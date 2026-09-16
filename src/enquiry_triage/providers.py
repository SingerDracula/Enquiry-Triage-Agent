"""Provider adapters, including runnable deterministic demo providers."""

from __future__ import annotations

import json
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlparse

from .models import (
    CaseType,
    Confidence,
    Priority,
    ProviderResponse,
    SafetyStatus,
    TriageResult,
    Usage,
)
from .policies import is_low_priority, is_urgent, requires_refusal, safe_refusal_result


class ProviderError(RuntimeError):
    """Raised when a remote model cannot be reached or produces no usable message."""

    def __init__(self, message: str, *, latency_ms: float = 0.0) -> None:
        super().__init__(message)
        self.latency_ms = latency_ms


def _error_detail(error: urllib.error.HTTPError, limit: int = 300) -> str:
    """Return a bounded, single-line excerpt of an HTTP error body for diagnostics."""

    body = error.read() if error.fp is not None else b""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    return " ".join(text.split())[:limit]


class StructuredOutputMode(str, Enum):
    """Server-side output modes supported by the chat-completions adapter."""

    JSON_SCHEMA_STRICT = "json_schema_strict"
    JSON_OBJECT = "json_object"


class TriageProvider(Protocol):
    name: str

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        """Return a JSON string intended to satisfy the supplied schema."""


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname in local_hosts:
        return
    raise ProviderError("Remote provider URL must use HTTPS; plain HTTP is allowed only for localhost.")


def _choose_case_type(text: str) -> CaseType:
    normalized = text.lower()
    if requires_refusal(normalized):
        return CaseType.OTHER
    if any(word in normalized for word in ("complaint", "complain", "unacceptable", "ombudsman", "angry")):
        return CaseType.COMPLAINT
    if any(word in normalized for word in ("claim", "accident", "hospital", "injury", "loss")):
        return CaseType.CLAIM
    if any(word in normalized for word in ("address", "moved", "move house", "relocate", "postal")):
        return CaseType.ADDRESS_CHANGE
    if any(word in normalized for word in ("premium", "billing", "payment", "invoice", "direct debit", "pay")):
        return CaseType.PREMIUM_BILLING
    if any(word in normalized for word in ("policy", "coverage", "beneficiary", "surrender", "plan")):
        return CaseType.POLICY_QUERY
    return CaseType.OTHER


def _topic_phrase(case_type: CaseType) -> str:
    return {
        CaseType.POLICY_QUERY: "your policy query",
        CaseType.PREMIUM_BILLING: "your premium or billing query",
        CaseType.ADDRESS_CHANGE: "your address-change request",
        CaseType.CLAIM: "your claim query",
        CaseType.COMPLAINT: "your complaint",
        CaseType.OTHER: "your enquiry",
    }[case_type]


@dataclass(frozen=True)
class DemoRuleBasedProvider:
    """Offline baseline for a complete demo; it is explicitly not an LLM comparison."""

    profile: str = "fast"

    @property
    def name(self) -> str:
        return f"demo-{self.profile}"

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        started = time.perf_counter()
        case_type = _choose_case_type(email_text)
        priority = Priority.URGENT if is_urgent(email_text) else Priority.NORMAL
        if priority is Priority.NORMAL and is_low_priority(email_text):
            priority = Priority.LOW
        if case_type is CaseType.OTHER and not is_urgent(email_text):
            priority = Priority.LOW

        if requires_refusal(email_text):
            result = safe_refusal_result()
        else:
            urgency_line = (
                "We understand that this matter may be urgent and will make it visible to the reviewing agent."
                if priority is Priority.URGENT
                else "A customer-service agent will review the details before any action is taken."
            )
            result = TriageResult(
                case_type=case_type,
                priority=priority,
                summary=(
                    f"Customer is contacting us about {_topic_phrase(case_type)}. "
                    "The draft should be reviewed against the original email before it is used."
                ),
                draft_reply=(
                    f"Hello,\n\nThank you for contacting us about {_topic_phrase(case_type)}. "
                    f"{urgency_line} If further information is needed, we will ask for it through the "
                    "appropriate verified process.\n\nKind regards,\nCustomer Service"
                ),
                confidence=Confidence(
                    score=0.84 if self.profile == "fast" else 0.76,
                    methodology=(
                        "Derived from matched enquiry keywords and reduced when the message may be ambiguous."
                    ),
                ),
                safety_status=SafetyStatus.PENDING_REVIEW,
            )

        content = result.model_dump_json()
        return ProviderResponse(
            model_name=self.name,
            content=content,
            latency_ms=(time.perf_counter() - started) * 1_000,
            usage=Usage(),
        )


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """Chat-completions adapter with strict-schema and JSON-object compatibility modes."""

    model: str
    base_url: str
    api_key: str
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA_STRICT
    max_tokens: int = 800

    @property
    def name(self) -> str:
        return self.model

    def build_payload(self, *, system_prompt: str, email_text: str, response_schema: dict) -> dict:
        """Build the provider request without exposing a local schema-validation fallback as strict mode."""

        if self.structured_output_mode is StructuredOutputMode.JSON_SCHEMA_STRICT:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "triage_result", "strict": True, "schema": response_schema},
            }
            formatted_system_prompt = system_prompt
        else:
            response_format = {"type": "json_object"}
            formatted_system_prompt = (
                f"{system_prompt}\n\n"
                "DeepSeek JSON mode requirements: return exactly one JSON object and no markdown, "
                "preamble, explanation, or extra fields. Use the exact field names and enum values below.\n"
                "Field contract:\n"
                "- case_type: exactly one of POLICY_QUERY, PREMIUM_BILLING, ADDRESS_CHANGE, CLAIM, COMPLAINT, OTHER.\n"
                "- priority: exactly one of URGENT, NORMAL, LOW. Decide priority independently from case_type.\n"
                "  Select URGENT when the email describes a time-sensitive impact: a deadline or effective date "
                "today/tomorrow, immediate loss of cover or access, an active accident/claim/fraud event, "
                "hospitalisation, or an explicit urgent need. If a policy cancellation takes effect today and "
                "the customer asks what cover remains, priority MUST be URGENT.\n"
                "  Select NORMAL by default for a customer request that needs an explanation, human review, "
                "identity verification, missing details, or a future account/policy change, even without a deadline. "
                "A beneficiary question, a waiting-period question with a missing attachment, and an address-change "
                "request missing postal details are all NORMAL. Do not treat anger alone as URGENT; capital letters or "
                "repeated complaints alone also do not make a case URGENT.\n"
                "  Select LOW only when the message is clearly optional or general and needs no customer-specific "
                "follow-up, verification, explanation, or change; do not use LOW merely because information is missing.\n"
                "- summary: one or two factual sentences, 10-360 characters.\n"
                "- draft_reply: professional customer-service draft, 20-2000 characters; do not invent facts, "
                "amounts, dates, coverage, or timelines.\n"
                "- confidence.score: JSON number from 0.0 to 1.0 meaning the estimated probability that case_type "
                "is correct; it is not confidence in the reply quality or priority. Use 0.90-0.98 for an explicit, "
                "unambiguous case_type with direct evidence. Use 0.75-0.89 only when the category is clear but some "
                "supporting details are missing. Use 0.50-0.74 only when multiple case_types are genuinely plausible. "
                "Do not lower the score merely because details are missing when the case_type itself is explicit.\n"
                "- confidence.methodology: exactly one concise sentence, 12-120 characters, explaining only "
                "the classification evidence and uncertainty. Do not repeat the email, policy rules, or a long rationale.\n"
                "- safety_status: exactly PENDING_REVIEW or REFUSE_AND_ESCALATE.\n"
                "Example JSON shape (use the email facts, not these example values):\n"
                '{"case_type":"OTHER","priority":"NORMAL","summary":"Customer asks for help with a service issue.",'
                '"draft_reply":"Hello, thank you for contacting us. A customer-service agent will review your request. Kind regards, Customer Service.",'
                '"confidence":{"score":0.5,"methodology":"The request is ambiguous and needs human review."},'
                '"safety_status":"PENDING_REVIEW"}\n'
                f"JSON Schema:\n{json.dumps(response_schema, ensure_ascii=False)}"
            )

        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": email_text},
            ],
            "response_format": response_format,
        }

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        payload = self.build_payload(
            system_prompt=system_prompt,
            email_text=email_text,
            response_schema=response_schema,
        )
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310 - configured endpoint
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"Remote provider rejected the request with HTTP {exc.code}: {_error_detail(exc)}",
                latency_ms=(time.perf_counter() - started) * 1_000,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"Remote provider failed: {exc}",
                latency_ms=(time.perf_counter() - started) * 1_000,
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1_000
        try:
            choice = decoded["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ProviderError("Remote provider output was truncated", latency_ms=elapsed_ms)
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                raise TypeError("message content was not a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Remote provider response did not include a text message", latency_ms=elapsed_ms
            ) from exc

        usage_raw = decoded.get("usage", {})
        input_tokens = int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0)
        output_tokens = int(usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0)
        cost = (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000
        return ProviderResponse(
            model_name=self.name,
            content=content,
            latency_ms=(time.perf_counter() - started) * 1_000,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
        )


def _gemini_response_schema(response_schema: dict) -> dict:
    """Convert the small Pydantic JSON Schema subset used here to Gemini's schema format.

    Gemini's GenerateContent API uses upper-case JSON schema type names and does
    not accept Pydantic's local ``$defs``/``$ref`` representation directly. Its
    ``responseSchema`` is an OpenAPI-style subset that also rejects keywords
    such as ``additionalProperties``, so Pydantic's ``extra="forbid"`` marker is
    dropped (the remote schema still only permits the declared properties).
    Keeping this conversion deliberately narrow means unsupported contract
    constructs fail safely in local Pydantic validation rather than being
    silently approximated in a remote request.
    """

    definitions = response_schema.get("$defs", {})

    def convert(node: object) -> object:
        if isinstance(node, list):
            return [convert(value) for value in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            reference = node["$ref"]
            if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
                raise ProviderError("Gemini provider received an unsupported JSON Schema reference")
            name = reference.removeprefix("#/$defs/")
            if name not in definitions:
                raise ProviderError("Gemini provider received an unresolved JSON Schema reference")
            return convert(definitions[name])

        converted: dict[str, object] = {}
        for key in (
            "title",
            "description",
            "enum",
            "required",
            "minimum",
            "maximum",
            "minItems",
            "maxItems",
        ):
            if key in node:
                converted[key] = convert(node[key])
        if "additionalProperties" in node:
            # Gemini's responseSchema has no ``additionalProperties`` field. A
            # closed object (``false``) matches the remote default of only
            # accepting declared properties, so it can be dropped safely. Open
            # or free-form objects have no faithful equivalent and must fail
            # loudly instead of being approximated.
            if node["additionalProperties"] is not False:
                raise ProviderError(
                    "Gemini provider does not support open JSON Schema objects"
                )
        if "type" in node:
            schema_type = node["type"]
            if not isinstance(schema_type, str):
                raise ProviderError("Gemini provider supports only single JSON Schema types")
            converted["type"] = schema_type.upper()
        if "properties" in node:
            properties = node["properties"]
            if not isinstance(properties, dict):
                raise ProviderError("Gemini provider received invalid JSON Schema properties")
            converted["properties"] = {name: convert(value) for name, value in properties.items()}
        if "items" in node:
            converted["items"] = convert(node["items"])
        return converted

    converted_schema = convert(response_schema)
    if not isinstance(converted_schema, dict):  # defensive: the root contract must be an object
        raise ProviderError("Gemini provider requires an object JSON Schema")
    return converted_schema


@dataclass(frozen=True)
class GeminiGenerateContentProvider:
    """Gemini GenerateContent adapter using server-side JSON Schema output."""

    model: str
    base_url: str
    api_key: str
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0
    max_output_tokens: int = 800

    @property
    def name(self) -> str:
        return self.model

    def build_payload(self, *, system_prompt: str, email_text: str, response_schema: dict) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": email_text}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_response_schema(response_schema),
            },
        }

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        payload = self.build_payload(
            system_prompt=system_prompt,
            email_text=email_text,
            response_schema=response_schema,
        )
        endpoint = f"{self.base_url.rstrip('/')}/models/{quote(self.model, safe='')}:generateContent"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310 - configured endpoint
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"Gemini provider rejected the request with HTTP {exc.code}: {_error_detail(exc)}",
                latency_ms=(time.perf_counter() - started) * 1_000,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"Gemini provider failed: {exc}", latency_ms=(time.perf_counter() - started) * 1_000
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1_000
        try:
            candidate = decoded["candidates"][0]
            if candidate.get("finishReason") in {"MAX_TOKENS", "LENGTH"}:
                raise ProviderError("Gemini provider output was truncated", latency_ms=elapsed_ms)
            parts = candidate["content"]["parts"]
            content = "".join(part["text"] for part in parts if isinstance(part, dict) and "text" in part)
            if not content:
                raise TypeError("candidate included no text parts")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Gemini provider response did not include a text message", latency_ms=elapsed_ms
            ) from exc

        usage_raw = decoded.get("usageMetadata", {})
        input_tokens = int(usage_raw.get("promptTokenCount", 0) or 0)
        output_tokens = int(usage_raw.get("candidatesTokenCount", 0) or 0)
        cost = (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000
        return ProviderResponse(
            model_name=self.name,
            content=content,
            latency_ms=elapsed_ms,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
        )


def _load_provider_settings(config_path: Path) -> dict[str, dict[str, object]]:
    """Load a private TOML config without ever placing its secrets in errors."""

    if not config_path.is_file():
        raise ProviderError(
            f"Provider config not found: {config_path}. Copy config.example.toml to config.toml and add your key."
        )
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProviderError(f"Could not read provider config: {exc}") from exc
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raise ProviderError("Provider config must contain [providers.<name>] sections")
    return {name: value for name, value in providers.items() if isinstance(name, str) and isinstance(value, dict)}


def _required_string(settings: dict[str, object], key: str, provider_name: str) -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"Provider '{provider_name}' requires a non-empty '{key}' in config.toml")
    return value.strip()


def _optional_string(settings: dict[str, object], key: str, default: str, provider_name: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"Provider '{provider_name}' has an invalid '{key}' in config.toml")
    return value.strip()


def _price(settings: dict[str, object], key: str, provider_name: str) -> float:
    value = settings.get(key, 0)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ProviderError(f"Provider '{provider_name}' has an invalid non-negative '{key}' in config.toml")
    return float(value)


def provider_from_spec(spec: str, *, config_path: Path | None = None) -> TriageProvider:
    """Create a demo provider or a named provider configured in private TOML."""

    if spec in {"demo-fast", "demo-conservative"}:
        return DemoRuleBasedProvider(profile=spec.removeprefix("demo-"))
    if ":" in spec or spec not in {"gpt", "deepseek", "gemini", "compatible"}:
        raise ProviderError("Unknown provider. Use demo-fast, demo-conservative, gpt, deepseek, gemini, or compatible.")

    settings_by_name = _load_provider_settings(config_path or Path.cwd() / "config.toml")
    settings = settings_by_name.get(spec)
    if settings is None:
        raise ProviderError(f"Provider '{spec}' is missing [providers.{spec}] in config.toml")
    model = _required_string(settings, "model", spec)
    api_key = _required_string(settings, "api_key", spec)

    if spec == "gemini":
        base_url = _optional_string(
            settings, "base_url", "https://generativelanguage.googleapis.com/v1beta", spec
        )
        _validate_base_url(base_url)
        return GeminiGenerateContentProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            input_usd_per_million=_price(settings, "input_usd_per_million", spec),
            output_usd_per_million=_price(settings, "output_usd_per_million", spec),
        )

    default_base_url = "https://api.openai.com/v1" if spec == "gpt" else ""
    base_url = _optional_string(settings, "base_url", default_base_url, spec)
    _validate_base_url(base_url)
    return OpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        input_usd_per_million=_price(settings, "input_usd_per_million", spec),
        output_usd_per_million=_price(settings, "output_usd_per_million", spec),
        structured_output_mode=(
            StructuredOutputMode.JSON_OBJECT if spec == "deepseek" else StructuredOutputMode.JSON_SCHEMA_STRICT
        ),
    )
