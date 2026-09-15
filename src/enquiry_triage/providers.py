"""Provider adapters, including runnable deterministic demo providers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from urllib.parse import urlparse

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
                "DeepSeek json mode requirements: output one json object only, without markdown. "
                "Use exactly the fields and enum values in the JSON Schema below.\n"
                "Example JSON shape (use the email facts, not these example values):\n"
                '{"case_type":"OTHER","priority":"NORMAL","summary":"Short factual summary.",'
                '"draft_reply":"Professional draft.","confidence":{"score":0.5,'
                '"methodology":"Brief calibrated rationale."},"safety_status":"PENDING_REVIEW"}\n'
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


def provider_from_spec(spec: str) -> TriageProvider:
    """Create providers: demo-*, compatible:<model>, deepseek:<model>."""

    if spec in {"demo-fast", "demo-conservative"}:
        return DemoRuleBasedProvider(profile=spec.removeprefix("demo-"))
    if spec.startswith("compatible:"):
        model = spec.split(":", 1)[1].strip()
        base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").strip()
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
        if not model or not base_url or not api_key:
            raise ProviderError(
                "compatible:<model> requires OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY"
            )
        _validate_base_url(base_url)
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            input_usd_per_million=float(os.getenv("OPENAI_COMPATIBLE_INPUT_USD_PER_MILLION", "0")),
            output_usd_per_million=float(os.getenv("OPENAI_COMPATIBLE_OUTPUT_USD_PER_MILLION", "0")),
        )
    if spec.startswith("deepseek:"):
        model = spec.split(":", 1)[1].strip()
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY", "").strip()
            or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
        )
        if not model or not api_key:
            raise ProviderError("deepseek:<model> requires DEEPSEEK_API_KEY")
        _validate_base_url(base_url)
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            input_usd_per_million=float(os.getenv("DEEPSEEK_INPUT_USD_PER_MILLION", "0")),
            output_usd_per_million=float(os.getenv("DEEPSEEK_OUTPUT_USD_PER_MILLION", "0")),
            structured_output_mode=StructuredOutputMode.JSON_OBJECT,
        )
    raise ProviderError(f"Unknown provider spec: {spec}")
