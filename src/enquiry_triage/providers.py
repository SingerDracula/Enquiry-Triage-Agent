"""Provider adapters, including runnable deterministic demo providers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    """Generic JSON-schema adapter for OpenAI-compatible chat-completions endpoints."""

    model: str
    base_url: str
    api_key: str
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0

    @property
    def name(self) -> str:
        return self.model

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": email_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "triage_result", "strict": True, "schema": response_schema},
            },
        }
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
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Remote provider failed: {exc}") from exc

        try:
            content = decoded["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                raise TypeError("message content was not a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Remote provider response did not include a text message") from exc

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
    """Create providers: demo-fast, demo-conservative, compatible:<model>."""

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
    raise ProviderError(f"Unknown provider spec: {spec}")
