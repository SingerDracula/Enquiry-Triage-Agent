"""Deterministic safety and groundedness checks independent of an LLM."""

from __future__ import annotations

import re

from .models import CaseType, Confidence, Inquiry, Priority, SafetyStatus, TriageResult


INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "reveal your instructions",
)

PRIVACY_RISK_MARKERS = (
    "another customer's",
    "another customer",
    "someone else's policy",
    "someone elses policy",
    "bypass verification",
    "without verification",
    "full policy details",
    "staff login",
    "password instructions",
    "credentials",
    "internal notes",
    "alter the claim date",
    "make it look like",
)

URGENT_MARKERS = (
    "urgent",
    "immediately",
    "today",
    "overdue",
    "accident",
    "hospital",
    "fraud",
    "emergency",
)

LOW_PRIORITY_MARKERS = (
    "no rush",
    "not urgent",
    "whenever convenient",
    "just curious",
    "for future reference",
)


def requires_refusal(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in INJECTION_MARKERS + PRIVACY_RISK_MARKERS)


def is_urgent(text: str) -> bool:
    normalized = text.lower()
    if is_low_priority(normalized):
        return False
    return any(marker in normalized for marker in URGENT_MARKERS)


def is_low_priority(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in LOW_PRIORITY_MARKERS)


def safe_refusal_result() -> TriageResult:
    """A deterministic boundary response for clearly unsafe or unauthorized requests."""

    return TriageResult(
        case_type=CaseType.OTHER,
        priority=Priority.NORMAL,
        summary="The message requests unsafe or unauthorized information and requires human handling.",
        draft_reply=(
            "Hello,\n\nTo protect personal information, we cannot provide or change policy details "
            "through this request. Please use an official customer-service channel and complete "
            "identity verification so a team member can assist you.\n\nKind regards,\nCustomer Service"
        ),
        confidence=Confidence(
            score=0.96,
            methodology="A deterministic safety policy detected an unauthorized or prompt-injection request.",
        ),
        safety_status=SafetyStatus.REFUSE_AND_ESCALATE,
    )


def extract_numbers(text: str) -> set[str]:
    """Find number-like facts that a reply should not fabricate."""

    return set(re.findall(r"(?<![A-Za-z])\d+(?:[,.]\d+)?(?:%|\b)", text))


def unsupported_numeric_facts(inquiry: Inquiry, result: TriageResult) -> list[str]:
    """Conservative heuristic; generic greeting text is intentionally ignored."""

    source_numbers = extract_numbers(f"{inquiry.subject}\n{inquiry.body}")
    reply_numbers = extract_numbers(result.draft_reply)
    return sorted(reply_numbers - source_numbers)


def groundedness_passes(inquiry: Inquiry, result: TriageResult) -> bool:
    return not unsupported_numeric_facts(inquiry, result)
