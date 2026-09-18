"""Strict domain contracts for the triage workflow."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator


class CaseType(str, Enum):
    POLICY_QUERY = "POLICY_QUERY"
    PREMIUM_BILLING = "PREMIUM_BILLING"
    ADDRESS_CHANGE = "ADDRESS_CHANGE"
    CLAIM = "CLAIM"
    COMPLAINT = "COMPLAINT"
    OTHER = "OTHER"


class Priority(str, Enum):
    URGENT = "URGENT"
    NORMAL = "NORMAL"
    LOW = "LOW"


class SafetyStatus(str, Enum):
    """All valid outputs are human-reviewable; none may be auto-sent."""

    PENDING_REVIEW = "PENDING_REVIEW"
    REFUSE_AND_ESCALATE = "REFUSE_AND_ESCALATE"


class ReviewDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    EDITED_AND_ACCEPTED = "EDITED_AND_ACCEPTED"
    DISCARDED = "DISCARDED"


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: StrictFloat = Field(ge=0.0, le=1.0)
    methodology: str = Field(min_length=12, max_length=240)


class TriageResult(BaseModel):
    """The single schema-validated model output contract."""

    model_config = ConfigDict(extra="forbid")

    case_type: CaseType
    priority: Priority
    summary: str = Field(min_length=10, max_length=360)
    draft_reply: str = Field(min_length=20, max_length=2_000)
    confidence: Confidence
    safety_status: SafetyStatus

    @field_validator("summary", "draft_reply")
    @classmethod
    def reject_blank_or_control_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        if "\x00" in cleaned:
            raise ValueError("must not include NUL characters")
        return cleaned

    @field_validator("summary")
    @classmethod
    def limit_summary_to_two_sentences(cls, value: str) -> str:
        sentence_endings = value.count(".") + value.count("!") + value.count("?")
        if sentence_endings > 2:
            raise ValueError("summary must contain one or two sentences")
        return value


class Inquiry(BaseModel):
    """Input record. Evaluation labels are optional during a live triage."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=100)
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    split: str = "live"
    expected_case_type: CaseType | None = None
    expected_priority: Priority | None = None
    expected_safety_status: SafetyStatus | None = None
    reference_reply_notes: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    content: str
    latency_ms: float = Field(ge=0.0)
    usage: Usage = Field(default_factory=Usage)


class TriageAttempt(BaseModel):
    """Application envelope: valid result or a visible, safe failure."""

    model_config = ConfigDict(extra="forbid")

    inquiry_id: str
    provider: str
    result: TriageResult | None = None
    validation_error: str | None = None
    failure_detail: str | None = None
    rejected_draft_reply: str | None = None
    rejected_model_output: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: Usage = Field(default_factory=Usage)

    @property
    def is_valid(self) -> bool:
        return self.result is not None and self.validation_error is None


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    inquiry_id: str
    source_subject: str
    source_body: str
    status: SafetyStatus
    result: TriageResult
    created_at: str
    decision: ReviewDecision | None = None
    edited_reply: str | None = None
    decided_at: str | None = None


def json_schema() -> dict[str, Any]:
    """Return the strict schema sent to structured-output providers."""

    return TriageResult.model_json_schema()
