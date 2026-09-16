"""Local-only FastAPI UI for triage, human review, and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import sysconfig
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent import TriageAgent
from .evaluation import compare_agents, load_inquiries, verify_frozen_dataset, write_comparison
from .models import Inquiry, ReviewDecision
from .providers import ProviderError, provider_from_spec
from .review import ReviewError, ReviewStore


def _app_state_dir() -> Path:
    configured = os.getenv("AI_TRIAGE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ai-enquiry-triage"
    return Path.home() / ".local" / "share" / "ai-enquiry-triage"


def _default_golden_set() -> Path:
    installed = Path(sysconfig.get_path("data")) / "share" / "ai-enquiry-triage" / "golden_set.jsonl"
    source_tree = Path(__file__).resolve().parents[2] / "data" / "golden_set.jsonl"
    for candidate in (Path.cwd() / "data" / "golden_set.jsonl", installed, source_tree):
        if candidate.is_file():
            return candidate
    return installed


APP_STATE_DIR = _app_state_dir()
DEFAULT_DB = APP_STATE_DIR / "review_queue.db"
DEFAULT_RESULTS = APP_STATE_DIR / "results"
DEFAULT_GOLDEN_SET = _default_golden_set()
DEFAULT_CONFIG = Path(os.getenv("AI_TRIAGE_CONFIG", "config.toml"))
STATIC_DIR = Path(__file__).with_name("static")


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    provider: str = Field(default="demo-fast", min_length=1, max_length=80)
    queue_for_review: bool = True


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    edited_reply: str | None = Field(default=None, max_length=2_000)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=2, max_length=4
    )


app = FastAPI(
    title="AI Enquiry Triage",
    description="Local human-review workflow. It never sends email.",
    version="0.1.0",
)


def _provider(spec: str):
    try:
        return provider_from_spec(spec, config_path=DEFAULT_CONFIG)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _review_store() -> ReviewStore:
    return ReviewStore(DEFAULT_DB)


def _record_payload(record: object) -> dict:
    return record.model_dump(mode="json")  # type: ignore[union-attr]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "config": str(DEFAULT_CONFIG), "database": str(DEFAULT_DB)}


@app.post("/api/triage")
def triage(request: TriageRequest) -> dict:
    inquiry = Inquiry(id="web-email", subject=request.subject, body=request.body)
    attempt = TriageAgent(_provider(request.provider)).triage(inquiry)
    response = {"attempt": attempt.model_dump(mode="json"), "review_record": None}
    if attempt.is_valid and request.queue_for_review:
        response["review_record"] = _record_payload(_review_store().enqueue(attempt, inquiry))
    return response


@app.get("/api/reviews")
def list_reviews() -> dict:
    return {"records": [_record_payload(record) for record in _review_store().list_open()]}


@app.get("/api/reviews/{record_id}")
def get_review(record_id: int) -> dict:
    try:
        return _record_payload(_review_store().get(record_id))
    except ReviewError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/reviews/{record_id}/decision")
def decide_review(record_id: int, request: DecisionRequest) -> dict:
    try:
        return _record_payload(_review_store().decide(record_id, request.decision, request.edited_reply))
    except ReviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest) -> dict:
    specs = [spec.strip() for spec in request.providers if spec.strip()]
    if len(specs) < 2 or len(set(specs)) < 2:
        raise HTTPException(status_code=400, detail="Choose at least two distinct providers for evaluation.")
    agents = [TriageAgent(_provider(spec)) for spec in specs]
    try:
        golden_hash = verify_frozen_dataset(DEFAULT_GOLDEN_SET)
        comparison = compare_agents(agents, load_inquiries(DEFAULT_GOLDEN_SET), golden_hash)
        summary_path, records_path = write_comparison(comparison, DEFAULT_RESULTS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "comparison": comparison,
        "artifacts": {"summary": str(summary_path), "records": str(records_path)},
    }


def run() -> None:
    """Start the local UI at http://127.0.0.1:8000."""

    import uvicorn

    uvicorn.run("enquiry_triage.web:app", host="127.0.0.1", port=8000, reload=False)
