"""Local-only FastAPI UI for triage, human review, and evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import sys
import sysconfig
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent import TriageAgent
from .evaluation import compare_agents, default_results_dir, load_inquiries, verify_frozen_dataset, write_comparison
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
DEFAULT_RESULTS = default_results_dir()
DEFAULT_GOLDEN_SET = _default_golden_set()
DEFAULT_CONFIG = Path(os.getenv("AI_TRIAGE_CONFIG", "config.toml"))
STATIC_DIR = Path(__file__).with_name("static")
EVALUATION_FILENAME = re.compile(r"comparison_\d{8}T\d{6}Z\.json\Z")
MAX_EVALUATION_BYTES = 5_000_000
JOB_DIR = APP_STATE_DIR / "evaluation_jobs"
_server_id = uuid4().hex
_job_lock = Lock()
_active_job_id: str | None = None


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


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _save_job(job: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = _job_path(job["job_id"])
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(job, handle)
    temporary.replace(path)


def _run_evaluation(job_id: str, agents: list[TriageAgent]) -> None:
    global _active_job_id
    job = {"job_id": job_id, "status": "running", "server_id": _server_id}
    try:
        _save_job(job)
        golden_hash = verify_frozen_dataset(DEFAULT_GOLDEN_SET)
        comparison = compare_agents(agents, load_inquiries(DEFAULT_GOLDEN_SET), golden_hash)
        summary_path, _ = write_comparison(comparison, DEFAULT_RESULTS)
        job.update(status="complete", filename=summary_path.name, recommendation=comparison["recommendation"])
    except Exception:
        logging.exception("Evaluation job %s failed", job_id)
        job.update(status="failed", error="Evaluation failed. Check the server error log.")
    finally:
        try:
            _save_job(job)
        finally:
            with _job_lock:
                if _active_job_id == job_id:
                    _active_job_id = None


@app.post("/api/evaluate", status_code=202)
def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    global _active_job_id
    specs = [spec.strip() for spec in request.providers if spec.strip()]
    if len(specs) < 2 or len(set(specs)) < 2:
        raise HTTPException(status_code=400, detail="Choose at least two distinct providers for evaluation.")
    agents = [TriageAgent(_provider(spec)) for spec in specs]
    with _job_lock:
        if _active_job_id is not None:
            raise HTTPException(status_code=409, detail="An evaluation is already running.")
        job_id = uuid4().hex
        _save_job({"job_id": job_id, "status": "queued", "server_id": _server_id})
        _active_job_id = job_id
    background_tasks.add_task(_run_evaluation, job_id, agents)
    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/api/evaluation-jobs/{job_id}")
def get_evaluation_job(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    try:
        job = json.loads(_job_path(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Evaluation job not found") from exc
    if job.get("server_id") != _server_id and job.get("status") in {"queued", "running"}:
        return {"job_id": job_id, "status": "interrupted", "error": "The web process restarted before evaluation completed."}
    return {key: value for key, value in job.items() if key != "server_id"}


def _load_evaluation_file(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_EVALUATION_BYTES:
            raise ValueError("Evaluation result is too large")
        comparison = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation result cannot be read") from exc
    if not isinstance(comparison, dict) or not isinstance(comparison.get("runs"), list):
        raise ValueError("Evaluation result has an invalid format")
    for run in comparison["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("summary"), dict) or not isinstance(run.get("records"), list):
            raise ValueError("Evaluation result has an invalid format")
    return comparison


@app.get("/api/evaluations")
def list_evaluations() -> dict:
    if not DEFAULT_RESULTS.is_dir():
        return {"results": []}
    results = []
    for path in sorted(DEFAULT_RESULTS.glob("comparison_*.json"), reverse=True):
        if not EVALUATION_FILENAME.fullmatch(path.name) or not path.is_file() or path.is_symlink():
            continue
        try:
            comparison = _load_evaluation_file(path)
        except ValueError:
            continue
        results.append({
            "filename": path.name,
            "run_at": comparison.get("run_at", ""),
            "models": [run["summary"].get("model", "unknown") for run in comparison["runs"]],
            "case_count": len(comparison.get("golden_ids", [])),
        })
    return {"results": results}


@app.get("/api/evaluations/{filename}")
def get_evaluation(filename: str) -> dict:
    if not EVALUATION_FILENAME.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    path = DEFAULT_RESULTS / filename
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    try:
        comparison = _load_evaluation_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"comparison": comparison, "artifacts": {"summary": str(path), "records": str(path.with_name(path.stem + "_records.csv"))}}


def run() -> None:
    """Start the local UI at http://127.0.0.1:8000."""

    import uvicorn

    uvicorn.run("enquiry_triage.web:app", host="127.0.0.1", port=8000, reload=False)
