"""Repeatable golden-set evaluation and model comparison utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .agent import SYSTEM_PROMPT, TriageAgent
from .models import CaseType, Inquiry, Priority, SafetyStatus, TriageAttempt
from .policies import groundedness_passes


def default_results_dir() -> Path:
    """Keep evaluation artifacts in the source project when running from this repository."""

    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").is_file():
        return project_root / "evaluation_results"
    return Path.cwd() / "evaluation_results"


def load_inquiries(path: Path) -> list[Inquiry]:
    inquiries: list[Inquiry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            inquiries.append(Inquiry.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
    if not inquiries:
        raise ValueError(f"No enquiries found in {path}")
    return inquiries


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_dataset(path: Path) -> str:
    """Require a sidecar SHA-256 file before a dataset may be called frozen."""

    hash_path = path.with_suffix(".sha256")
    if not hash_path.is_file():
        raise ValueError(f"Frozen dataset requires an integrity file: {hash_path}")
    expected = hash_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen dataset hash mismatch for {path}")
    return actual


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _classification_metrics(records: list[dict[str, Any]]) -> tuple[float, dict[str, dict[str, float | int]]]:
    labeled = [record for record in records if record["expected_case_type"]]
    if not labeled:
        return 0.0, {}
    report: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for case_type in CaseType:
        expected = case_type.value
        true_positive = sum(
            record["expected_case_type"] == expected and record["actual_case_type"] == expected
            for record in labeled
        )
        false_positive = sum(
            record["expected_case_type"] != expected and record["actual_case_type"] == expected
            for record in labeled
        )
        false_negative = sum(
            record["expected_case_type"] == expected and record["actual_case_type"] != expected
            for record in labeled
        )
        precision = _safe_divide(true_positive, true_positive + false_positive) or 0.0
        recall = _safe_divide(true_positive, true_positive + false_negative) or 0.0
        f1 = _safe_divide(2 * precision * recall, precision + recall) or 0.0
        report[expected] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": true_positive + false_negative,
        }
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values), report


def _calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    if not valid:
        return {"ece": None, "brier_score": None, "buckets": []}

    boundaries = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.000001)]
    buckets: list[dict[str, Any]] = []
    ece = 0.0
    brier_values: list[float] = []
    for lower, upper in boundaries:
        items = [record for record in valid if lower <= record["confidence"] < upper]
        if not items:
            continue
        average_confidence = statistics.fmean(record["confidence"] for record in items)
        actual_accuracy = statistics.fmean(1.0 if record["classification_correct"] else 0.0 for record in items)
        ece += abs(average_confidence - actual_accuracy) * len(items) / len(valid)
        buckets.append(
            {
                "range": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                "count": len(items),
                "average_confidence": round(average_confidence, 4),
                "actual_accuracy": round(actual_accuracy, 4),
            }
        )
    for record in valid:
        actual = 1.0 if record["classification_correct"] else 0.0
        brier_values.append((record["confidence"] - actual) ** 2)
    return {
        "ece": round(ece, 4),
        "brier_score": round(statistics.fmean(brier_values), 4),
        "buckets": buckets,
    }


def _reply_quality_heuristic(attempt: TriageAttempt) -> float:
    """Transparent non-LLM baseline; replace or supplement with blinded human review."""

    if not attempt.is_valid or attempt.result is None:
        return 0.0
    reply = attempt.result.draft_reply.lower()
    score = 0.0
    score += 0.35 if any(greeting in reply for greeting in ("hello", "dear", "您好")) else 0.0
    score += 0.25 if any(closing in reply for closing in ("kind regards", "regards", "谢谢", "此致")) else 0.0
    score += 0.20 if "customer" in reply or "review" in reply else 0.0
    score += 0.20 if len(reply) >= 80 else 0.0
    return score


def evaluate_agent(agent: TriageAgent, inquiries: Iterable[Inquiry]) -> dict[str, Any]:
    """Run one model over a frozen set and return raw records plus summary metrics."""

    raw_records: list[dict[str, Any]] = []
    for inquiry in inquiries:
        print(f"Evaluating {agent.provider.name} on inquiry {inquiry.id}")
        attempt = agent.triage(inquiry)
        result = attempt.result
        valid = attempt.is_valid and result is not None
        expected_case_type = inquiry.expected_case_type.value if inquiry.expected_case_type else None
        expected_priority = inquiry.expected_priority.value if inquiry.expected_priority else None
        expected_safety_status = (
            inquiry.expected_safety_status.value if inquiry.expected_safety_status else None
        )
        raw_records.append(
            {
                "inquiry_id": inquiry.id,
                "split": inquiry.split,
                "valid": valid,
                "validation_error": attempt.validation_error,
                "failure_detail": attempt.failure_detail,
                "expected_case_type": expected_case_type,
                "actual_case_type": result.case_type.value if result else None,
                "expected_priority": expected_priority,
                "actual_priority": result.priority.value if result else None,
                "expected_safety_status": expected_safety_status,
                "actual_safety_status": result.safety_status.value if result else None,
                "classification_correct": valid and result.case_type.value == expected_case_type,
                "priority_correct": valid and result.priority.value == expected_priority,
                "safety_correct": valid and result.safety_status.value == expected_safety_status,
                "groundedness_pass": valid and groundedness_passes(inquiry, result),
                "reply_quality_score": round(_reply_quality_heuristic(attempt), 4),
                "confidence": result.confidence.score if result else None,
                "latency_ms": round(attempt.latency_ms, 3),
                "input_tokens": attempt.usage.input_tokens,
                "output_tokens": attempt.usage.output_tokens,
                "cost_usd": attempt.usage.cost_usd,
            }
        )

    if not raw_records:
        raise ValueError("No enquiries supplied for evaluation")
    macro_f1, per_class_f1 = _classification_metrics(raw_records)
    labeled = [record for record in raw_records if record["expected_case_type"]]
    priority_labeled = [record for record in raw_records if record["expected_priority"]]
    urgent = [record for record in raw_records if record["expected_priority"] == Priority.URGENT.value]
    classification_accuracy = statistics.fmean(
        1.0 if record["classification_correct"] else 0.0 for record in labeled
    )
    priority_accuracy = statistics.fmean(
        1.0 if record["priority_correct"] else 0.0 for record in priority_labeled
    )
    groundedness_rate = statistics.fmean(
        1.0 if record["groundedness_pass"] else 0.0 for record in raw_records
    )
    reply_quality = statistics.fmean(record["reply_quality_score"] for record in raw_records)
    schema_success_rate = statistics.fmean(1.0 if record["valid"] else 0.0 for record in raw_records)
    safety_labeled = [record for record in raw_records if record["expected_safety_status"]]
    safety_accuracy = statistics.fmean(
        1.0 if record["safety_correct"] else 0.0 for record in safety_labeled
    )
    urgent_recall = statistics.fmean(1.0 if record["priority_correct"] else 0.0 for record in urgent)
    latencies = [record["latency_ms"] for record in raw_records]
    costs = [record["cost_usd"] for record in raw_records]
    calibration = _calibration(raw_records)
    quality_score = 0.40 * macro_f1 + 0.20 * priority_accuracy + 0.25 * groundedness_rate + 0.15 * reply_quality
    calibration_score = 1.0 - (calibration["ece"] if calibration["ece"] is not None else 1.0)
    p95_latency = _percentile(latencies, 0.95) or 0.0
    average_cost = statistics.fmean(costs)
    latency_score = max(0.0, 1 - p95_latency / 5_000)
    cost_score = max(0.0, 1 - average_cost / 0.02)
    fitness_score = 0.55 * quality_score + 0.20 * calibration_score + 0.15 * latency_score + 0.10 * cost_score
    gates = {
        "schema_success_100pct": schema_success_rate == 1.0,
        "safety_action_100pct": safety_accuracy == 1.0,
        "urgent_recall_100pct": urgent_recall == 1.0,
        "macro_f1_at_least_0_85": macro_f1 >= 0.85,
        "priority_accuracy_at_least_0_85": priority_accuracy >= 0.85,
        "groundedness_at_least_0_98": groundedness_rate >= 0.98,
        "calibration_ece_at_most_0_15": (
            calibration["ece"] is not None and calibration["ece"] <= 0.15
        ),
    }
    summary = {
        "model": agent.provider.name,
        "case_count": len(raw_records),
        "classification_accuracy": round(classification_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "priority_accuracy": round(priority_accuracy, 4),
        "urgent_recall": round(urgent_recall, 4),
        "safety_action_accuracy": round(safety_accuracy, 4),
        "groundedness_rate": round(groundedness_rate, 4),
        "reply_quality_heuristic": round(reply_quality, 4),
        "schema_success_rate": round(schema_success_rate, 4),
        "calibration": calibration,
        "operational": {
            "latency_ms_p50": round(_percentile(latencies, 0.5) or 0.0, 3),
            "latency_ms_p95": round(p95_latency, 3),
            "average_cost_usd": round(average_cost, 8),
            "total_cost_usd": round(sum(costs), 8),
        },
        "fitness": {
            "weights": {"quality": 0.55, "calibration": 0.20, "latency": 0.15, "cost": 0.10},
            "quality_score": round(quality_score, 4),
            "score": round(fitness_score, 4),
        },
        "gates": gates,
    }
    return {"summary": summary, "records": raw_records}


def compare_agents(
    agents: Iterable[TriageAgent],
    inquiries: list[Inquiry],
    golden_set_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare at least two distinct provider names on exactly the same golden records."""

    agents = list(agents)
    names = [agent.provider.name for agent in agents]
    if len(agents) < 2 or len(set(names)) < 2:
        raise ValueError("Model comparison requires at least two distinct provider/model identifiers.")
    runs = [evaluate_agent(agent, inquiries) for agent in agents]
    eligible = [run for run in runs if all(run["summary"]["gates"].values())]
    ranked = sorted(eligible, key=lambda run: run["summary"]["fitness"]["score"], reverse=True)
    recommendation = (
        {
            "model": ranked[0]["summary"]["model"],
            "reason": "Highest composite score among models that passed every safety and quality gate.",
            "would_change_with": "A larger representative labelled set, human reply review, or a gate failure.",
        }
        if ranked
        else {
            "model": None,
            "reason": "No model passed every safety and quality gate; do not recommend deployment.",
            "would_change_with": "A corrected model run that passes all listed gates on the unchanged golden set.",
        }
    )
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "golden_ids": [inquiry.id for inquiry in inquiries],
        "golden_set_sha256": golden_set_sha256,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "runs": runs,
        "recommendation": recommendation,
    }


def write_comparison(comparison: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write auditable summary JSON and flat per-case CSV without customer email bodies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"comparison_{stamp}.json"
    records_path = output_dir / f"comparison_{stamp}_records.csv"
    summary_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [
        {key: _escape_csv_value(value) for key, value in {"model": run["summary"]["model"], **record}.items()}
        for run in comparison["runs"]
        for record in run["records"]
    ]
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, records_path


def _escape_csv_value(value: Any) -> Any:
    """Prevent spreadsheet applications interpreting an untrusted string as a formula."""

    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return f"'{value}"
    return value
