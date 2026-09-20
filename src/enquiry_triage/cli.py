"""Standard-library CLI for draft creation, review decisions, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import sysconfig
from typing import Sequence

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
DEFAULT_GOLDEN_SET = _default_golden_set()
DEFAULT_RESULTS = default_results_dir()


def _print_json(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load_provider(spec: str, config_path: Path):
    try:
        return provider_from_spec(spec, config_path=config_path)
    except ProviderError as exc:
        raise ValueError(f"Provider configuration error: {exc}") from exc


def _add_database_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="SQLite review queue path")


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Private provider TOML configuration path (default: config.toml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage-agent",
        description="Generate customer-service drafts for human review only; it never sends email.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    triage_parser = commands.add_parser("triage", help="Generate one validated draft from a text email")
    triage_parser.add_argument("email_file", type=Path, help="Plain-text email body")
    triage_parser.add_argument("--subject", default="", help="Optional subject line")
    triage_parser.add_argument("--inquiry-id", default=None, help="Defaults to email file stem")
    triage_parser.add_argument(
        "--provider",
        default="demo-fast",
        help="demo-fast, demo-conservative, gpt, deepseek, gemini, glm, jev, or compatible",
    )
    _add_config_option(triage_parser)
    triage_parser.add_argument("--no-queue", action="store_true", help="Do not store a pending local review")
    _add_database_option(triage_parser)

    review_parser = commands.add_parser("review", help="Inspect and decide local review proposals")
    review_commands = review_parser.add_subparsers(dest="review_command", required=True)
    list_parser = review_commands.add_parser("list", help="List pending proposals")
    _add_database_option(list_parser)
    show_parser = review_commands.add_parser("show", help="Show one generated proposal")
    show_parser.add_argument("record_id", type=int)
    _add_database_option(show_parser)
    decide_parser = review_commands.add_parser("decide", help="Record a human decision; does not send email")
    decide_parser.add_argument("record_id", type=int)
    decide_parser.add_argument("action", choices=[decision.value for decision in ReviewDecision])
    decide_parser.add_argument("--edited-reply", default=None)
    _add_database_option(decide_parser)

    evaluate_parser = commands.add_parser("evaluate", help="Compare at least two providers on frozen golden data")
    evaluate_parser.add_argument("--models", default="demo-fast,demo-conservative")
    _add_config_option(evaluate_parser)
    evaluate_parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_SET)
    evaluate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser


def _triage(args: argparse.Namespace) -> int:
    if not args.email_file.is_file():
        raise ValueError(f"Email file does not exist: {args.email_file}")
    inquiry = Inquiry(
        id=args.inquiry_id or args.email_file.stem,
        subject=args.subject,
        body=args.email_file.read_text(encoding="utf-8"),
    )
    attempt = TriageAgent(_load_provider(args.provider, args.config)).triage(inquiry)
    if not attempt.is_valid:
        failure_payload = attempt.model_dump(mode="json")
        _print_json(failure_payload)
        print("Draft was not queued: model output failed safely and visibly.")
        return 2
    payload = attempt.model_dump(mode="json")
    if not args.no_queue:
        record = ReviewStore(args.database).enqueue(attempt, inquiry)
        payload["review_record_id"] = record.id
        payload["review_queue"] = str(args.database)
    _print_json(payload)
    print("No email was sent. The draft remains pending human review.")
    return 0


def _review(args: argparse.Namespace) -> int:
    store = ReviewStore(args.database)
    if args.review_command == "list":
        records = store.list_open()
        if not records:
            print("No pending review records.")
            return 0
        for record in records:
            print(
                f"#{record.id} {record.inquiry_id} {record.result.case_type.value} "
                f"{record.result.priority.value} {record.status.value} {record.created_at}"
            )
        return 0
    if args.review_command == "show":
        _print_json(store.get(args.record_id))
        return 0
    if args.review_command == "decide":
        decision = ReviewDecision(args.action)
        _print_json(store.decide(args.record_id, decision, args.edited_reply))
        print("Decision recorded locally. No email was sent.")
        return 0
    raise ValueError(f"Unknown review command: {args.review_command}")


def _print_run_summaries(runs: list[dict]) -> None:
    """Print one compact, auditable block per completed provider run."""

    for index, run in enumerate(runs, start=1):
        summary = run["summary"]
        operational = summary["operational"]
        calibration = summary["calibration"]
        print(f"\nRun {index}/{len(runs)}: {summary['model']} ({summary['case_count']} cases)")
        print(
            f"  classification_accuracy={summary['classification_accuracy']} "
            f"macro_f1={summary['macro_f1']} "
            f"priority_accuracy={summary['priority_accuracy']} "
            f"urgent_recall={summary['urgent_recall']}"
        )
        print(
            f"  groundedness_rate={summary['groundedness_rate']} "
            f"reply_quality={summary['reply_quality_heuristic']} "
            f"schema_success_rate={summary['schema_success_rate']} "
            f"safety_action_accuracy={summary['safety_action_accuracy']}"
        )
        print(
            f"  calibration_ece={'n/a' if calibration['ece'] is None else calibration['ece']} "
            f"calibration_brier_score={calibration['brier_score']} "
            f"fitness_score={summary['fitness']['score']} "
            f"latency_ms_p95={operational['latency_ms_p95']} "
            f"average_cost_usd={operational['average_cost_usd']}"
        )
        print("  gates:")
        for name, passed in summary["gates"].items():
            print(f"    {'PASS' if passed else 'FAIL'} {name}")
        issues = [
            record
            for record in run["records"]
            if not record["valid"]
            or not record["classification_correct"]
            or not record["priority_correct"]
            or not record["safety_correct"]
            or not record["groundedness_pass"]
        ]
        if issues:
            print("  case issues:")
            for record in issues:
                if not record["valid"]:
                    reason = record["validation_error"] or "INVALID_RESULT"
                    if record.get("failure_detail"):
                        reason = f"{reason}: {record['failure_detail']}"
                else:
                    mismatches = []
                    if not record["classification_correct"]:
                        mismatches.append(
                            f"case_type expected={record['expected_case_type']} actual={record['actual_case_type']}"
                        )
                    if not record["priority_correct"]:
                        mismatches.append(
                            f"priority expected={record['expected_priority']} actual={record['actual_priority']}"
                        )
                    if not record["safety_correct"]:
                        mismatches.append(
                            f"safety expected={record['expected_safety_status']} actual={record['actual_safety_status']}"
                        )
                    if not record["groundedness_pass"]:
                        mismatches.append("groundedness check failed")
                    reason = "; ".join(mismatches)
                print(f"    {record['inquiry_id']}: {reason}")


def _evaluate(args: argparse.Namespace) -> int:
    specs = [spec.strip() for spec in args.models.split(",") if spec.strip()]
    agents = [TriageAgent(_load_provider(spec, args.config)) for spec in specs]
    golden_set_sha256 = verify_frozen_dataset(args.dataset)
    comparison = compare_agents(agents, load_inquiries(args.dataset), golden_set_sha256)
    summary_path, records_path = write_comparison(comparison, args.output_dir)
    _print_run_summaries(comparison["runs"])
    _print_json({"recommendation": comparison["recommendation"]})
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {records_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "triage":
            return _triage(args)
        if args.command == "review":
            return _review(args)
        if args.command == "evaluate":
            return _evaluate(args)
        raise ValueError(f"Unknown command: {args.command}")
    except (ReviewError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


def app() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    app()
