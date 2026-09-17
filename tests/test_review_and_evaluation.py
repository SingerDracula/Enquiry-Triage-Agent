from __future__ import annotations

import tempfile
import unittest
import os
import json
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from enquiry_triage.agent import TriageAgent
from enquiry_triage.cli import build_parser
from enquiry_triage.evaluation import compare_agents, default_results_dir, load_inquiries, write_comparison
from enquiry_triage.models import Inquiry, ProviderResponse, ReviewDecision, Usage
from enquiry_triage.providers import DemoRuleBasedProvider
from enquiry_triage.review import ReviewError, ReviewStore
from enquiry_triage.web import DEFAULT_RESULTS as WEB_DEFAULT_RESULTS
from enquiry_triage.web import get_evaluation, list_evaluations


ROOT = Path(__file__).resolve().parents[1]


class BrokenProvider:
    name = "broken-provider"

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        return ProviderResponse(model_name=self.name, content="not json", latency_ms=1.0, usage=Usage())


class ReviewAndEvaluationTests(unittest.TestCase):
    def test_cli_and_web_save_evaluations_in_project_folder_by_default(self) -> None:
        expected = ROOT / "evaluation_results"
        self.assertEqual(default_results_dir(), expected)
        self.assertEqual(build_parser().parse_args(["evaluate"]).output_dir, expected)
        self.assertEqual(WEB_DEFAULT_RESULTS, expected)

    def test_review_requires_valid_attempt_and_never_sends(self) -> None:
        inquiry = Inquiry(id="review-001", body="Please explain my policy coverage.")
        attempt = TriageAgent(DemoRuleBasedProvider()).triage(inquiry)
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "review.db")
            record = store.enqueue(attempt, inquiry)
            self.assertIsNone(record.decision)
            self.assertEqual(record.source_body, inquiry.body)
            self.assertEqual(store.get(record.id).source_body, inquiry.body)
            self.assertEqual(os.stat(Path(directory) / "review.db").st_mode & 0o777, 0o600)
            decided = store.decide(record.id, ReviewDecision.EDITED_AND_ACCEPTED, "Edited safe reply")
            self.assertEqual(decided.decision, ReviewDecision.EDITED_AND_ACCEPTED)
            self.assertEqual(decided.edited_reply, "Edited safe reply")
            with self.assertRaises(ReviewError):
                store.decide(record.id, ReviewDecision.DISCARDED)

    def test_invalid_attempt_cannot_enter_review_queue(self) -> None:
        attempt = TriageAgent(BrokenProvider()).triage(Inquiry(id="bad-review-001", body="Hello"))
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "review.db")
            with self.assertRaises(ReviewError):
                store.enqueue(attempt, Inquiry(id="bad-review-001", body="Hello"))

    def test_comparison_writes_raw_records_and_summary(self) -> None:
        golden = load_inquiries(ROOT / "data" / "golden_set.jsonl")
        comparison = compare_agents(
            [TriageAgent(DemoRuleBasedProvider("fast")), TriageAgent(DemoRuleBasedProvider("conservative"))],
            golden,
        )
        self.assertEqual(len(comparison["runs"]), 2)
        self.assertEqual(len(comparison["golden_ids"]), len(golden))
        self.assertIn("fitness", comparison["runs"][0]["summary"])
        self.assertIn("failure_detail", comparison["runs"][0]["records"][0])
        for run in comparison["runs"]:
            self.assertEqual(len(run["records"]), len(golden))
            for record in run["records"]:
                self.assertIn("actual_case_type", record)
                self.assertIn("actual_priority", record)
                self.assertIn("actual_safety_status", record)
                self.assertIn("groundedness_pass", record)
                self.assertIn("reply_quality_score", record)
                self.assertIn("draft_reply", record)
        with tempfile.TemporaryDirectory() as directory:
            summary, records = write_comparison(comparison, Path(directory))
            self.assertTrue(summary.is_file())
            self.assertTrue(records.is_file())
            self.assertEqual(
                json.loads(summary.read_text(encoding="utf-8"))["runs"][0]["records"][0]["draft_reply"],
                comparison["runs"][0]["records"][0]["draft_reply"],
            )
            self.assertNotIn("draft_reply", records.read_text(encoding="utf-8").splitlines()[0])

    def test_csv_output_escapes_formula_like_identifiers(self) -> None:
        comparison = {
            "runs": [
                {
                    "summary": {"model": "=unsafe-model"},
                    "records": [{"inquiry_id": "@unsafe-id", "valid": True}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            _, records = write_comparison(comparison, Path(directory))
            output = records.read_text(encoding="utf-8")
            self.assertIn("'=unsafe-model", output)
            self.assertIn("'@unsafe-id", output)

    def test_saved_evaluations_can_be_listed_and_loaded_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = "comparison_20260917T120000Z.json"
            (root / filename).write_text(json.dumps({
                "run_at": "2026-09-17T12:00:00Z",
                "golden_ids": ["golden-pq-01"],
                "runs": [{"summary": {"model": "demo-fast"}, "records": []}],
            }), encoding="utf-8")
            (root / "comparison_20260917T130000Z.json").write_text("not json", encoding="utf-8")
            with patch("enquiry_triage.web.DEFAULT_RESULTS", root):
                listed = list_evaluations()["results"]
                self.assertEqual([item["filename"] for item in listed], [filename])
                self.assertEqual(listed[0]["models"], ["demo-fast"])
                self.assertEqual(get_evaluation(filename)["comparison"]["runs"][0]["summary"]["model"], "demo-fast")
                with self.assertRaises(HTTPException) as invalid:
                    get_evaluation("../config.toml")
                self.assertEqual(invalid.exception.status_code, 404)
                with self.assertRaises(HTTPException) as corrupt:
                    get_evaluation("comparison_20260917T130000Z.json")
                self.assertEqual(corrupt.exception.status_code, 422)
