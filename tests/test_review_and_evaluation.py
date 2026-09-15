from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from enquiry_triage.agent import TriageAgent
from enquiry_triage.evaluation import compare_agents, load_inquiries, write_comparison
from enquiry_triage.models import Inquiry, ProviderResponse, ReviewDecision, Usage
from enquiry_triage.providers import DemoRuleBasedProvider
from enquiry_triage.review import ReviewError, ReviewStore


ROOT = Path(__file__).resolve().parents[1]


class BrokenProvider:
    name = "broken-provider"

    def generate(self, *, system_prompt: str, email_text: str, response_schema: dict) -> ProviderResponse:
        return ProviderResponse(model_name=self.name, content="not json", latency_ms=1.0, usage=Usage())


class ReviewAndEvaluationTests(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as directory:
            summary, records = write_comparison(comparison, Path(directory))
            self.assertTrue(summary.is_file())
            self.assertTrue(records.is_file())

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
