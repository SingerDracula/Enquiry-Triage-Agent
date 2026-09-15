from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from enquiry_triage.evaluation import load_inquiries, verify_frozen_dataset
from enquiry_triage.models import CaseType, SafetyStatus


ROOT = Path(__file__).resolve().parents[1]


class DataContractTests(unittest.TestCase):
    def test_dataset_has_required_coverage_and_frozen_golden_subset(self) -> None:
        golden_path = ROOT / "data" / "golden_set.jsonl"
        golden = load_inquiries(golden_path)
        development = load_inquiries(ROOT / "data" / "dev_set.jsonl")
        all_inquiries = golden + development

        self.assertGreaterEqual(len(all_inquiries), 25)
        self.assertEqual(len({record.id for record in all_inquiries}), len(all_inquiries))
        self.assertEqual({record.expected_case_type for record in golden}, set(CaseType))
        self.assertTrue(any(record.expected_safety_status is SafetyStatus.REFUSE_AND_ESCALATE for record in all_inquiries))
        self.assertTrue(any("complaint" in record.body.lower() for record in all_inquiries))
        self.assertTrue(any("ignore previous" in record.body.lower() for record in all_inquiries))
        self.assertTrue(any("missing" in record.body.lower() for record in all_inquiries))
        self.assertTrue(all(record.reference_reply_notes for record in golden))

    def test_golden_set_hash_is_pinned(self) -> None:
        golden_path = ROOT / "data" / "golden_set.jsonl"
        digest = hashlib.sha256(golden_path.read_bytes()).hexdigest()
        expected = (ROOT / "data" / "golden_set.sha256").read_text(encoding="utf-8").strip()
        self.assertEqual(digest, expected)
        self.assertEqual(verify_frozen_dataset(golden_path), expected)
