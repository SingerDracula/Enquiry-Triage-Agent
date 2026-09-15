from __future__ import annotations

import unittest

from enquiry_triage.models import CaseType, Confidence, Inquiry, Priority, SafetyStatus, TriageResult
from enquiry_triage.policies import groundedness_passes, unsupported_numeric_facts


class PolicyTests(unittest.TestCase):
    def test_unsupported_numeric_fact_is_detected(self) -> None:
        inquiry = Inquiry(id="grounding-001", body="My premium is 20 dollars.")
        result = TriageResult(
            case_type=CaseType.PREMIUM_BILLING,
            priority=Priority.NORMAL,
            summary="Customer asks about a premium.",
            draft_reply="Hello. Your premium will be 99 dollars. Kind regards, Customer Service.",
            confidence=Confidence(score=0.8, methodology="Test methodology for numeric fact validation."),
            safety_status=SafetyStatus.PENDING_REVIEW,
        )
        self.assertEqual(unsupported_numeric_facts(inquiry, result), ["99"])
        self.assertFalse(groundedness_passes(inquiry, result))
