from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from enquiry_triage import web


class CapturedBackgroundTasks:
    def __init__(self) -> None:
        self.task = None
        self.args = ()

    def add_task(self, task, *args) -> None:
        self.task = task
        self.args = args


class WebEvaluationJobsTests(unittest.TestCase):
    def test_evaluation_returns_before_work_and_exposes_saved_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = CapturedBackgroundTasks()
            with (
                patch.object(web, "JOB_DIR", root / "jobs"),
                patch.object(web, "DEFAULT_RESULTS", root / "results"),
                patch.object(web, "_active_job_id", None),
            ):
                response = web.evaluate(
                    web.EvaluationRequest(providers=["demo-fast", "demo-conservative"]), background
                )
                self.assertEqual(response.status_code, 202)
                job_id = json.loads(response.body)["job_id"]
                self.assertEqual(web.get_evaluation_job(job_id)["status"], "queued")
                with self.assertRaises(HTTPException) as duplicate:
                    web.evaluate(web.EvaluationRequest(providers=["demo-fast", "demo-conservative"]), CapturedBackgroundTasks())
                self.assertEqual(duplicate.exception.status_code, 409)

                background.task(*background.args)
                finished = web.get_evaluation_job(job_id)
                self.assertEqual(finished["status"], "complete")
                self.assertIn("model", finished["recommendation"])
                self.assertEqual(web.get_evaluation(finished["filename"])["comparison"]["recommendation"], finished["recommendation"])

    def test_running_job_from_previous_web_process_is_reported_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(web, "JOB_DIR", Path(directory)):
            job_id = "a" * 32
            web._save_job({"job_id": job_id, "status": "running", "server_id": "previous-process"})
            self.assertEqual(web.get_evaluation_job(job_id)["status"], "interrupted")

    def test_background_failure_is_visible_and_releases_job_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(web, "JOB_DIR", Path(directory)),
                patch.object(web, "_active_job_id", "b" * 32),
                patch.object(web, "verify_frozen_dataset", side_effect=ValueError("bad golden set")),
                patch("enquiry_triage.web.logging.exception"),
            ):
                web._run_evaluation("b" * 32, [])
                self.assertEqual(web.get_evaluation_job("b" * 32)["status"], "failed")
                self.assertIsNone(web._active_job_id)


if __name__ == "__main__":
    unittest.main()
