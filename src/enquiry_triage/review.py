"""Local human-review queue. It has no sending capability by design."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import Inquiry, ReviewDecision, ReviewRecord, SafetyStatus, TriageAttempt, TriageResult


class ReviewError(ValueError):
    """A safe workflow error that should be shown to a reviewer."""


class ReviewStore:
    """A local queue for authorized reviewers, not an event log or email transport."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inquiry_id TEXT NOT NULL,
                    source_subject TEXT NOT NULL DEFAULT '',
                    source_body TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decision TEXT,
                    edited_reply TEXT,
                    decided_at TEXT
                )
                """
            )
            existing_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(review_records)").fetchall()
            }
            for column in ("source_subject", "source_body"):
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE review_records ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                    )
        os.chmod(self.database_path, 0o600)

    def enqueue(self, attempt: TriageAttempt, inquiry: Inquiry) -> ReviewRecord:
        """Store source content only in the private local queue so review remains meaningful."""

        if not attempt.is_valid or attempt.result is None:
            raise ReviewError("Invalid model output cannot be placed in the review queue.")
        if inquiry.id != attempt.inquiry_id:
            raise ReviewError("Inquiry and triage attempt identifiers do not match.")

        result = attempt.result
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_records
                    (inquiry_id, source_subject, source_body, status, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.inquiry_id,
                    inquiry.subject,
                    inquiry.body,
                    result.safety_status.value,
                    result.model_dump_json(),
                    created_at,
                ),
            )
            record_id = int(cursor.lastrowid)
        return ReviewRecord(
            id=record_id,
            inquiry_id=attempt.inquiry_id,
            source_subject=inquiry.subject,
            source_body=inquiry.body,
            status=result.safety_status,
            result=result,
            created_at=created_at,
        )

    def list_open(self) -> list[ReviewRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM review_records WHERE decision IS NULL ORDER BY id ASC"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get(self, record_id: int) -> ReviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_records WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            raise ReviewError(f"No review record found for id={record_id}.")
        return self._record_from_row(row)

    def decide(
        self,
        record_id: int,
        decision: ReviewDecision,
        edited_reply: str | None = None,
    ) -> ReviewRecord:
        record = self.get(record_id)
        if record.decision is not None:
            raise ReviewError(f"Review record {record_id} already has decision {record.decision.value}.")
        if decision is ReviewDecision.EDITED_AND_ACCEPTED and not (edited_reply or "").strip():
            raise ReviewError("An edited reply is required for EDITED_AND_ACCEPTED.")
        if decision is not ReviewDecision.EDITED_AND_ACCEPTED:
            edited_reply = None

        decided_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE review_records
                SET decision = ?, edited_reply = ?, decided_at = ?
                WHERE id = ? AND decision IS NULL
                """,
                (decision.value, edited_reply, decided_at, record_id),
            )
        if cursor.rowcount != 1:
            raise ReviewError(f"Review record {record_id} was changed by another reviewer.")
        payload = record.model_dump()
        payload.update(
            decision=decision,
            edited_reply=edited_reply,
            decided_at=decided_at,
        )
        return ReviewRecord.model_validate(payload)

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ReviewRecord:
        return ReviewRecord(
            id=int(row["id"]),
            inquiry_id=row["inquiry_id"],
            source_subject=row["source_subject"],
            source_body=row["source_body"],
            status=SafetyStatus(row["status"]),
            result=TriageResult.model_validate(json.loads(row["result_json"])),
            created_at=row["created_at"],
            decision=ReviewDecision(row["decision"]) if row["decision"] else None,
            edited_reply=row["edited_reply"],
            decided_at=row["decided_at"],
        )
