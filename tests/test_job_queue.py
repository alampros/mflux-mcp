"""Tests for the SQLite-backed job queue.

All tests use a temporary on-disk database — no real DB files are left behind.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from job_queue import JobQueue


class TestJobQueueSchema:
    """Tests for database creation, WAL mode, and table schema."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_db_file_created(self, tmp_path):
        db = tmp_path / "test_jobs.db"
        assert not db.exists()
        JobQueue(db_path=db)
        assert db.exists()

    def test_wal_mode_enabled(self, tmp_path):
        queue = self._make_queue(tmp_path)
        db_path = queue._db_path
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_table_exists_with_correct_columns(self, tmp_path):
        queue = self._make_queue(tmp_path)
        db_path = queue._db_path
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(jobs)")
        rows = cursor.fetchall()
        conn.close()

        columns = {row[1] for row in rows}
        expected = {
            "job_id",
            "status",
            "command",
            "params",
            "backend",
            "output_path",
            "created_at",
            "started_at",
            "completed_at",
            "pid",
            "progress",
            "error",
            "timeout_s",
        }
        assert columns == expected


class TestJobQueueSubmit:
    """Tests for job submission."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_submit_returns_job_descriptor(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(
            command="generate_image",
            params={"prompt": "a cat", "width": 512},
            output_path="output/cat.png",
        )
        assert isinstance(job, dict)
        assert "job_id" in job
        assert job["status"] == "queued"
        assert job["command"] == "generate_image"
        assert job["output_path"] == "output/cat.png"
        assert job["backend"] == "thread"
        assert job["timeout_s"] == 300.0
        assert job["params"] == {"prompt": "a cat", "width": 512}
        assert job["started_at"] is None
        assert job["completed_at"] is None
        assert job["pid"] is None
        assert job["progress"] is None
        assert job["error"] is None

    def test_submit_generates_uuid4_job_id(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(
            command="generate_image",
            params={},
            output_path="out.png",
        )
        # UUID4 has 36 chars with hyphens
        assert len(job["job_id"]) == 36
        assert job["job_id"].count("-") == 4

    def test_submit_allows_custom_backend_and_timeout(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(
            command="edit_image",
            params={},
            output_path="out.png",
            backend="subprocess",
            timeout_s=600.0,
        )
        assert job["backend"] == "subprocess"
        assert job["timeout_s"] == 600.0

    def test_submit_stores_params_as_json(self, tmp_path):
        queue = self._make_queue(tmp_path)
        queue.submit(
            command="generate_image",
            params={"prompt": "a dog", "steps": 8},
            output_path="out.png",
        )
        # Verify raw JSON in DB
        conn = sqlite3.connect(str(queue._db_path))
        cursor = conn.execute("SELECT params FROM jobs LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        parsed = json.loads(row[0])
        assert parsed == {"prompt": "a dog", "steps": 8}


class TestJobQueueQuery:
    """Tests for get_job and list_jobs."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_get_job_returns_correct_data(self, tmp_path):
        queue = self._make_queue(tmp_path)
        submitted = queue.submit(
            command="generate_image",
            params={"prompt": "hello"},
            output_path="out.png",
        )
        fetched = queue.get_job(submitted["job_id"])
        assert fetched is not None
        assert fetched["job_id"] == submitted["job_id"]
        assert fetched["command"] == "generate_image"
        assert fetched["params"] == {"prompt": "hello"}

    def test_get_job_missing_returns_none(self, tmp_path):
        queue = self._make_queue(tmp_path)
        result = queue.get_job("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_list_jobs_returns_all(self, tmp_path):
        queue = self._make_queue(tmp_path)
        queue.submit(command="a", params={}, output_path="a.png")
        queue.submit(command="b", params={}, output_path="b.png")
        jobs = queue.list_jobs()
        assert len(jobs) == 2
        # Most recent first
        assert jobs[0]["command"] == "b"
        assert jobs[1]["command"] == "a"

    def test_list_jobs_with_status_filter(self, tmp_path):
        queue = self._make_queue(tmp_path)
        j1 = queue.submit(command="a", params={}, output_path="a.png")
        j2 = queue.submit(command="b", params={}, output_path="b.png")
        queue.update_status(j1["job_id"], "running")
        queue.update_status(j2["job_id"], "completed", completed_at=queue._now_iso())

        queued = queue.list_jobs(status="queued")
        assert len(queued) == 0

        running = queue.list_jobs(status="running")
        assert len(running) == 1
        assert running[0]["command"] == "a"

        completed = queue.list_jobs(status="completed")
        assert len(completed) == 1
        assert completed[0]["command"] == "b"

    def test_list_jobs_respects_limit(self, tmp_path):
        queue = self._make_queue(tmp_path)
        for i in range(5):
            queue.submit(command=f"job_{i}", params={}, output_path="out.png")
        jobs = queue.list_jobs(limit=2)
        assert len(jobs) == 2
        assert jobs[0]["command"] == "job_4"
        assert jobs[1]["command"] == "job_3"


class TestJobQueueUpdate:
    """Tests for update_status and update_progress."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_update_status_transitions(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_status(job["job_id"], "running")
        fetched = queue.get_job(job["job_id"])
        assert fetched["status"] == "running"

    def test_update_status_sets_timestamps(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_status(
            job["job_id"], "running", started_at="2024-06-01T12:00:00+00:00"
        )
        queue.update_status(job["job_id"], "completed", completed_at=queue._now_iso())
        fetched = queue.get_job(job["job_id"])
        assert fetched["started_at"] == "2024-06-01T12:00:00+00:00"
        assert fetched["completed_at"] is not None
        assert len(fetched["completed_at"]) > 0

    def test_update_status_sets_pid_and_error(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_status(job["job_id"], "failed", pid=1234, error="Out of memory")
        fetched = queue.get_job(job["job_id"])
        assert fetched["pid"] == 1234
        assert fetched["error"] == "Out of memory"

    def test_update_status_rejects_unknown_field(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        with pytest.raises(ValueError, match="unknown field"):
            queue.update_status(job["job_id"], "running", unknown_field="x")

    def test_update_progress_serializes_dict(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_progress(job["job_id"], {"phase": "generating", "step": 3})
        fetched = queue.get_job(job["job_id"])
        assert fetched["progress"] == {"phase": "generating", "step": 3}
        assert fetched["status"] == "running"

    def test_update_status_serializes_progress_dict(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_status(job["job_id"], "running", progress={"step": 2})
        fetched = queue.get_job(job["job_id"])
        assert fetched["progress"] == {"step": 2}


class TestJobQueuePurge:
    """Tests for lazy purge of expired terminal jobs."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def _insert_terminal_job(
        self, queue: JobQueue, status: str, completed_at: str
    ) -> str:
        """Insert a job directly with a specific completed_at timestamp."""
        job_id = "00000000-0000-0000-0000-000000000001"
        conn = sqlite3.connect(str(queue._db_path))
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, command, params, backend,
                output_path, created_at, completed_at, timeout_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                status,
                "generate_image",
                "{}",
                "thread",
                "out.png",
                "2024-01-01T00:00:00+00:00",
                completed_at,
                300.0,
            ),
        )
        conn.commit()
        conn.close()
        return job_id

    def test_purge_removes_expired_completed_jobs(self, tmp_path):
        queue = self._make_queue(tmp_path)
        # A job completed 10 minutes ago should be purged
        old_time = "2024-01-01T00:00:00+00:00"
        self._insert_terminal_job(queue, "completed", old_time)
        queue.purge_expired()
        assert queue.get_job("00000000-0000-0000-0000-000000000001") is None

    def test_purge_keeps_non_expired_jobs(self, tmp_path):
        queue = self._make_queue(tmp_path)
        # A job completed just now should be kept
        from datetime import datetime, timezone

        recent = datetime.now(timezone.utc).isoformat()
        job_id = self._insert_terminal_job(queue, "completed", recent)
        queue.purge_expired()
        assert queue.get_job(job_id) is not None

    def test_purge_keeps_non_terminal_jobs(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        queue.update_status(job["job_id"], "running")
        queue.purge_expired()
        assert queue.get_job(job["job_id"]) is not None

    def test_purge_runs_automatically_on_get_job(self, tmp_path):
        queue = self._make_queue(tmp_path)
        old_time = "2024-01-01T00:00:00+00:00"
        self._insert_terminal_job(queue, "failed", old_time)
        # get_job should trigger purge
        assert queue.get_job("00000000-0000-0000-0000-000000000001") is None

    def test_purge_runs_automatically_on_list_jobs(self, tmp_path):
        queue = self._make_queue(tmp_path)
        old_time = "2024-01-01T00:00:00+00:00"
        self._insert_terminal_job(queue, "cancelled", old_time)
        # list_jobs should trigger purge
        jobs = queue.list_jobs()
        ids = [j["job_id"] for j in jobs]
        assert "00000000-0000-0000-0000-000000000001" not in ids


class TestJobQueueCancel:
    """Tests for cancelling jobs."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_cancel_queued_job_succeeds(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        cancelled = queue.cancel(job["job_id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert cancelled["completed_at"] is not None

        fetched = queue.get_job(job["job_id"])
        assert fetched["status"] == "cancelled"

    def test_cancel_non_existent_returns_none(self, tmp_path):
        queue = self._make_queue(tmp_path)
        result = queue.cancel("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_cancel_sets_completed_at(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        cancelled = queue.cancel(job["job_id"])
        assert cancelled["completed_at"] is not None
        assert len(cancelled["completed_at"]) > 0


class TestJobQueueConcurrency:
    """Tests for thread-safe concurrent access."""

    def _make_queue(self, tmp_path: Path) -> JobQueue:
        db = tmp_path / "test_jobs.db"
        return JobQueue(db_path=db)

    def test_concurrent_submits_and_reads(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job_ids = []
        errors = []

        def submit_jobs():
            try:
                for i in range(20):
                    job = queue.submit(
                        command="generate_image",
                        params={"idx": i},
                        output_path=f"out_{i}.png",
                    )
                    job_ids.append(job["job_id"])
            except Exception as e:
                errors.append(e)

        def read_jobs():
            try:
                for _ in range(20):
                    queue.list_jobs(limit=10)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(4):
            t = threading.Thread(target=submit_jobs)
            threads.append(t)
            t.start()
        for _ in range(4):
            t = threading.Thread(target=read_jobs)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        # 4 submitters * 20 jobs each = 80 jobs
        assert len(job_ids) == 80
        # All IDs should be unique
        assert len(set(job_ids)) == 80

    def test_concurrent_updates(self, tmp_path):
        queue = self._make_queue(tmp_path)
        job = queue.submit(command="a", params={}, output_path="a.png")
        errors = []

        def update_job():
            try:
                for i in range(50):
                    queue.update_progress(job["job_id"], {"step": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_job) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent updates: {errors}"
        # Job should still be readable
        fetched = queue.get_job(job["job_id"])
        assert fetched is not None
        assert fetched["status"] == "running"
        assert isinstance(fetched["progress"], dict)
