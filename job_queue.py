"""SQLite-backed job queue for async inference work.

Provides a thread-safe, WAL-enabled SQLite queue that stores job descriptors
for async image generation and editing. Jobs are persisted to disk so that
subprocess workers can read and update them safely.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobQueue:
    """SQLite-backed job queue for async inference work.

    All public methods are thread-safe. A new SQLite connection is created
    per method call so that the queue can be used safely from both threads
    and subprocess workers.
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id         TEXT PRIMARY KEY,
        status         TEXT NOT NULL DEFAULT 'queued',
        command        TEXT NOT NULL,
        params         TEXT NOT NULL,
        backend        TEXT NOT NULL DEFAULT 'thread',
        output_path    TEXT NOT NULL,
        created_at     TEXT NOT NULL DEFAULT (datetime('now')),
        started_at     TEXT,
        completed_at   TEXT,
        pid            INTEGER,
        progress       TEXT,
        error          TEXT,
        timeout_s      REAL NOT NULL DEFAULT 300.0
    );
    """

    _PURGE_SQL = """
    DELETE FROM jobs
    WHERE status IN ('completed', 'failed', 'cancelled', 'timed_out')
      AND completed_at < datetime('now', '-5 minutes');
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open or create the SQLite job database.

        Args:
            db_path: Path to the SQLite database file. If None, defaults to
                jobs.db in the same directory as this module, ensuring a
                deterministic location independent of the caller's working directory.
        """
        if db_path is None:
            db_path = Path(__file__).parent / "jobs.db"
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the jobs table if it does not already exist."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE_SQL)
                conn.commit()

    def _now_iso(self) -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row into a plain dict with deserialized JSON."""
        job = dict(row)
        if job.get("params") is not None:
            job["params"] = json.loads(job["params"])
        if job.get("progress") is not None:
            job["progress"] = json.loads(job["progress"])
        return job

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        """Delete terminal jobs older than five minutes.

        This is called automatically before every read operation.
        """
        conn.execute(self._PURGE_SQL)
        conn.commit()

    def submit(
        self,
        command: str,
        params: dict,
        output_path: str,
        backend: str = "thread",
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Insert a new job into the queue.

        Args:
            command: The MCP tool command (e.g. ``"generate_image"``).
            params: JSON-serializable dict of tool parameters.
            output_path: Absolute or relative path where the output image
                will be written.
            backend: Execution backend — ``"thread"`` or ``"subprocess"``.
            timeout_s: Maximum seconds the job may run before timing out.

        Returns:
            A job descriptor dict containing ``job_id``, ``status``,
            ``command``, ``output_path``, ``backend``, and other fields.
        """
        job_id = str(uuid.uuid4())
        params_json = json.dumps(params)
        created_at = self._now_iso()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, status, command, params, backend,
                        output_path, created_at, timeout_s
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        "queued",
                        command,
                        params_json,
                        backend,
                        output_path,
                        created_at,
                        timeout_s,
                    ),
                )
                conn.commit()

        return {
            "job_id": job_id,
            "status": "queued",
            "command": command,
            "params": params,
            "backend": backend,
            "output_path": output_path,
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "pid": None,
            "progress": None,
            "error": None,
            "timeout_s": timeout_s,
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch a single job by ID.

        Runs the lazy purge before querying.

        Args:
            job_id: The UUID job identifier.

        Returns:
            The job descriptor dict, or ``None`` if not found.
        """
        with self._lock:
            with self._connect() as conn:
                self._purge_expired(conn)
                cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    def list_jobs(
        self, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List jobs, optionally filtered by status.

        Runs the lazy purge before querying.

        Args:
            status: Filter by job status (e.g. ``"queued"``). ``None`` returns
                jobs of all statuses.
            limit: Maximum number of rows to return.

        Returns:
            A list of job descriptor dicts, most-recently created first.
        """
        with self._lock:
            with self._connect() as conn:
                self._purge_expired(conn)
                if status is not None:
                    cursor = conn.execute(
                        """
                        SELECT * FROM jobs
                        WHERE status = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (status, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM jobs
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]

    def update_status(self, job_id: str, status: str, **kwargs: Any) -> None:
        """Update a job's status and optional fields.

        Args:
            job_id: The UUID job identifier.
            status: New status string.
            **kwargs: Additional columns to update. Recognised keys:
                ``started_at``, ``completed_at``, ``pid``, ``progress``,
                ``error``.  If ``progress`` is a dict it is serialized to JSON.
        """
        allowed = {"started_at", "completed_at", "pid", "progress", "error"}
        fields = ["status = ?"]
        values: list[Any] = [status]

        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Cannot update unknown field: {key}")
            if key == "progress" and isinstance(value, dict):
                value = json.dumps(value)
            fields.append(f"{key} = ?")
            values.append(value)

        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"

        with self._lock:
            with self._connect() as conn:
                conn.execute(sql, values)
                conn.commit()

    def update_progress(self, job_id: str, progress: dict) -> None:
        """Update only the progress JSON for a job.

        Args:
            job_id: The UUID job identifier.
            progress: JSON-serializable progress dict.
        """
        self.update_status(job_id, status="running", progress=progress)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        """Cancel a queued or running job.

        Sets ``status`` to ``'cancelled'`` and ``completed_at`` to the
        current UTC timestamp.

        Args:
            job_id: The UUID job identifier.

        Returns:
            The updated job descriptor dict, or ``None`` if the job was
            not found.
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row is None:
                    return None

                completed_at = self._now_iso()
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    ("cancelled", completed_at, job_id),
                )
                conn.commit()

                job = self._row_to_dict(row)
                job["status"] = "cancelled"
                job["completed_at"] = completed_at
                return job

    def claim_next(self, backend: str) -> dict[str, Any] | None:
        """Atomically claim the oldest queued job for the given backend.

        Selects the oldest job with ``status='queued'`` and the specified
        backend, transitions it to ``'running'``, and returns it — all in
        one transaction.  If no matching queued job exists, returns ``None``.

        This replaces the non-atomic read-then-lock pattern that allowed
        duplicate job processing.

        Args:
            backend: Execution backend to filter by (``"thread"`` or
                ``"subprocess"``).

        Returns:
            The claimed job descriptor dict (with ``status='running'`` and
            ``started_at`` set), or ``None`` if the queue is empty for this
            backend.
        """
        started_at = self._now_iso()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = 'queued' AND backend = ?
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (backend,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                job_id = row["job_id"]
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', started_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (started_at, job_id),
                )
                conn.commit()

                job = self._row_to_dict(row)
                job["status"] = "running"
                job["started_at"] = started_at
                return job

    def purge_expired(self) -> None:
        """Delete terminal jobs older than five minutes.

        This is called automatically before every read, but may also be
        invoked explicitly for testing or maintenance.
        """
        with self._lock:
            with self._connect() as conn:
                self._purge_expired(conn)
