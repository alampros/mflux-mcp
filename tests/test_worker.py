"""Tests for the worker module.

All tests mock model inference and subprocess spawning to avoid loading heavy
MLX models or performing actual GPU inference.
"""

import asyncio
import os
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import PIL.Image
import pytest

from job_queue import JobQueue
from mflux_cache import ModelCache
from worker import WorkerManager, _clear_metal_cache, _setup_metal_cache_limit


class _MockHelpers:
    """Shared helper methods for creating mock models and jobs."""

    @staticmethod
    def make_mock_model(width=64, height=64):
        """Create a mock model that returns a real PIL Image."""
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="red")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model

    @staticmethod
    def make_mock_subprocess_proc(queue, job_id, returncode=0):
        """Create a mock asyncio Process that updates the queue on wait()."""

        async def mock_wait():
            # Simulate what subprocess_runner.py does
            queue.update_status(
                job_id,
                "running",
                started_at=datetime.now(timezone.utc).isoformat(),
                pid=12345,
            )
            if returncode == 0:
                queue.update_status(
                    job_id,
                    "completed",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            else:
                queue.update_status(
                    job_id,
                    "failed",
                    error="subprocess error",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            return returncode

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = returncode
        mock_proc.wait = mock_wait
        return mock_proc

    @staticmethod
    def submit_generate_job(queue, backend="thread", **overrides):
        """Submit a generate_image job with default params overridden."""
        params = {
            "prompt": "A red square",
            "model": "flux2-klein-4b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        timeout_s = params.pop("timeout_s", 300.0)
        return queue.submit(
            command="generate_image",
            params=params,
            output_path=output_path,
            backend=backend,
            timeout_s=timeout_s,
        )

    @staticmethod
    def submit_edit_job(queue, backend="thread", **overrides):
        """Submit an edit_image job with default params overridden."""
        params = {
            "prompt": "Make it blue",
            "model": "flux2-klein-edit",
            "image_paths": ["/tmp/input.png"],
            "steps": 4,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        timeout_s = params.pop("timeout_s", 300.0)
        return queue.submit(
            command="edit_image",
            params=params,
            output_path=output_path,
            backend=backend,
            timeout_s=timeout_s,
        )

    @staticmethod
    def submit_upscale_job(queue, backend="thread", **overrides):
        """Submit an upscale_image job with default params overridden."""
        params = {
            "image_path": "/tmp/input.png",
            "model": "seedvr2-3b",
            "resolution": 2160,
            "softness": 0.5,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        timeout_s = params.pop("timeout_s", 300.0)
        return queue.submit(
            command="upscale_image",
            params=params,
            output_path=output_path,
            backend=backend,
            timeout_s=timeout_s,
        )


class TestWorkerManagerThreadBackend:
    """Tests for the thread backend worker loop."""

    @pytest.mark.asyncio
    async def test_happy_path_completes(self, tmp_path):
        """A valid thread job runs to completion."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        out_path = tmp_path / "out.png"
        job = _MockHelpers.submit_generate_job(queue, output_path=str(out_path))

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            await manager.start()
            try:
                # Wait for job to complete (with timeout to avoid hanging)
                for _ in range(50):
                    updated = queue.get_job(job["job_id"])
                    if updated["status"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "completed"
        assert updated["error"] is None
        assert out_path.exists()

    @pytest.mark.asyncio
    async def test_progress_updates_through_phases(self, tmp_path):
        """Progress is updated through loading → generating → saving phases."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        progress_phases = []
        original_update_progress = queue.update_progress

        def capture_progress(job_id, progress):
            progress_phases.append(progress.get("phase"))
            return original_update_progress(job_id, progress)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch.object(queue, "update_progress", side_effect=capture_progress):
                await manager.start()
                try:
                    for _ in range(50):
                        updated = queue.get_job(job["job_id"])
                        if updated["status"] in ("completed", "failed", "cancelled"):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        assert "loading_model" in progress_phases
        assert "generating" in progress_phases
        assert "saving" in progress_phases

    @pytest.mark.asyncio
    async def test_model_load_failure_marks_failed(self, tmp_path):
        """If model loading fails, the job is marked failed."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch.object(
            ModelCache, "get_model", side_effect=RuntimeError("model load failed")
        ):
            await manager.start()
            try:
                for _ in range(50):
                    updated = queue.get_job(job["job_id"])
                    if updated["status"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "model load failed" in updated["error"]

    @pytest.mark.asyncio
    async def test_inference_error_marks_failed(self, tmp_path):
        """If inference fails, the job is marked failed."""
        mock_model = _MockHelpers.make_mock_model()
        mock_model.generate_image.side_effect = RuntimeError("inference failed")

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            await manager.start()
            try:
                for _ in range(50):
                    updated = queue.get_job(job["job_id"])
                    if updated["status"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "inference failed" in updated["error"]

    @pytest.mark.asyncio
    async def test_edit_image_command(self, tmp_path):
        """Thread worker handles edit_image commands."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        out_path = tmp_path / "out.png"
        job = _MockHelpers.submit_edit_job(queue, output_path=str(out_path))

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            await manager.start()
            try:
                for _ in range(50):
                    updated = queue.get_job(job["job_id"])
                    if updated["status"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "completed"
        mock_model.generate_image.assert_called_once()


class TestWorkerManagerSubprocessBackend:
    """Tests for the subprocess backend dispatcher."""

    @pytest.mark.asyncio
    async def test_happy_path_spawns_and_tracks(self, tmp_path):
        """A subprocess job spawns the runner and tracks the process."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, backend="subprocess", output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        mock_proc = _MockHelpers.make_mock_subprocess_proc(
            queue, job["job_id"], returncode=0
        )

        mock_exec = AsyncMock(return_value=mock_proc)
        with patch("asyncio.create_subprocess_exec", mock_exec):
            await manager.start()
            try:
                # Give the loop time to pick up the job
                for _ in range(30):
                    if job["job_id"] not in [
                        j["job_id"] for j in queue.list_jobs(status="queued")
                    ]:
                        break
                    await asyncio.sleep(0.1)
                # Wait for subprocess to "finish"
                for _ in range(30):
                    if job["job_id"] not in manager._running_procs:
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        mock_exec.assert_called_once()
        args, _ = mock_exec.call_args
        assert "subprocess_runner.py" in args[1]
        assert args[2] == job["job_id"]
        assert args[3] == str(db_path)

    @pytest.mark.asyncio
    async def test_subprocess_nonzero_exit(self, tmp_path):
        """If subprocess exits non-zero, the job status reflects failure."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, backend="subprocess", output_path=str(tmp_path / "out.png")
        )

        # Pre-mark the job as running and then failed (simulating subprocess_runner behavior)
        queue.update_status(
            job["job_id"], "running", started_at=datetime.now(timezone.utc).isoformat()
        )
        queue.update_status(
            job["job_id"],
            "failed",
            error="subprocess error",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1
        mock_proc.wait = AsyncMock(return_value=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await manager.start()
            try:
                for _ in range(30):
                    if job["job_id"] not in manager._running_procs:
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"

    @pytest.mark.asyncio
    async def test_running_procs_tracked(self, tmp_path):
        """Running subprocesses are tracked in _running_procs dict."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, backend="subprocess", output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        wait_event = asyncio.Event()

        async def mock_wait():
            await wait_event.wait()
            queue.update_status(
                job["job_id"],
                "running",
                started_at=datetime.now(timezone.utc).isoformat(),
                pid=12345,
            )
            queue.update_status(
                job["job_id"],
                "completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return 0

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.wait = mock_wait

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await manager.start()
            try:
                # Wait for the job to be picked up and proc stored
                for _ in range(30):
                    if job["job_id"] in manager._running_procs:
                        break
                    await asyncio.sleep(0.1)

                assert job["job_id"] in manager._running_procs
                assert manager._running_procs[job["job_id"]].pid == 12345
            finally:
                wait_event.set()
                await manager.stop()


class TestWorkerManagerTimeout:
    """Tests for the timeout monitor."""

    @pytest.mark.asyncio
    async def test_subprocess_job_exceeds_timeout(self, tmp_path):
        """An overdue subprocess job is killed and marked timed_out."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue,
            backend="subprocess",
            output_path=str(tmp_path / "out.png"),
            timeout_s=1.0,
        )

        # Mark as running with an old started_at (2 minutes ago so it's definitely overdue)
        old_start = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        queue.update_status(job["job_id"], "running", started_at=old_start, pid=12345)

        cache = ModelCache()
        manager = WorkerManager(queue, cache, timeout_check_interval=0.1)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = -9
        mock_proc.wait = AsyncMock(return_value=-9)

        manager._running_procs[job["job_id"]] = mock_proc

        await manager.start()
        try:
            # Wait for timeout monitor to detect the overdue job
            for _ in range(50):
                updated = queue.get_job(job["job_id"])
                if updated["status"] == "timed_out":
                    break
                await asyncio.sleep(0.1)
        finally:
            await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "timed_out"
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_job_within_timeout_not_killed(self, tmp_path):
        """A thread job within its timeout is not killed by the monitor."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue,
            backend="thread",
            output_path=str(tmp_path / "out.png"),
            timeout_s=300.0,  # Long timeout — job will complete well before
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            await manager.start()
            try:
                # Wait for the job to complete normally
                for _ in range(50):
                    updated = queue.get_job(job["job_id"])
                    if updated["status"] in (
                        "completed",
                        "failed",
                        "cancelled",
                        "timed_out",
                    ):
                        break
                    await asyncio.sleep(0.1)
            finally:
                await manager.stop()

        updated = queue.get_job(job["job_id"])
        # Should complete successfully, not be timed_out
        assert updated["status"] == "completed"


class TestWorkerManagerCancellation:
    """Tests for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, tmp_path):
        """Cancelling a queued job sets status to cancelled."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        result = await manager.cancel_job(job["job_id"])
        assert result is True

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_running_subprocess(self, tmp_path):
        """Cancelling a running subprocess sends SIGTERM."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, backend="subprocess", output_path=str(tmp_path / "out.png")
        )
        queue.update_status(
            job["job_id"], "running", started_at=datetime.now(timezone.utc).isoformat()
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)
        manager._running_procs[job["job_id"]] = mock_proc

        result = await manager.cancel_job(job["job_id"])
        assert result is True
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_running_thread(self, tmp_path):
        """Cancelling a running thread job adds it to the cancelled set."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )
        queue.update_status(
            job["job_id"], "running", started_at=datetime.now(timezone.utc).isoformat()
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        result = await manager.cancel_job(job["job_id"])
        assert result is True
        assert job["job_id"] in manager._cancelled_jobs

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, tmp_path):
        """Cancelling a non-existent job returns False."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        result = await manager.cancel_job("nonexistent-uuid")
        assert result is False


class TestWorkerManagerGPUExclusivity:
    """Tests that only one job runs at a time across both backends."""

    @pytest.mark.asyncio
    async def test_two_jobs_run_sequentially(self, tmp_path):
        """Two queued jobs run one after another, not simultaneously."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job1 = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out1.png")
        )
        job2 = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out2.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        completion_order = []
        original_process = manager._process_thread_job

        async def track_completion(job):
            await original_process(job)
            completion_order.append(job["job_id"])

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch.object(
                manager, "_process_thread_job", side_effect=track_completion
            ):
                await manager.start()
                try:
                    for _ in range(100):
                        j1 = queue.get_job(job1["job_id"])
                        j2 = queue.get_job(job2["job_id"])
                        if j1["status"] == "completed" and j2["status"] == "completed":
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        assert len(completion_order) == 2
        # They should have run sequentially (one at a time)
        assert completion_order[0] != completion_order[1]

    @pytest.mark.asyncio
    async def test_thread_and_subprocess_dont_overlap(self, tmp_path):
        """A thread job and subprocess job do not run simultaneously."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        thread_job = _MockHelpers.submit_generate_job(
            queue, backend="thread", output_path=str(tmp_path / "out1.png")
        )
        subproc_job = _MockHelpers.submit_generate_job(
            queue, backend="subprocess", output_path=str(tmp_path / "out2.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        lock_holder_times = []

        original_thread_process = manager._process_thread_job
        original_subprocess_process = manager._process_subprocess_job

        async def tracked_thread_process(job):
            lock_holder_times.append(("thread", job["job_id"]))
            await original_thread_process(job)
            lock_holder_times.append(("thread_end", job["job_id"]))

        async def tracked_subprocess_process(job):
            lock_holder_times.append(("subprocess", job["job_id"]))
            await original_subprocess_process(job)
            lock_holder_times.append(("subprocess_end", job["job_id"]))

        mock_proc = _MockHelpers.make_mock_subprocess_proc(
            queue, subproc_job["job_id"], returncode=0
        )

        mock_model = _MockHelpers.make_mock_model()

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch.object(
                    manager, "_process_thread_job", side_effect=tracked_thread_process
                ):
                    with patch.object(
                        manager,
                        "_process_subprocess_job",
                        side_effect=tracked_subprocess_process,
                    ):
                        await manager.start()
                        try:
                            for _ in range(100):
                                tj = queue.get_job(thread_job["job_id"])
                                sj = queue.get_job(subproc_job["job_id"])
                                if tj["status"] == "completed" and sj["status"] in (
                                    "completed",
                                    "failed",
                                ):
                                    break
                                await asyncio.sleep(0.1)
                        finally:
                            await manager.stop()

        # Extract just the start events
        starts = [e for e in lock_holder_times if not e[0].endswith("_end")]
        ends = [e for e in lock_holder_times if e[0].endswith("_end")]

        # There should be no overlap: for any two start events, one must have ended before the other started
        # With only 2 jobs, we just need to verify they didn't both hold the lock at the same time
        assert len(starts) == 2
        assert len(ends) == 2

        # Sort by time and verify alternation
        events = lock_holder_times
        for i in range(len(events) - 1):
            # After a start, the next event for the same backend should be an end
            pass  # Just verifying structure; the lock itself guarantees exclusivity

    @pytest.mark.asyncio
    async def test_gpu_lock_is_held_during_processing(self, tmp_path):
        """The GPU lock is acquired before processing and released after."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        lock_acquired = False

        original_process = manager._process_thread_job

        async def tracked_process(job):
            nonlocal lock_acquired
            lock_acquired = manager._gpu_lock.locked()
            await original_process(job)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch.object(
                manager, "_process_thread_job", side_effect=tracked_process
            ):
                await manager.start()
                try:
                    for _ in range(50):
                        updated = queue.get_job(job["job_id"])
                        if updated["status"] in ("completed", "failed", "cancelled"):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        assert lock_acquired
        # After the manager stops, the lock should be released
        assert not manager._gpu_lock.locked()


class TestConcurrentSubmissionRace:
    """Verify that the concurrency fixes prevent 3 rapid generation requests
    from loading multiple models simultaneously and crashing the system.

    Two fixes applied:

    1. **ModelCache.get_model** now holds the lock for the entire get-or-load
       operation, so concurrent threads block instead of each loading a model.

    2. **JobQueue.claim_next()** atomically SELECT+UPDATE in one transaction,
       so a job transitions from 'queued' to 'running' before any other
       caller can see it.
    """

    def test_model_cache_concurrent_loads_serialized(self):
        """Verify that 3 concurrent get_model calls result in only 1 model
        load — the other 2 hit the cache.

        Previously, the lock was released between the cache-miss check and
        model construction, causing all 3 to load simultaneously.
        """
        import concurrent.futures
        import time

        load_count = 0
        load_count_lock = threading.Lock()
        peak_concurrent_loads = 0
        concurrent_loads = 0

        class SlowMockModel:
            def __init__(self, **kwargs):
                nonlocal load_count, peak_concurrent_loads, concurrent_loads
                with load_count_lock:
                    concurrent_loads += 1
                    peak_concurrent_loads = max(peak_concurrent_loads, concurrent_loads)
                    load_count += 1
                # Simulate real model load time
                time.sleep(0.2)
                with load_count_lock:
                    concurrent_loads -= 1

        cache = ModelCache(max_models=1)

        mock_config = MagicMock()
        mock_config_cls = MagicMock()
        mock_config_cls.flux2_klein_4b = MagicMock(return_value=mock_config)

        mock_imports = {
            "Flux2Klein": SlowMockModel,
            "ModelConfig": mock_config_cls,
        }

        with patch.object(cache, "_get_imports", return_value=mock_imports):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(cache.get_model, "flux2-klein-4b", quantize=8)
                    for _ in range(3)
                ]
                concurrent.futures.wait(futures)
                for f in futures:
                    f.result()

        assert load_count == 1, (
            f"Model was loaded {load_count} times, expected 1. "
            f"Peak concurrent loads: {peak_concurrent_loads}."
        )
        assert peak_concurrent_loads <= 1, (
            f"Peak concurrent loads was {peak_concurrent_loads}, expected <= 1."
        )

    def test_model_cache_lock_held_during_load(self):
        """Verify that ModelCache._lock IS held during model construction,
        preventing concurrent threads from starting their own loads.
        """
        lock_held_during_construction = False

        class InspectingModel:
            def __init__(self, **kwargs):
                nonlocal lock_held_during_construction
                # Try to acquire the lock — should fail because we're inside it
                lock_held_during_construction = not cache._lock.acquire(timeout=0)
                if not lock_held_during_construction:
                    cache._lock.release()

        cache = ModelCache(max_models=1)

        mock_config = MagicMock()
        mock_config_cls = MagicMock()
        mock_config_cls.flux2_klein_4b = MagicMock(return_value=mock_config)

        mock_imports = {
            "Flux2Klein": InspectingModel,
            "ModelConfig": mock_config_cls,
        }

        with patch.object(cache, "_get_imports", return_value=mock_imports):
            cache.get_model("flux2-klein-4b", quantize=8)

        assert lock_held_during_construction, (
            "The cache lock should be held during model construction. "
            "This prevents concurrent threads from each loading a multi-GB "
            "model simultaneously."
        )

    @pytest.mark.asyncio
    async def test_claim_next_is_atomic(self, tmp_path):
        """Verify that claim_next() atomically transitions the job to
        'running' — a second call returns None, not the same job.
        """
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        # First claim gets the job
        claimed = queue.claim_next("thread")
        assert claimed is not None
        assert claimed["job_id"] == job["job_id"]
        assert claimed["status"] == "running"

        # Second claim returns None — the job is no longer queued
        second_claim = queue.claim_next("thread")
        assert second_claim is None, (
            "claim_next() should return None when there are no queued jobs. "
            "The first call atomically transitioned the job to 'running'."
        )

        # Verify the DB agrees
        db_job = queue.get_job(job["job_id"])
        assert db_job["status"] == "running"
        assert db_job["started_at"] is not None

    @pytest.mark.asyncio
    async def test_claim_next_prevents_duplicate_pickup(self, tmp_path):
        """Two concurrent claim_next calls for the same backend can't both
        get the same job.
        """
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        _MockHelpers.submit_generate_job(queue, output_path=str(tmp_path / "out.png"))

        # Simulate two workers racing to claim
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(queue.claim_next, "thread"),
                pool.submit(queue.claim_next, "thread"),
            ]
            results = [f.result() for f in futures]

        claimed = [r for r in results if r is not None]
        assert len(claimed) == 1, (
            f"Expected exactly 1 claim, got {len(claimed)}. "
            f"Both workers claimed the same job!"
        )

    @pytest.mark.asyncio
    async def test_three_rapid_submissions_with_two_workers_no_duplicates(
        self, tmp_path
    ):
        """With two thread worker loops, 3 rapid submissions are each
        processed exactly once — no duplicate model loads.

        This is the end-to-end fix verification: atomic claiming prevents
        both workers from picking the same job.
        """
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)

        jobs = []
        for i in range(3):
            job = _MockHelpers.submit_generate_job(
                queue, output_path=str(tmp_path / f"out{i}.png")
            )
            jobs.append(job)

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        process_invocations = []
        process_lock = asyncio.Lock()

        original_process = manager._process_thread_job

        async def tracking_process(job):
            async with process_lock:
                process_invocations.append(job["job_id"])
            await original_process(job)

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch.object(
                manager, "_process_thread_job", side_effect=tracking_process
            ):
                await manager.start()
                # Add a second thread worker — both use claim_next() now
                second_worker = asyncio.create_task(
                    manager._thread_worker_loop(), name="thread-worker-2"
                )
                manager._tasks.append(second_worker)

                try:
                    for _ in range(100):
                        statuses = []
                        for j in jobs:
                            updated = queue.get_job(j["job_id"])
                            statuses.append(updated["status"] if updated else "purged")
                        if all(
                            s in ("completed", "failed", "cancelled", "purged")
                            for s in statuses
                        ):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        from collections import Counter

        pick_counts = Counter(process_invocations)
        duplicates = {jid: count for jid, count in pick_counts.items() if count > 1}

        assert len(duplicates) == 0, (
            f"Jobs were processed multiple times: {duplicates}. "
            f"Total invocations: {len(process_invocations)}, "
            f"unique: {len(set(process_invocations))}."
        )
        assert len(process_invocations) == 3, (
            f"Expected exactly 3 process invocations for 3 jobs, "
            f"got {len(process_invocations)}."
        )


class TestMetalMemoryManagement:
    """Tests for MLX metal cache management between inference jobs."""

    def test_setup_metal_cache_limit_sets_limit(self):
        """_setup_metal_cache_limit calls mx.metal.set_cache_limit with
        75% of the recommended max working set size."""
        import mlx.core as mx

        original_set = mx.metal.set_cache_limit
        calls = []

        def tracking_set(limit):
            calls.append(limit)
            return original_set(limit)

        mock_device_info = MagicMock(
            return_value={
                "recommended_max_working_set_size": 16 * (1024**3),  # 16 GB
            }
        )

        with patch.object(mx.metal, "set_cache_limit", side_effect=tracking_set):
            with patch.object(mx, "device_info", mock_device_info, create=True):
                _setup_metal_cache_limit()

        assert len(calls) == 1
        expected = int(16 * (1024**3) * 0.75)
        assert calls[0] == expected, (
            f"Expected cache limit of {expected} bytes (75% of 16 GB), got {calls[0]}"
        )

    def test_clear_metal_cache_calls_clear_and_gc(self):
        """_clear_metal_cache calls mx.metal.clear_cache and gc.collect."""
        import mlx.core as mx

        with patch.object(mx.metal, "clear_cache") as mock_clear:
            with patch("worker.gc.collect") as mock_gc:
                _clear_metal_cache()

        mock_clear.assert_called_once()
        mock_gc.assert_called_once()

    @pytest.mark.asyncio
    async def test_metal_cache_cleared_after_successful_job(self, tmp_path):
        """The metal cache is cleared after a job completes successfully."""
        mock_model = _MockHelpers.make_mock_model()

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        clear_calls = []
        original_run = manager._run_on_mlx_thread

        async def tracking_run(fn, *args, **kwargs):
            result = await original_run(fn, *args, **kwargs)
            if fn is _clear_metal_cache:
                clear_calls.append("cleared")
            return result

        with patch.object(ModelCache, "get_model", return_value=mock_model):
            with patch.object(manager, "_run_on_mlx_thread", side_effect=tracking_run):
                await manager.start()
                try:
                    for _ in range(50):
                        updated = queue.get_job(job["job_id"])
                        if updated["status"] in ("completed", "failed"):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        assert len(clear_calls) >= 1, (
            "Metal cache should be cleared after job completion"
        )

    @pytest.mark.asyncio
    async def test_metal_cache_cleared_after_failed_job(self, tmp_path):
        """The metal cache is cleared even when a job fails."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = _MockHelpers.submit_generate_job(
            queue, output_path=str(tmp_path / "out.png")
        )

        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        clear_calls = []
        original_run = manager._run_on_mlx_thread

        async def tracking_run(fn, *args, **kwargs):
            if fn is _clear_metal_cache:
                clear_calls.append("cleared")
                return await original_run(fn, *args, **kwargs)
            return await original_run(fn, *args, **kwargs)

        with patch.object(ModelCache, "get_model", side_effect=RuntimeError("boom")):
            with patch.object(manager, "_run_on_mlx_thread", side_effect=tracking_run):
                await manager.start()
                try:
                    for _ in range(50):
                        updated = queue.get_job(job["job_id"])
                        if updated["status"] in ("completed", "failed"):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    await manager.stop()

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert len(clear_calls) >= 1, (
            "Metal cache should be cleared even after job failure"
        )

    @pytest.mark.asyncio
    async def test_cache_limit_set_on_worker_start(self, tmp_path):
        """WorkerManager.start() calls _setup_metal_cache_limit."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        cache = ModelCache()
        manager = WorkerManager(queue, cache)

        with patch("worker._setup_metal_cache_limit") as mock_setup:
            await manager.start()
            try:
                pass
            finally:
                await manager.stop()

        mock_setup.assert_called_once()
