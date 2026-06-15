"""Worker module for mflux-mcp async job execution.

Provides a WorkerManager that processes inference jobs from the SQLite queue
using two backends:
- Thread: in-process using ModelCache (fast, uses cached models)
- Subprocess: isolated subprocess via subprocess_runner.py (safe, reloads models)

Only one job runs at a time across both backends (GPU exclusivity).
"""

import asyncio
import concurrent.futures
import functools
import gc
import io
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from job_queue import JobQueue
from mflux_cache import ModelCache


def _log(msg: str) -> None:
    """Write a log line to stderr."""
    print(f"[mflux-worker] {msg}", file=sys.stderr, flush=True)


def _setup_metal_cache_limit() -> None:
    """Set the MLX metal cache limit to a fraction of available GPU memory.

    This prevents the metal allocator from caching unlimited intermediate
    tensors between inference runs.  Without a cap, sequential generations
    accumulate cached metal buffers until the system runs out of unified
    memory.
    """
    try:
        import mlx.core as mx

        _device_info = getattr(mx, "device_info", None) or getattr(
            mx.metal, "device_info", None
        )
        info = _device_info() if _device_info else {}
        rec_max = info.get(
            "recommended_max_working_set_size",
            info.get("max_recommended_working_set_size", 0),
        )
        if rec_max > 0:
            # Use 75% of the recommended max as the cache limit
            limit = int(rec_max * 0.75)
            mx.metal.set_cache_limit(limit)
            _log(
                f"Metal cache limit set to {limit / (1024**3):.1f} GB "
                f"(75% of {rec_max / (1024**3):.1f} GB recommended max)"
            )
        else:
            _log("Could not determine GPU memory size; metal cache limit not set")
    except Exception as exc:
        _log(f"Failed to set metal cache limit: {exc}")


def _clear_metal_cache() -> None:
    """Clear the MLX metal cache and run garbage collection.

    Called after each job completes to free cached GPU allocations before
    the next job begins.  Without this, intermediate tensors from previous
    inference runs stay resident in the metal cache, and memory usage
    grows monotonically across jobs.
    """
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:
        pass  # MLX not available or clear_cache failed — not fatal
    gc.collect()


class WorkerManager:
    """Manages background workers that process inference jobs from the queue."""

    def __init__(
        self, queue: JobQueue, cache: ModelCache, timeout_check_interval: float = 10.0
    ) -> None:
        """Initialize the worker manager.

        Args:
            queue: JobQueue instance for reading/updating jobs.
            cache: ModelCache instance for thread backend model loading.
            timeout_check_interval: Seconds between timeout monitor checks.
        """
        self._queue = queue
        self._cache = cache
        self._timeout_check_interval = timeout_check_interval
        self._gpu_lock = asyncio.Lock()  # One job at a time across both backends
        self._cancelled_jobs: set[str] = set()  # Job IDs flagged for cancellation
        self._running_procs: dict[
            str, asyncio.subprocess.Process
        ] = {}  # job_id → Process
        self._tasks: list[asyncio.Task] = []  # Background tasks
        self._running = False
        # Dedicated single-thread executor for MLX GPU work (same constraint as server.py)
        self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-gpu"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background worker tasks."""
        _setup_metal_cache_limit()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._thread_worker_loop(), name="thread-worker"),
            asyncio.create_task(
                self._subprocess_worker_loop(), name="subprocess-worker"
            ),
            asyncio.create_task(self._timeout_monitor_loop(), name="timeout-monitor"),
        ]
        _log("WorkerManager started")

    async def stop(self) -> None:
        """Stop background worker tasks and clean up."""
        self._running = False
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        # Kill any running subprocesses
        for job_id, proc in list(self._running_procs.items()):
            _log(f"Killing subprocess for job {job_id} (pid={proc.pid})")
            proc.kill()
        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._mlx_executor.shutdown(wait=False)
        _log("WorkerManager stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: The UUID job identifier.

        Returns:
            True if an action was taken, False if not found or not cancellable.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return False

        status = job.get("status")
        backend = job.get("backend")

        if status == "queued":
            self._queue.cancel(job_id)
            _log(f"Cancelled queued job {job_id}")
            return True

        if status == "running":
            if backend == "subprocess":
                proc = self._running_procs.get(job_id)
                if proc is not None:
                    _log(f"Sending SIGTERM to subprocess job {job_id} (pid={proc.pid})")
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        _log(
                            f"Subprocess job {job_id} did not exit in 5s, sending SIGKILL"
                        )
                        proc.kill()
                        await proc.wait()
                    self._queue.update_status(
                        job_id, "cancelled", completed_at=self._now_iso()
                    )
                    _log(f"Cancelled running subprocess job {job_id}")
                    return True
            elif backend == "thread":
                self._cancelled_jobs.add(job_id)
                _log(f"Flagged thread job {job_id} for cancellation (best-effort)")
                return True

        return False

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    async def _thread_worker_loop(self) -> None:
        """Background loop that polls for and processes thread backend jobs."""
        while self._running:
            try:
                job = None
                async with self._gpu_lock:
                    job = self._queue.claim_next("thread")
                    if job is not None:
                        await self._process_thread_job(job)
                if job is None:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log(f"Thread worker loop error: {exc}")
                await asyncio.sleep(1)

    async def _subprocess_worker_loop(self) -> None:
        """Background loop that polls for and dispatches subprocess backend jobs."""
        while self._running:
            try:
                job = None
                async with self._gpu_lock:
                    job = self._queue.claim_next("subprocess")
                    if job is not None:
                        await self._process_subprocess_job(job)
                if job is None:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log(f"Subprocess worker loop error: {exc}")
                await asyncio.sleep(1)

    async def _timeout_monitor_loop(self) -> None:
        """Periodic check for overdue running jobs."""
        while self._running:
            try:
                await asyncio.sleep(self._timeout_check_interval)
            except asyncio.CancelledError:
                raise

            try:
                running_jobs = self._queue.list_jobs(status="running", limit=100)
                now = datetime.now(timezone.utc)
                for job in running_jobs:
                    started_at_str = job.get("started_at")
                    timeout_s = job.get("timeout_s", 300.0)
                    if started_at_str is None:
                        continue
                    try:
                        started_at = datetime.fromisoformat(started_at_str)
                        if started_at.tzinfo is None:
                            started_at = started_at.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue

                    elapsed = (now - started_at).total_seconds()
                    if elapsed > timeout_s:
                        job_id = job["job_id"]
                        backend = job.get("backend")
                        _log(f"Job {job_id} timed out after {elapsed:.1f}s")

                        if backend == "subprocess":
                            proc = self._running_procs.get(job_id)
                            if proc is not None:
                                proc.terminate()
                                try:
                                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                                except asyncio.TimeoutError:
                                    proc.kill()
                                    await proc.wait()

                        self._queue.update_status(
                            job_id,
                            "timed_out",
                            error=f"Timed out after {elapsed:.1f}s (limit: {timeout_s}s)",
                            completed_at=self._now_iso(),
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log(f"Timeout monitor error: {exc}")

    # ------------------------------------------------------------------
    # Job processing helpers
    # ------------------------------------------------------------------

    async def _process_thread_job(self, job: dict[str, Any]) -> None:
        """Process a single thread backend job."""
        job_id = job["job_id"]
        params = job["params"]
        command = job["command"]
        output_path = job["output_path"]

        _log(f"Processing thread job {job_id} ({command})")

        # Job is already 'running' from claim_next(); just record the pid
        self._queue.update_status(job_id, "running", pid=os.getpid())

        try:
            # Phase: loading model
            self._queue.update_progress(job_id, {"phase": "loading_model"})
            model_name = params.get("model", "flux2-klein-4b")
            quantize = params.get("quantize", 8)
            lora_style = params.get("lora_style")

            loaded_model = await self._run_on_mlx_thread(
                self._cache.get_model,
                model_name,
                quantize=quantize,
                lora_style=lora_style,
            )

            if self._is_cancelled(job_id):
                self._handle_cancelled(job_id)
                return

            # Handle seed
            seed = params.get("seed")
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            # Phase: inference
            if command == "generate_image":
                self._queue.update_progress(job_id, {"phase": "generating"})
                result = await self._run_on_mlx_thread(
                    loaded_model.generate_image,
                    seed=seed,
                    prompt=params["prompt"],
                    num_inference_steps=params.get("steps", 4),
                    width=params.get("width", 1024),
                    height=params.get("height", 1024),
                )
            elif command == "edit_image":
                self._queue.update_progress(job_id, {"phase": "editing"})
                class_key = ModelCache._REGISTRY[model_name][0]
                if class_key == "FIBOEdit":
                    inference_kwargs = dict(
                        seed=seed,
                        prompt=params["prompt"],
                        image_path=params["image_paths"][0],
                        num_inference_steps=params.get("steps", 4),
                    )
                else:
                    inference_kwargs = dict(
                        seed=seed,
                        prompt=params["prompt"],
                        image_paths=params["image_paths"],
                        num_inference_steps=params.get("steps", 4),
                    )
                result = await self._run_on_mlx_thread(
                    loaded_model.generate_image, **inference_kwargs
                )
            elif command == "upscale_image":
                self._queue.update_progress(job_id, {"phase": "upscaling"})
                result = await self._run_on_mlx_thread(
                    loaded_model.generate_image,
                    seed=seed,
                    image_path=params["image_path"],
                    resolution=params.get("resolution", 2160),
                    softness=params.get("softness", 0.5),
                )
            else:
                raise ValueError(f"Unknown command: {command}")

            if self._is_cancelled(job_id):
                self._handle_cancelled(job_id)
                return

            # Phase: saving
            self._queue.update_progress(job_id, {"phase": "saving"})
            abs_path = os.path.abspath(output_path)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            buf = io.BytesIO()
            result.image.save(buf, format="PNG")
            with open(abs_path, "wb") as f:
                f.write(buf.getvalue())

            # Mark completed
            self._queue.update_status(job_id, "completed", completed_at=self._now_iso())
            _log(f"Thread job {job_id} completed")

        except Exception as e:
            _log(f"Thread job {job_id} failed: {e}")
            self._queue.update_status(
                job_id,
                "failed",
                error=str(e),
                completed_at=self._now_iso(),
            )
        finally:
            # Free MLX metal cache and Python references between jobs.
            # Without this, intermediate tensors from inference accumulate
            # in the metal cache across sequential jobs until OOM.
            await self._run_on_mlx_thread(_clear_metal_cache)

    async def _process_subprocess_job(self, job: dict[str, Any]) -> None:
        """Spawn a subprocess to process a single subprocess backend job."""
        job_id = job["job_id"]
        _log(f"Spawning subprocess for job {job_id}")

        runner_path = Path(__file__).parent / "subprocess_runner.py"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(runner_path),
            job_id,
            str(self._queue._db_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._running_procs[job_id] = proc
        try:
            await proc.wait()
            _log(f"Subprocess for job {job_id} exited with code {proc.returncode}")
        finally:
            self._running_procs.pop(job_id, None)

    # ------------------------------------------------------------------
    # Cancellation helpers
    # ------------------------------------------------------------------

    def _is_cancelled(self, job_id: str) -> bool:
        """Check whether a thread job has been flagged for cancellation."""
        return job_id in self._cancelled_jobs

    def _handle_cancelled(self, job_id: str) -> None:
        """Update job status to cancelled and remove from the cancellation set."""
        self._cancelled_jobs.discard(job_id)
        self._queue.update_status(job_id, "cancelled", completed_at=self._now_iso())
        _log(f"Thread job {job_id} cancelled mid-flight")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def _run_on_mlx_thread(self, fn, *args, **kwargs):
        """Run *fn* on the dedicated MLX GPU thread and return the result."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._mlx_executor, functools.partial(fn, *args, **kwargs)
        )

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()
