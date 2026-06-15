# mflux-mcp v2 — Async Queue Rewrite Plan

> **Every agent working on this rewrite MUST read this document before starting.**
> It is the single source of truth for architecture, tool contracts, and how the pieces fit together.

## Problem Statement

The current mflux-mcp server exposes synchronous MCP tools for image generation. Image inference on Apple Silicon can take 30–120+ seconds depending on model size, image dimensions, and step count. MCP tools were not designed for long-running operations — clients frequently time out, and the existing heartbeat/keepalive hacks fight against the protocol's request/response model.

## Goal

Rewrite the server so that all inference work (generation and editing) is submitted to a persistent work queue and executed asynchronously. Give callers tools to submit jobs, poll status, cancel work, and query system resources. Support two execution backends — **in-process thread** (fast, uses cached models) and **subprocess** (isolated, reloads models each time) — selectable per job.

## Architecture Overview

:::mermaid
graph TB
    Agent["LLM Agent"] -->|MCP tools| Server["server.py (FastMCP)"]

    Server -->|"submit job"| Queue["SQLite Queue (jobs.db)"]
    Server -->|"poll / cancel"| Queue

    Queue -->|"backend=thread"| ThreadWorker["In-Process Worker\n(MLX thread pool)"]
    Queue -->|"backend=subprocess"| SubprocessWorker["Subprocess Worker\n(mflux CLI / Python subprocess)"]

    ThreadWorker -->|"write status"| Queue
    SubprocessWorker -->|"write status"| Queue

    ThreadWorker --> GPU["Apple Silicon GPU\n(MLX)"]
    SubprocessWorker --> GPU

    ThreadWorker -->|"write image"| Disk["Output File\n(caller-provided path)"]
    SubprocessWorker -->|"write image"| Disk

    Server -->|"read-only"| ModelCache["Model Cache\n(in-process only)"]
    Server -->|"read-only"| SystemInfo["System Status\n(RAM, Metal, queue)"]
:::

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Queue storage | SQLite (stdlib `sqlite3`) | Zero dependencies, subprocess-safe, single file, lazy purge is trivial |
| Execution backends | Thread + Subprocess | Thread preserves model cache (fast); subprocess gives crash isolation (safe) |
| Result storage | Caller-provided output path (required) | No managed output directory. Caller decides where files go. |
| Job retention | Short TTL after terminal state (~5 min), lazy purge | Caller can poll for final status; old records cleaned on next query |
| Purge mechanism | Lazy (on read) | Every `list_jobs`/`get_job` call purges expired records first. No background timer needed. |
| Progress tracking | Job record updated by worker (phase + timestamp) | Subprocess writes directly to SQLite; thread worker updates in-process |
| Concurrency | One job at a time per backend | MLX GPU is single-stream. Semaphore for thread backend; natural for subprocess (one at a time). |

## MCP Tool Surface (10 tools)

### Inference Tools (queue-based)

| Tool | Description |
|------|-------------|
| `generate_image` | Submit a text-to-image generation job. Returns a job descriptor immediately. |
| `edit_image` | Submit an image editing job. Returns a job descriptor immediately. |

Both accept a `backend` parameter: `"thread"` (default) or `"subprocess"`.

Both **require** an `output_path` parameter — the caller specifies where the result image is written.

Return value (immediate):
```json
{
  "job_id": "uuid",
  "status": "queued",
  "command": "generate_image",
  "output_path": "/path/to/output.png",
  "backend": "thread"
}
```

### Queue Management Tools

| Tool | Description |
|------|-------------|
| `list_jobs` | List jobs in the queue. Optional filters: `status`, `limit`. |
| `get_job` | Get full details of a single job by `job_id`, including progress and timing. |
| `cancel_job` | Cancel a queued or running job. Queued jobs are removed; running subprocess jobs are killed (SIGTERM → SIGKILL). Running thread jobs are marked for cancellation (best-effort). |

### Utility Tools (synchronous, no queue)

| Tool | Description |
|------|-------------|
| `list_models` | List available mflux model variants with metadata. Unchanged from v1. |
| `get_image_metadata` | Read EXIF/XMP metadata from an mflux-generated image. Unchanged from v1. |
| `clear_cache` | Clear all cached models from the in-process LRU model cache, reclaiming unified memory. Unchanged from v1. Only affects the thread backend (subprocess jobs use their own ephemeral cache). |
| `get_system_status` | Query system RAM, MLX Metal memory, GPU info, queue snapshot, and cached models. |

## SQLite Schema

Single table, single file (`jobs.db` adjacent to `server.py`):

```sql
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'queued',
        -- queued | running | completed | failed | cancelled | timed_out
    command        TEXT NOT NULL,
        -- 'generate_image' | 'edit_image'
    params         TEXT NOT NULL,
        -- JSON blob of all tool parameters
    backend        TEXT NOT NULL DEFAULT 'thread',
        -- 'thread' | 'subprocess'
    output_path    TEXT NOT NULL,

    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    started_at     TEXT,
    completed_at   TEXT,

    pid            INTEGER,
        -- subprocess PID (NULL for thread backend)
    progress       TEXT,
        -- JSON: {"phase": "generating", "elapsed_s": 12.3} or similar
    error          TEXT,
        -- error message if status is 'failed' or 'timed_out'
    timeout_s      REAL NOT NULL DEFAULT 300.0
        -- per-job timeout in seconds
);
```

**WAL mode** should be enabled at connection time (`PRAGMA journal_mode=WAL`) to allow concurrent reads from the server while a subprocess writes status updates.

**Lazy purge query** (run before every read operation):
```sql
DELETE FROM jobs
WHERE status IN ('completed', 'failed', 'cancelled', 'timed_out')
  AND completed_at < datetime('now', '-5 minutes');
```

## Job Lifecycle

:::mermaid
stateDiagram-v2
    [*] --> queued : submit
    queued --> running : worker picks up
    queued --> cancelled : cancel_job

    running --> completed : success
    running --> failed : error
    running --> timed_out : timeout exceeded
    running --> cancelled : cancel_job (kill subprocess / flag thread)

    completed --> [*] : purged after TTL
    failed --> [*] : purged after TTL
    timed_out --> [*] : purged after TTL
    cancelled --> [*] : purged after TTL
:::

### Thread Backend Flow

1. Job submitted → row inserted with `status='queued'`, `backend='thread'`
2. In-process worker loop (asyncio task) picks up next queued thread job
3. Worker acquires MLX semaphore, loads model from cache, runs inference
4. Worker updates `progress` column in SQLite as phases change (loading → generating → saving)
5. On completion: writes image to `output_path`, sets `status='completed'`, `completed_at=now()`
6. On error: sets `status='failed'`, `error=<message>`, `completed_at=now()`

### Subprocess Backend Flow

1. Job submitted → row inserted with `status='queued'`, `backend='subprocess'`
2. In-process dispatcher spawns a Python subprocess for the job
3. Subprocess opens `jobs.db`, updates its own row: `status='running'`, `pid=os.getpid()`, `started_at=now()`
4. Subprocess loads model, runs inference, updates `progress` column periodically
5. On completion: writes image to `output_path`, sets `status='completed'`, `completed_at=now()`
6. On error: sets `status='failed'`, `error=<message>`, `completed_at=now()`
7. Server-side timeout monitor: if `started_at + timeout_s < now()` and status is still `running`, send SIGTERM to PID, then SIGKILL after grace period, set `status='timed_out'`

### Cancellation

- **Queued jobs**: Set `status='cancelled'`, `completed_at=now()`. Worker skips them.
- **Running subprocess jobs**: Send SIGTERM to PID. If still alive after 5s, SIGKILL. Set `status='cancelled'`.
- **Running thread jobs**: Set a cancellation flag. Best-effort — MLX inference is not interruptible mid-step, so the job will complete the current operation then check the flag. Document this limitation.

## `get_system_status` Tool

Returns a single dict with system resource information. All fields are best-effort — gracefully return `null` for anything unavailable.

```json
{
  "ram": {
    "total_gb": 36.0,
    "available_gb": 12.4,
    "percent_used": 65.5
  },
  "metal": {
    "active_mb": 4200,
    "peak_mb": 8100,
    "cache_mb": 320
  },
  "chip": {
    "name": "Apple M3 Max",
    "gpu_cores": 40,
    "recommended_max_gb": 36.0
  },
  "queue": {
    "queued": 2,
    "running": 1
  },
  "cached_models": ["flux2-klein-4b (q8)", "z-image (q8)"]
}
```

**Implementation notes:**
- RAM: `psutil.virtual_memory()` or `os.sysconf` (prefer psutil if available, fall back to sysconf)
- Metal: `mlx.core.metal.get_active_memory()`, `get_peak_memory()`, `get_cache_memory()`. These trigger MLX import — guard with try/except and return nulls if MLX hasn't been imported yet.
- Chip: `mlx.core.metal.device_info()` — same lazy-import guard.
- Queue: Single SQL `SELECT COUNT(*) ... GROUP BY status` query.
- Cached models: Read from the in-process `ModelCache` instance.

## Existing Code: What Stays, What Goes

| File | Disposition |
|------|-------------|
| `mflux_cache.py` | **Keep as-is.** Thread-safe LRU model cache (`max_models=1`, `OrderedDict`-based eviction), model registry, HF cache detection — all still needed for the thread backend. The LRU eviction and `clear()` method are used by the `clear_cache` MCP tool. |
| `server.py` | **Rewrite.** New tool definitions, queue integration, worker loops. Remove heartbeat machinery. |
| `tests/test_server.py` | **Rewrite.** New tests for queue-based tools, both backends, job lifecycle, system status. |
| `tests/test_model_cache.py` | **Keep as-is.** Model cache tests are independent of the server rewrite. |
| `tests/conftest.py` | **Modify.** Add fixtures for SQLite test database, mock subprocess, etc. |
| `pyproject.toml` | **Modify.** May need `psutil` dependency for system status. |
| `PLAN.md` | **Keep as historical reference.** This document (`REWRITE-PLAN.md`) supersedes it for v2 work. |
| `health.sh` | **Modify.** Update to check for new files, possibly `jobs.db` presence. |
| `AGENTS.md` | **Modify.** Update file table, tool list, and project structure for v2. |

## New Files

| File | Purpose |
|------|---------|
| `job_queue.py` | SQLite queue manager class. Schema creation, job CRUD, lazy purge, WAL mode setup. |
| `worker.py` | Worker logic for both backends. Thread worker loop, subprocess entry point, timeout monitor. |
| `subprocess_runner.py` | Standalone script that a subprocess executes. Opens `jobs.db`, runs inference, updates status. Self-contained so it can be invoked as `python subprocess_runner.py <job_id> <db_path>`. |

## Implementation Phases

Each phase is a discrete harness task. Phases should be executed in order — later phases depend on earlier ones. **Every agent working on a phase must read this entire document first** to understand how their piece fits into the whole.

---

### Phase 1: Queue Infrastructure (`job_queue.py`)

**What:** Create the `JobQueue` class that manages the SQLite database — schema creation, WAL mode, job insertion, status queries, status updates, lazy purge, and cancellation. This is the foundation everything else builds on.

**Key concerns:**
- Thread-safe and subprocess-safe (multiple processes may write concurrently)
- WAL mode enabled at connection time
- All timestamp handling in UTC
- Lazy purge runs before every read operation
- Job ID generation (UUID4)
- `params` stored as JSON text

**Does NOT include:** Worker logic, MCP tool definitions, or inference code.

**Acceptance criteria:**
- `JobQueue` class with full CRUD operations
- Schema auto-created on first connection
- Lazy purge tested and working
- Concurrent access from multiple threads tested
- Comprehensive unit tests in a new `tests/test_job_queue.py`

---

### Phase 2: Subprocess Runner (`subprocess_runner.py`)

**What:** Create the standalone script that executes an inference job in its own process. It receives a `job_id` and `db_path`, opens the database, reads job params, loads the model, runs inference, writes the output image, and updates job status throughout.

**Key concerns:**
- Must be runnable as `python subprocess_runner.py <job_id> <db_path>`
- Opens its own SQLite connection (WAL mode)
- Updates `status`, `pid`, `started_at`, `progress`, `completed_at`, `error` in the job row
- Catches all exceptions and writes error to the job row before exiting
- Handles both `generate_image` and `edit_image` commands
- Uses `mflux_cache.py` for model loading (fresh cache per process — that's fine)
- Writes image to the `output_path` from the job params

**Does NOT include:** Being spawned by anything — that's Phase 4. This phase just makes the script work when invoked directly.

**Acceptance criteria:**
- Script runs standalone and processes a job from the database
- All job phases update the `progress` column
- Errors are caught and written to the job row
- Output image is written to the specified path
- Unit tests mock the mflux model (same pattern as existing tests)

---

### Phase 3: Worker Logic (`worker.py`)

**What:** Create the worker module containing: (a) the in-process thread worker loop that processes `backend='thread'` jobs, (b) the subprocess dispatcher that spawns `subprocess_runner.py` for `backend='subprocess'` jobs, and (c) the timeout monitor that kills overdue subprocess jobs.

**Key concerns:**
- Thread worker: asyncio background task, acquires MLX semaphore, uses existing `ModelCache`, updates job row in SQLite
- Subprocess dispatcher: asyncio background task, spawns `subprocess_runner.py` via `asyncio.create_subprocess_exec`, tracks the `Process` handle
- Timeout monitor: periodic check (every ~10s) for subprocess jobs past their `timeout_s`, sends SIGTERM then SIGKILL
- Concurrency: one thread job at a time (MLX constraint), one subprocess at a time (GPU constraint), but a thread job and subprocess job should NOT run simultaneously (they'd fight over the GPU)
- Cancellation support: thread jobs check a flag between phases; subprocess jobs receive SIGTERM

**Does NOT include:** MCP tool definitions. This module exposes a `WorkerManager` (or similar) that the server calls.

**Acceptance criteria:**
- Thread worker processes jobs and updates status
- Subprocess dispatcher spawns and tracks child processes
- Timeout monitor kills overdue jobs
- Cancellation works for both backends
- GPU exclusivity enforced (one job at a time across both backends)
- Unit tests with mocked inference and mocked subprocess

---

### Phase 4: MCP Server Rewrite (`server.py`)

**What:** Rewrite `server.py` with the new tool surface. Wire up the `JobQueue` and `WorkerManager`. Start background worker tasks on server startup. Keep `list_models` and `get_image_metadata` unchanged.

**New/changed tools:**
- `generate_image` — validates params, inserts job into queue, returns job descriptor
- `edit_image` — validates params, inserts job into queue, returns job descriptor
- `list_jobs` — queries queue with optional filters
- `get_job` — returns full job details
- `cancel_job` — cancels a job via `WorkerManager`
- `get_system_status` — new, queries RAM/Metal/queue/cache
- `list_models` — unchanged
- `get_image_metadata` — unchanged
- `clear_cache` — unchanged (clears in-process LRU model cache)

**Key concerns:**
- Background worker tasks must start when the server starts (FastMCP lifecycle hooks or startup in `main()`)
- All submission tools return immediately (no awaiting inference)
- `get_system_status` must handle MLX not-yet-imported gracefully
- Remove all heartbeat/keepalive machinery from v1
- Remove the old synchronous `generate_image`/`edit_image` implementations
- CLI args (`--transport`, `--port`) remain

**Does NOT include:** The queue, worker, or subprocess runner — those are imported from Phases 1–3.

**Acceptance criteria:**
- All 10 tools registered and callable
- Submission tools return job descriptors immediately
- `get_system_status` returns valid data (with graceful nulls)
- Integration test: submit → poll → verify completion (with mocked inference)
- Old heartbeat code fully removed
- Server starts cleanly, background workers initialize

---

### Phase 5: Test Suite Rewrite (`tests/`)

**What:** Rewrite `tests/test_server.py` to cover the new tool contracts and job lifecycle. Add integration-style tests that exercise the full submit → poll → complete flow. Keep `tests/test_model_cache.py` unchanged.

**Key concerns:**
- All tests still use mocks — no real GPU inference
- Test the queue lifecycle: submit, poll, complete, purge
- Test both backends (thread and subprocess — mock the subprocess spawn)
- Test cancellation for both backends
- Test timeout handling
- Test `get_system_status` with mocked `psutil`/`mlx` values
- Test lazy purge behavior
- Test error propagation (model load failure, inference error, file write error)
- Ensure `list_models`, `get_image_metadata`, and `clear_cache` tests still pass (they shouldn't need changes)

**Acceptance criteria:**
- `uv run pytest tests/` passes
- Coverage of all 10 tools
- Job lifecycle tested end-to-end (with mocks)
- Both backends tested
- Edge cases: concurrent submissions, cancel during different phases, timeout, DB errors

---

### Phase 6: Documentation and Cleanup

**What:** Update all project documentation and configuration to reflect the v2 architecture. Run `health.sh` and fix anything that breaks.

**Key concerns:**
- Update `AGENTS.md`: file table, project structure, tool list, gotchas
- Update `README.md`: new tool descriptions, usage examples, backend explanation
- Update `pyproject.toml`: add `psutil` if needed, add new modules to `[tool.setuptools]`
- Update `health.sh`: check for new files (`job_queue.py`, `worker.py`, `subprocess_runner.py`)
- Verify `health.sh` exits 0
- Verify full test suite passes

**Acceptance criteria:**
- `health.sh` passes
- `uv run pytest tests/` passes
- `AGENTS.md` accurately describes the v2 codebase
- `README.md` documents all 10 tools with examples

---

## Appendix: Parameter Reference

### `generate_image` Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | str | *required* | Text description of the image to generate |
| `output_path` | str | *required* | File path to write the output image |
| `model` | str | `"flux2-klein-4b"` | Model variant name |
| `width` | int | `1024` | Image width in pixels |
| `height` | int | `1024` | Image height in pixels |
| `steps` | int | `4` | Number of inference steps |
| `seed` | int \| None | `None` | Random seed (auto-generated if None) |
| `quantize` | int \| None | `8` | Quantization (4, 8, or None) |
| `lora_style` | str \| None | `None` | LoRA style name |
| `backend` | str | `"thread"` | `"thread"` or `"subprocess"` |
| `timeout` | float | `300.0` | Per-job timeout in seconds |

### `edit_image` Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `image_paths` | list[str] | *required* | Input image file paths |
| `prompt` | str | *required* | Text description of the edit |
| `output_path` | str | *required* | File path to write the output image |
| `model` | str | `"flux2-klein-edit"` | Edit model variant name |
| `steps` | int | `4` | Number of inference steps |
| `seed` | int \| None | `None` | Random seed (auto-generated if None) |
| `quantize` | int \| None | `8` | Quantization (4, 8, or None) |
| `lora_style` | str \| None | `None` | LoRA style name |
| `backend` | str | `"thread"` | `"thread"` or `"subprocess"` |
| `timeout` | float | `300.0` | Per-job timeout in seconds |
