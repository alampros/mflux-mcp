# AGENTS.md

## What this is

MCP server wrapping [mflux](https://github.com/filipstrand/mflux) for local image generation on Apple Silicon via MLX. Exposes 10 tools (generate, edit, upscale, queue management, metadata, system status) over the Model Context Protocol.

## Commands

```bash
uv sync                      # install deps (Python 3.12+)
uv run pytest tests/         # run all tests (~200, all mocked, no GPU needed)
uv run pytest tests/test_server.py::TestGenerateImage -k test_name  # single test
uv run server.py             # start server (stdio transport)
uv run server.py --transport http --port 8080  # HTTP transport
```

No linter, formatter, or type checker is configured. No CI workflows exist.

## Architecture

Flat single-package layout -- all modules live at repo root, no `src/` directory.

| File | Role |
|---|---|
| `server.py` | FastMCP entry point, all 10 MCP tool definitions, CLI arg parsing |
| `job_queue.py` | SQLite-backed job queue (WAL mode, thread-safe) |
| `worker.py` | `WorkerManager` -- async job loop, thread + subprocess backends, GPU lock |
| `subprocess_runner.py` | Standalone script invoked as isolated subprocess per job |
| `mflux_cache.py` | Thread-safe LRU model cache, model registry (`_REGISTRY`), HF cache helpers |

Key wiring:
- `server.py:main()` creates `JobQueue`, `ModelCache`, and `WorkerManager`, then calls `mcp.run()`
- Only one job runs at a time (GPU exclusivity via `asyncio.Lock` in `WorkerManager`)
- **stdout is the MCP protocol channel** -- all logging must go to stderr (`_log()` helper)
- `jobs.db` is auto-created at repo root at runtime (gitignored)

## Testing

- pytest-asyncio with `asyncio_mode = "auto"` -- no `@pytest.mark.asyncio` needed on async tests (but existing tests do use it; either works)
- `conftest.py` adds repo root to `sys.path` so flat modules are importable
- All tests mock mflux/MLX -- no Apple Silicon, GPU, or downloaded models required
- Fixtures: `mock_queue`, `mock_cache`, `mock_worker_manager` in `conftest.py`
- Tests patch `server._queue`, `server._cache`, `server._worker_manager` module globals
- Test DB files (`test.db`) are gitignored via `.gitignore` patterns

## Gotchas

- The `mflux_cache.py` `_REGISTRY` dict is the single source of truth for supported models -- `server.py` reads it for validation and `list_models`
- Model class imports in `mflux_cache.py` are lazy (`_lazy_imports()`) to avoid loading MLX at module import time
- `subprocess_runner.py` is invoked as `python subprocess_runner.py <job_id> <db_path>` by the worker -- it reads/writes the same SQLite DB
