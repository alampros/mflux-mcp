# AGENTS.md — mflux-mcp

> **Read this file first.** It is the navigation map for every AI agent working in this repository.

## Project

**mflux-mcp** — An MCP server exposing the [mflux](https://github.com/filipstrand/mflux) image generation tool to LLM agents via the Model Context Protocol. Built with Python, FastMCP, and the mflux Python API. Runs on macOS with Apple Silicon (MLX).

## Health check (run before starting)

```bash
bash health.sh
```

If it exits non-zero, stop and report the issue. Do not proceed with tasks until health is green.

## Key files

| File | Purpose |
|------|---------|
| `PLAN.md` | Project plan with architecture, tool specs, test plan, and mflux API reference |
| `server.py` | MCP server entry point (FastMCP, registers 4 tools) |
| `mflux_cache.py` | Thread-safe lazy model cache, model registry (17 variants), HF cache detection |
| `pyproject.toml` | Python project config (uv), dependencies |
| `tests/` | pytest test suite (2 test files, 19 test classes) |
| `health.sh` | Health check script (toolchain, tests, hardware, harness) |
| `README.md` | User-facing documentation |

## Project structure

```
mflux-mcp/
├── server.py              # MCP server entry point (FastMCP, 4 tools)
├── mflux_cache.py         # Thread-safe lazy model cache & registry
├── pyproject.toml         # Python project config (uv)
├── health.sh              # Project health check script
├── PLAN.md                # Architecture & design document
├── AGENTS.md              # This file — agent navigation guide
├── README.md              # User-facing documentation
├── LICENSE                # MIT License
├── tests/
│   ├── conftest.py        # Test config (adds repo root to sys.path)
│   ├── test_server.py     # MCP tool tests (15 test classes, ~1041 lines)
│   └── test_model_cache.py # Model cache tests (4 test classes, ~214 lines)
├── output/                # Generated images directory (gitignored)
└── .harness/              # Harness data (see below)
```

## Harness data (source of truth)

| File | Purpose |
|------|---------|
| `.harness/harness.db` | SQLite: all tasks, actions, file changes, tool calls |
| `.harness/current.md` | Markdown fallback -- read this if MCP server is unavailable |
| `.harness/feature_list.json` | Human-editable task seed list |

## MCP tools (preferred)

The harness exposes tools via MCP server on port 3742. Use these instead of reading files directly.

```
actions.start        taskId agent                           -> start an action, returns actionId
actions.write        actionId section text                  -> record a section (result, blockers, ...)
actions.record_tool  actionId toolName [argsJson] [summary] -> log a tool call to the Tools dashboard
actions.record_file  actionId filePath operation [notes]    -> log a file touch to the Files dashboard
actions.complete     actionId summary                       -> close the action
actions.get          taskId                                 -> full action history for a task
tasks.add            title [slug] [description] [acceptance] -> create a new task from natural language
tasks.get            [status]                               -> list tasks (pending | in_progress | done | blocked)
tasks.claim          id                                     -> atomically claim a pending task
tasks.update         id status                              -> change task status
tasks.acceptance.update criterionId                         -> mark an acceptance criterion as met
docs.search          query                                  -> search ./docs for relevant content
```

## Workflow

```
1. INIT
   - Run health.sh -> exit 1 means stop
   - tasks.get('in_progress') -> resume if something is in progress
   - tasks.get('pending') -> pick lowest id

2. WORK  (lead -> explorer -> builder -> reviewer)
   - Each agent calls actions.start(taskId, agentName) -> actionId
   - After EVERY tool call: actions.record_tool(actionId, toolName, args, summary)
   - After EVERY file change: actions.record_file(actionId, filePath, operation, notes)
   - Closes with actions.complete(actionId, summary)

3. CLOSE
   - tasks.update(taskId, 'done')
   - Run health.sh -> must be green before closing
```

## Agent roles

| Agent | Responsibility |
|-------|---------------|
| lead | Decomposes the task into a plan, assigns sub-agents |
| explorer | Reads and maps relevant code, never writes |
| builder | Implements the plan, writes files |
| reviewer | Verifies acceptance criteria, approves or blocks |

## Technology stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Package manager | uv |
| MCP framework | FastMCP |
| Image generation | mflux (MLX native, Apple Silicon) |
| Testing | pytest, pytest-asyncio |
| Transport | stdio (default), HTTP (optional) |

## Development

### Running tests

```bash
uv run pytest tests/
```

All tests use mocks — no GPU or downloaded models are needed. The two test files cover:

- **`test_server.py`** (15 classes): Tool registration, parameter defaults, happy-path generation/editing with mock models, error handling, output path behavior, CLI transport parsing, list_models structure, metadata inspection
- **`test_model_cache.py`** (4 classes): Registry structure, cache get/hit/miss, cache clearing, lazy import verification

### What health.sh checks

1. Toolchain — `uv` and `python3` exist
2. mflux CLI — `mflux-generate-flux2` exists (installed as a uv tool)
3. Project files — `pyproject.toml` and `server.py` exist
4. Tests — `tests/` dir has `test_*.py` files, `uv run pytest tests/` passes
5. Docs — `PLAN.md` exists
6. Harness — `.harness/feature_list.json` exists and is non-empty
7. Hardware — `uname -m == arm64` (Apple Silicon)

## What to read

```
Always:          PLAN.md, .harness/current.md (or MCP tasks.get)
If implementing: server.py, mflux_cache.py, pyproject.toml, tests/
If debugging:    mflux docs at https://github.com/filipstrand/mflux
```

## Gotchas

- **Apple Silicon is a hard requirement.** mflux depends on MLX which only runs on arm64 macOS. There is no Linux or Intel Mac support.
- **mflux is both a dependency and a uv tool.** `uv sync` installs it as a Python library (imported by server.py). `uv tool install mflux` installs it as a CLI tool (checked by health.sh). Both are needed.
- **Lazy imports.** `mflux_cache.py` defers all mflux imports until the first model is loaded. This keeps `list_models` fast and avoids importing heavy ML libraries at startup. Tests verify this with `sys.modules` checks.
- **All tests use mocks.** No actual model loading or GPU inference happens in the test suite. Tests mock `ModelCache.get_model()` and the mflux model classes.
- **Generated images are gitignored.** The `output/` directory and all `*.png`, `*.jpg`, `*.jpeg`, `*.webp` files are in `.gitignore`.
- **No entry point scripts.** The server is run directly via `uv run server.py`, not through a console_scripts entry point.
- **PLAN.md references `test_tools.py`** but the actual test file is `test_server.py`. Trust the filesystem, not PLAN.md.
