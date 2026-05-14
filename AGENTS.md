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
| `server.py` | MCP server entry point (FastMCP) |
| `pyproject.toml` | Python project config (uv), dependencies |
| `tests/` | pytest test suite |

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

## What to read

```
Always:          PLAN.md, .harness/current.md (or MCP tasks.get)
If implementing: server.py, pyproject.toml, tests/
If debugging:    mflux docs at https://github.com/filipstrand/mflux
```
