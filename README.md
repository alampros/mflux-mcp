# mflux-mcp

An MCP server exposing [mflux](https://github.com/filipstrand/mflux) image generation to LLM agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

## What is this?

mflux-mcp wraps the mflux Python API in a set of MCP tools that any MCP-compatible LLM agent (Claude Desktop, OpenCode, etc.) can call to generate images, edit images, inspect metadata, and list available models — all running locally on Apple Silicon via MLX.

No cloud API keys. No GPU servers. Just your Mac.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **macOS** | Apple Silicon (M1 or later) — mflux depends on MLX |
| **Python** | 3.12 or later |
| **uv** | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) — used for dependency management and running the server |

## Installation

### Quick start (zero-clone)

If you just want to use mflux-mcp as an MCP tool — no need to clone the repo:

```bash
uvx --from "git+https://github.com/alampros/mflux-mcp.git" mflux-mcp
```

This installs the server from git and runs it in one command. To pick up new versions after the repo is updated:

```bash
uv cache clean
```

### Local development setup

```bash
# Clone the repo
git clone git@github.com:alampros/mflux-mcp.git
cd mflux-mcp

# Install Python dependencies
uv sync
```

## Usage

### Running the server

The server supports two transport modes:

```bash
# Default: stdio transport (for MCP client integration)
uv run server.py

# HTTP transport on a custom port
uv run server.py --transport http --port 8080
```

### MCP client integration

#### OpenCode

Add to your OpenCode config (`opencode.jsonc`):

**Using uvx (zero-clone):**

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mflux": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/alampros/mflux-mcp.git", "mflux-mcp"]
    }
  }
}
```

**Using a local clone:**

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mflux": {
      "type": "local",
      "command": ["uv", "run", "server.py"],
      "environment": {
        "PATH": "/path/to/mflux-mcp"
      }
    }
  }
}
```

#### Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

**Using uvx (zero-clone):**

```json
{
  "mcpServers": {
    "mflux": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/alampros/mflux-mcp.git", "mflux-mcp"]
    }
  }
}
```

**Using a local clone:**

```json
{
  "mcpServers": {
    "mflux-mcp": {
      "command": "uv",
      "args": ["run", "server.py"],
      "cwd": "/path/to/mflux-mcp"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Submit a text-to-image generation job to the async queue |
| `edit_image` | Submit an image editing job to the async queue |
| `upscale_image` | Submit an image upscaling job to the async queue |
| `list_jobs` | List jobs in the queue with optional status filter and limit |
| `get_job` | Get full details of one or more jobs by job_id |
| `cancel_job` | Cancel a queued or running job |
| `list_models` | List all available model families and variants with download status |
| `get_image_metadata` | Read EXIF/XMP metadata embedded by mflux in generated images |
| `clear_cache` | Clear all cached models and reclaim memory |
| `get_system_status` | Query RAM, MLX Metal memory, GPU info, queue snapshot, and cached models |

### `generate_image`

Submit a text-to-image generation job to the async queue. Returns a job descriptor immediately — the job runs in the background. Use `get_job()` to poll for completion.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `prompt` | `str` | — | Yes | Text description of the image to generate |
| `output_path` | `str` | — | Yes | File path to write the output image to. Parent directories are created automatically. |
| `model` | `str` | `"flux2-klein-4b"` | | Model name (see [Supported Models](#supported-models)) |
| `width` | `int` | `1024` | | Image width in pixels |
| `height` | `int` | `1024` | | Image height in pixels |
| `steps` | `int` | `4` | | Number of inference steps |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility (auto-generated if omitted) |
| `quantize` | `int \| None` | `8` | | Quantization bit-width (4, 8, or None for full precision) |
| `lora_style` | `str \| None` | `None` | | Optional LoRA style to apply |
| `backend` | `str` | `"thread"` | | Execution backend — `"thread"` or `"subprocess"` |
| `timeout` | `float` | `300.0` | | Per-job timeout in seconds |

### `edit_image`

Submit an image editing job to the async queue. Returns a job descriptor immediately — the job runs in the background. Use `get_job()` to poll for completion.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `image_paths` | `list[str]` | — | Yes | Input image file paths |
| `prompt` | `str` | — | Yes | Text description of the desired edit |
| `output_path` | `str` | — | Yes | File path to write the output image to |
| `model` | `str` | `"flux2-klein-edit"` | | Edit model name |
| `steps` | `int` | `4` | | Number of inference steps |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility |
| `quantize` | `int \| None` | `8` | | Quantization bit-width |
| `lora_style` | `str \| None` | `None` | | Optional LoRA style to apply |
| `backend` | `str` | `"thread"` | | Execution backend — `"thread"` or `"subprocess"` |
| `timeout` | `float` | `300.0` | | Per-job timeout in seconds |

> **Note:** FIBOEdit models use only the first image path. Flux2KleinEdit and QwenImageEdit can accept multiple reference images.

### `upscale_image`

Submit an image upscaling job to the async queue. Returns a job descriptor immediately — the job runs in the background. Use `get_job()` to poll for completion.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `image_path` | `str` | — | Yes | Path to the input image to upscale |
| `output_path` | `str` | — | Yes | File path to write the upscaled image to |
| `model` | `str` | `"seedvr2-3b"` | | Upscale model name (e.g. `"seedvr2-3b"`, `"seedvr2-7b"`) |
| `resolution` | `int` | `2160` | | Target shortest side in pixels (e.g. 2160 for ~4K) |
| `softness` | `float` | `0.5` | | Pre-downsampling factor (0.0–1.0). Lower = sharper |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility |
| `quantize` | `int \| None` | `8` | | Quantization bit-width (4, 8, or None for full precision) |
| `backend` | `str` | `"thread"` | | Execution backend — `"thread"` or `"subprocess"` |
| `timeout` | `float` | `300.0` | | Per-job timeout in seconds |

### `list_jobs`

List jobs in the queue with optional status filter and limit.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `status` | `str \| None` | `None` | | Filter by job status (`"queued"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`). None returns all statuses. |
| `limit` | `int` | `50` | | Maximum number of jobs to return |

Returns a list of job descriptor dicts, most-recently created first.

### `get_job`

Get full details of one or more jobs, including progress and timing.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `job_id` | `str \| list[str]` | — | Yes | A single UUID string, or a list of UUIDs to fetch multiple jobs at once |

When `job_id` is a string, returns the job descriptor dict or `None` if not found. When `job_id` is a list, returns a list of job descriptor dicts (or `None` per missing job) in the same order as the input IDs.

### `cancel_job`

Cancel a queued or running job.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `job_id` | `str` | — | Yes | The UUID job identifier |

Returns a dict with `job_id` and `cancelled` (True if action was taken).

### `list_models`

Returns a list of all registered model variants with their family, capability, LoRA support, quantization options, and whether the model weights are already downloaded locally.

No parameters required.

### `get_image_metadata`

Reads EXIF and XMP metadata embedded by mflux in generated images, including prompt, model, seed, steps, guidance, dimensions, timing, and LoRA configuration.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `image_path` | `str` | — | Yes | Path to the image file to inspect |

### `clear_cache`

Clear all cached models from the in-process LRU model cache, reclaiming unified memory on Apple Silicon. Use this when switching tasks or when memory pressure is high.

No parameters required. Returns a dict with `status`, `models_cleared`, and `message`.

### `get_system_status`

Query system RAM, MLX Metal memory, GPU info, queue snapshot, and cached models. All fields are best-effort — gracefully returns null for anything unavailable.

No parameters required.

## Architecture

### Queue-based async architecture

All inference work (generation and editing) is submitted to a persistent SQLite work queue and executed asynchronously. This avoids MCP client timeouts for long-running operations that can take 30–120+ seconds.

:::mermaid
graph TB
    Agent["LLM Agent"] -->|MCP tools| Server["server.py (FastMCP)"]
    Server -->|"submit job"| Queue["SQLite Queue (jobs.db)"]
    Server -->|"poll / cancel"| Queue
    Queue -->|"backend=thread"| ThreadWorker["In-Process Worker<br/>(MLX thread pool)"]
    Queue -->|"backend=subprocess"| SubprocessWorker["Subprocess Worker<br/>(isolated process)"]
    ThreadWorker -->|"write status"| Queue
    SubprocessWorker -->|"write status"| Queue
    ThreadWorker --> GPU["Apple Silicon GPU<br/>(MLX)"]
    SubprocessWorker --> GPU
    ThreadWorker -->|"write image"| Disk["Output File<br/>(caller-provided path)"]
    SubprocessWorker -->|"write image"| Disk
:::

### Thread vs subprocess backends

| Backend | Speed | Isolation | Use case |
|---------|-------|-----------|----------|
| `thread` (default) | Fast — uses cached models | In-process — shares memory | Repeated generations with the same model |
| `subprocess` | Slower — reloads model per job | Full process isolation | Stability, memory cleanup, crash containment |

### Job lifecycle

:::mermaid
stateDiagram-v2
    [*] --> queued : submit
    queued --> running : worker picks up
    queued --> cancelled : cancel_job
    running --> completed : success
    running --> failed : error
    running --> timed_out : timeout exceeded
    running --> cancelled : cancel_job
    completed --> [*] : purged after TTL
    failed --> [*] : purged after TTL
    timed_out --> [*] : purged after TTL
    cancelled --> [*] : purged after TTL
:::

### GPU exclusivity

Only one job runs at a time across both backends due to an `asyncio.Lock`. The thread backend is fast but blocks; the subprocess backend is isolated but slower. This ensures the Apple Silicon GPU is not contended.

### Usage example

1. **Submit:** `generate_image(prompt="a cat astronaut floating in space, digital art", output_path="/tmp/out.png")`
   → Returns: `{"job_id": "abc123", "status": "queued", ...}`
2. **Poll:** `get_job(job_id="abc123")`
   → Returns: `{"status": "running", "progress": {"phase": "generating"}, ...}`
3. **Complete:** `get_job(job_id="abc123")`
   → Returns: `{"status": "completed", "output_path": "/tmp/out.png", ...}`

## Supported Models

17 model variants across 6 families:

| Name | Family | Capability | LoRA |
|------|--------|------------|------|
| `schnell` | FLUX.1 | txt2img | Yes |
| `dev` | FLUX.1 | txt2img | Yes |
| `flux2-klein-4b` | FLUX.2 | txt2img | Yes |
| `flux2-klein-9b` | FLUX.2 | txt2img | Yes |
| `flux2-klein-base-4b` | FLUX.2 | txt2img | Yes |
| `flux2-klein-base-9b` | FLUX.2 | txt2img | Yes |
| `flux2-klein-edit` | FLUX.2 | edit | Yes |
| `z-image` | Z-Image | txt2img | Yes |
| `z-image-turbo` | Z-Image | txt2img | Yes |
| `fibo` | FIBO | txt2img | Yes |
| `fibo-lite` | FIBO | txt2img | Yes |
| `fibo-edit` | FIBO | edit | Yes |
| `fibo-edit-rmbg` | FIBO | edit | Yes |
| `qwen-image` | Qwen | txt2img | Yes |
| `qwen-image-edit` | Qwen | edit | Yes |
| `seedvr2-3b` | SeedVR2 | upscale | No |
| `seedvr2-7b` | SeedVR2 | upscale | No |

Use `list_models` to check which models are downloaded and ready to use.

## Configuration

### Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_HUB_CACHE` | Override HuggingFace Hub cache directory for model weights | `~/.cache/huggingface/hub` |
| `HF_HOME` | Override HuggingFace home directory (cache is `$HF_HOME/hub`) | `~/.cache/huggingface` |

### CLI flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--transport` | `stdio`, `http` | `stdio` | MCP transport mode |
| `--port` | integer | `8000` | HTTP port (only used with `--transport http`) |

## Development

### Running tests

```bash
uv run pytest tests/
```

All tests use mocks — no GPU or downloaded models required.

### Project structure

```
mflux-mcp/
├── server.py              # MCP server entry point (FastMCP, 10 tools)
├── job_queue.py            # SQLite job queue (WAL mode)
├── worker.py               # WorkerManager (thread + subprocess backends)
├── subprocess_runner.py    # Standalone subprocess runner
├── mflux_cache.py          # Thread-safe lazy model cache & registry
├── pyproject.toml          # Python project config (uv)
├── AGENTS.md               # Agent navigation guide
├── README.md               # This file
├── LICENSE                 # MIT License
└── tests/
    ├── conftest.py        # Test config & shared fixtures
    ├── test_server.py     # MCP tool tests
    ├── test_model_cache.py # Model cache tests
    ├── test_job_queue.py  # Job queue tests
    ├── test_worker.py     # Worker tests
    └── test_subprocess_runner.py # Subprocess runner tests
```

## License

MIT — see [LICENSE](LICENSE) for details.
