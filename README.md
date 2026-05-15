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

```bash
# Clone the repo
git clone git@github.com:alampros/mflux-mcp.git
cd mflux-mcp

# Install Python dependencies
uv sync

# Install mflux as a uv tool (provides CLI commands used by health.sh)
uv tool install mflux
```

> **Note:** mflux is both a Python dependency (imported by the server) and a standalone CLI tool. `uv sync` handles the library dependency; `uv tool install mflux` provides CLI commands like `mflux-generate-flux2`.

## Usage

### Running the server

The server supports two transport modes:

```bash
# Default: stdio transport (for MCP client integration)
uv run server.py

# HTTP transport on a custom port
uv run server.py --transport http --port 8080
```

### Claude Desktop integration

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

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

### Example tool calls

Generate an image:
```json
{
  "tool": "generate_image",
  "arguments": {
    "prompt": "a cat astronaut floating in space, digital art",
    "model": "schnell",
    "width": 512,
    "height": 512,
    "steps": 4
  }
}
```

Edit an existing image:
```json
{
  "tool": "edit_image",
  "arguments": {
    "image_paths": ["/path/to/input.png"],
    "prompt": "add a rainbow in the background",
    "model": "flux2-klein-edit"
  }
}
```

Save to a specific path (returns the absolute file path instead of image bytes):
```json
{
  "tool": "generate_image",
  "arguments": {
    "prompt": "mountain landscape at sunset",
    "output_path": "output/landscape.png"
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate an image from a text prompt |
| `edit_image` | Edit an image using a text prompt and one or more reference images |
| `list_models` | List all available model families and variants with download status |
| `get_image_metadata` | Read EXIF/XMP metadata embedded by mflux in generated images |

### `generate_image`

Generate an image from a text prompt using mflux.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `prompt` | `str` | — | Yes | Text description of the image to generate |
| `model` | `str` | `"flux2-klein-4b"` | | Model name (see [Supported Models](#supported-models)) |
| `width` | `int` | `1024` | | Image width in pixels |
| `height` | `int` | `1024` | | Image height in pixels |
| `steps` | `int` | `4` | | Number of inference steps |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility (auto-generated if omitted) |
| `quantize` | `int` | `8` | | Quantization bit-width (4, 8, or None for full precision) |
| `output_path` | `str \| None` | `None` | | File path to save the image. Returns raw image bytes when omitted, absolute path string when provided. Parent directories are created automatically. |

### `edit_image`

Edit an image using a text prompt and one or more input images.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `image_paths` | `list[str]` | — | Yes | Input image file paths |
| `prompt` | `str` | — | Yes | Text description of the desired edit |
| `model` | `str` | `"flux2-klein-edit"` | | Edit model name |
| `steps` | `int` | `4` | | Number of inference steps |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility |
| `quantize` | `int` | `8` | | Quantization bit-width |
| `output_path` | `str \| None` | `None` | | File path to save the image |

> **Note:** FIBOEdit models use only the first image path. Flux2KleinEdit and QwenImageEdit can accept multiple reference images.

### `list_models`

Returns a list of all registered model variants with their family, capability, LoRA support, quantization options, and whether the model weights are already downloaded locally.

No parameters required.

### `get_image_metadata`

Reads EXIF and XMP metadata embedded by mflux in generated images, including prompt, model, seed, steps, guidance, dimensions, timing, and LoRA configuration.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `image_path` | `str` | — | Yes | Path to the image file to inspect |

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

### Health check

```bash
bash health.sh
```

Checks toolchain (uv, python3), mflux CLI, project files, test suite, docs, harness data, and Apple Silicon hardware. Must exit 0 before closing any task.

### Project structure

```
mflux-mcp/
├── server.py              # MCP server entry point (FastMCP, 4 tools)
├── mflux_cache.py         # Thread-safe lazy model cache & registry
├── pyproject.toml         # Python project config (uv)
├── health.sh              # Project health check script
├── PLAN.md                # Architecture & design document
├── AGENTS.md              # Agent navigation guide
├── README.md              # This file
├── LICENSE                # MIT License
├── tests/
│   ├── conftest.py        # Test config (sys.path setup)
│   ├── test_server.py     # MCP tool tests (15 test classes)
│   └── test_model_cache.py # Model cache tests (4 test classes)
└── output/                # Generated images directory (gitignored)
```

## License

MIT — see [LICENSE](LICENSE) for details.
