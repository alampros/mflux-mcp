# mflux-mcp

MCP server exposing the [mflux](https://github.com/filipstrand/mflux) image generation tool to LLM agents via the Model Context Protocol.

## Language Choice

Python. Rationale:
- mflux has a clean Python API (`Flux2Klein`, `ZImageTurbo`, etc.) -- far better than shelling out from Node.js.
- Existing MCP server pattern in `sage-mcp` using `fastmcp` to follow.
- Holding the model in memory after first load enables much faster subsequent generations.
- mflux depends on `mlx` (macOS/Apple Silicon only), so this server only runs on Mac anyway.

## Project Structure

```
mflux-mcp/
  pyproject.toml          # uv project w/ deps + [tool.pytest] config
  server.py               # MCP server entry point
  tests/
    conftest.py           # Shared fixtures (mock model, mock image, etc.)
    test_tools.py         # Test each MCP tool's logic
    test_server.py        # Test server registration, tool listing, MCP protocol
```

## Dependencies

| Type | Packages |
|------|----------|
| Runtime | `mflux`, `fastmcp`, `mcp` |
| Dev | `pytest`, `pytest-asyncio` |

## MCP Tools

| Tool | Description | Key Params |
|------|-------------|------------|
| `generate_image` | Text-to-image generation | prompt, model, width, height, steps, seed, quantize |
| `edit_image` | Image-conditioned editing (FLUX.2) | image_paths, prompt, model, steps, seed |
| `list_models` | List available mflux model families and variants | -- |
| `get_image_metadata` | Inspect metadata from an mflux-generated image file | image_path |

All image-producing tools return images via the `fastmcp` Image type (base64 data URI).

## Design Decisions

- **Python API, not subprocess** -- call mflux classes directly for generation.
- **Lazy model loading** -- models are large; load on first use and cache in memory.
- **Default model**: FLUX.2 klein-4b with q8 quantization.
- **Transport**: stdio by default (for local agent use), `--http` flag for HTTP transport.

## Architecture

:::mermaid
graph LR
    Agent["LLM Agent"] -->|MCP protocol| Server["server.py<br/>(FastMCP)"]
    Server -->|Python API| MFlux["mflux library"]
    MFlux -->|MLX| GPU["Apple Silicon GPU"]
    MFlux -->|weights| HF["HuggingFace Cache /<br/>Local Models"]
:::

### Model Cache

The server holds a dict of loaded models keyed by `(model_name, quantize)`. On first call to `generate_image` or `edit_image`, the requested model is loaded and cached. Subsequent calls with the same config reuse the cached model, avoiding the multi-second load time.

```python
_model_cache: dict[tuple[str, int | None], Model] = {}
```

## Tests

Using pytest with mocks (no GPU required to run tests).

| Test | Coverage |
|------|----------|
| Tool registration | All 4 tools registered with correct names and schemas |
| `generate_image` params | Valid/invalid dimensions, steps, quantize values, missing prompt |
| `generate_image` happy path | Mock model returns mock PIL Image, verify base64 image returned |
| `edit_image` validation | Missing image paths, invalid file paths |
| `list_models` | Returns expected model list structure |
| `get_image_metadata` | Valid image path returns metadata, missing file returns error |
| Error handling | Model load failures, OOM simulation, graceful error messages |

## Supported Models

From mflux v0.17.5 (installed via `uv tool`):

| Model Family | Variants | Strengths |
|-------------|----------|-----------|
| FLUX.2 | klein-4b, klein-9b, klein-base-4b, klein-base-9b | Fast, small, good quality, edit support |
| Z-Image | z-image, z-image-turbo | Fast, small, very good realism |
| FIBO | fibo, fibo-edit | JSON-based prompts, edit support |
| FLUX.1 | schnell, dev, kontext | Legacy, ControlNet, Kontext editing |
| Qwen Image | qwen | Large model, strong prompt understanding |
| SeedVR2 | seedvr2 | Best upscaling |
| Depth Pro | depth-pro | Fast depth estimation |

## CLI Commands (mflux reference)

Key commands available via `uv tool`:

```
mflux-generate-flux2          # FLUX.2 text-to-image
mflux-generate-flux2-edit     # FLUX.2 image editing
mflux-generate-z-image        # Z-Image text-to-image
mflux-generate-z-image-turbo  # Z-Image Turbo text-to-image
mflux-generate-fibo           # FIBO text-to-image
mflux-generate-fibo-edit      # FIBO image editing
mflux-generate-qwen           # Qwen text-to-image
mflux-generate-qwen-edit      # Qwen image editing
mflux-upscale-seedvr2         # SeedVR2 upscaling
mflux-info                    # Inspect image metadata
mflux-save                    # Save quantized model
mflux-train                   # LoRA training
```

## Python API Reference (FLUX.2 examples)

### Text-to-image

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein

model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A puffin standing on a cliff",
    num_inference_steps=4,
    width=1024,
    height=560,
)
image.save("puffin.png")
```

### Image editing

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit

model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_9b())
image = model.generate_image(
    seed=42,
    prompt="Make the woman wear the eyeglasses",
    image_paths=["person.jpg", "glasses.jpg"],
    num_inference_steps=4,
)
image.save("edited.png")
```

### Metadata inspection

```python
from mflux.utils.metadata_reader import MetadataReader

metadata = MetadataReader.read_all_metadata("./image.png")
```
