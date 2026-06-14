"""mflux-mcp — MCP server exposing mflux image generation to LLM agents."""

import asyncio
import concurrent.futures
import functools
import io
import os
import random
import time

from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
from huggingface_hub.errors import GatedRepoError
from mflux.utils.metadata_reader import MetadataReader

from mflux_cache import ModelCache, _REPO_MAP, is_model_cached

INFERENCE_TIMEOUT = 300.0  # seconds — timeout for model inference tools
HEARTBEAT_INTERVAL = 10.0  # seconds — interval for progress heartbeats during inference

mcp = FastMCP("mflux-mcp")
cache = ModelCache()
_inference_lock = asyncio.Semaphore(1)

# ---------------------------------------------------------------------------
# Dedicated single-thread executor for all MLX GPU work.
#
# MLX binds ``Stream(gpu, 0)`` to the thread that first creates it.  If model
# loading happens on thread A and inference on thread B, thread B has no GPU
# stream and ``mx.eval`` raises:
#     RuntimeError: There is no Stream(gpu, 0) in current thread.
#
# Pinning *all* MLX work (model loading + inference) to a single persistent
# thread guarantees the stream is always available.
# ---------------------------------------------------------------------------
_mlx_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-gpu"
)


async def _run_on_mlx_thread(fn, *args, **kwargs):
    """Run *fn* on the dedicated MLX GPU thread and return the result.

    All MLX operations (model loading, inference, evaluation) must happen
    on the same thread so they share a single ``Stream(gpu, 0)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _mlx_executor, functools.partial(fn, *args, **kwargs)
    )


async def _run_with_heartbeat(
    blocking_fn,
    kwargs: dict,
    ctx,
    stage_message: str,
    progress_current: int = 2,
    progress_total: int = 4,
):
    """Run a blocking callable on the MLX thread while sending periodic progress heartbeats.

    Dispatches the blocking function to ``_mlx_executor`` and, if a FastMCP
    context is available, sends progress notifications every HEARTBEAT_INTERVAL
    seconds to keep the MCP client's timeout clock from expiring.
    """
    start = time.monotonic()
    task = asyncio.ensure_future(_run_on_mlx_thread(blocking_fn, **kwargs))

    if ctx is None:
        return await task

    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            await ctx.report_progress(
                progress_current,
                progress_total,
                f"{stage_message} (elapsed: {int(elapsed)}s)",
            )
    return task.result()


@mcp.tool(timeout=INFERENCE_TIMEOUT)
async def generate_image(
    prompt: str,
    model: str = "flux2-klein-4b",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int | None = None,
    quantize: int = 8,
    output_path: str | None = None,
    lora_style: str | None = None,
    ctx: Context | None = None,
) -> Image | str:
    """Generate an image from a text prompt using mflux.

    Args:
        prompt: Text description of the image to generate.
        model: Model name (e.g. "flux2-klein-4b", "schnell", "z-image").
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of inference steps.
        seed: Random seed for reproducibility. Auto-generated if not provided.
        quantize: Quantization bit-width (4, 8, or None for full precision).
        output_path: Optional file path to save the image to. When provided,
            parent directories are created automatically and the absolute path
            is returned as a string. When omitted, raw image bytes are returned
            via the FastMCP Image type.
        lora_style: Optional LoRA style to apply. Valid values: couple, font,
            home, identity, illustration, portrait, ppt, sandstorm, sparklers,
            storyboard.

    Returns:
        FastMCP Image when output_path is None, otherwise the absolute file
        path as a string.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if ctx is not None:
        await ctx.report_progress(0, 4, "queued")

    async with _inference_lock:
        if ctx is not None:
            await ctx.report_progress(1, 4, "loading model")
        try:
            loaded_model = await _run_on_mlx_thread(
                cache.get_model, model, quantize=quantize, lora_style=lora_style
            )
        except GatedRepoError:
            _, config_factory_name, _ = ModelCache._REGISTRY[model]
            repo_id = _REPO_MAP.get(config_factory_name, model)
            raise RuntimeError(
                f"Model '{model}' requires access to a gated HuggingFace repository.\n\n"
                f"To resolve this:\n"
                f"1. Visit https://huggingface.co/{repo_id} and request access\n"
                f"2. Authenticate locally by running: huggingface-cli login\n"
                f"   Or set the HF_TOKEN environment variable with your access token\n"
                f"3. Retry the operation"
            )

        if ctx is not None:
            await ctx.report_progress(2, 4, "generating")
        result = await _run_with_heartbeat(
            loaded_model.generate_image,
            kwargs=dict(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
            ),
            ctx=ctx,
            stage_message="generating",
        )

        if ctx is not None:
            await ctx.report_progress(3, 4, "saving")
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        if output_path is not None:
            output_path = os.path.abspath(output_path)
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path

        return Image(data=image_bytes, format="png")


@mcp.tool(timeout=INFERENCE_TIMEOUT)
async def edit_image(
    image_paths: list[str],
    prompt: str,
    model: str = "flux2-klein-edit",
    steps: int = 4,
    seed: int | None = None,
    quantize: int = 8,
    output_path: str | None = None,
    lora_style: str | None = None,
    ctx: Context | None = None,
) -> Image | str:
    """Edit an image using an mflux edit model.

    Accepts one or more input images and a text prompt describing the edit.

    Args:
        image_paths: List of input image file paths. FIBOEdit models use
            only the first path; Flux2KleinEdit and QwenImageEdit can use
            multiple reference images.
        prompt: Text description of the desired edit.
        model: Edit model name (e.g. "flux2-klein-edit", "fibo-edit",
            "qwen-image-edit").
        steps: Number of inference steps.
        seed: Random seed for reproducibility. Auto-generated if not provided.
        quantize: Quantization bit-width (4, 8, or None for full precision).
        output_path: Optional file path to save the image to. When provided,
            parent directories are created automatically and the absolute path
            is returned as a string. When omitted, raw image bytes are returned
            via the FastMCP Image type.
        lora_style: Optional LoRA style to apply. Valid values: couple, font,
            home, identity, illustration, portrait, ppt, sandstorm, sparklers,
            storyboard.

    Returns:
        FastMCP Image when output_path is None, otherwise the absolute file
        path as a string.

    Raises:
        ValueError: If image_paths is empty.
    """
    if not image_paths:
        raise ValueError("image_paths must contain at least one image path.")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if ctx is not None:
        await ctx.report_progress(0, 4, "queued")

    async with _inference_lock:
        if ctx is not None:
            await ctx.report_progress(1, 4, "loading model")
        try:
            loaded_model = await _run_on_mlx_thread(
                cache.get_model, model, quantize=quantize, lora_style=lora_style
            )
        except GatedRepoError:
            _, config_factory_name, _ = ModelCache._REGISTRY[model]
            repo_id = _REPO_MAP.get(config_factory_name, model)
            raise RuntimeError(
                f"Model '{model}' requires access to a gated HuggingFace repository.\n\n"
                f"To resolve this:\n"
                f"1. Visit https://huggingface.co/{repo_id} and request access\n"
                f"2. Authenticate locally by running: huggingface-cli login\n"
                f"   Or set the HF_TOKEN environment variable with your access token\n"
                f"3. Retry the operation"
            )

        if ctx is not None:
            await ctx.report_progress(2, 4, "generating")
        class_key = ModelCache._REGISTRY[model][0]
        if class_key == "FIBOEdit":
            inference_kwargs = dict(
                seed=seed,
                prompt=prompt,
                image_path=image_paths[0],
                num_inference_steps=steps,
            )
        else:
            inference_kwargs = dict(
                seed=seed,
                prompt=prompt,
                image_paths=image_paths,
                num_inference_steps=steps,
            )
        result = await _run_with_heartbeat(
            loaded_model.generate_image,
            kwargs=inference_kwargs,
            ctx=ctx,
            stage_message="generating",
        )

        if ctx is not None:
            await ctx.report_progress(3, 4, "saving")
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        if output_path is not None:
            output_path = os.path.abspath(output_path)
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path

        return Image(data=image_bytes, format="png")


# ---------------------------------------------------------------------------
# Metadata mappings for list_models (static — no mflux imports triggered)
# ---------------------------------------------------------------------------

_CAPABILITY_MAP: dict[str, str] = {
    "Flux1": "txt2img",
    "Flux2Klein": "txt2img",
    "Flux2KleinEdit": "edit",
    "ZImage": "txt2img",
    "FIBO": "txt2img",
    "FIBOEdit": "edit",
    "QwenImage": "txt2img",
    "QwenImageEdit": "edit",
    "SeedVR2": "upscale",
}

_FAMILY_MAP: dict[str, str] = {
    "Flux1": "FLUX.1",
    "Flux2Klein": "FLUX.2",
    "Flux2KleinEdit": "FLUX.2",
    "ZImage": "Z-Image",
    "FIBO": "FIBO",
    "FIBOEdit": "FIBO",
    "QwenImage": "Qwen",
    "QwenImageEdit": "Qwen",
    "SeedVR2": "SeedVR2",
}


@mcp.tool()
def list_models() -> list[dict]:
    """List available mflux model families and variants.

    Returns a structured list of all registered models with their family,
    capability (txt2img, edit, or upscale), LoRA support, and valid
    quantization options.

    Returns:
        A list of dicts, one per model, each containing:
            - name: Model identifier to pass to generate_image / edit_image.
            - family: Model family (e.g. "FLUX.1", "FLUX.2", "FIBO").
            - capability: One of "txt2img", "edit", or "upscale".
            - supports_lora: Whether the model accepts LoRA adapters.
            - quantize_options: Valid quantize values (list of ints or None).
            - is_downloaded: Whether the model weights are locally cached.
    """
    models: list[dict] = []
    for name, (
        class_key,
        config_factory_name,
        supports_lora,
    ) in ModelCache._REGISTRY.items():
        repo_id = _REPO_MAP.get(config_factory_name)
        downloaded = is_model_cached(repo_id) if repo_id else False
        models.append(
            {
                "name": name,
                "family": _FAMILY_MAP[class_key],
                "capability": _CAPABILITY_MAP[class_key],
                "supports_lora": supports_lora,
                "quantize_options": [4, 8, None],
                "is_downloaded": downloaded,
            }
        )
    return models


@mcp.tool()
def get_image_metadata(image_path: str) -> dict:
    """Inspect metadata embedded in an mflux-generated image file.

    Reads EXIF and XMP metadata that mflux embeds during image generation,
    including prompt, model, seed, steps, guidance, dimensions, timing,
    and LoRA configuration.

    Args:
        image_path: Path to the image file to inspect.

    Returns:
        A dict with "exif" and "xmp" keys containing the parsed metadata,
        or a dict with a "message" key if no mflux metadata was found.

    Raises:
        ValueError: If the file does not exist at the given path.
    """
    if not os.path.isfile(image_path):
        raise ValueError(f"File not found: {image_path}")

    metadata = MetadataReader.read_all_metadata(image_path)

    if not metadata.get("exif") and not metadata.get("xmp"):
        return {
            "message": "No mflux metadata found in this image.",
            "image_path": image_path,
        }

    return metadata


def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments for transport configuration.

    Args:
        argv: Argument list to parse (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace with transport and port attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="mflux-mcp server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    return parser.parse_args(argv)


def main():
    """Entry point for the mflux-mcp console script."""
    args = parse_args()

    kwargs: dict = {"transport": args.transport}
    if args.transport == "http":
        kwargs["port"] = args.port
    mcp.run(**kwargs)


if __name__ == "__main__":
    main()
