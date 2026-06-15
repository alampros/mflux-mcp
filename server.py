"""mflux-mcp — MCP server exposing mflux image generation to LLM agents."""

import asyncio
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from mflux.utils.metadata_reader import MetadataReader

from job_queue import JobQueue
from mflux_cache import ModelCache, _REPO_MAP, is_model_cached
from worker import WorkerManager

try:
    import psutil
except ImportError:
    psutil = None

_VALID_LORA_STYLES = {
    "couple",
    "font",
    "home",
    "identity",
    "illustration",
    "portrait",
    "ppt",
    "sandstorm",
    "sparklers",
    "storyboard",
}
_VALID_QUANTIZE = {4, 8, None}

_queue: JobQueue | None = None
_cache: ModelCache | None = None
_worker_manager: WorkerManager | None = None


def _log(msg: str) -> None:
    """Write a log line to stderr (stdout is the MCP protocol channel)."""
    print(f"[mflux-mcp] {msg}", file=sys.stderr, flush=True)


@asynccontextmanager
async def _app_lifespan(server: FastMCP):
    """FastMCP lifespan — start workers on startup, stop on shutdown."""
    global _worker_manager
    if _worker_manager is not None:
        await _worker_manager.start()
    try:
        yield {}
    finally:
        if _worker_manager is not None:
            await _worker_manager.stop()


mcp = FastMCP("mflux-mcp", lifespan=_app_lifespan)


# ---------------------------------------------------------------------------
# Inference tools (queue-based)
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate_image(
    prompt: str,
    output_path: str,
    model: str = "flux2-klein-4b",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int | None = None,
    quantize: int | None = 8,
    lora_style: str | None = None,
    backend: str = "thread",
    timeout: float = 300.0,
) -> dict:
    """Submit a text-to-image generation job to the async queue.

    Returns a job descriptor immediately — the job runs in the background.
    Use get_job() to poll for completion. Image generation typically takes
    30–90 seconds depending on model and parameters, so poll no more
    frequently than every 10 seconds to avoid unnecessary overhead.

    Args:
        prompt: Text description of the image to generate.
        output_path: File path to write the output image to.
        model: Model name (e.g. "flux2-klein-4b", "schnell", "z-image").
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of inference steps.
        seed: Random seed for reproducibility. Auto-generated if not provided.
        quantize: Quantization bit-width (4, 8, or None for full precision).
        lora_style: Optional LoRA style to apply.
        backend: Execution backend — "thread" or "subprocess".
        timeout: Per-job timeout in seconds.

    Returns:
        A job descriptor dict with job_id, status, command, output_path, backend.
    """
    if _queue is None:
        raise RuntimeError("Server not initialized — queue is not available.")

    if model not in ModelCache._REGISTRY:
        available = ", ".join(sorted(ModelCache._REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{model}'. Available models: {available}")

    if quantize not in _VALID_QUANTIZE:
        raise ValueError(f"Invalid quantize={quantize}. Must be one of: 4, 8, or None.")

    if lora_style is not None and lora_style not in _VALID_LORA_STYLES:
        valid = ", ".join(sorted(_VALID_LORA_STYLES))
        raise ValueError(f"Invalid lora_style='{lora_style}'. Valid: {valid}")

    if backend not in ("thread", "subprocess"):
        raise ValueError(
            f"Invalid backend='{backend}'. Must be 'thread' or 'subprocess'."
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    _log(
        f"generate_image [queue-submit] model={model} size={width}x{height} steps={steps} "
        f"seed={seed} quantize={quantize} lora={lora_style} backend={backend} prompt={prompt!r}"
    )

    job = _queue.submit(
        command="generate_image",
        params={
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "quantize": quantize,
            "lora_style": lora_style,
        },
        output_path=output_path,
        backend=backend,
        timeout_s=timeout,
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "command": job["command"],
        "output_path": job["output_path"],
        "backend": job["backend"],
    }


@mcp.tool()
async def edit_image(
    image_paths: list[str],
    prompt: str,
    output_path: str,
    model: str = "flux2-klein-edit",
    steps: int = 4,
    seed: int | None = None,
    quantize: int | None = 8,
    lora_style: str | None = None,
    backend: str = "thread",
    timeout: float = 300.0,
) -> dict:
    """Submit an image editing job to the async queue.

    Returns a job descriptor immediately — the job runs in the background.
    Use get_job() to poll for completion. Image editing typically takes
    30–90 seconds depending on model and parameters, so poll no more
    frequently than every 10 seconds to avoid unnecessary overhead.

    Args:
        image_paths: List of input image file paths.
        prompt: Text description of the desired edit.
        output_path: File path to write the output image to.
        model: Edit model name (e.g. "flux2-klein-edit", "fibo-edit").
        steps: Number of inference steps.
        seed: Random seed for reproducibility. Auto-generated if not provided.
        quantize: Quantization bit-width (4, 8, or None for full precision).
        lora_style: Optional LoRA style to apply.
        backend: Execution backend — "thread" or "subprocess".
        timeout: Per-job timeout in seconds.

    Returns:
        A job descriptor dict with job_id, status, command, output_path, backend.

    Raises:
        ValueError: If image_paths is empty.
    """
    if _queue is None:
        raise RuntimeError("Server not initialized — queue is not available.")

    if not image_paths:
        raise ValueError("image_paths must contain at least one image path.")

    if model not in ModelCache._REGISTRY:
        available = ", ".join(sorted(ModelCache._REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{model}'. Available models: {available}")

    # Verify the model supports editing
    class_key = ModelCache._REGISTRY[model][0]
    if class_key not in ("Flux2KleinEdit", "FIBOEdit", "QwenImageEdit"):
        raise ValueError(f"Model '{model}' does not support image editing.")

    if quantize not in _VALID_QUANTIZE:
        raise ValueError(f"Invalid quantize={quantize}. Must be one of: 4, 8, or None.")

    if lora_style is not None and lora_style not in _VALID_LORA_STYLES:
        valid = ", ".join(sorted(_VALID_LORA_STYLES))
        raise ValueError(f"Invalid lora_style='{lora_style}'. Valid: {valid}")

    if backend not in ("thread", "subprocess"):
        raise ValueError(
            f"Invalid backend='{backend}'. Must be 'thread' or 'subprocess'."
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    _log(
        f"edit_image [queue-submit] model={model} steps={steps} seed={seed} "
        f"quantize={quantize} lora={lora_style} backend={backend} images={image_paths} prompt={prompt!r}"
    )

    job = _queue.submit(
        command="edit_image",
        params={
            "image_paths": image_paths,
            "prompt": prompt,
            "model": model,
            "steps": steps,
            "seed": seed,
            "quantize": quantize,
            "lora_style": lora_style,
        },
        output_path=output_path,
        backend=backend,
        timeout_s=timeout,
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "command": job["command"],
        "output_path": job["output_path"],
        "backend": job["backend"],
    }


# ---------------------------------------------------------------------------
# Queue management tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_jobs(status: str | None = None, limit: int = 50) -> list[dict]:
    """List jobs in the queue with optional status filter.

    Args:
        status: Filter by job status (e.g. "queued", "running", "completed").
            None returns jobs of all statuses.
        limit: Maximum number of jobs to return.

    Returns:
        A list of job descriptor dicts, most-recently created first.
    """
    if _queue is None:
        raise RuntimeError("Server not initialized — queue is not available.")
    return _queue.list_jobs(status=status, limit=limit)


@mcp.tool()
def get_job(job_id: str) -> dict | None:
    """Get full details of a single job.

    Use this to check on a previously submitted generate_image or edit_image
    job. Jobs typically take 30–90 seconds to complete. Do not poll more
    frequently than every 10 seconds.

    Args:
        job_id: The UUID job identifier.

    Returns:
        The job descriptor dict, or None if not found.
    """
    if _queue is None:
        raise RuntimeError("Server not initialized — queue is not available.")
    return _queue.get_job(job_id)


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running job.

    Args:
        job_id: The UUID job identifier.

    Returns:
        A dict with job_id and cancelled (True if action was taken).
    """
    if _worker_manager is None:
        raise RuntimeError("Server not initialized — worker manager is not available.")
    result = await _worker_manager.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": result}


# ---------------------------------------------------------------------------
# Utility tools (unchanged from v1)
# ---------------------------------------------------------------------------

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


@mcp.tool()
def clear_cache() -> dict:
    """Clear all cached models and reclaim memory.

    Removes all loaded model instances from the in-memory cache, freeing
    unified memory on Apple Silicon. Use this when switching tasks or when
    memory pressure is high.

    Returns:
        A dict containing:
            - status: "ok" on success.
            - models_cleared: Number of models that were cached before clearing.
            - message: Human-readable summary.
    """
    if _cache is None:
        raise RuntimeError("Server not initialized — cache is not available.")
    count = _cache.size
    _cache.clear()
    return {
        "status": "ok",
        "models_cleared": count,
        "message": f"Cleared {count} cached model(s).",
    }


# ---------------------------------------------------------------------------
# System status tool
# ---------------------------------------------------------------------------


@mcp.tool()
def get_system_status() -> dict:
    """Query system RAM, MLX Metal memory, GPU info, queue snapshot, and cached models.

    All fields are best-effort — gracefully returns null for anything unavailable.
    """
    status: dict[str, Any] = {}

    # RAM
    try:
        if psutil is not None:
            mem = psutil.virtual_memory()
            status["ram"] = {
                "total_gb": round(mem.total / (1024**3), 1),
                "available_gb": round(mem.available / (1024**3), 1),
                "percent_used": mem.percent,
            }
        else:
            status["ram"] = None
    except Exception:
        status["ram"] = None

    # Metal (MLX GPU memory)
    try:
        import mlx.core as mx

        # Prefer non-deprecated APIs; fall back to deprecated mx.metal.* for compatibility
        _get_active = getattr(mx, "get_active_memory", None) or getattr(
            mx.metal, "get_active_memory", None
        )
        _get_peak = getattr(mx, "get_peak_memory", None) or getattr(
            mx.metal, "get_peak_memory", None
        )
        _get_cache = getattr(mx, "get_cache_memory", None) or getattr(
            mx.metal, "get_cache_memory", None
        )

        status["metal"] = {
            "active_mb": round(_get_active() / (1024**2)) if _get_active else None,
            "peak_mb": round(_get_peak() / (1024**2)) if _get_peak else None,
            "cache_mb": round(_get_cache() / (1024**2)) if _get_cache else None,
        }
    except Exception:
        status["metal"] = None

    # Chip
    try:
        import mlx.core as mx

        _device_info = getattr(mx, "device_info", None) or getattr(
            mx.metal, "device_info", None
        )
        info = _device_info() if _device_info else {}
        status["chip"] = {
            "name": info.get("device_name", None),
            "gpu_cores": info.get("gpu_core_count", None),
            "recommended_max_gb": (
                round(
                    info.get(
                        "recommended_max_working_set_size",
                        info.get("max_recommended_working_set_size", 0),
                    )
                    / (1024**3),
                    1,
                )
                or None
            ),
        }
    except Exception:
        status["chip"] = None

    # Queue counts
    try:
        if _queue is not None:
            all_jobs = _queue.list_jobs()
            queued = sum(1 for j in all_jobs if j["status"] == "queued")
            running = sum(1 for j in all_jobs if j["status"] == "running")
            status["queue"] = {"queued": queued, "running": running}
        else:
            status["queue"] = None
    except Exception:
        status["queue"] = None

    # Cached models
    try:
        if _cache is not None:
            cached_models = []
            for key in _cache._cache.keys():
                name, quantize, _lora = key
                q_label = f"q{quantize}" if quantize else "fp"
                cached_models.append(f"{name} ({q_label})")
            status["cached_models"] = cached_models
        else:
            status["cached_models"] = []
    except Exception:
        status["cached_models"] = []

    return status


# ---------------------------------------------------------------------------
# CLI and entry point
# ---------------------------------------------------------------------------


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
    global _queue, _cache, _worker_manager

    args = parse_args()

    # Initialize components
    db_path = Path(__file__).parent / "jobs.db"
    _cache = ModelCache()
    _queue = JobQueue(db_path)
    _worker_manager = WorkerManager(_queue, _cache)

    kwargs: dict = {"transport": args.transport}
    if args.transport == "http":
        kwargs["port"] = args.port
    mcp.run(**kwargs)


if __name__ == "__main__":
    main()
