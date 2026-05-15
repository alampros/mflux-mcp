"""mflux-mcp — MCP server exposing mflux image generation to LLM agents."""

import io
import os
import random

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from mflux.utils.metadata_reader import MetadataReader

from mflux_cache import ModelCache, _REPO_MAP, is_model_cached

mcp = FastMCP("mflux-mcp")
cache = ModelCache()


@mcp.tool()
def generate_image(
    prompt: str,
    model: str = "flux2-klein-4b",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int | None = None,
    quantize: int = 8,
    output_path: str | None = None,
    lora_style: str | None = None,
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

    loaded_model = cache.get_model(model, quantize=quantize, lora_style=lora_style)

    result = loaded_model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )

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


@mcp.tool()
def edit_image(
    image_paths: list[str],
    prompt: str,
    model: str = "flux2-klein-edit",
    steps: int = 4,
    seed: int | None = None,
    quantize: int = 8,
    output_path: str | None = None,
    lora_style: str | None = None,
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

    loaded_model = cache.get_model(model, quantize=quantize, lora_style=lora_style)

    # FIBOEdit takes a singular image_path argument, while Flux2KleinEdit
    # and QwenImageEdit accept a plural image_paths list.
    class_key = ModelCache._REGISTRY[model][0]
    if class_key == "FIBOEdit":
        result = loaded_model.generate_image(
            seed=seed,
            prompt=prompt,
            image_path=image_paths[0],
            num_inference_steps=steps,
        )
    else:
        result = loaded_model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=image_paths,
            num_inference_steps=steps,
        )

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
    for name, (class_key, config_factory_name, supports_lora) in ModelCache._REGISTRY.items():
        repo_id = _REPO_MAP.get(config_factory_name)
        downloaded = is_model_cached(repo_id) if repo_id else False
        models.append({
            "name": name,
            "family": _FAMILY_MAP[class_key],
            "capability": _CAPABILITY_MAP[class_key],
            "supports_lora": supports_lora,
            "quantize_options": [4, 8, None],
            "is_downloaded": downloaded,
        })
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
        return {"message": "No mflux metadata found in this image.", "image_path": image_path}

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
