"""Lazy model cache for mflux models.

Provides a thread-safe cache that lazily loads mflux model instances on first
use, keyed by (model_name, quantize). Models are heavy MLX nn.Module objects
that take several seconds to load — caching avoids reloading on every
generation call.
"""

import os
import threading
from pathlib import Path
from typing import Any, Callable


def _lazy_imports() -> dict[str, Any]:
    """Import mflux model classes lazily to avoid heavy import at module load time."""
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.models.fibo.variants.txt2img.fibo import FIBO
    from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    from mflux.models.common.config.model_config import ModelConfig

    return {
        "Flux1": Flux1,
        "Flux2Klein": Flux2Klein,
        "Flux2KleinEdit": Flux2KleinEdit,
        "ZImage": ZImage,
        "FIBO": FIBO,
        "FIBOEdit": FIBOEdit,
        "QwenImage": QwenImage,
        "QwenImageEdit": QwenImageEdit,
        "SeedVR2": SeedVR2,
        "ModelConfig": ModelConfig,
    }


# Sentinel for models that do NOT accept lora_paths / lora_scales
_NO_LORA = object()


def _build_registry() -> dict[str, tuple[str, Callable, bool]]:
    """Build the model registry mapping model name -> (class_key, config_factory, supports_lora).

    Returns a dict keyed by model name string. Each value is a tuple of:
      - class_key: key into the _lazy_imports() dict
      - config_factory_name: name of the ModelConfig static factory method
      - supports_lora: whether the model class accepts lora_paths/lora_scales
    """
    return {
        # FLUX.1
        "schnell": ("Flux1", "schnell", True),
        "dev": ("Flux1", "dev", True),
        # FLUX.2 Klein
        "flux2-klein-4b": ("Flux2Klein", "flux2_klein_4b", True),
        "flux2-klein-9b": ("Flux2Klein", "flux2_klein_9b", True),
        "flux2-klein-base-4b": ("Flux2Klein", "flux2_klein_base_4b", True),
        "flux2-klein-base-9b": ("Flux2Klein", "flux2_klein_base_9b", True),
        # FLUX.2 Klein Edit
        "flux2-klein-edit": ("Flux2KleinEdit", "flux2_klein_4b", True),
        # Z-Image
        "z-image": ("ZImage", "z_image", True),
        "z-image-turbo": ("ZImage", "z_image_turbo", True),
        # FIBO
        "fibo": ("FIBO", "fibo", True),
        "fibo-lite": ("FIBO", "fibo_lite", True),
        # FIBO Edit
        "fibo-edit": ("FIBOEdit", "fibo_edit", True),
        "fibo-edit-rmbg": ("FIBOEdit", "fibo_edit_rmbg", True),
        # Qwen
        "qwen-image": ("QwenImage", "qwen_image", True),
        # Qwen Edit
        "qwen-image-edit": ("QwenImageEdit", "qwen_image_edit", True),
        # SeedVR2 (no lora support)
        "seedvr2-3b": ("SeedVR2", "seedvr2_3b", False),
        "seedvr2-7b": ("SeedVR2", "seedvr2_7b", False),
    }


# ---------------------------------------------------------------------------
# HuggingFace cache detection (static — no mflux imports triggered)
# ---------------------------------------------------------------------------

# Static mapping from config_factory_name -> HuggingFace repo ID.
# Kept in sync with the values returned by ModelConfig factory methods.
# This avoids triggering heavy mflux imports just to read repo names.
_REPO_MAP: dict[str, str] = {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
    "flux2_klein_4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2_klein_9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2_klein_base_4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2_klein_base_9b": "black-forest-labs/FLUX.2-klein-base-9B",
    "z_image": "Tongyi-MAI/Z-Image",
    "z_image_turbo": "Tongyi-MAI/Z-Image-Turbo",
    "fibo": "briaai/FIBO",
    "fibo_lite": "briaai/Fibo-lite",
    "fibo_edit": "briaai/Fibo-Edit",
    "fibo_edit_rmbg": "briaai/Fibo-Edit-RMBG",
    "qwen_image": "Qwen/Qwen-Image",
    "qwen_image_edit": "Qwen/Qwen-Image-Edit-2509",
    "seedvr2_3b": "numz/SeedVR2_comfyUI",
    "seedvr2_7b": "numz/SeedVR2_comfyUI",
}


def _default_hf_cache_dir() -> Path:
    """Return the default HuggingFace Hub cache directory.

    Respects the ``HF_HOME`` and ``HF_HUB_CACHE`` environment variables used
    by the ``huggingface_hub`` library.  Falls back to ``~/.cache/huggingface/hub``.
    """
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env)
    hf_home = os.environ.get("HF_HOME", os.path.join("~", ".cache", "huggingface"))
    return Path(os.path.expanduser(os.path.join(hf_home, "hub")))


def is_model_cached(
    repo_id: str,
    cache_dir: Path | str | None = None,
) -> bool:
    """Check whether a HuggingFace model's weights are locally cached.

    Looks for ``.safetensors`` files inside the model's snapshot directory.
    A directory that exists but contains only metadata (no weight files) is
    treated as *not* cached.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``"black-forest-labs/FLUX.2-klein-4B"``).
        cache_dir: Override for the HF Hub cache directory. Defaults to the
            standard HuggingFace cache location.

    Returns:
        ``True`` if the model directory exists and contains at least one
        ``.safetensors`` file, ``False`` otherwise.
    """
    if cache_dir is None:
        cache_dir = _default_hf_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    # HF Hub stores repos in dirs named models--{org}--{model}
    dir_name = "models--" + repo_id.replace("/", "--")
    model_dir = cache_dir / dir_name

    if not model_dir.is_dir():
        return False

    # Walk the snapshots directory looking for .safetensors weight files.
    # We avoid walking blobs (which are content-addressed and lack extensions)
    # to keep the check fast and unambiguous.
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False

    for _root, _dirs, files in os.walk(snapshots_dir):
        for f in files:
            if f.endswith(".safetensors"):
                return True

    return False


class ModelCache:
    """Thread-safe lazy model cache keyed by (model_name, quantize).

    Usage::

        cache = ModelCache()
        model = cache.get_model("schnell", quantize=4)
        # subsequent calls with same args return the cached instance
        same_model = cache.get_model("schnell", quantize=4)
        assert model is same_model
    """

    _REGISTRY = _build_registry()

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int | None], Any] = {}
        self._lock = threading.Lock()
        self._imports: dict[str, Any] | None = None
        self._imports_lock = threading.Lock()

    def _get_imports(self) -> dict[str, Any]:
        """Lazily import mflux classes on first use."""
        if self._imports is None:
            with self._imports_lock:
                if self._imports is None:
                    self._imports = _lazy_imports()
        return self._imports

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported model names."""
        return list(self._REGISTRY.keys())

    def get_model(self, model_name: str, quantize: int | None = None) -> Any:
        """Get or lazily load a cached model instance.

        Args:
            model_name: Model name string (e.g. "schnell", "flux2-klein-4b").
            quantize: Quantization bit-width (4, 8, or None for full precision).

        Returns:
            The loaded model instance.

        Raises:
            ValueError: If model_name is not in the registry.
            RuntimeError: If the model fails to load.
        """
        key = (model_name, quantize)

        # Fast path: check cache without loading anything
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Validate model name before doing expensive work
        if model_name not in self._REGISTRY:
            available = ", ".join(sorted(self._REGISTRY.keys()))
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )

        class_key, config_factory_name, supports_lora = self._REGISTRY[model_name]

        # Get lazily-imported classes
        imports = self._get_imports()
        model_cls = imports[class_key]
        model_config_cls = imports["ModelConfig"]
        config_factory = getattr(model_config_cls, config_factory_name)

        try:
            model = model_cls(quantize=quantize, model_config=config_factory())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' with quantize={quantize}: {e}"
            ) from e

        with self._lock:
            # Double-check after loading (another thread may have loaded it)
            if key not in self._cache:
                self._cache[key] = model
            return self._cache[key]

    def clear(self) -> None:
        """Clear all cached models, releasing memory."""
        with self._lock:
            self._cache.clear()
