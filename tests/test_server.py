"""Tests for the MCP server, generate_image, and edit_image tools.

All tests mock the model cache and mflux generation to avoid loading heavy
MLX models or performing actual GPU inference.
"""

import io
import os
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest
from fastmcp.utilities.types import Image

from server import mcp, generate_image, edit_image, list_models, get_image_metadata, cache, parse_args
from mflux_cache import ModelCache, _REPO_MAP, is_model_cached, _default_hf_cache_dir


class TestServerSetup:
    """Tests for basic server configuration."""

    def test_server_has_name(self):
        assert mcp.name == "mflux-mcp"

    @pytest.mark.asyncio
    async def test_generate_image_tool_is_registered(self):
        """The generate_image tool should be registered on the MCP server."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "generate_image" in tool_names

    @pytest.mark.asyncio
    async def test_edit_image_tool_is_registered(self):
        """The edit_image tool should be registered on the MCP server."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "edit_image" in tool_names


class TestGenerateImageDefaults:
    """Tests that generate_image has the correct parameter defaults."""

    def test_default_model(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["model"].default == "flux2-klein-4b"

    def test_default_dimensions(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["width"].default == 1024
        assert sig.parameters["height"].default == 1024

    def test_default_steps(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["steps"].default == 4

    def test_default_seed_is_none(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["seed"].default is None

    def test_default_quantize(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["quantize"].default == 8

    def test_prompt_is_required(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["prompt"].default is inspect.Parameter.empty


class TestGenerateImageHappyPath:
    """Tests for successful image generation with mocked model."""

    def _make_mock_model(self, width=64, height=64):
        """Create a mock model that returns a real PIL Image."""
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="red")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model, mock_generated

    @patch.object(cache, "get_model")
    def test_returns_image_with_default_params(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = generate_image(prompt="A red square")

        assert result is not None
        # Result should be a FastMCP Image with PNG data
        assert result.data is not None
        assert len(result.data) > 0

    @patch.object(cache, "get_model")
    def test_calls_cache_with_correct_model_and_quantize(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        generate_image(prompt="test", model="schnell", quantize=4)

        mock_get_model.assert_called_once_with("schnell", quantize=4, lora_style=None)

    @patch.object(cache, "get_model")
    def test_passes_params_to_model_generate(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        generate_image(
            prompt="A blue cat",
            width=512,
            height=768,
            steps=8,
            seed=42,
        )

        mock_model.generate_image.assert_called_once_with(
            seed=42,
            prompt="A blue cat",
            num_inference_steps=8,
            width=512,
            height=768,
        )

    @patch.object(cache, "get_model")
    def test_auto_generates_seed_when_none(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        generate_image(prompt="test", seed=None)

        call_kwargs = mock_model.generate_image.call_args[1]
        seed = call_kwargs["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    @patch.object(cache, "get_model")
    def test_returned_data_is_valid_png(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = generate_image(prompt="test")

        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert result.data[:4] == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_returned_image_format_is_png(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = generate_image(prompt="test")

        assert result._format == "png"

    @patch.object(cache, "get_model")
    def test_custom_dimensions_used(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        generate_image(prompt="test", width=256, height=384)

        call_kwargs = mock_model.generate_image.call_args[1]
        assert call_kwargs["width"] == 256
        assert call_kwargs["height"] == 384

    @patch.object(cache, "get_model")
    def test_explicit_seed_is_used(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        generate_image(prompt="test", seed=12345)

        call_kwargs = mock_model.generate_image.call_args[1]
        assert call_kwargs["seed"] == 12345


class TestGenerateImageErrors:
    """Tests for error handling in generate_image."""

    @patch.object(cache, "get_model")
    def test_invalid_model_raises_value_error(self, mock_get_model):
        mock_get_model.side_effect = ValueError("Unknown model: 'bad-model'")

        with pytest.raises(ValueError, match="Unknown model"):
            generate_image(prompt="test", model="bad-model")

    @patch.object(cache, "get_model")
    def test_model_load_failure_raises_runtime_error(self, mock_get_model):
        mock_get_model.side_effect = RuntimeError("Failed to load model")

        with pytest.raises(RuntimeError, match="Failed to load"):
            generate_image(prompt="test")

    @patch.object(cache, "get_model")
    def test_generation_failure_propagates(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.generate_image.side_effect = RuntimeError("MLX error")
        mock_get_model.return_value = mock_model

        with pytest.raises(RuntimeError, match="MLX error"):
            generate_image(prompt="test")


# ---------------------------------------------------------------------------
# edit_image tests
# ---------------------------------------------------------------------------


class TestEditImageDefaults:
    """Tests that edit_image has the correct parameter defaults."""

    def test_default_model(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["model"].default == "flux2-klein-edit"

    def test_default_steps(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["steps"].default == 4

    def test_default_seed_is_none(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["seed"].default is None

    def test_default_quantize(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["quantize"].default == 8

    def test_image_paths_is_required(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["image_paths"].default is inspect.Parameter.empty

    def test_prompt_is_required(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["prompt"].default is inspect.Parameter.empty


class TestEditImageHappyPath:
    """Tests for successful image editing with mocked model."""

    def _make_mock_model(self, width=64, height=64):
        """Create a mock model that returns a real PIL Image."""
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="blue")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model, mock_generated

    @patch.object(cache, "get_model")
    def test_returns_image_with_default_params(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = edit_image(image_paths=["input.jpg"], prompt="Make it blue")

        assert result is not None
        assert result.data is not None
        assert len(result.data) > 0

    @patch.object(cache, "get_model")
    def test_calls_cache_with_correct_model_and_quantize(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(
            image_paths=["input.jpg"],
            prompt="test",
            model="fibo-edit",
            quantize=4,
        )

        mock_get_model.assert_called_once_with("fibo-edit", quantize=4, lora_style=None)

    @patch.object(cache, "get_model")
    def test_flux2_klein_edit_passes_image_paths_list(self, mock_get_model):
        """Flux2KleinEdit should receive image_paths (plural list)."""
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(
            image_paths=["person.jpg", "glasses.jpg"],
            prompt="Make the woman wear the eyeglasses",
            model="flux2-klein-edit",
            steps=6,
            seed=42,
        )

        mock_model.generate_image.assert_called_once_with(
            seed=42,
            prompt="Make the woman wear the eyeglasses",
            image_paths=["person.jpg", "glasses.jpg"],
            num_inference_steps=6,
        )

    @patch.object(cache, "get_model")
    def test_fibo_edit_passes_singular_image_path(self, mock_get_model):
        """FIBOEdit should receive image_path (singular), not image_paths."""
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(
            image_paths=["photo.jpg"],
            prompt="Remove background",
            model="fibo-edit",
            steps=10,
            seed=99,
        )

        mock_model.generate_image.assert_called_once_with(
            seed=99,
            prompt="Remove background",
            image_path="photo.jpg",
            num_inference_steps=10,
        )

    @patch.object(cache, "get_model")
    def test_fibo_edit_rmbg_passes_singular_image_path(self, mock_get_model):
        """fibo-edit-rmbg should also use singular image_path (same FIBOEdit class)."""
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(
            image_paths=["photo.jpg", "extra.jpg"],
            prompt="Remove background",
            model="fibo-edit-rmbg",
            steps=5,
            seed=7,
        )

        mock_model.generate_image.assert_called_once_with(
            seed=7,
            prompt="Remove background",
            image_path="photo.jpg",
            num_inference_steps=5,
        )

    @patch.object(cache, "get_model")
    def test_qwen_image_edit_passes_image_paths_list(self, mock_get_model):
        """QwenImageEdit should receive image_paths (plural list)."""
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(
            image_paths=["scene.jpg"],
            prompt="Add a hat",
            model="qwen-image-edit",
            steps=4,
            seed=123,
        )

        mock_model.generate_image.assert_called_once_with(
            seed=123,
            prompt="Add a hat",
            image_paths=["scene.jpg"],
            num_inference_steps=4,
        )

    @patch.object(cache, "get_model")
    def test_auto_generates_seed_when_none(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(image_paths=["input.jpg"], prompt="test", seed=None)

        call_kwargs = mock_model.generate_image.call_args[1]
        seed = call_kwargs["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    @patch.object(cache, "get_model")
    def test_returned_data_is_valid_png(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = edit_image(image_paths=["input.jpg"], prompt="test")

        # PNG magic bytes
        assert result.data[:4] == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_returned_image_format_is_png(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        result = edit_image(image_paths=["input.jpg"], prompt="test")

        assert result._format == "png"

    @patch.object(cache, "get_model")
    def test_explicit_seed_is_used(self, mock_get_model):
        mock_model, _ = self._make_mock_model()
        mock_get_model.return_value = mock_model

        edit_image(image_paths=["input.jpg"], prompt="test", seed=54321)

        call_kwargs = mock_model.generate_image.call_args[1]
        assert call_kwargs["seed"] == 54321


class TestEditImageErrors:
    """Tests for error handling in edit_image."""

    def test_empty_image_paths_raises_value_error(self):
        with pytest.raises(ValueError, match="image_paths must contain at least one"):
            edit_image(image_paths=[], prompt="test")

    @patch.object(cache, "get_model")
    def test_invalid_model_raises_value_error(self, mock_get_model):
        mock_get_model.side_effect = ValueError("Unknown model: 'bad-model'")

        with pytest.raises(ValueError, match="Unknown model"):
            edit_image(image_paths=["input.jpg"], prompt="test", model="bad-model")

    @patch.object(cache, "get_model")
    def test_model_load_failure_raises_runtime_error(self, mock_get_model):
        mock_get_model.side_effect = RuntimeError("Failed to load model")

        with pytest.raises(RuntimeError, match="Failed to load"):
            edit_image(image_paths=["input.jpg"], prompt="test")

    @patch.object(cache, "get_model")
    def test_generation_failure_propagates(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.generate_image.side_effect = RuntimeError("MLX error")
        mock_get_model.return_value = mock_model

        with pytest.raises(RuntimeError, match="MLX error"):
            edit_image(image_paths=["input.jpg"], prompt="test")


# ---------------------------------------------------------------------------
# list_models tests
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for the list_models MCP tool."""

    @pytest.mark.asyncio
    async def test_list_models_tool_is_registered(self):
        """The list_models tool should be registered on the MCP server."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_models" in tool_names

    def test_returns_a_list(self):
        result = list_models()
        assert isinstance(result, list)

    def test_returns_non_empty_list(self):
        result = list_models()
        assert len(result) > 0

    def test_returns_correct_number_of_models(self):
        result = list_models()
        assert len(result) == len(ModelCache._REGISTRY)

    def test_each_entry_is_a_dict(self):
        result = list_models()
        for entry in result:
            assert isinstance(entry, dict)

    def test_each_entry_has_required_fields(self):
        required_fields = {"name", "family", "capability", "supports_lora", "quantize_options", "is_downloaded"}
        result = list_models()
        for entry in result:
            assert required_fields.issubset(entry.keys()), (
                f"Missing fields in {entry.get('name', '?')}: "
                f"{required_fields - entry.keys()}"
            )

    def test_all_registry_models_are_present(self):
        result = list_models()
        result_names = {entry["name"] for entry in result}
        registry_names = set(ModelCache._REGISTRY.keys())
        assert result_names == registry_names

    def test_all_capabilities_represented(self):
        result = list_models()
        capabilities = {entry["capability"] for entry in result}
        assert capabilities == {"txt2img", "edit", "upscale"}

    def test_all_families_represented(self):
        result = list_models()
        families = {entry["family"] for entry in result}
        expected = {"FLUX.1", "FLUX.2", "Z-Image", "FIBO", "Qwen", "SeedVR2"}
        assert families == expected

    def test_supports_lora_is_boolean(self):
        result = list_models()
        for entry in result:
            assert isinstance(entry["supports_lora"], bool), (
                f"supports_lora is not bool for {entry['name']}"
            )

    def test_seedvr2_models_do_not_support_lora(self):
        result = list_models()
        seedvr2_models = [e for e in result if e["family"] == "SeedVR2"]
        assert len(seedvr2_models) == 2
        for entry in seedvr2_models:
            assert entry["supports_lora"] is False

    def test_quantize_options_present(self):
        result = list_models()
        for entry in result:
            assert entry["quantize_options"] == [4, 8, None]

    def test_is_downloaded_is_boolean(self):
        """is_downloaded field should be a boolean for every model."""
        result = list_models()
        for entry in result:
            assert isinstance(entry["is_downloaded"], bool), (
                f"is_downloaded is not bool for {entry['name']}"
            )

    def test_does_not_trigger_mflux_import(self):
        """list_models should only read the static _REGISTRY, not trigger mflux imports."""
        import sys
        # Remove mflux from sys.modules if present, to detect fresh imports
        mflux_modules = [k for k in sys.modules if k.startswith("mflux.")]
        for mod in mflux_modules:
            del sys.modules[mod]

        list_models()

        imported = [k for k in sys.modules if k.startswith("mflux.")]
        assert imported == [], f"list_models triggered mflux imports: {imported}"


# ---------------------------------------------------------------------------
# is_model_cached / HF cache detection tests
# ---------------------------------------------------------------------------


class TestIsModelCached:
    """Tests for the is_model_cached function using tmp_path fixtures."""

    def _make_cached_model(self, tmp_path, repo_id, with_safetensors=True):
        """Create a fake HF cache directory structure for a model.

        Args:
            tmp_path: pytest tmp_path fixture.
            repo_id: HuggingFace repo ID (e.g. "org/model").
            with_safetensors: If True, create .safetensors files in snapshots.

        Returns:
            The tmp_path (to pass as cache_dir).
        """
        dir_name = "models--" + repo_id.replace("/", "--")
        model_dir = tmp_path / dir_name
        snapshots_dir = model_dir / "snapshots" / "abc123"
        transformer_dir = snapshots_dir / "transformer"
        transformer_dir.mkdir(parents=True)

        # Always create metadata files
        (transformer_dir / "config.json").write_text("{}")

        if with_safetensors:
            (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 100)

        # Create blobs and refs dirs (realistic structure)
        (model_dir / "blobs").mkdir(exist_ok=True)
        (model_dir / "refs").mkdir(exist_ok=True)

        return tmp_path

    def test_returns_true_for_fully_cached_model(self, tmp_path):
        """A model with .safetensors files should be detected as cached."""
        cache_dir = self._make_cached_model(tmp_path, "org/my-model", with_safetensors=True)
        assert is_model_cached("org/my-model", cache_dir=cache_dir) is True

    def test_returns_false_for_missing_model(self, tmp_path):
        """A model not in the cache at all should return False."""
        assert is_model_cached("org/nonexistent-model", cache_dir=tmp_path) is False

    def test_returns_false_for_partial_download(self, tmp_path):
        """A model dir with metadata but no .safetensors should return False."""
        cache_dir = self._make_cached_model(tmp_path, "org/partial-model", with_safetensors=False)
        assert is_model_cached("org/partial-model", cache_dir=cache_dir) is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        """A model dir that exists but is empty should return False."""
        dir_name = "models--org--empty-model"
        (tmp_path / dir_name).mkdir()
        assert is_model_cached("org/empty-model", cache_dir=tmp_path) is False

    def test_returns_false_when_no_snapshots_dir(self, tmp_path):
        """A model dir without a snapshots subdirectory should return False."""
        dir_name = "models--org--no-snapshots"
        model_dir = tmp_path / dir_name
        model_dir.mkdir()
        (model_dir / "blobs").mkdir()
        (model_dir / "refs").mkdir()
        assert is_model_cached("org/no-snapshots", cache_dir=tmp_path) is False

    def test_handles_nested_safetensors(self, tmp_path):
        """Safetensors files in nested subdirectories should be detected."""
        dir_name = "models--org--nested-model"
        deep_dir = tmp_path / dir_name / "snapshots" / "rev1" / "text_encoder" / "sub"
        deep_dir.mkdir(parents=True)
        (deep_dir / "model.safetensors").write_bytes(b"\x00" * 50)
        # Also need blobs/refs for realism
        (tmp_path / dir_name / "blobs").mkdir(exist_ok=True)
        (tmp_path / dir_name / "refs").mkdir(exist_ok=True)
        assert is_model_cached("org/nested-model", cache_dir=tmp_path) is True

    def test_repo_id_with_special_chars(self, tmp_path):
        """Repo IDs with dots and hyphens should map correctly to cache dirs."""
        cache_dir = self._make_cached_model(tmp_path, "black-forest-labs/FLUX.2-klein-4B")
        assert is_model_cached("black-forest-labs/FLUX.2-klein-4B", cache_dir=cache_dir) is True

    def test_does_not_load_model_weights(self, tmp_path):
        """Calling is_model_cached should not import mflux or load models."""
        import sys
        mflux_before = set(k for k in sys.modules if k.startswith("mflux."))

        cache_dir = self._make_cached_model(tmp_path, "org/test-model")
        is_model_cached("org/test-model", cache_dir=cache_dir)

        mflux_after = set(k for k in sys.modules if k.startswith("mflux."))
        new_imports = mflux_after - mflux_before
        assert new_imports == set(), f"is_model_cached triggered mflux imports: {new_imports}"


class TestRepoMap:
    """Tests for the _REPO_MAP static mapping."""

    def test_repo_map_covers_all_registry_config_factories(self):
        """Every config_factory_name in _REGISTRY should have an entry in _REPO_MAP."""
        for name, (_class_key, config_factory_name, _lora) in ModelCache._REGISTRY.items():
            assert config_factory_name in _REPO_MAP, (
                f"Model '{name}' uses config factory '{config_factory_name}' "
                f"which is missing from _REPO_MAP"
            )

    def test_repo_map_values_are_valid_hf_repo_ids(self):
        """Each _REPO_MAP value should be a valid org/model format."""
        for factory_name, repo_id in _REPO_MAP.items():
            assert "/" in repo_id, (
                f"_REPO_MAP['{factory_name}'] = '{repo_id}' is not a valid HF repo ID"
            )
            parts = repo_id.split("/")
            assert len(parts) == 2, (
                f"_REPO_MAP['{factory_name}'] = '{repo_id}' should have exactly one '/'"
            )
            assert all(part for part in parts), (
                f"_REPO_MAP['{factory_name}'] = '{repo_id}' has empty org or model name"
            )


class TestDefaultHfCacheDir:
    """Tests for the _default_hf_cache_dir helper."""

    def test_default_path_ends_with_hub(self):
        """Default cache dir should end with 'hub'."""
        from pathlib import Path
        result = _default_hf_cache_dir()
        assert result.name == "hub"

    @patch.dict(os.environ, {"HF_HUB_CACHE": "/custom/cache/path"}, clear=False)
    def test_respects_hf_hub_cache_env(self):
        """HF_HUB_CACHE env var should override the default."""
        from pathlib import Path
        result = _default_hf_cache_dir()
        assert result == Path("/custom/cache/path")

    @patch.dict(os.environ, {"HF_HOME": "/custom/hf_home"}, clear=False)
    def test_respects_hf_home_env(self):
        """HF_HOME env var should be used when HF_HUB_CACHE is not set."""
        from pathlib import Path
        # Make sure HF_HUB_CACHE isn't set
        env = os.environ.copy()
        env.pop("HF_HUB_CACHE", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["HF_HOME"] = "/custom/hf_home"
            result = _default_hf_cache_dir()
            assert result == Path("/custom/hf_home/hub")


class TestListModelsDownloadStatus:
    """Integration tests for the is_downloaded field in list_models."""

    @patch("server.is_model_cached")
    def test_list_models_calls_is_model_cached_for_each_model(self, mock_cached):
        """list_models should check cache status for every model."""
        mock_cached.return_value = False
        result = list_models()
        # Should have been called once per model in registry
        assert mock_cached.call_count == len(ModelCache._REGISTRY)

    @patch("server.is_model_cached", return_value=True)
    def test_all_cached_shows_all_downloaded(self, mock_cached):
        """When all models are cached, every entry should have is_downloaded=True."""
        result = list_models()
        for entry in result:
            assert entry["is_downloaded"] is True

    @patch("server.is_model_cached", return_value=False)
    def test_none_cached_shows_none_downloaded(self, mock_cached):
        """When no models are cached, every entry should have is_downloaded=False."""
        result = list_models()
        for entry in result:
            assert entry["is_downloaded"] is False

    @patch("server.is_model_cached")
    def test_mixed_cache_status(self, mock_cached):
        """When some models are cached and others aren't, is_downloaded reflects reality."""
        # Make only "schnell" appear cached
        def side_effect(repo_id):
            return repo_id == "black-forest-labs/FLUX.1-schnell"
        mock_cached.side_effect = side_effect

        result = list_models()
        schnell = next(e for e in result if e["name"] == "schnell")
        dev = next(e for e in result if e["name"] == "dev")
        assert schnell["is_downloaded"] is True
        assert dev["is_downloaded"] is False


# ---------------------------------------------------------------------------
# get_image_metadata tests
# ---------------------------------------------------------------------------


class TestGetImageMetadata:
    """Tests for the get_image_metadata MCP tool."""

    @pytest.mark.asyncio
    async def test_get_image_metadata_tool_is_registered(self):
        """The get_image_metadata tool should be registered on the MCP server."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "get_image_metadata" in tool_names

    @patch("server.MetadataReader.read_all_metadata")
    @patch("server.os.path.isfile", return_value=True)
    def test_happy_path_returns_metadata(self, mock_isfile, mock_read):
        """When metadata is present, the full dict is returned."""
        mock_read.return_value = {
            "exif": {
                "prompt": "A red panda",
                "model": "flux2-klein-4b",
                "seed": 42,
                "steps": 4,
                "width": 1024,
                "height": 1024,
            },
            "xmp": {
                "description": "A red panda",
                "creator_tool": "mflux",
            },
        }

        result = get_image_metadata("/tmp/test_image.png")

        mock_isfile.assert_called_once_with("/tmp/test_image.png")
        mock_read.assert_called_once_with("/tmp/test_image.png")
        assert result["exif"]["prompt"] == "A red panda"
        assert result["exif"]["seed"] == 42
        assert result["xmp"]["creator_tool"] == "mflux"

    @patch("server.MetadataReader.read_all_metadata")
    @patch("server.os.path.isfile", return_value=True)
    def test_exif_only_returns_metadata(self, mock_isfile, mock_read):
        """When only EXIF metadata is present (no XMP), the dict is still returned."""
        mock_read.return_value = {
            "exif": {"prompt": "A blue cat", "seed": 7},
            "xmp": None,
        }

        result = get_image_metadata("/tmp/test.png")

        assert result["exif"]["prompt"] == "A blue cat"
        assert result["xmp"] is None

    def test_file_not_found_raises_value_error(self):
        """When the file does not exist, a ValueError is raised."""
        with pytest.raises(ValueError, match="File not found"):
            get_image_metadata("/nonexistent/path/to/image.png")

    @patch("server.MetadataReader.read_all_metadata")
    @patch("server.os.path.isfile", return_value=True)
    def test_no_metadata_returns_message(self, mock_isfile, mock_read):
        """When no mflux metadata is found, a message dict is returned."""
        mock_read.return_value = {"exif": None, "xmp": None}

        result = get_image_metadata("/tmp/plain_photo.jpg")

        assert "message" in result
        assert "No mflux metadata found" in result["message"]
        assert result["image_path"] == "/tmp/plain_photo.jpg"


# ---------------------------------------------------------------------------
# CLI transport option tests
# ---------------------------------------------------------------------------


class TestCLITransport:
    """Tests for CLI argument parsing (--transport, --port)."""

    def test_default_transport_is_stdio(self):
        """Running with no arguments defaults to stdio transport."""
        args = parse_args([])
        assert args.transport == "stdio"

    def test_default_port_is_8000(self):
        """Running with no arguments defaults to port 8000."""
        args = parse_args([])
        assert args.port == 8000

    def test_transport_http(self):
        """--transport http sets transport to http."""
        args = parse_args(["--transport", "http"])
        assert args.transport == "http"

    def test_transport_stdio_explicit(self):
        """--transport stdio explicitly sets stdio transport."""
        args = parse_args(["--transport", "stdio"])
        assert args.transport == "stdio"

    def test_port_custom(self):
        """--port sets a custom port number."""
        args = parse_args(["--port", "9090"])
        assert args.port == 9090

    def test_transport_http_with_port(self):
        """--transport http --port 3000 sets both correctly."""
        args = parse_args(["--transport", "http", "--port", "3000"])
        assert args.transport == "http"
        assert args.port == 3000

    def test_invalid_transport_raises_system_exit(self):
        """An invalid transport choice causes argparse to exit with error."""
        with pytest.raises(SystemExit):
            parse_args(["--transport", "websocket"])

    def test_invalid_port_raises_system_exit(self):
        """A non-integer port causes argparse to exit with error."""
        with pytest.raises(SystemExit):
            parse_args(["--port", "not-a-number"])

    @patch.object(mcp, "run")
    def test_mcp_run_called_with_stdio_defaults(self, mock_run):
        """When transport is stdio, mcp.run is called with transport only (no port)."""
        args = parse_args([])
        kwargs: dict = {"transport": args.transport}
        if args.transport == "http":
            kwargs["port"] = args.port
        mcp.run(**kwargs)

        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(mcp, "run")
    def test_mcp_run_called_with_http_and_port(self, mock_run):
        """When transport is http, mcp.run is called with transport and port."""
        args = parse_args(["--transport", "http", "--port", "4567"])
        kwargs: dict = {"transport": args.transport}
        if args.transport == "http":
            kwargs["port"] = args.port
        mcp.run(**kwargs)

        mock_run.assert_called_once_with(transport="http", port=4567)


# ---------------------------------------------------------------------------
# output_path tests for generate_image and edit_image
# ---------------------------------------------------------------------------


class TestGenerateImageOutputPath:
    """Tests for the output_path parameter on generate_image."""

    def _make_mock_model(self, width=64, height=64):
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="red")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image
        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model

    def test_output_path_default_is_none(self):
        import inspect
        sig = inspect.signature(generate_image)
        assert sig.parameters["output_path"].default is None

    @patch.object(cache, "get_model")
    def test_without_output_path_returns_image(self, mock_get_model):
        """When output_path is omitted, returns a FastMCP Image (unchanged behavior)."""
        mock_get_model.return_value = self._make_mock_model()

        result = generate_image(prompt="test")

        assert isinstance(result, Image)
        assert result.data[:4] == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_with_output_path_returns_string(self, mock_get_model, tmp_path):
        """When output_path is provided, returns the absolute file path as a string."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "output.png")

        result = generate_image(prompt="test", output_path=out)

        assert isinstance(result, str)
        assert result == out

    @patch.object(cache, "get_model")
    def test_with_output_path_file_is_created(self, mock_get_model, tmp_path):
        """When output_path is provided, the file is actually written to disk."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "output.png")

        generate_image(prompt="test", output_path=out)

        assert os.path.isfile(out)

    @patch.object(cache, "get_model")
    def test_with_output_path_file_is_valid_png(self, mock_get_model, tmp_path):
        """The file written to disk should be a valid PNG."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "output.png")

        generate_image(prompt="test", output_path=out)

        with open(out, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_with_output_path_creates_parent_dirs(self, mock_get_model, tmp_path):
        """Parent directories should be created automatically."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "nested" / "deep" / "output.png")

        result = generate_image(prompt="test", output_path=out)

        assert os.path.isfile(out)
        assert result == out

    @patch.object(cache, "get_model")
    def test_with_output_path_returns_absolute_path(self, mock_get_model, tmp_path):
        """Even if a relative path is given, the returned path should be absolute."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "output.png")

        result = generate_image(prompt="test", output_path=out)

        assert os.path.isabs(result)


class TestEditImageOutputPath:
    """Tests for the output_path parameter on edit_image."""

    def _make_mock_model(self, width=64, height=64):
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="blue")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image
        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model

    def test_output_path_default_is_none(self):
        import inspect
        sig = inspect.signature(edit_image)
        assert sig.parameters["output_path"].default is None

    @patch.object(cache, "get_model")
    def test_without_output_path_returns_image(self, mock_get_model):
        """When output_path is omitted, returns a FastMCP Image (unchanged behavior)."""
        mock_get_model.return_value = self._make_mock_model()

        result = edit_image(image_paths=["input.jpg"], prompt="test")

        assert isinstance(result, Image)
        assert result.data[:4] == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_with_output_path_returns_string(self, mock_get_model, tmp_path):
        """When output_path is provided, returns the absolute file path as a string."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "edited.png")

        result = edit_image(image_paths=["input.jpg"], prompt="test", output_path=out)

        assert isinstance(result, str)
        assert result == out

    @patch.object(cache, "get_model")
    def test_with_output_path_file_is_created(self, mock_get_model, tmp_path):
        """When output_path is provided, the file is actually written to disk."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "edited.png")

        edit_image(image_paths=["input.jpg"], prompt="test", output_path=out)

        assert os.path.isfile(out)

    @patch.object(cache, "get_model")
    def test_with_output_path_file_is_valid_png(self, mock_get_model, tmp_path):
        """The file written to disk should be a valid PNG."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "edited.png")

        edit_image(image_paths=["input.jpg"], prompt="test", output_path=out)

        with open(out, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    @patch.object(cache, "get_model")
    def test_with_output_path_creates_parent_dirs(self, mock_get_model, tmp_path):
        """Parent directories should be created automatically."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "nested" / "deep" / "edited.png")

        result = edit_image(image_paths=["input.jpg"], prompt="test", output_path=out)

        assert os.path.isfile(out)
        assert result == out

    @patch.object(cache, "get_model")
    def test_with_output_path_returns_absolute_path(self, mock_get_model, tmp_path):
        """Even if a relative path is given, the returned path should be absolute."""
        mock_get_model.return_value = self._make_mock_model()
        out = str(tmp_path / "edited.png")

        result = edit_image(image_paths=["input.jpg"], prompt="test", output_path=out)

        assert os.path.isabs(result)


# ---------------------------------------------------------------------------
# lora_style parameter tests
# ---------------------------------------------------------------------------


class TestGenerateImageLoraStyle:
    """Tests for the lora_style parameter on generate_image."""

    def _make_mock_model(self, width=64, height=64):
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="red")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image
        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model

    @patch.object(cache, "get_model")
    def test_lora_style_passed_to_get_model(self, mock_get_model):
        """lora_style should be forwarded to cache.get_model."""
        mock_get_model.return_value = self._make_mock_model()

        generate_image(prompt="x", lora_style="portrait")

        mock_get_model.assert_called_once_with("flux2-klein-4b", quantize=8, lora_style="portrait")

    @patch.object(cache, "get_model")
    def test_no_lora_style_default(self, mock_get_model):
        """When lora_style is omitted, None should be passed to cache.get_model."""
        mock_get_model.return_value = self._make_mock_model()

        generate_image(prompt="x")

        mock_get_model.assert_called_once_with("flux2-klein-4b", quantize=8, lora_style=None)


class TestEditImageLoraStyle:
    """Tests for the lora_style parameter on edit_image."""

    def _make_mock_model(self, width=64, height=64):
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="blue")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image
        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model

    @patch.object(cache, "get_model")
    def test_lora_style_passed_to_get_model(self, mock_get_model):
        """lora_style should be forwarded to cache.get_model for edit_image."""
        mock_get_model.return_value = self._make_mock_model()

        edit_image(image_paths=["img.png"], prompt="x", lora_style="portrait")

        mock_get_model.assert_called_once_with("flux2-klein-edit", quantize=8, lora_style="portrait")

    @patch.object(cache, "get_model")
    def test_no_lora_style_default(self, mock_get_model):
        """When lora_style is omitted, None should be passed to cache.get_model."""
        mock_get_model.return_value = self._make_mock_model()

        edit_image(image_paths=["img.png"], prompt="x")

        mock_get_model.assert_called_once_with("flux2-klein-edit", quantize=8, lora_style=None)
