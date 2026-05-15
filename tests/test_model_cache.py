"""Tests for the lazy model cache.

All tests mock the mflux model classes to avoid loading heavy MLX models.
"""

from unittest.mock import MagicMock, patch
import pytest

from mflux_cache import ModelCache, _build_registry


class TestModelCacheRegistry:
    """Tests for the model registry and supported_models."""

    def test_supported_models_returns_list(self):
        cache = ModelCache()
        models = cache.supported_models
        assert isinstance(models, list)
        assert len(models) > 0

    def test_supported_models_contains_expected_entries(self):
        cache = ModelCache()
        models = cache.supported_models
        expected = [
            "schnell",
            "dev",
            "flux2-klein-4b",
            "flux2-klein-9b",
            "z-image",
            "z-image-turbo",
            "fibo",
            "fibo-edit",
            "qwen-image",
            "qwen-image-edit",
            "seedvr2-3b",
            "seedvr2-7b",
        ]
        for name in expected:
            assert name in models, f"Expected '{name}' in supported models"

    def test_registry_has_correct_structure(self):
        registry = _build_registry()
        for name, entry in registry.items():
            assert len(entry) == 3, f"Registry entry for '{name}' should be a 3-tuple"
            class_key, config_name, supports_lora = entry
            assert isinstance(class_key, str)
            assert isinstance(config_name, str)
            assert isinstance(supports_lora, bool)

    def test_seedvr2_models_do_not_support_lora(self):
        registry = _build_registry()
        assert registry["seedvr2-3b"][2] is False
        assert registry["seedvr2-7b"][2] is False

    def test_non_seedvr2_models_support_lora(self):
        registry = _build_registry()
        for name, (_, _, supports_lora) in registry.items():
            if not name.startswith("seedvr2"):
                assert supports_lora is True, f"Expected '{name}' to support lora"


class TestModelCacheGetModel:
    """Tests for get_model with mocked model classes."""

    def _make_cache_with_mock(self):
        """Create a ModelCache with mocked imports."""
        cache = ModelCache()

        mock_model_config = MagicMock()
        # Create distinct factory methods that return unique config objects
        for method_name in [
            "schnell", "dev", "flux2_klein_4b", "flux2_klein_9b",
            "flux2_klein_base_4b", "flux2_klein_base_9b",
            "z_image", "z_image_turbo", "fibo", "fibo_lite",
            "fibo_edit", "fibo_edit_rmbg", "qwen_image",
            "qwen_image_edit", "seedvr2_3b", "seedvr2_7b",
        ]:
            getattr(mock_model_config, method_name).return_value = MagicMock(
                name=f"config_{method_name}"
            )

        # Use side_effect to return a new MagicMock on each call, so that
        # different cache keys produce distinct instances (MagicMock() returns
        # the same child mock by default).
        mock_imports = {
            "Flux1": MagicMock(name="Flux1", side_effect=lambda **kw: MagicMock()),
            "Flux2Klein": MagicMock(name="Flux2Klein", side_effect=lambda **kw: MagicMock()),
            "Flux2KleinEdit": MagicMock(name="Flux2KleinEdit", side_effect=lambda **kw: MagicMock()),
            "ZImage": MagicMock(name="ZImage", side_effect=lambda **kw: MagicMock()),
            "FIBO": MagicMock(name="FIBO", side_effect=lambda **kw: MagicMock()),
            "FIBOEdit": MagicMock(name="FIBOEdit", side_effect=lambda **kw: MagicMock()),
            "QwenImage": MagicMock(name="QwenImage", side_effect=lambda **kw: MagicMock()),
            "QwenImageEdit": MagicMock(name="QwenImageEdit", side_effect=lambda **kw: MagicMock()),
            "SeedVR2": MagicMock(name="SeedVR2", side_effect=lambda **kw: MagicMock()),
            "ModelConfig": mock_model_config,
        }
        # Inject mocked imports
        cache._imports = mock_imports
        return cache, mock_imports

    def test_returns_same_instance_on_repeated_calls(self):
        cache, mocks = self._make_cache_with_mock()
        model1 = cache.get_model("schnell", quantize=4)
        model2 = cache.get_model("schnell", quantize=4)
        assert model1 is model2

    def test_model_constructor_called_once_for_same_key(self):
        cache, mocks = self._make_cache_with_mock()
        cache.get_model("schnell", quantize=4)
        cache.get_model("schnell", quantize=4)
        # Constructor should only be called once due to caching
        assert mocks["Flux1"].call_count == 1

    def test_different_model_names_get_different_instances(self):
        cache, mocks = self._make_cache_with_mock()
        model_schnell = cache.get_model("schnell")
        model_dev = cache.get_model("dev")
        assert model_schnell is not model_dev

    def test_different_quantize_values_get_different_instances(self):
        cache, mocks = self._make_cache_with_mock()
        model_none = cache.get_model("schnell", quantize=None)
        model_4 = cache.get_model("schnell", quantize=4)
        model_8 = cache.get_model("schnell", quantize=8)
        assert model_none is not model_4
        assert model_none is not model_8
        assert model_4 is not model_8

    def test_invalid_model_name_raises_value_error(self):
        cache = ModelCache()
        with pytest.raises(ValueError, match="Unknown model"):
            cache.get_model("nonexistent-model")

    def test_invalid_model_error_includes_available_models(self):
        cache = ModelCache()
        with pytest.raises(ValueError, match="schnell"):
            cache.get_model("nonexistent-model")

    def test_model_constructor_receives_correct_args(self):
        cache, mocks = self._make_cache_with_mock()
        cache.get_model("schnell", quantize=8)
        mocks["Flux1"].assert_called_once()
        call_kwargs = mocks["Flux1"].call_args[1]
        assert call_kwargs["quantize"] == 8
        assert "model_config" in call_kwargs

    def test_load_error_raises_runtime_error(self):
        cache, mocks = self._make_cache_with_mock()
        mocks["Flux1"].side_effect = Exception("OOM: not enough memory")
        with pytest.raises(RuntimeError, match="Failed to load model 'schnell'"):
            cache.get_model("schnell")

    def test_load_error_preserves_original_exception(self):
        cache, mocks = self._make_cache_with_mock()
        original = MemoryError("out of memory")
        mocks["Flux1"].side_effect = original
        with pytest.raises(RuntimeError) as exc_info:
            cache.get_model("schnell")
        assert exc_info.value.__cause__ is original

    def test_flux2_klein_models_use_correct_class(self):
        cache, mocks = self._make_cache_with_mock()
        cache.get_model("flux2-klein-4b")
        mocks["Flux2Klein"].assert_called_once()

    def test_seedvr2_uses_correct_class(self):
        cache, mocks = self._make_cache_with_mock()
        cache.get_model("seedvr2-3b")
        mocks["SeedVR2"].assert_called_once()


class TestModelCacheClear:
    """Tests for cache clearing."""

    def test_clear_empties_cache(self):
        cache = ModelCache()
        # Manually populate the internal cache
        cache._cache[("schnell", None)] = MagicMock(name="fake_model")
        assert len(cache._cache) == 1
        cache.clear()
        assert len(cache._cache) == 0

    def test_clear_allows_reload(self):
        cache = ModelCache()
        mock_imports = {
            "Flux1": MagicMock(name="Flux1", side_effect=lambda **kw: MagicMock()),
            "ModelConfig": MagicMock(),
        }
        mock_imports["ModelConfig"].schnell.return_value = MagicMock()
        cache._imports = mock_imports

        model1 = cache.get_model("schnell")
        cache.clear()
        model2 = cache.get_model("schnell")

        # After clear, a new instance should be created
        assert model1 is not model2
        assert mock_imports["Flux1"].call_count == 2


class TestModelCacheLazyImport:
    """Tests that mflux imports are deferred."""

    def test_no_mflux_import_on_init(self):
        """ModelCache.__init__ should NOT trigger mflux imports."""
        cache = ModelCache()
        # _imports should be None until first get_model call
        assert cache._imports is None

    def test_supported_models_does_not_trigger_import(self):
        """Accessing supported_models should NOT trigger mflux imports."""
        cache = ModelCache()
        _ = cache.supported_models
        assert cache._imports is None
