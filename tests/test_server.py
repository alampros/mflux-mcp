"""Tests for mflux-mcp server v2 — queue-based architecture.

All tests mock the model cache, job queue, and worker manager to avoid loading
heavy MLX models or performing actual GPU inference.
"""

import asyncio
import inspect
import json
import os
import sqlite3
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import PIL.Image
import pytest

import server as server_module
from job_queue import JobQueue
from mflux_cache import ModelCache, _REPO_MAP, is_model_cached, _default_hf_cache_dir
from server import (
    mcp,
    generate_image,
    edit_image,
    upscale_image,
    list_jobs,
    get_job,
    cancel_job,
    list_models,
    get_image_metadata,
    clear_cache,
    get_system_status,
    parse_args,
)
from worker import WorkerManager


# ---------------------------------------------------------------------------
# Server setup / tool registration
# ---------------------------------------------------------------------------


class TestServerSetup:
    """Tests for basic server configuration and tool registration."""

    def test_server_has_name(self):
        assert mcp.name == "mflux-mcp"

    @pytest.mark.asyncio
    async def test_generate_image_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "generate_image" in tool_names

    @pytest.mark.asyncio
    async def test_edit_image_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "edit_image" in tool_names

    @pytest.mark.asyncio
    async def test_upscale_image_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "upscale_image" in tool_names

    @pytest.mark.asyncio
    async def test_list_jobs_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_jobs" in tool_names

    @pytest.mark.asyncio
    async def test_get_job_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "get_job" in tool_names

    @pytest.mark.asyncio
    async def test_cancel_job_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "cancel_job" in tool_names

    @pytest.mark.asyncio
    async def test_list_models_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_models" in tool_names

    @pytest.mark.asyncio
    async def test_get_image_metadata_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "get_image_metadata" in tool_names

    @pytest.mark.asyncio
    async def test_clear_cache_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "clear_cache" in tool_names

    @pytest.mark.asyncio
    async def test_get_system_status_tool_registered(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "get_system_status" in tool_names


# ---------------------------------------------------------------------------
# generate_image tool
# ---------------------------------------------------------------------------


class TestGenerateImageTool:
    """Tests for the generate_image MCP tool (queue-based)."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_descriptor(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "test-job-id",
            "status": "queued",
            "command": "generate_image",
            "output_path": "/tmp/out.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            result = await generate_image(
                prompt="A red square", output_path="/tmp/out.png"
            )

        assert result["job_id"] == "test-job-id"
        assert result["status"] == "queued"
        assert result["command"] == "generate_image"
        assert result["output_path"] == "/tmp/out.png"
        assert result["backend"] == "thread"

    def test_default_params(self):
        sig = inspect.signature(generate_image)
        assert sig.parameters["model"].default == "flux2-klein-4b"
        assert sig.parameters["width"].default == 1024
        assert sig.parameters["height"].default == 1024
        assert sig.parameters["steps"].default == 4
        assert sig.parameters["seed"].default is None
        assert sig.parameters["quantize"].default == 8
        assert sig.parameters["lora_style"].default is None
        assert sig.parameters["backend"].default == "thread"
        assert sig.parameters["timeout"].default == 300.0

    @pytest.mark.asyncio
    async def test_custom_params(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "test-id",
            "status": "queued",
            "command": "generate_image",
            "output_path": "/tmp/out.png",
            "backend": "subprocess",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await generate_image(
                prompt="custom",
                output_path="/tmp/out.png",
                model="schnell",
                width=512,
                height=768,
                steps=8,
                seed=42,
                quantize=4,
                lora_style="portrait",
                backend="subprocess",
                timeout=600.0,
            )

        mock_queue.submit.assert_called_once()
        call_kwargs = mock_queue.submit.call_args[1]
        assert call_kwargs["command"] == "generate_image"
        assert call_kwargs["params"]["model"] == "schnell"
        assert call_kwargs["params"]["width"] == 512
        assert call_kwargs["params"]["height"] == 768
        assert call_kwargs["params"]["steps"] == 8
        assert call_kwargs["params"]["seed"] == 42
        assert call_kwargs["params"]["quantize"] == 4
        assert call_kwargs["params"]["lora_style"] == "portrait"
        assert call_kwargs["backend"] == "subprocess"
        assert call_kwargs["timeout_s"] == 600.0

    @pytest.mark.asyncio
    async def test_seed_auto_generation(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "test-id",
            "status": "queued",
            "command": "generate_image",
            "output_path": "/tmp/out.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await generate_image(prompt="test", output_path="/tmp/out.png", seed=None)

        call_kwargs = mock_queue.submit.call_args[1]
        seed = call_kwargs["params"]["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    @pytest.mark.asyncio
    async def test_invalid_model(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Unknown model"):
                await generate_image(
                    prompt="test", output_path="/tmp/out.png", model="not-a-model"
                )

    @pytest.mark.asyncio
    async def test_invalid_backend(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Invalid backend"):
                await generate_image(
                    prompt="test", output_path="/tmp/out.png", backend="gpu"
                )

    @pytest.mark.asyncio
    async def test_invalid_quantize(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Invalid quantize"):
                await generate_image(
                    prompt="test", output_path="/tmp/out.png", quantize=16
                )

    @pytest.mark.asyncio
    async def test_invalid_lora_style(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Invalid lora_style"):
                await generate_image(
                    prompt="test",
                    output_path="/tmp/out.png",
                    lora_style="cyberpunk",
                )

    @pytest.mark.asyncio
    async def test_queue_not_initialized(self):
        with patch.object(server_module, "_queue", None):
            with pytest.raises(RuntimeError, match="queue is not available"):
                await generate_image(prompt="test", output_path="/tmp/out.png")


# ---------------------------------------------------------------------------
# edit_image tool
# ---------------------------------------------------------------------------


class TestEditImageTool:
    """Tests for the edit_image MCP tool (queue-based)."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_descriptor(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "edit-job-id",
            "status": "queued",
            "command": "edit_image",
            "output_path": "/tmp/out.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            result = await edit_image(
                image_paths=["input.jpg"],
                prompt="Make it blue",
                output_path="/tmp/out.png",
            )

        assert result["job_id"] == "edit-job-id"
        assert result["status"] == "queued"
        assert result["command"] == "edit_image"

    def test_edit_specific_defaults(self):
        sig = inspect.signature(edit_image)
        assert sig.parameters["model"].default == "flux2-klein-edit"
        assert sig.parameters["steps"].default == 4
        assert sig.parameters["seed"].default is None
        assert sig.parameters["quantize"].default == 8

    @pytest.mark.asyncio
    async def test_invalid_edit_model(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="does not support image editing"):
                await edit_image(
                    image_paths=["input.jpg"],
                    prompt="test",
                    output_path="/tmp/out.png",
                    model="schnell",
                )

    @pytest.mark.asyncio
    async def test_requires_image_paths(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(
                ValueError, match="image_paths must contain at least one"
            ):
                await edit_image(
                    image_paths=[],
                    prompt="test",
                    output_path="/tmp/out.png",
                )

    @pytest.mark.asyncio
    async def test_valid_edit_models(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "id",
            "status": "queued",
            "command": "edit_image",
            "output_path": "/tmp/out.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await edit_image(
                image_paths=["in.png"],
                prompt="test",
                output_path="/tmp/out.png",
                model="flux2-klein-edit",
            )
            await edit_image(
                image_paths=["in.png"],
                prompt="test",
                output_path="/tmp/out.png",
                model="fibo-edit",
            )
            await edit_image(
                image_paths=["in.png"],
                prompt="test",
                output_path="/tmp/out.png",
                model="qwen-image-edit",
            )


# ---------------------------------------------------------------------------
# upscale_image tool
# ---------------------------------------------------------------------------


class TestUpscaleImageTool:
    """Tests for the upscale_image MCP tool."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_descriptor(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "u1",
            "status": "queued",
            "command": "upscale_image",
            "output_path": "/tmp/up.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            result = await upscale_image(
                image_path="input.png",
                output_path="/tmp/up.png",
            )

        assert result["job_id"] == "u1"
        assert result["status"] == "queued"
        assert result["command"] == "upscale_image"

    @pytest.mark.asyncio
    async def test_default_params(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "u2",
            "status": "queued",
            "command": "upscale_image",
            "output_path": "/tmp/up.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await upscale_image(
                image_path="input.png",
                output_path="/tmp/up.png",
            )

        call_kwargs = mock_queue.submit.call_args[1]
        params = call_kwargs["params"]
        assert params["model"] == "seedvr2-3b"
        assert params["resolution"] == 2160
        assert params["softness"] == 0.5
        assert params["quantize"] == 8
        assert call_kwargs["command"] == "upscale_image"

    @pytest.mark.asyncio
    async def test_custom_params(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "u3",
            "status": "queued",
            "command": "upscale_image",
            "output_path": "/tmp/up.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await upscale_image(
                image_path="input.png",
                output_path="/tmp/up.png",
                model="seedvr2-7b",
                resolution=4320,
                softness=0.3,
                seed=42,
                quantize=4,
            )

        params = mock_queue.submit.call_args[1]["params"]
        assert params["model"] == "seedvr2-7b"
        assert params["resolution"] == 4320
        assert params["softness"] == 0.3
        assert params["seed"] == 42
        assert params["quantize"] == 4

    @pytest.mark.asyncio
    async def test_invalid_model(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Unknown model"):
                await upscale_image(
                    image_path="input.png",
                    output_path="/tmp/up.png",
                    model="not-a-model",
                )

    @pytest.mark.asyncio
    async def test_non_upscale_model_rejected(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="does not support upscaling"):
                await upscale_image(
                    image_path="input.png",
                    output_path="/tmp/up.png",
                    model="flux2-klein-4b",
                )

    @pytest.mark.asyncio
    async def test_queue_not_initialized(self):
        with patch.object(server_module, "_queue", None):
            with pytest.raises(RuntimeError, match="queue is not available"):
                await upscale_image(
                    image_path="input.png",
                    output_path="/tmp/up.png",
                )

    @pytest.mark.asyncio
    async def test_seed_auto_generation(self, mock_queue):
        mock_queue.submit.return_value = {
            "job_id": "u4",
            "status": "queued",
            "command": "upscale_image",
            "output_path": "/tmp/up.png",
            "backend": "thread",
        }
        with patch.object(server_module, "_queue", mock_queue):
            await upscale_image(
                image_path="input.png",
                output_path="/tmp/up.png",
            )

        params = mock_queue.submit.call_args[1]["params"]
        assert isinstance(params["seed"], int)


# ---------------------------------------------------------------------------
# list_jobs tool
# ---------------------------------------------------------------------------


class TestListJobsTool:
    """Tests for the list_jobs MCP tool."""

    def test_list_all_jobs(self, mock_queue):
        mock_queue.list_jobs.return_value = [
            {"job_id": "1", "status": "queued"},
            {"job_id": "2", "status": "completed"},
        ]
        with patch.object(server_module, "_queue", mock_queue):
            result = list_jobs()

        assert len(result) == 2
        mock_queue.list_jobs.assert_called_once_with(status=None, limit=50)

    def test_filter_by_status(self, mock_queue):
        mock_queue.list_jobs.return_value = [{"job_id": "1", "status": "running"}]
        with patch.object(server_module, "_queue", mock_queue):
            result = list_jobs(status="running")

        mock_queue.list_jobs.assert_called_once_with(status="running", limit=50)

    def test_with_limit(self, mock_queue):
        mock_queue.list_jobs.return_value = []
        with patch.object(server_module, "_queue", mock_queue):
            result = list_jobs(limit=10)

        mock_queue.list_jobs.assert_called_once_with(status=None, limit=10)

    def test_queue_not_initialized(self):
        with patch.object(server_module, "_queue", None):
            with pytest.raises(RuntimeError, match="queue is not available"):
                list_jobs()


# ---------------------------------------------------------------------------
# get_job tool
# ---------------------------------------------------------------------------


class TestGetJobTool:
    """Tests for the get_job MCP tool."""

    def test_get_existing_job(self, mock_queue):
        mock_queue.get_job.return_value = {"job_id": "abc", "status": "queued"}
        with patch.object(server_module, "_queue", mock_queue):
            result = get_job("abc")

        assert result["job_id"] == "abc"
        assert result["status"] == "queued"
        mock_queue.get_job.assert_called_once_with("abc")

    def test_get_missing_job(self, mock_queue):
        mock_queue.get_job.return_value = None
        with patch.object(server_module, "_queue", mock_queue):
            result = get_job("missing")

        assert result is None

    def test_get_multiple_jobs(self, mock_queue):
        mock_queue.get_job.side_effect = [
            {"job_id": "a", "status": "completed"},
            {"job_id": "b", "status": "running"},
            None,
        ]
        with patch.object(server_module, "_queue", mock_queue):
            result = get_job(["a", "b", "missing"])

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["job_id"] == "a"
        assert result[1]["job_id"] == "b"
        assert result[2] is None

    def test_get_multiple_preserves_order(self, mock_queue):
        mock_queue.get_job.side_effect = lambda jid: {"job_id": jid, "status": "queued"}
        with patch.object(server_module, "_queue", mock_queue):
            result = get_job(["z", "a", "m"])

        assert [r["job_id"] for r in result] == ["z", "a", "m"]

    def test_get_empty_list_returns_empty(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            result = get_job([])

        assert result == []

    def test_queue_not_initialized(self):
        with patch.object(server_module, "_queue", None):
            with pytest.raises(RuntimeError, match="queue is not available"):
                get_job("abc")

    def test_queue_not_initialized_with_list(self):
        with patch.object(server_module, "_queue", None):
            with pytest.raises(RuntimeError, match="queue is not available"):
                get_job(["abc", "def"])


# ---------------------------------------------------------------------------
# cancel_job tool
# ---------------------------------------------------------------------------


class TestCancelJobTool:
    """Tests for the cancel_job MCP tool."""

    @pytest.mark.asyncio
    async def test_cancel_success(self, mock_worker_manager):
        mock_worker_manager.cancel_job.return_value = True
        with patch.object(server_module, "_worker_manager", mock_worker_manager):
            result = await cancel_job("job-123")

        assert result["job_id"] == "job-123"
        assert result["cancelled"] is True
        mock_worker_manager.cancel_job.assert_awaited_once_with("job-123")

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, mock_worker_manager):
        mock_worker_manager.cancel_job.return_value = False
        with patch.object(server_module, "_worker_manager", mock_worker_manager):
            result = await cancel_job("missing")

        assert result["cancelled"] is False

    @pytest.mark.asyncio
    async def test_worker_manager_not_initialized(self):
        with patch.object(server_module, "_worker_manager", None):
            with pytest.raises(RuntimeError, match="worker manager is not available"):
                await cancel_job("abc")


# ---------------------------------------------------------------------------
# get_system_status tool
# ---------------------------------------------------------------------------


class TestGetSystemStatusTool:
    """Tests for the get_system_status MCP tool."""

    def _make_mlx_module(self, active_mb=100, peak_mb=200, cache_mb=50):
        """Build a fake mlx.core module using types.ModuleType."""
        mock_mx = types.ModuleType("mlx.core")
        mock_mx.metal = types.ModuleType("mlx.core.metal")
        mock_mx.get_active_memory = MagicMock(return_value=active_mb * 1024 * 1024)
        mock_mx.get_peak_memory = MagicMock(return_value=peak_mb * 1024 * 1024)
        mock_mx.get_cache_memory = MagicMock(return_value=cache_mb * 1024 * 1024)
        mock_mx.device_info = MagicMock(
            return_value={
                "device_name": "Apple M3 Max",
                "gpu_core_count": 40,
                "recommended_max_working_set_size": 24 * 1024**3,
            }
        )

        mock_mlx = types.ModuleType("mlx")
        mock_mlx.core = mock_mx
        return mock_mlx, mock_mx

    def test_with_all_data(self, mock_queue, mock_cache):
        mock_queue.list_jobs.return_value = [
            {"job_id": "1", "status": "queued"},
            {"job_id": "2", "status": "running"},
            {"job_id": "3", "status": "completed"},
        ]
        mock_cache._cache.keys.return_value = [("schnell", 8, None), ("dev", 4, None)]

        mock_psutil = MagicMock()
        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 16 * 1024**3
        mock_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_mem

        mock_mlx, mock_mx = self._make_mlx_module()

        with patch.object(server_module, "_queue", mock_queue):
            with patch.object(server_module, "_cache", mock_cache):
                with patch.object(server_module, "psutil", mock_psutil):
                    with patch.dict(
                        "sys.modules",
                        {"mlx": mock_mlx, "mlx.core": mock_mx},
                    ):
                        result = get_system_status()

        assert result["ram"] is not None
        assert result["ram"]["total_gb"] == 32.0
        assert result["ram"]["available_gb"] == 16.0
        assert result["ram"]["percent_used"] == 50.0

        assert result["metal"] is not None
        assert result["metal"]["active_mb"] == 100
        assert result["metal"]["peak_mb"] == 200
        assert result["metal"]["cache_mb"] == 50

        assert result["chip"] is not None
        assert result["chip"]["name"] == "Apple M3 Max"
        assert result["chip"]["gpu_cores"] == 40
        assert result["chip"]["recommended_max_gb"] == 24.0

        assert result["queue"] == {"queued": 1, "running": 1}
        assert result["cached_models"] == ["schnell (q8)", "dev (q4)"]

    def test_without_psutil(self, mock_queue, mock_cache):
        with patch.object(server_module, "_queue", mock_queue):
            with patch.object(server_module, "_cache", mock_cache):
                with patch.object(server_module, "psutil", None):
                    result = get_system_status()

        assert result["ram"] is None

    def test_without_mlx(self, mock_queue, mock_cache):
        mock_mx = types.ModuleType("mlx.core")
        mock_mx.metal = types.ModuleType("mlx.core.metal")
        mock_mx.get_active_memory = MagicMock(side_effect=RuntimeError("No MLX"))
        mock_mx.get_peak_memory = MagicMock(side_effect=RuntimeError("No MLX"))
        mock_mx.get_cache_memory = MagicMock(side_effect=RuntimeError("No MLX"))
        mock_mx.device_info = MagicMock(side_effect=RuntimeError("No MLX"))

        mock_mlx = types.ModuleType("mlx")
        mock_mlx.core = mock_mx

        with patch.object(server_module, "_queue", mock_queue):
            with patch.object(server_module, "_cache", mock_cache):
                with patch.dict(
                    "sys.modules",
                    {"mlx": mock_mlx, "mlx.core": mock_mx},
                ):
                    result = get_system_status()

        assert result["metal"] is None
        assert result["chip"] is None

    def test_queue_counts(self, mock_queue, mock_cache):
        mock_queue.list_jobs.return_value = [
            {"job_id": "1", "status": "queued"},
            {"job_id": "2", "status": "queued"},
            {"job_id": "3", "status": "running"},
        ]
        with patch.object(server_module, "_queue", mock_queue):
            with patch.object(server_module, "_cache", mock_cache):
                with patch.object(server_module, "psutil", None):
                    mock_mlx = types.ModuleType("mlx")
                    mock_mlx.core = types.ModuleType("mlx.core")
                    with patch.dict(
                        "sys.modules",
                        {"mlx": mock_mlx, "mlx.core": mock_mlx.core},
                    ):
                        result = get_system_status()

        assert result["queue"] == {"queued": 2, "running": 1}


# ---------------------------------------------------------------------------
# list_models tool
# ---------------------------------------------------------------------------


class TestListModelsTool:
    """Tests for the list_models MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_list(self):
        result = list_models()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_model_structure(self):
        required_fields = {
            "name",
            "family",
            "capability",
            "tool",
            "supports_lora",
            "quantize_options",
            "is_downloaded",
        }
        result = list_models()
        for entry in result:
            assert required_fields.issubset(entry.keys()), (
                f"Missing fields in {entry.get('name', '?')}: "
                f"{required_fields - entry.keys()}"
            )

    def test_all_registry_models_present(self):
        result = list_models()
        result_names = {entry["name"] for entry in result}
        assert result_names == set(ModelCache._REGISTRY.keys())

    def test_all_capabilities_represented(self):
        result = list_models()
        capabilities = {entry["capability"] for entry in result}
        assert capabilities == {"txt2img", "edit", "upscale"}

    def test_tool_field_matches_capability(self):
        expected_map = {
            "txt2img": "generate_image",
            "edit": "edit_image",
            "upscale": "upscale_image",
        }
        result = list_models()
        for entry in result:
            assert entry["tool"] == expected_map[entry["capability"]], (
                f"Model {entry['name']} has capability={entry['capability']} "
                f"but tool={entry['tool']}, expected {expected_map[entry['capability']]}"
            )

    def test_all_families_represented(self):
        result = list_models()
        families = {entry["family"] for entry in result}
        expected = {"FLUX.1", "FLUX.2", "Z-Image", "FIBO", "Qwen", "SeedVR2"}
        assert families == expected

    def test_supports_lora_is_boolean(self):
        result = list_models()
        for entry in result:
            assert isinstance(entry["supports_lora"], bool)

    def test_quantize_options_present(self):
        result = list_models()
        for entry in result:
            assert entry["quantize_options"] == [4, 8, None]

    def test_does_not_trigger_mflux_import(self):
        mflux_modules = [k for k in sys.modules if k.startswith("mflux.")]
        for mod in mflux_modules:
            del sys.modules[mod]

        list_models()

        imported = [k for k in sys.modules if k.startswith("mflux.")]
        assert imported == [], f"list_models triggered mflux imports: {imported}"

    @patch("server.is_model_cached")
    def test_download_status(self, mock_cached):
        mock_cached.return_value = False
        result = list_models()
        for entry in result:
            assert entry["is_downloaded"] is False
        assert mock_cached.call_count == len(ModelCache._REGISTRY)

    @patch("server.is_model_cached", return_value=True)
    def test_all_cached_shows_all_downloaded(self, mock_cached):
        result = list_models()
        for entry in result:
            assert entry["is_downloaded"] is True

    @patch("server.is_model_cached", return_value=False)
    def test_none_cached_shows_none_downloaded(self, mock_cached):
        result = list_models()
        for entry in result:
            assert entry["is_downloaded"] is False

    @patch("server.is_model_cached")
    def test_mixed_cache_status(self, mock_cached):
        def side_effect(repo_id):
            return repo_id == "black-forest-labs/FLUX.1-schnell"

        mock_cached.side_effect = side_effect
        result = list_models()
        schnell = next(e for e in result if e["name"] == "schnell")
        dev = next(e for e in result if e["name"] == "dev")
        assert schnell["is_downloaded"] is True
        assert dev["is_downloaded"] is False


# ---------------------------------------------------------------------------
# get_image_metadata tool
# ---------------------------------------------------------------------------


class TestGetImageMetadataTool:
    """Tests for the get_image_metadata MCP tool."""

    @patch("server.MetadataReader.read_all_metadata")
    @patch("server.os.path.isfile", return_value=True)
    def test_valid_image(self, mock_isfile, mock_read):
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

    def test_missing_image(self):
        with pytest.raises(ValueError, match="File not found"):
            get_image_metadata("/nonexistent/path/to/image.png")

    @patch("server.MetadataReader.read_all_metadata")
    @patch("server.os.path.isfile", return_value=True)
    def test_no_metadata_returns_message(self, mock_isfile, mock_read):
        mock_read.return_value = {"exif": None, "xmp": None}

        result = get_image_metadata("/tmp/plain_photo.jpg")

        assert "message" in result
        assert "No mflux metadata found" in result["message"]
        assert result["image_path"] == "/tmp/plain_photo.jpg"


# ---------------------------------------------------------------------------
# clear_cache tool
# ---------------------------------------------------------------------------


class TestClearCacheTool:
    """Tests for the clear_cache MCP tool."""

    def test_clear_success(self, mock_cache):
        with patch.object(server_module, "_cache", mock_cache):
            clear_cache()
        mock_cache.clear.assert_called_once()

    def test_reports_cleared_count(self, mock_cache):
        mock_cache.size = 3
        with patch.object(server_module, "_cache", mock_cache):
            result = clear_cache()

        assert result["status"] == "ok"
        assert result["models_cleared"] == 3
        assert "3" in result["message"]

    def test_cache_not_initialized(self):
        with patch.object(server_module, "_cache", None):
            with pytest.raises(RuntimeError, match="cache is not available"):
                clear_cache()


# ---------------------------------------------------------------------------
# Job lifecycle integration
# ---------------------------------------------------------------------------


class TestJobLifecycleIntegration:
    """Integration-style tests using a real JobQueue with mocked inference."""

    @pytest.mark.asyncio
    async def test_submit_poll_complete(self, tmp_path):
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)

        with patch.object(server_module, "_queue", queue):
            job_desc = await generate_image(
                prompt="test", output_path=str(tmp_path / "out.png")
            )
            assert job_desc["status"] == "queued"

            # Manually complete the job (simulating worker completion)
            queue.update_status(
                job_desc["job_id"], "completed", completed_at=queue._now_iso()
            )

            fetched = get_job(job_desc["job_id"])
            assert fetched["status"] == "completed"

    def test_lazy_purge(self, tmp_path):
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)

        # Insert an old completed job directly into SQLite
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, command, params, backend,
                output_path, created_at, completed_at, timeout_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-job",
                "completed",
                "generate_image",
                "{}",
                "thread",
                "out.png",
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:00:00+00:00",
                300.0,
            ),
        )
        conn.commit()
        conn.close()

        with patch.object(server_module, "_queue", queue):
            # get_job triggers lazy purge
            result = get_job("old-job")
            assert result is None

    @pytest.mark.asyncio
    async def test_multiple_jobs_queued(self, tmp_path):
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)

        with patch.object(server_module, "_queue", queue):
            job1 = await generate_image(
                prompt="first", output_path=str(tmp_path / "1.png")
            )
            job2 = await generate_image(
                prompt="second", output_path=str(tmp_path / "2.png")
            )
            job3 = await edit_image(
                image_paths=["in.png"],
                prompt="third",
                output_path=str(tmp_path / "3.png"),
            )

            jobs = list_jobs()
            assert len(jobs) == 3
            ids = {j["job_id"] for j in jobs}
            assert job1["job_id"] in ids
            assert job2["job_id"] in ids
            assert job3["job_id"] in ids

    @pytest.mark.asyncio
    async def test_purge_via_list_jobs(self, tmp_path):
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, command, params, backend,
                output_path, created_at, completed_at, timeout_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-job",
                "failed",
                "generate_image",
                "{}",
                "thread",
                "out.png",
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:00:00+00:00",
                300.0,
            ),
        )
        conn.commit()
        conn.close()

        with patch.object(server_module, "_queue", queue):
            jobs = list_jobs()
            ids = [j["job_id"] for j in jobs]
            assert "old-job" not in ids


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for validation errors and error propagation."""

    @pytest.mark.asyncio
    async def test_invalid_lora_style(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Invalid lora_style"):
                await generate_image(
                    prompt="test",
                    output_path="/tmp/out.png",
                    lora_style="cyberpunk",
                )

    @pytest.mark.asyncio
    async def test_validation_errors_dont_create_jobs(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Unknown model"):
                await generate_image(
                    prompt="test",
                    output_path="/tmp/out.png",
                    model="bad-model",
                )
        mock_queue.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_model(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="Unknown model"):
                await generate_image(
                    prompt="test",
                    output_path="/tmp/out.png",
                    model="not-real",
                )

    @pytest.mark.asyncio
    async def test_edit_validation_does_not_create_job(self, mock_queue):
        with patch.object(server_module, "_queue", mock_queue):
            with pytest.raises(ValueError, match="does not support image editing"):
                await edit_image(
                    image_paths=["in.png"],
                    prompt="test",
                    output_path="/tmp/out.png",
                    model="schnell",
                )
        mock_queue.submit.assert_not_called()


# ---------------------------------------------------------------------------
# CLI transport
# ---------------------------------------------------------------------------


class TestCLITransport:
    """Tests for CLI argument parsing (--transport, --port)."""

    def test_default_transport_is_stdio(self):
        args = parse_args([])
        assert args.transport == "stdio"

    def test_default_port_is_8000(self):
        args = parse_args([])
        assert args.port == 8000

    def test_transport_http(self):
        args = parse_args(["--transport", "http"])
        assert args.transport == "http"

    def test_transport_stdio_explicit(self):
        args = parse_args(["--transport", "stdio"])
        assert args.transport == "stdio"

    def test_port_custom(self):
        args = parse_args(["--port", "9090"])
        assert args.port == 9090

    def test_transport_http_with_port(self):
        args = parse_args(["--transport", "http", "--port", "3000"])
        assert args.transport == "http"
        assert args.port == 3000

    def test_invalid_transport_raises_system_exit(self):
        with pytest.raises(SystemExit):
            parse_args(["--transport", "websocket"])

    def test_invalid_port_raises_system_exit(self):
        with pytest.raises(SystemExit):
            parse_args(["--port", "not-a-number"])

    @patch.object(mcp, "run")
    def test_mcp_run_called_with_stdio_defaults(self, mock_run):
        args = parse_args([])
        kwargs: dict = {"transport": args.transport}
        if args.transport == "http":
            kwargs["port"] = args.port
        mcp.run(**kwargs)
        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(mcp, "run")
    def test_mcp_run_called_with_http_and_port(self, mock_run):
        args = parse_args(["--transport", "http", "--port", "4567"])
        kwargs: dict = {"transport": args.transport}
        if args.transport == "http":
            kwargs["port"] = args.port
        mcp.run(**kwargs)
        mock_run.assert_called_once_with(transport="http", port=4567)


# ---------------------------------------------------------------------------
# is_model_cached / HF cache detection
# ---------------------------------------------------------------------------


class TestIsModelCached:
    """Tests for the is_model_cached function."""

    def _make_cached_model(self, tmp_path, repo_id, with_safetensors=True):
        dir_name = "models--" + repo_id.replace("/", "--")
        model_dir = tmp_path / dir_name
        snapshots_dir = model_dir / "snapshots" / "abc123"
        transformer_dir = snapshots_dir / "transformer"
        transformer_dir.mkdir(parents=True)
        (transformer_dir / "config.json").write_text("{}")
        if with_safetensors:
            (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(
                b"\x00" * 100
            )
        (model_dir / "blobs").mkdir(exist_ok=True)
        (model_dir / "refs").mkdir(exist_ok=True)
        return tmp_path

    def test_returns_true_for_fully_cached_model(self, tmp_path):
        cache_dir = self._make_cached_model(tmp_path, "org/my-model")
        assert is_model_cached("org/my-model", cache_dir=cache_dir) is True

    def test_returns_false_for_missing_model(self, tmp_path):
        assert is_model_cached("org/nonexistent", cache_dir=tmp_path) is False

    def test_returns_false_for_partial_download(self, tmp_path):
        cache_dir = self._make_cached_model(
            tmp_path, "org/partial-model", with_safetensors=False
        )
        assert is_model_cached("org/partial-model", cache_dir=cache_dir) is False

    def test_does_not_load_model_weights(self, tmp_path):
        mflux_before = set(k for k in sys.modules if k.startswith("mflux."))
        cache_dir = self._make_cached_model(tmp_path, "org/test-model")
        is_model_cached("org/test-model", cache_dir=cache_dir)
        mflux_after = set(k for k in sys.modules if k.startswith("mflux."))
        new_imports = mflux_after - mflux_before
        assert new_imports == set(), (
            f"is_model_cached triggered mflux imports: {new_imports}"
        )


# ---------------------------------------------------------------------------
# Repo map
# ---------------------------------------------------------------------------


class TestRepoMap:
    """Tests for the _REPO_MAP static mapping."""

    def test_repo_map_covers_all_registry_config_factories(self):
        for name, (
            _class_key,
            config_factory_name,
            _lora,
        ) in ModelCache._REGISTRY.items():
            assert config_factory_name in _REPO_MAP, (
                f"Model '{name}' uses config factory '{config_factory_name}' "
                f"which is missing from _REPO_MAP"
            )

    def test_repo_map_values_are_valid_hf_repo_ids(self):
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


# ---------------------------------------------------------------------------
# Default HF cache dir
# ---------------------------------------------------------------------------


class TestDefaultHfCacheDir:
    """Tests for the _default_hf_cache_dir helper."""

    def test_default_path_ends_with_hub(self):
        result = _default_hf_cache_dir()
        assert result.name == "hub"

    @patch.dict(os.environ, {"HF_HUB_CACHE": "/custom/cache/path"}, clear=False)
    def test_respects_hf_hub_cache_env(self):
        from pathlib import Path

        result = _default_hf_cache_dir()
        assert result == Path("/custom/cache/path")

    @patch.dict(os.environ, {"HF_HOME": "/custom/hf_home"}, clear=False)
    def test_respects_hf_home_env(self):
        from pathlib import Path

        env = os.environ.copy()
        env.pop("HF_HUB_CACHE", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["HF_HOME"] = "/custom/hf_home"
            result = _default_hf_cache_dir()
            assert result == Path("/custom/hf_home/hub")
