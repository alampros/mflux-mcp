"""Tests for the subprocess runner.

All tests mock the model cache and mflux generation to avoid loading heavy
MLX models or performing actual GPU inference.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest

from job_queue import JobQueue
from mflux_cache import ModelCache
from subprocess_runner import run_job
import subprocess_runner


class _MockHelpers:
    """Shared helper methods for creating mock models."""

    @staticmethod
    def make_mock_model(width=64, height=64):
        """Create a mock model that returns a real PIL Image."""
        mock_pil_image = PIL.Image.new("RGB", (width, height), color="red")
        mock_generated = MagicMock()
        mock_generated.image = mock_pil_image

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_generated
        return mock_model, mock_generated


class TestSubprocessRunnerGenerate:
    """Tests for happy-path generate_image via subprocess runner."""

    def _submit_generate_job(self, queue, **overrides):
        """Submit a generate_image job with default params overridden."""
        params = {
            "prompt": "A red square",
            "model": "flux2-klein-4b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        return queue.submit(
            command="generate_image",
            params=params,
            output_path=output_path,
            backend="subprocess",
        )

    @patch.object(ModelCache, "get_model")
    def test_happy_path_completes(self, mock_get_model, tmp_path):
        """A valid generate job runs to completion."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(tmp_path / "out.png"))

        run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "completed"
        assert updated["error"] is None

    @patch.object(ModelCache, "get_model")
    def test_progress_updates_recorded(self, mock_get_model, tmp_path):
        """Progress is updated through loading → generating → saving phases."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(tmp_path / "out.png"))

        run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        # Final progress should be saving (the last phase before completion)
        assert updated["progress"] == {"phase": "saving"}

    @patch.object(ModelCache, "get_model")
    def test_output_file_exists_and_is_png(self, mock_get_model, tmp_path):
        """The output image is written and is a valid PNG."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        out_path = tmp_path / "generated.png"
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(out_path))

        run_job(job["job_id"], str(db_path))

        assert out_path.exists()
        with open(out_path, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    @patch.object(ModelCache, "get_model")
    def test_generate_image_called_with_correct_kwargs(self, mock_get_model, tmp_path):
        """The mock model's generate_image receives the right arguments."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(
            queue,
            output_path=str(tmp_path / "out.png"),
            prompt="A blue cat",
            width=512,
            height=768,
            steps=8,
            seed=123,
        )

        run_job(job["job_id"], str(db_path))

        mock_model.generate_image.assert_called_once_with(
            seed=123,
            prompt="A blue cat",
            num_inference_steps=8,
            width=512,
            height=768,
        )

    @patch.object(ModelCache, "get_model")
    def test_auto_generates_seed_when_none(self, mock_get_model, tmp_path):
        """When seed is None, a random seed is generated."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(
            queue, output_path=str(tmp_path / "out.png"), seed=None
        )

        run_job(job["job_id"], str(db_path))

        call_kwargs = mock_model.generate_image.call_args[1]
        seed = call_kwargs["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    @patch.object(ModelCache, "get_model")
    def test_creates_parent_directories(self, mock_get_model, tmp_path):
        """Parent directories for output_path are created automatically."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        out_path = tmp_path / "nested" / "deep" / "out.png"
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(out_path))

        run_job(job["job_id"], str(db_path))

        assert out_path.exists()

    @patch.object(ModelCache, "get_model")
    def test_lora_style_passed_to_get_model(self, mock_get_model, tmp_path):
        """lora_style from params is forwarded to cache.get_model."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(
            queue,
            output_path=str(tmp_path / "out.png"),
            lora_style="portrait",
        )

        run_job(job["job_id"], str(db_path))

        mock_get_model.assert_called_once_with(
            "flux2-klein-4b", quantize=8, lora_style="portrait"
        )


class TestSubprocessRunnerEdit:
    """Tests for happy-path edit_image via subprocess runner."""

    def _submit_edit_job(self, queue, **overrides):
        """Submit an edit_image job with default params overridden."""
        params = {
            "prompt": "Make it blue",
            "model": "flux2-klein-edit",
            "image_paths": ["input.jpg"],
            "steps": 4,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        return queue.submit(
            command="edit_image",
            params=params,
            output_path=output_path,
            backend="subprocess",
        )

    @patch.object(ModelCache, "get_model")
    def test_happy_path_edit_completes(self, mock_get_model, tmp_path):
        """A valid edit job runs to completion."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_edit_job(queue, output_path=str(tmp_path / "out.png"))

        run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "completed"

    @patch.object(ModelCache, "get_model")
    def test_non_fibo_edit_uses_image_paths_list(self, mock_get_model, tmp_path):
        """Flux2KleinEdit receives image_paths (plural list)."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_edit_job(
            queue,
            model="flux2-klein-edit",
            image_paths=["person.jpg", "glasses.jpg"],
            output_path=str(tmp_path / "out.png"),
            steps=6,
            seed=99,
        )

        run_job(job["job_id"], str(db_path))

        mock_model.generate_image.assert_called_once_with(
            seed=99,
            prompt="Make it blue",
            image_paths=["person.jpg", "glasses.jpg"],
            num_inference_steps=6,
        )

    @patch.object(ModelCache, "get_model")
    def test_fibo_edit_uses_singular_image_path(self, mock_get_model, tmp_path):
        """FIBOEdit receives image_path (singular), not image_paths."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_edit_job(
            queue,
            model="fibo-edit",
            image_paths=["photo.jpg", "extra.jpg"],
            output_path=str(tmp_path / "out.png"),
            steps=10,
            seed=77,
        )

        run_job(job["job_id"], str(db_path))

        mock_model.generate_image.assert_called_once_with(
            seed=77,
            prompt="Make it blue",
            image_path="photo.jpg",
            num_inference_steps=10,
        )

    @patch.object(ModelCache, "get_model")
    def test_fibo_edit_rmbg_uses_singular_image_path(self, mock_get_model, tmp_path):
        """fibo-edit-rmbg also uses singular image_path (same FIBOEdit class)."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_edit_job(
            queue,
            model="fibo-edit-rmbg",
            image_paths=["photo.jpg"],
            output_path=str(tmp_path / "out.png"),
            steps=5,
            seed=7,
        )

        run_job(job["job_id"], str(db_path))

        mock_model.generate_image.assert_called_once_with(
            seed=7,
            prompt="Make it blue",
            image_path="photo.jpg",
            num_inference_steps=5,
        )

    @patch.object(ModelCache, "get_model")
    def test_edit_progress_updates(self, mock_get_model, tmp_path):
        """Edit job updates progress through loading → editing → saving."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_edit_job(queue, output_path=str(tmp_path / "out.png"))

        run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["progress"] == {"phase": "saving"}


class TestSubprocessRunnerErrors:
    """Tests for error handling in the subprocess runner."""

    def _submit_generate_job(self, queue, **overrides):
        params = {
            "prompt": "A red square",
            "model": "flux2-klein-4b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
            "quantize": 8,
        }
        params.update(overrides)
        output_path = params.pop("output_path", "/tmp/test_output.png")
        return queue.submit(
            command="generate_image",
            params=params,
            output_path=output_path,
            backend="subprocess",
        )

    def test_missing_job_raises_value_error(self, tmp_path):
        """A non-existent job_id raises ValueError."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        # Don't submit any job

        with pytest.raises(ValueError, match="not found"):
            run_job("nonexistent-id", str(db_path))

    def test_job_not_queued_raises_value_error(self, tmp_path):
        """A job that is not in 'queued' status raises ValueError."""
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(tmp_path / "out.png"))
        # Manually set status to running
        queue.update_status(job["job_id"], "running")

        with pytest.raises(ValueError, match="not queued"):
            run_job(job["job_id"], str(db_path))

    @patch.object(ModelCache, "get_model")
    def test_model_load_failure_marks_failed(self, mock_get_model, tmp_path):
        """When model loading fails, status is 'failed' and error is recorded."""
        mock_get_model.side_effect = RuntimeError("Failed to load model")

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(tmp_path / "out.png"))

        with pytest.raises(SystemExit):
            run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "Failed to load model" in updated["error"]
        assert updated["completed_at"] is not None

    @patch.object(ModelCache, "get_model")
    def test_inference_error_marks_failed(self, mock_get_model, tmp_path):
        """When inference fails, status is 'failed' and error is recorded."""
        mock_model = MagicMock()
        mock_model.generate_image.side_effect = RuntimeError("MLX error")
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(queue, output_path=str(tmp_path / "out.png"))

        with pytest.raises(SystemExit):
            run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "MLX error" in updated["error"]
        assert updated["completed_at"] is not None

    @patch.object(ModelCache, "get_model")
    def test_invalid_command_marks_failed(self, mock_get_model, tmp_path):
        """An unknown command marks the job as failed."""
        mock_model, _ = _MockHelpers.make_mock_model()
        mock_get_model.return_value = mock_model

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = queue.submit(
            command="unknown_command",
            params={"prompt": "test"},
            output_path=str(tmp_path / "out.png"),
            backend="subprocess",
        )

        with pytest.raises(SystemExit):
            run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "Unknown command" in updated["error"]

    @patch.object(ModelCache, "get_model")
    def test_gated_repo_error_user_friendly_message(self, mock_get_model, tmp_path):
        """GatedRepoError produces a user-friendly error message."""
        from huggingface_hub.errors import GatedRepoError

        mock_get_model.side_effect = GatedRepoError("test", response=MagicMock())

        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path)
        job = self._submit_generate_job(
            queue, model="fibo-edit", output_path=str(tmp_path / "out.png")
        )

        with pytest.raises(SystemExit):
            run_job(job["job_id"], str(db_path))

        updated = queue.get_job(job["job_id"])
        assert updated["status"] == "failed"
        assert "huggingface-cli login" in updated["error"]
        assert "HF_TOKEN" in updated["error"]


class TestSubprocessRunnerCLI:
    """Tests for the subprocess runner CLI entry point."""

    @patch("subprocess_runner.run_job")
    def test_correct_args_invoke_run_job(self, mock_run_job, tmp_path):
        """Passing exactly two args invokes run_job."""
        db_path = tmp_path / "test.db"
        with patch.object(
            sys, "argv", ["subprocess_runner.py", "job-123", str(db_path)]
        ):
            subprocess_runner.main()

        mock_run_job.assert_called_once_with("job-123", str(db_path))

    def test_missing_args_exits_with_code_2(self, tmp_path):
        """Missing arguments cause exit code 2."""
        with patch.object(sys, "argv", ["subprocess_runner.py"]):
            with pytest.raises(SystemExit) as exc_info:
                subprocess_runner.main()

            assert exc_info.value.code == 2

    def test_too_many_args_exits_with_code_2(self, tmp_path):
        """Too many arguments cause exit code 2."""
        db_path = tmp_path / "test.db"
        with patch.object(
            sys, "argv", ["subprocess_runner.py", "job-123", str(db_path), "extra"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                subprocess_runner.main()

            assert exc_info.value.code == 2
