"""Subprocess runner for mflux-mcp inference jobs.

Standalone script that executes a single inference job in its own process.
Receives a job_id and db_path, opens the database, reads job params,
loads the model, runs inference, writes the output image, and updates
job status throughout.

Usage:
    python subprocess_runner.py <job_id> <db_path>
"""

import io
import os
import random
import sys

from huggingface_hub.errors import GatedRepoError

from job_queue import JobQueue
from mflux_cache import ModelCache, _REPO_MAP


def run_job(job_id: str, db_path: str) -> None:
    """Execute a single inference job.

    Args:
        job_id: UUID of the job to run.
        db_path: Path to the SQLite job database.

    Raises:
        ValueError: If the job is not found or not in 'queued' status.
    """
    queue = JobQueue(db_path)
    job = queue.get_job(job_id)

    # Validate
    if job is None:
        raise ValueError(f"Job {job_id} not found")
    if job["status"] != "queued":
        raise ValueError(f"Job {job_id} is not queued (status={job['status']})")

    # Mark running
    queue.update_status(
        job_id,
        "running",
        started_at=queue._now_iso(),
        pid=os.getpid(),
    )

    try:
        params = job["params"]  # already deserialized by JobQueue
        command = job["command"]
        output_path = job["output_path"]

        # Phase: loading model
        queue.update_progress(job_id, {"phase": "loading_model"})
        cache = ModelCache()
        model_name = params.get("model", "flux2-klein-4b")
        quantize = params.get("quantize", 8)
        lora_style = params.get("lora_style")

        try:
            loaded_model = cache.get_model(
                model_name, quantize=quantize, lora_style=lora_style
            )
        except GatedRepoError:
            _, config_factory_name, _ = ModelCache._REGISTRY[model_name]
            repo_id = _REPO_MAP.get(config_factory_name, model_name)
            raise RuntimeError(
                f"Model '{model_name}' requires access to a gated "
                f"HuggingFace repository.\n\n"
                f"To resolve this:\n"
                f"1. Visit https://huggingface.co/{repo_id} and request access\n"
                f"2. Authenticate locally by running: huggingface-cli login\n"
                f"   Or set the HF_TOKEN environment variable with your "
                f"access token\n"
                f"3. Retry the operation"
            )

        # Handle seed
        seed = params.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Phase: inference
        if command == "generate_image":
            queue.update_progress(job_id, {"phase": "generating"})
            result = loaded_model.generate_image(
                seed=seed,
                prompt=params["prompt"],
                num_inference_steps=params.get("steps", 4),
                width=params.get("width", 1024),
                height=params.get("height", 1024),
            )
        elif command == "edit_image":
            queue.update_progress(job_id, {"phase": "editing"})
            class_key = ModelCache._REGISTRY[model_name][0]
            if class_key == "FIBOEdit":
                inference_kwargs = dict(
                    seed=seed,
                    prompt=params["prompt"],
                    image_path=params["image_paths"][0],
                    num_inference_steps=params.get("steps", 4),
                )
            else:
                inference_kwargs = dict(
                    seed=seed,
                    prompt=params["prompt"],
                    image_paths=params["image_paths"],
                    num_inference_steps=params.get("steps", 4),
                )
            result = loaded_model.generate_image(**inference_kwargs)
        else:
            raise ValueError(f"Unknown command: {command}")

        # Phase: saving
        queue.update_progress(job_id, {"phase": "saving"})
        abs_path = os.path.abspath(output_path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        with open(abs_path, "wb") as f:
            f.write(buf.getvalue())

        # Mark completed
        queue.update_status(job_id, "completed", completed_at=queue._now_iso())

    except Exception as e:
        # Catch ALL exceptions and write to job row
        queue.update_status(
            job_id,
            "failed",
            error=str(e),
            completed_at=queue._now_iso(),
        )
        sys.exit(1)


def main() -> None:
    """CLI entry point for the subprocess runner."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <job_id> <db_path>", file=sys.stderr)
        sys.exit(2)
    run_job(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
