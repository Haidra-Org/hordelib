"""Executable single-stage runner for the subprocess-per-stage disaggregation reproduction.

Each invocation initializes hordelib fresh in its own process, rebuilds the SDK parameters from a
JSON file, runs exactly ONE disaggregated stage entry point on :class:`~hordelib.horde.HordeLib`
(``encode_text_stage`` / ``vae_encode_stage`` / ``sample_stage`` / ``decode_stage``), and writes the
resulting blob(s) to the paths named in the job spec. This is the per-process analog of the in-process
stage pipeline: no prior ``generate`` has run in the process, so every disk load is genuinely cold and
ComfyUI global state is freshly minted.

The job spec is a JSON file whose path is the sole argv. Shape::

    {
        "stage": "encode" | "vae_encode" | "sample" | "decode",
        "params_json_path": "<path to ImageGenerationParameters.model_dump_json output>",
        "inputs": {"positive": "<path>", "negative": "<path>", "latent": "<path>"},
        "outputs": {"positive": "<path>", "negative": "<path>", "latent": "<path>", "image": "<path>"}
    }

Only the input/output keys a given stage consumes or produces need be present. When the environment
variable ``HORDELIB_DISABLE_ZERO_COPY`` is set to a truthy value the runner replaces the zero-copy
state-dict adoption context manager with a null context before any load, so the loader falls back to
the ordinary copying ``load_state_dict``; this isolates the cross-process mmap-adoption path as a
suspect. ``HORDELIB_STAGE_AGGRESSIVE_UNLOADING`` (default ``1``) sets the HordeLib ``aggressive_unloading``
flag for the fidelity variant.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _disable_zero_copy_if_requested() -> None:
    """Replace the zero-copy adoption context manager with a null context, forcing copying loads.

    The loader imports ``zero_copy_state_dict_assignment`` from the module at call time, so replacing
    the module attribute here (before any stage load) makes the loader use the ordinary copying
    ``load_state_dict`` path instead of adopting mmap-backed tensors.
    """
    import hordelib.execution.zero_copy_load as zero_copy_load

    @contextlib.contextmanager
    def _null_assignment():  # type: ignore[no-untyped-def]
        yield

    zero_copy_load.zero_copy_state_dict_assignment = _null_assignment  # type: ignore[assignment]


def _initialise_hordelib() -> None:
    """Initialise hordelib and load model managers, mirroring the GPU-test session fixtures."""
    import hordelib

    hordelib.initialise(
        setup_logging=True,
        logging_verbosity=5,
        disable_smart_memory=True,
        force_normal_vram_mode=True,
        do_not_load_model_mangers=True,
        extra_comfyui_args=["--reserve-vram", "1.4"],
    )

    from hordelib.model_manager.hyper import ALL_MODEL_MANAGER_TYPES
    from hordelib.settings import UserSettings
    from hordelib.shared_model_manager import SharedModelManager

    UserSettings.set_ram_to_leave_free_mb("100%")
    UserSettings.set_vram_to_leave_free_mb("90%")

    SharedModelManager(do_not_load_model_mangers=True)
    SharedModelManager.load_model_managers(ALL_MODEL_MANAGER_TYPES)


def _ensure_checkpoint_available(model_name: str) -> None:
    """Make sure the checkpoint the stage needs is present on disk and validated."""
    from hordelib.shared_model_manager import SharedModelManager

    compvis = SharedModelManager.manager.compvis
    assert compvis is not None
    if model_name in compvis.model_reference and model_name not in compvis.available_models:
        assert compvis.download_model(model_name)
        assert compvis.validate_model(model_name)


def _load_params(params_json_path: str) -> Any:
    from horde_sdk.generation_parameters.image import ImageGenerationParameters

    return ImageGenerationParameters.model_validate_json(Path(params_json_path).read_text(encoding="utf-8"))


def _run_stage(spec: dict[str, Any]) -> None:
    from hordelib.horde import HordeLib

    stage = spec["stage"]
    params = _load_params(spec["params_json_path"])
    inputs: dict[str, str] = spec.get("inputs", {})
    outputs: dict[str, str] = spec.get("outputs", {})

    _ensure_checkpoint_available(params.base_params.model)

    aggressive_unloading = _truthy(os.getenv("HORDELIB_STAGE_AGGRESSIVE_UNLOADING", "1"))
    hordelib_instance = HordeLib(aggressive_unloading=aggressive_unloading)

    if stage == "encode":
        positive_bytes, negative_bytes = hordelib_instance.encode_text_stage(params)
        Path(outputs["positive"]).write_bytes(positive_bytes)
        Path(outputs["negative"]).write_bytes(negative_bytes)
    elif stage == "vae_encode":
        source_latent_bytes = hordelib_instance.vae_encode_stage(params)
        Path(outputs["latent"]).write_bytes(source_latent_bytes)
    elif stage == "sample":
        source_latent_bytes = Path(inputs["latent"]).read_bytes() if inputs.get("latent") else None
        latent_bytes = hordelib_instance.sample_stage(
            params,
            positive_conditioning_bytes=Path(inputs["positive"]).read_bytes(),
            negative_conditioning_bytes=Path(inputs["negative"]).read_bytes(),
            source_latent_bytes=source_latent_bytes,
        )
        Path(outputs["latent"]).write_bytes(latent_bytes)
    elif stage == "decode":
        results, _faults = hordelib_instance.decode_stage(
            params,
            latent_bytes=Path(inputs["latent"]).read_bytes(),
        )
        assert len(results) == 1, f"decode stage produced {len(results)} image(s), expected 1"
        Path(outputs["image"]).write_bytes(results[0].rawpng.getvalue())
    else:
        raise ValueError(f"Unknown stage: {stage!r}")


def main() -> None:
    os.environ["TESTS_ONGOING"] = "1"

    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <job_spec.json>")

    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    if _truthy(os.getenv("HORDELIB_DISABLE_ZERO_COPY")):
        _disable_zero_copy_if_requested()

    _initialise_hordelib()
    _run_stage(spec)


if __name__ == "__main__":
    main()
