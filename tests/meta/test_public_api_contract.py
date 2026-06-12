"""The hordelib.api consumer contract.

horde-worker-reGen (and any other consumer) imports exclusively from ``hordelib.api``. This
test is the tripwire that makes breaking that surface loud inside hordelib CI: it asserts the
module imports without ``hordelib.initialise()`` (no ComfyUI), that every declared name
exists, and that the signatures consumers call don't drift silently.
"""

import inspect
import subprocess
import sys

_COMFY_FREE_IMPORT_CHECK = """
import sys

import hordelib.api

for name in hordelib.api.__all__:
    assert hasattr(hordelib.api, name), f"hordelib.api.__all__ declares missing name {name!r}"

# Importing the API surface must never pull ComfyUI into the process: the worker's
# safety process imports these helpers and must stay comfy-free.
assert "comfy" not in sys.modules
assert "execution" not in sys.modules
"""


def test_api_imports_without_comfy() -> None:
    # A subprocess gives a clean sys.modules; in-process the check would false-fail
    # whenever an earlier test in the session had already imported ComfyUI.
    result = subprocess.run(
        [sys.executable, "-c", _COMFY_FREE_IMPORT_CHECK],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"hordelib.api comfy-free import check failed:\n{result.stderr}"


def test_consumer_called_signatures() -> None:
    """The exact keyword surfaces the worker calls; changing these is a coordinated release."""
    from hordelib import api

    def params(func) -> set[str]:
        return set(inspect.signature(func).parameters)

    assert {"comfyui_callback", "aggressive_unloading"} <= params(api.HordeLib.__init__)
    assert {"payload", "progress_callback"} <= params(api.HordeLib.basic_inference)
    assert {"horde_model_name", "will_load_loras", "seamless_tiling_enabled"} <= params(api.HordeLib.preload_model)
    assert {"payload"} <= params(api.HordeLib.post_process)

    assert {"managers_to_load", "multiprocessing_lock", "lora_reference_backups"} <= params(
        api.SharedModelManager.load_model_managers.__func__,
    )

    assert {
        "setup_logging",
        "process_id",
        "logging_verbosity",
        "force_normal_vram_mode",
        "models_not_to_force_load",
        "extra_comfyui_args",
    } <= params(api.initialise)

    assert {"setup_logging", "process_id", "verbosity_count"} <= params(api.HordeLog.initialise.__func__)

    # The execution backend protocol the worker uses for memory control
    assert {"free_vram", "free_ram", "vram_stats", "interrupt", "run_pipeline", "start"} <= {
        name for name in dir(api.ExecutionBackend) if not name.startswith("_")
    }


def test_blessed_lora_manager_surface() -> None:
    """Manager methods the worker calls directly; blessed as public in HL-2."""
    from hordelib.model_manager.lora import LoraModelManager

    for method in (
        "load_model_database",
        "reset_adhoc_loras",
        "fetch_adhoc_lora",
        "wait_for_downloads",
        "save_cached_reference_to_disk",
        "is_model_available",
        "download_model",
    ):
        assert callable(getattr(LoraModelManager, method, None)), f"LoraModelManager.{method} missing"
