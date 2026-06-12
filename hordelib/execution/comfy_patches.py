"""hordelib's ComfyUI monkeypatches.

These hijacks override ComfyUI's model-management behavior so the AI Horde worker can run
multiple aggressive instances on one GPU. They are **order-sensitive**: they are captured and
applied at specific points during ``hordelib.comfy_horde.do_comfy_import()`` — do not apply
them at import time, and do not reorder the capture calls there.

ComfyUI is only imported inside functions here (after ``do_comfy_import`` has run); the module
itself is importable at any time.
"""

import contextlib
import hashlib
import json
import typing
from collections.abc import Callable

import logfire
import PIL.Image
import torch
from loguru import logger

_originals: dict[str, Callable] = {}
"""The original ComfyUI callables, keyed by monkeypatch name. Populated by capture_and_patch()."""

_comfy_execution_module: typing.Any = None
"""ComfyUI's ``execution`` module, registered during do_comfy_import()."""

models_not_to_force_load: list = [
    "cascade",
    "sdxl",
    "flux",
    "qwen_image",
]  # other possible values could be `basemodel` or `sd1`
"""Models which should not be forced to load in the comfy model loading hijack.

Possible values include `cascade`, `sdxl`, `basemodel`, `sd1` or any other comfyui classname
which can be passed to comfyui's `load_models_gpu` function (as a `ModelPatcher.model`).
"""

disable_force_loading: bool = False


def capture_and_patch(name: str, module: typing.Any, attr: str, hijack: Callable) -> None:
    """Capture the original ``module.attr`` under ``name`` and install the hijack."""
    _originals[name] = getattr(module, attr)
    setattr(module, attr, hijack)
    logger.debug("Applied ComfyUI monkeypatch: name={}, target={}.{}", name, module, attr)


def register_execution_module(execution_module: typing.Any) -> None:
    """Record ComfyUI's execution module so the registry can address IsChangedCache."""
    global _comfy_execution_module
    _comfy_execution_module = execution_module


def _do_not_force_load_model_in_patcher(model_patcher) -> bool:
    import comfy.model_base

    # Only the big diffusion models are ever skipped. VAEs/CLIP/controlnets must always be
    # force-loaded: the name match below is against the full class path, and e.g. the cascade
    # stage-C VAE (comfy.ldm.cascade.stage_c_coder) would otherwise match "cascade" and be left
    # partially on CPU, crashing encode with a CUDA/CPU device mismatch.
    if not isinstance(model_patcher.model, comfy.model_base.BaseModel):
        return False

    model_name_lower = str(type(model_patcher.model)).lower()
    if "clip" in model_name_lower:
        return False

    for model in models_not_to_force_load:
        if model in model_name_lower:
            return True

    return False


@logfire.instrument("comfy.load_models_gpu_hijack")
def _load_models_gpu_hijack(*args, **kwargs):
    """Intercepts the comfy load_models_gpu function to force full load.

    ComfyUI is too conservative in its loading to GPU for the worker/horde use case where we can have
    multiple ComfyUI instances running on the same GPU. This function forces a full load of the model
    and the worker/horde-engine takes responsibility for managing the memory or the problems this may
    cause.
    """

    found_model_to_skip = False
    for model_patcher in args[0]:
        found_model_to_skip = _do_not_force_load_model_in_patcher(model_patcher)
        if found_model_to_skip:
            break

    if found_model_to_skip:
        logger.debug("Not overriding model load")
        logger.info("comfy.model_load_skipped", model_count=len(args[0]))
        kwargs["memory_required"] = 1e30
        _originals["load_models_gpu"](*args, **kwargs)
        return

    if "force_full_load" in kwargs:
        kwargs.pop("force_full_load")

    kwargs["force_full_load"] = True
    logger.info("comfy.force_full_load", model_count=len(args[0]))
    _originals["load_models_gpu"](*args, **kwargs)


def _model_patcher_load_hijack(*args, **kwargs):
    """Intercepts the comfy ModelPatcher.load function to force full load.

    See _load_models_gpu_hijack for more information
    """
    model_patcher = args[0]
    if _do_not_force_load_model_in_patcher(model_patcher):
        logger.debug("Not overriding model load")
        _originals["model_patcher_load"](*args, **kwargs)
        return

    if "full_load" in kwargs:
        kwargs.pop("full_load")

    kwargs["full_load"] = True
    _originals["model_patcher_load"](*args, **kwargs)


def _calculate_weight_hijack(*args, **kwargs):
    patches = args[0]

    for p in patches:
        v = p[1]
        if not isinstance(v, tuple):
            continue
        patch_type = v[0]
        if patch_type != "diff":
            continue
        if len(v) == 2 and isinstance(v[1], list):
            for idx, val in enumerate(v[1]):
                if val is None:
                    v[1][idx] = {"pad_weight": False}
                    break

    return _originals["lora_calculate_weight"](*args, **kwargs)


_last_pipeline_settings_hash = ""


def default_json_serializer_pil_image(obj):
    if isinstance(obj, PIL.Image.Image):
        return str(hash(obj.__str__()))
    return obj


async def IsChangedCache_get_hijack(self, *args, **kwargs):
    result = await _originals["is_changed_cache_get"](self, *args, **kwargs)

    global _last_pipeline_settings_hash

    prompt = self.dynprompt.original_prompt

    pipeline_settings_hash = hashlib.md5(
        json.dumps(prompt, default=default_json_serializer_pil_image).encode(),
    ).hexdigest()

    if pipeline_settings_hash != _last_pipeline_settings_hash:
        _last_pipeline_settings_hash = pipeline_settings_hash
        logger.debug("Pipeline settings changed: hash={}", pipeline_settings_hash)
        logger.debug("Outputs cache size: count={}", len(self.outputs_cache.cache))
        logger.debug("Outputs subcache size: count={}", len(self.outputs_cache.subcaches))

        logger.debug("IsChangedCache dynprompt node IDs: ids={}", self.dynprompt.all_node_ids())

    if result:
        logger.debug("IsChangedCache.get: result={}", result)

    return result


def text_encoder_initial_device_hijack(*args, **kwargs):
    # This ensures clip models are loaded on the CPU first
    return torch.device("cpu")


class _MonkeyPatchBinding(typing.NamedTuple):
    module: typing.Any
    attr: str
    patched: typing.Callable | None
    original: typing.Callable | None


def _ensure_comfy_initialised() -> None:
    if "load_models_gpu" not in _originals:
        raise RuntimeError("ComfyUI is not initialised. Call hordelib.initialise() before using monkeypatch helpers.")


def _build_monkeypatch_registry() -> dict[str, _MonkeyPatchBinding]:
    _ensure_comfy_initialised()

    import comfy.lora
    import comfy.model_management as model_management
    from comfy.model_patcher import ModelPatcher

    bindings: dict[str, _MonkeyPatchBinding] = {
        "load_models_gpu": _MonkeyPatchBinding(
            model_management,
            "load_models_gpu",
            _load_models_gpu_hijack,
            _originals.get("load_models_gpu"),
        ),
        "model_patcher_load": _MonkeyPatchBinding(
            ModelPatcher,
            "load",
            _model_patcher_load_hijack,
            _originals.get("model_patcher_load"),
        ),
        "lora_calculate_weight": _MonkeyPatchBinding(
            comfy.lora,
            "calculate_weight",
            _calculate_weight_hijack,
            _originals.get("lora_calculate_weight"),
        ),
        "text_encoder_initial_device": _MonkeyPatchBinding(
            model_management,
            "text_encoder_initial_device",
            text_encoder_initial_device_hijack,
            _originals.get("text_encoder_initial_device"),
        ),
    }

    if _comfy_execution_module is not None and _originals.get("is_changed_cache_get") is not None:
        bindings["is_changed_cache_get"] = _MonkeyPatchBinding(
            _comfy_execution_module.IsChangedCache,
            "get",
            IsChangedCache_get_hijack,
            _originals.get("is_changed_cache_get"),
        )

    return {
        name: binding
        for name, binding in bindings.items()
        if binding.original is not None and binding.patched is not None
    }


def get_monkeypatch_names() -> list[str]:
    """Return the ordered list of known ComfyUI monkeypatch identifiers."""
    return list(_build_monkeypatch_registry().keys())


def get_monkeypatch_state() -> dict[str, bool]:
    """Return which monkeypatches are currently active."""
    registry = _build_monkeypatch_registry()
    state: dict[str, bool] = {}
    for name, binding in registry.items():
        state[name] = getattr(binding.module, binding.attr) is binding.patched
    return state


def set_monkeypatch_state(
    *,
    enable: bool,
    patch_names: typing.Iterable[str] | None = None,
) -> None:
    """Force a set of monkeypatches to be enabled or disabled."""
    registry = _build_monkeypatch_registry()
    target_names = list(patch_names) if patch_names is not None else list(registry.keys())
    for name in target_names:
        if name not in registry:
            raise KeyError(f"Unknown monkeypatch '{name}'")
        binding = registry[name]
        setattr(binding.module, binding.attr, binding.patched if enable else binding.original)


@contextlib.contextmanager
def temporary_monkeypatch_state(
    *,
    enable: bool,
    patch_names: typing.Iterable[str] | None = None,
):
    """Context manager to temporarily enable or disable specific monkeypatches."""
    registry = _build_monkeypatch_registry()
    target_names = list(patch_names) if patch_names is not None else list(registry.keys())

    previous: dict[str, typing.Callable | None] = {}
    for name in target_names:
        if name not in registry:
            raise KeyError(f"Unknown monkeypatch '{name}'")
        binding = registry[name]
        previous[name] = getattr(binding.module, binding.attr)

    try:
        set_monkeypatch_state(enable=enable, patch_names=target_names)
        yield
    finally:
        for name in target_names:
            binding = registry[name]
            setattr(binding.module, binding.attr, previous[name])
