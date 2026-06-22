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

FORCE_LOAD_SKIP_CLASS_NAMES: tuple[str, ...] = (
    "StableCascade_C",
    "StableCascade_B",
    "SDXL",
    "SDXLRefiner",
    "Flux",
    "QwenImage",
    "Lumina2",
)
"""The comfy ``model_base`` class names hordelib must keep in lockstep with ComfyUI.

This is the single source of truth for the force-load skip policy: the default
:data:`models_not_to_force_load` is derived from it, and :data:`BASELINE_FORCE_LOAD_CLASS_NAMES`
may only reference names listed here. ComfyUI has no notion of horde baselines, so this mapping
cannot be derived from comfy and must be hand-maintained — :func:`assert_force_load_class_names_exist`
fails fast at startup if any name here has drifted from ``comfy.model_base`` (a silent miss would
let an oversized model be force-loaded and OOM/segfault).
"""

models_not_to_force_load: list = list(FORCE_LOAD_SKIP_CLASS_NAMES)
"""comfy ``model_base`` class names whose models must NOT be force-loaded onto the GPU (default).

Entries are matched by **class identity** (``isinstance`` against the resolved
``comfy.model_base`` class), not by substring, so naming quirks — e.g. the comfy class
``QwenImage`` versus the baseline value ``qwen_image`` — cannot silently break the match the
way the old ``str(type(model)).lower()`` substring test did. Consumers may override this via
``initialise(models_not_to_force_load=...)``; any entry that does not resolve to a real
``comfy.model_base`` class falls back to a lowercase substring match for backwards compatibility
with older raw-string configs.
"""

disable_force_loading: bool = False

FORCE_LOAD_VRAM_SAFETY_FRACTION: float = 0.95
"""Force a full GPU load only when the model(s) fit within this fraction of free VRAM.

Defense-in-depth that does not depend on ``models_not_to_force_load``: a model larger than the
available VRAM (e.g. a 20GB unet on a 16GB GPU) is never force-loaded, because ComfyUI's
force_full_load path crashes (segfaults) rather than falling back to offloading when the
weights cannot fit.
"""

BASELINE_FORCE_LOAD_CLASS_NAMES: dict = {}
"""``KNOWN_IMAGE_GENERATION_BASELINE`` -> comfy ``model_base`` class names for the skip list.

hordelib owns the knowledge of which comfy class implements which horde baseline; consumers
(the worker) express force-load policy in baseline enum members and never ship raw class names.
Populated lazily to keep this module importable without horde_model_reference.
"""


def _baseline_class_names() -> dict:
    if not BASELINE_FORCE_LOAD_CLASS_NAMES:
        from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

        BASELINE_FORCE_LOAD_CLASS_NAMES.update(
            {
                KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade: ("StableCascade_C", "StableCascade_B"),
                KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl: ("SDXL", "SDXLRefiner"),
                KNOWN_IMAGE_GENERATION_BASELINE.flux_1: ("Flux",),
                KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell: ("Flux",),
                KNOWN_IMAGE_GENERATION_BASELINE.flux_dev: ("Flux",),
                KNOWN_IMAGE_GENERATION_BASELINE.qwen_image: ("QwenImage",),
                # Z-Image (incl. Z-Image-Turbo) loads as comfy's Lumina2 model_base class.
                KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo: ("Lumina2",),
            },
        )
    return BASELINE_FORCE_LOAD_CLASS_NAMES


def resolve_force_load_skip_entries(entries: list) -> list[str]:
    """Translate a mixed list of baseline enum members and raw strings to comfy class names.

    Raw strings pass through unchanged (back-compat: a comfy class name or a legacy lowercase
    fragment); baselines are mapped to their comfy ``model_base`` class names. Baselines with
    no known comfy class are skipped with a warning.
    """
    names: list[str] = []
    mapping = _baseline_class_names()
    for entry in entries:
        if isinstance(entry, str) and entry not in mapping:
            if entry not in names:
                names.append(entry)
            continue
        class_names = mapping.get(entry)
        if class_names is None:
            logger.warning("No comfy class known for baseline; ignoring: baseline={}", entry)
            continue
        for class_name in class_names:
            if class_name not in names:
                names.append(class_name)
    return names


def assert_force_load_class_names_exist() -> None:
    """Fail fast when a comfy class name hordelib hard-codes has drifted from ComfyUI.

    The force-load skip policy names the ``comfy.model_base`` classes hordelib refuses to
    force-load. Because ComfyUI has no notion of horde baselines this list can't be derived and
    must be hand-maintained, so it can silently rot when ComfyUI renames/removes a class — and a
    silent miss is dangerous (the model gets force-loaded and OOM/segfaults). This is the tripwire:
    it raises a clear, actionable error if any referenced name is missing, and also verifies the
    two declarations (:data:`FORCE_LOAD_SKIP_CLASS_NAMES` and :data:`BASELINE_FORCE_LOAD_CLASS_NAMES`)
    agree with each other. Called once when the load_models_gpu hijack is installed.
    """
    import comfy.model_base

    baseline_names = {name for class_names in _baseline_class_names().values() for name in class_names}

    # Internal consistency: the baseline map and the flat default must reference the same classes,
    # so neither can be updated without the other.
    only_in_baseline = baseline_names - set(FORCE_LOAD_SKIP_CLASS_NAMES)
    only_in_flat = set(FORCE_LOAD_SKIP_CLASS_NAMES) - baseline_names
    if only_in_baseline or only_in_flat:
        raise RuntimeError(
            "hordelib force-load class-name declarations disagree: "
            f"only in BASELINE_FORCE_LOAD_CLASS_NAMES={sorted(only_in_baseline)}, "
            f"only in FORCE_LOAD_SKIP_CLASS_NAMES={sorted(only_in_flat)}. "
            "Keep both in hordelib/execution/comfy_patches.py in sync.",
        )

    referenced = set(FORCE_LOAD_SKIP_CLASS_NAMES) | baseline_names
    missing = sorted(name for name in referenced if not isinstance(getattr(comfy.model_base, name, None), type))
    if missing:
        available = sorted(
            name for name in dir(comfy.model_base) if isinstance(getattr(comfy.model_base, name, None), type)
        )
        raise RuntimeError(
            f"hordelib's force-load skip list references comfy.model_base classes that no longer exist: {missing}. "
            "ComfyUI's model classes have changed; update FORCE_LOAD_SKIP_CLASS_NAMES / "
            f"BASELINE_FORCE_LOAD_CLASS_NAMES in hordelib/execution/comfy_patches.py. "
            f"Available comfy.model_base classes: {available}",
        )


def capture_and_patch(name: str, module: typing.Any, attr: str, hijack: Callable) -> None:
    """Capture the original ``module.attr`` under ``name`` and install the hijack."""
    _originals[name] = getattr(module, attr)
    setattr(module, attr, hijack)
    logger.debug("Applied ComfyUI monkeypatch: name={}, target={}.{}", name, module, attr)


def register_execution_module(execution_module: typing.Any) -> None:
    """Record ComfyUI's execution module so the registry can address IsChangedCache."""
    global _comfy_execution_module
    _comfy_execution_module = execution_module


def _resolve_skip_classes_and_fragments() -> tuple[tuple[type, ...], list[str]]:
    """Resolve ``models_not_to_force_load`` entries into comfy classes + legacy fragments.

    Entries naming a real ``comfy.model_base`` class are matched by identity; any that do not
    resolve are kept as lowercase substring fragments for backwards compatibility with older
    raw-string configs.
    """
    import comfy.model_base

    classes: list[type] = []
    fragments: list[str] = []
    for entry in models_not_to_force_load:
        candidate = getattr(comfy.model_base, entry, None) if isinstance(entry, str) else None
        if isinstance(candidate, type):
            classes.append(candidate)
        elif isinstance(entry, str):
            fragments.append(entry.lower())
    return tuple(classes), fragments


def _do_not_force_load_model_in_patcher(model_patcher) -> bool:
    import comfy.model_base

    # Only the big diffusion (BaseModel) models are ever skipped; VAEs/CLIP/controlnets must
    # always be force-loaded. Matching is by class identity so e.g. the cascade stage-C coder
    # (comfy.ldm.cascade.stage_c_coder, not a StableCascade BaseModel) can't be mistaken for the
    # cascade diffusion model and left partially on CPU (which crashes encode on a device mismatch).
    model = model_patcher.model
    if not isinstance(model, comfy.model_base.BaseModel):
        return False

    skip_classes, legacy_fragments = _resolve_skip_classes_and_fragments()
    if skip_classes and isinstance(model, skip_classes):
        return True
    if legacy_fragments:
        class_name_lower = type(model).__name__.lower()
        if any(fragment in class_name_lower for fragment in legacy_fragments):
            return True
    return False


def _force_full_load_would_overflow_vram(models) -> bool:
    """Return True when force-fully loading *models* would not fit in currently-free VRAM.

    ComfyUI's ``force_full_load`` path crashes (segfault) when asked to load weights larger than
    the free VRAM instead of erroring or offloading, so we preempt that here. Fails open
    (returns False) when the device isn't CUDA or sizes can't be determined, preserving the
    force-load default.
    """
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        if getattr(device, "type", None) != "cuda":
            return False
        free_memory = model_management.get_free_memory(device)
        required = sum(model_patcher.model_size() for model_patcher in models)
    except Exception as exc:
        logger.debug("Could not estimate force-load VRAM fit; proceeding to force load: error={}", exc)
        return False

    overflow = required > free_memory * FORCE_LOAD_VRAM_SAFETY_FRACTION
    if overflow:
        logger.info(
            "Skipping force-full-load; model would not fit in free VRAM: required_mb={:.0f}, free_mb={:.0f}",
            required / 2**20,
            free_memory / 2**20,
        )
    return overflow


@logfire.instrument("comfy.load_models_gpu_hijack")
def _load_models_gpu_hijack(*args, **kwargs):
    """Intercepts the comfy load_models_gpu function to force full load.

    ComfyUI is too conservative in its loading to GPU for the worker/horde use case where we can have
    multiple ComfyUI instances running on the same GPU. This function forces a full load of the model
    and the worker/horde-engine takes responsibility for managing the memory or the problems this may
    cause.
    """

    models = args[0]
    skip_force_load = any(_do_not_force_load_model_in_patcher(model_patcher) for model_patcher in models)
    if not skip_force_load and _force_full_load_would_overflow_vram(models):
        skip_force_load = True

    if skip_force_load:
        logger.debug("Not overriding model load")
        logger.info("comfy.model_load_skipped", model_count=len(models))
        kwargs["memory_required"] = 1e30
        kwargs.pop("force_full_load", None)
        _originals["load_models_gpu"](*args, **kwargs)
        return

    kwargs.pop("force_full_load", None)
    kwargs["force_full_load"] = True
    logger.info("comfy.force_full_load", model_count=len(models))
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
