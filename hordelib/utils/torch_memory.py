"""Memory and accelerator statistics that work with or without ComfyUI loaded.

Consumers (notably horde-worker-reGen's process monitoring, including its safety process,
which must never trigger a ComfyUI import) need VRAM/RAM numbers and a device inventory in
processes where ``hordelib.initialise()`` may not have run. When ComfyUI *is* loaded in this
process, these delegate to its backend-agnostic memory management for exact agreement with
what the execution backend sees; otherwise they fall back to plain torch device queries that
cover every torch backend (CUDA/ROCm, Intel XPU, Apple MPS, CPU), not just CUDA.

This module is the single source of accelerator truth for the worker: it never assumes
NVIDIA. NVML is treated as optional NVIDIA-only enrichment (see :mod:`hordelib.utils.nvml`),
never as a hard requirement here; :func:`get_accelerator_utilization_percent` is the one place
that consults it, and only for the CUDA/NVIDIA backend.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import psutil
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum

_MB = 1024 * 1024

_COMFY_MODEL_MANAGEMENT = "comfy.model_management"


class AcceleratorKind(StrEnum):
    """The compute backend a device is driven through.

    ``cuda`` and ``rocm`` both present through ``torch.cuda`` in PyTorch; they are
    distinguished by ``torch.version.hip``. ``directml`` cannot be enumerated without ComfyUI
    loaded, so the plain-torch fallback never reports it (ComfyUI does).
    """

    cuda = "cuda"
    rocm = "rocm"
    xpu = "xpu"
    npu = "npu"
    mlu = "mlu"
    mps = "mps"
    directml = "directml"
    cpu = "cpu"


class AcceleratorInfo(BaseModel):
    """A single compute device discovered on this machine."""

    index: int
    name: str
    total_vram_mb: int
    kind: AcceleratorKind


def _comfy_model_management() -> ModuleType | None:
    """Return ComfyUI's ``model_management`` module iff it is already imported here.

    Uses ``sys.modules`` rather than a fresh import so that calling this never triggers the
    heavy (and, in the safety process, forbidden) ComfyUI import.
    """
    return sys.modules.get(_COMFY_MODEL_MANAGEMENT)


def _comfy_memory_funcs() -> tuple | None:
    """Return comfy's (get_total_memory, get_free_memory) if comfy is imported here."""
    mm = _comfy_model_management()
    if mm is None:
        return None
    total = getattr(mm, "get_total_memory", None)
    free = getattr(mm, "get_free_memory", None)
    if total is None or free is None:
        return None
    return total, free


def _active_torch_kind() -> AcceleratorKind:
    """Detect the backend the active torch build talks to, without ComfyUI loaded."""
    import torch

    if getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
        return AcceleratorKind.rocm
    if getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available():
        return AcceleratorKind.cuda
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return AcceleratorKind.xpu
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return AcceleratorKind.mps
    return AcceleratorKind.cpu


def torch_build_is_cpu_only() -> bool:
    """Return whether the installed torch wheel has no GPU backend compiled in at all.

    This is deliberately stricter than ``_active_torch_kind() is AcceleratorKind.cpu``: that returns
    ``cpu`` whenever no usable device is *found at runtime* (which also happens for a CUDA/ROCm build
    whose GPU is merely masked or has a broken driver), and it does not know about the NPU/MLU/DirectML
    backends ComfyUI can still drive. The distinction matters because ComfyUI's ``cpu_state`` defaults to
    GPU and only flips to CPU on the ``--cpu`` flag (never from ``torch.cuda.is_available()`` being
    ``False``); telling it to use CPU is correct only when the build genuinely cannot reach any
    accelerator. A CPU-only wheel reports no CUDA/HIP version and exposes no XPU/MPS/NPU/MLU backend, so
    that is what is checked here.

    Returns ``False`` on any probe error, so an unexpected torch shape never forces CPU mode onto a box
    that may actually have a GPU.
    """
    try:
        import torch

        if getattr(torch.version, "cuda", None) is not None:
            return False  # CUDA or ROCm build (ROCm also sets version.cuda); a masked GPU is not CPU-only.
        if getattr(torch.version, "hip", None) is not None:
            return False
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return False
        if getattr(torch, "npu", None) is not None and torch.npu.is_available():
            return False
        if getattr(torch, "mlu", None) is not None and torch.mlu.is_available():
            return False
    except Exception as e:
        logger.debug(f"Could not determine whether torch is a CPU-only build; assuming not: {e}")
        return False
    return True


def _torch_fallback_vram_bytes(*, free: bool) -> int:
    """Total or free device memory of the active torch backend, in bytes (0 on failure).

    Mirrors ComfyUI's per-backend handling for the case where ComfyUI is not loaded: CUDA/ROCm
    and XPU expose ``mem_get_info``; MPS and CPU have no device VRAM, so system RAM is the
    honest answer (matching ComfyUI's ``get_total_memory`` for those backends).
    """
    import torch

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes if free else total_bytes
        if kind is AcceleratorKind.xpu and hasattr(torch.xpu, "mem_get_info"):
            free_bytes, total_bytes = torch.xpu.mem_get_info()
            return free_bytes if free else total_bytes
    except Exception as e:
        logger.debug(f"torch VRAM probe failed for {kind}: {e}")
        return 0

    vmem = psutil.virtual_memory()
    return vmem.available if free else vmem.total


def get_torch_total_vram_mb() -> int:
    """Total VRAM of the active torch device, in MB (system RAM for CPU/MPS backends)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[0]() / _MB)
    return round(_torch_fallback_vram_bytes(free=False) / _MB)


def get_torch_free_vram_mb() -> int:
    """Free VRAM of the active torch device, in MB (available RAM for CPU/MPS backends)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[1]() / _MB)
    return round(_torch_fallback_vram_bytes(free=True) / _MB)


def get_torch_device_free_vram_mb() -> int:
    """Physically free VRAM on the device, in MB, from the driver's device-wide accounting.

    Unlike :func:`get_torch_free_vram_mb` (which, when ComfyUI is loaded, returns comfy's
    ``get_free_memory`` == device-wide free PLUS this process's reserved-but-inactive torch allocator
    cache), this is the raw device-wide free from ``mem_get_info``. The distinction is load-bearing for
    cross-process residency budgeting and for predicting the driver's system-memory fallback: the comfy
    number adds back a single process's reclaimable cache, so it over-states what a *new* allocation can
    use and hides VRAM held by *other* processes (including leaked/orphaned ones). The driver spills to
    host RAM on device-wide free nearing zero, not on the inflated number, so the worker must budget
    against this. Falls back to system RAM on CPU/MPS, matching the other helpers here.
    """
    return round(_torch_fallback_vram_bytes(free=True) / _MB)


def _get_aimdo_usage_mb() -> int:
    """This process's device memory held by the engine's direct-IO weight pool, in MB (0 unless it is initialised).

    ComfyUI's ``comfy_aimdo`` subsystem *can* load model weights through native direct-IO into device memory
    it reserves itself (``vrambuf_create``/``vrambuf_grow``), bypassing the torch caching allocator entirely;
    such memory would be invisible to :func:`torch.cuda.memory_reserved`. This captures that pool *if the
    subsystem is initialised*: ``comfy_aimdo.control`` exposes its own byte-count (summed across the device
    contexts it initialised), which this returns in MB.

    In hordelib's embedding the subsystem is inert, so this is normally 0: nothing calls ``comfy_aimdo.control``'s
    native init (only ComfyUI's ``main.py`` does), so ``control.lib`` stays ``None`` and the byte-count is 0.
    Weights therefore flow through the torch caching allocator and are counted by ``reserved_mb`` in the current
    build. The field is kept as a future-proof complement should comfy's embedding ever initialise the pool: it
    is disjoint from ``reserved_mb`` (``comfy_aimdo`` does not install its ``CUDAPluggableAllocator`` as torch's,
    so the two pools never count the same bytes). Returns 0 when the package is absent, its native library is
    uninitialised (the default), or any probe error occurs, so it never raises and is safe on the safety process
    or a CPU build.
    """
    try:
        from comfy_aimdo import control as aimdo_control

        return round(aimdo_control.get_total_vram_usage() / _MB)
    except Exception as e:
        logger.debug(f"comfy_aimdo VRAM usage unavailable: {e}")
        return 0


class ProcessVramStats(BaseModel):
    """This process's own committed device memory (MB), read from the torch allocator and the direct-IO pool.

    ``torch.cuda.memory_reserved()`` is a byte-exact, platform-independent attribution of *this* process's
    committed device memory (its live allocations plus the allocator's cached free blocks), and it excludes
    the fixed CUDA context overhead. It moves only for the process that allocates: a sibling process's
    allocations never appear here, verified identical on both Windows/WDDM and native Linux. That makes it the
    honest per-process VRAM charge the device-wide ``mem_get_info`` reading cannot give on Windows (where
    ``mem_get_info`` is a per-process view, not a device view).

    ``aimdo_mb`` is the complement the torch allocator cannot see: the engine's native direct-IO
    weight pool (see :func:`_get_aimdo_usage_mb`). It captures that pool only *if* the subsystem is
    initialised, which nothing does in hordelib's embedding, so it is normally 0 and weights are counted by
    ``reserved_mb`` instead. It is kept as a future-proof, disjoint complement: a process's full device
    footprint is ``context_constant + reserved_mb + aimdo_mb``, the context constant being resolved per
    platform by the consumer, and the two memory terms never counting the same bytes.

    ``allocated_mb`` is the live (in-use) subset of ``reserved_mb``; ``peak_reserved_mb`` is the reserved
    high-water since the allocator's peak counters were last reset (see :func:`get_process_vram_stats`).
    """

    allocated_mb: int
    reserved_mb: int
    peak_reserved_mb: int
    aimdo_mb: int


def get_process_vram_stats(*, reset_peak: bool = True) -> ProcessVramStats | None:
    """Return this process's own committed VRAM (MB) from the torch allocator, or None when unavailable.

    Byte-exact per-process attribution that is independent of platform and of siblings' usage, unlike the
    device-wide/per-process-view ``mem_get_info`` figure. Returns None cleanly when there is no CUDA/ROCm (or
    XPU) allocator to read (CPU/MPS builds, or any probe error), so a caller can guard on None without
    catching. When ``reset_peak`` is set the allocator's peak counter is reset after it is read, so each
    successive call's ``peak_reserved_mb`` is the high-water *since the previous call* (the report interval).
    """
    try:
        import torch
    except Exception as e:
        logger.debug(f"process VRAM stats unavailable (torch import failed): {e}")
        return None

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            peak_reserved = torch.cuda.max_memory_reserved()
            if reset_peak:
                torch.cuda.reset_peak_memory_stats()
            return ProcessVramStats(
                allocated_mb=round(allocated / _MB),
                reserved_mb=round(reserved / _MB),
                peak_reserved_mb=round(peak_reserved / _MB),
                aimdo_mb=_get_aimdo_usage_mb(),
            )
        if kind is AcceleratorKind.xpu and hasattr(torch, "xpu") and hasattr(torch.xpu, "memory_reserved"):
            allocated = torch.xpu.memory_allocated()
            reserved = torch.xpu.memory_reserved()
            max_reserved = getattr(torch.xpu, "max_memory_reserved", None)
            peak_reserved = max_reserved() if max_reserved is not None else reserved
            reset_peak_stats = getattr(torch.xpu, "reset_peak_memory_stats", None)
            if reset_peak and reset_peak_stats is not None:
                reset_peak_stats()
            return ProcessVramStats(
                allocated_mb=round(allocated / _MB),
                reserved_mb=round(reserved / _MB),
                peak_reserved_mb=round(peak_reserved / _MB),
                aimdo_mb=_get_aimdo_usage_mb(),
            )
    except Exception as e:
        logger.debug(f"process VRAM stats probe failed for {kind}: {e}")
        return None
    return None


def offthread_vram_sampling_ready() -> bool:
    """Whether a background thread may sample device VRAM without itself triggering a lazy device init.

    A dedicated reporter thread that reads per-process/device VRAM must never be the caller that *creates*
    the device context: several torch query primitives lazily initialise CUDA on the calling thread when no
    context exists yet (verified: ``torch.cuda.mem_get_info`` and ``torch.cuda.reset_peak_memory_stats``, the
    latter used by :func:`get_process_vram_stats`, both flip ``torch.cuda.is_initialized()`` to True when
    called first). Initialising the context off the main thread is the hazard this guards against. Once the
    context exists (the main thread has loaded a model / read VRAM), those same primitives are ordinary
    runtime queries that are safe to call from any thread, so this returns True and the sampler proceeds.

    Reads only the runtime's already-initialised flag (``torch.cuda.is_initialized`` / the XPU equivalent),
    which never creates a context. Returns True on CPU/MPS backends, where VRAM sampling falls back to psutil
    and there is no device context to lazily create, and False on any probe error (conservatively withholding
    an off-thread sample rather than risking an init). The caller reports Nones for VRAM until this is True.
    """
    try:
        import torch
    except Exception:
        return False

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            return bool(torch.cuda.is_initialized())
        if kind is AcceleratorKind.xpu and hasattr(torch, "xpu"):
            xpu_is_initialized = getattr(torch.xpu, "is_initialized", None)
            return bool(xpu_is_initialized()) if xpu_is_initialized is not None else True
    except Exception as e:
        logger.debug(f"offthread VRAM sampling readiness probe failed for {kind}: {e}")
        return False
    # CPU/MPS: no lazy device context to create; sampling falls back to psutil.
    return True


def get_free_ram_mb() -> int:
    """Available system RAM, in MB."""
    return round(psutil.virtual_memory().available / _MB)


def log_free_ram() -> None:
    offloaded_mb = get_loaded_weights_offloaded_mb()
    offloaded_note = f", Weights offloaded to RAM: {offloaded_mb:0.0f} MB" if offloaded_mb > 0 else ""
    logger.debug(
        f"Free VRAM: {get_torch_free_vram_mb():0.0f} MB, Free RAM: {get_free_ram_mb():0.0f} MB{offloaded_note}",
    )


def get_accelerator_utilization_percent(index: int = 0) -> int | None:
    """Return device ``index``'s core-utilization percentage (0-100), or None when unavailable.

    Utilization is vendor telemetry with no portable torch/ComfyUI equivalent, so this is best-effort and
    backend-specific: the NVIDIA CUDA backend is read via NVML (:mod:`hordelib.utils.nvml`); every other
    backend (ROCm, XPU, MPS, CPU) returns None until a vendor source is wired, so callers simply report no
    GPU duty cycle there. ROCm presents through ``torch.cuda`` but is not an NVML device, so it returns None.
    """
    if _active_torch_kind() is not AcceleratorKind.cuda:
        return None
    from hordelib.utils import nvml

    return nvml.get_device_utilization_percent(index)


def _enumerate_via_comfy(mm: ModuleType) -> list[AcceleratorInfo]:
    """Enumerate devices through ComfyUI's backend-agnostic device APIs."""
    devices = mm.get_all_torch_devices()
    accelerators: list[AcceleratorInfo] = []
    for index, device in enumerate(devices):
        device_type = getattr(device, "type", "cpu")
        if device_type == "cuda":
            import torch

            kind = AcceleratorKind.rocm if getattr(torch.version, "hip", None) is not None else AcceleratorKind.cuda
        else:
            kind = _COMFY_TYPE_TO_KIND.get(device_type, AcceleratorKind.cpu)

        try:
            total_mb = round(mm.get_total_memory(device) / _MB)
        except Exception as e:
            logger.debug(f"Could not read total memory for device {device}: {e}")
            total_mb = 0

        try:
            name = mm.get_torch_device_name(device)
        except Exception:
            name = str(device)

        accelerators.append(AcceleratorInfo(index=index, name=name, total_vram_mb=total_mb, kind=kind))
    return accelerators


_COMFY_TYPE_TO_KIND: dict[str, AcceleratorKind] = {
    "cuda": AcceleratorKind.cuda,
    "xpu": AcceleratorKind.xpu,
    "npu": AcceleratorKind.npu,
    "mlu": AcceleratorKind.mlu,
    "mps": AcceleratorKind.mps,
    "cpu": AcceleratorKind.cpu,
}


def _enumerate_via_torch() -> list[AcceleratorInfo]:
    """Enumerate devices with plain torch (ComfyUI not loaded in this process).

    Cannot see DirectML (ComfyUI owns that path); reports a single CPU pseudo-device when no
    accelerator is present so callers always get at least one device to reason about.
    """
    import torch

    kind = _active_torch_kind()
    accelerators: list[AcceleratorInfo] = []

    if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accelerators.append(
                AcceleratorInfo(
                    index=i,
                    name=getattr(props, "name", f"cuda:{i}"),
                    total_vram_mb=round(getattr(props, "total_memory", 0) / _MB),
                    kind=kind,
                ),
            )
    elif kind is AcceleratorKind.xpu:
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            accelerators.append(
                AcceleratorInfo(
                    index=i,
                    name=getattr(props, "name", f"xpu:{i}"),
                    total_vram_mb=round(getattr(props, "total_memory", 0) / _MB),
                    kind=kind,
                ),
            )
    elif kind is AcceleratorKind.mps:
        accelerators.append(
            AcceleratorInfo(
                index=0,
                name="Apple MPS (unified memory)",
                total_vram_mb=round(psutil.virtual_memory().total / _MB),
                kind=kind,
            ),
        )

    if not accelerators:
        accelerators.append(
            AcceleratorInfo(
                index=0,
                name="CPU",
                total_vram_mb=round(psutil.virtual_memory().total / _MB),
                kind=AcceleratorKind.cpu,
            ),
        )
    return accelerators


def enumerate_accelerators() -> list[AcceleratorInfo]:
    """Return every compute device on this machine, across all torch/ComfyUI backends.

    Prefers ComfyUI's detection when it is loaded in this process (it covers DirectML, NPU and
    MLU too); otherwise uses a plain-torch fallback. Always returns at least one device (a CPU
    pseudo-device when no accelerator is found), so callers never see an empty inventory the way
    a bare ``torch.cuda.device_count()`` loop does on non-CUDA backends.
    """
    mm = _comfy_model_management()
    if mm is not None:
        try:
            return _enumerate_via_comfy(mm)
        except Exception as e:
            logger.debug(f"ComfyUI device enumeration failed, falling back to torch: {e}")
    return _enumerate_via_torch()


def get_loaded_weights_offloaded_mb() -> int:
    """Return how much of the currently-loaded models' weights sit off the compute device, in MB.

    ComfyUI keeps a model fully resident when it fits the weight budget; otherwise it offloads the
    remainder to host RAM and streams it to VRAM during sampling (the slow path that drops device-free
    to near zero and runs several times slower). A non-zero result here is the positive runtime signal
    that weight streaming is happening for the loaded model(s), which the worker uses to confirm a
    streaming forecast and to grant the over-budget step grace instead of killing a slow-but-progressing
    job. Returns 0 when ComfyUI is not loaded in this process or nothing is resident.
    """
    mm = _comfy_model_management()
    if mm is None:
        return 0
    loaded_models = getattr(mm, "current_loaded_models", None)
    if not loaded_models:
        return 0
    offloaded_bytes = 0
    for loaded_model in loaded_models:
        read_offloaded = getattr(loaded_model, "model_offloaded_memory", None)
        if read_offloaded is None:
            continue
        try:
            offloaded_bytes += max(0, int(read_offloaded()))
        except Exception as e:
            logger.debug(f"offloaded-memory read failed for a loaded model: {e}")
    return round(offloaded_bytes / _MB)


def _is_core_diffusion_module(module: object) -> bool:
    """Whether ``module`` is a core diffusion model (the checkpoint's UNet/DiT), not a support component.

    ComfyUI loads a checkpoint's diffusion model as a :class:`comfy.model_base.BaseModel`; its text
    encoders and VAE are distinct modules loaded through their own patchers. The distinction sizes the
    streaming-vs-residency split: streaming judgments apply to the core weights, while sibling-room and
    whole-card judgments must charge the core plus every support component. Returns False (support, the
    conservative side for a room judgment) when ComfyUI is not importable.
    """
    try:
        from comfy.model_base import BaseModel
    except Exception:
        return False
    return isinstance(module, BaseModel)


class ResidentFootprint(BaseModel):
    """Measured resident weight footprint of a job, split into core diffusion weights and support.

    ``support_mb`` is the text encoders and VAE force-loaded alongside the core weights; ``total_mb`` is
    the full set a scheduler must charge when deciding whether a card can host this model beside anything
    else. Distinct from an activation-inclusive VRAM high-water mark: this is weights only.
    """

    core_mb: float
    support_mb: float

    @property
    def total_mb(self) -> float:
        """Core diffusion weights plus every force-loaded support component."""
        return self.core_mb + self.support_mb


class ResidentFootprintRecorder:
    """Accumulates the distinct model components ComfyUI loads to the device, deduplicated by patcher.

    Recording every load (rather than snapshotting ``current_loaded_models`` once) is what makes the
    measurement robust to mid-job eviction: a text encoder freed before VAE decode is still counted,
    which is exactly the component an end-of-job snapshot would miss and the one whose omission from a
    burden seed under-counts a multi-component checkpoint.
    """

    def __init__(self) -> None:
        self._by_patcher: dict[int, tuple[bool, float]] = {}

    def record_load(self, models: object) -> None:
        """Record each patcher in a ``load_models_gpu`` call, classified core vs support by its module."""
        try:
            patchers = list(models)  # type: ignore[call-overload]
        except TypeError:
            return
        for patcher in patchers:
            module = getattr(patcher, "model", None)
            model_size = getattr(patcher, "model_size", None)
            if module is None or model_size is None:
                continue
            try:
                size_mb = float(model_size()) / _MB
            except Exception as e:
                logger.debug(f"model_size read failed for a loaded patcher: {e}")
                continue
            self._by_patcher[id(patcher)] = (_is_core_diffusion_module(module), size_mb)

    def resident_footprint(self) -> ResidentFootprint:
        """Return the accumulated core and support weight totals (MB)."""
        core = sum(size for is_core, size in self._by_patcher.values() if is_core)
        support = sum(size for is_core, size in self._by_patcher.values() if not is_core)
        return ResidentFootprint(core_mb=core, support_mb=support)


@contextmanager
def record_resident_footprint() -> Iterator[ResidentFootprintRecorder]:
    """Record the resident weight footprint of the ComfyUI work done inside the block.

    Wraps ``model_management.load_models_gpu`` so every component the backend force-loads to the device
    during a job is captured (deduplicated), then classified into core diffusion weights and support
    components. Yields a no-op recorder when ComfyUI is not loaded, so callers need no guard. The wrapper
    is removed on exit even if the block raises. This is the measurement side of the burden registry: a
    seed can be verified against what the backend actually loads for a real job on real hardware.
    """
    recorder = ResidentFootprintRecorder()
    mm = _comfy_model_management()
    original = getattr(mm, "load_models_gpu", None) if mm is not None else None
    if mm is None or original is None:
        yield recorder
        return

    def _recording_load_models_gpu(models: object, *args: object, **kwargs: object) -> object:
        recorder.record_load(models)
        return original(models, *args, **kwargs)

    mm.load_models_gpu = _recording_load_models_gpu  # type: ignore[attr-defined]  # patching a dynamic module
    try:
        yield recorder
    finally:
        mm.load_models_gpu = original  # type: ignore[attr-defined]  # restore the backend's loader


def _sum_resident_weights_mb() -> float:
    """Sum the weights *currently on the device* across ComfyUI's loaded models (MB), 0 without ComfyUI.

    Reads each loaded model's ``model_loaded_memory`` (the weights actually resident, excluding any offloaded
    to host RAM), so block-swapped or partially-offloaded weights are counted at their on-device size, not
    their full checkpoint size. A snapshot: sampled over a job it reveals the peak *simultaneous* residency,
    which is distinct from the union of every component ever loaded (components that time-share the device are
    never all resident at once).
    """
    mm = _comfy_model_management()
    if mm is None:
        return 0.0
    loaded_models = getattr(mm, "current_loaded_models", None)
    if not loaded_models:
        return 0.0
    resident_bytes = 0
    for loaded_model in list(loaded_models):  # copy: the backend mutates this list from the worker thread
        read_loaded = getattr(loaded_model, "model_loaded_memory", None)
        if read_loaded is None:
            continue
        try:
            resident_bytes += max(0, int(read_loaded()))
        except Exception as e:
            logger.debug(f"loaded-memory read failed for a loaded model: {e}")
    return resident_bytes / _MB


class JobVramProfile(BaseModel):
    """Peak VRAM a job actually consumed, separating simultaneous residency from the union of components.

    ``peak_resident_weights_mb`` is the largest the *on-device weight set* ever got at one instant. It is the
    honest figure for a co-residency judgment: when the backend evicts a text encoder before sampling or
    block-swaps a large diffusion model, the peak simultaneous weight set is below the summed component
    weights (``sum_component_weights_mb``, the conservative upper bound). ``peak_device_used_mb`` adds the
    transient activation high-water, so it is what an activation-inclusive peak seed (``vram_base_mb``) should
    be measured against. Zero fields mean the profile ran without ComfyUI or a device to read.
    """

    peak_resident_weights_mb: float
    peak_device_used_mb: float
    sum_component_weights_mb: float

    @property
    def time_shared_mb(self) -> float:
        """How much the summed components exceed the peak simultaneous weights (the eviction/swap savings)."""
        return max(0.0, self.sum_component_weights_mb - self.peak_resident_weights_mb)


class _JobVramProfiler:
    """Samples on-device weights and device-used VRAM on a background thread while a job runs."""

    def __init__(self, recorder: ResidentFootprintRecorder, *, poll_interval_s: float) -> None:
        self._recorder = recorder
        self._poll_interval_s = poll_interval_s
        self._peak_resident_weights_mb = 0.0
        self._peak_device_used_mb = 0.0
        self._total_vram_mb = float(get_torch_total_vram_mb())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vram-profiler", daemon=True)

    def _sample(self) -> None:
        self._peak_resident_weights_mb = max(self._peak_resident_weights_mb, _sum_resident_weights_mb())
        device_used = self._total_vram_mb - float(get_torch_device_free_vram_mb())
        self._peak_device_used_mb = max(self._peak_device_used_mb, device_used)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as e:
                logger.debug(f"vram profile sample failed: {e}")
            self._stop.wait(self._poll_interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()  # a final reading so a peak between the last poll and stop is not lost

    def profile(self) -> JobVramProfile:
        """Return the measured peaks alongside the summed-component upper bound from the recorder."""
        footprint = self._recorder.resident_footprint()
        return JobVramProfile(
            peak_resident_weights_mb=self._peak_resident_weights_mb,
            peak_device_used_mb=self._peak_device_used_mb,
            sum_component_weights_mb=footprint.total_mb,
        )


@contextmanager
def record_job_vram_profile(*, poll_interval_s: float = 0.02) -> Iterator[_JobVramProfiler]:
    """Measure a job's peak *simultaneous* VRAM, not just the union of components it loaded.

    Samples the on-device weight set and the device-used high-water on a background thread while wrapping
    ``load_models_gpu`` for the component union, so one run yields all three views: the summed components
    (conservative), the peak simultaneous weights (the true co-residency footprint once time-sharing is
    accounted for), and the peak device use (weights plus activations). Yields an inert profiler when
    ComfyUI is not loaded. This is how a burden seed is set from what a model *actually* holds at once,
    rather than from an assumption about whether its components co-reside.
    """
    with record_resident_footprint() as recorder:
        profiler = _JobVramProfiler(recorder, poll_interval_s=poll_interval_s)
        if _comfy_model_management() is None:
            yield profiler
            return
        profiler.start()
        try:
            yield profiler
        finally:
            profiler.stop()


def clear_accelerator_cache() -> None:
    """Release cached device memory on whatever backend is active (a no-op on CPU).

    Delegates to ComfyUI's ``soft_empty_cache`` when loaded (backend-aware for CUDA/XPU/NPU/
    MLU/MPS); otherwise clears the active torch backend's cache directly. Never imports from
    ``torch.cuda`` unconditionally, so it is safe on non-NVIDIA builds.
    """
    mm = _comfy_model_management()
    if mm is not None:
        soft_empty_cache = getattr(mm, "soft_empty_cache", None)
        if soft_empty_cache is not None:
            soft_empty_cache()
            return

    import torch

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            torch.cuda.empty_cache()
        elif kind is AcceleratorKind.xpu and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
        elif kind is AcceleratorKind.mps and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception as e:
        logger.debug(f"clear_accelerator_cache failed for {kind}: {e}")
