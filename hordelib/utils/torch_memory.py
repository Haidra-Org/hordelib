"""Memory statistics helpers that work with or without ComfyUI loaded.

Consumers (notably horde-worker-reGen's process monitoring, including its safety process,
which must never trigger a ComfyUI import) need VRAM/RAM numbers in processes where
``hordelib.initialise()`` may not have run. When ComfyUI *is* loaded in this process, these
delegate to its memory management for exact agreement with what the execution backend sees;
otherwise they fall back to plain torch device queries.
"""

import psutil
from loguru import logger

_MB = 1024 * 1024


def _comfy_memory_funcs() -> tuple | None:
    """Return comfy's (get_total_memory, get_free_memory) if comfy is imported here."""
    try:
        from hordelib import comfy_horde
    except ImportError:
        return None
    total = getattr(comfy_horde, "_comfy_get_total_memory", None)
    free = getattr(comfy_horde, "_comfy_get_free_memory", None)
    if total is None or free is None:
        return None
    return total, free


def get_torch_total_vram_mb() -> int:
    """Total VRAM of the active torch device, in MB (0 if no CUDA device)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[0]() / _MB)

    import torch

    if not torch.cuda.is_available():
        return 0
    _, total = torch.cuda.mem_get_info()
    return round(total / _MB)


def get_torch_free_vram_mb() -> int:
    """Free VRAM of the active torch device, in MB (0 if no CUDA device)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[1]() / _MB)

    import torch

    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return round(free / _MB)


def get_free_ram_mb() -> int:
    """Available system RAM, in MB."""
    return round(psutil.virtual_memory().available / _MB)


def log_free_ram() -> None:
    logger.debug(f"Free VRAM: {get_torch_free_vram_mb():0.0f} MB, Free RAM: {get_free_ram_mb():0.0f} MB")
