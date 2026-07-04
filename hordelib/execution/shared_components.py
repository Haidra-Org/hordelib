"""Cross-process sharing of immutable model-component weights (CUDA IPC on Linux).

Several worker processes serving the same family of checkpoints hold byte-identical component
weights: most SDXL finetunes embed the same two text encoders and the same VAE (~1.7GB per
checkpoint), and sibling processes serving the same checkpoint duplicate its UNet. On Linux,
PyTorch can pass CUDA tensors between processes zero-copy (``torch.multiprocessing`` registers
``ForkingPickler`` reductions that ship a cudaIpc handle instead of bytes), so one process's
device-resident copy can serve every process on the card. Windows/WDDM does not support CUDA IPC:
everything here degrades to a no-op there, and CPU-RAM dedupe is already provided by the mmap
zero-copy load path (see :mod:`hordelib.execution.zero_copy_load`).

Topology (the orchestrator parent is torch-free by invariant, so tensors never route through it):

- The parent constructs a :class:`SharedComponentBus` with plain ``multiprocessing`` primitives
  (queues and a manager dict hold only picklable metadata and tensor payloads in transit; the
  parent never gets or puts tensors) and hands each GPU child its endpoint.
- Each child wraps its endpoint in a :class:`SharedComponentClient`. A child that has a component
  on-device *publishes* it (content hash -> its process id in the registry, tensors pinned locally,
  a serve thread answering fetch requests). A child that needs a component *fetches* it: the
  producer ships the tensor dict through the requester's inbox queue; unpickling maps the same VRAM.

Sharing contract (contamination safety):

- Only immutable BASE weights may be published. LoRA/controlnet patching must never write through a
  shared tensor: ComfyUI's patcher computes patched weights into fresh per-process tensors rather
  than mutating the base in place (verify against the vendored comfy before enabling by default);
  published tensors additionally have ``requires_grad`` off, and :func:`assert_unchanged` gives a
  debug-time data check for patched-job validation.
- A published component is pinned in the producer for the producer's lifetime: a cudaIpc mapping
  dies with its source allocation, so publication is only offered for components whose residency is
  process-long. Producer death invalidates consumers' mappings; consumers must treat fetch results
  as best-effort and fall back to loading normally (the registry entry is dropped when the producer
  is unregistered).
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import multiprocessing
    from collections.abc import Mapping


_FETCH_TIMEOUT_SECONDS = 10.0
"""How long a consumer waits for a producer to answer before falling back to a normal load."""


def is_cuda_ipc_supported() -> bool:
    """Whether this host can share CUDA tensors across processes (Linux with CUDA available).

    Windows/WDDM has no cudaIpc; macOS has no CUDA. Import failures (CPU-only installs) are a
    normal "no". Never raises.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def hash_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Content-hash a state dict's tensor bytes (keys, shapes, dtypes, and data).

    Intended for the CPU (mmap-view) state dict *before* upload, where reading the bytes is cheap;
    byte-identical components in different checkpoint files hash equal, which is the only condition
    under which substitution is permitted (never "known-good component" heuristics: finetunes bake
    genuinely different VAEs).
    """
    import torch

    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class SharedComponentEndpoint:
    """One child's picklable handle onto the bus: its identity, inboxes, and the shared registry."""

    process_id: int
    registry: Any
    """Manager dict: component hash -> producer process id. Metadata only, parent-safe."""
    request_queues: dict[int, Any]
    """Per-child request inboxes: (component_hash, requester_process_id) tuples."""
    response_queues: dict[int, Any]
    """Per-child fetch inboxes: (component_hash, {name: tensor} | None) tuples."""


class SharedComponentBus:
    """Parent-side constructor of the sharing fabric. Torch-free: builds only mp primitives.

    Build once before spawning GPU children, then pass ``endpoint_for(process_id)`` into each
    child's entry kwargs. The parent must never get/put on the queues (tensor payloads transit
    them; unpickling one would import torch into the parent).
    """

    def __init__(self, ctx: multiprocessing.context.BaseContext, process_ids: list[int]) -> None:
        """Create per-child queues and the metadata registry for the given child process ids."""
        self._manager = ctx.Manager()
        self._registry = self._manager.dict()
        self._request_queues = {pid: ctx.Queue() for pid in process_ids}
        self._response_queues = {pid: ctx.Queue() for pid in process_ids}

    def endpoint_for(self, process_id: int) -> SharedComponentEndpoint:
        """The endpoint to pass into the child with this process id."""
        if process_id not in self._request_queues:
            raise KeyError(f"process id {process_id} was not declared to the bus")
        return SharedComponentEndpoint(
            process_id=process_id,
            registry=self._registry,
            request_queues=dict(self._request_queues),
            response_queues=dict(self._response_queues),
        )

    def shutdown(self) -> None:
        """Tear down the manager (registry proxies become invalid; children should already be gone)."""
        self._manager.shutdown()


@dataclass
class _Published:
    tensors: dict[str, Any]
    published_at: float = field(default_factory=time.time)


class SharedComponentClient:
    """Child-side publish/fetch API over a :class:`SharedComponentEndpoint`.

    All methods degrade to "not shared" (False / None) rather than raising: sharing is an
    optimization of a healthy worker, never a load dependency.
    """

    def __init__(self, endpoint: SharedComponentEndpoint, *, enabled: bool | None = None) -> None:
        """Wrap the endpoint; ``enabled`` overrides the platform gate (tests use CPU tensors anywhere)."""
        self._endpoint = endpoint
        self._enabled = is_cuda_ipc_supported() if enabled is None else enabled
        self._published: dict[str, _Published] = {}
        self._serve_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._fetch_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether sharing is active on this host."""
        return self._enabled

    def publish(self, component_hash: str, tensors: Mapping[str, Any]) -> bool:
        """Offer this process's device-resident copy of a component to siblings.

        Pins the tensors locally for this process's lifetime (a cudaIpc mapping dies with its source
        allocation) and registers this process as the producer unless another producer is already
        registered. Tensors are marked non-differentiable; they must be the immutable base weights.
        """
        if not self._enabled:
            return False
        try:
            pinned = {name: tensor.detach().requires_grad_(False) for name, tensor in tensors.items()}
            self._published[component_hash] = _Published(tensors=pinned)
            self._ensure_serving()
            existing = self._endpoint.registry.setdefault(component_hash, self._endpoint.process_id)
            return existing == self._endpoint.process_id
        except Exception as publish_error:
            logger.debug(f"Shared-component publish failed ({publish_error})")
            return False

    def fetch(self, component_hash: str) -> dict[str, Any] | None:
        """Map a sibling's copy of the component, or None (unknown, timeout, producer gone, disabled).

        The returned CUDA tensors alias the producer's VRAM: adopt them as module weights only for
        read (never in-place writes; patching must clone). One outstanding fetch at a time per child.
        """
        if not self._enabled:
            return None
        try:
            producer = self._endpoint.registry.get(component_hash)
            if producer is None or producer == self._endpoint.process_id:
                return None
            request_queue = self._endpoint.request_queues.get(producer)
            response_queue = self._endpoint.response_queues.get(self._endpoint.process_id)
            if request_queue is None or response_queue is None:
                return None
            with self._fetch_lock:
                request_queue.put((component_hash, self._endpoint.process_id))
                deadline = time.time() + _FETCH_TIMEOUT_SECONDS
                while time.time() < deadline:
                    got_hash, payload = response_queue.get(timeout=max(0.1, deadline - time.time()))
                    if got_hash == component_hash:
                        return payload
            return None
        except Exception as fetch_error:
            logger.debug(f"Shared-component fetch failed ({fetch_error})")
            return None

    def unregister(self) -> None:
        """Withdraw this process's publications (call before exit so consumers stop fetching from it)."""
        self._stop_event.set()
        try:
            for component_hash in list(self._published):
                if self._endpoint.registry.get(component_hash) == self._endpoint.process_id:
                    self._endpoint.registry.pop(component_hash, None)
        except Exception:
            pass
        self._published.clear()

    def _ensure_serving(self) -> None:
        if self._serve_thread is not None and self._serve_thread.is_alive():
            return
        self._serve_thread = threading.Thread(target=self._serve_loop, name="shared-components", daemon=True)
        self._serve_thread.start()

    def _serve_loop(self) -> None:
        inbox = self._endpoint.request_queues[self._endpoint.process_id]
        while not self._stop_event.is_set():
            try:
                component_hash, requester = inbox.get(timeout=0.5)
            except Exception:
                continue
            entry = self._published.get(component_hash)
            payload = dict(entry.tensors) if entry is not None else None
            response_queue = self._endpoint.response_queues.get(requester)
            if response_queue is None:
                continue
            try:
                response_queue.put((component_hash, payload))
            except Exception as serve_error:
                logger.debug(f"Shared-component serve failed ({serve_error})")


def assert_unchanged(tensors: Mapping[str, Any], fingerprints: Mapping[str, float]) -> bool:
    """Debug check that shared base weights were not written through by a patched job.

    ``fingerprints`` maps names to a cheap pre-job statistic (``tensor.float().sum().item()``);
    returns False (and logs) if any tensor's statistic changed. Intended for validation runs, not
    the hot path.
    """
    for name, before in fingerprints.items():
        tensor = tensors.get(name)
        if tensor is None:
            continue
        after = float(tensor.detach().float().sum().item())
        if after != before:
            logger.error(f"Shared component tensor {name!r} changed under a patched job ({before} -> {after})")
            return False
    return True
