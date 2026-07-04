"""Tests for cross-process shared model components.

The transport/registry logic is validated here with CPU tensors (runs on any platform, exercising
the exact queue/registry/serve-thread paths); the CUDA IPC mapping itself only exists on Linux with
CUDA, so the true zero-copy test is gated to that environment and is expected to run on a Linux rig.
"""

from __future__ import annotations

import multiprocessing
import sys

import pytest
import torch

from hordelib.execution.shared_components import (
    SharedComponentBus,
    SharedComponentClient,
    assert_unchanged,
    hash_state_dict,
    is_cuda_ipc_supported,
)


@pytest.fixture()
def bus() -> SharedComponentBus:  # type: ignore[misc]
    made = SharedComponentBus(multiprocessing.get_context("spawn"), [1, 2])
    yield made
    made.shutdown()


def _clients(bus: SharedComponentBus) -> tuple[SharedComponentClient, SharedComponentClient]:
    # enabled=True forces the platform gate open so CPU tensors exercise the full path anywhere.
    return (
        SharedComponentClient(bus.endpoint_for(1), enabled=True),
        SharedComponentClient(bus.endpoint_for(2), enabled=True),
    )


class TestHashing:
    """Content hashing: equality on identical bytes, sensitivity to any semantic difference."""

    def test_identical_state_dicts_hash_equal(self) -> None:
        a = {"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
        b = {"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
        assert hash_state_dict(a) == hash_state_dict(b)

    def test_data_shape_and_dtype_all_matter(self) -> None:
        base = {"w": torch.zeros(4, 4, dtype=torch.float32)}
        assert hash_state_dict({"w": torch.ones(4, 4)}) != hash_state_dict(base)
        assert hash_state_dict({"w": torch.zeros(16)}) != hash_state_dict(base)
        assert hash_state_dict({"w": torch.zeros(4, 4, dtype=torch.float16)}) != hash_state_dict(base)


class TestPublishFetch:
    """The registry/serve/fetch protocol, with CPU tensors standing in for CUDA ones."""

    def test_fetch_returns_published_tensors(self, bus: SharedComponentBus) -> None:
        producer, consumer = _clients(bus)
        tensors = {"weight": torch.full((2, 2), 7.0)}
        assert producer.publish("hash-a", tensors) is True

        fetched = consumer.fetch("hash-a")

        assert fetched is not None
        assert torch.equal(fetched["weight"], tensors["weight"])
        producer.unregister()

    def test_fetch_unknown_hash_is_none(self, bus: SharedComponentBus) -> None:
        _producer, consumer = _clients(bus)
        assert consumer.fetch("nope") is None

    def test_own_publication_is_not_fetched(self, bus: SharedComponentBus) -> None:
        producer, _consumer = _clients(bus)
        producer.publish("hash-a", {"w": torch.zeros(1)})
        assert producer.fetch("hash-a") is None
        producer.unregister()

    def test_unregister_withdraws_offer(self, bus: SharedComponentBus) -> None:
        producer, consumer = _clients(bus)
        producer.publish("hash-a", {"w": torch.zeros(1)})
        producer.unregister()
        assert consumer.fetch("hash-a") is None

    def test_disabled_client_noops(self, bus: SharedComponentBus) -> None:
        client = SharedComponentClient(bus.endpoint_for(1), enabled=False)
        assert client.publish("hash-a", {"w": torch.zeros(1)}) is False
        assert client.fetch("hash-a") is None

    def test_published_tensors_are_grad_free(self, bus: SharedComponentBus) -> None:
        producer, consumer = _clients(bus)
        producer.publish("hash-a", {"w": torch.zeros(2, requires_grad=True)})
        fetched = consumer.fetch("hash-a")
        assert fetched is not None
        assert fetched["w"].requires_grad is False
        producer.unregister()


class TestContaminationGuard:
    def test_assert_unchanged_detects_write_through(self) -> None:
        tensors = {"w": torch.ones(4)}
        fingerprints = {"w": float(tensors["w"].sum().item())}
        assert assert_unchanged(tensors, fingerprints) is True
        tensors["w"] += 1.0
        assert assert_unchanged(tensors, fingerprints) is False


def _cuda_child(endpoint, result_queue) -> None:
    import torch as child_torch

    client = SharedComponentClient(endpoint, enabled=True)
    fetched = client.fetch("cuda-hash")
    ok = (
        fetched is not None
        and fetched["w"].is_cuda
        and bool(child_torch.equal(fetched["w"].cpu(), child_torch.full((8,), 3.0)))
    )
    result_queue.put(ok)


@pytest.mark.skipif(not is_cuda_ipc_supported(), reason="CUDA IPC requires Linux with CUDA (run on the Linux rig)")
def test_cuda_tensor_maps_across_real_processes() -> None:
    """END-TO-END (Linux+CUDA only): a child process maps the parent's CUDA tensor via IPC."""
    ctx = multiprocessing.get_context("spawn")
    bus = SharedComponentBus(ctx, [1, 2])
    try:
        producer = SharedComponentClient(bus.endpoint_for(1), enabled=True)
        producer.publish("cuda-hash", {"w": torch.full((8,), 3.0, device="cuda")})

        result_queue = ctx.Queue()
        child = ctx.Process(target=_cuda_child, args=(bus.endpoint_for(2), result_queue))
        child.start()
        assert result_queue.get(timeout=30) is True
        child.join(timeout=10)
        producer.unregister()
    finally:
        bus.shutdown()


if __name__ == "__main__" and "--cuda-smoke" in sys.argv:
    # Convenience for the Linux rig: python tests/execution/test_shared_components.py --cuda-smoke
    test_cuda_tensor_maps_across_real_processes()
    print("CUDA IPC smoke passed")
