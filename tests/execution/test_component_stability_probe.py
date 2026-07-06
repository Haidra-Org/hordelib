"""Tests for the cross-machine component stability probe.

The record model, dtype histogram and comparison verdict are pure and torch-light, so they are exercised
here with synthetic records and small CPU modules on any platform. The GPU probe path
(:func:`probe_checkpoint`) needs a real checkpoint and ComfyUI environment and is run manually on reference
rigs, not in this suite.

The load-bearing scenario is ``test_content_stable_module_diverges``: two machines produce the *same* content
hash but *different* module hashes. That is the empirical shape the whole sharing design assumes, and the
reason the content hash (not the module hash) is the shipped canonical identity.
"""

from __future__ import annotations

import torch
from horde_model_reference.component_hash import ComponentKind

from hordelib.execution.component_stability_probe import (
    ComponentProbe,
    ComponentProbeRecord,
    _module_probe,
    _render_comparison,
    compare_records,
    resolved_dtypes,
)


def _record(label: str, components: dict[str, ComponentProbe]) -> ComponentProbeRecord:
    """Build a probe record with only the fields the comparison logic reads populated."""
    return ComponentProbeRecord(
        label=label,
        model_name="test_sdxl",
        file_name="test_sdxl.safetensors",
        file_size=123,
        platform="test",
        python_version="3.12.0",
        torch_version="2.0.0",
        device="cuda",
        gpu_name="TestGPU",
        components=components,
    )


def _probe(kind: ComponentKind, content: str | None, module: str | None, dtypes: dict[str, int]) -> ComponentProbe:
    """Build a single component probe."""
    return ComponentProbe(
        kind=kind.value,
        content_hash=content,
        module_hash=module,
        resolved_dtypes=dtypes,
        num_tensors=sum(dtypes.values()),
    )


class TestResolvedDtypes:
    """The dtype histogram: only tensor-like entries counted, keyed by their dtype string."""

    def test_counts_tensors_by_dtype(self) -> None:
        state_dict = {
            "a": torch.zeros(2, dtype=torch.float16),
            "b": torch.zeros(2, dtype=torch.float16),
            "c": torch.zeros(2, dtype=torch.float32),
        }
        assert resolved_dtypes(state_dict) == {"torch.float16": 2, "torch.float32": 1}

    def test_ignores_non_tensor_entries(self) -> None:
        assert resolved_dtypes({"meta": "not-a-tensor", "w": torch.zeros(1)}) == {"torch.float32": 1}


class TestRecordRoundTrip:
    """A record survives a JSON-shaped dict round trip with its nested component probes intact."""

    def test_to_dict_from_dict_is_identity(self) -> None:
        record = _record(
            "machine-1",
            {
                "vae": _probe(ComponentKind.VAE, "c-vae", "m-vae", {"torch.float16": 3}),
                "text_encoders": _probe(ComponentKind.TEXT_ENCODERS, "c-te", "m-te", {"torch.float16": 5}),
            },
        )
        assert ComponentProbeRecord.from_dict(record.to_dict()) == record


class TestCompareRecords:
    """The cross-machine verdict: content hash is the pass condition, module divergence is the demonstration."""

    def test_all_agree_is_stable(self) -> None:
        components = {"vae": _probe(ComponentKind.VAE, "c-vae", "m-vae", {"torch.float16": 3})}
        comparison = compare_records([_record("a", components), _record("b", dict(components))])
        assert comparison.content_hash_stable is True
        assert comparison.module_hash_diverged is False

    def test_content_stable_module_diverges(self) -> None:
        # Same file bytes on both machines (content equal), but a device-dependent dtype pick splits the
        # module hash. This is the shape that justifies shipping the content hash as the identity.
        machine_a = _record("a", {"vae": _probe(ComponentKind.VAE, "c-vae", "m-fp16", {"torch.float16": 3})})
        machine_b = _record("b", {"vae": _probe(ComponentKind.VAE, "c-vae", "m-bf16", {"torch.bfloat16": 3})})
        comparison = compare_records([machine_a, machine_b])
        assert comparison.content_hash_stable is True
        assert comparison.module_hash_diverged is True
        (vae,) = comparison.per_component
        assert vae.content_hash_by_value == {"c-vae": ["a", "b"]}
        assert vae.module_hash_by_value == {"m-fp16": ["a"], "m-bf16": ["b"]}

    def test_content_divergence_fails(self) -> None:
        machine_a = _record("a", {"vae": _probe(ComponentKind.VAE, "c-one", "m", {"torch.float16": 3})})
        machine_b = _record("b", {"vae": _probe(ComponentKind.VAE, "c-two", "m", {"torch.float16": 3})})
        comparison = compare_records([machine_a, machine_b])
        assert comparison.content_hash_stable is False

    def test_missing_content_hash_does_not_fabricate_instability(self) -> None:
        # A machine that could not extract the component (None content hash) must not read as a divergence.
        machine_a = _record("a", {"vae": _probe(ComponentKind.VAE, "c-vae", "m", {"torch.float16": 3})})
        machine_b = _record("b", {"vae": _probe(ComponentKind.VAE, None, "m", {})})
        comparison = compare_records([machine_a, machine_b])
        assert comparison.content_hash_stable is True

    def test_empty_records_are_not_a_pass(self) -> None:
        assert compare_records([]).content_hash_stable is False

    def test_render_names_the_verdict(self) -> None:
        components = {"vae": _probe(ComponentKind.VAE, "c-vae", "m-vae", {"torch.float16": 3})}
        text = _render_comparison(compare_records([_record("a", components), _record("b", dict(components))]))
        assert "PASS" in text
        assert "vae" in text


class TestModuleProbe:
    """The loaded-module probe reads the same submodules the runtime adoption path hashes."""

    def _holder(self, attr: str, dtype: torch.dtype) -> object:
        module = torch.nn.Linear(4, 4, bias=False).to(dtype)
        holder = type("Holder", (), {})()
        setattr(holder, attr, module)
        return holder

    def test_probes_vae_first_stage_model(self) -> None:
        loaded = {"vae": self._holder("first_stage_model", torch.float16)}
        module_hash, dtypes, num_tensors = _module_probe(ComponentKind.VAE, loaded)
        assert module_hash is not None
        assert dtypes == {"torch.float16": 1}
        assert num_tensors == 1

    def test_probes_clip_cond_stage_model(self) -> None:
        loaded = {"clip": self._holder("cond_stage_model", torch.float32)}
        module_hash, dtypes, _ = _module_probe(ComponentKind.TEXT_ENCODERS, loaded)
        assert module_hash is not None
        assert dtypes == {"torch.float32": 1}

    def test_absent_submodule_is_empty_probe(self) -> None:
        assert _module_probe(ComponentKind.VAE, {"vae": object()}) == (None, {}, 0)
