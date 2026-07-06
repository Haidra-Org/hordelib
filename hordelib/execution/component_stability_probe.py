"""Empirically confirm which component identity is reproducible across machines, and which is not.

Cross-process component sharing relies on two different hashes, and the whole design rests on knowing
which one is safe to *ship* as the canonical identity in the model reference:

* the **content hash** (:mod:`horde_model_reference.component_hash`): a torch-free hash of the component's
  stored file bytes. It has no device, dtype-selection or driver input, so the claim is that it is identical
  on every machine in a heterogeneous fleet. This is the identity the reference publishes.
* the **module hash** (:func:`hordelib.execution.shared_components.hash_state_dict`): a hash of the
  ComfyUI-loaded module's state dict. It folds in ``str(tensor.dtype)``, and ComfyUI's load-time dtype
  selection is device-dependent (fp16 vs bf16 vs fp32 by card/config), so the claim is that it can differ
  between machines. This is the *runtime* adoption key used by sibling processes on one card, where they
  share a dtype regime; it must never be the shipped identity.

This module makes both claims falsifiable on real hardware. On one machine, for one monolithic SD1.5/SDXL
checkpoint, :func:`probe_checkpoint` records both hashes per component (VAE and text encoders) alongside the
load-time environment that could perturb them (device, GPU, resolved dtypes, torch version). Run it on
several reference machines, collect the JSON records, and :func:`compare_records` reports whether the content
hash held identical everywhere (the pass condition) and whether the module hash diverged (the demonstration
of why only the content hash is shippable).

The pure record and comparison logic is torch-free and needs no GPU (``import`` and ``compare`` run
anywhere, including CPU-only CI). Only :func:`probe_checkpoint` and the ``probe`` CLI subcommand load a real
checkpoint, so they require a working ComfyUI/torch environment and the model files.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from horde_model_reference.component_hash import (
    ComponentKind,
    NoComponentTensorsError,
    UnsupportedContainerError,
    hash_embedded_component_file,
)
from loguru import logger

# The monolithic components in scope for v1: an SD1.5/SDXL checkpoint embeds exactly these, and each maps to
# a ComfyUI-loaded module whose state dict the runtime hashes for adoption.
_MONOLITHIC_KINDS: tuple[ComponentKind, ...] = (ComponentKind.VAE, ComponentKind.TEXT_ENCODERS)

# Where each component's loaded weights hang off the ComfyUI objects returned by the checkpoint loader. A
# probe hashes the module at this path, so the resulting module hash corresponds to the same component
# identified by content hash in the model-reference component registry.
_LOADED_MODULE_ATTR: dict[ComponentKind, tuple[str, str]] = {
    ComponentKind.VAE: ("vae", "first_stage_model"),
    ComponentKind.TEXT_ENCODERS: ("clip", "cond_stage_model"),
}


@dataclass(frozen=True)
class ComponentProbe:
    """Both identities and the resolved dtypes for one component of one checkpoint on one machine."""

    kind: str
    """The :class:`~horde_model_reference.component_hash.ComponentKind` value (``vae``/``text_encoders``)."""
    content_hash: str | None
    """The torch-free file-content hash, or None when the component could not be extracted from the file."""
    module_hash: str | None
    """The ComfyUI-loaded-module hash, or None when the loaded module was absent."""
    resolved_dtypes: dict[str, int]
    """The loaded module's dtype histogram (``str(dtype) -> tensor count``): the variable that moves the
    module hash without moving the content hash, so a divergence can be correlated to it."""
    num_tensors: int


@dataclass(frozen=True)
class ComponentProbeRecord:
    """One machine's probe of one checkpoint: per-component identities plus the environment that could move them."""

    label: str
    """Machine identifier used to attribute results in a comparison (defaults to the hostname)."""
    model_name: str
    file_name: str
    file_size: int
    """Total file size in bytes: a cheap guard that every machine hashed the same checkpoint bytes."""
    platform: str
    python_version: str
    torch_version: str | None
    device: str
    gpu_name: str | None
    components: dict[str, ComponentProbe] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (nested :class:`ComponentProbe` values included)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentProbeRecord:
        """Rebuild a record from :meth:`to_dict` output (rehydrates the nested component probes)."""
        components = {
            kind: ComponentProbe(
                kind=probe["kind"],
                content_hash=probe.get("content_hash"),
                module_hash=probe.get("module_hash"),
                resolved_dtypes=dict(probe.get("resolved_dtypes", {})),
                num_tensors=int(probe.get("num_tensors", 0)),
            )
            for kind, probe in data.get("components", {}).items()
        }
        return cls(
            label=data["label"],
            model_name=data["model_name"],
            file_name=data["file_name"],
            file_size=int(data["file_size"]),
            platform=data["platform"],
            python_version=data["python_version"],
            torch_version=data.get("torch_version"),
            device=data["device"],
            gpu_name=data.get("gpu_name"),
            components=components,
        )


@dataclass(frozen=True)
class ComponentComparison:
    """Cross-machine agreement for one component kind, and the dtypes that explain any module-hash split."""

    kind: str
    content_hash_by_value: dict[str, list[str]]
    """content_hash -> the labels that produced it. More than one key means the shipped identity is unstable."""
    module_hash_by_value: dict[str, list[str]]
    """module_hash -> the labels that produced it. More than one key demonstrates device dependence."""
    dtypes_by_label: dict[str, dict[str, int]]
    """label -> its loaded-module dtype histogram, to correlate a module-hash split to a dtype difference."""
    content_hash_stable: bool
    module_hash_stable: bool


@dataclass(frozen=True)
class ProbeComparison:
    """The verdict over a set of machines' records: is the shipped identity stable, did the runtime key diverge."""

    labels: list[str]
    per_component: list[ComponentComparison]
    content_hash_stable: bool
    """True only if every probed component's content hash was identical across all machines (the pass condition)."""
    module_hash_diverged: bool
    """True if any component's module hash differed across machines (the reason it is not the shipped identity)."""


def resolved_dtypes(state_dict: Mapping[str, Any]) -> dict[str, int]:
    """Return a ``str(dtype) -> tensor count`` histogram of a loaded module's state dict.

    Duck-typed on a ``.dtype`` attribute so it needs no torch import: non-tensor entries are ignored. This is
    the load-time variable that moves the module hash while leaving the content hash untouched.
    """
    counts: dict[str, int] = {}
    for value in state_dict.values():
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            continue
        key = str(dtype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compare_records(records: list[ComponentProbeRecord]) -> ProbeComparison:
    """Reduce several machines' probe records to a cross-machine agreement verdict.

    For each component kind seen in any record, groups the machines by content hash and (separately) by module
    hash. A single content-hash group across all machines confirms the shipped identity is reproducible; more
    than one module-hash group demonstrates the runtime key's device dependence. Records where a component was
    not extractable (a None hash) simply do not contribute that hash to the grouping.
    """
    labels = [record.label for record in records]
    kinds = sorted({kind for record in records for kind in record.components})
    per_component: list[ComponentComparison] = []
    for kind in kinds:
        content_by_value: dict[str, list[str]] = {}
        module_by_value: dict[str, list[str]] = {}
        dtypes_by_label: dict[str, dict[str, int]] = {}
        for record in records:
            probe = record.components.get(kind)
            if probe is None:
                continue
            if probe.content_hash is not None:
                content_by_value.setdefault(probe.content_hash, []).append(record.label)
            if probe.module_hash is not None:
                module_by_value.setdefault(probe.module_hash, []).append(record.label)
            dtypes_by_label[record.label] = probe.resolved_dtypes
        per_component.append(
            ComponentComparison(
                kind=kind,
                content_hash_by_value=content_by_value,
                module_hash_by_value=module_by_value,
                dtypes_by_label=dtypes_by_label,
                content_hash_stable=len(content_by_value) <= 1,
                module_hash_stable=len(module_by_value) <= 1,
            ),
        )
    content_hash_stable = bool(per_component) and all(item.content_hash_stable for item in per_component)
    module_hash_diverged = any(not item.module_hash_stable for item in per_component)
    return ProbeComparison(
        labels=labels,
        per_component=per_component,
        content_hash_stable=content_hash_stable,
        module_hash_diverged=module_hash_diverged,
    )


def _content_hash_for(checkpoint_path: Path, kind: ComponentKind) -> str | None:
    """Compute the torch-free embedded content hash for *kind*, or None when the file has no such component."""
    try:
        return hash_embedded_component_file(checkpoint_path, kind)
    except (NoComponentTensorsError, UnsupportedContainerError) as hash_error:
        logger.warning(f"No content hash for {kind} in {checkpoint_path.name}: {hash_error}")
        return None


def _module_probe(kind: ComponentKind, loaded: Mapping[str, Any]) -> tuple[str | None, dict[str, int], int]:
    """Return the module hash, dtype histogram and tensor count for a loaded component, or the empty probe.

    ``loaded`` maps the loader-return name (``vae``/``clip``) to the object ComfyUI returned. The component's
    weights hang off a fixed submodule attribute (:data:`_LOADED_MODULE_ATTR`), the same one the runtime
    adoption path hashes; when that submodule is absent the module could not be probed.
    """
    from hordelib.execution.shared_components import hash_state_dict

    owner_name, submodule_name = _LOADED_MODULE_ATTR[kind]
    owner = loaded.get(owner_name)
    module = getattr(owner, submodule_name, None)
    if module is None:
        return None, {}, 0
    state_dict = module.state_dict()
    return hash_state_dict(state_dict), resolved_dtypes(state_dict), len(state_dict)


def probe_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_name: str | None = None,
    label: str | None = None,
) -> ComponentProbeRecord:
    """Load a monolithic SD1.5/SDXL checkpoint and record both identities per component on this machine.

    Requires an initialised hordelib/ComfyUI environment (call :func:`hordelib.initialise` first) and the
    checkpoint file present. The content hash is read torch-free from the file; the module hash is taken from
    the ComfyUI-loaded modules, exactly as the runtime adoption path would.

    Args:
        checkpoint_path: Path to the monolithic ``.safetensors`` checkpoint.
        model_name: Human label for the model (defaults to the file stem).
        label: Machine identifier for later comparison (defaults to the hostname).

    Returns:
        The per-component probe record for this machine.
    """
    import comfy.sd  # imported lazily so the pure/compare paths stay torch-free
    import torch

    path = Path(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    logger.info(f"Loading {path.name} to probe component identities (device={device}).")
    with torch.no_grad():
        loaded_tuple = comfy.sd.load_checkpoint_guess_config(
            str(path),
            output_vae=True,
            output_clip=True,
            embedding_directory=None,
        )
    # load_checkpoint_guess_config returns (model, clip, vae, ...); the loader node relies on this ordering.
    loaded = {"clip": loaded_tuple[1], "vae": loaded_tuple[2]}

    components: dict[str, ComponentProbe] = {}
    for kind in _MONOLITHIC_KINDS:
        content_hash = _content_hash_for(path, kind)
        module_hash, dtypes, num_tensors = _module_probe(kind, loaded)
        components[kind.value] = ComponentProbe(
            kind=kind.value,
            content_hash=content_hash,
            module_hash=module_hash,
            resolved_dtypes=dtypes,
            num_tensors=num_tensors,
        )
        logger.info(
            f"{kind.value}: content={_short(content_hash)} module={_short(module_hash)} "
            f"dtypes={dtypes} tensors={num_tensors}",
        )

    return ComponentProbeRecord(
        label=label or socket.gethostname(),
        model_name=model_name or path.stem,
        file_name=path.name,
        file_size=path.stat().st_size,
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        device=device,
        gpu_name=gpu_name,
        components=components,
    )


def _short(value: str | None) -> str:
    """Abbreviate a hash for logging (or ``-`` when absent)."""
    return value[:12] if value else "-"


def _render_comparison(comparison: ProbeComparison) -> str:
    """Render a human-readable verdict of a cross-machine comparison."""
    lines = [
        f"Machines compared ({len(comparison.labels)}): {', '.join(comparison.labels)}",
        "",
    ]
    for item in comparison.per_component:
        content_state = "STABLE" if item.content_hash_stable else "UNSTABLE"
        module_state = "stable" if item.module_hash_stable else "diverged"
        lines.append(f"[{item.kind}] content hash: {content_state}   module hash: {module_state}")
        if not item.content_hash_stable:
            for value, who in item.content_hash_by_value.items():
                lines.append(f"    content {value[:16]} <- {', '.join(who)}")
        if not item.module_hash_stable:
            for value, who in item.module_hash_by_value.items():
                dtypes = "; ".join(f"{label}:{item.dtypes_by_label.get(label, {})}" for label in who)
                lines.append(f"    module  {value[:16]} <- {dtypes}")
    lines.append("")
    verdict = "PASS" if comparison.content_hash_stable else "FAIL"
    lines.append(f"Shipped identity (content hash) cross-machine reproducible: {verdict}")
    if comparison.module_hash_diverged:
        lines.append("Runtime module hash diverged across machines (expected: it folds device-dependent dtype).")
    return "\n".join(lines)


def _run_probe(args: argparse.Namespace) -> int:
    """Load a checkpoint, build the record, and write it as JSON to stdout or a file."""
    import hordelib

    hordelib.initialise(setup_logging=False)
    record = probe_checkpoint(args.checkpoint, model_name=args.model_name, label=args.label)
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        logger.info(f"Wrote probe record to {args.output}")
    else:
        print(payload)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Read several probe-record JSON files and print the cross-machine verdict."""
    records = [
        ComponentProbeRecord.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.records
    ]
    if len({record.label for record in records}) < 2:
        logger.warning("Fewer than two distinct machine labels: the comparison is only meaningful across machines.")
    comparison = compare_records(records)
    print(_render_comparison(comparison))
    return 0 if comparison.content_hash_stable else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``probe`` a checkpoint into a JSON record, or ``compare`` several records."""
    parser = argparse.ArgumentParser(description="Probe and compare cross-machine component identities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Load a checkpoint and emit its probe record as JSON.")
    probe_parser.add_argument("checkpoint", help="Path to a monolithic SD1.5/SDXL .safetensors checkpoint.")
    probe_parser.add_argument("--model-name", default=None, help="Model label (defaults to the file stem).")
    probe_parser.add_argument("--label", default=None, help="Machine label for comparison (defaults to hostname).")
    probe_parser.add_argument("--output", default=None, help="Write the JSON record here (defaults to stdout).")
    probe_parser.set_defaults(func=_run_probe)

    compare_parser = subparsers.add_parser("compare", help="Compare probe records from several machines.")
    compare_parser.add_argument("records", nargs="+", help="Probe-record JSON files to compare.")
    compare_parser.set_defaults(func=_run_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
