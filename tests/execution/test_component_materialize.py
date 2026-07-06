"""Tests for canonical-component source selection and materialisation data types (GPU-free).

The comfy load paths in ``materialize()`` need a real backend and model files and are rig-validated; the
source-selection logic, the data types, and the module's lazy-import contract are exercised here.
"""

from __future__ import annotations

import pytest
from horde_model_reference.canonical_components import CanonicalComponent, CanonicalComponentSource
from horde_model_reference.component_hash import ComponentKind

from hordelib.execution import component_materialize
from hordelib.execution.component_materialize import (
    MaterializedComponent,
    NoMaterializationSourceError,
    _choose_source,
    _require_file_name,
)


def _source(
    *,
    embedded: bool,
    kind: ComponentKind = ComponentKind.VAE,
    file_name: str | None = "ae.safetensors",
    model_name: str = "carrier_model",
) -> CanonicalComponentSource:
    """Build a canonical component source; an embedded source carries no standalone file."""
    return CanonicalComponentSource(
        model_name=model_name,
        kind=kind,
        embedded=embedded,
        file_name=None if embedded else file_name,
        file_purpose=None if embedded else "vae",
    )


def _component(
    sources: list[CanonicalComponentSource],
    *,
    kind: ComponentKind = ComponentKind.VAE,
    content_hash: str = "a" * 64,
) -> CanonicalComponent:
    """Build a canonical component wrapping the given sources."""
    return CanonicalComponent(
        content_hash=content_hash,
        kind=kind,
        shared_by_model_count=max(1, len(sources)),
        sources=sources,
    )


class TestChooseSource:
    """A standalone source is preferred; embedded is the fallback; no sources is an error."""

    def test_prefers_standalone_over_embedded(self) -> None:
        embedded = _source(embedded=True)
        standalone = _source(embedded=False)
        source, is_embedded = _choose_source(_component([embedded, standalone]))
        assert source is standalone
        assert is_embedded is False

    def test_falls_back_to_embedded(self) -> None:
        embedded = _source(embedded=True)
        source, is_embedded = _choose_source(_component([embedded]))
        assert source is embedded
        assert is_embedded is True

    def test_no_sources_raises(self) -> None:
        with pytest.raises(NoMaterializationSourceError):
            _choose_source(_component([]))


class TestDataTypes:
    """The materialisation data types and label mapping."""

    def test_materialized_component_holds_fields(self) -> None:
        component = MaterializedComponent(module="MODULE", content_hash="h", label="vae")
        assert (component.module, component.content_hash, component.label) == ("MODULE", "h", "vae")

    def test_label_mapping_matches_publish_labels(self) -> None:
        assert component_materialize._LABEL_FOR_KIND[ComponentKind.VAE] == "vae"
        assert component_materialize._LABEL_FOR_KIND[ComponentKind.TEXT_ENCODERS] == "clip"


class TestRequireFileName:
    """A standalone source must carry a file_name, else materialisation cannot resolve a path."""

    def test_returns_file_name(self) -> None:
        assert _require_file_name(_source(embedded=False, file_name="x.safetensors")) == "x.safetensors"

    def test_missing_file_name_raises(self) -> None:
        with pytest.raises(NoMaterializationSourceError):
            _require_file_name(_source(embedded=True))


def test_module_import_keeps_torch_and_comfy_lazy() -> None:
    """The module must bind no top-level torch/comfy import (they load only inside the comfy path)."""
    assert not hasattr(component_materialize, "torch")
    assert not hasattr(component_materialize, "comfy")
