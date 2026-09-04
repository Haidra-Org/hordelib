"""The versioned kudos feature manifest and the encoder it drives.

The manifest is the only definition of the kudos model's input contract: the ordered feature list,
each feature's normalization constant and default, the categorical vocabularies, and the
unknown-collapse target every vocabulary uses. The trainer, the exporter, the golden-vector
generator and the server-side evaluator all encode through this module, so one payload encodes to
one vector wherever it is priced.

A published manifest is immutable. hordelib's live vocabularies (:mod:`hordelib.pipeline.constants`)
inform the authoring of a new manifest revision and are checked against it by
``tests/test_kudos_feature_manifest.py``, but they never alter the encoding of an existing revision:
a layout that shifted underneath a trained model would reprice every job silently. A vocabulary that
has to grow is a new manifest file and a new model.

This module imports numpy and the standard library only. It is consumed by processes (the AI-Horde
server, the worker) that must not pay for torch to price a job.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from horde_sdk.generation_parameters.image.constraints_document import SAMPLER_CONSTRAINTS_DOCUMENT_SCHEMA_VERSION
from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion
from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_FILENAME = "kudos_feature_manifest_v22.json"
"""Filename of the manifest revision this package ships."""

DEFAULT_MANIFEST_PATH = Path(__file__).parent / MANIFEST_FILENAME
"""Location of the shipped manifest. The AI-Horde tree carries a byte-identical copy."""

VECTOR_DTYPE = np.float32
"""Encoded vectors are float32, the precision the served model evaluates in."""


class SamplerSemanticsReference(BaseModel):
    """Represents the SDK sampler semantics against which a feature manifest was authored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    horde_sdk_version: str = Field(min_length=1)
    """Released SDK version that supplied the sampler vocabulary and work semantics."""

    constraints_document_schema_version: Literal["1.0"] = SAMPLER_CONSTRAINTS_DOCUMENT_SCHEMA_VERSION
    """JSON schema version of the referenced sampler-constraints document."""

    constraints_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """SHA-256 of the canonical committed sampler-constraints JSON artifact."""

    execution_contract_version: SamplerExecutionContractVersion
    """Execution behavior required for observations whose runtime depends on that contract."""


class DerivedQuantity(StrEnum):
    """A float feature computed from other resolved features rather than read from a payload key.

    Membership is closed on purpose: a manifest names a derivation this module implements, rather
    than carrying an expression that each consumer would have to evaluate the same way.
    """

    MEGAPIXELS = auto()
    """``width * height / 1e6``, in payload units, from the resolved width and height features."""


class _FeatureBase(BaseModel):
    """Fields shared by every feature entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """Feature name. Unique within a manifest and stable across a manifest's lifetime."""

    payload_keys: tuple[str, ...] = ()
    """Payload keys read in order; the first one present and non-null supplies the value.

    Several keys are listed where a field has more than one spelling in the wild (the v21 payload
    calls the step count ``ddim_steps``, the worker stats record calls it ``steps``).
    """

    description: str = ""
    """What the feature measures, for a reader of the manifest file."""


class FloatFeature(_FeatureBase):
    """A single-slot numeric feature: resolve, clamp in payload units, then divide."""

    kind: Literal["float"] = "float"

    derived: DerivedQuantity | None = None
    """When set, the value is computed from earlier features instead of read from the payload."""

    bool_as_float: bool = False
    """Encode the resolved value's truthiness as 1.0 or 0.0 rather than coercing it to a number."""

    divisor: float = 1.0
    """Normalization constant. The encoded slot is the clamped payload-unit value over this."""

    default: float = 0.0
    """Used when no payload key is present and no fallback resolves, and when coercion fails."""

    fallback_feature: str | None = None
    """Name of an earlier feature whose resolved value substitutes for an absent payload value.

    ``control_strength`` falls back to ``denoising_strength`` this way, as the v21 encoder did.
    """

    clamp_min: float | None = None
    """Lower bound, in payload units, applied before the divisor."""

    clamp_max: float | None = None
    """Upper bound, in payload units, applied before the divisor."""

    @property
    def width(self) -> int:
        """Number of vector slots this feature occupies."""
        return 1

    def slot_names(self) -> tuple[str, ...]:
        """Return the per-slot labels this feature contributes to the vector layout."""
        return (self.name,)

    @model_validator(mode="after")
    def _check_source(self) -> "FloatFeature":
        if self.derived is not None and self.payload_keys:
            raise ValueError(f"feature {self.name!r} is derived and must not declare payload_keys")
        if self.derived is None and not self.payload_keys:
            raise ValueError(f"feature {self.name!r} declares neither payload_keys nor a derivation")
        if self.divisor == 0:
            raise ValueError(f"feature {self.name!r} has a zero divisor")
        if self.clamp_min is not None and self.clamp_max is not None and self.clamp_min > self.clamp_max:
            raise ValueError(f"feature {self.name!r} has clamp_min above clamp_max")
        return self


class _VocabularyFeature(_FeatureBase):
    """Fields shared by the two vocabulary-backed feature kinds."""

    vocabulary: tuple[str, ...]
    """The encoded values, in vector order. Frozen for the life of the manifest revision."""

    value_aliases: dict[str, str] = Field(default_factory=dict)
    """Values folded onto a vocabulary entry before lookup, as a named, reviewable decision.

    Distinct from the unknown-collapse target: an alias records that two spellings are the same
    thing to the model, whereas a collapse records that an unrecognised value is being priced as
    something else.
    """

    @property
    def width(self) -> int:
        """Number of vector slots this feature occupies."""
        return len(self.vocabulary)

    def slot_names(self) -> tuple[str, ...]:
        """Return the per-slot labels this feature contributes to the vector layout."""
        return tuple(f"{self.name}={value}" for value in self.vocabulary)

    @model_validator(mode="after")
    def _check_vocabulary(self) -> "_VocabularyFeature":
        if not self.vocabulary:
            raise ValueError(f"feature {self.name!r} has an empty vocabulary")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError(f"feature {self.name!r} has a duplicate vocabulary entry")
        for alias, target in self.value_aliases.items():
            if target not in self.vocabulary:
                raise ValueError(f"feature {self.name!r} aliases {alias!r} onto non-vocabulary {target!r}")
            if alias in self.vocabulary:
                raise ValueError(f"feature {self.name!r} aliases {alias!r}, which is itself in the vocabulary")
        return self


class BooleanFallback(BaseModel):
    """A legacy boolean payload key standing in for an absent categorical value.

    Horde payloads predate several of the fields this manifest reads, and the older spelling is
    still what a live request carries: a request selects its sigma schedule through a ``karras``
    boolean, which the SDK defaults to true, rather than through a ``scheduler`` name. Encoding
    such a payload from the categorical's plain default would price ``karras: false`` as karras.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_key: str
    """The boolean key consulted when the categorical's own keys are absent or empty."""

    when_true: str
    """Vocabulary entry encoded when the boolean is truthy."""

    when_false: str
    """Vocabulary entry encoded when the boolean is falsy."""


class CategoricalFeature(_VocabularyFeature):
    """A one-hot feature over a closed vocabulary, with an explicit unknown-collapse target."""

    kind: Literal["categorical"] = "categorical"

    boolean_fallback: BooleanFallback | None = None
    """Legacy boolean key consulted when this feature's own payload keys carry nothing.

    Resolution order is: a present, non-empty value under :attr:`payload_keys` (which then faces
    vocabulary collapse like any other), then this boolean, then :attr:`default`.
    """

    unknown_collapse: str
    """The vocabulary entry an unrecognised value is priced as.

    Naming it in the manifest is the point: collapsing a genuinely slower new sampler onto a fast
    one is then a visible decision carried by a manifest revision, not an accident of encoding.
    """

    default: str
    """The vocabulary entry used when the payload carries no value for this feature."""

    @model_validator(mode="after")
    def _check_targets(self) -> "CategoricalFeature":
        if self.unknown_collapse not in self.vocabulary:
            raise ValueError(f"feature {self.name!r} collapses onto non-vocabulary {self.unknown_collapse!r}")
        if self.default not in self.vocabulary:
            raise ValueError(f"feature {self.name!r} defaults to non-vocabulary {self.default!r}")
        if self.boolean_fallback is not None:
            for target in (self.boolean_fallback.when_true, self.boolean_fallback.when_false):
                if target not in self.vocabulary:
                    raise ValueError(f"feature {self.name!r} falls back onto non-vocabulary {target!r}")
        return self


class MultiHotFeature(_VocabularyFeature):
    """A multi-hot feature over a closed vocabulary, one slot per entry, counting occurrences.

    A value outside the vocabulary is dropped rather than collapsed, because there is no honest
    entry to fold an unknown post-processor onto. The drop is counted into
    :attr:`EncodingResult.dropped_unknown` so a caller can see it happen.
    """

    kind: Literal["multihot"] = "multihot"


Feature = Annotated[FloatFeature | CategoricalFeature | MultiHotFeature, Field(discriminator="kind")]


@dataclass(frozen=True)
class EncodingResult:
    """One encoded payload and what the encoder had to do to it."""

    vector: np.ndarray
    """The encoded feature vector, in manifest order, of length :meth:`KudosFeatureManifest.vector_length`."""

    collapsed: dict[str, str] = field(default_factory=dict)
    """Categorical feature name to the vocabulary entry its unrecognised value was priced as."""

    dropped_unknown: dict[str, int] = field(default_factory=dict)
    """Multi-hot feature name to how many unrecognised entries were dropped from it."""

    @property
    def has_unknowns(self) -> bool:
        """Whether any value was collapsed or dropped while encoding."""
        return bool(self.collapsed or self.dropped_unknown)


class KudosFeatureManifest(BaseModel):
    """An ordered, versioned feature contract plus the encoder that realizes it.

    The vector layout is exactly the manifest order: each feature contributes its slots
    consecutively, floats one slot and vocabulary features one slot per entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str
    """Revision identifier, e.g. ``v22``. Part of every model card and every exported artifact."""

    sampler_semantics: SamplerSemanticsReference
    """SDK vocabulary, constraint artifact, and execution contract used to interpret sampler inputs."""

    target: str
    """The measured duration the model trained against this manifest predicts."""

    basis_payload: dict[str, Any]
    """The reference job used to anchor prices across machines, in payload form."""

    features: tuple[Feature, ...]
    """The features, in vector order."""

    def content_sha256(self) -> str:
        """Hash of the manifest's content, independent of file formatting.

        Two manifests that encode identically hash identically, whatever their whitespace or line
        endings, so a corpus definition can carry this and an assembler can refuse rows encoded
        under a manifest other than the one being trained against.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _check_features(self) -> "KudosFeatureManifest":
        seen: set[str] = set()
        for feature in self.features:
            if feature.name in seen:
                raise ValueError(f"duplicate feature name {feature.name!r}")
            if isinstance(feature, FloatFeature):
                # Both back-references read a value the encoder has already resolved, so a manifest
                # may only point backwards; a forward reference would encode a default silently.
                if feature.fallback_feature is not None and feature.fallback_feature not in seen:
                    raise ValueError(
                        f"feature {feature.name!r} falls back to {feature.fallback_feature!r}, "
                        "which does not precede it",
                    )
                if feature.derived is DerivedQuantity.MEGAPIXELS and not {"width", "height"} <= seen:
                    raise ValueError(f"feature {feature.name!r} needs width and height to precede it")
            seen.add(feature.name)
        if not seen:
            raise ValueError("manifest declares no features")
        return self

    def feature_names(self) -> tuple[str, ...]:
        """Return the feature names in vector order."""
        return tuple(feature.name for feature in self.features)

    def slot_names(self) -> tuple[str, ...]:
        """Return one label per vector slot, in vector order."""
        return tuple(name for feature in self.features for name in feature.slot_names())

    def vector_length(self) -> int:
        """Return the number of slots an encoded vector has."""
        return sum(feature.width for feature in self.features)

    def encode(self, payload: dict[str, Any]) -> EncodingResult:
        """Encode *payload* into a feature vector, reporting collapsed and dropped values.

        Encoding never raises on bad input: a value that cannot be coerced falls back to the
        feature's default, an unrecognised categorical is collapsed onto the manifest's named
        target, and an unrecognised multi-hot entry is dropped. Every such event is recorded on
        the result so a caller can label the row instead of trusting it blindly.

        Args:
            payload: A horde job payload, or a stats record carrying the same fields.

        Returns:
            The vector and the record of what was collapsed or dropped.
        """
        vector = np.zeros(self.vector_length(), dtype=VECTOR_DTYPE)
        collapsed: dict[str, str] = {}
        dropped_unknown: dict[str, int] = {}
        resolved: dict[str, float] = {}

        offset = 0
        for feature in self.features:
            if isinstance(feature, FloatFeature):
                value = self._resolve_float(feature, payload, resolved)
                resolved[feature.name] = value
                vector[offset] = value / feature.divisor
            elif isinstance(feature, CategoricalFeature):
                value_name = self._resolve_categorical(feature, payload, collapsed)
                vector[offset + feature.vocabulary.index(value_name)] = 1.0
            else:
                dropped = self._fill_multihot(feature, payload, vector, offset)
                if dropped:
                    dropped_unknown[feature.name] = dropped
            offset += feature.width

        return EncodingResult(vector=vector, collapsed=collapsed, dropped_unknown=dropped_unknown)

    def to_vector(self, payload: dict[str, Any]) -> np.ndarray:
        """Encode *payload* and return only the vector.

        Use :meth:`encode` where the collapse and drop record matters, which is anywhere a row is
        being admitted into training data.
        """
        return self.encode(payload).vector

    def _resolve_float(
        self,
        feature: FloatFeature,
        payload: dict[str, Any],
        resolved: dict[str, float],
    ) -> float:
        """Return the clamped payload-unit value of *feature*, before its divisor is applied."""
        raw: Any = None
        if feature.derived is DerivedQuantity.MEGAPIXELS:
            raw = resolved["width"] * resolved["height"] / 1e6
        else:
            raw = _first_present(payload, feature.payload_keys)
            if raw is None and feature.fallback_feature is not None:
                raw = resolved[feature.fallback_feature]

        if feature.bool_as_float:
            if raw is None:
                return 1.0 if feature.default else 0.0
            return 1.0 if raw else 0.0

        if raw is None:
            value = feature.default
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = feature.default
            if not np.isfinite(value):
                value = feature.default

        if feature.clamp_min is not None:
            value = max(value, feature.clamp_min)
        if feature.clamp_max is not None:
            value = min(value, feature.clamp_max)
        return value

    def _resolve_categorical(
        self,
        feature: CategoricalFeature,
        payload: dict[str, Any],
        collapsed: dict[str, str],
    ) -> str:
        """Return the vocabulary entry *feature* encodes for *payload*, recording any collapse.

        A named value wins outright; an absent or empty one falls to the feature's legacy boolean
        (where it declares one) and then to its default. Only a named value can be collapsed:
        the boolean and the default resolve to vocabulary entries by construction.
        """
        raw = _first_present(payload, feature.payload_keys)
        if raw is None or raw == "":
            if feature.boolean_fallback is not None:
                flag = payload.get(feature.boolean_fallback.payload_key)
                if flag is not None:
                    return feature.boolean_fallback.when_true if flag else feature.boolean_fallback.when_false
            return feature.default

        value = raw if isinstance(raw, str) else str(raw)
        value = feature.value_aliases.get(value, value)
        if value not in feature.vocabulary:
            collapsed[feature.name] = feature.unknown_collapse
            return feature.unknown_collapse
        return value

    def _fill_multihot(
        self,
        feature: MultiHotFeature,
        payload: dict[str, Any],
        vector: np.ndarray,
        offset: int,
    ) -> int:
        """Write *feature*'s slots into *vector* and return how many entries were dropped."""
        raw = _first_present(payload, feature.payload_keys)
        if raw is None:
            return 0
        entries = [raw] if isinstance(raw, str) else list(raw)

        dropped = 0
        for entry in entries:
            value = entry if isinstance(entry, str) else str(entry)
            value = feature.value_aliases.get(value, value)
            if value in feature.vocabulary:
                vector[offset + feature.vocabulary.index(value)] += 1.0
            else:
                dropped += 1
        return dropped


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the value of the first key in *keys* that is present and non-null, else ``None``."""
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def load_manifest(path: str | Path | None = None) -> KudosFeatureManifest:
    """Load and validate a feature manifest.

    Args:
        path: Manifest file to read. Defaults to the revision shipped with this package.

    Returns:
        The validated manifest.
    """
    resolved = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    return KudosFeatureManifest.model_validate_json(resolved.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def default_manifest() -> KudosFeatureManifest:
    """Return the manifest revision shipped with this package, parsed once per process."""
    return load_manifest()


def to_vector(payload: dict[str, Any]) -> np.ndarray:
    """Encode *payload* against the shipped manifest revision.

    This is the module-level convenience the server-side evaluator and the trainer both call; it
    exists so no consumer has to know where the manifest file lives.
    """
    return default_manifest().to_vector(payload)


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "MANIFEST_FILENAME",
    "VECTOR_DTYPE",
    "BooleanFallback",
    "CategoricalFeature",
    "DerivedQuantity",
    "EncodingResult",
    "Feature",
    "FloatFeature",
    "KudosFeatureManifest",
    "MultiHotFeature",
    "SamplerSemanticsReference",
    "default_manifest",
    "load_manifest",
    "to_vector",
]
