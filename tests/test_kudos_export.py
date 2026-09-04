"""Behavioural tests for the export stage: distillation, the acceptance gate, and the bundle.

The forward-pass test pins the served contract against arithmetic worked out by hand, so a change
to how the artifact is read is caught without a trained model in the way. The remaining tests drive
train into export on the same synthetic corpus the pipeline suite uses, with a search small enough
to run in CI, and check that what lands on disk is loadable with numpy alone and reproducible from
the artifact itself.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hordelib.kudos_training.export import (
    LAYER_COUNT,
    AcceptanceGateError,
    ExportConfig,
    ExportResult,
    build_model_golden_document,
    export,
    load_npz_weights,
    predict_seconds_npz,
)
from hordelib.kudos_training.manifest import CategoricalFeature, default_manifest
from hordelib.kudos_training.train import TrainConfig, train

# The synthetic corpus builders are shared with the pipeline suite rather than restated here, so a
# schema change to snapshot rows lands in one place.
from tests.test_kudos_training_pipeline import _snapshot_row, _write_snapshot_parquet

pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

_CORPUS_CELLS = (
    ("g1.fast", 10, 512, 1, 4.0),
    ("g1.slow", 50, 512, 1, 16.0),
    ("g2.big", 30, 1024, 1, 11.0),
    ("g3.batch", 30, 512, 4, 25.0),
)
"""Cell id, trajectory steps, square dimension, batch size and sampler-window seconds."""

_TEST_EXPORT_CONFIG = ExportConfig(
    trials=3,
    synthetic_samples_per_row=6,
    hidden_width_choices=(32, 64),
    learning_rate_range=(3e-3, 6e-3),
    min_epochs=600,
    max_epochs=900,
    early_stopping_patience=200,
    batch_size=64,
)
"""A search small enough for CI; the acceptance thresholds stay at their shipped defaults."""


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train the reference model once on a synthetic corpus, for every export test to distil from."""
    run_root = tmp_path_factory.mktemp("kudos-export")
    rows = []
    popped = 1000.0
    for replicate in range(20):
        for cell_id, steps, dimension, batch, window in _CORPUS_CELLS:
            rows.append(
                _snapshot_row(
                    job_id=f"job-{cell_id}-{replicate}",
                    cell_id=cell_id,
                    time_popped=popped,
                    trajectory_steps=steps,
                    width=dimension,
                    height=dimension,
                    n_images=batch,
                    replicate=replicate,
                    sampler_window_seconds=window + 0.2 * (replicate % 5),
                ),
            )
            popped += 30.0

    clean_path = run_root / "clean-export.parquet"
    _write_snapshot_parquet(clean_path, rows)
    run = train(clean_path, out_dir=run_root / "runs", config=TrainConfig(n_estimators=80, early_stopping_rounds=20))
    return run.run_dir, clean_path


@pytest.fixture(scope="module")
def exported(trained_run: tuple[Path, Path]) -> ExportResult:
    """Export the trained run once, so the bundle assertions all read the same artifact."""
    run_dir, clean_path = trained_run
    return export(run_dir, clean_path, config=_TEST_EXPORT_CONFIG)


def test_forward_pass_matches_a_hand_computed_network(tmp_path: Path) -> None:
    weights = {
        "w0": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float32),
        "b0": np.asarray([0.5, 1.0], dtype=np.float32),
        "w1": np.asarray([[2.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
        "b1": np.asarray([0.0, -1.0], dtype=np.float32),
        "w2": np.asarray([[1.0, 1.0], [0.5, -2.0]], dtype=np.float32),
        "b2": np.asarray([-1.0, 0.25], dtype=np.float32),
        "w3": np.asarray([[2.0, -1.0]], dtype=np.float32),
        "b3": np.asarray([0.5], dtype=np.float32),
    }
    vector = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    # relu([1.5, 0.0]) -> relu([3.0, -2.5]) -> relu([2.0, 1.75]) -> 2 * 2.0 - 1.75 + 0.5
    assert predict_seconds_npz(weights, vector) == pytest.approx(2.75)

    npz_path = tmp_path / "tiny.npz"
    np.savez(npz_path, **weights)
    assert predict_seconds_npz(load_npz_weights(npz_path), vector) == pytest.approx(2.75)


def test_export_produces_a_loadable_bundle_that_passes_the_gate(exported: ExportResult) -> None:
    assert exported.model_path.exists()
    assert exported.model_path.name.startswith("kudos-v22-")
    assert exported.median_ape <= _TEST_EXPORT_CONFIG.median_ape_threshold
    assert exported.p90_ape <= _TEST_EXPORT_CONFIG.p90_ape_threshold
    assert exported.held_out_rows > 0
    assert exported.basis_seconds > 0

    manifest = default_manifest()
    with np.load(exported.model_path) as loaded:
        assert sorted(loaded.files) == sorted(f"{prefix}{index}" for index in range(LAYER_COUNT) for prefix in "wb")
        weights = {key: loaded[key] for key in loaded.files}
    for index in range(LAYER_COUNT):
        assert weights[f"w{index}"].ndim == 2
        assert weights[f"b{index}"].shape == (weights[f"w{index}"].shape[0],)
    assert weights["w0"].shape[1] == manifest.vector_length()
    assert weights["w3"].shape[0] == 1

    metadata = json.loads(exported.metadata_path.read_text(encoding="utf-8"))
    assert metadata["manifest_version"] == manifest.manifest_version
    assert metadata["model_file"] == exported.model_path.name
    assert metadata["seeds"]["export"] == _TEST_EXPORT_CONFIG.seed
    assert metadata["acceptance"]["median_ape"] == exported.median_ape
    assert metadata["hpo"]["best_params"] == exported.best_params
    assert set(metadata["library_versions"]) >= {"numpy", "torch", "lightgbm", "optuna"}
    assert (exported.export_dir / "hpo.sqlite3").exists()


def test_gate_failure_raises_and_publishes_nothing(trained_run: tuple[Path, Path]) -> None:
    run_dir, clean_path = trained_run
    # The distilled net still has to predict a positive basis job: export refuses an artifact that
    # prices nothing before it ever reaches the accuracy gate, and the gate is what this test is
    # about. The thresholds, not the training budget, are what make the run unacceptable.
    impossible = ExportConfig(
        trials=1,
        synthetic_samples_per_row=2,
        hidden_width_choices=(16,),
        min_epochs=200,
        max_epochs=300,
        early_stopping_patience=50,
        batch_size=64,
        median_ape_threshold=1e-9,
        p90_ape_threshold=1e-9,
    )

    with pytest.raises(AcceptanceGateError) as raised:
        export(run_dir, clean_path, config=impossible)

    assert raised.value.median_ape > 0
    assert not list((run_dir / "export").glob("*.staging.npz"))


def test_golden_vectors_re_evaluate_from_the_artifact(exported: ExportResult) -> None:
    document = json.loads(exported.golden_vectors_path.read_text(encoding="utf-8"))
    manifest = default_manifest()

    sampled_samplers = {
        case["payload"].get("sampler_name") for case in document["cases"] if case["source"] == "vocabulary_sweep"
    }
    sampler_feature = next(
        feature
        for feature in manifest.features
        if isinstance(feature, CategoricalFeature) and feature.name == "sampler_name"
    )
    assert set(sampler_feature.vocabulary) <= sampled_samplers
    assert len(document["cases"]) >= 50
    assert any(case["user_price"] is not None for case in document["cases"])

    rebuilt = build_model_golden_document(
        load_npz_weights(exported.model_path),
        model_filename=exported.model_path.name,
        seed=_TEST_EXPORT_CONFIG.seed,
    )
    assert json.dumps(rebuilt, indent=2) + "\n" == exported.golden_vectors_path.read_text(encoding="utf-8")
