"""Snapshot-row to manifest-payload bridging.

The feature manifest encodes horde job payloads; a snapshot row is a flattened stats record with
labels. This module maps a row back into payload form so the manifest is the only encoder anywhere
in the pipeline, and turns a snapshot frame into the feature matrix the trainer and the
residual-pruning sanitation rule both consume.
"""

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from hordelib.kudos_training.manifest import KudosFeatureManifest

if TYPE_CHECKING:
    import pandas as pd

_SOURCE_MODES_WITH_IMAGE = ("img2img", "inpainting")
"""Source-processing modes that carry a source image; inpainting additionally carries a mask."""


def row_to_payload(row: Mapping[Hashable, Any]) -> dict[str, Any]:
    """Convert one snapshot row into the payload form the feature manifest encodes.

    Args:
        row: A snapshot row as a plain mapping (one record of the snapshot frame).

    Returns:
        A payload dictionary keyed the way the manifest's ``payload_keys`` expect.
    """
    source_processing = row.get("source_processing")
    payload: dict[str, Any] = {
        "width": row["width"],
        "height": row["height"],
        "steps": row["trajectory_steps"],
        "cfg_scale": row["cfg_scale"],
        "denoising_strength": row["denoising_strength"],
        "hires_fix": row["hires_fix"],
        "source_image": source_processing in _SOURCE_MODES_WITH_IMAGE,
        "source_mask": source_processing == "inpainting",
        "n_images": row["n_images"],
        "loras_count": row["loras_count"],
        "tis_count": row["tis_count"],
        "sampler_name": row["sampler_name"],
        "scheduler": row["scheduler"],
        "baseline": row["baseline"],
        "control_type": row["control_type"],
        "source_processing": source_processing,
        "post_processing": list(row["post_processing"]) if row["post_processing"] is not None else [],
    }
    return payload


def frame_to_matrix(frame: "pd.DataFrame", manifest: KudosFeatureManifest) -> np.ndarray:
    """Encode every row of a snapshot frame into the manifest's feature matrix.

    Args:
        frame: Snapshot rows (any stage's output).
        manifest: The feature manifest revision to encode against.

    Returns:
        A float32 matrix of shape ``(len(frame), manifest.vector_length())``, in frame order.
    """
    matrix = np.zeros((len(frame), manifest.vector_length()), dtype=np.float32)
    for index, row in enumerate(frame.to_dict(orient="records")):
        matrix[index] = manifest.to_vector(row_to_payload(row))
    return matrix


__all__ = ["frame_to_matrix", "row_to_payload"]
