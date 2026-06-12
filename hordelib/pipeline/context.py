"""Resolved model context: facts about the requested model that pipeline selection needs."""

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from pydantic import BaseModel

FLUX_BASELINES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
    },
)


class ModelContext(BaseModel):
    """What the pipeline layer knows about the model a payload requests."""

    horde_model_name: str
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None = None

    @property
    def is_flux(self) -> bool:
        return self.baseline in FLUX_BASELINES

    @property
    def is_cascade(self) -> bool:
        return self.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade

    @property
    def is_qwen(self) -> bool:
        return self.baseline == KNOWN_IMAGE_GENERATION_BASELINE.qwen_image
