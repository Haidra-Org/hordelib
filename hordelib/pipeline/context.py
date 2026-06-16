"""Resolved model context: facts about the requested model that selection/materialization need."""

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from pydantic import BaseModel, ConfigDict

from hordelib.pipeline.patches import ResolvedLora


class PostProcessingContext(BaseModel):
    """Resolved facts for a post-processing payload: where the model lives on disk.

    Post-processing selection has no baseline; the payload type and model name are the only
    selection keys, which is why this is a separate, lighter context than :class:`ModelContext`.
    """

    model_name: str
    model_file: str
    """The on-disk filename to feed the graph's model loader."""


class ModelContext(BaseModel):
    """What the pipeline layer knows about the model a payload requests.

    The baseline fields drive pipeline *selection*; the file/lora fields are *materialization*
    facts filled in by :mod:`hordelib.pipeline.resolution` (selection-only callers may leave
    them at their defaults).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    horde_model_name: str
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None = None

    main_file: str | None = None
    """The checkpoint/unet filename for the graph's model loader."""
    extra_files: dict[str, str] = {}
    """Additional ``file_type -> file_path`` entries (e.g. stable cascade stage b/c)."""
    is_inpainting_model: bool = False
    resolved_loras: list[ResolvedLora] = []
    """LoRAs that have been validated/downloaded, in application order."""
    will_load_loras: bool = False
    """Whether the model loader should prepare for LoRA application."""
