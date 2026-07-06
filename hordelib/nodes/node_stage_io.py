"""ComfyUI nodes that cut a generation graph at a disaggregation boundary.

A stage entry point (see :mod:`hordelib.execution.interface` / ``HordeLib``) builds a minimal
graph ending at one of the *Output* nodes here and points ``execute_outputs`` at it, so the
executor runs only that stage's ancestor subgraph and hands the serialized intermediate back
through ``history_result`` (the exact path ``HordeImageOutput`` uses for image bytes). The
matching *Input* node injects a received intermediate blob back into a downstream stage's graph,
mirroring how ``HordeImageLoader`` injects a raw PIL object via a sentinel input type.

The blobs are produced/consumed by :mod:`hordelib.execution.stage_payloads`; they are byte-for-byte
the tensors a monolithic run would have passed in memory, so a disaggregated run reproduces the
monolithic image at a fixed seed.
"""

from io import BytesIO

from loguru import logger

from hordelib.execution.results import UI_ENTRY_DATA_KEY, UI_ENTRY_TYPE_KEY
from hordelib.execution.stage_payloads import (
    CONDITIONING_TYPE,
    LATENT_TYPE,
    deserialize_conditioning,
    deserialize_latent,
    serialize_conditioning,
    serialize_latent,
)

# Sentinel input types: unknown to ComfyUI, so the entry point's raw blob value is passed straight
# through to the node (the mechanism HordeImageLoader relies on with "<PIL Instance>").
CONDITIONING_BYTES_TYPE = "<CONDITIONING_BYTES>"
LATENT_BYTES_TYPE = "<LATENT_BYTES>"


def _target_device() -> str:
    """The device injected intermediates are materialized on (the active inference device)."""
    import comfy.model_management as mm

    return str(mm.get_torch_device())


class HordeConditioningOutput:
    """Serialize the graph's CONDITIONING and hand it back as an output artifact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ()
    FUNCTION = "get_conditioning"
    OUTPUT_NODE = True
    CATEGORY = "horde/stage"

    def get_conditioning(self, conditioning):
        blob = serialize_conditioning(conditioning)
        logger.debug("stage.conditioning_output: bytes={}", len(blob))
        entry = {UI_ENTRY_DATA_KEY: BytesIO(blob), UI_ENTRY_TYPE_KEY: CONDITIONING_TYPE}
        return {"ui": {"conditioning": [entry]}}


class HordeConditioningInput:
    """Inject a received CONDITIONING blob back into a downstream stage's graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning_bytes": (CONDITIONING_BYTES_TYPE,)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_conditioning"
    CATEGORY = "horde/stage"

    def load_conditioning(self, conditioning_bytes):
        if not isinstance(conditioning_bytes, (bytes, bytearray)):
            raise ValueError(f"HordeConditioningInput expected bytes, got {type(conditioning_bytes).__name__}")
        return (deserialize_conditioning(bytes(conditioning_bytes), device=_target_device()),)


class HordeLatentOutput:
    """Serialize the graph's LATENT and hand it back as an output artifact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "get_latent"
    OUTPUT_NODE = True
    CATEGORY = "horde/stage"

    def get_latent(self, samples):
        blob = serialize_latent(samples)
        logger.debug("stage.latent_output: bytes={}", len(blob))
        entry = {UI_ENTRY_DATA_KEY: BytesIO(blob), UI_ENTRY_TYPE_KEY: LATENT_TYPE}
        return {"ui": {"latent": [entry]}}


class HordeLatentInput:
    """Inject a received LATENT blob back into a downstream stage's graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_bytes": (LATENT_BYTES_TYPE,)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "horde/stage"

    def load_latent(self, latent_bytes):
        if not isinstance(latent_bytes, (bytes, bytearray)):
            raise ValueError(f"HordeLatentInput expected bytes, got {type(latent_bytes).__name__}")
        return (deserialize_latent(bytes(latent_bytes), device=_target_device()),)


NODE_CLASS_MAPPINGS = {
    "HordeConditioningOutput": HordeConditioningOutput,
    "HordeConditioningInput": HordeConditioningInput,
    "HordeLatentOutput": HordeLatentOutput,
    "HordeLatentInput": HordeLatentInput,
}
