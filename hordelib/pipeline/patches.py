"""First-class graph patch operations.

These are the dynamic graph mutations previously inlined in
``HordeLib._final_pipeline_adjustments``, as pure functions over API-format graph dicts so
they can be unit-tested without a GPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from hordelib.execution.graph_utils import GraphDict, reconnect_input


@dataclass(frozen=True)
class ResolvedLora:
    """A LoRA that has been validated/downloaded and resolved to an on-disk filename."""

    filename: str
    strength_model: float
    strength_clip: float


def insert_lora_chain(
    graph: GraphDict,
    loras: Sequence[ResolvedLora],
    *,
    flux: bool = False,
) -> None:
    """Inject a chain of HordeLoraLoader nodes between the model loader and its consumers.

    The first LoRA connects to ``model_loader``; each subsequent LoRA chains from the previous
    one. The last LoRA replaces the model/clip sources of the samplers and clip_skip (or, for
    Flux pipelines, of ``cfg_guider``/``basic_scheduler``). Targets absent from the graph are
    skipped, matching the variant pipelines that lack them (e.g. no ``upscale_sampler``).

    Args:
        graph: The pipeline graph to mutate.
        loras: The LoRAs to chain, in application order.
        flux: Whether this is a Flux pipeline (different model consumers).
    """
    if not loras:
        return

    for index, lora in enumerate(loras):
        source = "model_loader" if index == 0 else f"lora_{index - 1}"
        graph[f"lora_{index}"] = {
            "inputs": {
                "model": [source, 0],
                "clip": [source, 1],
                "lora_name": lora.filename,
                "strength_model": lora.strength_model,
                "strength_clip": lora.strength_clip,
            },
            "class_type": "HordeLoraLoader",
            "_meta": {"title": f"lora_{index}"},
        }

    last = f"lora_{len(loras) - 1}"
    if flux:
        reconnect_input(graph, "cfg_guider.model", last)
        reconnect_input(graph, "basic_scheduler.model", last)
    else:
        reconnect_input(graph, "sampler.model", last)
        reconnect_input(graph, "upscale_sampler.model", last)
        reconnect_input(graph, "clip_skip.clip", last)
