"""Typed pipeline-run results and the modality-agnostic output-entry collection.

After a run, ComfyUI's ``PromptExecutor`` exposes ``success``, ``status_messages`` and
``history_result`` (``{"outputs": {node_title: ui_dict}, "meta": ...}``). This module turns
those attributes into a :class:`PipelineRunResult` and flattens the per-node ui dicts into
the bridge's wire-shaped artifact entries.

Collection is keyed by the output node, not by artifact type: every list-valued key of a
node's ui dict is walked (``"images"`` today, ``"audio"`` or others later), so a new output
modality needs a new output node but no change to this collection path. Entries must carry
an in-memory ``BytesIO`` under :data:`UI_ENTRY_DATA_KEY`, the contract hordelib output nodes
implement (see ``hordelib/nodes/node_image_output.py``).

This module must remain importable before ``hordelib.initialise()``: it never imports ComfyUI.
"""

import io
import typing

from loguru import logger
from pydantic import BaseModel, ConfigDict

from hordelib.execution.comfy_events import ExecutionErrorEvent

UI_ENTRY_DATA_KEY: typing.Final[str] = "imagedata"
"""The ui-entry key carrying the artifact bytes (a ``BytesIO``).

Historical name from the image-only era; audio/video output nodes reuse it for their encoded
bytes so the whole wire path stays modality-agnostic.
"""

UI_ENTRY_TYPE_KEY: typing.Final[str] = "type"
"""The ui-entry key naming the encoded format (e.g. ``"PNG"``)."""

UI_ENTRY_SOURCE_NODE_KEY: typing.Final[str] = "source_node"
"""The key collection adds to each entry, carrying the producing node's graph title."""

_EXPECTED_UI_ENTRY_KEYS: typing.Final[frozenset[str]] = frozenset({UI_ENTRY_DATA_KEY, UI_ENTRY_TYPE_KEY})


class PipelineRunResult(BaseModel):
    """Represents the outcome of one executed pipeline graph.

    ``entries`` holds the wire-shaped artifact dicts
    (``{"imagedata": BytesIO, "type": "PNG", "source_node": <node title>}``) consumed by
    :meth:`hordelib.execution.in_process.InProcessComfyBackend._to_artifacts` and, through the
    compatibility surface, by direct ``run_pipeline`` callers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    entries: list[dict[str, typing.Any]] = []
    error: ExecutionErrorEvent | None = None

    @property
    def produced_nodes(self) -> set[str]:
        """The set of node titles that produced at least one collected entry."""
        return {
            entry[UI_ENTRY_SOURCE_NODE_KEY]
            for entry in self.entries
            if isinstance(entry.get(UI_ENTRY_SOURCE_NODE_KEY), str)
        }


def collect_output_entries(history_outputs: dict[str, typing.Any]) -> list[dict[str, typing.Any]]:
    """Flatten the executor's per-node ui outputs into tagged artifact entries.

    Walks every list-valued key of every node's ui dict and keeps the entries that satisfy
    the bridge's artifact contract (a ``BytesIO`` under :data:`UI_ENTRY_DATA_KEY`). Malformed
    entries are logged and skipped rather than failing the run; the declared-output check in
    ``run_pipeline`` is what turns a missing artifact into a hard error.

    Args:
        history_outputs: The ``history_result["outputs"]`` mapping of node title to ui dict.

    Returns:
        list[dict[str, typing.Any]]: Wire-shaped entries, each tagged with its source node,
        in node-then-list order.
    """
    entries: list[dict[str, typing.Any]] = []
    for node_title, ui_dict in history_outputs.items():
        if not isinstance(ui_dict, dict):
            logger.warning(
                "Output node produced a non-dict ui payload; skipping",
                source_node=node_title,
                payload_type=type(ui_dict).__name__,
            )
            continue
        for ui_key, ui_value in ui_dict.items():
            if not isinstance(ui_value, list):
                continue
            for raw_entry in ui_value:
                if not isinstance(raw_entry, dict):
                    logger.error(
                        "Received non dict output entry from comfyui",
                        source_node=node_title,
                        ui_key=ui_key,
                        entry_type=type(raw_entry).__name__,
                    )
                    continue
                if not isinstance(raw_entry.get(UI_ENTRY_DATA_KEY), io.BytesIO):
                    logger.error(
                        "Received output entry without in-memory artifact bytes from comfyui",
                        source_node=node_title,
                        ui_key=ui_key,
                        keys=list(raw_entry),
                    )
                    continue
                unexpected_keys = set(raw_entry) - _EXPECTED_UI_ENTRY_KEYS
                if unexpected_keys:
                    logger.error(
                        "Received unexpected output entry keys from comfyui",
                        source_node=node_title,
                        ui_key=ui_key,
                        keys=sorted(unexpected_keys),
                    )
                collected = dict(raw_entry)
                collected[UI_ENTRY_SOURCE_NODE_KEY] = node_title
                entries.append(collected)
    return entries
