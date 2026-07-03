"""Typed representations of the events and results ComfyUI's executor hands the bridge.

ComfyUI delivers execution lifecycle information two ways: events pushed into the server's
``send_sync`` during a run, and result attributes on the ``PromptExecutor`` after a run.
Both arrive as string-labeled raw dicts. This module is the single place those wire shapes
are named: every string key of a ComfyUI event payload lives in a model definition here, so
the rest of the bridge dispatches on enums and reads typed fields.

Critical members:

- :class:`ComfyEventLabel`: every event label the executor can deliver (pinned by
  ``tests/test_comfy_contract_drift.py`` against the vendored ComfyUI).
- :func:`parse_event`: convert a raw ``(label, data)`` pair into a typed event; never raises
  (unrecognized labels or malformed payloads become :class:`UnknownEvent`).
- :class:`ValidationResult`: the typed form of ComfyUI's ``validate_prompt`` 4-tuple.

This module must remain importable before ``hordelib.initialise()``: it never imports ComfyUI.
"""

import typing
from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

VAE_DECODE_NODE_TITLE: typing.Final[str] = "vae_decode"
"""The graph node title hordelib pipelines give their VAE decode step.

Used to surface a user-facing log line when decoding starts, since large decodes can look
like a stall.
"""

MODEL_LOADER_NODE_TITLE: typing.Final[str] = "model_loader"
"""The graph node title hordelib pipelines give their checkpoint loader."""

HORDE_MODEL_NAME_INPUT: typing.Final[str] = "horde_model_name"
"""The loader input carrying the horde model name, extracted for log context."""


class ComfyEventLabel(StrEnum):
    """Every event label ComfyUI's execution path delivers through the server's send_sync."""

    EXECUTION_START = "execution_start"
    EXECUTION_CACHED = "execution_cached"
    EXECUTING = "executing"
    EXECUTED = "executed"
    PROGRESS_STATE = "progress_state"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    EXECUTION_SUCCESS = "execution_success"


class ComfyEventBase(BaseModel):
    """Represents one event delivered by ComfyUI's executor during a pipeline run.

    Extra payload keys are retained rather than rejected so a ComfyUI version that enriches
    an event does not break parsing; the drift tests are what flag new labels or shapes.
    """

    model_config = ConfigDict(extra="allow", frozen=True)


class ExecutionStartEvent(ComfyEventBase):
    """Represents the start of a prompt execution."""

    prompt_id: str | None = None


class ExecutionCachedEvent(ComfyEventBase):
    """Represents the set of nodes satisfied from cache at the start of a run."""

    nodes: list[str] = []
    prompt_id: str | None = None


class ExecutingEvent(ComfyEventBase):
    """Represents a node beginning execution.

    Graphs are title-keyed (see :class:`hordelib.pipeline.graph.ComfyGraph`), so ``node`` is
    the pipeline-declared node title.
    """

    node: str | None = None
    display_node: str | None = None
    prompt_id: str | None = None


class ExecutedEvent(ComfyEventBase):
    """Represents a node having produced UI output.

    ``output`` is the node's raw ui dict (e.g. ``{"images": [...]}``); artifact collection
    reads the executor's ``history_result`` instead of this event, so the field is carried
    for logging/diagnostics only.
    """

    node: str | None = None
    display_node: str | None = None
    output: dict[str, typing.Any] | None = None
    prompt_id: str | None = None


class ProgressStateEvent(ComfyEventBase):
    """Represents per-node progress state (pending/running/finished) for the whole graph.

    Delivered by ComfyUI's WebUIProgressHandler, which the executor force-registers each run;
    the bridge receives it for free without adopting the ProgressRegistry.
    """

    prompt_id: str | None = None
    nodes: dict[str, dict[str, typing.Any]] = {}


class ExecutionErrorEvent(ComfyEventBase):
    """Represents a node execution failure, with the full exception context."""

    prompt_id: str | None = None
    node_id: str | None = None
    node_type: str | None = None
    executed: list[str] = []
    exception_message: str = ""
    exception_type: str = ""
    traceback: list[str] | str | None = None
    current_inputs: typing.Any = None
    current_outputs: typing.Any = None

    def summary(self) -> str:
        """Return a one-line human-readable description of the failure."""
        node = self.node_id or "unknown node"
        node_type = f" ({self.node_type})" if self.node_type else ""
        exception_type = self.exception_type or "Exception"
        return f"{node}{node_type}: {exception_type}: {self.exception_message}"


class ExecutionInterruptedEvent(ComfyEventBase):
    """Represents a run cut short by an interrupt request."""

    prompt_id: str | None = None
    node_id: str | None = None
    node_type: str | None = None
    executed: list[str] = []


class ExecutionSuccessEvent(ComfyEventBase):
    """Represents a run that completed every staged node."""

    prompt_id: str | None = None


class UnknownEvent(ComfyEventBase):
    """Represents an event this bridge version does not recognize.

    Produced for labels outside :class:`ComfyEventLabel` or payloads that fail model
    validation; carried so callers can still log the raw information.
    """

    label: str = ""
    data: dict[str, typing.Any] = {}


ComfyEvent = (
    ExecutionStartEvent
    | ExecutionCachedEvent
    | ExecutingEvent
    | ExecutedEvent
    | ProgressStateEvent
    | ExecutionErrorEvent
    | ExecutionInterruptedEvent
    | ExecutionSuccessEvent
    | UnknownEvent
)
"""The union of all typed events :func:`parse_event` can return."""


_EVENT_MODEL_BY_LABEL: dict[ComfyEventLabel, type[ComfyEvent]] = {
    ComfyEventLabel.EXECUTION_START: ExecutionStartEvent,
    ComfyEventLabel.EXECUTION_CACHED: ExecutionCachedEvent,
    ComfyEventLabel.EXECUTING: ExecutingEvent,
    ComfyEventLabel.EXECUTED: ExecutedEvent,
    ComfyEventLabel.PROGRESS_STATE: ProgressStateEvent,
    ComfyEventLabel.EXECUTION_ERROR: ExecutionErrorEvent,
    ComfyEventLabel.EXECUTION_INTERRUPTED: ExecutionInterruptedEvent,
    ComfyEventLabel.EXECUTION_SUCCESS: ExecutionSuccessEvent,
}


def parse_event(label: str, data: dict[str, typing.Any]) -> ComfyEvent:
    """Convert a raw ComfyUI event into its typed form.

    Never raises: an unrecognized label or a payload that fails validation is returned as an
    :class:`UnknownEvent` (with a warning for the malformed-payload case, since that indicates
    contract drift rather than mere unfamiliarity).

    Args:
        label: The event label as delivered to ``send_sync``.
        data: The raw event payload.

    Returns:
        ComfyEvent: The typed event, or :class:`UnknownEvent` when the label or payload is
        not recognized.
    """
    try:
        known_label = ComfyEventLabel(label)
    except ValueError:
        return UnknownEvent(label=label, data=data)

    event_model = _EVENT_MODEL_BY_LABEL[known_label]
    try:
        return event_model.model_validate(data)
    except ValidationError as validation_error:
        logger.warning(
            "ComfyUI event payload no longer matches its typed model (contract drift?)",
            label=label,
            error=str(validation_error),
        )
        return UnknownEvent(label=label, data=data)


class ValidationResult(BaseModel):
    """Represents the result of ComfyUI's ``validate_prompt`` for a materialized graph."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error: dict[str, typing.Any] | None = None
    output_node_ids: list[str] = []
    node_errors: dict[str, typing.Any] = {}

    @classmethod
    def from_comfy(cls, raw: tuple) -> "ValidationResult":
        """Convert the raw ``validate_prompt`` return value into a typed result.

        Args:
            raw: The tuple returned by ComfyUI's ``validate_prompt``.

        Returns:
            ValidationResult: The typed validation outcome.

        Raises:
            ValueError: If the tuple does not have the pinned 4-element shape, indicating
                the ComfyUI validation contract has drifted.
        """
        if len(raw) != 4:
            raise ValueError(
                f"validate_prompt returned {len(raw)} elements, expected 4 "
                "(is_valid, error, output_node_ids, node_errors); the ComfyUI validation "
                "contract has drifted.",
            )
        is_valid, error, output_node_ids, node_errors = raw
        return cls(
            is_valid=bool(is_valid),
            error=error,
            output_node_ids=list(output_node_ids or []),
            node_errors=dict(node_errors or {}),
        )
