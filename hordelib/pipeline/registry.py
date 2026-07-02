"""Explicit, tier-ordered pipeline selection over registered definitions.

Each registered :class:`hordelib.pipeline.definition.PipelineDefinition` declares *when* it
applies via its typed :class:`hordelib.pipeline.definition.Selector`; the registry orders
definitions by (tier, order, registration order) and returns the first match. This replaces
both the legacy if/elif tree in ``HordeLib._get_appropriate_pipeline`` and the interim
lambda-predicate/magic-priority registry.

Registration audits every definition against its graph (see :func:`audit_definition`), so a
typo'd binding target or missing output node fails at import time instead of silently doing
nothing at materialization time.

The registry is generic over the payload model and a family-specific selection context, so
each pipeline family (image generation, post-processing, future modalities) gets its own
fully typed registry. The context type is deliberately unconstrained: families that select on
model facts use ``ModelContext``, while simpler families may use a lighter object (or None).
"""

import json
from collections import Counter

from loguru import logger
from pydantic import BaseModel

from hordelib.pipeline.definition import PipelineDefinition, SelectionTier
from hordelib.pipeline.graph import HORDE_OUTPUT_CLASS_TYPES
from hordelib.pipeline.node_schemas import load_node_input_schemas

__all__ = [
    "PipelineRegistry",
    "audit_definition",
]


def _audit_target(
    definition: PipelineDefinition,
    graph_class_types: dict[str, str],
    target: str,
    node_classes: tuple[str, ...],
) -> list[str]:
    """Audit one dotted binding/extra-input target against the graph and node schemas."""
    title, _, input_name = target.partition(".")
    if not input_name:
        return [f"target {target!r} is not a dotted 'node_title.input_name' path"]

    actual_class = graph_class_types.get(title)
    if actual_class is None:
        return [f"target {target!r} names a node absent from the graph"]

    violations: list[str] = []
    if node_classes and actual_class not in node_classes:
        violations.append(
            f"target {target!r}: graph node is class {actual_class!r}, "
            f"but the binding declares {sorted(node_classes)}",
        )

    schema = load_node_input_schemas().get(actual_class)
    if schema is None:
        violations.append(
            f"target {target!r}: class {actual_class!r} is not in the node schema snapshot; "
            "regenerate it with `python -m hordelib.pipeline.node_schemas`",
        )
        return violations

    input_is_declared = input_name in schema.all_inputs
    grandfathered = target in definition.known_spurious_inputs
    if not input_is_declared and not grandfathered:
        violations.append(
            f"target {target!r}: class {actual_class!r} declares no input {input_name!r} "
            "(ComfyUI would silently drop it)",
        )
    if input_is_declared and grandfathered:
        violations.append(
            f"target {target!r} is listed in known_spurious_inputs but class {actual_class!r} "
            "does declare that input; remove the stale grandfather entry",
        )
    return violations


def audit_definition(
    definition: PipelineDefinition,
    *,
    payload_types: tuple[type[BaseModel], ...] = (),
) -> list[str]:
    """Validate a definition against its graph, returning every violation found.

    Checks:
        - The graph file's raw ``_meta.title`` values are unique (``fix_node_names`` would
          otherwise silently drop colliding nodes).
        - Every binding and extra-input target names a node present in the loaded graph,
          with the class the binding declares (when authored via ``node()``), and an input
          name that class actually declares in the committed node schema snapshot (ComfyUI
          silently drops unknown input names; see ``hordelib/pipeline/node_schemas.py``).
        - Every binding source is a field on one of ``payload_types`` (when provided).
        - At least one output is declared; every output node exists and its post-replacement
          class type is a registered Horde output class for its kind.
        - A non-fallback selector declares at least one criterion.

    Args:
        definition: The pipeline definition to audit.
        payload_types: The family's payload model(s); binding sources are validated against
            their fields when non-empty.

    Returns:
        list[str]: Human-readable violations; empty when the definition is sound.
    """
    violations: list[str] = []

    raw_nodes = json.loads(definition.graph_file.read_text(encoding="utf-8"))
    raw_titles = [
        node["_meta"]["title"]
        for node in raw_nodes.values()
        if isinstance(node, dict) and isinstance(node.get("_meta"), dict) and "title" in node["_meta"]
    ]
    duplicate_titles = sorted(
        title
        for title, count in Counter(raw_titles).items()
        if count > 1 and title not in definition.known_title_collisions
    )
    if duplicate_titles:
        violations.append(
            f"graph {definition.graph_file.name} has duplicate node titles {duplicate_titles}; "
            "colliding nodes would be silently dropped at load time",
        )

    graph = definition.load_graph()
    node_titles = set(graph.node_titles())
    graph_class_types = {title: graph.node(title).get("class_type", "") for title in node_titles}

    for binding in definition.bindings:
        violations.extend(_audit_target(definition, graph_class_types, binding.target, binding.node_classes))
    for extra_target in definition.extra_inputs:
        violations.extend(_audit_target(definition, graph_class_types, extra_target, ()))

    if payload_types:
        known_fields = {field for payload_type in payload_types for field in payload_type.model_fields}
        for binding in definition.bindings:
            if binding.source is not None and binding.source not in known_fields:
                violations.append(
                    f"binding for {binding.target!r} reads payload field {binding.source!r}, "
                    f"which none of {[t.__name__ for t in payload_types]} declares",
                )

    stale_grandfathers = definition.known_spurious_inputs - {binding.target for binding in definition.bindings}
    if stale_grandfathers:
        violations.append(
            f"known_spurious_inputs entries with no matching binding: {sorted(stale_grandfathers)}",
        )

    if not definition.outputs:
        violations.append("no outputs declared; every pipeline must declare at least one output node")

    for output in definition.outputs:
        if output.node not in node_titles:
            violations.append(f"declared output node {output.node!r} does not exist in the graph")
            continue
        allowed_class_types = HORDE_OUTPUT_CLASS_TYPES.get(output.kind, frozenset())
        node_class_type = graph.node(output.node).get("class_type")
        if node_class_type not in allowed_class_types:
            violations.append(
                f"output node {output.node!r} has class_type {node_class_type!r}, "
                f"not a registered {output.kind} output class ({sorted(allowed_class_types)})",
            )

    fallback_needs_no_criteria = definition.selector.tier is SelectionTier.FALLBACK
    if not fallback_needs_no_criteria and not definition.selector.has_criteria():
        violations.append(
            f"selector in tier {definition.selector.tier.name} declares no criteria and would "
            "shadow every lower-precedence pipeline",
        )

    return violations


class PipelineRegistry[PayloadT: BaseModel, ContextT]:
    """Holds a family's pipeline definitions and selects among them.

    Registration audits each definition against its graph (see :func:`audit_definition`);
    selection walks the definitions in declared precedence order and returns the first whose
    selector matches.
    """

    def __init__(self, *, payload_types: tuple[type[BaseModel], ...] = ()) -> None:
        """Create an empty registry.

        Args:
            payload_types: The family's payload model(s); when provided, registration also
                validates every binding source against their declared fields.
        """
        self._definitions: list[PipelineDefinition[PayloadT, ContextT]] = []
        self._payload_types = payload_types

    def register(self, definition: PipelineDefinition[PayloadT, ContextT]) -> None:
        """Register a definition, auditing it against its graph.

        Raises:
            ValueError: If a definition with the same name is already registered, or the
                definition fails the audit (see :func:`audit_definition`).
        """
        if any(existing.name == definition.name for existing in self._definitions):
            raise ValueError(f"Pipeline {definition.name!r} is already registered")

        violations = audit_definition(definition, payload_types=self._payload_types)
        if violations:
            details = "\n  - ".join(violations)
            raise ValueError(f"Pipeline definition {definition.name!r} failed its audit:\n  - {details}")

        self._definitions.append(definition)
        # Stable sort preserves registration order within a (tier, order) slot
        self._definitions.sort(key=lambda d: (-d.selector.tier, d.selector.order))

    def select(self, payload: PayloadT, context: ContextT) -> PipelineDefinition[PayloadT, ContextT] | None:
        """Return the highest-precedence definition whose selector matches, or None."""
        for definition in self._definitions:
            if definition.selector.matches(payload, context):
                logger.debug(
                    "Pipeline selected: name={}, tier={}, order={}",
                    definition.name,
                    definition.selector.tier.name,
                    definition.selector.order,
                )
                return definition
        return None

    def get(self, name: str) -> PipelineDefinition[PayloadT, ContextT] | None:
        """Return the registered definition with the given name, or None."""
        for definition in self._definitions:
            if definition.name == name:
                return definition
        return None

    def all_definitions(self) -> list[PipelineDefinition[PayloadT, ContextT]]:
        """Return all registered definitions in selection precedence order."""
        return list(self._definitions)
