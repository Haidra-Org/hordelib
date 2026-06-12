"""Pure functions for manipulating ComfyUI API-format pipeline graphs.

A graph is a ``dict[str, dict]`` mapping node names to node dicts of the shape::

    {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["model_loader", 0]}, "_meta": {...}}

Nothing in this module imports ComfyUI; everything here is unit-testable without a GPU.
"""

from typing import Any

from loguru import logger

GraphDict = dict[str, Any]
"""An API-format pipeline graph, keyed by node name."""


def apply_dotted_params(graph: GraphDict, params: dict[str, Any]) -> int:
    """Set parameters on the graph using dotted ``node.input`` keys.

    The ``inputs`` segment may be omitted from the key; it is inserted automatically
    (``"sampler.steps"`` is equivalent to ``"sampler.inputs.steps"``).

    Args:
        graph: The graph to mutate.
        params: A mapping of dotted parameter names to values.

    Returns:
        int: The number of parameters that were skipped because their node did not exist
        in this graph.
    """
    num_skipped = 0

    for key, value in params.items():
        keys = key.split(".")
        skip = False
        if "inputs" not in keys:
            keys.insert(1, "inputs")
        current = graph

        for k in keys[:-1]:
            if k not in current:
                skip = True
                num_skipped += 1
                break

            current = current[k]

        if not skip:
            if not current.get(keys[-1]):
                logger.debug("Template parameter created: key={}", key)
            current[keys[-1]] = value

    logger.debug(
        "Attempted to set parameters: requested_count={}, skipped_count={}",
        len(params),
        num_skipped,
    )
    return num_skipped


def reconnect_input(graph: GraphDict, input: str, output: str) -> bool | None:
    """Connect the named node input to the named node's output.

    Used for dynamic switching of pipeline graphs.

    Args:
        graph: The graph to mutate.
        input: The dotted input to rewire (e.g. ``"sampler.model"``).
        output: The name of the node to connect the input to.

    Returns:
        ``True`` on success, ``None`` if the input or output does not exist in this graph.
    """
    if output not in graph:
        logger.debug(
            "Cannot reconnect input to missing output",
            input_name=input,
            output_name=output,
        )
        return None

    keys = input.split(".")
    if "inputs" not in keys:
        keys.insert(1, "inputs")
    # The walk ends on the input's connection list (e.g. ["model_loader", 0]), not a dict
    current: Any = graph
    for k in keys:
        if k not in current:
            logger.debug("Attempt to reconnect unknown input: input_name={}", input)
            return None

        current = current[k]

    logger.debug("Reconnected input to output", input_name=input, output_name=output)
    current[0] = output
    return True


def fix_node_names(graph: GraphDict) -> GraphDict:
    """Rename nodes to the ``_meta.title`` set in the design file, fixing up references.

    Node titles become the stable parameter namespace (a KSampler titled ``sampler`` is
    addressed as ``sampler.steps`` etc.), so they must be unique within a pipeline.

    Args:
        graph: The graph to rename.

    Returns:
        GraphDict: A new graph dict with renamed nodes (node dicts are shared, not copied).
    """
    newnodes: GraphDict = {}
    renames: dict[str, str] = {}
    for nodename, nodedata in graph.items():
        newname = nodename
        if nodedata.get("_meta", {}).get("title"):
            newname = nodedata["_meta"]["title"]
        renames[nodename] = newname
        newnodes[newname] = nodedata

    # Now we've renamed the node names, change any references to them also
    for node in newnodes.values():
        if "inputs" in node:
            for _, input in node["inputs"].items():
                if isinstance(input, list) and input and input[0] in renames:
                    input[0] = renames[input[0]]
    return newnodes


def fix_pipeline_types(
    graph: GraphDict,
    node_replacements: dict[str, str],
    node_parameter_replacements: dict[str, dict[str, str]] | None = None,
) -> GraphDict:
    """Replace node class types (and optionally rename node inputs) throughout the graph.

    Used to swap ComfyUI standard node types for hordelib node types (e.g.
    ``CheckpointLoaderSimple`` -> ``HordeCheckpointLoader``).

    Args:
        graph: The graph to mutate.
        node_replacements: A mapping of class types to their replacement class types.
        node_parameter_replacements: Per (replaced) class type, a mapping of input names to
            their new names.

    Returns:
        GraphDict: The same graph dict, mutated.
    """
    for nodename, node in graph.items():
        if ("class_type" in node) and (node["class_type"] in node_replacements):
            old_type = node["class_type"]
            new_type = node_replacements[old_type]
            logger.debug("Changed node type: node={}, old_type={}, new_type={}", nodename, old_type, new_type)
            node["class_type"] = new_type

    if node_parameter_replacements:
        for nodename, node in graph.items():
            if ("class_type" in node) and (node["class_type"] in node_parameter_replacements):
                for oldname, newname in node_parameter_replacements[node["class_type"]].items():
                    if "inputs" in node and oldname in node["inputs"]:
                        node["inputs"][newname] = node["inputs"].pop(oldname)
                        logger.debug(
                            "Renamed node input: node={}, old_name={}, new_name={}",
                            nodename,
                            oldname,
                            newname,
                        )

    return graph
