# node_v3_canary.py
# Proves ComfyUI's typed V3 node API works in hordelib's headless embedding.
#
# hordelib's policy is that NEW nodes (especially for new modalities, where the V3 typed
# sockets like Audio/Video/SVG are the payoff) are written against comfy_api's V3
# ComfyExtension API, while the existing classic-API nodes stay as they are. This canary is
# the proof that policy rests on: it registers through comfy_entrypoint, executes headless,
# and returns a ui entry satisfying the bridge's BytesIO artifact contract (see
# hordelib.execution.results). It is exercised by tests/test_comfy_contract_drift.py and is
# not used by any pipeline.
from io import BytesIO

import numpy as np
from PIL import Image
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class HordeV3CanaryOutput(io.ComfyNode):
    """A V3 output node encoding its input image to an in-memory PNG ui entry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define the typed V3 schema: one IMAGE input, no sockets out, output node."""
        return io.Schema(
            node_id="HordeV3CanaryOutput",
            display_name="Horde V3 Canary Output",
            category="image",
            inputs=[io.Image.Input("images")],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        """Encode each input image to a PNG BytesIO entry, mirroring HordeImageOutput."""
        results = []
        for image in images:
            array = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            byte_stream = BytesIO()
            pil_image.save(byte_stream, format="PNG", compress_level=4)
            byte_stream.seek(0)
            results.append({"imagedata": byte_stream, "type": "PNG"})
        return io.NodeOutput(ui={"images": results})


class HordeV3CanaryExtension(ComfyExtension):
    """Registers the canary through the V3 extension entry point."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the V3 nodes this extension provides."""
        return [HordeV3CanaryOutput]


async def comfy_entrypoint() -> HordeV3CanaryExtension:
    """Create the extension instance ComfyUI's custom-node loader awaits."""
    return HordeV3CanaryExtension()
