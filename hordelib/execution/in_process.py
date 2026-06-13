"""The in-process ComfyUI execution backend.

Wraps the legacy :class:`hordelib.comfy_horde.Comfy_Horde` machinery behind the
:class:`hordelib.execution.interface.ExecutionBackend` protocol. ComfyUI runs inside this
process, with hordelib's monkeypatches applied (see ``hordelib.comfy_horde.do_comfy_import``).
"""

from collections.abc import Callable
from typing import Any

from loguru import logger

from hordelib.execution.interface import OutputArtifact, ProgressCallback, VRAMStats


class InProcessComfyBackend:
    """Runs pipelines on the ComfyUI embedded in this process.

    ``hordelib.initialise()`` must have completed before :meth:`start` is called.
    """

    def __init__(
        self,
        *,
        comfyui_callback: Callable[[str, dict, str], None] | None = None,
        aggressive_unloading: bool = True,
    ):
        self._comfyui_callback = comfyui_callback
        self._aggressive_unloading = aggressive_unloading
        self._comfy: Any | None = None

    def start(self) -> None:
        import hordelib

        if not hordelib.is_initialised():
            raise RuntimeError(
                "hordelib.initialise() must be called before starting the in-process ComfyUI backend.",
            )

        if self._comfy is None:
            from hordelib.comfy_horde import Comfy_Horde

            self._comfy = Comfy_Horde(
                comfyui_callback=self._comfyui_callback,
                aggressive_unloading=self._aggressive_unloading,
            )

            # Must happen before any pipeline runs: comfy's ProgressBar captures the
            # global hook at construction time.
            from hordelib.execution.progress_hook import install_native_progress_hook

            install_native_progress_hook()

            # Hold a host-injected cross-process lease around the sampling loop so multiple
            # inference processes pipeline (one samples while others stage) instead of idling
            # the GPU in lockstep. A no-op until the host sets a lease.
            from hordelib.execution.sampling_lease import install_sampling_lease_hook

            install_sampling_lease_hook()

            # Record RAM->VRAM and VAE phase durations into the metrics collector regardless
            # of whether logfire is active, so the host can see where non-sampling time goes.
            from hordelib.execution.phase_timing import install_phase_timing_hooks

            install_phase_timing_hooks()

            # When the host has opted into high-memory mode (aggressive_unloading off), keep
            # models resident in VRAM so back-to-back jobs skip the per-job RAM->VRAM reload
            # that otherwise dominates non-sampling time. Gated by the host's VRAM assertion.
            if not self._aggressive_unloading:
                from hordelib.comfy_horde import pin_models_in_vram

                pin_models_in_vram()

    @property
    def comfy_horde(self) -> Any:
        """The underlying Comfy_Horde instance (transitional escape hatch)."""
        self._ensure_started()
        return self._comfy

    def _ensure_started(self) -> None:
        if self._comfy is None:
            self.start()

    def run_pipeline(
        self,
        graph: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[OutputArtifact]:
        self._ensure_started()
        assert self._comfy is not None

        results = self._comfy.run_image_pipeline(graph, {}, progress_callback)
        return self._to_artifacts(results)

    @staticmethod
    def _to_artifacts(results: list[dict[str, Any]]) -> list[OutputArtifact]:
        artifacts: list[OutputArtifact] = []
        for result in results:
            data = result.get("imagedata")
            if data is None:
                logger.warning("Pipeline result entry without imagedata; skipping: keys={}", list(result))
                continue
            mime_type = "image/png" if result.get("type", "PNG").upper() == "PNG" else "application/octet-stream"
            artifacts.append(OutputArtifact(data=data, mime_type=mime_type))
        return artifacts

    def interrupt(self) -> None:
        from hordelib.comfy_horde import interrupt_comfyui_processing

        interrupt_comfyui_processing()

    def free_vram(self) -> None:
        from hordelib.comfy_horde import unload_all_models_vram

        unload_all_models_vram()

    def free_ram(self) -> None:
        from hordelib.comfy_horde import unload_all_models_ram

        unload_all_models_ram()

    def vram_stats(self) -> VRAMStats:
        from hordelib.comfy_horde import get_torch_free_vram_mb, get_torch_total_vram_mb

        return VRAMStats(total_mb=get_torch_total_vram_mb(), free_mb=get_torch_free_vram_mb())
