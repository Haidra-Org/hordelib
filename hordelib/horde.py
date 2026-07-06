# horde.py
# Main interface for the horde to this library.
from __future__ import annotations

import io
import time
from collections.abc import Callable
from enum import Enum, auto

import logfire
from horde_sdk.ai_horde_api.apimodels.base import (
    GenMetadataEntry,
)
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_FACEFIXERS
from horde_sdk.generation_parameters.image import ImageGenerationParameters
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.execution.interface import OutputSpec
from hordelib.execution.stage_graph import (
    cut_decode_stage,
    cut_encode_text_stage,
    cut_sample_stage,
    cut_vae_encode_stage,
)
from hordelib.pipeline import constants as pipeline_constants
from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import PipelineDefinition
from hordelib.pipeline.families.post_processing import POST_PROCESSING_REGISTRY
from hordelib.pipeline.graph import ComfyGraph
from hordelib.pipeline.identifiers import AUTO_PIPELINE, AutoPipeline, ImagePipeline
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.payload_pp import (
    FacefixPayload,
    PostProcessingPayload,
    PostProcessorKind,
    StripBackgroundPayload,
    UpscalePayload,
    classify_post_processor,
    post_processing_payload_from_horde_dict,
)
from hordelib.pipeline.resolution import resolve_post_processing_model
from hordelib.utils.image_utils import ImageUtils
from hordelib.utils.ioredirect import ComfyUIProgress


class ProgressState(Enum):
    """The state of the progress report"""

    started = auto()
    progress = auto()
    post_processing = auto()
    finished = auto()


class ProgressReport(BaseModel):
    """A progress message sent to a callback"""

    hordelib_progress_state: ProgressState
    comfyui_progress: ComfyUIProgress | None = None
    progress: float | None = None
    hordelib_message: str | None = None


class ResultingImageReturn:
    image: Image.Image | None
    rawpng: io.BytesIO | None
    faults: list[GenMetadataEntry]

    def __init__(
        self,
        image: Image.Image | None,
        rawpng: io.BytesIO | None,
        faults: list[GenMetadataEntry],
    ):
        if faults is None:
            faults = []

        for fault in faults:
            if not isinstance(fault, GenMetadataEntry):
                raise TypeError("faults must be a list of GenMetadataEntry")

        if image is not None and not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        if rawpng is not None and not isinstance(rawpng, io.BytesIO):
            raise TypeError("rawpng must be a io.BytesIO")

        self.image = image
        self.rawpng = rawpng
        self.faults = faults


# Module-level metrics for inference performance tracking
inference_duration_histogram = logfire.metric_histogram(
    "horde.inference.duration_ms",
    unit="ms",
    description="Total inference duration including model loading and generation",
)

progress_gauge = logfire.metric_gauge(
    "horde.progress.percent",
    unit="percent",
    description="Current progress percentage of inference job",
)

post_process_duration_histogram = logfire.metric_histogram(
    "horde.post_process.duration_ms",
    unit="ms",
    description="Duration of individual post-processing operations",
)


class HordeLib:
    _instance: HordeLib | None = None
    _initialised = False

    # Payload vocabularies; the canonical copies live in hordelib.pipeline.constants and the
    # payload bounds/clamping live in hordelib.pipeline.payload.ImageGenPayload.
    SAMPLERS_MAP = pipeline_constants.SAMPLERS_MAP
    CONTROLNET_IMAGE_PREPROCESSOR_MAP = pipeline_constants.CONTROLNET_IMAGE_PREPROCESSOR_MAP
    CONTROLNET_MODEL_MAP = pipeline_constants.CONTROLNET_MODEL_MAP
    SOURCE_IMAGE_PROCESSING_OPTIONS = pipeline_constants.SOURCE_IMAGE_PROCESSING_OPTIONS
    SCHEDULERS = pipeline_constants.SCHEDULERS

    _comfyui_callback: Callable[[str, dict, str], None] | None = None

    # We are a singleton
    def __new__(
        cls,
        *,
        comfyui_callback: Callable[[str, dict, str], None] | None = None,
        aggressive_unloading: bool = True,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._comfyui_callback = comfyui_callback
            cls.aggressive_unloading = aggressive_unloading

        return cls._instance

    # We initialise only ever once (in the lifetime of the singleton)
    def __init__(
        self,
        *,
        comfyui_callback: Callable[[str, dict, str], None] | None = None,
        aggressive_unloading: bool | None = True,
        # If you add any more parameters here, you should also add them to __new__ above
        # and follow the same pattern
    ):
        if not self._initialised:
            self.backend = InProcessComfyBackend(
                comfyui_callback=comfyui_callback if comfyui_callback else self._comfyui_callback,
                aggressive_unloading=(
                    aggressive_unloading if aggressive_unloading is not None else self.aggressive_unloading
                ),
            )
            # Eager start preserves the historical timing: ComfyUI spins up when HordeLib
            # is constructed, not on the first job.
            self.backend.start()
            self.__class__._initialised = True

    @logfire.instrument("horde.materialize_graph", extract_args=False)
    def _materialize_image_graph(
        self,
        payload: dict,
    ) -> tuple[
        ComfyGraph,
        ImageGenPayload,
        ModelContext,
        list[GenMetadataEntry],
        tuple[OutputSpec, ...],
    ]:
        """The legacy-dict pipeline flow: resolve -> normalize -> validate -> select -> materialize.

        The compatibility front end for raw Horde payload dicts: the AI-Horde-specific dict
        hacks (``###`` prompt splitting, the karras flag, hires-fix disable rules) are applied
        before validation, then the shared typed tail runs with automatic pipeline selection.

        Returns the fully materialized graph, the typed payload, the resolved model context,
        any faults to attach to the results, and the pipeline's declared outputs.
        """
        from hordelib.pipeline.horde_compat import normalize_horde_payload
        from hordelib.pipeline.resolution import resolve_image_model

        model = payload.get("model")
        if model is None:
            raise RuntimeError("No model specified in payload")

        context = resolve_image_model(str(model))
        normalized, compat_faults = normalize_horde_payload(payload, context)
        typed = ImageGenPayload.from_horde_dict(normalized)
        return self._finalize_and_materialize(typed, context, compat_faults, pipeline=AUTO_PIPELINE)

    def _finalize_and_materialize(
        self,
        typed: ImageGenPayload,
        context: ModelContext,
        faults: list[GenMetadataEntry],
        *,
        pipeline: ImagePipeline | AutoPipeline,
    ) -> tuple[
        ComfyGraph,
        ImageGenPayload,
        ModelContext,
        list[GenMetadataEntry],
        tuple[OutputSpec, ...],
    ]:
        """The shared typed tail of the pipeline flow: resize -> resolve -> select -> materialize."""
        from hordelib.pipeline.horde_compat import resize_sources_to_request
        from hordelib.pipeline.resolution import resolve_image_generation

        typed = resize_sources_to_request(typed)
        typed, context, resolution_faults = resolve_image_generation(typed, context)

        definition = self._resolve_pipeline_definition(typed, context, pipeline)

        logger.info(
            "Pipeline validation complete",
            pipeline=definition.name,
            faults_count=len(faults) + len(resolution_faults),
        )
        graph = definition.materialize(typed, context)
        return graph, typed, context, faults + resolution_faults, definition.outputs

    @staticmethod
    def _resolve_pipeline_definition(
        typed: ImageGenPayload,
        context: ModelContext,
        pipeline: ImagePipeline | AutoPipeline,
    ) -> PipelineDefinition[ImageGenPayload, ModelContext]:
        """Resolve the pipeline definition to run: the caller's explicit choice, or the registry's.

        An explicit choice is trusted even when its selector would not have matched (the
        registry audits guarantee any payload can materialize any registered definition); the
        mismatch is logged, along with what automatic selection would have picked.
        """
        from hordelib.pipeline.families.image_gen import DEFAULT_REGISTRY

        if isinstance(pipeline, AutoPipeline):
            definition = DEFAULT_REGISTRY.select(typed, context)
            if definition is None:
                # The registry has a fallback-tier catch-all, so this is purely defensive
                logger.warning("No pipeline matched payload; falling back to stable_diffusion")
                definition = DEFAULT_REGISTRY.get(ImagePipeline.STABLE_DIFFUSION.value)
                assert definition is not None
            return definition

        definition = DEFAULT_REGISTRY.get(pipeline.value)
        if definition is None:
            # Unreachable while the import-time enum/registry sync audit holds.
            raise RuntimeError(f"Pipeline {pipeline.value!r} is not registered")

        if not definition.selector.matches(typed, context):
            auto_definition = DEFAULT_REGISTRY.select(typed, context)
            logger.warning(
                "Explicit pipeline does not match the payload's features; proceeding as requested",
                pipeline=definition.name,
                auto_would_select=auto_definition.name if auto_definition else None,
            )

        return definition

    # ------------------------------------------------------------------------------------------
    # Disaggregated stage entry points.
    #
    # Each materializes the full family graph (so LoRA/TI/hires/SDXL-conditioning wiring is reused
    # verbatim), then hands it to a cut helper in hordelib.execution.stage_graph, which points the
    # loader at only the component the stage runs, drops the original image output, and adds/rewires
    # the stage IO nodes so ancestor-only execution runs just that stage. Those helpers validate the
    # graph shape structurally and raise StageGraphUnsupportedError for families they cannot cut.
    # Canonical node titles come from pipeline/families/image_gen/bindings.py (model_loader, prompt,
    # negative_prompt, sampler, vae_encode, vae_decode, output_image). v1 families only
    # (txt2img/img2img/hires/LoRA/TI); controlnet/inpaint/cascade, whose graphs differ, stay on the
    # monolithic path.
    # ------------------------------------------------------------------------------------------

    def _materialize_stage_graph(
        self,
        params: ImageGenerationParameters,
    ) -> tuple[ComfyGraph, tuple[OutputSpec, ...], list[GenMetadataEntry]]:
        """Materialize the full family graph for a job, ready to be cut into a single stage."""
        from hordelib.pipeline.horde_compat import apply_model_compat
        from hordelib.pipeline.resolution import resolve_image_model
        from hordelib.pipeline.sdk_adapter import to_image_gen_payload

        typed, conversion_faults = to_image_gen_payload(params)
        context = resolve_image_model(typed.model)
        typed, compat_faults = apply_model_compat(typed, context)
        graph, _typed, _context, faults, outputs = self._finalize_and_materialize(
            typed,
            context,
            conversion_faults + compat_faults,
            pipeline=AUTO_PIPELINE,
        )
        return graph, outputs, faults

    def encode_text_stage(self, params: ImageGenerationParameters) -> tuple[bytes, bytes]:
        """Encode a job's prompts to (positive, negative) CONDITIONING blobs (loads only the CLIP).

        Supported only on the v1 family graph shape: a combined ``model_loader``
        (``HordeCheckpointLoader``) and ``prompt``/``negative_prompt`` CLIPTextEncode nodes. Raises
        :class:`~hordelib.execution.interface.StageGraphUnsupportedError` for any other shape
        (split-loader families such as Qwen/Z-Image, where the loader-subset flags are no-ops).
        """
        graph, _outputs, _faults = self._materialize_stage_graph(params)
        outputs = cut_encode_text_stage(graph)
        artifacts = self.backend.run_pipeline(graph.to_api_dict(), outputs=outputs)
        by_node = {a.source_node: a.data.getvalue() for a in artifacts}
        return by_node["cond_positive_output"], by_node["cond_negative_output"]

    def sample_stage(
        self,
        params: ImageGenerationParameters,
        *,
        positive_conditioning_bytes: bytes,
        negative_conditioning_bytes: bytes,
        source_latent_bytes: bytes | None = None,
    ) -> bytes:
        """Sample a LATENT from injected conditioning (loads only the UNet).

        ``source_latent_bytes`` injects an img2img/remix start latent (VAE-encoded by the image
        lane) in place of the graph's own VAE-encode; None runs txt2img from the empty latent.

        Supported only on the v1 family graph shape: a combined ``model_loader`` and a ``sampler``
        KSampler (plus an optional hires ``upscale_sampler``). Raises
        :class:`~hordelib.execution.interface.StageGraphUnsupportedError` for any other shape (Flux's
        ``SamplerCustomAdvanced`` sampler, split-loader families), when the job's graph is img2img
        (VAE-encode feeds the sampler) but ``source_latent_bytes`` was not supplied, or when the
        conditioning injection fails to displace the graph's text encoders.
        """
        graph, _outputs, _faults = self._materialize_stage_graph(params)
        outputs = cut_sample_stage(
            graph,
            positive_bytes=positive_conditioning_bytes,
            negative_bytes=negative_conditioning_bytes,
            source_latent_bytes=source_latent_bytes,
        )
        artifacts = self.backend.run_pipeline(graph.to_api_dict(), outputs=outputs)
        return artifacts[0].data.getvalue()

    def vae_encode_stage(self, params: ImageGenerationParameters) -> bytes:
        """VAE-encode a job's source image to a LATENT (loads only the VAE), for img2img/remix.

        Supported only on the v1 family graph shape with a ``vae_encode`` node present (a txt2img
        graph has none). Raises :class:`~hordelib.execution.interface.StageGraphUnsupportedError`
        otherwise (a txt2img graph, or a split-loader family).
        """
        graph, _outputs, _faults = self._materialize_stage_graph(params)
        outputs = cut_vae_encode_stage(graph)
        artifacts = self.backend.run_pipeline(graph.to_api_dict(), outputs=outputs)
        return artifacts[0].data.getvalue()

    def decode_stage(
        self,
        params: ImageGenerationParameters,
        *,
        latent_bytes: bytes,
    ) -> tuple[list[ResultingImageReturn], list[GenMetadataEntry]]:
        """Decode an injected LATENT to images (loads only the VAE), reusing the graph's image output.

        Supported only on the v1 family graph shape: a combined ``model_loader`` and a ``vae_decode``
        node feeding the reused image output. Raises
        :class:`~hordelib.execution.interface.StageGraphUnsupportedError` for any other shape
        (split-loader families).
        """
        graph, outputs, faults = self._materialize_stage_graph(params)
        cut_decode_stage(graph, latent_bytes=latent_bytes)
        artifacts = self.backend.run_pipeline(graph.to_api_dict(), outputs=outputs)
        results = [ResultingImageReturn(image=Image.open(a.data), rawpng=a.data, faults=faults) for a in artifacts]
        return results, faults

    @logfire.instrument("horde.inference", extract_args=False)
    def _inference(
        self,
        payload: dict,
        *,
        single_image_expected: bool = True,
        comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn] | ResultingImageReturn:
        graph_bundle = self._materialize_image_graph(payload)
        return self._run_materialized(
            graph_bundle,
            single_image_expected=single_image_expected,
            comfyui_progress_callback=comfyui_progress_callback,
            defer_vram_unload=defer_vram_unload,
        )

    def _run_materialized(
        self,
        graph_bundle: tuple[
            ComfyGraph,
            ImageGenPayload,
            ModelContext,
            list[GenMetadataEntry],
            tuple[OutputSpec, ...],
        ],
        *,
        single_image_expected: bool = True,
        comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn] | ResultingImageReturn:
        """Run a materialized graph on the backend and wrap the artifacts as results."""
        start_time = time.time()

        graph, typed, context, faults, declared_outputs = graph_bundle

        resolution = f"{typed.width}x{typed.height}"
        logger.info(
            "Starting inference",
            model=context.horde_model_name,
            resolution=resolution,
            steps=typed.ddim_steps,
            sampler=typed.sampler_name,
            single_image=single_image_expected,
            loras_count=len(context.resolved_loras),
            tis_count=len(typed.tis),
            has_controlnet=typed.control_type is not None,
        )
        logger.info(
            "Generating image: resolution={}, steps={}, model={}",
            resolution,
            typed.ddim_steps,
            context.horde_model_name,
        )
        if typed.hires_fix:
            logger.info("Using hiresfix: final_resolution={}", resolution)
        if typed.control_type:
            logger.info("Using controlnet: control_type={}", typed.control_type)
        if context.resolved_loras:
            logger.info("Using LORAs: count={}", len(context.resolved_loras))
        if typed.tis:
            logger.info("Using TIs: count={}", len(typed.tis))

        artifacts = self.backend.run_pipeline(
            graph.to_api_dict(),
            outputs=declared_outputs,
            progress_callback=comfyui_progress_callback,
            defer_vram_unload=defer_vram_unload,
        )

        ret_results = []
        for artifact in artifacts:
            ret_results.append(
                ResultingImageReturn(
                    image=Image.open(artifact.data),
                    rawpng=artifact.data,
                    faults=faults,
                ),
            )

        # Record inference duration metric
        duration_ms = (time.time() - start_time) * 1000
        inference_duration_histogram.record(duration_ms)
        logger.info("Inference complete", duration_ms=duration_ms, image_count=len(ret_results))

        if single_image_expected:
            if len(ret_results) != 1:
                raise RuntimeError("Expected a single image but got multiple")
            return ret_results[0]

        return ret_results

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[ProgressReport], None] | None,
        report: ProgressReport,
    ) -> None:
        """Send a progress report to the callback, never letting a callback error propagate."""
        if progress_callback is None:
            return
        try:
            progress_callback(report)
        except Exception:
            logger.exception("Progress callback failed")

    @classmethod
    def _make_comfyui_progress_adapter(
        cls,
        progress_callback: Callable[[ProgressReport], None] | None,
    ) -> Callable[[ComfyUIProgress, str], None]:
        """Create the backend-facing progress adapter: metrics, logging, and callback fan-out."""

        def _comfyui_progress_adapter(comfyui_progress: ComfyUIProgress, message: str) -> None:
            # Record progress metric
            progress_gauge.set(comfyui_progress.percent)

            # Log progress event
            logger.info(
                "horde.progress_update",
                percent=comfyui_progress.percent,
                current_step=comfyui_progress.current_step,
                total_steps=comfyui_progress.total_steps,
                rate=comfyui_progress.rate,
                rate_unit=comfyui_progress.rate_unit.name,
            )

            cls._emit_progress(
                progress_callback,
                ProgressReport(
                    hordelib_progress_state=ProgressState.progress,
                    hordelib_message=message,
                    comfyui_progress=comfyui_progress,
                ),
            )

        return _comfyui_progress_adapter

    @logfire.instrument("horde.basic_inference", extract_args=False)
    def basic_inference(
        self,
        payload: dict,
        *,
        progress_callback: Callable[[ProgressReport], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn]:
        post_processing_requested: list[str] | None = payload.get("post_processing")
        n_iter = payload.get("n_iter", 1)

        logger.info(
            "Basic inference started",
            n_iter=n_iter,
            post_processing_count=len(post_processing_requested) if post_processing_requested else 0,
            has_callback=progress_callback is not None,
        )

        faults: list[GenMetadataEntry] = []

        self._emit_progress(
            progress_callback,
            ProgressReport(
                hordelib_progress_state=ProgressState.started,
                hordelib_message="Initiating inference...",
                progress=0,
            ),
        )

        result = self._inference(
            payload,
            single_image_expected=False,
            comfyui_progress_callback=self._make_comfyui_progress_adapter(progress_callback),
            defer_vram_unload=defer_vram_unload,
        )

        if not isinstance(result, list):
            raise RuntimeError(f"Expected a list of PIL.Image.Image but got {type(result)}")

        return_list = [x for x in result if isinstance(x.image, Image.Image)]
        pptext = ""
        if post_processing_requested is not None:
            pptext = " Initiating post-processing..."
        logger.debug("Inference complete: image_count={}, post_processing={}", len(return_list), pptext)

        post_processed: list[ResultingImageReturn] | None = None
        if post_processing_requested is not None:
            with logfire.span(
                "horde.post_processing",
                image_count=len(return_list),
                operation_count=len(post_processing_requested),
            ):
                if progress_callback is not None:
                    try:
                        progress_callback(
                            ProgressReport(
                                hordelib_progress_state=ProgressState.post_processing,
                                hordelib_message="Post Processing.",
                            ),
                        )
                    except Exception:
                        logger.exception("Progress callback failed")

                post_processed = []
                for img_idx, ret in enumerate(return_list):
                    with logfire.span("horde.post_process_image", image_index=img_idx):
                        single_image_faults = faults[:]
                        final_image = ret.image
                        final_rawpng = ret.rawpng

                        if progress_callback is not None:
                            try:
                                progress_callback(
                                    ProgressReport(
                                        hordelib_progress_state=ProgressState.progress,
                                        hordelib_message="Post Processing new image.",
                                    ),
                                )
                            except Exception:
                                logger.exception("Progress callback failed")

                        # Facefixers sort last (legacy ordering; images_expected/ encodes it)
                        post_processing_requested = sorted(
                            post_processing_requested,
                            key=lambda x: 1 if x in KNOWN_FACEFIXERS.__members__ else 0,
                        )

                        for post_processing in post_processing_requested:
                            if final_image is None:
                                logger.error(
                                    "No image available to post-process; aborting remaining operations",
                                )
                                break

                            pp_start = time.perf_counter()
                            pp_kind = classify_post_processor(post_processing)

                            if pp_kind is PostProcessorKind.upscaler:
                                with logfire.span(
                                    "pp.upscale",
                                    model=post_processing,
                                    image_index=img_idx,
                                ):
                                    image_ret = self.post_process(
                                        UpscalePayload(
                                            model=post_processing,
                                            source_image=final_image,
                                        ),
                                    )
                                    single_image_faults += image_ret.faults
                                    final_rawpng = image_ret.rawpng
                                    final_image = image_ret.image
                                    pp_duration = (time.perf_counter() - pp_start) * 1000
                                    post_process_duration_histogram.record(pp_duration)
                                    logger.info(
                                        "pp.upscale_complete",
                                        model=post_processing,
                                        duration_ms=pp_duration,
                                        fault_count=len(image_ret.faults),
                                    )

                            elif pp_kind is PostProcessorKind.facefixer:
                                with logfire.span(
                                    "pp.facefix",
                                    model=post_processing,
                                    strength=payload.get("facefixer_strength", 1.0),
                                    image_index=img_idx,
                                ):
                                    # facefixer_strength is deliberately not forwarded: the
                                    # legacy mapping never wired it to codeformer_fidelity, so
                                    # honoring it would change existing job output.
                                    image_ret = self.post_process(
                                        FacefixPayload(
                                            model=post_processing,
                                            source_image=final_image,
                                        ),
                                    )
                                    single_image_faults += image_ret.faults
                                    final_rawpng = image_ret.rawpng
                                    final_image = image_ret.image
                                    pp_duration = (time.perf_counter() - pp_start) * 1000
                                    post_process_duration_histogram.record(pp_duration)
                                    logger.info(
                                        "pp.facefix_complete",
                                        model=post_processing,
                                        duration_ms=pp_duration,
                                        fault_count=len(image_ret.faults),
                                    )

                            elif pp_kind is PostProcessorKind.strip_background:
                                with logfire.span("pp.strip_background", image_index=img_idx):
                                    if final_image is not None:
                                        # The pre-strip rawpng is intentionally kept (legacy parity)
                                        final_image = self.post_process(
                                            StripBackgroundPayload(source_image=final_image),
                                        ).image
                                    pp_duration = (time.perf_counter() - pp_start) * 1000
                                    post_process_duration_histogram.record(pp_duration)
                                    logger.info(
                                        "pp.strip_background_complete",
                                        duration_ms=pp_duration,
                                    )

                            else:
                                logger.warning(
                                    "Unknown post-processor requested; skipping: name={}",
                                    post_processing,
                                )

                        if final_image is None:
                            # TODO: Allow to return a partially PP image?
                            logger.error("Post processing failed and there is no output image!")
                            logger.error("pp.failed_no_output", image_index=img_idx)
                        else:
                            post_processed.append(
                                ResultingImageReturn(
                                    image=final_image,
                                    rawpng=final_rawpng,
                                    faults=single_image_faults,
                                ),
                            )

        if progress_callback is not None:
            try:
                progress_callback(
                    ProgressReport(
                        hordelib_progress_state=ProgressState.finished,
                        hordelib_message="Inference complete.",
                        progress=100,
                    ),
                )
            except Exception:
                logger.exception("Progress callback failed")

        if post_processed is not None:
            logger.debug("Post-processing complete: image_count={}", len(post_processed))
            return post_processed

        if len(return_list) == len(result):
            return return_list

        raise RuntimeError("Expected a list of PIL.Image.Image but got a mix of types!")

    @logfire.instrument("horde.basic_inference_single_image", extract_args=False)
    def basic_inference_single_image(self, payload: dict) -> ResultingImageReturn:
        result = self._inference(payload, single_image_expected=True)
        if isinstance(result, ResultingImageReturn):
            return result

        raise RuntimeError(f"Expected a PIL.Image.Image but got {type(result)}")

    def basic_inference_rawpng(self, payload: dict) -> list[io.BytesIO]:
        """Return the results directly from comfy as (a) raw PNG byte stream(s)."""
        result = self._inference(payload, single_image_expected=False)

        if isinstance(result, list):
            bytes_io_list = [x.rawpng for x in result if isinstance(x.rawpng, io.BytesIO)]
            if len(bytes_io_list) == len(result):
                return bytes_io_list

            raise RuntimeError("Expected a list of io.BytesIO but got a mix of types!")

        if isinstance(result.image, io.BytesIO):
            return [result.image]

        raise RuntimeError(f"Expected at least one io.BytesIO. Got {result}.")

    @logfire.instrument("horde.generate", extract_args=False)
    def generate(
        self,
        params: ImageGenerationParameters,
        *,
        pipeline: ImagePipeline | AutoPipeline,
        progress_callback: Callable[[ProgressReport], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn]:
        """Run an image generation from backend-agnostic parameters. Inference only.

        The typed counterpart of `basic_inference`: it accepts the ecosystem-shared
        [`ImageGenerationParameters`][horde_sdk.generation_parameters.image.ImageGenerationParameters]
        and requires the pipeline choice to be visible at the call site, either a specific
        [`ImagePipeline`][hordelib.pipeline.identifiers.ImagePipeline] member or the explicit
        [`AUTO_PIPELINE`][hordelib.pipeline.identifiers.AUTO_PIPELINE] opt-in to registry
        selection. Unlike `basic_inference`, no post-processing is performed (and
        `ProgressState.post_processing` is never emitted); callers drive post-processing
        explicitly via `post_process`.

        Args:
            params: The generation parameters. Any embedded `alchemy_params` are ignored.
            pipeline: The pipeline to run, or `AUTO_PIPELINE` for registry-based selection.
                An explicit pipeline is trusted even if its selector would not have matched
                the payload (a warning is logged with what AUTO would have chosen).
            progress_callback: Receives started/progress/finished reports.
            defer_vram_unload: Keep the model resident in VRAM after the job.

        Returns:
            One result per generated image, each carrying the image, its raw PNG stream, and
            any faults accumulated during conversion and generation.

        Raises:
            RuntimeError: If the model cannot be resolved (unknown to the reference or
                missing from disk).
        """
        from hordelib.pipeline.horde_compat import apply_model_compat
        from hordelib.pipeline.resolution import resolve_image_model
        from hordelib.pipeline.sdk_adapter import to_image_gen_payload

        logger.info(
            "Typed generation started",
            batch_size=params.batch_size,
            pipeline=pipeline.value if isinstance(pipeline, ImagePipeline) else "auto",
            has_callback=progress_callback is not None,
        )

        typed, conversion_faults = to_image_gen_payload(params)
        context = resolve_image_model(typed.model)
        typed, compat_faults = apply_model_compat(typed, context)

        self._emit_progress(
            progress_callback,
            ProgressReport(
                hordelib_progress_state=ProgressState.started,
                hordelib_message="Initiating inference...",
                progress=0,
            ),
        )

        graph_bundle = self._finalize_and_materialize(
            typed,
            context,
            conversion_faults + compat_faults,
            pipeline=pipeline,
        )

        result = self._run_materialized(
            graph_bundle,
            single_image_expected=False,
            comfyui_progress_callback=self._make_comfyui_progress_adapter(progress_callback),
            defer_vram_unload=defer_vram_unload,
        )

        if not isinstance(result, list):
            raise RuntimeError(f"Expected a list of results but got {type(result)}")

        return_list = [single_result for single_result in result if isinstance(single_result.image, Image.Image)]

        self._emit_progress(
            progress_callback,
            ProgressReport(
                hordelib_progress_state=ProgressState.finished,
                hordelib_message="Inference complete.",
                progress=100,
            ),
        )

        if len(return_list) != len(result):
            raise RuntimeError("Expected a list of PIL.Image.Image but got a mix of types!")

        return return_list

    def select_pipeline(self, params: ImageGenerationParameters) -> ImagePipeline:
        """Return the pipeline that `AUTO_PIPELINE` would select for these parameters.

        A preview of selection, not a pure function: the parameters are resolved exactly as
        `generate` would resolve them, which touches the model managers and may download
        referenced ad-hoc models (LoRAs) as an IO boundary.

        Raises:
            RuntimeError: If the model cannot be resolved (unknown to the reference or
                missing from disk).
        """
        from hordelib.pipeline.horde_compat import apply_model_compat, resize_sources_to_request
        from hordelib.pipeline.resolution import resolve_image_generation, resolve_image_model
        from hordelib.pipeline.sdk_adapter import to_image_gen_payload

        typed, _conversion_faults = to_image_gen_payload(params)
        context = resolve_image_model(typed.model)
        typed, _compat_faults = apply_model_compat(typed, context)
        typed = resize_sources_to_request(typed)
        typed, context, _resolution_faults = resolve_image_generation(typed, context)

        definition = self._resolve_pipeline_definition(typed, context, AUTO_PIPELINE)
        return ImagePipeline(definition.name)

    def preload_model(
        self,
        horde_model_name: str,
        *,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
    ) -> None:
        """Load a model into RAM ahead of inference (the worker's preload path).

        Wraps the HordeCheckpointLoader preload dance so consumers don't need to touch
        hordelib's ComfyUI node classes directly.
        """
        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        HordeCheckpointLoader().load_checkpoint(
            will_load_loras=will_load_loras,
            seamless_tiling_enabled=seamless_tiling_enabled,
            horde_model_name=horde_model_name,
            preloading=True,
        )

    @logfire.instrument("horde.post_process", extract_args=False)
    def post_process(self, payload: PostProcessingPayload | dict) -> ResultingImageReturn:
        """Run a single post-processing operation (upscale, facefix, or strip-background).

        This is the standalone post-processing entry point (the alchemy surface); the embedded
        ``post_processing`` list inside :meth:`basic_inference` flows through here too. Legacy
        dict payloads are accepted (see ``post_processing_payload_from_horde_dict``).
        """
        from hordelib.comfy_horde import log_free_ram

        if isinstance(payload, dict):
            payload = post_processing_payload_from_horde_dict(payload)

        if isinstance(payload, StripBackgroundPayload):
            return ResultingImageReturn(
                image=ImageUtils.strip_background(payload.source_image),
                rawpng=None,
                faults=[],
            )

        log_free_ram()

        context = resolve_post_processing_model(payload.model)
        definition = POST_PROCESSING_REGISTRY.select(payload, context)
        if definition is None:
            raise RuntimeError(f"No post-processing pipeline matched payload type {type(payload).__name__}")

        graph = definition.materialize(payload, context)
        artifacts = self.backend.run_pipeline(graph.to_api_dict(), outputs=definition.outputs)
        if len(artifacts) != 1:
            raise RuntimeError("Expected a single image but got multiple")

        rawpng = artifacts[0].data
        image = Image.open(rawpng)

        # Upscale models produce a fixed multiple of the input size; an explicit rescale request
        # shrinks the result back down. Such results carry no raw PNG stream (legacy parity).
        if isinstance(payload, UpscalePayload) and (payload.rescale_width or payload.rescale_height):
            return ResultingImageReturn(
                image=ImageUtils.shrink_image(image, payload.rescale_width, payload.rescale_height),
                rawpng=None,
                faults=[],
            )

        log_free_ram()
        return ResultingImageReturn(image=image, rawpng=rawpng, faults=[])

    def image_upscale(self, payload: dict) -> ResultingImageReturn:
        """Upscale an image (legacy dict surface; prefer :meth:`post_process`)."""
        logger.debug("image_upscale called")
        return self.post_process(
            UpscalePayload(
                model=payload["model"],
                source_image=payload["source_image"],
                rescale_width=payload.get("width"),
                rescale_height=payload.get("height"),
            ),
        )

    def image_facefix(self, payload: dict) -> ResultingImageReturn:
        """Fix faces in an image (legacy dict surface; prefer :meth:`post_process`).

        ``facefixer_strength`` in the dict is ignored, as it always has been: the legacy
        parameter mapping never wired it to the graph's ``codeformer_fidelity`` input. Typed
        callers can set :attr:`FacefixPayload.fidelity` explicitly.
        """
        logger.debug("image_facefix called")
        return self.post_process(
            FacefixPayload(
                model=payload["model"],
                source_image=payload["source_image"],
            ),
        )
