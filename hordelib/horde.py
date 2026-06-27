# horde.py
# Main interface for the horde to this library.
from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from enum import Enum, auto

import logfire
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.apimodels.base import (
    GenMetadataEntry,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_FACEFIXERS
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.pipeline import constants as pipeline_constants
from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.post_processing import POST_PROCESSING_REGISTRY
from hordelib.pipeline.graph import ComfyGraph
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
    ) -> tuple[ComfyGraph, ImageGenPayload, ModelContext, list[GenMetadataEntry]]:
        """The typed pipeline flow: resolve -> normalize -> validate -> select -> materialize.

        Returns the fully materialized graph, the typed payload, the resolved model context,
        and any faults to attach to the results.
        """
        from hordelib.pipeline.families.image import DEFAULT_REGISTRY
        from hordelib.pipeline.horde_compat import normalize_horde_payload, resize_sources_to_request
        from hordelib.pipeline.resolution import resolve_image_generation, resolve_image_model

        model = payload.get("model")
        if model is None:
            raise RuntimeError("No model specified in payload")

        context = resolve_image_model(str(model))
        normalized, compat_faults = normalize_horde_payload(payload, context)
        typed = ImageGenPayload.from_horde_dict(normalized)
        typed = resize_sources_to_request(typed)
        typed, context, resolution_faults = resolve_image_generation(typed, context)

        template = DEFAULT_REGISTRY.select(typed, context)
        if template is None:
            # The registry has a priority-0 catch-all, so this is purely defensive
            logger.warning("No pipeline matched payload; falling back to stable_diffusion")
            template = DEFAULT_REGISTRY.get("stable_diffusion")
            assert template is not None

        logger.info(
            "Pipeline validation complete",
            pipeline=template.name,
            faults_count=len(compat_faults) + len(resolution_faults),
        )
        graph = template.materialize(typed, context)
        return graph, typed, context, compat_faults + resolution_faults

    @logfire.instrument("horde.inference", extract_args=False)
    def _inference(
        self,
        payload: dict,
        *,
        single_image_expected: bool = True,
        comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn] | ResultingImageReturn:
        start_time = time.time()

        graph, typed, context, faults = self._materialize_image_graph(payload)

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

    @logfire.instrument("horde.basic_inference", extract_args=False)
    def basic_inference(
        self,
        payload: dict | ImageGenerateJobPopResponse,
        *,
        progress_callback: Callable[[ProgressReport], None] | None = None,
        defer_vram_unload: bool = False,
    ) -> list[ResultingImageReturn]:
        post_processing_requested: list[str] | None = None
        if isinstance(payload, dict):
            post_processing_requested = payload.get("post_processing")
            n_iter = payload.get("n_iter", 1)
        else:
            n_iter = getattr(payload.payload, "n_iter", 1)

        logger.info(
            "Basic inference started",
            n_iter=n_iter,
            post_processing_count=len(post_processing_requested) if post_processing_requested else 0,
            has_callback=progress_callback is not None,
        )

        faults = []
        if isinstance(payload, ImageGenerateJobPopResponse):  # TODO move this to _inference()
            for post_processor_requested in payload.payload.post_processing:
                if post_processing_requested is None:
                    post_processing_requested = []
                post_processing_requested.append(post_processor_requested)
                logger.debug("Post-processing requested: processors={}", post_processor_requested)

            sub_payload = payload.payload.model_dump()

            def handle_images(
                payload: ImageGenerateJobPopResponse,
                image_type: str,
                get_downloaded_image_func: Callable,
            ):
                image = getattr(payload, image_type)

                if image is not None and "http" in image:
                    image = get_downloaded_image_func()

                    if image is None:
                        logger.error(
                            f"{image_type.capitalize()} is a URL but wasn't downloaded, "
                            "this is not supported in this context. Run the `async_download_*` methods first.",
                        )

                        return None

                return image

            source_image = handle_images(
                payload,
                "source_image",
                payload.get_downloaded_source_image,
            )
            if source_image is None:
                logger.info("No source image found in payload.")

            mask_image = handle_images(
                payload,
                "source_mask",
                payload.get_downloaded_source_mask,
            )
            if mask_image is None:
                logger.info("No mask image found in payload.")

            extra_source_images = payload.extra_source_images

            if extra_source_images is not None:
                extra_source_images = payload.get_downloaded_extra_source_images()
                if extra_source_images is not None:
                    logger.info("Using downloaded extra source images: count={}", len(extra_source_images))
                else:
                    logger.info("No extra source images found in payload.")

            esi_to_remove = []
            if extra_source_images is not None:
                for esi in extra_source_images:
                    if "http" in esi.image:
                        logger.warning("Extra source image is a URL, this is not supported in this context.")
                        esi_to_remove.append(esi)

                extra_source_images = [esi for esi in extra_source_images if esi not in esi_to_remove]
            # If its a base64 encoded image, decode it
            if isinstance(source_image, str):
                try:
                    source_image_bytes = base64.b64decode(source_image)
                    source_image_pil = Image.open(io.BytesIO(source_image_bytes))
                    sub_payload["source_image"] = source_image_pil
                except Exception as err:
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_image,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    logger.warning("Failed to parse source image, falling back to text2img: error={}", err)

            if isinstance(mask_image, str):
                try:
                    mask_image_bytes = base64.b64decode(mask_image)
                    mask_image_pil = Image.open(io.BytesIO(mask_image_bytes))
                    sub_payload["source_mask"] = mask_image_pil
                except Exception as err:
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_mask,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    logger.warning("Failed to parse source mask, ignoring it: error={}", err)

            if isinstance(extra_source_images, list):
                extra_source_images_sub = []
                for esi_index, esi in enumerate(extra_source_images):
                    try:
                        esi_bytes = base64.b64decode(esi.image)
                        esi_pil = Image.open(io.BytesIO(esi_bytes))
                        extra_source_images_sub.append(
                            {
                                "image": esi_pil,
                                "strength": esi.strength,
                            },
                        )
                    except Exception as err:
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.extra_source_images,
                                value=METADATA_VALUE.parse_failed,
                                ref=str(esi_index),
                            ),
                        )
                        logger.warning(
                            "Failed to parse extra source image, ignoring: index={}, error={}",
                            esi_index,
                            err,
                        )
                sub_payload["extra_source_images"] = extra_source_images_sub

            sub_payload["source_processing"] = payload.source_processing
            sub_payload["model"] = payload.model
            payload = sub_payload

        if progress_callback is not None:
            try:
                progress_callback(
                    ProgressReport(
                        hordelib_progress_state=ProgressState.started,
                        hordelib_message="Initiating inference...",
                        progress=0,
                    ),
                )
            except Exception:
                logger.exception("Progress callback failed")

        def _default_progress_callback(comfyui_progress: ComfyUIProgress, message: str) -> None:
            nonlocal progress_callback
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

            if progress_callback is not None:
                try:
                    progress_callback(
                        ProgressReport(
                            hordelib_progress_state=ProgressState.progress,
                            hordelib_message=message,
                            comfyui_progress=comfyui_progress,
                        ),
                    )
                except Exception:
                    logger.exception("Progress callback failed")

        result = self._inference(
            payload,
            single_image_expected=False,
            comfyui_progress_callback=_default_progress_callback,
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
        template = POST_PROCESSING_REGISTRY.select(payload, context)
        if template is None:
            raise RuntimeError(f"No post-processing pipeline matched payload type {type(payload).__name__}")

        graph = template.materialize(payload, context)
        artifacts = self.backend.run_pipeline(graph.to_api_dict())
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
