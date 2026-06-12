"""The IO boundary of the pipeline layer: resolving payload intent against installed models.

Everything else in ``hordelib.pipeline`` is pure; this module talks to the model managers
(disk lookups, downloads). Imports of manager machinery are deferred to call time so the
pipeline package stays importable without hordelib initialisation.
"""

import random

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels.base import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

from hordelib.pipeline.context import ModelContext, PostProcessingContext
from hordelib.pipeline.patches import ResolvedLora
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.utils.dynamicprompt import DynamicPromptParser


def resolve_post_processing_model(model_name: str) -> PostProcessingContext:
    """Resolve a post-processor model name to its on-disk file, downloading if needed.

    Mirrors the legacy ``_apply_aihorde_compatibility_hacks`` post-processor branch (search the
    codeformer/esrgan/gfpgan managers, take the first file entry), with one addition: a model
    that is known to the reference but missing from disk is downloaded rather than failing
    outright.

    Raises:
        RuntimeError: If the model is not in any post-processor model reference, or could not
            be made available on disk.
    """
    from hordelib.consts import MODEL_CATEGORY_NAMES
    from hordelib.shared_model_manager import SharedModelManager

    if SharedModelManager.manager is None:
        raise RuntimeError("SharedModelManager must be loaded before resolving post-processing models")

    managers = SharedModelManager.manager.get_model_manager_instances(
        [MODEL_CATEGORY_NAMES.codeformer, MODEL_CATEGORY_NAMES.esrgan, MODEL_CATEGORY_NAMES.gfpgan],
    )

    found_in_reference = False
    for manager in managers:
        if model_name not in manager.model_reference:
            continue
        found_in_reference = True

        if not manager.is_model_available(model_name):
            logger.info("Post-processing model not on disk; downloading: model={}", model_name)
            manager.download_model(model_name)

        if not manager.is_model_available(model_name):
            continue

        model_files = manager.get_model_filenames(model_name)
        if len(model_files) == 0 or not isinstance(model_files[0], dict) or "file_path" not in model_files[0]:
            raise RuntimeError(f"Model {model_name} has no files in its reference entry!")

        return PostProcessingContext(model_name=model_name, model_file=str(model_files[0]["file_path"]))

    if not found_in_reference:
        raise RuntimeError(f"Model {model_name} not found in model reference!")
    raise RuntimeError(f"Model {model_name} not found on disk!")


def _compvis_manager():
    """Return the loaded compvis manager, lazily loading it like the legacy compat hacks."""
    from hordelib.consts import MODEL_CATEGORY_NAMES
    from hordelib.shared_model_manager import SharedModelManager

    if SharedModelManager.manager is None or SharedModelManager.manager.compvis is None:
        try:
            SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])
        except Exception as exc:
            raise RuntimeError("Cannot load the compvis model manager required for image generation!") from exc

    compvis = SharedModelManager.manager.compvis
    if compvis is None:
        raise RuntimeError("Cannot resolve image generation models without compvis loaded!")
    return compvis


def resolve_image_model(model_name: str) -> ModelContext:
    """Resolve a horde image-generation model to its on-disk files and baseline facts.

    Raises:
        RuntimeError: If the model is unknown to the reference or missing from disk
            (same messages as the legacy compatibility hacks).
    """
    compvis = _compvis_manager()

    if model_name not in compvis.model_reference:
        raise RuntimeError(f"Model {model_name} not found in model reference!")
    if not compvis.is_model_available(model_name):
        raise RuntimeError(f"Model {model_name} not found on disk!")

    model_files = compvis.get_model_filenames(model_name)
    if len(model_files) == 0 or not isinstance(model_files[0], dict) or "file_path" not in model_files[0]:
        raise RuntimeError(f"Model {model_name} has no files in its reference entry!")

    extra_files = {
        str(file_entry["file_type"]): str(file_entry["file_path"])
        for file_entry in model_files
        if "file_type" in file_entry
    }

    record = compvis.get_model_reference_info(model_name)
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None = None
    if record is not None:
        try:
            baseline = KNOWN_IMAGE_GENERATION_BASELINE(record.baseline)
        except ValueError:
            logger.warning("Model has an unrecognized baseline: model={}, baseline={}", model_name, record.baseline)

    return ModelContext(
        horde_model_name=model_name,
        baseline=baseline,
        main_file=str(model_files[0]["file_path"]),
        extra_files=extra_files,
        is_inpainting_model=bool(record is not None and record.inpainting is True),
    )


def _resolve_tis(payload: ImageGenPayload, context: ModelContext) -> list[GenMetadataEntry]:
    """Validate/download requested TIs and inject their embeddings into the prompts.

    Mutates ``payload.prompt``/``payload.negative_prompt`` (faithful port of the legacy TI
    loop in ``_final_pipeline_adjustments``).
    """
    from hordelib.shared_model_manager import SharedModelManager

    faults: list[GenMetadataEntry] = []
    ti_manager = SharedModelManager.manager.ti if SharedModelManager.manager is not None else None
    if not payload.tis or ti_manager is None:
        return faults

    compvis = _compvis_manager()
    model_details = compvis.get_model_reference_info(context.horde_model_name)

    for ti in payload.tis:
        ti_name_requested = str(ti.name)
        if not ti_manager.is_local_model(ti_name_requested):
            try:
                adhoc_ti = ti_manager.fetch_adhoc_ti(ti_name_requested)
            except Exception:
                logger.bind(ti_name=ti.name).exception("Error fetching adhoc TI")
                faults.append(
                    GenMetadataEntry(type=METADATA_TYPE.ti, value=METADATA_VALUE.download_failed, ref=ti.name),
                )
                adhoc_ti = None
            if not adhoc_ti:
                logger.info(f"Adhoc TI requested '{ti.name}' could not be found in CivitAI. Ignoring!")
                faults.append(
                    GenMetadataEntry(type=METADATA_TYPE.ti, value=METADATA_VALUE.download_failed, ref=ti.name),
                )
                continue

        ti_name = ti_manager.get_ti_name(ti_name_requested)
        if not ti_name:
            continue

        logger.debug("Found valid TI: ti_name={}", ti_name)
        if not ti_manager.do_baselines_match(ti_name, model_details):
            logger.info("Skipped TI due to baseline mismatch: ti_name={}", ti_name)
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.ti, value=METADATA_VALUE.baseline_mismatch, ref=ti_name),
            )
            continue

        ti_strength = ti.strength if isinstance(ti.strength, (float, int)) else 1.0
        ti_id = ti_manager.get_ti_id(str(ti.name))
        logger.info("ti.injecting", ti_id=ti_id, inject_target=ti.inject_ti, strength=ti_strength)
        if ti.inject_ti == "prompt":
            payload.prompt = f"(embedding:{ti_id}:{ti_strength}),{payload.prompt}"
        elif ti.inject_ti == "negprompt":
            had_leading_comma = payload.negative_prompt.startswith(",")
            payload.negative_prompt = f"{payload.negative_prompt},(embedding:{ti_id}:{ti_strength})"
            if not had_leading_comma:
                payload.negative_prompt = payload.negative_prompt.strip(",")
        ti_manager.touch_ti(ti_name)

    return faults


def _resolve_loras(
    payload: ImageGenPayload,
    context: ModelContext,
) -> tuple[list[ResolvedLora], list[GenMetadataEntry]]:
    """Validate/download requested LoRAs, injecting triggers into the prompt.

    Mutates ``payload.prompt`` and replaces ``payload.loras`` with the valid entries
    (faithful port of the legacy LoRA loop in ``_final_pipeline_adjustments``).
    """
    from hordelib.shared_model_manager import SharedModelManager

    faults: list[GenMetadataEntry] = []
    lora_manager = SharedModelManager.manager.lora if SharedModelManager.manager is not None else None
    if not payload.loras or lora_manager is None:
        return [], faults

    compvis = _compvis_manager()
    model_details = compvis.get_model_reference_info(context.horde_model_name)

    job_context = {
        "model": context.horde_model_name,
        "resolution": f"{payload.width}x{payload.height}",
        "steps": payload.ddim_steps,
        "sampler": payload.sampler_name,
        "trigger_source": "adhoc_generation",
    }

    valid_specs = []
    resolved: list[ResolvedLora] = []
    for lora in payload.loras:
        is_version = bool(lora.is_version)
        verstext = " version" if is_version else ""

        if not lora_manager.is_lora_available(str(lora.name), is_version=is_version):
            logger.debug("Adhoc lora not yet downloaded, downloading: lora_name={}", lora.name)
            try:
                adhoc_lora = lora_manager.fetch_adhoc_lora(
                    str(lora.name),
                    is_version=is_version,
                    job_context=job_context,
                )
            except Exception:
                logger.bind(lora_name=lora.name).exception("Error fetching adhoc lora")
                faults.append(
                    GenMetadataEntry(type=METADATA_TYPE.lora, value=METADATA_VALUE.download_failed, ref=lora.name),
                )
                adhoc_lora = None
            if not adhoc_lora:
                logger.info(
                    f"Adhoc lora requested{verstext} '{lora.name} could not be found in CivitAI. Ignoring!",
                )
                faults.append(
                    GenMetadataEntry(type=METADATA_TYPE.lora, value=METADATA_VALUE.download_failed, ref=lora.name),
                )
                continue

        # If a version is requested, the lora name we need is the exact version
        lora_name = str(lora.name) if is_version else lora_manager.get_lora_name(str(lora.name))
        if lora_name is None:
            logger.debug("Lora not found in reference DB, ignoring: lora_name={}", lora.name)
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.lora, value=METADATA_VALUE.download_failed, ref=lora_name),
            )
            continue

        logger.debug("Found valid lora: lora_name={}, version={}", lora_name, verstext)
        if not lora_manager.do_baselines_match(lora_name, model_details, is_version=is_version):
            logger.info("Skipped lora due to baseline mismatch: lora_name={}, is_version={}", lora_name, verstext)
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.lora, value=METADATA_VALUE.baseline_mismatch, ref=lora_name),
            )
            continue

        trigger = None
        if lora.inject_trigger == "any":
            triggers = lora_manager.get_lora_triggers(lora_name, is_version=is_version)
            if triggers:
                trigger = random.choice(triggers)
        elif lora.inject_trigger == "all":
            triggers = lora_manager.get_lora_triggers(lora_name, is_version=is_version)
            if triggers:
                trigger = ", ".join(triggers)
        elif lora.inject_trigger is not None:
            trigger = lora_manager.find_lora_trigger(lora_name, lora.inject_trigger, is_version)
        if trigger:
            # Injected at the start, to avoid throwing it in a negative prompt
            payload.prompt = f"{trigger}, {payload.prompt}"

        filename = lora_manager.get_lora_filename(lora_name, is_version=is_version)
        if filename is None:
            logger.debug("Lora has no downloaded file, ignoring: lora_name={}", lora_name)
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.lora, value=METADATA_VALUE.download_failed, ref=lora_name),
            )
            continue
        lora_manager._touch_lora(lora_name, is_version=is_version)

        valid_specs.append(lora)
        resolved.append(
            ResolvedLora(filename=filename, strength_model=lora.model, strength_clip=lora.clip),
        )

    payload.loras = valid_specs
    return resolved, faults


def resolve_image_generation(
    payload: ImageGenPayload,
    context: ModelContext,
) -> tuple[ImageGenPayload, ModelContext, list[GenMetadataEntry]]:
    """Resolve all adhoc models for an image generation and finalize the prompts.

    Applies, in the legacy order: dynamic-prompt expansion, TI embedding injection, LoRA
    trigger injection. Returns the updated payload, a context enriched with the resolved
    LoRAs, and any faults to report.
    """
    payload = payload.model_copy(deep=False)

    if new_prompt := DynamicPromptParser(payload.seed).parse(payload.prompt):
        payload.prompt = new_prompt

    ti_faults = _resolve_tis(payload, context)
    resolved_loras, lora_faults = _resolve_loras(payload, context)

    from hordelib.shared_model_manager import SharedModelManager

    lora_manager_loaded = SharedModelManager.manager is not None and SharedModelManager.manager.lora is not None
    will_load_loras = bool(resolved_loras) if lora_manager_loaded else bool(payload.loras)

    context = context.model_copy(
        update={"resolved_loras": resolved_loras, "will_load_loras": will_load_loras},
    )
    return payload, context, ti_faults + lora_faults
