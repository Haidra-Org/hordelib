"""Vocabularies shared by the payload models and pipeline templates.

These are the canonical copies; ``HordeLib``'s class attributes alias them for backwards
compatibility until the legacy payload path is removed.
"""

from collections.abc import Iterable
from enum import StrEnum, auto

SAMPLERS_MAP = {
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_heun": "heun",
    "k_dpm_2": "dpm_2",
    "k_dpm_2_a": "dpm_2_ancestral",
    "k_lms": "lms",
    "k_dpm_fast": "dpm_fast",
    "k_dpm_adaptive": "dpm_adaptive",
    "k_dpmpp_2s_a": "dpmpp_2s_ancestral",
    "k_dpmpp_sde": "dpmpp_sde",
    "k_dpmpp_2m": "dpmpp_2m",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
    "uni_pc_bh2": "uni_pc_bh2",
    "plms": "euler",
    "lcm": "lcm",
    # `dpmsolver` names the diffusers-era DPM-Solver multistep solver that predates this backend.
    # DPM-Solver++ 2M is its closest ComfyUI equivalent; before this entry existed the name fell
    # through the payload validator's clamp and rendered as `euler`, which the horde never advertised.
    "dpmsolver": "dpmpp_2m",
    # ComfyUI-native solvers, named as ComfyUI names them: unlike the `k_` block above they are not
    # k-diffusion samplers, so the prefix would assert a lineage they do not have.
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    # `dpmpp_3m_sde` needs a low-noise sigma schedule to converge. Under ComfyUI's `normal` scheduler
    # it diverges to colour noise (SD15 and SDXL alike, at every step count from 8 to 50); under
    # karras, simple or sgm_uniform it renders cleanly at the same seed and step count. This is a
    # property of the solver, not of this package: it reproduces through ComfyUI's own nodes with no
    # hordelib code in the path. It matters here because a horde payload selects a schedule only
    # indirectly, through the `karras` flag that horde_compat maps to `karras` or `normal`.
    "dpmpp_3m_sde": "dpmpp_3m_sde",
    "ddpm": "ddpm",
    "deis": "deis",
    "ipndm": "ipndm",
    "res_multistep": "res_multistep",
    "gradient_estimation": "gradient_estimation",
    "heunpp2": "heunpp2",
    "er_sde": "er_sde",
    "sa_solver": "sa_solver",
    "ipndm_v": "ipndm_v",
    "dpmpp_2m_sde_heun": "dpmpp_2m_sde_heun",
    "sa_solver_pece": "sa_solver_pece",
    "seeds_2": "seeds_2",
    "seeds_3": "seeds_3",
    "res_multistep_ancestral": "res_multistep_ancestral",
    # The exp_heun pair delegates to `sample_seeds_2`, so its `solver_type` is the phi vocabulary
    # (`phi_1`/`phi_2`), not the midpoint/heun pair in SOLVER_TYPES. Nothing here has to enforce that:
    # a solver_type a sampler does not implement is either filtered out or refused upstream.
    "exp_heun_2_x0": "exp_heun_2_x0",
    "exp_heun_2_x0_sde": "exp_heun_2_x0_sde",
    # CFG++ variants. They rescale the guidance step, so the cfg_scale that suits a request changes with
    # them: the community range is roughly 1.0 to 2.0, and a conventional cfg_scale near 7 oversaturates.
    # They are offered anyway, because the choice of guidance strength belongs to the caller.
    "euler_cfg_pp": "euler_cfg_pp",
    "euler_ancestral_cfg_pp": "euler_ancestral_cfg_pp",
    "dpmpp_2s_ancestral_cfg_pp": "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_2m_cfg_pp": "dpmpp_2m_cfg_pp",
    "res_multistep_cfg_pp": "res_multistep_cfg_pp",
    "res_multistep_ancestral_cfg_pp": "res_multistep_ancestral_cfg_pp",
    "gradient_estimation_cfg_pp": "gradient_estimation_cfg_pp",
}
"""Horde sampler names to ComfyUI sampler names.

A name absent here is clamped to the default sampler rather than rejected (see
``hordelib.pipeline.payload``), so an entry whose value ComfyUI no longer offers degrades silently.
``tests/test_comfy_contract_drift.py`` pins every value against ``comfy.samplers.SAMPLER_NAMES`` so
that a ComfyUI rename fails loudly instead.

ComfyUI's ``_gpu`` sampler variants are deliberately absent: they differ from their siblings only in
which device generates the sampling noise, so they name a worker-side implementation detail rather
than a solver a requester can meaningfully choose between.
"""

# Horde control_type on the left, comfyui_controlnet_aux preprocessor on the right
CONTROLNET_IMAGE_PREPROCESSOR_MAP = {
    "binary": "BinaryPreprocessor",
    "canny": "CannyEdgePreprocessor",
    "hed": "HEDPreprocessor",
    "depth": "LeReS-DepthMapPreprocessor",
    "normal": "MiDaS-NormalMapPreprocessor",
    "openpose": "OpenposePreprocessor",
    "seg": "SemSegPreprocessor",
    "scribble": "ScribblePreprocessor",
    "fakescribbles": "FakeScribblePreprocessor",
    "hough": "M-LSDPreprocessor",  # horde backward compatibility
    "mlsd": "M-LSDPreprocessor",
    "standard_lineart": "LineartStandardPreprocessor",
    "lineart": "LineArtPreprocessor",
    "lineart_anime": "AnimeLineArtPreprocessor",
    "lineart_anime_denoise": "Manga2Anime_LineArt_Preprocessor",
    "pidinet": "PiDiNetPreprocessor",
    "scribble_xdog": "Scribble_XDoG_Preprocessor",
    "scribble_pidinet": "Scribble_PiDiNet_Preprocessor",
    "teed": "TEEDPreprocessor",
    "pyracanny": "PyraCannyPreprocessor",
    "midas_depth": "MiDaS-DepthMapPreprocessor",
    "zoe_depth": "Zoe-DepthMapPreprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "depth_anything_v2": "DepthAnythingV2Preprocessor",
    "normal_bae": "BAE-NormalMapPreprocessor",
    "oneformer_ade20k": "OneFormer-ADE20K-SemSegPreprocessor",
    "oneformer_coco": "OneFormer-COCO-SemSegPreprocessor",
    "recolor_luminance": "ImageLuminanceDetector",
    "recolor_intensity": "ImageIntensityDetector",
    "tile": "TilePreprocessor",
    "tile_ttplanet_guided": "TTPlanet_TileGF_Preprocessor",
    "tile_ttplanet_simple": "TTPlanet_TileSimple_Preprocessor",
    "color": "ColorPreprocessor",
    "shuffle": "ShufflePreprocessor",
}

ONNXRUNTIME_GATED_PREPROCESSORS: frozenset[str] = frozenset()
"""comfyui_controlnet_aux preprocessors that need the onnxruntime-backed ``controlnet`` extra to run.

No currently exposed preprocessor requires ONNX Runtime. ``OpenposePreprocessor`` is the classic Torch
implementation at the pinned auxiliary-node revision; DWPose is a separate, unexposed node.
"""

CONTROLNET_MODEL_MAP = {
    "canny": "diff_control_sd15_canny_fp16.safetensors",
    "hed": "diff_control_sd15_hed_fp16.safetensors",
    "depth": "diff_control_sd15_depth_fp16.safetensors",
    "normal": "control_normal_fp16.safetensors",
    "openpose": "control_openpose_fp16.safetensors",
    "seg": "control_seg_fp16.safetensors",
    "scribble": "control_scribble_fp16.safetensors",
    "fakescribbles": "control_scribble_fp16.safetensors",
    "mlsd": "control_mlsd_fp16.safetensors",
    "hough": "control_mlsd_fp16.safetensors",
    "binary": "control_scribble_fp16.safetensors",
    "standard_lineart": "control_v11p_sd15_lineart_fp16.safetensors",
    "lineart": "control_v11p_sd15_lineart_fp16.safetensors",
    "lineart_anime": "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
    "lineart_anime_denoise": "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
    "pidinet": "diff_control_sd15_hed_fp16.safetensors",
    "scribble_xdog": "control_scribble_fp16.safetensors",
    "scribble_pidinet": "control_scribble_fp16.safetensors",
    "teed": "diff_control_sd15_hed_fp16.safetensors",
    "pyracanny": "diff_control_sd15_canny_fp16.safetensors",
    "midas_depth": "diff_control_sd15_depth_fp16.safetensors",
    "zoe_depth": "diff_control_sd15_depth_fp16.safetensors",
    "depth_anything": "diff_control_sd15_depth_fp16.safetensors",
    "depth_anything_v2": "diff_control_sd15_depth_fp16.safetensors",
    "normal_bae": "control_v11p_sd15_normalbae_fp16.safetensors",
    "oneformer_ade20k": "control_seg_fp16.safetensors",
    "oneformer_coco": "control_seg_fp16.safetensors",
    "recolor_luminance": "ioclab_sd15_recolor.safetensors",
    "recolor_intensity": "ioclab_sd15_recolor.safetensors",
    "tile": "control_v11f1e_sd15_tile_fp16.safetensors",
    "tile_ttplanet_guided": "control_v11f1e_sd15_tile_fp16.safetensors",
    "tile_ttplanet_simple": "control_v11f1e_sd15_tile_fp16.safetensors",
    "color": "t2iadapter_color_sd14v1.pth",
    # ComfyUI keys global average pooling on a "_shuffle" in the filename, which the shuffle
    # controlnet needs to guide on the whole image rather than position by position; renaming the
    # file loses that.
    "shuffle": "control_v11e_sd15_shuffle_fp16.safetensors",
}
"""Horde control_type to controlnet model filename."""

# Rough order-of-magnitude of the checkpoint(s) each control_type's comfyui_controlnet_aux detector
# downloads from the HuggingFace hub on first use, into AUX_ANNOTATOR_CKPTS_PATH (see hordelib.preload).
# These are *annotator* (preprocessor) weights, distinct from the controlnet model weights in
# CONTROLNET_MODEL_MAP. Used only for download/disk previews (e.g. the worker's benchmark planner), so a
# ROM estimate is sufficient; the real fetch verifies actual sizes. canny/scribble/mlsd are pure-cv2 and
# download nothing (0). Keep keys aligned with CONTROLNET_IMAGE_PREPROCESSOR_MAP.
CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES = {
    "canny": 0,
    "hed": 56 * 1024**2,  # ControlNetHED.pth
    "depth": 800 * 1024**2,  # LeReS: res101.pth (~470MB) + latest_net_G.pth (~320MB)
    "normal": 470 * 1024**2,  # Intel/dpt-hybrid-midas pytorch_model.bin
    "openpose": 200 * 1024**2,  # body/hand/face pose models (pure Torch at the pinned aux revision)
    "seg": 170 * 1024**2,  # UniFormer segmentation
    "scribble": 0,
    "fakescribbles": 56 * 1024**2,  # FakeScribble runs the HED detector underneath
    "hough": 6 * 1024**2,  # M-LSD
    "mlsd": 6 * 1024**2,
    "binary": 0,
    "standard_lineart": 0,
    "lineart": 35 * 1024**2,
    "lineart_anime": 208 * 1024**2,
    "lineart_anime_denoise": 165 * 1024**2,
    "pidinet": 3 * 1024**2,
    "scribble_xdog": 0,
    "scribble_pidinet": 3 * 1024**2,
    "teed": 1024**2,
    "pyracanny": 0,
    "midas_depth": 470 * 1024**2,
    "zoe_depth": 1_400 * 1024**2,
    "depth_anything": 1_400 * 1024**2,
    "depth_anything_v2": 1_280 * 1024**2,
    "normal_bae": 280 * 1024**2,
    "oneformer_ade20k": 850 * 1024**2,
    "oneformer_coco": 850 * 1024**2,
    "recolor_luminance": 0,
    "recolor_intensity": 0,
    "tile": 0,
    "tile_ttplanet_guided": 0,
    "tile_ttplanet_simple": 0,
    "color": 0,
    "shuffle": 0,
}
"""Horde control_type to an estimated annotator-checkpoint download size in bytes (ROM)."""


def controlnet_annotator_download_bytes(control_types: Iterable[str | None]) -> int:
    """Return the summed ROM annotator-download size for *control_types*.

    Unknown or ``None`` control types contribute 0 (they may be pure-cv2 or not annotator-backed), so a
    consumer can pass a level's raw control types without filtering. Duplicates are de-duplicated first:
    an annotator is fetched once and shared on disk, so sweeping the same type twice costs it once.

    Args:
        control_types: Horde ``control_type`` values (``None`` entries allowed and ignored).

    Returns:
        The total estimated annotator download size in bytes (0 when none are annotator-backed).
    """
    distinct = {control_type for control_type in control_types if control_type}
    return sum(CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES.get(control_type, 0) for control_type in distinct)


# Horde upscaler name (KNOWN_UPSCALERS value) to its linear scale factor. The upscaler enlarges the
# generated image by this factor on each axis, so the post-processing activation peak scales with the
# *output* megapixels (factor**2 the generation megapixels), not the generation resolution. A 4x upscale
# of a 1 MP image produces a 16 MP tensor, which is the dominant VRAM cost of the post-processing phase
# (the model weights themselves are tens of MB). Used by the feature-impact estimate to size that peak.
UPSCALER_SCALE_FACTORS = {
    "BACKEND_DEFAULT": 4,  # the worker's default upscaler is a 4x ESRGAN; assume the larger factor
    "RealESRGAN_x4plus": 4,
    "RealESRGAN_x2plus": 2,
    "RealESRGAN_x4plus_anime_6B": 4,
    "NMKD_Siax": 4,
    "4x_AnimeSharp": 4,
    "4xNomos8kSC": 4,
    "4xLSDIRplus": 4,
    "4xNomosWebPhoto_RealPLKSR": 4,
    "4xNomos2_realplksr_dysample": 4,
    "4xNomos2_hq_dat2": 4,
    "2xModernSpanimationV1": 2,
}
"""Horde upscaler name to its linear (per-axis) scale factor (ROM)."""

_DEFAULT_UPSCALE_FACTOR = 4
"""Assumed factor for an upscaler absent from the ROM: err high so the activation peak is not under-sized."""


def upscaler_scale_factor(name: str | None) -> int:
    """Return the linear scale factor for upscaler *name*, or the conservative default when unknown.

    ``None`` returns 1 (no upscaler, no enlargement). An unrecognised upscaler returns
    :data:`_DEFAULT_UPSCALE_FACTOR` rather than 1, so a new upscaler the ROM has not learned yet
    over-reserves rather than under-reserving the post-processing activation peak.
    """
    if name is None:
        return 1
    return UPSCALER_SCALE_FACTORS.get(name, _DEFAULT_UPSCALE_FACTOR)


def max_upscale_factor(names: Iterable[str | None]) -> int:
    """Return the largest linear scale factor among *names*, or 1 when none enlarge the image.

    A job may request several upscalers; the output size (and thus the activation peak) is driven by the
    largest factor. ``None`` entries and an empty iterable contribute the no-op factor of 1.
    """
    factors = [upscaler_scale_factor(name) for name in names if name is not None]
    return max(factors) if factors else 1


SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting", "remix"]

SCHEDULERS = [
    "normal",
    "karras",
    "simple",
    "ddim_uniform",
    "sgm_uniform",
    "exponential",
    "beta",
    "linear_quadratic",
    "kl_optimal",
    "align_your_steps",
    "gits",
]
"""Sigma schedules offered to callers: every schedule ComfyUI names, plus the two it implements as nodes.

``normal`` stays first because it is the payload default and the schedule a horde request resolves to
when its karras flag is false. ``tests/test_comfy_contract_drift.py`` pins the list against
``comfy.samplers.SCHEDULER_NAMES``, which matters because an unrecognised schedule is substituted
rather than rejected: ``KSampler.__init__`` falls back to its first entry. The two members of
:data:`SIGMA_GENERATOR_SCHEDULES` are exempt from that pin and carried by
:mod:`hordelib.execution.sigma_schedules` instead, for the reason given there.
"""


class SigmaGeneratorSchedule(StrEnum):
    """A schedule ComfyUI produces from a node rather than from a name ``calculate_sigmas`` accepts.

    Both come from published research schedules rather than from a closed-form function of the model's
    sigma range: `align_your_steps` interpolates NVIDIA's per-family noise levels and `gits` indexes a
    table of step-count-specific schedules. Upstream exposes each as a scheduler node emitting a SIGMAS
    output, which only the custom-sampler graph shape can consume; the graphs this package runs take a
    schedule by name, so the name is carried beside the graph instead.
    """

    ALIGN_YOUR_STEPS = auto()
    GITS = auto()


SIGMA_GENERATOR_SCHEDULES: frozenset[str] = frozenset(str(schedule) for schedule in SigmaGeneratorSchedule)
"""The schedule names ComfyUI's ``calculate_sigmas`` does not know, spelled as a request spells them."""

SIGMA_GENERATOR_GRAPH_SCHEDULE = "normal"
"""The schedule a graph input carries while a sigma generator supplies the real one.

The input still has to name a schedule ComfyUI recognises: ``KSampler.__init__`` silently substitutes
its first entry for anything else, and prompt validation rejects a value outside the node's declared
list. The value is never used, because the generator replaces the sigmas the node would compute from it.
"""

SOLVER_TYPES = frozenset({"midpoint", "heun"})
"""Accepted ``solver_type`` values, which only ``dpmpp_2m_sde`` and ``dpmpp_2m_sde_heun`` take.

Other samplers carrying a parameter of the same name use a different vocabulary and are not covered
here: `seeds_2` takes `phi_1`/`phi_2` and the `exp_heun_2_x0` pair defaults to `phi_2`. A value from
this set handed to one of those would name nothing they implement, so the per-sampler applicability
filter in :mod:`hordelib.execution.sampler_options` is what keeps them apart.

Lives here rather than beside the sampler-option plumbing because it is request vocabulary: the payload
validates against it, and the payload layer must not depend on the execution layer.
"""


class SolverOption(StrEnum):
    """A solver tuning argument a request can set, spelled as ComfyUI's sampler functions name it.

    The members are the keyword-argument names in ``comfy.k_diffusion.sampling``, so a value here can be
    passed straight through to the sampler that accepts it. ``ORDER`` and ``MAX_ORDER`` are the same
    concept under two upstream spellings (`lms` and `dpm_adaptive` take `order`, the multistep solvers
    take `max_order`), not two separate controls.
    """

    ETA = auto()
    S_NOISE = auto()
    S_CHURN = auto()
    S_TMIN = auto()
    S_TMAX = auto()
    SOLVER_TYPE = auto()
    ORDER = auto()
    MAX_ORDER = auto()


SOLVER_OPTION_FALLBACK_BOUNDS: dict[SolverOption, tuple[float, float]] = {
    SolverOption.ETA: (0.0, 100.0),
    SolverOption.S_NOISE: (0.0, 100.0),
    SolverOption.S_CHURN: (0.0, 100.0),
    SolverOption.S_TMIN: (0.0, 100.0),
    SolverOption.S_TMAX: (0.0, 100.0),
    SolverOption.ORDER: (2.0, 4.0),
    SolverOption.MAX_ORDER: (2.0, 4.0),
}
"""Bounds applied to a numeric solver option when no per-sampler range is known.

The eta and noise-scale bounds are the ones ComfyUI's own sampler nodes declare. The churn thresholds
`s_tmin` and `s_tmax` bound the sigma window in which churn is injected (upstream defaults 0 and
infinity); they are sigma values, so any positive number is meaningful and the upper bound only keeps
a request finite. The order bound starts at 2 because order 1 raises inside `deis` and `ipndm` rather
than degrading, and it is uniformly narrower than what individual samplers allow: `lms` accepts 1 to
100 and `dpm_adaptive` 2 to 3. Those per-sampler ranges are read through
:func:`hordelib.execution.sampler_options.option_bounds`, which falls back to this mapping.

``SolverOption.SOLVER_TYPE`` is absent because it is a vocabulary, not a range; see :data:`SOLVER_TYPES`.
"""

FLOW_SHIFT_BOUNDS: tuple[float, float] = (0.0, 100.0)
"""Bounds ComfyUI's model-sampling nodes declare for a flow model's timestep shift.

The same range is declared by ``ModelSamplingSD3``, ``ModelSamplingAuraFlow`` and the two shift
inputs of ``ModelSamplingFlux``, so one range covers every graph the knob can reach.
"""

SCHEDULE_SENSITIVE_SAMPLERS = frozenset({"dpmpp_3m_sde"})
"""Samplers that need a low-noise sigma schedule to converge at all.

On a schedule in :data:`DIVERGENT_SCHEDULES` these return high-frequency colour noise instead of an
image, at every step count rather than only at low ones. Membership is evidence-based: `dpmpp_3m_sde`
was verified diverging on SD15 from 8 to 50 steps and on SDXL at 25, and it reproduces through
ComfyUI's own nodes with none of this package in the path, so the constraint belongs to the solver.
Its one-order-lower sibling `dpmpp_2m_sde` is unaffected and must not be added here.
"""

DIVERGENT_SCHEDULES = frozenset({"normal"})
"""Schedules on which a :data:`SCHEDULE_SENSITIVE_SAMPLERS` member diverges.

Only `normal` is listed, because only `normal` was confirmed to diverge while `karras`, `simple`,
`sgm_uniform` and `exponential` were confirmed to converge. `beta` and `ddim_uniform` measured as
suspect but were never confirmed either way, so they are deliberately absent rather than guessed at.
"""

SCHEDULE_SENSITIVE_FALLBACK = "simple"
"""The schedule substituted for a sensitive sampler that would otherwise diverge.

`simple` rather than `karras`, on measurement: sweeping `dpmpp_3m_sde` across all nine schedules at 8
and 25 steps, `simple` and `sgm_uniform` were the only two that converge at *both* step counts. `karras`
converges at 25 steps but returns dithered colour noise at 8, so substituting it would have fixed the
schedule a low-step request landed on while leaving the image broken. A substitute has to hold across
the step range, not just at the step count it was checked at.
"""


def resolve_schedule(sampler_name: str | None, scheduler: str | None) -> tuple[str | None, bool]:
    """Return the schedule to run for *sampler_name*, and whether it was substituted.

    A sensitive sampler asked for a divergent schedule is moved onto
    :data:`SCHEDULE_SENSITIVE_FALLBACK` rather than failed or served as noise: the request is
    physically serviceable, just not on the schedule named. Callers are expected to disclose the
    substitution on the result rather than silently altering what was asked for.

    Args:
        sampler_name: The horde sampler name for the request.
        scheduler: The schedule the request resolved to before this check.

    Returns:
        The schedule to run, and whether it differs from the one passed in.
    """
    if sampler_name in SCHEDULE_SENSITIVE_SAMPLERS and scheduler in DIVERGENT_SCHEDULES:
        return SCHEDULE_SENSITIVE_FALLBACK, True
    return scheduler, False
