"""Measure real post-processing op burdens on the local accelerator and compare them to the seeds.

The :mod:`hordelib.feature_impact` registry ships *seed* estimates for post-processing VRAM peaks; a
scheduler admitting work against them is only as good as their fit to reality. This module runs each
post-processor the way a worker's dedicated post-processing lane does (one ``HordeLib.post_process``
call per op, models kept resident) across a spread of input sizes, samples the true device peak, and
reports measured wall time and VRAM peak next to the registry's estimate for the same configuration.

Run it on any machine with the models installed::

    python -m hordelib.profiling [--out results.json]

Key empirical properties this measurement exists to keep honest (they are also pinned by the
envelope tests in ``tests/test_feature_impact_envelope.py``):

- Upscaler activation peaks are effectively flat with input size (the execution backend tiles the
  activation); wall time scales with output megapixels.
- Face-fixer peaks scale roughly linearly with *input* megapixels (whole-image face detection), and
  in a chain the face-fixer's input is the upscaled image.
- A chain's peak is the maximum of its ops' peaks, not their sum.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_OPS: list[str] = [
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "GFPGAN",
    "CodeFormers",
    "strip_background",
]
_DEFAULT_SIZES: list[int] = [512, 1024, 1536]


@dataclass(frozen=True)
class OpMeasurement:
    """One measured post-processing run beside the registry's estimate for the same configuration."""

    operation: str
    input_px: int
    output_px: int | None
    ok: bool
    error: str
    wall_s: float
    torch_peak_alloc_mb: int
    device_peak_used_delta_mb: int
    seed_estimate_mb: int | None


class _DeviceVramSampler:
    """Poll device-wide free VRAM on a thread and retain the minimum seen (the usage peak)."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._stop = threading.Event()
        self.min_free_mb = float("inf")
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _DeviceVramSampler:
        self._stop.clear()
        self.min_free_mb = float("inf")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            free, _total = self._torch.cuda.mem_get_info()
            self.min_free_mb = min(self.min_free_mb, free / 2**20)
            time.sleep(0.025)

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _seed_estimate_mb(operation: str, input_px: int) -> int | None:
    """Return the registry's post-processing peak estimate for a 1-op job at ``input_px`` square."""
    from hordelib.feature_impact import FEATURE_KIND, estimate_job_burden
    from hordelib.pipeline.constants import max_upscale_factor
    from hordelib.pipeline.payload_pp import PostProcessorKind, classify_post_processor

    kind = classify_post_processor(operation)
    factor: float
    if kind is PostProcessorKind.upscaler:
        features = [FEATURE_KIND.post_processing_upscale]
        factor = float(max_upscale_factor([operation]))
    elif kind is PostProcessorKind.facefixer:
        features = [FEATURE_KIND.post_processing_facefix]
        factor = 1.0
    elif kind is PostProcessorKind.strip_background:
        features = [FEATURE_KIND.strip_background]
        factor = 1.0
    else:
        return None
    burden = estimate_job_burden(
        baseline="stable_diffusion_xl",
        width=input_px,
        height=input_px,
        features=features,
        post_processing_upscale_factor=factor,
    )
    return burden.vram_post_processing_mb


def measure_post_process_burden(
    operations: list[str] | None = None,
    input_sizes: list[int] | None = None,
    repetitions: int = 2,
) -> list[OpMeasurement]:
    """Run each op at each input size on the local accelerator and return the measurements.

    The first run of an op includes its model load (reported like any other run; compare against the
    later repetition for the warm figure). Requires hordelib to already be initialised
    (:func:`hordelib.initialise`) and the post-processing models to be obtainable.
    """
    import torch
    from PIL import Image

    from hordelib.api import HordeLib

    horde = HordeLib(aggressive_unloading=False)
    images_dir = Path(__file__).parent.parent / "images"
    face = Image.open(images_dir / "test_facefix.png").convert("RGB")
    scene = Image.open(images_dir / "test_db0.jpg").convert("RGB")

    measurements: list[OpMeasurement] = []
    for operation in operations or _DEFAULT_OPS:
        source = face if operation in ("GFPGAN", "GFPGANv1.3", "CodeFormers", "RestoreFormer") else scene
        for input_px in input_sizes or _DEFAULT_SIZES:
            image = source.resize((input_px, input_px), Image.LANCZOS)
            for _rep in range(repetitions):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                free_before, _ = torch.cuda.mem_get_info()
                started = time.perf_counter()
                ok, error, output_px = True, "", None
                sampler = _DeviceVramSampler()
                try:
                    with sampler:
                        result = horde.post_process({"model": operation, "source_image": image})
                    if result.image is None:
                        ok, error = False, "no output image"
                    else:
                        output_px = result.image.size[0]
                except Exception as e:
                    ok, error = False, f"{type(e).__name__}: {e}"
                wall_s = time.perf_counter() - started
                torch.cuda.synchronize()
                measurements.append(
                    OpMeasurement(
                        operation=operation,
                        input_px=input_px,
                        output_px=output_px,
                        ok=ok,
                        error=error,
                        wall_s=round(wall_s, 3),
                        torch_peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20),
                        device_peak_used_delta_mb=(
                            round(free_before / 2**20 - sampler.min_free_mb)
                            if sampler.min_free_mb != float("inf")
                            else 0
                        ),
                        seed_estimate_mb=_seed_estimate_mb(operation, input_px),
                    ),
                )
    return measurements


def main() -> None:
    """Initialise hordelib, run the measurement matrix, and print/save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write measurements to this JSON file")
    parser.add_argument("--sizes", type=int, nargs="*", default=None, help="square input sizes in pixels")
    parser.add_argument("--ops", type=str, nargs="*", default=None, help="post-processor names to measure")
    args = parser.parse_args()

    import hordelib

    hordelib.initialise(setup_logging=None, logging_verbosity=0)

    from hordelib.api import SharedModelManager

    SharedModelManager.load_model_managers()

    measurements = measure_post_process_burden(operations=args.ops, input_sizes=args.sizes)
    for m in measurements:
        status = "OK  " if m.ok else f"FAIL {m.error}"
        seed = f"{m.seed_estimate_mb}" if m.seed_estimate_mb is not None else "n/a"
        print(
            f"{m.operation:28s} {m.input_px:5d}px {status} wall={m.wall_s:7.2f}s "
            f"torch_peak={m.torch_peak_alloc_mb:5d}MB device_peak_delta={m.device_peak_used_delta_mb:5d}MB "
            f"seed_estimate={seed}MB",
        )
    if args.out is not None:
        args.out.write_text(json.dumps([asdict(m) for m in measurements], indent=2), encoding="utf-8")
        print(f"wrote {len(measurements)} measurements to {args.out}")


if __name__ == "__main__":
    main()
