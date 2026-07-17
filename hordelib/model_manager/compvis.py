import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

from horde_model_reference import COMPONENT_PURPOSE_FOLDERS, component_relative_path
from horde_model_reference.component_hash import UnsupportedContainerError
from horde_model_reference.component_identity import ensure_sidecar
from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from loguru import logger

from hordelib.model_manager.base import BaseModelManager


def _custom_model_entry_to_record(model_name: str, entry: dict[str, Any]) -> ImageGenerationModelRecord:
    """Convert a HORDELIB_CUSTOM_MODELS JSON entry into a pydantic record.

    Custom model entries use the legacy flat-dict shape::

        {"name": ..., "baseline": ..., "type": "ckpt", "config": {"files": [{"path": "<abs path>"}]}}

    Normalising them here means the rest of hordelib only ever sees pydantic records.
    """
    download_entries: list[DownloadRecord] = []
    for file_entry in entry.get("config", {}).get("files", []):
        file_path = file_entry.get("path")
        if not file_path:
            continue
        download_entries.append(
            DownloadRecord(
                file_name=file_path,
                file_url=file_entry.get("url", ""),
                sha256sum=file_entry.get("sha256sum", "FIXME"),
            ),
        )

    if not download_entries:
        raise ValueError(f"Custom model {model_name} has no usable file entries")

    return ImageGenerationModelRecord(
        record_type=MODEL_REFERENCE_CATEGORY.image_generation,
        name=entry.get("name", model_name),
        description=entry.get("description", "Custom model (HORDELIB_CUSTOM_MODELS)"),
        baseline=entry.get("baseline", "stable_diffusion_1"),
        nsfw=bool(entry.get("nsfw", False)),
        inpainting=bool(entry.get("inpainting", False)),
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        config=GenericModelRecordConfig(download=download_entries),
    )


class CompVisModelManager(BaseModelManager[ImageGenerationModelRecord]):
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        kwargs.pop("model_category", None)  # consumed by this subclass
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.image_generation,
            download_reference=download_reference,
            **kwargs,
        )

    @override
    def load_model_database(self) -> None:
        super().load_model_database()

        num_custom_models = 0

        try:
            extra_models_path_str = os.getenv("HORDELIB_CUSTOM_MODELS")
            if extra_models_path_str:
                extra_models_path = Path(extra_models_path_str)
                if extra_models_path.exists():
                    extra_models = json.loads((extra_models_path).read_text())
                    for mname in extra_models:
                        # Avoid clobbering
                        if mname in self.model_reference:
                            continue
                        try:
                            self.model_reference[mname] = _custom_model_entry_to_record(mname, extra_models[mname])
                        except Exception as e:
                            logger.error("Skipping invalid custom model: model={}, error={}", mname, e)
                            continue
                        if self.is_model_available(mname):
                            self.available_models.append(mname)
                        num_custom_models += 1

        except json.decoder.JSONDecodeError as e:
            logger.error("Custom model database is not valid JSON: path={}, error={}", self.models_db_path, e)
            raise

        logger.info("Loaded custom models: count={}, path={}", num_custom_models, os.getenv("HORDELIB_CUSTOM_MODELS"))

    @override
    def download_model(
        self,
        model_name: str,
        *,
        callback: Callable[[int, int], None] | None = None,
        connections: int = 1,
    ) -> bool | None:
        """Download *model_name*, then build its component-identity sidecar(s) and extract the embedded VAE.

        The sidecar lets a later VAE-only decode load a small standalone VAE (and share it across models with
        byte-identical VAE weights) instead of subset-loading the whole checkpoint. Sidecar failures never
        fail the download: they are logged and swallowed inside :meth:`_ensure_sidecars_for_model`.
        """
        outcome = super().download_model(model_name, callback=callback, connections=connections)
        if outcome:
            self._ensure_sidecars_for_model(model_name)
        return outcome

    def ensure_component_identity_sweep(self) -> None:
        """Ensure a component-identity sidecar (and extracted VAE) for every on-disk image checkpoint.

        Idempotent and safe to call repeatedly: a fresh sidecar is left untouched. A single bad file never
        aborts the sweep (a legitimate ``.ckpt`` pickle that cannot be identified torch-free is logged and
        skipped, as is a file that cannot be read). This is deliberately NOT called at manager init: the
        hashing pass is scheduled explicitly by the worker so a child process never pays for it at boot.
        """
        for model_name in list(self.model_reference):
            if not self.is_model_available(model_name):
                continue
            self._ensure_sidecars_for_model(model_name)

    def _ensure_sidecars_for_model(self, model_name: str) -> None:
        """Ensure sidecars for *model_name*'s on-disk monolithic checkpoint file(s)."""
        record = self._get_generic_record(model_name)
        if record is None:
            return
        for ckpt_path in self._monolithic_checkpoint_paths(record):
            self._ensure_one_sidecar(ckpt_path)

    def _monolithic_checkpoint_paths(self, record: GenericModelRecord) -> list[Path]:
        """Return the on-disk paths of *record*'s monolithic checkpoint files (the VAE-embedding files).

        Only files that stay in this manager's own folder are candidates; component files whose
        ``file_purpose`` routes them to a sibling folder (the standalone ``vae``/``text_encoders``) are the
        extraction targets, not sources, so they are skipped.
        """
        paths: list[Path] = []
        for download in record.config.download:
            if download.file_purpose in COMPONENT_PURPOSE_FOLDERS:
                continue
            relative = component_relative_path(download.file_name, download.file_purpose)
            candidate = self.model_folder_path / relative
            if candidate.exists():
                paths.append(candidate)
        return paths

    def _ensure_one_sidecar(self, ckpt_path: Path) -> None:
        """Ensure one checkpoint's sidecar and extracted VAE, swallowing per-file container/IO errors."""
        try:
            ensure_sidecar(ckpt_path, extract_vae=True, extraction_dir=self._vae_extraction_dir())
        except UnsupportedContainerError:
            # A pickle .ckpt cannot be identified torch-free; that is expected, not an error.
            logger.info("Skipping component-identity sidecar for non-safetensors checkpoint: path={}", ckpt_path)
        except OSError as sidecar_error:
            logger.warning(
                "Could not build component-identity sidecar: path={}, error={}",
                ckpt_path,
                sidecar_error,
            )

    def _vae_extraction_dir(self) -> Path:
        """Return the ``vae`` sibling folder extracted standalone VAEs are written into.

        The same folder ComfyUI searches for VAEs (see :mod:`hordelib.execution.model_dirs`), so a VAE
        extracted here is found by the loader's standalone-VAE path without any further wiring.
        """
        return self._weights_root / COMPONENT_PURPOSE_FOLDERS["vae"]
