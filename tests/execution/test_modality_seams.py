"""Seam test: the execution protocol carries non-image artifacts unmolested.

Audio/video support is explicitly deferred (see docs/modality-readiness.md), but the seams it
will ride through must not silently grow image-only assumptions: ``OutputArtifact`` is
mime-typed and the ``ExecutionBackend`` protocol is artifact-typed, not image-typed.
"""

import io
from typing import Any

from hordelib.execution.interface import ExecutionBackend, OutputArtifact, ProgressCallback, VRAMStats


class _FakeAudioBackend:
    """A minimal ExecutionBackend whose pipeline produces audio artifacts."""

    def start(self) -> None:
        pass

    def run_pipeline(
        self,
        graph: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
        defer_vram_unload: bool = False,
    ) -> list[OutputArtifact]:
        return [OutputArtifact(data=io.BytesIO(b"fLaC..."), mime_type="audio/flac", metadata={"duration_s": 4.2})]

    def interrupt(self) -> None:
        pass

    def free_vram(self) -> None:
        pass

    def free_ram(self) -> None:
        pass

    def vram_stats(self) -> VRAMStats:
        return VRAMStats(total_mb=0, free_mb=0)


def test_backend_protocol_accepts_non_image_artifacts() -> None:
    backend: ExecutionBackend = _FakeAudioBackend()

    artifacts = backend.run_pipeline({})

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.mime_type == "audio/flac"
    assert artifact.data.getvalue().startswith(b"fLaC")
    assert artifact.metadata["duration_s"] == 4.2
