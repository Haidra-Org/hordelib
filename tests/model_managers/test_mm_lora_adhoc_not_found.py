"""Behavioural tests for the ad-hoc LoRA not-found verdict versus a transient CivitAI outage.

A LoRA whose metadata endpoint returns a definitive client error (a not-found or unauthorized status) can
never be placed on disk: the reference simply does not exist. That is a terminal rejection, categorically
different from a transient outage (a timeout, a connection error, a 5xx) which may succeed on retry. These
tests encode the contract that :meth:`LoraModelManager.fetch_adhoc_lora_with_reason` surfaces a terminal
rejection reason for the definitive-not-found case (rather than laundering it into a bare ``(None, None)``
that the caller cannot distinguish from a retryable failure), while a transient outage stays retryable and
raises :class:`CivitAIDown` at the metadata layer. They also guard that a parse-level invalid item is
surfaced as a rejection.

They build a :class:`LoraModelManager` shell wired only for the metadata-fetch and rejection-classification
path (no CivitAI network, no GPU, no real downloads): the definitive-status cases drive the real
``_fetch_civitai_json`` with a mocked HTTP layer so the status-code handling runs, and the parse-level and
outage cases stub the JSON fetch directly.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from hordelib.exceptions import CivitAIDown
from hordelib.model_manager.lora import MAX_ADHOC_LORA_SIZE_MB, LoraModelManager, LoRaRejectionReason
from hordelib.model_manager.ti import TextualInversionModelManager, TIRejectionReason

_FETCH_TARGET = "hordelib.model_manager.civitai_adhoc.requests.get"
_SLEEP_TARGET = "hordelib.model_manager.civitai_adhoc.time.sleep"
_MODELS_URL = "https://civitai.com/api/v1/models/123456"


class _FakeResponse:
    """A stand-in ``requests`` response whose ``raise_for_status`` fails with the given HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return {}


def _adhoc_manager(*, nsfw: bool = True) -> LoraModelManager:
    """Build a LoraModelManager shell wired for the metadata-fetch and rejection-classification path only."""
    manager = object.__new__(LoraModelManager)
    manager.read_only = False
    manager.nsfw = nsfw
    manager.model_reference = {}
    manager._index_ids = {}
    manager._default_lora_ids = []
    manager.METRIC_PREFIX = "lora"
    manager.total_retries_attempted = 0
    # The real _fetch_civitai_json records to these metrics on the error path; inert stand-ins keep it running.
    manager._metric_metadata_duration = Mock()
    manager._metric_network_errors = Mock()
    manager._metric_retries = Mock()
    return manager


def _adhoc_ti_manager(*, nsfw: bool = True) -> TextualInversionModelManager:
    """Build a TextualInversionModelManager shell wired for the metadata-fetch/rejection path only.

    The textual-inversion fetch shares the same ``_fetch_civitai_json`` layer, so its 404 handling must
    surface the same terminal not-found verdict the LoRA path does.
    """
    manager = object.__new__(TextualInversionModelManager)
    manager.read_only = False
    manager.nsfw = nsfw
    manager.model_reference = {}
    manager.METRIC_PREFIX = "ti"
    manager.total_retries_attempted = 0
    manager._metric_metadata_duration = Mock()
    manager._metric_network_errors = Mock()
    manager._metric_retries = Mock()
    return manager


def _too_large_item() -> dict[str, object]:
    """A CivitAI item that parses to a TOO_LARGE rejection (a complete version whose weight exceeds the cap)."""
    return {
        "name": "chonk",
        "id": 777,
        "nsfw": False,
        "modelVersions": [
            {
                "id": 7771,
                "baseModel": "SD 1.5",
                "availability": "Public",
                "trainedWords": [],
                "files": [
                    {
                        "primary": True,
                        "name": "chonk.safetensors",
                        "sizeKB": (MAX_ADHOC_LORA_SIZE_MB + 100) * 1024,
                        "downloadUrl": "http://example/chonk",
                        "hashes": {"SHA256": "a" * 64},
                    },
                ],
            },
        ],
    }


def _invalid_item() -> dict[str, object]:
    """A CivitAI item that parses to an INVALID rejection (no primary safetensors file present)."""
    return {"name": "ghost", "id": 555, "nsfw": False, "modelVersions": [{"id": 5551, "files": []}]}


def test_metadata_not_found_yields_terminal_rejection() -> None:
    """A 404 for an ad-hoc LoRA surfaces a terminal rejection reason, not a laundered ``(None, None)``.

    A definitive not-found means the reference does not exist and never will, so the caller must be able to
    tell it apart from a retryable failure. The laundering defect collapses the 404 to ``CivitAIDown``, caught
    as a bare ``(None, None)`` carrying no rejection reason.
    """
    manager = _adhoc_manager()
    with patch(_SLEEP_TARGET), patch(_FETCH_TARGET, return_value=_FakeResponse(404)):
        key, reason = manager.fetch_adhoc_lora_with_reason(123456)

    assert key is None
    assert reason is not None


def test_metadata_unauthorized_yields_terminal_rejection() -> None:
    """A 401 for an ad-hoc LoRA surfaces a terminal rejection reason, mirroring the not-found contract.

    An unauthorized status is as definitive as a not-found (retrying the same public metadata request will
    not change it), so it must not be laundered into a retryable ``(None, None)`` either.
    """
    manager = _adhoc_manager()
    with patch(_SLEEP_TARGET), patch(_FETCH_TARGET, return_value=_FakeResponse(401)):
        key, reason = manager.fetch_adhoc_lora_with_reason(123457)

    assert key is None
    assert reason is not None


def test_get_lora_metadata_distinguishes_not_found_from_outage() -> None:
    """The metadata layer distinguishes a definitive not-found from a transient outage.

    A transient outage (an exhausted retry budget) is retryable and raises :class:`CivitAIDown`. A definitive
    not-found is terminal and must be reported through a different channel, so it must not be laundered into
    the same ``CivitAIDown`` an outage raises.
    """
    manager = _adhoc_manager()

    # Control: an exhausted transient fetch (None from the JSON layer) stays a retryable CivitAIDown outage.
    with patch.object(manager, "_fetch_civitai_json", return_value=None):
        with pytest.raises(CivitAIDown):
            manager.get_lora_metadata(_MODELS_URL)

    # A definitive not-found must be distinguishable from that outage: anything but CivitAIDown is acceptable.
    with patch(_SLEEP_TARGET), patch(_FETCH_TARGET, return_value=_FakeResponse(404)):
        try:
            manager.get_lora_metadata(_MODELS_URL)
        except CivitAIDown:
            pytest.fail("a definitive not-found was laundered into a transient CivitAIDown outage")
        except Exception:
            pass


def test_ti_metadata_not_found_yields_terminal_rejection() -> None:
    """A 404 for an ad-hoc textual inversion surfaces a terminal NOT_FOUND rejection, mirroring the LoRA path.

    The textual-inversion fetch shares ``_fetch_civitai_json``, so a definitive not-found must not launder into
    a reason-less ``(None, None)`` there either: the caller must be able to memoize the doomed reference.
    """
    manager = _adhoc_ti_manager()
    with patch(_SLEEP_TARGET), patch(_FETCH_TARGET, return_value=_FakeResponse(404)):
        key, reason = manager.fetch_adhoc_ti_with_reason(123456)

    assert key is None
    assert reason == TIRejectionReason.NOT_FOUND


def test_parse_level_invalid_is_surfaced_as_rejection() -> None:
    """A parse-level invalid item is surfaced as a rejection reason, not laundered into ``(None, None)``.

    An item CivitAI returns that cannot be parsed into a usable record (here, no primary weight file) is a
    terminal rejection: the caller must learn the LoRA is unusable rather than see a bare ``(None, None)``.
    """
    manager = _adhoc_manager()
    with patch.object(manager, "_fetch_civitai_json", return_value={"items": [_invalid_item()]}):
        key, reason = manager.fetch_adhoc_lora_with_reason("ghost")

    assert key is None
    assert reason is not None


def test_too_large_lora_returns_its_rejection() -> None:
    """A LoRA whose weight exceeds the ad-hoc cap returns a TOO_LARGE rejection reason (control).

    This is the already-working rejection contract the not-found case must be brought in line with: a terminal
    property of the file is surfaced as a concrete rejection reason.
    """
    manager = _adhoc_manager()
    with patch.object(manager, "_fetch_civitai_json", return_value={"items": [_too_large_item()]}):
        key, reason = manager.fetch_adhoc_lora_with_reason("chonk")

    assert key is None
    assert reason == LoRaRejectionReason.TOO_LARGE


def test_transient_outage_stays_retryable_without_rejection() -> None:
    """A transient outage yields no rejection reason and raises CivitAIDown at the metadata layer (control).

    The terminal/transient distinction cuts both ways: a genuine outage must keep raising ``CivitAIDown`` (so
    it stays retryable) and must produce no terminal rejection reason through ``fetch_adhoc_lora_with_reason``,
    proving the not-found handling does not swallow retryable failures into terminal verdicts.
    """
    manager = _adhoc_manager()

    with patch.object(manager, "_fetch_civitai_json", return_value=None):
        with pytest.raises(CivitAIDown):
            manager.get_lora_metadata(_MODELS_URL)

    with patch.object(manager, "_fetch_civitai_json", return_value=None):
        key, reason = manager.fetch_adhoc_lora_with_reason(123456)

    assert key is None
    assert reason is None
