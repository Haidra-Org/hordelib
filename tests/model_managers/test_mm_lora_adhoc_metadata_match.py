"""Behavioural tests for ad-hoc LoRA name matching and terminal-mismatch classification.

CivitAI stores a LoRA's display name with spaces where a job requests it with underscores (a job asking for
``add_detail`` gets back an item named ``add detail - slider``). The name comparison in the ad-hoc match path
must fold underscores to spaces on both sides before its substring/fuzzy logic, so a request that differs only
by that separator is recognised as the same LoRA and proceeds to download rather than being discarded as a
mismatch.

When a returned item genuinely names a different LoRA and no cached entry covers the request, the outcome is a
terminal mismatch: :meth:`LoraModelManager.fetch_adhoc_lora_with_reason` must surface
``LoRaRejectionReason.MISMATCH`` rather than a reason-less ``(None, None)`` the caller cannot tell apart from a
retryable outage. A cached fallback for the request is still honoured with no rejection, and a transient outage
stays reason-less.

These build a :class:`LoraModelManager` shell wired only for the metadata-fetch and match-classification path
(no CivitAI network, no GPU, no real downloads): the JSON fetch is stubbed and the download enqueue is
replaced so the match path can be observed to reach (or not reach) the download attempt.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from hordelib.exceptions import CivitAIDown
from hordelib.model_manager.lora import LoraModelManager, LoRaRejectionReason


def _adhoc_manager(*, nsfw: bool = True) -> LoraModelManager:
    """Build a LoraModelManager shell wired for the metadata-fetch and match-classification path only."""
    manager = object.__new__(LoraModelManager)
    manager.read_only = False
    manager.nsfw = nsfw
    manager.model_reference = {}
    manager._index_ids = {}
    manager._index_version_ids = {}
    manager._index_orig_names = {}
    manager._default_lora_ids = []
    manager.METRIC_PREFIX = "lora"
    manager.total_retries_attempted = 0
    manager._metric_metadata_duration = Mock()
    manager._metric_network_errors = Mock()
    manager._metric_retries = Mock()
    return manager


def _named_item(name: str, *, model_id: int = 4242, version_id: int = 42421) -> dict[str, object]:
    """A CivitAI item that parses to a valid, downloadable record with the given display *name*."""
    return {
        "name": name,
        "id": model_id,
        "nsfw": False,
        "modelVersions": [
            {
                "id": version_id,
                "baseModel": "SD 1.5",
                "availability": "Public",
                "trainedWords": [],
                "files": [
                    {
                        "primary": True,
                        "name": "weight.safetensors",
                        "sizeKB": 144 * 1024,
                        "downloadUrl": "http://example/weight",
                        "hashes": {"SHA256": "b" * 64},
                    },
                ],
            },
        ],
    }


def test_underscore_name_matches_spaced_metadata_and_reaches_download() -> None:
    """A request differing from the returned name only by ``_``/space folds to a match and downloads.

    A job asking for ``add_detail`` against an item named ``add detail - slider`` must recognise the two as the
    same LoRA (the underscore folds to a space, making the request a substring of the returned name) and proceed
    to the download attempt rather than discarding the metadata as a mismatch.
    """
    manager = _adhoc_manager()
    item = _named_item("add detail - slider")

    with (
        patch.object(manager, "_fetch_civitai_json", return_value={"items": [item]}),
        patch.object(manager, "_enqueue_download") as enqueue,
    ):
        key, reason = manager.fetch_adhoc_lora_with_reason("add_detail", timeout=None)

    assert reason is None, "a folded name match must not surface a rejection reason"
    assert enqueue.called, "a folded name match must reach the download enqueue"
    assert key is None  # timeout=None returns before the download completes, but the enqueue was reached


def test_genuine_name_mismatch_without_cache_yields_terminal_mismatch() -> None:
    """A returned item naming a genuinely different LoRA, with no cached fallback, is a terminal MISMATCH.

    The caller must be able to memoize the doomed request rather than see a reason-less ``(None, None)`` it
    cannot distinguish from a retryable failure.
    """
    manager = _adhoc_manager()
    item = _named_item("add detail - slider")

    with patch.object(manager, "_fetch_civitai_json", return_value={"items": [item]}):
        key, reason = manager.fetch_adhoc_lora_with_reason("totally_other_lora", timeout=None)

    assert key is None
    assert reason == LoRaRejectionReason.MISMATCH


def test_metadata_mismatch_with_cached_fallback_returns_cached_key_without_rejection() -> None:
    """A metadata mismatch that has a cached entry for the request returns the cached key, no rejection (control).

    The mismatch verdict is only terminal when nothing local covers the request; an existing cached reference
    must still be served, and no rejection reason may be surfaced for it.
    """
    manager = _adhoc_manager()
    manager.model_reference = {"cachedlora": Mock()}
    wrong_item = _named_item("wrong result")

    with (
        patch.object(manager, "_fetch_civitai_json", return_value={"items": [wrong_item]}),
        patch.object(manager, "_touch_lora"),
    ):
        key, reason = manager.fetch_adhoc_lora_with_reason("cachedlora", timeout=None)

    assert key == "cachedlora"
    assert reason is None


def test_transient_outage_stays_reason_less_through_match_path() -> None:
    """A transient outage keeps raising CivitAIDown and yields a reason-less ``(None, None)`` (control).

    The mismatch classification must not swallow a retryable outage into a terminal verdict: a ``None`` metadata
    body still surfaces as a retryable failure carrying no rejection reason.
    """
    manager = _adhoc_manager()

    with patch.object(manager, "_fetch_civitai_json", return_value=None):
        with pytest.raises(CivitAIDown):
            manager.get_lora_metadata("https://civitai.com/api/v1/models/123456")

    with patch.object(manager, "_fetch_civitai_json", return_value=None):
        key, reason = manager.fetch_adhoc_lora_with_reason(123456)

    assert key is None
    assert reason is None
