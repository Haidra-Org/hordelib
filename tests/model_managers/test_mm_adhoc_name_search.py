"""Ad-hoc LoRA and textual inversion name search: query encoding and result selection.

A name lookup goes to CivitAI's model search. The name must reach the API unchanged, and an exact name match
is used ahead of the first search result.

The managers are shells with the CivitAI fetch and item parsing stubbed, so the tests do not use the network
or download files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from hordelib.model_manager.lora import LoraModelManager, LoRaRejectionReason
from hordelib.model_manager.ti import TextualInversionModelManager, TIRejectionReason

_AWKWARD_NAMES = [
    "Negative & Positive Embeddings By Stable Yogi",
    "detail+tweaker",
    "style #2",
    "ネガティブ embedding",
    "100% realism?",
]


def _ti_manager() -> TextualInversionModelManager:
    manager = object.__new__(TextualInversionModelManager)
    manager.read_only = False
    manager.nsfw = True
    manager.model_reference = {}
    return manager


def _lora_manager() -> LoraModelManager:
    manager = object.__new__(LoraModelManager)
    manager.read_only = False
    manager.nsfw = True
    manager.model_reference = {}
    manager._index_ids = {}
    return manager


def _query_of(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


@pytest.mark.parametrize("name", _AWKWARD_NAMES)
def test_ti_search_sends_the_exact_name(name: str) -> None:
    manager = _ti_manager()
    urls: list[str] = []

    def fake_fetch(url: str) -> dict[str, Any]:
        urls.append(url)
        return {"items": []}

    with patch.object(manager, "_fetch_civitai_json", side_effect=fake_fetch):
        key, reason = manager.fetch_adhoc_ti_with_reason(name)

    assert (key, reason) == (None, TIRejectionReason.NOT_FOUND)
    query = _query_of(urls[0])
    assert query["query"] == [name]
    assert query["nsfw"] == ["true"]
    assert query["types"] == ["TextualInversion"]


@pytest.mark.parametrize("name", _AWKWARD_NAMES)
def test_lora_search_sends_the_exact_name(name: str) -> None:
    manager = _lora_manager()
    urls: list[str] = []

    def fake_fetch(url: str) -> dict[str, Any]:
        urls.append(url)
        return {"items": [{"name": name}]}

    with (
        patch.object(manager, "_fetch_civitai_json", side_effect=fake_fetch),
        patch.object(
            manager,
            "_parse_civitai_item_with_reason",
            return_value=(None, LoRaRejectionReason.INVALID),
        ),
    ):
        manager.fetch_adhoc_lora_with_reason(name)

    query = _query_of(urls[0])
    assert query["query"] == [name]
    assert query["types"] == ["LORA"]


def _search_results(requested: str) -> list[dict[str, Any]]:
    return [
        {"id": 56519, "name": "negative_hand Negative Embedding"},
        {"id": 177792, "name": requested.upper()},
    ]


def test_ti_search_prefers_the_exact_name_over_the_first_result() -> None:
    manager = _ti_manager()
    requested = "Negative & Positive Embeddings By Stable Yogi"
    parsed: list[dict[str, Any]] = []

    def fake_parse(item: dict[str, Any], adhoc: bool = False) -> None:
        parsed.append(item)

    with (
        patch.object(manager, "_fetch_civitai_json", return_value={"items": _search_results(requested)}),
        patch.object(manager, "_parse_civitai_item", side_effect=fake_parse),
    ):
        manager.fetch_adhoc_ti_with_reason(requested)

    assert [item["id"] for item in parsed] == [177792]


def test_lora_search_prefers_the_exact_name_over_the_first_result() -> None:
    manager = _lora_manager()
    requested = "Negative & Positive Embeddings By Stable Yogi"
    parsed: list[dict[str, Any]] = []

    def fake_parse(item: dict[str, Any], adhoc: bool = False) -> tuple[None, LoRaRejectionReason]:
        parsed.append(item)
        return None, LoRaRejectionReason.INVALID

    with (
        patch.object(manager, "_fetch_civitai_json", return_value={"items": _search_results(requested)}),
        patch.object(manager, "_parse_civitai_item_with_reason", side_effect=fake_parse),
    ):
        manager.fetch_adhoc_lora_with_reason(requested)

    assert [item["id"] for item in parsed] == [177792]


def test_search_without_an_exact_name_keeps_the_first_result() -> None:
    items = [{"id": 1, "name": "detail tweaker xl"}, {"id": 2, "name": "detail tweaker"}]

    assert TextualInversionModelManager._select_search_item(items, "detail")["id"] == 1
    assert TextualInversionModelManager._select_search_item(items, " Detail Tweaker ")["id"] == 2
