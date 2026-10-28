"""Opt-in "beta" models sourced from a horde-model-reference PRIMARY's pending queue.

Historically a beta model could be run by pointing horde-model-reference's GitHub backend
at a branch. With a service-only source of truth that trick is gone, so instead the
PRIMARY's pending-queue entries are treated as beta models and exposed through a
:class:`~horde_model_reference.PendingModelProvider` registered under the ``"pending"``
source. This module owns the (env-driven) opt-in: which categories are in beta, building
the provider, and choosing the source selector so beta records win over canonical ones.

Environment variables:
    HORDELIB_BETA_MODEL_CATEGORIES: Comma-separated category values to opt into beta
        (e.g. ``"image_generation"``). Empty/unset disables beta entirely.
    HORDELIB_BETA_MODELS_API_KEY: A reader-level AI-Horde API key used to authenticate the
        pending-model reads. Typically populated by the worker from its existing key; unset
        falls back to the anonymous key, which the PRIMARY accepts for pending reads. Setting
        it to an empty string disables beta.
    HORDELIB_BETA_MODELS_DIR: A directory holding one ``{category}.json`` file of pending
        records per opted-in category, keyed by model name. When set, the pending provider
        serves those files and makes no network request, so the API key and the PRIMARY URL
        are not required. A process that owns the pending read (the worker's parent) writes
        the files and sets this for the processes it spawns, so those processes never fetch
        the pending queue themselves.

The PRIMARY URL is taken from ``horde_model_reference_settings.primary_api_url`` (env
``HORDE_MODEL_REFERENCE_PRIMARY_API_URL``), so it stays consistent with the canonical fetch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from horde_model_reference import (
    HORDE_SOURCE_ID,
    MODEL_REFERENCE_CATEGORY,
    PENDING_SOURCE_ID,
    ModelProvider,
    ModelReferenceManager,
    PendingModelProvider,
    SourceSelector,
    StaticModelProvider,
    horde_model_reference_settings,
)
from loguru import logger

BETA_CATEGORIES_ENV_VAR = "HORDELIB_BETA_MODEL_CATEGORIES"
BETA_API_KEY_ENV_VAR = "HORDELIB_BETA_MODELS_API_KEY"
BETA_MODELS_DIR_ENV_VAR = "HORDELIB_BETA_MODELS_DIR"

ANONYMOUS_API_KEY = "0000000000"


def beta_model_categories() -> set[MODEL_REFERENCE_CATEGORY]:
    """Return the categories opted into beta via the environment.

    Unknown category values are logged and skipped rather than raising, so a typo never
    breaks model loading.
    """
    raw = os.getenv(BETA_CATEGORIES_ENV_VAR, "")
    categories: set[MODEL_REFERENCE_CATEGORY] = set()
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        try:
            categories.add(MODEL_REFERENCE_CATEGORY(name))
        except ValueError:
            logger.warning(
                "Ignoring unknown category {!r} in {}; valid values: {}",
                name,
                BETA_CATEGORIES_ENV_VAR,
                [c.value for c in MODEL_REFERENCE_CATEGORY],
            )
    return categories


def build_pending_provider(
    *,
    timeout_seconds: float | None = None,
    retry_max_attempts: int | None = None,
) -> ModelProvider | None:
    """Construct the pending (beta) model provider from the environment, or ``None``.

    Returns ``None`` (with a log line explaining why) when beta is not configured: no categories
    opted in, no PRIMARY URL to read from, or the API key explicitly set empty. An *unset* key
    resolves to :data:`ANONYMOUS_API_KEY`.

    When :data:`BETA_MODELS_DIR_ENV_VAR` is set, the provider is a :class:`StaticModelProvider`
    built from that directory's files (see :func:`build_file_pending_provider`) and the key, the
    PRIMARY URL and both overrides are ignored.

    Args:
        timeout_seconds: Per-request HTTP timeout override. ``None`` keeps the provider's default.
            Callers on a latency-sensitive path (a catalog build, a UI) pass a tight budget here;
            beta is additive, so giving up quickly costs only the overlay, never the canonical set.
        retry_max_attempts: Retry-count override, same rationale.
    """
    categories = beta_model_categories()
    if not categories:
        return None

    beta_models_dir = os.getenv(BETA_MODELS_DIR_ENV_VAR, "")
    if beta_models_dir:
        return build_file_pending_provider(Path(beta_models_dir), categories)

    apikey = os.getenv(BETA_API_KEY_ENV_VAR, ANONYMOUS_API_KEY)
    if not apikey:
        logger.warning(
            "{} is set but {} is empty; beta models will not be loaded.",
            BETA_CATEGORIES_ENV_VAR,
            BETA_API_KEY_ENV_VAR,
        )
        return None

    primary_api_url = horde_model_reference_settings.primary_api_url
    if not primary_api_url:
        logger.warning(
            "Beta models requested but HORDE_MODEL_REFERENCE_PRIMARY_API_URL is unset; "
            "beta models require a PRIMARY service and cannot be served from GitHub.",
        )
        return None

    logger.info(
        "Beta models enabled for categories {} via PRIMARY {}.",
        sorted(c.value for c in categories),
        primary_api_url,
    )
    overrides: dict[str, float | int] = {}
    if timeout_seconds is not None:
        overrides["timeout_seconds"] = timeout_seconds
    if retry_max_attempts is not None:
        overrides["retry_max_attempts"] = retry_max_attempts
    return PendingModelProvider(
        primary_api_url=primary_api_url,
        apikey=apikey,
        categories=categories,
        **overrides,  # type: ignore[arg-type]  # working around this is a pain, and the overrides are trivial anyway
    )


def build_file_pending_provider(
    directory: Path,
    categories: set[MODEL_REFERENCE_CATEGORY],
) -> StaticModelProvider:
    """Serve the pending (beta) records for *categories* from ``{directory}/{category}.json``.

    Each file is a mapping of model name to record fields, the shape
    :meth:`StaticModelProvider.from_raw` reads. A missing, unreadable or invalid file is an
    empty category, so a process reading the files never fails a reference load over beta,
    which only adds to the canonical records. Only the files of *categories* are read.

    Args:
        directory: The directory the files live in.
        categories: The categories opted into beta.

    Returns:
        A provider registered under :data:`PENDING_SOURCE_ID` that serves every category in
        *categories*, empty where its file did not supply records.
    """
    raw_by_category: dict[MODEL_REFERENCE_CATEGORY | str, dict[str, dict[str, object]]] = {}
    for category in sorted(categories, key=lambda c: c.value):
        raw_by_category[category] = _read_pending_file(directory / f"{category.value}.json", category)
    logger.debug(
        "Beta models for categories {} served from the files in {}.",
        sorted(c.value for c in categories),
        directory,
    )
    return StaticModelProvider.from_raw(PENDING_SOURCE_ID, raw_by_category)


def _read_pending_file(path: Path, category: MODEL_REFERENCE_CATEGORY) -> dict[str, dict[str, object]]:
    """Return the raw records in *path*, or an empty mapping when the file does not supply valid records."""
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("Beta {} is empty because no file is at {}.", category.value, path)
        return {}
    except (OSError, ValueError) as read_error:
        logger.debug("Beta {} is empty because the file at {} could not be read: {}", category.value, path, read_error)
        return {}

    if not isinstance(loaded, dict) or not all(isinstance(fields, dict) for fields in loaded.values()):
        logger.debug("Beta {} is empty because the file at {} is not a mapping of records.", category.value, path)
        return {}
    raw: dict[str, dict[str, object]] = {str(name): fields for name, fields in loaded.items()}

    try:
        StaticModelProvider.from_raw(PENDING_SOURCE_ID, {category: raw})
    except ValueError as validation_error:
        logger.debug(
            "Beta {} is empty because the file at {} holds an invalid record: {}",
            category.value,
            path,
            validation_error,
        )
        return {}
    return raw


def beta_source_for(
    category: MODEL_REFERENCE_CATEGORY,
    manager: ModelReferenceManager,
) -> SourceSelector:
    """Return the source selector to load *category* with.

    Beta-opted categories that have a registered pending provider resolve to
    ``[PENDING_SOURCE_ID, HORDE_SOURCE_ID]`` so pending (beta) records override canonical
    ones on name collisions. Everything else uses the canonical source only. The provider
    registration is checked so a missing/failed provider degrades to canonical rather than
    raising.
    """
    if category in beta_model_categories() and manager.get_provider(PENDING_SOURCE_ID) is not None:
        return [PENDING_SOURCE_ID, HORDE_SOURCE_ID]
    return HORDE_SOURCE_ID
