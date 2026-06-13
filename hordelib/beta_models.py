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
        pending-model reads. Typically populated by the worker from its existing key.

The PRIMARY URL is taken from ``horde_model_reference_settings.primary_api_url`` (env
``HORDE_MODEL_REFERENCE_PRIMARY_API_URL``), so it stays consistent with the canonical fetch.
"""

from __future__ import annotations

import os

from horde_model_reference import (
    HORDE_SOURCE_ID,
    MODEL_REFERENCE_CATEGORY,
    PENDING_SOURCE_ID,
    ModelReferenceManager,
    PendingModelProvider,
    SourceSelector,
    horde_model_reference_settings,
)
from loguru import logger

BETA_CATEGORIES_ENV_VAR = "HORDELIB_BETA_MODEL_CATEGORIES"
BETA_API_KEY_ENV_VAR = "HORDELIB_BETA_MODELS_API_KEY"


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


def build_pending_provider() -> PendingModelProvider | None:
    """Construct the pending (beta) model provider from the environment, or ``None``.

    Returns ``None`` (with a log line explaining why) when beta is not fully configured:
    no categories opted in, no API key, or no PRIMARY URL to read from.
    """
    categories = beta_model_categories()
    if not categories:
        return None

    apikey = os.getenv(BETA_API_KEY_ENV_VAR)
    if not apikey:
        logger.warning(
            "{} is set but {} is not; beta models will not be loaded.",
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
    return PendingModelProvider(
        primary_api_url=primary_api_url,
        apikey=apikey,
        categories=categories,
    )


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
