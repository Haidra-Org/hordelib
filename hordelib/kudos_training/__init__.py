"""Kudos model training and the feature contract it is trained against.

The manifest and its encoder (:mod:`hordelib.kudos_training.manifest`) are the single definition of
what a kudos model reads from a job payload. They replace the hand-written encoders that previously
existed in three places, which is where pricing drift entered: hordelib's training script,
``examples/kudos.py``, and the AI-Horde server's ``payload_to_vector``. See
``docs/kudos-model-training.md`` for the design.

The policy ledger (:mod:`hordelib.kudos_training.ledger`) is the other half of the contract: the
model predicts measurable resource-seconds, and every cost that is not a per-job duration is a named
line item there, composed over the prediction by ``compose_user_price``.

The manifest and encoder are numpy and standard library only, and the ledger is standard library
only, so the server and the worker can price a job without importing torch. The pipeline stages
(assemble, sanitize, train, evaluate; ``kudos-train`` CLI) additionally need the ``kudos-training``
extra (pandas, pyarrow, lightgbm, optuna) and guard those imports, so importing this package stays
cheap.
"""

from hordelib.kudos_training.ledger import (
    DEFAULT_LEDGER_PATH,
    LEDGER_FILENAME,
    KudosPolicyLedger,
    LedgerUnit,
    LineItem,
    PayloadFeatures,
    PredictedSeconds,
    PriceBreakdown,
    PricedFeature,
    PricingBasis,
    UnknownBaselineError,
    WorkerRewardBreakdown,
    compose_user_price,
    compose_worker_reward,
    default_ledger,
    load_ledger,
)
from hordelib.kudos_training.manifest import (
    DEFAULT_MANIFEST_PATH,
    MANIFEST_FILENAME,
    CategoricalFeature,
    EncodingResult,
    FloatFeature,
    KudosFeatureManifest,
    MultiHotFeature,
    SamplerSemanticsReference,
    default_manifest,
    load_manifest,
    to_vector,
)

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_MANIFEST_PATH",
    "LEDGER_FILENAME",
    "MANIFEST_FILENAME",
    "CategoricalFeature",
    "EncodingResult",
    "FloatFeature",
    "KudosFeatureManifest",
    "KudosPolicyLedger",
    "LedgerUnit",
    "LineItem",
    "MultiHotFeature",
    "PayloadFeatures",
    "PredictedSeconds",
    "PriceBreakdown",
    "PricedFeature",
    "PricingBasis",
    "SamplerSemanticsReference",
    "UnknownBaselineError",
    "WorkerRewardBreakdown",
    "compose_user_price",
    "compose_worker_reward",
    "default_ledger",
    "default_manifest",
    "load_ledger",
    "load_manifest",
    "to_vector",
]
