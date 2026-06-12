"""Per-model-family pipeline registrations.

Importing this package builds the default registry. Adding a new model family means adding a
module here with its templates and `register(...)` calls — no central mapping edits.
"""

from hordelib.pipeline.families.image import build_default_registry

__all__ = ["build_default_registry"]
