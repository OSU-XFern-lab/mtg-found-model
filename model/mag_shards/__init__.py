"""Immutable, columnar MAG training shards.

The package intentionally keeps storage concerns separate from training.  The
schema/manifest/writer modules do not import torch; reader/collate are the only
pieces used by a training process.
"""

from .schema import (
    ALL_TOKEN_MODES,
    ESMC_DIM,
    MISSING_U32,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    InputSpec,
    SchemaError,
)

__all__ = [
    "ALL_TOKEN_MODES",
    "ESMC_DIM",
    "MISSING_U32",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "InputSpec",
    "SchemaError",
]
