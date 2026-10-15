"""Stable labeling and train/validation split policies."""

from __future__ import annotations

import hashlib
from typing import List, Mapping

from .schema import SchemaError


def ec_top2_labels(ec: str) -> List[str]:
    """Return the level-one and level-two labels represented by an EC value."""
    parts = [part for part in (ec or "").split(".") if part and part != "-"]
    labels = []
    if parts:
        labels.append(parts[0])
    if len(parts) >= 2:
        labels.append("{}.{}".format(parts[0], parts[1]))
    return labels


def deterministic_split(
    mag_id: str,
    source: str,
    mode: str,
    seed: int,
    val_frac: float,
    registry: Mapping[str, str],
    source_splits: Mapping[str, str],
) -> str:
    """Assign a stable split without depending on ingestion order."""
    if mode == "registry":
        return str(registry.get(mag_id, "train"))
    if mode == "source":
        return str(source_splits.get(source, "train"))
    if mode == "random_mag":
        digest = hashlib.md5("{}:{}".format(seed, mag_id).encode("utf-8")).hexdigest()
        return "val" if int(digest, 16) % 10000 < int(val_frac * 10000) else "train"
    raise SchemaError("unknown split mode {}".format(mode))
