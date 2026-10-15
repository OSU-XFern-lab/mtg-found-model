"""Versioned storage schema and objective-to-column dependency rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from tokenization import ALL_MODES, DISCRETE_MODES

SCHEMA_NAME = "mag-columnar-shards"
SCHEMA_VERSION = 1
ESMC_DIM = 1152
MISSING_U32 = np.uint32(0xFFFFFFFF)
IGNORE_INDEX = -100
ALL_TOKEN_MODES = tuple(DISCRETE_MODES)
TAXONOMY_RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
VALID_SPLITS = ("train", "val", "test")

DTYPES = {
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "u32": np.dtype("<u4"),
    "u64": np.dtype("<u8"),
    "f16": np.dtype("<f2"),
}


class SchemaError(ValueError):
    """The shard or requested training configuration violates the schema."""


def dtype_for(name: str) -> np.dtype:
    try:
        return DTYPES[name]
    except KeyError:
        raise SchemaError("unsupported shard dtype {!r}".format(name))


def token_column(mode: str) -> str:
    if mode not in ALL_TOKEN_MODES:
        raise SchemaError("{} is not a discrete token mode".format(mode))
    return "token.{}.u32".format(mode)


@dataclass(frozen=True)
class InputSpec:
    """Exact storage/tensor dependencies for one train or probe configuration.

    A zero-weight objective has no data dependency. `probe_tasks` allows the
    same reader to expose labels needed by downstream probes without teaching
    the training script about storage columns.
    """

    mode: str
    loss_weights: Mapping[str, float]
    probe_tasks: Tuple[str, ...] = ()
    taxonomy_ranks: Tuple[str, ...] = TAXONOMY_RANKS

    def __post_init__(self) -> None:
        if self.mode not in ALL_MODES:
            raise SchemaError("unknown mode {!r}; expected one of {}".format(self.mode, ALL_MODES))
        unknown = set(self.loss_weights) - {"gene_id", "ko", "ec", "mag_tax", "patch_disc"}
        if unknown:
            raise SchemaError("unknown loss objective(s): {}".format(sorted(unknown)))
        if float(self.loss_weights.get("gene_id", 0.0)) > 0 and self.mode not in ALL_TOKEN_MODES:
            raise SchemaError("gene_id requires a discrete input/target mode")
        bad_ranks = set(self.taxonomy_ranks) - set(TAXONOMY_RANKS)
        if bad_ranks:
            raise SchemaError("unknown taxonomy rank(s): {}".format(sorted(bad_ranks)))

    def active(self, name: str) -> bool:
        return float(self.loss_weights.get(name, 0.0)) > 0.0 or name in self.probe_tasks

    @property
    def needs_esmc_input(self) -> bool:
        return self.mode == "esmc"

    @property
    def needs_esmc_target(self) -> bool:
        return self.active("patch_disc")

    @property
    def needs_esmc(self) -> bool:
        return self.needs_esmc_input or self.needs_esmc_target

    @property
    def needs_token(self) -> bool:
        return self.mode in ALL_TOKEN_MODES

    @property
    def needs_ko(self) -> bool:
        return self.active("ko")

    @property
    def needs_ec(self) -> bool:
        return self.active("ec")

    @property
    def needs_taxonomy(self) -> bool:
        return self.active("mag_tax")

    @property
    def required_files(self) -> Tuple[str, ...]:
        files = ["mags.jsonl", "mag_gene_offsets.u64", "mag_contig_offsets.u64",
                 "contig_gene_offsets.u64"]
        if self.needs_esmc:
            files.append("esmc.f16")
        if self.needs_token:
            files.append(token_column(self.mode))
        if self.needs_ko:
            files.extend(("ko_indptr.u64", "ko_ids.u32"))
        if self.needs_ec:
            files.extend(("ec_indptr.u64", "ec_ids.u32"))
        return tuple(files)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "loss_weights": {k: float(v) for k, v in self.loss_weights.items()},
            "probe_tasks": list(self.probe_tasks),
            "taxonomy_ranks": list(self.taxonomy_ranks),
            "required_files": list(self.required_files),
        }


def available_token_modes(columns: Iterable[str]) -> Tuple[str, ...]:
    present = set(columns)
    return tuple(mode for mode in ALL_TOKEN_MODES if token_column(mode) in present)
