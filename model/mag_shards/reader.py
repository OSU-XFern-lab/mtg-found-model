"""Objective-aware mmap reader for immutable MAG shards."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from .manifest import DatasetManifest, ShardRef, load_manifest, sha256_file
from .schema import ESMC_DIM, InputSpec, MISSING_U32, SchemaError, token_column

@dataclass(frozen=True)
class MagLocator:
    shard_index: int
    local_index: int
    mag_id: str
    source_name: str
    split: str
    n_genes: int
    n_contigs: int


@dataclass
class MagExample:
    mag_id: str
    source_name: str
    split: str
    taxonomy: Mapping[str, int]
    contig_gene_offsets: np.ndarray
    esmc: Optional[np.ndarray] = None
    esmc_valid: Optional[np.ndarray] = None
    tokens: Optional[np.ndarray] = None
    ko_indptr: Optional[np.ndarray] = None
    ko_ids: Optional[np.ndarray] = None
    ec_indptr: Optional[np.ndarray] = None
    ec_ids: Optional[np.ndarray] = None
    sample_seed: int = 0

    @property
    def n_genes(self) -> int:
        return int(self.contig_gene_offsets[-1]) if len(self.contig_gene_offsets) else 0

    @property
    def n_contigs(self) -> int:
        return max(0, len(self.contig_gene_offsets) - 1)


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _ShardHandle:
    def __init__(self, path: Path, spec: InputSpec):
        self.path = Path(path)
        with (self.path / "shard.json").open() as handle:
            self.meta = json.load(handle)
        columns = dict(self.meta["columns"])
        missing = set(spec.required_files) - set(columns)
        if missing:
            raise SchemaError("shard {} lacks required columns {}".format(self.path, sorted(missing)))
        self.mags = _read_jsonl(self.path / "mags.jsonl")
        self.mag_gene = np.memmap(self.path / "mag_gene_offsets.u64", dtype="<u8", mode="r")
        self.mag_contig = np.memmap(self.path / "mag_contig_offsets.u64", dtype="<u8", mode="r")
        self.contig_gene = np.memmap(self.path / "contig_gene_offsets.u64", dtype="<u8", mode="r")
        self.esmc = None
        self.esmc_bits = None
        self.tokens = None
        self.ko_indptr = self.ko_ids = None
        self.ec_indptr = self.ec_ids = None
        n_genes = int(self.meta["n_genes"])
        if spec.needs_esmc:
            self.esmc = np.memmap(self.path / "esmc.f16", dtype="<f2", mode="r",
                                  shape=(n_genes, ESMC_DIM))
            valid_path = self.path / "esmc_valid.bit"
            if valid_path.exists():
                self.esmc_bits = np.memmap(valid_path, dtype="u1", mode="r")
        if spec.needs_token:
            self.tokens = np.memmap(self.path / token_column(spec.mode), dtype="<u4", mode="r")
        if spec.needs_ko:
            self.ko_indptr = np.memmap(self.path / "ko_indptr.u64", dtype="<u8", mode="r")
            self.ko_ids = np.memmap(self.path / "ko_ids.u32", dtype="<u4", mode="r")
        if spec.needs_ec:
            self.ec_indptr = np.memmap(self.path / "ec_indptr.u64", dtype="<u8", mode="r")
            self.ec_ids = np.memmap(self.path / "ec_ids.u32", dtype="<u4", mode="r")

    def example(self, local_index: int) -> MagExample:
        g0, g1 = int(self.mag_gene[local_index]), int(self.mag_gene[local_index + 1])
        c0, c1 = int(self.mag_contig[local_index]), int(self.mag_contig[local_index + 1])
        row = self.mags[local_index]
        contigs = np.asarray(self.contig_gene[c0:c1 + 1], dtype=np.uint64) - np.uint64(g0)
        esmc = self.esmc[g0:g1] if self.esmc is not None else None
        valid = None
        if self.esmc is not None:
            if self.esmc_bits is None:
                valid = np.ones(g1 - g0, dtype=bool)
            else:
                positions = np.arange(g0, g1, dtype=np.uint64)
                valid = ((self.esmc_bits[positions // 8] >> (positions % 8).astype(np.uint8)) & 1).astype(bool)
        ko_ptr, ko_ids = _slice_csr(self.ko_indptr, self.ko_ids, g0, g1)
        ec_ptr, ec_ids = _slice_csr(self.ec_indptr, self.ec_ids, g0, g1)
        return MagExample(
            mag_id=str(row["mag_id"]), source_name=str(row["source_name"]), split=str(row["split"]),
            taxonomy={str(k): int(v) for k, v in dict(row.get("taxonomy", {})).items()},
            contig_gene_offsets=contigs,
            esmc=esmc,
            esmc_valid=valid,
            tokens=self.tokens[g0:g1] if self.tokens is not None else None,
            ko_indptr=ko_ptr, ko_ids=ko_ids, ec_indptr=ec_ptr, ec_ids=ec_ids,
        )


def _slice_csr(indptr: Optional[np.ndarray], ids: Optional[np.ndarray], g0: int, g1: int):
    if indptr is None:
        return None, None
    start, end = int(indptr[g0]), int(indptr[g1])
    ptr = np.asarray(indptr[g0:g1 + 1], dtype=np.uint64) - np.uint64(start)
    return ptr, ids[start:end]


class ManifestDataset:
    """Map-style dataset over one pinned manifest.

    Metadata for MAGs is small and loaded once.  Payload files are lazily
    memmapped per process and only for columns selected by `InputSpec`.
    """

    def __init__(self, manifest_path: Path, spec: InputSpec, splits: Optional[Iterable[str]] = None,
                 sources: Optional[Iterable[str]] = None, root_overrides: Optional[Mapping[str, str]] = None,
                 verify_metadata_digest: bool = True,
                 assignments: Optional[Mapping[str, object]] = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.spec = spec
        self.root_overrides = dict(root_overrides or {})
        self.selected_splits = set(splits or ())
        self.selected_sources = set(sources or ())
        self.assignments = assignments
        self._handles = {}
        self.locators = []
        self._shard_paths = []
        self._build_index(verify_metadata_digest)

    def select(self, splits: Optional[Iterable[str]] = None,
               sources: Optional[Iterable[str]] = None) -> "ManifestDataset":
        """Create a filtered view without rescanning or rehashing the shards."""
        result = object.__new__(ManifestDataset)
        result.manifest_path = self.manifest_path
        result.manifest = self.manifest
        result.spec = self.spec
        result.root_overrides = self.root_overrides
        result.selected_splits = set(splits or ())
        result.selected_sources = set(sources or ())
        result.assignments = self.assignments
        result._handles = {}
        result._shard_paths = self._shard_paths
        result.locators = [
            locator for locator in self.locators
            if (not result.selected_splits or locator.split in result.selected_splits)
            and (not result.selected_sources or locator.source_name in result.selected_sources)
        ]
        return result

    def _build_index(self, verify_metadata_digest: bool) -> None:
        seen = set()
        for shard_index, ref in enumerate(self.manifest.shards):
            path = self.manifest.resolve_shard(ref, self.root_overrides)
            if not (path / "COMPLETE").is_file():
                raise SchemaError("incomplete or missing shard: {}".format(path))
            if verify_metadata_digest and sha256_file(path / "shard.json") != ref.metadata_sha256:
                raise SchemaError("shard metadata digest mismatch: {}".format(ref.shard_id))
            with (path / "shard.json").open() as handle:
                meta = json.load(handle)
            missing = set(self.spec.required_files) - set(meta["columns"])
            if missing:
                raise SchemaError("shard {} lacks required columns {}".format(ref.shard_id, sorted(missing)))
            missing_files = [name for name in self.spec.required_files if not (path / name).is_file()]
            if missing_files:
                raise SchemaError("shard {} is missing required files {}".format(ref.shard_id, missing_files))
            mags = _read_jsonl(path / "mags.jsonl")
            if len(mags) != ref.n_mags:
                raise SchemaError("{} MAG metadata count mismatch".format(ref.shard_id))
            self._shard_paths.append(path)
            for local_index, row in enumerate(mags):
                mag_id = str(row["mag_id"])
                if mag_id in seen:
                    raise SchemaError("duplicate MAG ID in dataset: {}".format(mag_id))
                seen.add(mag_id)
                source = str(row["source_name"])
                if self.assignments is not None:
                    assignment = self.assignments.get(mag_id)
                    if assignment is None:
                        raise SchemaError("MAG is absent from the split plan: {}".format(mag_id))
                    if assignment.source_name != source:
                        raise SchemaError("split-plan source mismatch for {}".format(mag_id))
                    split = assignment.label
                else:
                    split = str(row["split"])
                if self.selected_splits and split not in self.selected_splits:
                    continue
                if self.selected_sources and source not in self.selected_sources:
                    continue
                self.locators.append(MagLocator(
                    shard_index=shard_index, local_index=local_index, mag_id=mag_id,
                    source_name=source, split=split, n_genes=int(row["n_genes"]),
                    n_contigs=int(row["n_contigs"])))

    def __len__(self) -> int:
        return len(self.locators)

    def __getitem__(self, index) -> MagExample:
        sample_seed = 0
        if isinstance(index, tuple):
            index, sample_seed = index
        locator = self.locators[index]
        handle = self._handles.get(locator.shard_index)
        if handle is None:
            handle = _ShardHandle(self._shard_paths[locator.shard_index], self.spec)
            self._handles[locator.shard_index] = handle
        example = handle.example(locator.local_index)
        example.sample_seed = int(sample_seed)
        return example

    def gene_counts(self) -> np.ndarray:
        return np.asarray([x.n_genes for x in self.locators], dtype=np.int64)

    def summary(self) -> Mapping[str, object]:
        digest = hashlib.sha256()
        sources = Counter()
        genes = contigs = 0
        for locator in sorted(self.locators, key=lambda value: value.mag_id):
            digest.update("{}\t{}\t{}\n".format(
                locator.mag_id, locator.source_name, locator.split
            ).encode())
            sources[locator.source_name] += 1
            genes += locator.n_genes
            contigs += locator.n_contigs
        return {
            "digest": digest.hexdigest(),
            "mags": len(self.locators),
            "contigs": contigs,
            "genes": genes,
            "sources": dict(sorted(sources.items())),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    @property
    def manifest_sha256(self) -> str:
        return self.manifest.sha256
