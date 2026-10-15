"""Canonical manifests, path resolution, hashing, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .schema import SCHEMA_NAME, SCHEMA_VERSION, VALID_SPLITS, SchemaError


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write and rename on the destination filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(Path(path), canonical_json_bytes(value))


@dataclass(frozen=True)
class ShardRef:
    shard_id: str
    root: str
    path: str
    metadata_sha256: str
    n_mags: int
    n_contigs: int
    n_genes: int
    bytes_by_column: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "root": self.root,
            "path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "n_mags": int(self.n_mags),
            "n_contigs": int(self.n_contigs),
            "n_genes": int(self.n_genes),
            "bytes_by_column": {k: int(v) for k, v in self.bytes_by_column.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShardRef":
        return cls(
            shard_id=str(value["shard_id"]),
            root=str(value["root"]),
            path=str(value["path"]),
            metadata_sha256=str(value["metadata_sha256"]),
            n_mags=int(value["n_mags"]),
            n_contigs=int(value["n_contigs"]),
            n_genes=int(value["n_genes"]),
            bytes_by_column={str(k): int(v) for k, v in dict(value.get("bytes_by_column", {})).items()},
        )


@dataclass
class DatasetManifest:
    dataset_id: str
    generation: int
    storage_roots: Mapping[str, str]
    shards: List[ShardRef]
    totals: Mapping[str, int]
    sources: Mapping[str, object]
    representations: Mapping[str, object]
    vocabularies: Mapping[str, object]
    split: Mapping[str, object]
    created_utc: str
    builder: Mapping[str, object]
    parent_sha256: Optional[str] = None
    validation: Mapping[str, object] = field(default_factory=dict)
    unavailable_token_modes: Mapping[str, str] = field(default_factory=dict)
    indexes: Mapping[str, object] = field(default_factory=dict)
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION
    state: str = "complete"
    manifest_path: Optional[Path] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, object]:
        value = {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "state": self.state,
            "dataset_id": self.dataset_id,
            "generation": int(self.generation),
            "created_utc": self.created_utc,
            "parent_sha256": self.parent_sha256,
            "storage_roots": dict(self.storage_roots),
            "shards": [s.to_dict() for s in self.shards],
            "totals": {k: int(v) for k, v in self.totals.items()},
            "sources": dict(self.sources),
            "representations": dict(self.representations),
            "vocabularies": dict(self.vocabularies),
            "split": dict(self.split),
            "builder": dict(self.builder),
            "validation": dict(self.validation),
            "unavailable_token_modes": dict(self.unavailable_token_modes),
            "indexes": dict(self.indexes),
        }
        return value

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def validate(self) -> None:
        if self.schema_name != SCHEMA_NAME or self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported manifest schema {} v{}".format(self.schema_name, self.schema_version))
        if self.state != "complete":
            raise SchemaError("manifest is not complete")
        if not self.dataset_id:
            raise SchemaError("manifest dataset_id is empty")
        if not self.shards:
            raise SchemaError("manifest has no shards")
        ids = [s.shard_id for s in self.shards]
        if len(ids) != len(set(ids)):
            raise SchemaError("manifest contains duplicate shard IDs")
        for shard in self.shards:
            if shard.root not in self.storage_roots:
                raise SchemaError("shard {} uses unknown storage root {}".format(shard.shard_id, shard.root))
            rel = Path(shard.path)
            if rel.is_absolute() or ".." in rel.parts:
                raise SchemaError("shard path must be a safe relative path: {}".format(shard.path))
        expected = {
            "n_mags": sum(s.n_mags for s in self.shards),
            "n_contigs": sum(s.n_contigs for s in self.shards),
            "n_genes": sum(s.n_genes for s in self.shards),
        }
        for key, value in expected.items():
            if int(self.totals.get(key, -1)) != value:
                raise SchemaError("manifest {}={} but shard total is {}".format(key, self.totals.get(key), value))
        split_counts = dict(self.split.get("counts", {}))
        invalid_splits = set(split_counts) - set(VALID_SPLITS)
        if invalid_splits:
            raise SchemaError("manifest has invalid split names: {}".format(sorted(invalid_splits)))
        if split_counts and sum(int(v) for v in split_counts.values()) != expected["n_mags"]:
            raise SchemaError("manifest split counts do not sum to n_mags")
        token_reps = dict(self.representations.get("tokens", {}))
        for mode, meta in token_reps.items():
            if int(dict(meta).get("size", 0)) <= 0:
                raise SchemaError("token representation {} has no positive vocabulary size".format(mode))
        protein_index = dict(self.indexes.get("protein", {}))
        if protein_index:
            if protein_index.get("type") != "per_shard_sorted_v1":
                raise SchemaError("unsupported protein index type {!r}".format(protein_index.get("type")))
            segments = list(protein_index.get("segments", []))
            if segments != ids:
                raise SchemaError("protein index segments must exactly match manifest shard order")

    def resolve_shard(self, shard: ShardRef, overrides: Optional[Mapping[str, str]] = None) -> Path:
        roots = dict(self.storage_roots)
        roots.update(dict(overrides or {}))
        if shard.root not in roots:
            raise SchemaError("no path for storage root {!r}".format(shard.root))
        return Path(roots[shard.root]).expanduser() / shard.path

    def publish(self, path: Path) -> str:
        self.validate()
        atomic_write_json(Path(path), self.to_dict())
        self.manifest_path = Path(path)
        return self.sha256

    @classmethod
    def from_dict(cls, value: Mapping[str, object], manifest_path: Optional[Path] = None) -> "DatasetManifest":
        result = cls(
            dataset_id=str(value["dataset_id"]),
            generation=int(value["generation"]),
            storage_roots={str(k): str(v) for k, v in dict(value["storage_roots"]).items()},
            shards=[ShardRef.from_dict(v) for v in list(value["shards"])],
            totals={str(k): int(v) for k, v in dict(value["totals"]).items()},
            sources=dict(value.get("sources", {})),
            representations=dict(value.get("representations", {})),
            vocabularies=dict(value.get("vocabularies", {})),
            split=dict(value.get("split", {})),
            created_utc=str(value["created_utc"]),
            builder=dict(value.get("builder", {})),
            parent_sha256=value.get("parent_sha256"),
            validation=dict(value.get("validation", {})),
            unavailable_token_modes=dict(value.get("unavailable_token_modes", {})),
            indexes=dict(value.get("indexes", {})),
            schema_name=str(value.get("schema_name", "")),
            schema_version=int(value.get("schema_version", -1)),
            state=str(value.get("state", "")),
            manifest_path=manifest_path,
        )
        result.validate()
        return result


def load_manifest(path: Path) -> DatasetManifest:
    path = Path(path)
    with path.open() as handle:
        value = json.load(handle)
    return DatasetManifest.from_dict(value, manifest_path=path)
