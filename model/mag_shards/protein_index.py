"""Mmap lookup across immutable per-shard sorted protein-hash indexes."""

from __future__ import annotations

import json
from typing import Mapping, Optional

import numpy as np

from .manifest import DatasetManifest
from .schema import SchemaError


class _Segment:
    def __init__(self, path):
        with (path / "shard.json").open() as handle:
            meta = json.load(handle)
        columns = meta["columns"]
        if "protein_index_hash.u8" not in columns or "protein_index_row.u32" not in columns:
            raise SchemaError("shard has no protein index: {}".format(path))
        count = int(columns["protein_index_row.u32"]["shape"][0])
        self.hashes = np.memmap(path / "protein_index_hash.u8", dtype="S16", mode="r", shape=(count,))
        self.rows = np.memmap(path / "protein_index_row.u32", dtype="<u4", mode="r", shape=(count,))


class ProteinIndex:
    def __init__(self, manifest: DatasetManifest, root_overrides: Optional[Mapping[str, str]] = None):
        index_meta = manifest.indexes.get("protein", {})
        if index_meta.get("type") != "per_shard_sorted_v1":
            raise SchemaError("manifest has no supported protein index")
        self.manifest = manifest
        self.paths = [manifest.resolve_shard(ref, root_overrides) for ref in manifest.shards]
        self._segments = {}

    def _segment(self, shard_index: int):
        segment = self._segments.get(shard_index)
        if segment is None:
            segment = _Segment(self.paths[shard_index]); self._segments[shard_index] = segment
        return segment

    def lookup(self, hashes: np.ndarray):
        values = np.asarray(hashes, dtype=np.uint8)
        if values.ndim != 2 or values.shape[1] != 16:
            raise ValueError("hashes must be uint8[n,16]")
        keys = np.ascontiguousarray(values).view("S16").reshape(-1)
        shard_result = np.full(len(keys), -1, dtype=np.int32)
        row_result = np.zeros(len(keys), dtype=np.uint32)
        unresolved = np.arange(len(keys), dtype=np.int64)
        # Newest shards first makes append releases authoritative while old
        # manifests remain immutable and independently readable.
        for shard_index in range(len(self.paths) - 1, -1, -1):
            if not len(unresolved):
                break
            segment = self._segment(shard_index)
            wanted = keys[unresolved]
            positions = np.searchsorted(segment.hashes, wanted)
            in_range = positions < len(segment.hashes)
            matched = np.zeros(len(wanted), dtype=bool)
            matched[in_range] = segment.hashes[positions[in_range]] == wanted[in_range]
            if matched.any():
                destination = unresolved[matched]
                shard_result[destination] = shard_index
                row_result[destination] = segment.rows[positions[matched]]
                unresolved = unresolved[~matched]
        return shard_result, row_result
