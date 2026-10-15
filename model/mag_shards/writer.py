"""Atomic shard writer.

The writer accepts already-aligned arrays.  Conversion and incremental
ingestion are responsible for producing those arrays; this module owns format
validation, checksummed writes, and publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

from .manifest import canonical_json_bytes, sha256_bytes
from .schema import ESMC_DIM, MISSING_U32, SCHEMA_NAME, SCHEMA_VERSION, SchemaError, token_column


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ShardPayload:
    mags: List[Mapping[str, object]]
    mag_gene_offsets: np.ndarray
    mag_contig_offsets: np.ndarray
    contig_gene_offsets: np.ndarray
    esmc: Optional[np.ndarray] = None
    esmc_valid: Optional[np.ndarray] = None
    tokens: Mapping[str, np.ndarray] = field(default_factory=dict)
    protein_hash: Optional[np.ndarray] = None
    ko_indptr: Optional[np.ndarray] = None
    ko_ids: Optional[np.ndarray] = None
    ec_indptr: Optional[np.ndarray] = None
    ec_ids: Optional[np.ndarray] = None

    @property
    def n_mags(self) -> int:
        return len(self.mags)

    @property
    def n_contigs(self) -> int:
        return int(len(self.contig_gene_offsets) - 1)

    @property
    def n_genes(self) -> int:
        return int(self.mag_gene_offsets[-1]) if len(self.mag_gene_offsets) else 0

    def validate(self) -> None:
        _check_offsets("mag_gene_offsets", self.mag_gene_offsets, self.n_mags + 1)
        _check_offsets("mag_contig_offsets", self.mag_contig_offsets, self.n_mags + 1)
        _check_offsets("contig_gene_offsets", self.contig_gene_offsets, self.n_contigs + 1)
        if int(self.mag_contig_offsets[-1]) != self.n_contigs:
            raise SchemaError("mag_contig_offsets does not end at n_contigs")
        if int(self.contig_gene_offsets[-1]) != self.n_genes:
            raise SchemaError("contig_gene_offsets does not end at n_genes")
        for i, mag in enumerate(self.mags):
            expected_genes = int(self.mag_gene_offsets[i + 1] - self.mag_gene_offsets[i])
            expected_contigs = int(self.mag_contig_offsets[i + 1] - self.mag_contig_offsets[i])
            if int(mag.get("n_genes", expected_genes)) != expected_genes:
                raise SchemaError("MAG {} has inconsistent gene count".format(mag.get("mag_id", i)))
            if int(mag.get("n_contigs", expected_contigs)) != expected_contigs:
                raise SchemaError("MAG {} has inconsistent contig count".format(mag.get("mag_id", i)))
        if self.esmc is not None:
            if self.esmc.dtype != np.dtype("float16") or self.esmc.shape != (self.n_genes, ESMC_DIM):
                raise SchemaError("esmc must have dtype float16 and shape ({}, {})".format(self.n_genes, ESMC_DIM))
        if self.esmc_valid is not None:
            if np.asarray(self.esmc_valid).shape != (self.n_genes,):
                raise SchemaError("esmc_valid must have one value per gene")
        for mode, values in self.tokens.items():
            if np.asarray(values).dtype != np.dtype("uint32") or np.asarray(values).shape != (self.n_genes,):
                raise SchemaError("token {} must be uint32[n_genes]".format(mode))
            token_column(mode)
        if self.protein_hash is not None:
            if np.asarray(self.protein_hash).dtype != np.dtype("uint8") or np.asarray(self.protein_hash).shape != (self.n_genes, 16):
                raise SchemaError("protein_hash must be uint8[n_genes,16]")
        _check_csr("ko", self.ko_indptr, self.ko_ids, self.n_genes)
        _check_csr("ec", self.ec_indptr, self.ec_ids, self.n_genes)


def _check_offsets(name: str, values: np.ndarray, expected_len: int) -> None:
    arr = np.asarray(values)
    if arr.dtype != np.dtype("uint64"):
        raise SchemaError("{} must be uint64".format(name))
    if arr.ndim != 1 or len(arr) != expected_len:
        raise SchemaError("{} length must be {}".format(name, expected_len))
    if not len(arr) or int(arr[0]) != 0:
        raise SchemaError("{} must start at zero".format(name))
    if np.any(arr[1:] < arr[:-1]):
        raise SchemaError("{} must be monotonic".format(name))


def _check_csr(name: str, indptr: Optional[np.ndarray], ids: Optional[np.ndarray], n_genes: int) -> None:
    if indptr is None and ids is None:
        return
    if indptr is None or ids is None:
        raise SchemaError("{} CSR requires both indptr and ids".format(name))
    _check_offsets("{}_indptr".format(name), indptr, n_genes + 1)
    values = np.asarray(ids)
    if values.dtype != np.dtype("uint32") or values.ndim != 1:
        raise SchemaError("{}_ids must be a uint32 vector".format(name))
    if int(indptr[-1]) != len(values):
        raise SchemaError("{}_indptr does not end at len(ids)".format(name))


def _write_array(path: Path, values: np.ndarray, dtype_name: str, shape: List[int], chunk_bytes: int) -> Dict[str, object]:
    arr = np.asarray(values)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    raw = arr.view(np.uint8).reshape(-1)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for start in range(0, len(raw), chunk_bytes):
            block = raw[start:start + chunk_bytes].tobytes()
            handle.write(block)
            digest.update(block)
        handle.flush()
        os.fsync(handle.fileno())
    return {"dtype": dtype_name, "shape": shape, "bytes": int(path.stat().st_size),
            "sha256": digest.hexdigest()}


class ShardWriter:
    def __init__(self, root: Path, relative_dataset_dir: str, builder: Optional[Mapping[str, object]] = None,
                 chunk_bytes: int = 16 * 1024 * 1024):
        self.root = Path(root)
        self.relative_dataset_dir = Path(relative_dataset_dir)
        if self.relative_dataset_dir.is_absolute() or ".." in self.relative_dataset_dir.parts:
            raise SchemaError("relative_dataset_dir must be a safe relative path")
        self.builder = dict(builder or {})
        self.chunk_bytes = int(chunk_bytes)

    def write(self, shard_id: str, payload: ShardPayload, overwrite_staging: bool = True) -> Path:
        payload.validate()
        dataset_dir = self.root / self.relative_dataset_dir
        final = dataset_dir / shard_id
        if final.exists():
            raise FileExistsError("published shard already exists: {}".format(final))
        staging_root = dataset_dir / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / "{}.{}".format(shard_id, uuid.uuid4().hex)
        staging.mkdir()
        try:
            columns = {}
            columns["mag_gene_offsets.u64"] = _write_array(
                staging / "mag_gene_offsets.u64", payload.mag_gene_offsets, "u64",
                [len(payload.mag_gene_offsets)], self.chunk_bytes)
            columns["mag_contig_offsets.u64"] = _write_array(
                staging / "mag_contig_offsets.u64", payload.mag_contig_offsets, "u64",
                [len(payload.mag_contig_offsets)], self.chunk_bytes)
            columns["contig_gene_offsets.u64"] = _write_array(
                staging / "contig_gene_offsets.u64", payload.contig_gene_offsets, "u64",
                [len(payload.contig_gene_offsets)], self.chunk_bytes)

            mags_bytes = b"".join(canonical_json_bytes(dict(mag)) for mag in payload.mags)
            mags_path = staging / "mags.jsonl"
            with mags_path.open("wb") as handle:
                handle.write(mags_bytes); handle.flush(); os.fsync(handle.fileno())
            columns["mags.jsonl"] = {"dtype": "jsonl", "shape": [payload.n_mags],
                                      "bytes": len(mags_bytes), "sha256": sha256_bytes(mags_bytes)}

            all_esmc_valid = False
            if payload.esmc is not None:
                columns["esmc.f16"] = _write_array(staging / "esmc.f16", payload.esmc, "f16",
                                                     [payload.n_genes, ESMC_DIM], self.chunk_bytes)
                valid = np.ones(payload.n_genes, dtype=bool) if payload.esmc_valid is None else np.asarray(payload.esmc_valid, dtype=bool)
                all_esmc_valid = bool(valid.all())
                if not bool(valid.all()):
                    packed = np.packbits(valid, bitorder="little")
                    columns["esmc_valid.bit"] = _write_array(staging / "esmc_valid.bit", packed, "u8",
                                                               [payload.n_genes], self.chunk_bytes)
            for mode, values in sorted(payload.tokens.items()):
                name = token_column(mode)
                columns[name] = _write_array(staging / name, values, "u32", [payload.n_genes], self.chunk_bytes)
                valid_tokens = np.asarray(values)[np.asarray(values) != MISSING_U32]
                columns[name].update(
                    missing_count=int(payload.n_genes - len(valid_tokens)),
                    min_id=int(valid_tokens.min()) if len(valid_tokens) else None,
                    max_id=int(valid_tokens.max()) if len(valid_tokens) else None,
                )
            if payload.protein_hash is not None:
                columns["protein_hash.u8"] = _write_array(staging / "protein_hash.u8", payload.protein_hash,
                                                            "u8", [payload.n_genes, 16], self.chunk_bytes)
                keys = np.ascontiguousarray(payload.protein_hash).view("S16").reshape(-1)
                order = np.argsort(keys, kind="stable")
                sorted_keys = keys[order]
                keep = np.ones(len(sorted_keys), dtype=bool)
                if len(sorted_keys) > 1:
                    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
                unique_keys = np.ascontiguousarray(sorted_keys[keep]).view(np.uint8).reshape(-1, 16)
                unique_rows = order[keep].astype(np.uint32, copy=False)
                columns["protein_index_hash.u8"] = _write_array(
                    staging / "protein_index_hash.u8", unique_keys, "u8", [len(unique_keys), 16], self.chunk_bytes)
                columns["protein_index_row.u32"] = _write_array(
                    staging / "protein_index_row.u32", unique_rows, "u32", [len(unique_rows)], self.chunk_bytes)
            for prefix, indptr, ids in (("ko", payload.ko_indptr, payload.ko_ids),
                                        ("ec", payload.ec_indptr, payload.ec_ids)):
                if indptr is not None:
                    columns["{}_indptr.u64".format(prefix)] = _write_array(
                        staging / "{}_indptr.u64".format(prefix), indptr, "u64", [payload.n_genes + 1], self.chunk_bytes)
                    columns["{}_ids.u32".format(prefix)] = _write_array(
                        staging / "{}_ids.u32".format(prefix), ids, "u32", [len(ids)], self.chunk_bytes)

            meta = {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "state": "complete",
                "shard_id": shard_id,
                "created_utc": utc_now(),
                "builder": self.builder,
                "n_mags": payload.n_mags,
                "n_contigs": payload.n_contigs,
                "n_genes": payload.n_genes,
                "all_esmc_valid": all_esmc_valid,
                "columns": columns,
                "sources": sorted({str(m.get("source_name", "")) for m in payload.mags}),
                "splits": _count_values(payload.mags, "split"),
            }
            meta_bytes = canonical_json_bytes(meta)
            with (staging / "shard.json").open("wb") as handle:
                handle.write(meta_bytes); handle.flush(); os.fsync(handle.fileno())
            with (staging / "COMPLETE").open("wb") as handle:
                handle.write((sha256_bytes(meta_bytes) + "\n").encode("ascii")); handle.flush(); os.fsync(handle.fileno())
            dataset_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(staging), str(final))
            return final
        except BaseException:
            if overwrite_staging:
                shutil.rmtree(str(staging), ignore_errors=True)
            raise


def _count_values(rows: List[Mapping[str, object]], key: str) -> Dict[str, int]:
    out = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out
