"""Read-only validation for shards and complete dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from .manifest import DatasetManifest, load_manifest, sha256_file
from .schema import SCHEMA_NAME, SCHEMA_VERSION, SchemaError, dtype_for


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open() as handle:
        return json.load(handle)


def validate_shard(path: Path, verify_hashes: bool = False) -> Dict[str, object]:
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise SchemaError("shard has no COMPLETE marker: {}".format(path))
    meta_path = path / "shard.json"
    marker = (path / "COMPLETE").read_text().strip()
    if marker != sha256_file(meta_path):
        raise SchemaError("COMPLETE marker does not match shard metadata: {}".format(path))
    meta = _load_json(meta_path)
    if meta.get("schema_name") != SCHEMA_NAME or int(meta.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SchemaError("unsupported shard schema at {}".format(path))
    if meta.get("state") != "complete":
        raise SchemaError("shard metadata is not complete: {}".format(path))
    n_mags = int(meta["n_mags"]); n_contigs = int(meta["n_contigs"]); n_genes = int(meta["n_genes"])
    columns = dict(meta["columns"])
    required = {"mags.jsonl", "mag_gene_offsets.u64", "mag_contig_offsets.u64", "contig_gene_offsets.u64"}
    missing = required - set(columns)
    if missing:
        raise SchemaError("shard is missing structural columns: {}".format(sorted(missing)))
    for name, spec_value in columns.items():
        spec = dict(spec_value)
        file_path = path / name
        if not file_path.is_file():
            raise SchemaError("declared column is missing: {}".format(file_path))
        actual_bytes = file_path.stat().st_size
        if actual_bytes != int(spec["bytes"]):
            raise SchemaError("{} byte length {} != {}".format(name, actual_bytes, spec["bytes"]))
        if spec["dtype"] not in ("jsonl",):
            shape = [int(x) for x in spec["shape"]]
            logical_shape = shape
            if name == "esmc_valid.bit":
                expected = (shape[0] + 7) // 8
            else:
                expected = int(np.prod(shape, dtype=np.int64)) * dtype_for(str(spec["dtype"])).itemsize
            if expected != actual_bytes:
                raise SchemaError("{} shape/dtype imply {} bytes, found {}".format(name, expected, actual_bytes))
        if verify_hashes and sha256_file(file_path) != spec["sha256"]:
            raise SchemaError("checksum mismatch: {}".format(file_path))
    with (path / "mags.jsonl").open() as handle:
        mags = [json.loads(line) for line in handle if line.strip()]
    if len(mags) != n_mags:
        raise SchemaError("mags.jsonl has {} rows, expected {}".format(len(mags), n_mags))
    mg = np.memmap(path / "mag_gene_offsets.u64", dtype="<u8", mode="r")
    mc = np.memmap(path / "mag_contig_offsets.u64", dtype="<u8", mode="r")
    cg = np.memmap(path / "contig_gene_offsets.u64", dtype="<u8", mode="r")
    _validate_offsets("mag_gene_offsets", mg, n_mags + 1, n_genes)
    _validate_offsets("mag_contig_offsets", mc, n_mags + 1, n_contigs)
    _validate_offsets("contig_gene_offsets", cg, n_contigs + 1, n_genes)
    if "ko_indptr.u64" in columns:
        ki = np.memmap(path / "ko_indptr.u64", dtype="<u8", mode="r")
        _validate_offsets("ko_indptr", ki, n_genes + 1, columns["ko_ids.u32"]["shape"][0])
    if "ec_indptr.u64" in columns:
        ei = np.memmap(path / "ec_indptr.u64", dtype="<u8", mode="r")
        _validate_offsets("ec_indptr", ei, n_genes + 1, columns["ec_ids.u32"]["shape"][0])
    return {"shard_id": meta["shard_id"], "n_mags": n_mags, "n_contigs": n_contigs,
            "n_genes": n_genes, "metadata_sha256": sha256_file(meta_path), "columns": len(columns)}


def _validate_offsets(name: str, values: np.ndarray, expected_len: int, expected_end: int) -> None:
    if len(values) != expected_len or (len(values) and int(values[0]) != 0):
        raise SchemaError("{} has invalid length/start".format(name))
    if np.any(values[1:] < values[:-1]):
        raise SchemaError("{} is not monotonic".format(name))
    if int(values[-1]) != int(expected_end):
        raise SchemaError("{} ends at {}, expected {}".format(name, int(values[-1]), expected_end))


def validate_dataset(manifest_path: Path, root_overrides: Optional[Mapping[str, str]] = None,
                     verify_hashes: bool = False) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    totals = {"n_mags": 0, "n_contigs": 0, "n_genes": 0}
    reports = []
    for ref in manifest.shards:
        shard_path = manifest.resolve_shard(ref, root_overrides)
        report = validate_shard(shard_path, verify_hashes=verify_hashes)
        if report["metadata_sha256"] != ref.metadata_sha256:
            raise SchemaError("shard metadata digest mismatch: {}".format(ref.shard_id))
        for key in totals:
            totals[key] += int(report[key])
        reports.append(report)
        with (shard_path / "shard.json").open() as handle:
            shard_meta = json.load(handle)
        for mode, token_meta in manifest.representations.get("tokens", {}).items():
            column = shard_meta["columns"].get("token.{}.u32".format(mode))
            if column and column.get("max_id") is not None and int(column["max_id"]) >= int(token_meta["size"]):
                raise SchemaError("{} token ID {} exceeds vocab size {} in {}".format(
                    mode, column["max_id"], token_meta["size"], ref.shard_id))
    for key, value in totals.items():
        if value != int(manifest.totals[key]):
            raise SchemaError("dataset {} mismatch: {} != {}".format(key, value, manifest.totals[key]))
    return {"manifest_sha256": manifest.sha256, "dataset_id": manifest.dataset_id,
            "n_shards": len(reports), "totals": totals, "verify_hashes": bool(verify_hashes)}
