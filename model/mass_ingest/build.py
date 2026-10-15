"""Build source-separated shards and publish final immutable manifests."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_sources import TAX_KEYS, build_record, load_taxonomy, source_paths
from mag_shards.ingest import build_shard_payload
from mag_shards.manifest import DatasetManifest, ShardRef, load_manifest, sha256_file
from mag_shards.schema import ESMC_DIM
from mag_shards.validation import validate_shard
from mag_shards.writer import ShardWriter
from mass_ingest.common import atomic_json, load_plan, require_path, safe_name, task_id


class Bucket:
    def __init__(self, root: Path, bucket: int):
        protein = root / "proteins" / f"bucket-{bucket:04d}"
        embedding = root / "embeddings" / f"bucket-{bucket:04d}"
        vq = root / "vq" / f"bucket-{bucket:04d}"
        for stage in (protein, embedding, vq):
            if not (stage / "COMPLETE").exists():
                raise RuntimeError(f"incomplete staged bucket: {stage}")
        count = (protein / "hashes.u8").stat().st_size // 16
        self.hashes = np.memmap(protein / "hashes.u8", dtype="S16", mode="r", shape=(count,))
        self.esmc = np.memmap(embedding / "esmc.f16", dtype="<f2", mode="r", shape=(count, ESMC_DIM))
        self.kmeans = np.memmap(vq / "vq_kmeans.u32", dtype="<u4", mode="r", shape=(count,))
        self.vae = np.memmap(vq / "vq_vae.u32", dtype="<u4", mode="r", shape=(count,))


class StagedResolver:
    def __init__(self, root: Path, bucket_count: int):
        if bucket_count <= 0 or bucket_count & (bucket_count - 1):
            raise ValueError("bucket_count must be a power of two")
        self.root = root
        self.bucket_count = bucket_count
        self.cache: dict[int, Bucket] = {}

    def bucket(self, index: int) -> Bucket:
        value = self.cache.get(index)
        if value is None:
            value = Bucket(self.root, index)
            self.cache[index] = value
        return value

    def resolve(self, hashes: np.ndarray, _sequences):
        count = len(hashes)
        vectors = np.empty((count, ESMC_DIM), dtype=np.float16)
        kmeans = np.empty(count, dtype=np.uint32)
        vae = np.empty(count, dtype=np.uint32)
        raw = np.ascontiguousarray(hashes, dtype=np.uint8)
        bucket_ids = raw[:, :4].copy().view(">u4").reshape(-1) & (self.bucket_count - 1)
        for bucket_index in np.unique(bucket_ids):
            positions = np.flatnonzero(bucket_ids == bucket_index)
            keys = np.ascontiguousarray(hashes[positions]).view("S16").reshape(-1)
            bucket = self.bucket(bucket_index)
            rows = np.searchsorted(bucket.hashes, keys)
            valid = rows < len(bucket.hashes)
            if not np.all(valid) or not np.all(bucket.hashes[rows] == keys):
                raise RuntimeError(f"staged embedding lookup failed in bucket {bucket_index}")
            vectors[positions] = bucket.esmc[rows]
            kmeans[positions] = bucket.kmeans[rows]
            vae[positions] = bucket.vae[rows]
        return vectors, {"vq_kmeans": kmeans, "vq_vae": vae}, np.ones(count, dtype=bool)


def load_records(source: str, mag_ids: list[str], workers: int) -> list[dict]:
    per_mag, manifest = source_paths(source)
    taxonomy = load_taxonomy(manifest)
    args = []
    for mag_id in mag_ids:
        tax = taxonomy.get(mag_id, {key: None for key in TAX_KEYS})
        args.append((mag_id, per_mag / mag_id, source, tax, True))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(build_record, args))
    if any(record is None for record in records):
        missing = [mag_id for mag_id, record in zip(mag_ids, records) if record is None]
        raise RuntimeError(f"MAGs lacked readable proteins: {missing[:10]}")
    return records


def remove_prepared_mags(source: str, mag_ids: list[str]) -> int:
    """Remove inputs made redundant by an already validated immutable shard."""
    if os.environ.get("REMOVE_BUILT_PER_MAG") != "1":
        return 0
    per_mag, _ = source_paths(source)
    parent = per_mag.resolve()
    removed = 0
    for mag_id in mag_ids:
        path = per_mag / mag_id
        if not path.exists():
            continue
        if path.is_symlink() or path.parent.resolve() != parent:
            raise RuntimeError(f"unsafe prepared-MAG cleanup target: {path}")
        shutil.rmtree(path)
        removed += 1
    print(f"[cleanup] task={task_id()} removed_prepared_mags={removed}", flush=True)
    return removed


def build() -> None:
    root = require_path("MASS_ROOT")
    parent_path = require_path("PARENT_MANIFEST")
    shard_root = require_path("SHARD_ROOT_PATH")
    root_alias = os.environ.get("SHARD_ROOT_ALIAS", "hpc_share")
    plan = load_plan(root)
    index = task_id()
    entry = plan["build_tasks"][index]
    source = entry["source"]
    dataset_base = safe_name(os.environ.get("DATASET_BASE", plan["dataset_base"]))
    relative = f"datasets/{dataset_base}/{safe_name(source)}/task-{index:04d}"
    shard_path = shard_root / relative / entry["shard_id"]
    report_path = root / "build_reports" / f"task-{index:04d}.json"
    if (shard_path / "COMPLETE").exists() and report_path.exists():
        validate_shard(shard_path, verify_hashes=True)
        remove_prepared_mags(source, entry["mag_ids"])
        print(f"build task {index} already complete")
        return
    parent = load_manifest(parent_path)
    records = load_records(source, entry["mag_ids"], int(os.environ.get("SOURCE_WORKERS", "8")))
    resolver = StagedResolver(root, plan["bucket_count"])
    payload = build_shard_payload(parent, records, resolver)
    writer = ShardWriter(shard_root, relative, builder={
        "operation": "bulk_incremental_ingest",
        "source": source,
        "task": index,
        "parent_manifest_sha256": parent.sha256,
    })
    output = writer.write(entry["shard_id"], payload)
    validation = validate_shard(output, verify_hashes=True)
    meta = json.loads((output / "shard.json").read_text())
    result = {
        "task": index,
        "source": source,
        "root": root_alias,
        "path": f"{relative}/{entry['shard_id']}",
        "shard_id": entry["shard_id"],
        "metadata_sha256": validation["metadata_sha256"],
        "n_mags": validation["n_mags"],
        "n_contigs": validation["n_contigs"],
        "n_genes": validation["n_genes"],
        "bytes_by_column": {key: int(value["bytes"]) for key, value in meta["columns"].items()},
    }
    atomic_json(report_path, result)
    remove_prepared_mags(source, entry["mag_ids"])
    print(json.dumps(result))


def candidate_refs(root: Path, plan: dict) -> list[tuple[str, ShardRef]]:
    result = []
    for entry in plan["build_tasks"]:
        path = root / "build_reports" / f"task-{entry['task']:04d}.json"
        if not path.exists():
            raise RuntimeError(f"missing build report: {path}")
        value = json.loads(path.read_text())
        result.append((value["source"], ShardRef(
            shard_id=value["shard_id"], root=value["root"], path=value["path"],
            metadata_sha256=value["metadata_sha256"], n_mags=value["n_mags"],
            n_contigs=value["n_contigs"], n_genes=value["n_genes"],
            bytes_by_column=value["bytes_by_column"],
        )))
    return result


def publish_one(parent: DatasetManifest, candidates: list[tuple[str, ShardRef]], sources_to_add: set[str],
                destination: Path, dataset_id: str, parent_path: Path,
                mass_root: Path, shard_root: Path, root_alias: str) -> DatasetManifest:
    selected = [(source, ref) for source, ref in candidates if source in sources_to_add]
    refs = [ref for _, ref in selected]
    roots = dict(parent.storage_roots); roots[root_alias] = str(shard_root)
    sources = dict(parent.sources)
    split_counts = dict(parent.split.get("counts", {}))
    for source, ref in selected:
        metadata = dict(sources.get(source, {}))
        metadata["n_mags"] = int(metadata.get("n_mags", 0)) + ref.n_mags
        _, manifest = source_paths(source)
        metadata["manifest"] = str(manifest)
        metadata["manifest_sha256"] = sha256_file(manifest)
        sources[source] = metadata
        shard_path = Path(roots[ref.root]) / ref.path
        with (shard_path / "mags.jsonl").open() as handle:
            for line in handle:
                split = json.loads(line)["split"]
                split_counts[split] = split_counts.get(split, 0) + 1
    all_refs = list(parent.shards) + refs
    child = DatasetManifest(
        dataset_id=dataset_id,
        generation=parent.generation + 1,
        parent_sha256=parent.sha256,
        storage_roots=roots,
        shards=all_refs,
        totals={
            "n_mags": parent.totals["n_mags"] + sum(ref.n_mags for ref in refs),
            "n_contigs": parent.totals["n_contigs"] + sum(ref.n_contigs for ref in refs),
            "n_genes": parent.totals["n_genes"] + sum(ref.n_genes for ref in refs),
        },
        sources=sources,
        representations=parent.representations,
        vocabularies=parent.vocabularies,
        split={**parent.split, "counts": split_counts},
        created_utc=datetime.now(timezone.utc).isoformat(),
        builder={"operation": "bulk_incremental_ingest", "parent_manifest": str(parent_path),
                 "parent_sha256": parent.sha256, "mass_root": str(mass_root)},
        validation={"new_shards_validated_with_hashes": len(refs)},
        unavailable_token_modes=parent.unavailable_token_modes,
        indexes={"protein": {"type": "per_shard_sorted_v1", "hash": "md5-aa-raw-16bytes",
                             "segments": [ref.shard_id for ref in all_refs]}},
    )
    child.publish(destination)
    return child


def publish() -> None:
    root = require_path("MASS_ROOT")
    parent_path = require_path("PARENT_MANIFEST")
    shard_root = require_path("SHARD_ROOT_PATH")
    root_alias = os.environ.get("SHARD_ROOT_ALIAS", "hpc_share")
    plan = load_plan(root)
    parent = load_manifest(parent_path)
    candidates = candidate_refs(root, plan)
    output_dir = Path(os.environ.get("FINAL_MANIFEST_DIR", str(parent_path.parent)))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = os.environ.get("DATASET_ID", plan["dataset_id"])
    combined_path = Path(os.environ.get(
        "OUTPUT_MANIFEST", str(output_dir / (safe_name(dataset_id) + ".manifest.json"))))
    publish_source_manifests = os.environ.get("PUBLISH_SOURCE_MANIFESTS", "0") == "1"
    reports = {}
    if publish_source_manifests:
        for source in plan["sources"]:
            source_path = output_dir / "{}-{}.manifest.json".format(
                safe_name(dataset_id), safe_name(source))
            child = publish_one(parent, candidates, {source}, source_path,
                                "{}-{}".format(dataset_id, safe_name(source)),
                                parent_path, root, shard_root, root_alias)
            reports[source] = {"manifest": str(source_path), "sha256": child.sha256,
                               "totals": child.totals}
    combined = publish_one(
        parent, candidates, set(plan["sources"]), combined_path, dataset_id,
        parent_path, root, shard_root, root_alias)
    reports["combined"] = {"manifest": str(combined_path), "sha256": combined.sha256,
                           "totals": combined.totals}
    atomic_json(root / "publication_report.json", reports)
    print(json.dumps(reports, indent=2))


def main(stage: str | None = None) -> None:
    stage = stage or os.environ.get("MODE", "")
    actions = {"build": build, "publish": publish}
    if stage not in actions:
        raise RuntimeError("stage must be build or publish")
    actions[stage]()


if __name__ == "__main__":
    main()
