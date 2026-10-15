"""Plan work, collect protein sequences, and deduplicate hash buckets.

Configuration is supplied through environment variables so the same commands
work on a login node and in Slurm arrays.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path

from data_sources import source_paths
from mass_ingest.common import (
    atomic_json,
    bucket_for_hash,
    fresh_staging,
    load_plan,
    publish_directory,
    require_path,
    task_id,
)
from mag_shards.manifest import load_manifest


def read_source(source: str) -> list[dict]:
    _, manifest = source_paths(source)
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["source"] = source
        row["n_genes"] = int(row.get("n_genes") or 0)
    return rows


def balanced_groups(rows: list[dict], count: int) -> list[list[dict]]:
    if count <= 0:
        raise ValueError("collect task count must be positive")
    groups = [[] for _ in range(count)]
    queue = [(0, index) for index in range(count)]
    for row in sorted(rows, key=lambda item: item["n_genes"], reverse=True):
        total, target = heapq.heappop(queue)
        groups[target].append({"source": row["source"], "mag_id": row["genome"]})
        heapq.heappush(queue, (total + row["n_genes"], target))
    return groups


def build_groups(rows: list[dict], target_genes: int) -> list[dict]:
    result, current, genes = [], [], 0
    for row in rows:
        size = row["n_genes"]
        if current and genes + size > target_genes:
            result.append({"source": current[0]["source"], "mag_ids": [x["genome"] for x in current],
                           "expected_genes": genes})
            current, genes = [], 0
        current.append(row)
        genes += size
    if current:
        result.append({"source": current[0]["source"], "mag_ids": [x["genome"] for x in current],
                       "expected_genes": genes})
    return result


def plan() -> None:
    root = require_path("MASS_ROOT")
    sources = [value.strip() for value in os.environ.get("SOURCES", "").split(",") if value.strip()]
    if not sources:
        raise RuntimeError("set SOURCES to one or more registered source names")
    parent_path = require_path("PARENT_MANIFEST")
    parent = load_manifest(parent_path)
    bucket_count = int(os.environ.get("BUCKET_COUNT", "512"))
    collect_count = int(os.environ.get("COLLECT_TASKS", "128"))
    target_genes = int(os.environ.get("BUILD_TARGET_GENES", "1000000"))
    bucket_for_hash(b"\0" * 16, bucket_count)
    if target_genes <= 0:
        raise ValueError("BUILD_TARGET_GENES must be positive")
    all_rows, by_source = [], {}
    for source in sources:
        rows = read_source(source)
        by_source[source] = rows
        all_rows.extend(rows)
    collect_count = min(collect_count, max(1, len(all_rows)))
    collect = balanced_groups(all_rows, collect_count)
    build = []
    for source in sources:
        build.extend(build_groups(by_source[source], target_genes))
    first_shard = int(os.environ.get("FIRST_SHARD", str(len(parent.shards))))
    for index, entry in enumerate(build):
        entry["task"] = index
        entry["shard_id"] = f"shard-{first_shard + index:06d}"
    value = {
        "version": 2,
        "dataset_id": os.environ.get("DATASET_ID", "{}-append".format(parent.dataset_id)),
        "dataset_base": os.environ.get("DATASET_BASE", "{}-append".format(parent.dataset_id)),
        "parent_manifest": str(parent_path),
        "parent_sha256": parent.sha256,
        "sources": sources,
        "bucket_count": bucket_count,
        "collect_tasks": collect,
        "build_tasks": build,
        "counts": {
            "mags": len(all_rows),
            "genes": sum(row["n_genes"] for row in all_rows),
            "collect_tasks": len(collect),
            "buckets": bucket_count,
            "build_tasks": len(build),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "plan.json", value)
    print(json.dumps(value["counts"], indent=2))


class BucketBuffers:
    def __init__(self, root: Path, bucket_count: int, flush_bytes: int = 64 * 1024):
        self.root = root
        self.bucket_count = bucket_count
        self.flush_bytes = flush_bytes
        self.buffers: dict[int, bytearray] = {}

    def add(self, bucket: int, line: bytes) -> None:
        buffer = self.buffers.setdefault(bucket, bytearray())
        buffer.extend(line)
        if len(buffer) >= self.flush_bytes:
            self.flush(bucket)

    def flush(self, bucket: int) -> None:
        payload = self.buffers.pop(bucket, None)
        if not payload:
            return
        path = self.root / f"bucket-{bucket:04d}.tsv.gz"
        with gzip.open(path, "ab", compresslevel=1) as handle:
            handle.write(payload)

    def close(self) -> None:
        for bucket in list(self.buffers):
            self.flush(bucket)


def iter_faa(path: Path):
    sequence = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if sequence:
                    yield "".join(sequence)
                sequence = []
            else:
                sequence.append(line.strip())
    if sequence:
        yield "".join(sequence)


def collect() -> None:
    root = require_path("MASS_ROOT")
    plan_value = load_plan(root)
    index = task_id()
    destination = root / "collect" / f"task-{index:04d}"
    if (destination / "COMPLETE").exists():
        print(f"collect task {index} already complete")
        return
    staging = fresh_staging(destination)
    buffers = BucketBuffers(staging, plan_value["bucket_count"])
    genes = residues = 0
    for item in plan_value["collect_tasks"][index]:
        per_mag, _ = source_paths(item["source"])
        faa = per_mag / item["mag_id"] / f"{item['mag_id']}_genes.faa"
        if not faa.exists():
            raise FileNotFoundError(faa)
        for sequence in iter_faa(faa):
            digest = hashlib.md5(sequence.encode()).digest()
            bucket = bucket_for_hash(digest, plan_value["bucket_count"])
            buffers.add(bucket, digest.hex().encode() + b"\t" + sequence.encode() + b"\n")
            genes += 1
            residues += len(sequence)
    buffers.close()
    atomic_json(staging / "report.json", {"task": index, "genes": genes, "residues": residues})
    (staging / "COMPLETE").write_text("complete\n")
    publish_directory(staging, destination)
    print(json.dumps({"task": index, "genes": genes, "residues": residues}))


def dedup() -> None:
    root = require_path("MASS_ROOT")
    plan_value = load_plan(root)
    bucket = task_id()
    destination = root / "proteins" / f"bucket-{bucket:04d}"
    if (destination / "COMPLETE").exists():
        print(f"dedup bucket {bucket} already complete")
        return
    values: dict[bytes, bytes] = {}
    occurrences = 0
    for task in range(len(plan_value["collect_tasks"])):
        path = root / "collect" / f"task-{task:04d}" / f"bucket-{bucket:04d}.tsv.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rb") as handle:
            for line in handle:
                raw_hash, sequence = line.rstrip(b"\n").split(b"\t", 1)
                digest = bytes.fromhex(raw_hash.decode())
                previous = values.setdefault(digest, sequence)
                if previous != sequence:
                    raise RuntimeError(f"MD5 collision in bucket {bucket}: {digest.hex()}")
                occurrences += 1
    staging = fresh_staging(destination)
    ordered = sorted(values.items())
    with (staging / "hashes.u8").open("wb") as hashes, gzip.open(
        staging / "sequences.txt.gz", "wb", compresslevel=1
    ) as sequences:
        for digest, sequence in ordered:
            hashes.write(digest)
            sequences.write(sequence + b"\n")
    report = {
        "bucket": bucket,
        "occurrences": occurrences,
        "unique_proteins": len(ordered),
        "residues": sum(len(sequence) for _, sequence in ordered),
    }
    atomic_json(staging / "report.json", report)
    (staging / "COMPLETE").write_text("complete\n")
    publish_directory(staging, destination)
    print(json.dumps(report))


def main(stage: str | None = None) -> None:
    stage = stage or os.environ.get("MODE", "")
    actions = {"plan": plan, "collect": collect, "dedup": dedup}
    if stage not in actions:
        raise RuntimeError("stage must be plan, collect, or dedup")
    actions[stage]()


if __name__ == "__main__":
    main()
