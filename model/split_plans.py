"""Versioned, checksum-pinned MAG assignments used above immutable shards."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from mag_shards.manifest import atomic_write_json, load_manifest, sha256_file


SCHEMA = "mag-split-plan"
VERSION = 1


@dataclass(frozen=True)
class Assignment:
    label: str
    source_name: str


@dataclass(frozen=True)
class SplitPlan:
    path: Path
    manifest_sha256: str
    table_sha256: str
    assignments: Mapping[str, Assignment]
    strategy: Mapping[str, object]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)


def load_split_plan(path: Path, manifest_sha256: str) -> SplitPlan:
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema_name") != SCHEMA or int(value.get("schema_version", -1)) != VERSION:
        raise ValueError("unsupported split-plan schema")
    if value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("split plan was compiled for a different dataset manifest")
    table = path.parent / value["assignment_table"]
    if sha256_file(table) != value.get("assignment_sha256"):
        raise ValueError("split-plan assignment checksum mismatch")
    assignments: Dict[str, Assignment] = {}
    with gzip.open(table, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            mag_id = row["mag_id"]
            if mag_id in assignments:
                raise ValueError("duplicate MAG ID in split plan: {}".format(mag_id))
            assignments[mag_id] = Assignment(row["label"], row["source_name"])
    if len(assignments) != int(value["counts"]["total"]):
        raise ValueError("split-plan row count mismatch")
    return SplitPlan(
        path=path,
        manifest_sha256=manifest_sha256,
        table_sha256=value["assignment_sha256"],
        assignments=assignments,
        strategy=dict(value.get("strategy", {})),
    )


def _manifest_rows(manifest, root_overrides=None):
    rows = []
    for ref in manifest.shards:
        shard = manifest.resolve_shard(ref, root_overrides)
        with (shard / "mags.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                rows.append((str(row["mag_id"]), str(row["source_name"]), str(row["split"])))
    return rows


def compile_assignments(manifest_path: Path, output_path: Path,
                        assignments: Mapping[str, str], strategy: Mapping[str, object],
                        root_overrides=None) -> Path:
    manifest = load_manifest(manifest_path)
    output_path = Path(output_path)
    table_path = output_path.with_suffix(".tsv.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest_rows = _manifest_rows(manifest, root_overrides)
    manifest_ids = {mag_id for mag_id, _, _ in manifest_rows}
    if set(assignments) != manifest_ids:
        missing = manifest_ids - set(assignments)
        extra = set(assignments) - manifest_ids
        raise ValueError("assignment registry mismatch: missing={} extra={}".format(len(missing), len(extra)))
    for mag_id, source, _ in manifest_rows:
        rows.append((mag_id, source, str(assignments[mag_id])))
    rows.sort()
    temporary_table = table_path.with_name(".{}.{}.tmp".format(table_path.name, os.getpid()))
    with temporary_table.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as compressed:
            compressed.write(b"mag_id\tsource_name\tlabel\n")
            for mag_id, source, label in rows:
                compressed.write("{}\t{}\t{}\n".format(mag_id, source, label).encode())
    os.replace(temporary_table, table_path)
    counts = Counter(label for _, _, label in rows)
    metadata = {
        "schema_name": SCHEMA,
        "schema_version": VERSION,
        "manifest_sha256": manifest.sha256,
        "assignment_table": table_path.name,
        "assignment_sha256": sha256_file(table_path),
        "counts": {"total": len(rows), "labels": dict(sorted(counts.items()))},
        "strategy": dict(strategy),
    }
    atomic_write_json(output_path, metadata)
    return output_path


def compile_source_holdout(manifest_path: Path, output_path: Path,
                           source_labels: Mapping[str, str], root_overrides=None) -> Path:
    manifest = load_manifest(manifest_path)
    assignments = {}
    for mag_id, source, _ in _manifest_rows(manifest, root_overrides):
        if source not in source_labels:
            raise ValueError("source has no split assignment: {}".format(source))
        assignments[mag_id] = source_labels[source]
    return compile_assignments(
        manifest_path, output_path, assignments,
        {"type": "source_holdout", "source_labels": dict(source_labels)},
        root_overrides,
    )


def compile_stored_labels(manifest_path: Path, output_path: Path,
                          root_overrides=None) -> Path:
    manifest = load_manifest(manifest_path)
    assignments = {
        mag_id: label
        for mag_id, _, label in _manifest_rows(manifest, root_overrides)
    }
    return compile_assignments(
        manifest_path, output_path, assignments,
        {"type": "manifest_labels", "manifest_split": dict(manifest.split)},
        root_overrides,
    )


def _seeded_order(seed: int, mag_id: str):
    digest = hashlib.sha256("{}\0{}".format(seed, mag_id).encode()).digest()
    return digest, mag_id


def _fraction_counts(size: int, fractions: Mapping[str, float]) -> Mapping[str, int]:
    labels = ("train", "val", "test")
    exact = {label: size * float(fractions[label]) for label in labels}
    counts = {label: int(exact[label]) for label in labels}
    remainder = size - sum(counts.values())
    priority = sorted(labels, key=lambda label: (-(exact[label] - counts[label]), labels.index(label)))
    for label in priority[:remainder]:
        counts[label] += 1
    return counts


def compile_random_mag(manifest_path: Path, output_path: Path, seed: int,
                       train_fraction: float, validation_fraction: float,
                       test_fraction: float, stratify_by_source: bool = True,
                       fixed_source_labels: Optional[Mapping[str, str]] = None,
                       root_overrides=None) -> Path:
    """Deterministically shuffle MAG IDs and assign exact split counts."""
    manifest = load_manifest(manifest_path)
    rows = _manifest_rows(manifest, root_overrides)
    fractions = {"train": float(train_fraction), "val": float(validation_fraction),
                 "test": float(test_fraction)}
    if any(value < 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("random split fractions must be nonnegative and sum to one")
    fixed_source_labels = dict(fixed_source_labels or {})
    unknown_sources = set(fixed_source_labels) - {source for _, source, _ in rows}
    if unknown_sources:
        raise ValueError("fixed source labels name unknown sources: {}".format(sorted(unknown_sources)))
    assignments = {}
    groups = {}
    for mag_id, source, _ in rows:
        if source in fixed_source_labels:
            assignments[mag_id] = fixed_source_labels[source]
        else:
            groups.setdefault(source if stratify_by_source else "all", []).append(mag_id)
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda mag_id: _seeded_order(seed, mag_id))
        counts = _fraction_counts(len(ordered), fractions)
        cursor = 0
        for label in ("train", "val", "test"):
            end = cursor + counts[label]
            assignments.update((mag_id, label) for mag_id in ordered[cursor:end])
            cursor = end
    return compile_assignments(
        manifest_path, output_path, assignments,
        {"type": "random_mag", "seed": int(seed), "fractions": fractions,
         "stratify_by_source": bool(stratify_by_source),
         "fixed_source_labels": dict(sorted(fixed_source_labels.items()))},
        root_overrides,
    )


def materialize_split_plan(manifest_path: Path, split_config, output_dir: Path,
                           root_overrides=None) -> Path:
    """Resolve an existing plan or build the configured deterministic plan."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if split_config.strategy == "existing":
        path = Path(split_config.plan).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        load_split_plan(path, manifest.sha256)
        return path

    specification = {
        "strategy": split_config.strategy,
        "seed": int(split_config.seed),
        "train_fraction": float(split_config.train_fraction),
        "validation_fraction": float(split_config.validation_fraction),
        "test_fraction": float(split_config.test_fraction),
        "stratify_by_source": bool(split_config.stratify_by_source),
        "source_labels": dict(sorted(split_config.source_labels.items())),
    }
    encoded = json.dumps(
        {"manifest_sha256": manifest.sha256, "split": specification},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    key = hashlib.sha256(encoded).hexdigest()[:20]
    output_path = Path(output_dir) / "{}-{}.json".format(split_config.strategy, key)
    if output_path.exists():
        try:
            load_split_plan(output_path, manifest.sha256)
            return output_path
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    if split_config.strategy == "stored_labels":
        compile_stored_labels(manifest_path, output_path, root_overrides)
    elif split_config.strategy == "source_holdout":
        compile_source_holdout(
            manifest_path, output_path, split_config.source_labels, root_overrides
        )
    elif split_config.strategy == "random_mag":
        compile_random_mag(
            manifest_path, output_path, split_config.seed,
            split_config.train_fraction, split_config.validation_fraction,
            split_config.test_fraction, split_config.stratify_by_source,
            split_config.source_labels, root_overrides,
        )
    else:
        raise ValueError("unsupported split strategy {!r}".format(split_config.strategy))
    load_split_plan(output_path, manifest.sha256)
    return output_path


def selection_digest(assignments: Mapping[str, Assignment], mag_ids) -> str:
    digest = hashlib.sha256()
    for mag_id in sorted(mag_ids):
        assignment = assignments[mag_id]
        digest.update("{}\t{}\t{}\n".format(
            mag_id, assignment.source_name, assignment.label
        ).encode())
    return digest.hexdigest()
