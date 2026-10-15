#!/usr/bin/env python
"""Normalize new MAGs and append them to a versioned shard dataset.

`convert` creates the generic canonical source layout.  `append` performs
known-protein reuse, ESM-C for new proteins, every available discrete
assignment, immutable shard writes, and child-manifest publication.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import data_sources as ds
from data_sources import TAX_KEYS, source_paths, stream_records
from mag_shards.ingest import append_records
from mag_shards.manifest import sha256_file

# ---------------------------------------------------------------------------
# Operation: convert | append. Edit these values for direct runs. The Slurm
# wrapper supplies the same named environment variables for scheduled runs.
# ---------------------------------------------------------------------------
ACTION = os.environ.get("MAG_INGEST_ACTION", "append")
SOURCE = os.environ.get("SOURCE", "NEW_SOURCE")

# convert settings
INPUT_ROOT = Path(os.environ["INPUT_ROOT"]) if os.environ.get("INPUT_ROOT") else None
PREFER_SEQUENCE = os.environ.get("PREFER_SEQUENCE", "faa")       # faa | fna
PRODIGAL_MODE = os.environ.get("PRODIGAL_MODE", "meta")          # meta | single

# append settings
PARENT_MANIFEST = (Path(os.environ["PARENT_MANIFEST"])
                   if os.environ.get("PARENT_MANIFEST") else None)
OUTPUT_MANIFEST = (Path(os.environ["OUTPUT_MANIFEST"])
                   if os.environ.get("OUTPUT_MANIFEST") else None)
DATASET_ID = os.environ.get("DATASET_ID") or None
SHARD_ROOT_ALIAS = os.environ.get("SHARD_ROOT_ALIAS", "project")
SHARD_ROOT_PATH = (Path(os.environ["SHARD_ROOT_PATH"])
                   if os.environ.get("SHARD_ROOT_PATH") else None)
SHARD_RELATIVE_DIR = os.environ.get("SHARD_RELATIVE_DIR", "datasets/incremental")
TARGET_SHARD_GIB = float(os.environ.get("TARGET_SHARD_GIB", "4"))
INGEST_SPLIT = os.environ.get("INGEST_SPLIT") or None
MAX_MAGS = int(os.environ["MAX_MAGS"]) if os.environ.get("MAX_MAGS") else None
SOURCE_WORKERS = int(os.environ.get("SOURCE_WORKERS", "8"))


def _read_proteins(mag_dir: Path, prefer: str, prodigal_mode: str):
    faa = next(iter(sorted(mag_dir.glob("*.faa"))), None)
    fna = next(iter(sorted(mag_dir.glob("*.fna"))), None)
    use_faa = faa if faa and (prefer == "faa" or not fna) else None
    if use_faa:
        return ds.read_faa(use_faa)
    if fna:
        output = mag_dir / "_prodigal.faa"
        subprocess.run(["prodigal", "-i", str(fna), "-a", str(output), "-p", prodigal_mode, "-q"], check=True)
        return ds.read_faa(output)
    raise FileNotFoundError("no .faa or .fna in {}".format(mag_dir))


def _taxonomy(mag_dir: Path):
    result = {rank: None for rank in TAX_KEYS}; path = mag_dir / "taxonomy.tsv"
    if path.exists():
        for line in path.read_text().splitlines():
            rank, _, value = line.partition("\t")
            if rank in result:
                result[rank] = value or None
    return result


def convert_input(input_root: Path, source: str, prefer="faa", prodigal_mode="meta"):
    per_mag, manifest = source_paths(source); per_mag.mkdir(parents=True, exist_ok=True)
    existing_taxonomy = ds.load_taxonomy(manifest); new_rows = []; converted = []
    for source_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        mag_id = source_dir.name; output_dir = per_mag / mag_id
        if (output_dir / "_ANNOTATION_DONE").exists():
            converted.append(mag_id); continue
        proteins = _read_proteins(source_dir, prefer, prodigal_mode)
        annotations = ds.read_annotations(source_dir / "ko_go_ec.tsv")
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "{}_genes.faa".format(mag_id)).open("w") as handle:
            for gene_id, sequence in proteins.items():
                handle.write(">{}\n{}\n".format(gene_id, sequence))
        with (output_dir / "ko_go_ec.tsv").open("w") as handle:
            handle.write("gene_id\tKO\tEC\n")
            for gene_id in proteins:
                value = annotations.get(gene_id, {})
                handle.write("{}\t{}\t{}\n".format(
                    gene_id, ",".join(value.get("ko_ids", [])), ",".join(value.get("ec_ids", []))))
        taxonomy = _taxonomy(source_dir)
        if mag_id not in existing_taxonomy:
            new_rows.append([mag_id] + [taxonomy[rank] or "" for rank in TAX_KEYS])
        (output_dir / "_ANNOTATION_DONE").touch(); converted.append(mag_id)
        print("[convert] {}: {} genes".format(mag_id, len(proteins)), flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new_file = not manifest.exists()
    with manifest.open("a") as handle:
        if new_file:
            handle.write("\t".join(["genome"] + [("class_" if rank == "class" else rank) for rank in TAX_KEYS]) + "\n")
        for row in new_rows:
            handle.write("\t".join(row) + "\n")
    print("[convert] {} MAGs available in data/{}".format(len(converted), source), flush=True)
    return converted


def main():
    if ACTION == "convert":
        if INPUT_ROOT is None:
            raise ValueError("set INPUT_ROOT for ACTION='convert'")
        if PREFER_SEQUENCE not in ("faa", "fna") or PRODIGAL_MODE not in ("meta", "single"):
            raise ValueError("invalid conversion settings")
        convert_input(INPUT_ROOT, SOURCE, PREFER_SEQUENCE, PRODIGAL_MODE)
        return
    if ACTION != "append":
        raise ValueError("ACTION must be 'convert' or 'append'")
    if PARENT_MANIFEST is None or OUTPUT_MANIFEST is None or SHARD_ROOT_PATH is None:
        raise ValueError("append requires PARENT_MANIFEST, OUTPUT_MANIFEST, and SHARD_ROOT_PATH")
    _, source_manifest = source_paths(SOURCE)
    source_meta = {SOURCE: {"manifest": str(source_manifest),
                            "manifest_sha256": sha256_file(source_manifest) if source_manifest.exists() else None}}
    records = stream_records([SOURCE], limit=MAX_MAGS, workers=SOURCE_WORKERS, keep_seq=True)
    child, report = append_records(
        PARENT_MANIFEST, records, OUTPUT_MANIFEST,
        root_alias=SHARD_ROOT_ALIAS, root_path=SHARD_ROOT_PATH,
        relative_dataset_dir=SHARD_RELATIVE_DIR,
        target_shard_bytes=int(TARGET_SHARD_GIB * 1024 ** 3),
        split_override=INGEST_SPLIT, dataset_id=DATASET_ID, source_metadata=source_meta)
    print(json.dumps({**report, "manifest": str(OUTPUT_MANIFEST), "sha256": child.sha256}, indent=2))


if __name__ == "__main__":
    main()
