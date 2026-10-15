#!/usr/bin/env python3
"""Normalize JGI archives into canonical per-MAG records.

The worker reads JGI outer ZIP parts directly, normalizes their bin tarballs
into the generic per-MAG layout. Rich text facts are built directly from the
archives by ``clip_text/facts.py``. Files are published atomically and are safe
to resume after preemption.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SOURCE_DIR = Path(os.environ["SOURCE_DIR"])
SOURCE_NAME = os.environ["SOURCE_NAME"]
MAG_PREFIX = os.environ.get("MAG_PREFIX", SOURCE_NAME)
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"])
WORKERS = int(os.environ.get("WORKERS", "16"))
MAX_ZIPS = int(os.environ.get("MAX_ZIPS", "0")) or None
MAX_BINS = int(os.environ.get("MAX_BINS", "0")) or None
WRITE_MAGS = os.environ.get("WRITE_MAGS", "1") == "1"
NORMALIZE_GENE_IDS = os.environ.get("NORMALIZE_GENE_IDS", "1") == "1"

TAX_RANKS = ("domain", "phylum", "class_", "order", "family", "genus", "species")
GENE_RE = re.compile(r"^(?P<contig>.+)_(?P<start>[0-9]+)_(?P<end>[0-9]+)$")
NORMALIZATION_VERSION = 2


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def fasta_records(text: str):
    header = None
    chunks = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:]
            chunks = []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def normalize_genes(faa_text: str):
    records = []
    old_to_new = {}
    new_seen = set()
    coordinate_counts = defaultdict(int)
    for header, sequence in fasta_records(faa_text):
        old_id = header.split()[0]
        match = GENE_RE.match(old_id)
        if not match:
            raise ValueError(f"unparseable JGI gene ID: {old_id}")
        contig = match.group("contig")
        start = int(match.group("start"))
        if NORMALIZE_GENE_IDS:
            coordinate_key = (contig, start)
            ordinal = coordinate_counts[coordinate_key]
            coordinate_counts[coordinate_key] += 1
            if ordinal >= 1000:
                raise ValueError(f"at least 1000 genes share one start coordinate: {contig}:{start}")
            # The generic reader treats the final integer as the within-contig
            # sort key. Scaling leaves room for legitimate same-start calls.
            new_id = f"{contig}_{start * 1000 + ordinal}"
        else:
            new_id = old_id
        if new_id in new_seen:
            raise ValueError(f"normalized gene-ID collision: {new_id}")
        if not sequence:
            raise ValueError(f"empty protein sequence: {old_id}")
        new_seen.add(new_id)
        old_to_new[old_id] = new_id
        records.append((new_id, old_id, sequence))
    if not records:
        raise ValueError("protein FASTA contains no records")
    return records, old_to_new


def assignments(text: str, prefix: str, output_prefix: str, mapping: dict[str, str]):
    result = defaultdict(list)
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0] not in mapping:
            continue
        value = parts[2]
        if prefix and value.startswith(prefix):
            value = value[len(prefix):]
        if value:
            result[mapping[parts[0]]].append(output_prefix + value)
    return result


def consensus_taxonomy(text: str):
    counts = Counter()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            lineage = tuple(parts[4].split(";")[:7])
            if lineage:
                counts[lineage] += 1
    if not counts:
        return [""] * 7
    best = counts.most_common(1)[0][0]
    return list(best) + [""] * (7 - len(best))


def gene_products(text: str, mapping: dict[str, str]):
    result = {}
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2 or parts[0] not in mapping:
            continue
        product = parts[1].strip()
        product_source = parts[2].strip() if len(parts) > 2 else ""
        if product:
            result[mapping[parts[0]]] = (parts[0], product, product_source)
    return result


def tar_members(payload: bytes):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    files[Path(member.name).name] = handle.read()
    return files


def prepare_bin(outer_zip: Path, inner_path: str, payload: bytes):
    original_bin_id = Path(inner_path).name.removesuffix(".tar.gz")
    mag_id = f"{MAG_PREFIX}__{original_bin_id}" if MAG_PREFIX else original_bin_id
    sample_folder = inner_path.split("/", 1)[0]
    files = tar_members(payload)
    faa_name = f"{original_bin_id}.faa"
    if faa_name not in files:
        raise ValueError(f"{inner_path}: missing {faa_name}")
    records, mapping = normalize_genes(files[faa_name].decode(errors="replace"))
    ko = assignments(files.get(f"{original_bin_id}.ko.txt", b"").decode(errors="replace"),
                     "KO:", "ko:", mapping)
    ec = assignments(files.get(f"{original_bin_id}.ec.txt", b"").decode(errors="replace"),
                     "EC:", "", mapping)
    tax = consensus_taxonomy(files.get(f"{original_bin_id}.phylodist.txt", b"").decode(errors="replace"))
    products = gene_products(files.get(f"{original_bin_id}.gene_product.txt", b"").decode(errors="replace"), mapping)
    provenance = {
        "mag_id": mag_id,
        "source_name": SOURCE_NAME,
        "source_folder": sample_folder,
        "original_bin_id": original_bin_id,
        "outer_zip": outer_zip.name,
        "inner_path": inner_path,
        "n_genes": len(records),
        "n_gene_products": len(products),
        "taxonomy": dict(zip(TAX_RANKS, tax)),
        "normalization_version": NORMALIZATION_VERSION,
    }
    if WRITE_MAGS:
        out_dir = OUTPUT_ROOT / "per_mag" / mag_id
        faa_lines = []
        for new_id, old_id, sequence in records:
            faa_lines.extend((f">{new_id} original_gene_id={old_id}\n", f"{sequence}\n"))
        ann_lines = ["gene_id\tKO\tGO\tEC\n"]
        for new_id, _, _ in records:
            ann_lines.append(f"{new_id}\t{','.join(ko[new_id])}\t\t{','.join(ec[new_id])}\n")
        product_lines = ["gene_id\toriginal_gene_id\tproduct_text\tproduct_source\n"]
        for new_id, _, _ in records:
            if new_id in products:
                old_id, product, product_source = products[new_id]
                product_lines.append(f"{new_id}\t{old_id}\t{product}\t{product_source}\n")
        taxonomy_lines = [f"{rank.removesuffix('_')}\t{value}\n" for rank, value in zip(TAX_RANKS, tax)]
        atomic_bytes(out_dir / f"{mag_id}_genes.faa", "".join(faa_lines).encode())
        atomic_bytes(out_dir / "ko_go_ec.tsv", "".join(ann_lines).encode())
        atomic_bytes(out_dir / "gene_product.tsv", "".join(product_lines).encode())
        atomic_bytes(out_dir / "taxonomy.tsv", "".join(taxonomy_lines).encode())
        atomic_bytes(out_dir / "provenance.json", (json.dumps(provenance, sort_keys=True) + "\n").encode())
        atomic_bytes(out_dir / "_ANNOTATION_DONE", b"")
    return provenance


def process_zip(zip_path_string: str, bin_limit: int | None = None):
    zip_path = Path(zip_path_string)
    report_path = OUTPUT_ROOT / "zip_reports" / f"{zip_path.stem}.json"
    if bin_limit is None and report_path.exists():
        prior = json.loads(report_path.read_text())
        if (prior.get("source_name") == SOURCE_NAME
                and prior.get("write_mags") == WRITE_MAGS
                and prior.get("normalization_version") == NORMALIZATION_VERSION
                and prior.get("normalize_gene_ids") == NORMALIZE_GENE_IDS):
            return prior["provenances"], prior.get("errors", [])
    provenances = []
    errors = []
    with zipfile.ZipFile(zip_path) as outer:
        inner_names = sorted(
            name for name in outer.namelist()
            if "/Binning_Data/" in name and name.endswith(".tar.gz")
        )
        if bin_limit is not None:
            inner_names = inner_names[:bin_limit]
        for inner_path in inner_names:
            try:
                provenance = prepare_bin(zip_path, inner_path, outer.read(inner_path))
                provenances.append(provenance)
            except Exception as exc:
                errors.append({"outer_zip": zip_path.name, "inner_path": inner_path, "error": repr(exc)})
    if bin_limit is None:
        atomic_bytes(report_path, (json.dumps({
            "source_name": SOURCE_NAME,
            "write_mags": WRITE_MAGS,
            "normalization_version": NORMALIZATION_VERSION,
            "normalize_gene_ids": NORMALIZE_GENE_IDS,
            "provenances": provenances,
            "errors": errors,
        }, sort_keys=True) + "\n").encode())
    return provenances, errors


def write_source_manifest(rows):
    columns = ["genome", *TAX_RANKS, "source_name", "source_folder", "original_bin_id",
               "outer_zip", "inner_path", "n_genes", "n_gene_products"]
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for row in sorted(rows, key=lambda value: value["mag_id"]):
        writer.writerow([
            row["mag_id"],
            *(row["taxonomy"].get(rank, "") for rank in TAX_RANKS),
            row["source_name"], row["source_folder"], row["original_bin_id"],
            row["outer_zip"], row["inner_path"], row["n_genes"], row["n_gene_products"],
        ])
    atomic_bytes(OUTPUT_ROOT / "manifest.tsv", buffer.getvalue().encode())


def main():
    zip_paths = sorted((SOURCE_DIR / "outputs" / "jgi_downloads").glob("*.zip"))
    if MAX_ZIPS is not None:
        zip_paths = zip_paths[:MAX_ZIPS]
    if not zip_paths:
        raise SystemExit("no JGI ZIP parts found")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_errors = []
    remaining_bins = MAX_BINS
    tasks = []
    for zip_path in zip_paths:
        if remaining_bins is not None and remaining_bins <= 0:
            break
        limit = remaining_bins
        tasks.append((str(zip_path), limit))
        if remaining_bins is not None:
            with zipfile.ZipFile(zip_path) as outer:
                count = sum("/Binning_Data/" in n and n.endswith(".tar.gz") for n in outer.namelist())
            remaining_bins -= min(remaining_bins, count)
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(tasks))) as pool:
        futures = {pool.submit(process_zip, path, limit): path for path, limit in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            rows, errors = future.result()
            all_rows.extend(rows)
            all_errors.extend(errors)
            print(f"[prepare] {number}/{len(tasks)} ZIPs; {len(all_rows)} MAGs; "
                  f"{len(all_errors)} errors", flush=True)
    if WRITE_MAGS:
        write_source_manifest(all_rows)
    report = {
        "source_name": SOURCE_NAME,
        "source_dir": str(SOURCE_DIR),
        "output_root": str(OUTPUT_ROOT),
        "zip_parts_processed": len(tasks),
        "mags_processed": len(all_rows),
        "genes": sum(row["n_genes"] for row in all_rows),
        "errors": all_errors,
        "configuration": {
            "mag_prefix": MAG_PREFIX,
            "write_mags": WRITE_MAGS,
            "max_zips": MAX_ZIPS,
            "max_bins": MAX_BINS,
            "normalize_gene_ids": NORMALIZE_GENE_IDS,
        },
    }
    atomic_bytes(OUTPUT_ROOT / "preparation_report.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    if all_errors:
        raise SystemExit(f"preparation had {len(all_errors)} bin errors; see preparation_report.json")


if __name__ == "__main__":
    main()
