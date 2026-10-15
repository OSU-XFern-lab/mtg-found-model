"""Read registered per-MAG sources as one canonical record stream."""

from __future__ import annotations

import csv
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:  # Minimal no-output fallback for lightweight conversion/test environments.
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count=1):
            pass

REPO_ROOT = Path(__file__).resolve().parents[1]

MOTUS_PER_MAG = REPO_ROOT / "data" / "motus" / "outputs" / "annotations" / "per_mag"
MOTUS_MANIFEST = REPO_ROOT / "data" / "motus" / "outputs" / "manifests" / "mag_manifest.tsv"

JGI_PER_MAG = REPO_ROOT / "data" / "jgi_img" / "outputs" / "aligned" / "per_mag"
JGI_MANIFEST = REPO_ROOT / "data" / "jgi_img" / "outputs" / "aligned" / "manifests" / "mag_manifest.tsv"

SOURCE_LAYOUTS = {
    "MOTUS": (MOTUS_PER_MAG, MOTUS_MANIFEST),
    "IMG_JGI": (JGI_PER_MAG, JGI_MANIFEST),
}

TAX_KEYS = ("domain", "phylum", "class", "order", "family", "genus", "species")


# ---------------------------------------------------------------------------
# Per-MAG file parsing (shared by MOTUS and IMG/JGI)
# ---------------------------------------------------------------------------

def _split_contig(gene_id: str, source_name: str) -> tuple[str, int]:
    """Return (contig_id, local_pos) for sorting genes within a contig."""
    if source_name == "IMG_JGI":
        # `<scaffold>_<start>_<end>`  e.g. `Ga0453081_0000381_140_502`
        parts = gene_id.split("_")
        if len(parts) >= 4:
            try:
                return "_".join(parts[:-2]), int(parts[-2])
            except ValueError:
                pass
        return gene_id, 0
    # MOTUS: `<mag_id>-scaffold_<c>_<g>`  -> contig=`<mag_id>-scaffold_<c>`, pos=g
    head, _, tail = gene_id.rpartition("_")
    try:
        return head, int(tail)
    except ValueError:
        return gene_id, 0


def read_faa(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    gid, buf = None, []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if gid is not None:
                    out[gid] = "".join(buf)
                gid = line[1:].split()[0]
                buf = []
            elif gid is not None:
                buf.append(line)
    if gid is not None:
        out[gid] = "".join(buf)
    return out


def read_annotations(path: Path) -> dict[str, dict]:
    """Parse MOTUS-style `ko_go_ec.tsv`. GO column is intentionally ignored."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            out[row["gene_id"]] = {
                "ko_ids": [k for k in (row.get("KO") or "").split(",") if k],
                "ec_ids": [e for e in (row.get("EC") or "").split(",") if e],
            }
    return out


def load_taxonomy(manifest_path: Path) -> dict[str, dict]:
    tax: dict[str, dict] = {}
    if not manifest_path.exists():
        return tax
    with manifest_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            # manifest column is `class_` because `class` is a reserved word.
            tax[row["genome"]] = {
                "domain": row.get("domain") or None,
                "phylum": row.get("phylum") or None,
                "class": row.get("class_") or None,
                "order": row.get("order") or None,
                "family": row.get("family") or None,
                "genus": row.get("genus") or None,
                "species": row.get("species") or None,
            }
    return tax


def build_record(arg: tuple) -> Optional[dict]:
    mag_id, mdir, source_name, taxonomy, keep_seq = arg
    faa = read_faa(mdir / f"{mag_id}_genes.faa")
    if not faa:
        return None
    ann = read_annotations(mdir / "ko_go_ec.tsv")
    contigs: dict[str, list] = {}
    for gid, aa in faa.items():
        contig_id, local = _split_contig(gid, source_name)
        contigs.setdefault(contig_id, []).append(
            {
                "gene_id": gid,
                "aa_sequence": aa if keep_seq else None,   # dropped at train time (huge RAM; only build needs it)
                "nt_sequence": None,
                "token_ids": {},
                "esmc_embedding": None,
                "ko_ids": ann.get(gid, {}).get("ko_ids", []),
                "ec_ids": ann.get(gid, {}).get("ec_ids", []),
                "go_ids": [],
                "metadata": {"local_pos": local},
            }
        )
    if not contigs:
        return None
    return {
        "mag_id": mag_id,
        "source_name": source_name,
        "taxonomy": taxonomy,
        "contigs": [
            {
                "contig_id": cid,
                "genes": sorted(genes, key=lambda g: g["metadata"]["local_pos"]),
                "metadata": {},
            }
            for cid, genes in sorted(contigs.items())
        ],
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Streaming loader (parallel I/O, bounded memory)
# ---------------------------------------------------------------------------

def _stream(
    per_mag_dir: Path,
    manifest_path: Path,
    source_name: str,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    skip: Optional[set] = None,
    keep_seq: bool = True,
) -> Iterator[dict]:
    """Yield canonical MAG records from a per-MAG dir, in parallel.

    `skip` is a set of mag_ids to ignore entirely (already ingested), so
    re-runs don't even read their files.

    Parallelism is bounded: at most ~2*workers records are ever in flight, so
    a slow consumer can't let completed (full-MAG) record dicts pile up in
    memory -- that unbounded buffering was what OOM-killed the old loader.
    """
    if not per_mag_dir.exists():
        return
    tax_table = load_taxonomy(manifest_path)
    mag_ids = sorted(os.listdir(per_mag_dir))
    if limit is not None:
        mag_ids = mag_ids[:limit]

    args: list[tuple] = []
    for mag_id in mag_ids:
        if skip and mag_id in skip:
            continue
        mdir = per_mag_dir / mag_id
        if not (mdir / "_ANNOTATION_DONE").exists():
            continue
        tax = tax_table.get(mag_id, {k: None for k in TAX_KEYS})
        args.append((mag_id, mdir, source_name, tax, keep_seq))

    if not args:
        return
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    args_it = iter(args)
    inflight: set = set()
    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(
        total=len(args), desc=f"Loading {source_name} (workers={workers})"
    ) as bar:
        for a in args_it:                              # prime the window
            inflight.add(ex.submit(build_record, a))
            if len(inflight) >= workers * 2:
                break
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                bar.update(1)
                rec = fut.result()
                if rec is not None:
                    yield rec
                nxt = next(args_it, None)              # top up as slots free
                if nxt is not None:
                    inflight.add(ex.submit(build_record, nxt))


def source_paths(source: str) -> tuple[Path, Path]:
    """Resolve a registered legacy source or the generic source layout."""
    if source in SOURCE_LAYOUTS:
        return SOURCE_LAYOUTS[source]
    base = REPO_ROOT / "data" / source
    return base / "per_mag", base / "manifest.tsv"


def stream_records(sources: Iterable[str], **kwargs) -> Iterator[dict]:
    """Generator across one or more sources (built-in or any dir under data/)."""
    for s in sources:
        per_mag, manifest = source_paths(s)
        yield from _stream(per_mag, manifest, s, **kwargs)


def load_records(source: str, **kwargs) -> List[dict]:
    return list(stream_records([source], **kwargs))
