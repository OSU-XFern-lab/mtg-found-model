"""Build the lossless, source-separated fact layer for MAG text training.

One worker reads one JGI outer ZIP and writes three compressed Parquet files:

``mags.parquet``
    Assembly quality, taxonomy, environment, and provenance.
``contigs.parquet``
    Contig order, length, GC content, gene count, and CRISPR count.
``genes.parquet``
    Ordered gene addresses, cleaned products, protein hashes, per-gene
    taxonomy, and every KO/EC/COG/Pfam/TIGRFAM hit with its evidence.

Protein sequences are represented by their 16-byte MD5 hashes. The immutable
MAG shards already contain the ESM-C vectors and the same hashes, so copying
850 million sequences into this layer would waste substantial space.

Commands::

    python facts.py metadata --source-name IMG_JGI \
        --input sample_summary.csv --output metadata.parquet

    python facts.py build --source-name IMG_JGI --zip part.zip \
        --metadata metadata.parquet --output-root /path/to/facts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from vocab import _clean_prose


SCHEMA_NAME = "hierarchical-mag-text-facts"
SCHEMA_VERSION = 1
TAXONOMY_RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
ENVIRONMENT_FIELDS = (
    "sample_name", "ecosystem", "ecosystem_category", "ecosystem_type",
    "ecosystem_subtype", "specific_ecosystem", "habitat", "geographic_location",
    "isolation", "isolation_country", "latitude", "longitude", "depth_m",
    "elevation_m", "host_name", "ph",
)

GENE_ID = re.compile(r"^(?P<contig>.+)_(?P<start>[0-9]+)_(?P<end>[0-9]+)$")


HIT_TYPE = pa.list_(pa.struct([
    pa.field("accession", pa.string(), nullable=False),
    pa.field("percent_identity", pa.float32()),
    pa.field("query_start", pa.int32()),
    pa.field("query_end", pa.int32()),
    pa.field("subject_start", pa.int32()),
    pa.field("subject_end", pa.int32()),
    pa.field("evalue", pa.float64()),
    pa.field("bit_score", pa.float32()),
    pa.field("align_length", pa.int32()),
]))

MAG_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("sample_id", pa.string()), ("mag_id", pa.string()),
    ("original_bin_id", pa.string()), ("sample_folder", pa.string()),
    ("source_zip", pa.string()), ("inner_path", pa.string()),
    ("quality", pa.string()), ("completeness", pa.float32()),
    ("contamination", pa.float32()), ("total_bases", pa.int64()),
    ("n_contigs", pa.int32()), ("n_genes", pa.int32()),
    ("n50", pa.int64()), ("l50", pa.int32()),
    ("taxonomy", pa.list_(pa.string())),
    *((name, pa.string()) for name in ENVIRONMENT_FIELDS),
])

CONTIG_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("sample_id", pa.string()), ("mag_id", pa.string()),
    ("contig_id", pa.string()), ("contig_index", pa.int32()),
    ("length_nt", pa.int64()), ("gc_fraction", pa.float32()),
    ("n_genes", pa.int32()), ("n_crispr_arrays", pa.int32()),
])

GENE_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("sample_id", pa.string()), ("mag_id", pa.string()),
    ("contig_id", pa.string()), ("contig_index", pa.int32()),
    ("gene_index", pa.int32()), ("contig_gene_index", pa.int32()),
    ("gene_id", pa.string()), ("start", pa.int32()), ("end", pa.int32()),
    ("strand", pa.string()), ("partial", pa.bool_()),
    ("aa_length", pa.int32()), ("protein_md5", pa.binary(16)),
    ("product", pa.string()), ("product_source", pa.string()),
    ("phylo_percent_identity", pa.float32()),
    ("phylo_lineage", pa.list_(pa.string())),
    ("ko", HIT_TYPE), ("ec", HIT_TYPE), ("cog", HIT_TYPE),
    ("pfam", HIT_TYPE), ("tigr", HIT_TYPE),
])

METADATA_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("sample_id", pa.string()),
    *((name, pa.string()) for name in ENVIRONMENT_FIELDS),
])

CONTEXT_INDEX_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("mag_id", pa.string()), ("split", pa.string()),
    ("shard_id", pa.string()), ("shard_root", pa.string()),
    ("shard_path", pa.string()), ("local_mag_index", pa.int32()),
    ("n_contigs", pa.int32()), ("n_genes", pa.int32()),
])


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _integer(value: object) -> int | None:
    try:
        return int(float(_text(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        return float(_text(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _sample_id(bin_id: str) -> str:
    return bin_id.split("_s", 1)[0]


def mag_id(source_name: str, original_bin_id: str) -> str:
    if source_name == "IMG_JGI":
        return original_bin_id
    return f"{source_name}__{original_bin_id}"


def fasta_records(text: str) -> Iterable[tuple[str, str]]:
    header = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:].split()[0]
            chunks = []
        elif header is not None:
            chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def parse_attributes(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(";"):
        key, found, text = item.partition("=")
        if found:
            result[key] = text
    return result


def parse_gff(text: str) -> dict[str, dict]:
    records = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9 or parts[2] != "CDS":
            continue
        attrs = parse_attributes(parts[8])
        gene_id = attrs.get("ID", "")
        if not gene_id:
            continue
        records[gene_id] = {
            "contig": parts[0], "start": _integer(parts[3]), "end": _integer(parts[4]),
            "strand": parts[6], "partial": bool(attrs.get("partial")),
        }
    return records


def parse_products(text: str) -> dict[str, tuple[str, str]]:
    result = {}
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 2 and parts[0] and parts[1].strip():
            result[parts[0]] = (_clean_prose(parts[1]), parts[2].strip() if len(parts) > 2 else "")
    return result


HIT_COLUMNS = {
    "ko": dict(accession=2, percent=3, query_start=4, query_end=5, subject_start=6,
               subject_end=7, evalue=8, bit_score=9, align_length=10),
    "ec": dict(accession=2, percent=3, query_start=4, query_end=5, subject_start=6,
               subject_end=7, evalue=8, bit_score=9, align_length=10),
    "cog": dict(accession=1, percent=2, align_length=3, query_start=4, query_end=5,
                subject_start=6, subject_end=7, evalue=8, bit_score=9),
    "pfam": dict(accession=1, percent=2, query_start=3, query_end=4, subject_start=5,
                 subject_end=6, evalue=7, bit_score=8, align_length=9),
    "tigr": dict(accession=1, percent=2, query_start=3, query_end=4, subject_start=5,
                 subject_end=6, evalue=7, bit_score=8, align_length=9),
}


def normalise_accession(kind: str, value: str) -> str:
    accession = value.strip()
    if kind in {"ko", "ec"}:
        accession = accession.split(":", 1)[-1]
    elif kind == "pfam" and accession.lower().startswith("pfam"):
        accession = "PF" + accession[4:]
    return accession


def parse_hits(kind: str, text: str) -> dict[str, list[dict]]:
    columns = HIT_COLUMNS[kind]
    result: dict[str, list[dict]] = defaultdict(list)
    required = max(columns.values())
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) <= required:
            continue
        accession = normalise_accession(kind, parts[columns["accession"]])
        if not accession:
            continue
        result[parts[0]].append({
            "accession": accession,
            "percent_identity": _float(parts[columns["percent"]]),
            "query_start": _integer(parts[columns["query_start"]]),
            "query_end": _integer(parts[columns["query_end"]]),
            "subject_start": _integer(parts[columns["subject_start"]]),
            "subject_end": _integer(parts[columns["subject_end"]]),
            "evalue": _float(parts[columns["evalue"]]),
            "bit_score": _float(parts[columns["bit_score"]]),
            "align_length": _integer(parts[columns["align_length"]]),
        })
    for values in result.values():
        values.sort(key=lambda hit: (hit["query_start"] or 0, hit["evalue"] or float("inf")))
    return result


def parse_phylo(text: str) -> dict[str, tuple[float | None, list[str]]]:
    result = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            result[parts[0]] = (_float(parts[3]), [_text(x) for x in parts[4].split(";") if _text(x)])
    return result


def parse_datafile(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        key, found, value = line.partition("\t")
        if found:
            result[_text(key)] = _text(value)
    return result


def parse_crispr(text: str) -> Counter:
    arrays = set()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            arrays.add((parts[0], parts[1]))
    return Counter(contig for contig, _ in arrays)


def read_tar(payload: bytes) -> dict[str, str]:
    import io
    result = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name == "README":
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                result[name] = handle.read().decode("utf-8", errors="replace")
    return result


def _metadata_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path or not path.is_file():
        return {}
    return {
        (str(row["source_name"]), str(row["sample_id"])): row
        for row in pq.read_table(path).to_pylist()
    }


def build_bin(source_name: str, source_zip: str, inner_path: str, payload: bytes,
              metadata: dict[tuple[str, str], dict[str, str]]):
    original_bin_id = Path(inner_path).name.removesuffix(".tar.gz")
    sample_id = _sample_id(original_bin_id)
    current_mag = mag_id(source_name, original_bin_id)
    files = read_tar(payload)
    prefix = original_bin_id
    faa = dict(fasta_records(files.get(f"{prefix}.faa", "")))
    if not faa:
        raise ValueError(f"{inner_path}: no protein FASTA records")
    gff = parse_gff(files.get(f"{prefix}.gff", ""))
    products = parse_products(files.get(f"{prefix}.gene_product.txt", ""))
    phylo = parse_phylo(files.get(f"{prefix}.phylodist.txt", ""))
    hits = {kind: parse_hits(kind, files.get(f"{prefix}.{kind}.txt", "")) for kind in HIT_COLUMNS}
    data = parse_datafile(files.get(f"{prefix}_datafile.txt", ""))
    crispr = parse_crispr(files.get(f"{prefix}.crispr.txt", ""))
    nt = dict(fasta_records(files.get(f"{prefix}.fna", "")))

    # Reproduce the immutable shard address order: contigs lexicographically,
    # then genes by start coordinate with FASTA order as the stable tie-break.
    grouped: dict[str, list[tuple[int, str, str, dict]]] = defaultdict(list)
    for fasta_order, (gene_id, sequence) in enumerate(faa.items()):
        feature = dict(gff.get(gene_id, {}))
        match = GENE_ID.match(gene_id)
        contig = feature.get("contig") or (match.group("contig") if match else gene_id)
        start = feature.get("start") or (_integer(match.group("start")) if match else 0) or 0
        end = feature.get("end") or (_integer(match.group("end")) if match else start) or start
        feature.update(contig=contig, start=start, end=end)
        grouped[contig].append((fasta_order, gene_id, sequence, feature))

    gene_rows = []
    contig_rows = []
    global_gene_index = 0
    for contig_index, contig in enumerate(sorted(grouped)):
        ordered = sorted(grouped[contig], key=lambda item: (item[3]["start"], item[0]))
        sequence = nt.get(contig, "")
        gc = ((sequence.upper().count("G") + sequence.upper().count("C")) / len(sequence)
              if sequence else None)
        contig_rows.append({
            "source_name": source_name, "sample_id": sample_id, "mag_id": current_mag,
            "contig_id": contig, "contig_index": contig_index,
            "length_nt": len(sequence) or None, "gc_fraction": gc,
            "n_genes": len(ordered), "n_crispr_arrays": int(crispr.get(contig, 0)),
        })
        for within, (_, gene_id, amino_acids, feature) in enumerate(ordered):
            product, product_source = products.get(gene_id, ("", ""))
            phylo_identity, lineage = phylo.get(gene_id, (None, []))
            gene_rows.append({
                "source_name": source_name, "sample_id": sample_id, "mag_id": current_mag,
                "contig_id": contig, "contig_index": contig_index,
                "gene_index": global_gene_index, "contig_gene_index": within,
                "gene_id": gene_id, "start": feature["start"], "end": feature["end"],
                "strand": feature.get("strand", ""), "partial": bool(feature.get("partial")),
                "aa_length": len(amino_acids),
                "protein_md5": hashlib.md5(amino_acids.encode("ascii", errors="ignore")).digest(),
                "product": product, "product_source": product_source,
                "phylo_percent_identity": phylo_identity, "phylo_lineage": lineage,
                **{kind: hits[kind].get(gene_id, []) for kind in HIT_COLUMNS},
            })
            global_gene_index += 1

    taxonomy_text = data.get("GTDB-TK lineage") or data.get("Bin Lineage") or ""
    taxonomy = [_text(x) for x in re.split(r"[,;]", taxonomy_text) if _text(x)]
    if not taxonomy:
        # Some engineered/host archives omit the optional bin datafile. The
        # shard preparation uses the same consensus fallback over per-gene
        # best-homolog lineages.
        lineages = Counter(tuple(values[1][:7]) for values in phylo.values() if values[1])
        if lineages:
            taxonomy = list(lineages.most_common(1)[0][0])
    env = metadata.get((source_name, sample_id), {})
    mag_row = {
        "source_name": source_name, "sample_id": sample_id, "mag_id": current_mag,
        "original_bin_id": original_bin_id, "sample_folder": inner_path.split("/", 1)[0],
        "source_zip": source_zip, "inner_path": inner_path,
        "quality": data.get("Bin Quality", ""),
        "completeness": _float(data.get("Bin Completeness") or data.get("Bin Completenesss")),
        "contamination": _float(data.get("Bin Contamination") or data.get("Bin Contamintation")),
        "total_bases": _integer(data.get("Total Number of Bases")),
        "n_contigs": len(contig_rows), "n_genes": len(gene_rows),
        "n50": _integer(data.get("Bin N50")), "l50": _integer(data.get("Bin L50")),
        "taxonomy": taxonomy,
        **{name: _text(env.get(name, "")) for name in ENVIRONMENT_FIELDS},
    }
    return mag_row, contig_rows, gene_rows


class Writers:
    def __init__(self, root: Path):
        self.mag = pq.ParquetWriter(root / "mags.parquet", MAG_SCHEMA, compression="zstd")
        self.contig = pq.ParquetWriter(root / "contigs.parquet", CONTIG_SCHEMA, compression="zstd")
        self.gene = pq.ParquetWriter(root / "genes.parquet", GENE_SCHEMA, compression="zstd",
                                     use_dictionary=True)

    def write(self, mag: dict, contigs: list[dict], genes: list[dict]) -> None:
        self.mag.write_table(pa.Table.from_pylist([mag], schema=MAG_SCHEMA))
        self.contig.write_table(pa.Table.from_pylist(contigs, schema=CONTIG_SCHEMA))
        self.gene.write_table(pa.Table.from_pylist(genes, schema=GENE_SCHEMA), row_group_size=65536)

    def close(self) -> None:
        self.mag.close(); self.contig.close(); self.gene.close()


def build_zip(args: argparse.Namespace) -> None:
    zip_path = args.zip.resolve()
    final = args.output_root / args.source_name / zip_path.stem
    if (final / "COMPLETE").is_file():
        print(json.dumps({"state": "already_complete", "path": str(final)}))
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise RuntimeError(f"refusing to overwrite incomplete output: {final}")
    stage = Path(tempfile.mkdtemp(prefix=f".{zip_path.stem}.", dir=final.parent))
    metadata = _metadata_rows(args.metadata)
    errors = []
    totals = Counter()
    writers = Writers(stage)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = sorted(name for name in archive.namelist()
                           if "/Binning_Data/" in name and name.endswith(".tar.gz"))
            if args.max_bins:
                names = names[:args.max_bins]
            for inner_path in names:
                try:
                    mag, contigs, genes = build_bin(
                        args.source_name, zip_path.name, inner_path, archive.read(inner_path), metadata)
                    writers.write(mag, contigs, genes)
                    totals.update(mags=1, contigs=len(contigs), genes=len(genes),
                                  informative_genes=sum(bool(row["product"] or row["ko"] or row["ec"] or
                                                             row["cog"] or row["pfam"] or row["tigr"])
                                                        for row in genes))
                except Exception as error:
                    errors.append({"inner_path": inner_path, "error": repr(error)})
        writers.close()
        report = {
            "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "source_name": args.source_name, "source_zip": str(zip_path),
            "max_bins": args.max_bins, "counts": dict(totals), "errors": errors,
        }
        (stage / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (stage / "COMPLETE").write_text("")
        os.replace(stage, final)
        print(json.dumps({**report, "path": str(final)}, sort_keys=True))
    except BaseException:
        try:
            writers.close()
        except BaseException:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


METADATA_COLUMNS = {
    "sample_name": "img_sample_name", "ecosystem": "img_ecosystem",
    "ecosystem_category": "img_ecosystem_category", "ecosystem_type": "img_ecosystem_type",
    "ecosystem_subtype": "img_ecosystem_subtype", "specific_ecosystem": "img_specific_ecosystem",
    "habitat": "img_habitat", "geographic_location": "img_geographic_location",
    "isolation": "img_isolation", "isolation_country": "img_isolation_country",
    "latitude": "img_latitude", "longitude": "img_longitude", "depth_m": "img_depth_in_meters",
    "elevation_m": "img_elevation_in_meters", "host_name": "img_host_name", "ph": "img_ph",
}


def build_metadata(args: argparse.Namespace) -> None:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with args.input.open(newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            sample_id = _text(row.get("query_value"))
            if not sample_id:
                continue
            values = {name: _text(row.get(column)) for name, column in METADATA_COLUMNS.items()}
            if not values["sample_name"]:
                values["sample_name"] = _text(row.get("organism_name"))
            if not values["ecosystem"]:
                values["ecosystem"] = {
                    "JGI_IMG_ENG": "Engineered",
                    "JGI_IMG_HOST_ASSOC": "Host-associated",
                }.get(args.source_name, "")
            rows.append({
                "source_name": args.source_name, "sample_id": sample_id,
                **values,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=METADATA_SCHEMA), temporary,
                       compression="zstd", use_dictionary=True)
        os.replace(temporary, args.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(json.dumps({"source_name": args.source_name, "samples": len(rows), "path": str(args.output)}))


def build_input_list(args: argparse.Namespace) -> None:
    paths = sorted((args.source_dir / "outputs" / "jgi_downloads").glob("*.zip"))
    if not paths:
        raise RuntimeError(f"no ZIP files found below {args.source_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{path.resolve()}\n" for path in paths)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(json.dumps({"zip_parts": len(paths), "path": str(args.output)}))


def build_context_index(args: argparse.Namespace) -> None:
    """Make the small MAG-address table used by rendered language examples."""
    manifest = json.loads(args.manifest.read_text())
    roots = {str(key): Path(value) for key, value in manifest["storage_roots"].items()}
    rows = []
    for shard in manifest["shards"]:
        path = roots[shard["root"]] / shard["path"]
        with (path / "mags.jsonl").open() as handle:
            for local_index, line in enumerate(handle):
                if not line.strip():
                    continue
                mag = json.loads(line)
                rows.append({
                    "source_name": str(mag["source_name"]), "mag_id": str(mag["mag_id"]),
                    "split": str(mag["split"]), "shard_id": str(shard["shard_id"]),
                    "shard_root": str(shard["root"]), "shard_path": str(shard["path"]),
                    "local_mag_index": local_index, "n_contigs": int(mag["n_contigs"]),
                    "n_genes": int(mag["n_genes"]),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=CONTEXT_INDEX_SCHEMA), temporary,
                       compression="zstd", use_dictionary=True)
        os.replace(temporary, args.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(json.dumps({"mags": len(rows), "path": str(args.output)}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata", help="compact a wide JGI sample-summary CSV")
    metadata.add_argument("--source-name", required=True)
    metadata.add_argument("--input", type=Path, required=True)
    metadata.add_argument("--output", type=Path, required=True)
    metadata.set_defaults(function=build_metadata)
    inputs = commands.add_parser("inputs", help="write a stable sorted list of source ZIP files")
    inputs.add_argument("--source-dir", type=Path, required=True)
    inputs.add_argument("--output", type=Path, required=True)
    inputs.set_defaults(function=build_input_list)
    context = commands.add_parser("context-index", help="index MAG shard addresses and splits")
    context.add_argument("--manifest", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    context.set_defaults(function=build_context_index)
    build = commands.add_parser("build", help="build one atomic fact part from one outer ZIP")
    build.add_argument("--source-name", required=True)
    build.add_argument("--zip", type=Path, required=True)
    build.add_argument("--metadata", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--max-bins", type=int)
    build.set_defaults(function=build_zip)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
