#!/usr/bin/env python3
"""Render medium/high grounded language examples from one MAG fact part.

This stage never rereads a JGI archive. It consumes the stable Parquet facts
from ``facts.py`` and writes one compact row per informative gene, contig, and
MAG. Each row carries both medium and high answers so the trainer can choose a
richness level without duplicating biological references.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import views as V
from vocab import _clean_prose


HERE = Path(__file__).resolve().parent
SCHEMA_NAME = "hierarchical-mag-text-examples"
SCHEMA_VERSION = 3

EXAMPLE_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("split", pa.string()),
    ("level", pa.string()), ("example_id", pa.string()),
    ("context_json", pa.string()), ("question", pa.string()),
    ("question_variants", pa.list_(pa.string())),
    ("answer_medium", pa.string()), ("answer_high", pa.string()),
    ("information_types", pa.list_(pa.string())), ("fact_part", pa.string()),
])

QUESTION_BANK = {
    "gene": (
        "What functional role is supported by the available evidence?",
        "Summarize the annotations associated with the encoded protein.",
        "What can be inferred about function from these annotations?",
        "Describe the biological evidence available for this sequence.",
    ),
    "multi_gene": (
        "How do the encoded functions of these genes compare?",
        "Summarize the functional evidence across the selected genes.",
        "Which roles connect or distinguish these nearby genes?",
        "What functional picture emerges from this group of genes?",
    ),
    "contig": (
        "What functional potential is encoded in this genomic segment?",
        "Summarize the supported biological functions across the sequence.",
        "Which functions and systems are represented by the encoded genes?",
        "Describe the functional evidence available at contig scale.",
    ),
    "multi_contig": (
        "How does functional potential differ across these genomic segments?",
        "Compare the supported functions encoded by the selected contigs.",
        "What functions are shared or distinctive across these sequences?",
        "Summarize the functional evidence across the contig set.",
    ),
    "mag": (
        "What does the available evidence suggest about this genome's functional potential?",
        "Summarize the assembly, taxonomy, and supported encoded functions.",
        "Which biological capabilities are supported at genome scale?",
        "Describe the main functional signals represented in the assembly.",
    ),
    "multi_mag": (
        "How do the supported functional potentials of these genomes compare?",
        "Compare the main biological signals across the assemblies.",
        "Which encoded capabilities appear shared or distinctive across the genomes?",
        "Summarize the strongest similarities and differences in functional evidence.",
    ),
}


def prompt_fields(level: str, identifier: str) -> dict[str, object]:
    values = list(QUESTION_BANK[level])
    offset = int(identifier[:8], 16) % len(values)
    ordered = values[offset:] + values[:offset]
    return {"question": ordered[0], "question_variants": ordered}

MAG_SUMMARY_SCHEMA = pa.schema([
    ("source_name", pa.string()), ("split", pa.string()), ("mag_id", pa.string()),
    ("context_json", pa.string()), ("taxonomy", pa.list_(pa.string())),
    ("environment", pa.string()), ("n_genes", pa.int64()), ("n_contigs", pa.int64()),
    ("informative_genes", pa.int64()), ("information_types", pa.list_(pa.string())),
    ("categories", pa.list_(pa.string())), ("pathways", pa.list_(pa.string())),
    ("representative_roles", pa.list_(pa.string())),
])


def sentence(text: str) -> str:
    value = " ".join((text or "").split()).strip().rstrip(".")
    return value + "." if value else ""


def example_id(level: str, *address: object) -> str:
    raw = "\x1f".join([level, *(str(value) for value in address)]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def context(mag_id: str, *, include_mag: bool = False, contig_index: int | None = None,
            include_uct: bool = False, gene_indices: list[int] | None = None) -> str:
    mag = {"mag_id": mag_id, "include_mag": include_mag, "contigs": []}
    if contig_index is not None:
        mag["contigs"].append({
            "contig_index": int(contig_index), "include_uct": include_uct,
            "gene_indices": [int(value) for value in (gene_indices or [])],
        })
    return json.dumps({"mags": [mag]}, separators=(",", ":"), sort_keys=True)


def context_contigs(mag_id: str, contig_indices: list[int]) -> str:
    mag = {
        "mag_id": mag_id,
        "include_mag": False,
        "contigs": [
            {"contig_index": int(index), "include_uct": True, "gene_indices": []}
            for index in contig_indices
        ],
    }
    return json.dumps({"mags": [mag]}, separators=(",", ":"), sort_keys=True)


def taxonomy_phrase(mag: dict) -> str:
    lineage = [value for value in mag.get("taxonomy", []) or [] if value]
    return "a genome classified as " + "; ".join(lineage) if lineage else ""


def environment_phrase(mag: dict) -> str:
    values = []
    for key in ("ecosystem", "ecosystem_type", "ecosystem_subtype", "habitat",
                "geographic_location", "sample_name"):
        value = mag.get(key, "") or ""
        if value and value not in values:
            values.append(value)
    return " / ".join(values[:4])


def info_types(gene: dict, vocab: dict) -> list[str]:
    """List annotation sources that resolve to meaningful language."""
    result = []
    if V.informative_label(gene.get("product", "")):
        result.append("product")
    for kind in ("ko", "ec", "cog", "pfam", "tigr"):
        resolved = False
        for hit in gene.get(kind, []) or []:
            entry = vocab.get(kind, {}).get(hit.get("accession"), {})
            if V.informative_label(entry.get("name", "")) or entry.get("description"):
                resolved = True
                break
        if resolved:
            result.append(kind)
    return result


def best_role(gene: dict, vocab: dict) -> str:
    if V.informative_label(gene.get("product", "")):
        return _clean_prose(gene["product"])
    for kind in ("ko", "ec", "cog", "tigr", "pfam"):
        for hit in gene.get(kind, []) or []:
            entry = vocab.get(kind, {}).get(hit.get("accession"), {})
            if V.informative_label(entry.get("name", "")):
                return entry["name"]
    return ""


def cog_context(gene: dict, vocab: dict) -> tuple[list[str], list[str]]:
    categories, pathways = [], []
    for hit in gene.get("cog", []) or []:
        entry = vocab.get("cog", {}).get(hit.get("accession"), {})
        categories.extend(entry.get("categories", []) or [])
        if entry.get("pathway"):
            pathways.append(entry["pathway"])
    return categories, pathways


@dataclass
class Aggregate:
    n_genes: int = 0
    informative: int = 0
    roles: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    pathways: Counter = field(default_factory=Counter)
    representative: list[str] = field(default_factory=list)
    information_types: set[str] = field(default_factory=set)

    def add(self, role: str, categories: list[str], pathways: list[str], types: list[str]) -> None:
        self.n_genes += 1
        if types:
            self.informative += 1
            self.information_types.update(types)
        if role:
            self.roles[role] += 1
            if role not in self.representative and len(self.representative) < 12:
                self.representative.append(role)
        self.categories.update(categories)
        self.pathways.update(pathways)


def top_text(counter: Counter, limit: int = 5) -> str:
    return ", ".join(name for name, _ in counter.most_common(limit) if name)


def top_names(counter: Counter, limit: int = 5) -> list[str]:
    return [name for name, _ in counter.most_common(limit) if name]


def merged_aggregate(aggregates: list[Aggregate]) -> Aggregate:
    merged = Aggregate()
    for aggregate in aggregates:
        merged.n_genes += aggregate.n_genes
        merged.informative += aggregate.informative
        merged.roles.update(aggregate.roles)
        merged.categories.update(aggregate.categories)
        merged.pathways.update(aggregate.pathways)
        merged.information_types.update(aggregate.information_types)
        for role in aggregate.representative:
            if role not in merged.representative and len(merged.representative) < 12:
                merged.representative.append(role)
    return merged


def multi_gene_answers(candidates: list[dict]) -> tuple[str, str]:
    roles = []
    categories = Counter()
    sources = set()
    for candidate in candidates:
        role = candidate["role"]
        if role and role not in roles:
            roles.append(role)
        categories.update(candidate["categories"])
        sources.update(candidate["types"])
    medium = sentence(
        f"Across {len(candidates)} genes from one contig, supported "
        "annotations include " + "; ".join(roles)
    )
    high = [medium]
    broad = top_text(categories)
    if broad:
        high.append(sentence(f"Together, their broad functional evidence includes {broad}"))
    if sources:
        high.append(sentence("The assignments draw on " + ", ".join(sorted(sources)) + " evidence"))
    high.append("The genes are nearby in the assembly, but proximity alone does not establish a shared operon or pathway.")
    return medium, " ".join(high)


def multi_contig_answers(selected: list[tuple[int, Aggregate]]) -> tuple[str, str]:
    combined = merged_aggregate([aggregate for _, aggregate in selected])
    coverage = 100.0 * combined.informative / max(1, combined.n_genes)
    medium = [sentence(
        f"Across {len(selected)} contigs, {combined.n_genes:,} genes are represented; "
        f"{combined.informative:,} ({coverage:.1f}%) have functional or domain annotations"
    )]
    broad = top_text(combined.categories, 7)
    if broad:
        medium.append(sentence(f"Across the contigs, broad functional evidence is strongest for {broad}"))
    high = list(medium)
    highlights = []
    ordinals = ("first", "second", "third", "fourth")
    for position, (_, aggregate) in enumerate(selected[:4]):
        signal = top_text(aggregate.categories, 2) or top_text(aggregate.roles, 2)
        if signal:
            highlights.append(f"the {ordinals[position]} contig: {signal}")
    if highlights:
        high.append(sentence("Contig-specific highlights are " + "; ".join(highlights)))
    pathways = top_text(combined.pathways, 8)
    if pathways:
        high.append(sentence(f"Named pathway or system context includes {pathways}"))
    high.append("These summaries describe encoded potential; they do not demonstrate that the contigs act together.")
    return " ".join(medium), " ".join(high)


def contig_answers(meta: dict, agg: Aggregate) -> tuple[str, str]:
    coverage = 100.0 * agg.informative / max(1, agg.n_genes)
    core = (f"The sequence segment contains {agg.n_genes:,} genes; {agg.informative:,} "
            f"({coverage:.1f}%) have functional or domain annotations.")
    categories = top_text(agg.categories)
    pathways = top_text(agg.pathways)
    roles = top_text(agg.roles)
    medium_parts = [core]
    if categories:
        medium_parts.append(sentence(f"The strongest broad functional categories are {categories}"))
    elif roles:
        medium_parts.append(sentence(f"Frequently supported roles include {roles}"))
    if int(meta.get("n_crispr_arrays") or 0):
        medium_parts.append(sentence(f"Sequence analysis identified {int(meta['n_crispr_arrays'])} CRISPR array(s)"))
    high_parts = list(medium_parts)
    if pathways:
        high_parts.append(sentence(f"Named pathway or system context includes {pathways}"))
    if agg.representative:
        high_parts.append(sentence("Representative gene annotations in genomic order include " +
                                   "; ".join(agg.representative[:10])))
    high_parts.append("These are computational annotations; gene proximity alone does not prove an operon.")
    return " ".join(medium_parts), " ".join(high_parts)


def mag_answers(meta: dict, agg: Aggregate) -> tuple[str, str]:
    lineage = "; ".join(meta.get("taxonomy", []) or [])
    environment = environment_phrase(meta)
    quality = []
    if meta.get("completeness") is not None:
        quality.append(f"{float(meta['completeness']):.1f}% complete")
    if meta.get("contamination") is not None:
        quality.append(f"{float(meta['contamination']):.1f}% contamination")
    opening = "The assembled genome"
    if lineage:
        opening += " is classified as " + lineage
    if environment:
        opening += " and was recovered from " + environment
    opening = sentence(opening)
    structure = sentence(
        f"The assembly contains {int(meta.get('n_genes') or agg.n_genes):,} genes across "
        f"{int(meta.get('n_contigs') or 0):,} contigs"
    )
    coverage = sentence(
        f"{agg.informative:,} genes ({100.0 * agg.informative / max(1, agg.n_genes):.1f}%) "
        "have functional or domain annotations"
    )
    medium = [opening, structure, coverage]
    categories = top_text(agg.categories, 7)
    roles = top_text(agg.roles, 7)
    if categories:
        medium.append(sentence(f"Broad functional evidence is strongest for {categories}"))
    elif roles:
        medium.append(sentence(f"Frequently supported roles include {roles}"))
    high = list(medium)
    if quality:
        high.append(sentence("Assembly-quality estimates report " + " and ".join(quality)))
    pathways = top_text(agg.pathways, 8)
    if pathways:
        high.append(sentence(f"Named pathway or system context includes {pathways}"))
    if agg.representative:
        high.append(sentence("Representative supported gene roles include " +
                             "; ".join(agg.representative[:12])))
    high.append("These annotations indicate encoded functional potential, not an experimentally demonstrated phenotype.")
    return " ".join(part for part in medium if part), " ".join(part for part in high if part)


class BufferedWriter:
    def __init__(self, path: Path, schema: pa.Schema = EXAMPLE_SCHEMA):
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
        self.buffer = []
        self.rows = 0

    def add(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= 10000:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.rows += len(self.buffer)
            self.buffer = []

    def close(self) -> None:
        self.flush(); self.writer.close()


def load_context_index(path: Path) -> dict[str, dict]:
    return {row["mag_id"]: row for row in pq.read_table(path).to_pylist()}


def render_part(args: argparse.Namespace) -> None:
    fact_part = args.fact_part.resolve()
    final = args.output_root / fact_part.parent.name / fact_part.name
    if (final / "COMPLETE").is_file():
        print(json.dumps({"state": "already_complete", "path": str(final)}))
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise RuntimeError(f"refusing to overwrite incomplete output: {final}")
    stage = Path(tempfile.mkdtemp(prefix=f".{fact_part.name}.", dir=final.parent))
    with gzip.open(args.vocab, "rt") as handle:
        vocab = json.load(handle)
    index = load_context_index(args.context_index)
    mag_meta = {row["mag_id"]: row for row in pq.read_table(fact_part / "mags.parquet").to_pylist()}
    contig_meta = {(row["mag_id"], int(row["contig_index"])): row
                   for row in pq.read_table(fact_part / "contigs.parquet").to_pylist()}
    gene_writer = BufferedWriter(stage / "gene_examples.parquet")
    multi_gene_writer = BufferedWriter(stage / "multi_gene_examples.parquet")
    contig_writer = BufferedWriter(stage / "contig_examples.parquet")
    multi_contig_writer = BufferedWriter(stage / "multi_contig_examples.parquet")
    mag_writer = BufferedWriter(stage / "mag_examples.parquet")
    summary_writer = BufferedWriter(stage / "mag_summaries.parquet", MAG_SUMMARY_SCHEMA)
    contig_agg: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    mag_agg: dict[str, Aggregate] = defaultdict(Aggregate)
    gene_candidates: dict[tuple[str, int], list[dict]] = defaultdict(list)
    missing_mags = set()
    try:
        parquet = pq.ParquetFile(fact_part / "genes.parquet")
        for batch in parquet.iter_batches(batch_size=10000):
            for gene in batch.to_pylist():
                mag_id = gene["mag_id"]
                address = index.get(mag_id)
                if address is None:
                    missing_mags.add(mag_id)
                    continue
                types = info_types(gene, vocab)
                role = best_role(gene, vocab)
                categories, pathways = cog_context(gene, vocab)
                key = (mag_id, int(gene["contig_index"]))
                contig_agg[key].add(role, categories, pathways, types)
                mag_agg[mag_id].add(role, categories, pathways, types)
                if not types:
                    continue
                if role and len(gene_candidates[key]) < 6:
                    gene_candidates[key].append({
                        "gene_index": int(gene["gene_index"]), "role": role,
                        "categories": categories, "types": types,
                    })
                meta = mag_meta[mag_id]
                rng = V.stable_rng(SCHEMA_VERSION, mag_id, gene["gene_index"])
                views = V.build_views_from_fact(
                    gene, vocab, rng, taxonomy_phrase(meta), environment_phrase(meta))
                identifier = example_id("gene", mag_id, gene["gene_index"])
                gene_writer.add({
                    "source_name": gene["source_name"], "split": address["split"],
                    "level": "gene", "example_id": identifier,
                    "context_json": context(mag_id, contig_index=gene["contig_index"],
                                            gene_indices=[gene["gene_index"]]),
                    **prompt_fields("gene", identifier),
                    "answer_medium": V.compose_richness(views, "medium"),
                    "answer_high": V.compose_richness(views, "high"),
                    "information_types": types, "fact_part": fact_part.name,
                })

        candidates_by_mag: dict[str, list[tuple[int, list[dict]]]] = defaultdict(list)
        for (mag_id, contig_index), candidates in gene_candidates.items():
            if len(candidates) >= 2:
                candidates_by_mag[mag_id].append((contig_index, candidates))
        for mag_id, choices in sorted(candidates_by_mag.items()):
            contig_index, candidates = max(
                choices, key=lambda item: (len(item[1]), sum(len(row["types"]) for row in item[1]), -item[0]))
            medium, high = multi_gene_answers(candidates)
            meta = mag_meta[mag_id]
            identifier = example_id("multi_gene", mag_id, contig_index)
            multi_gene_writer.add({
                "source_name": meta["source_name"], "split": index[mag_id]["split"],
                "level": "multi_gene", "example_id": identifier,
                "context_json": context(
                    mag_id, contig_index=contig_index,
                    gene_indices=[row["gene_index"] for row in candidates]),
                **prompt_fields("multi_gene", identifier),
                "answer_medium": medium, "answer_high": high,
                "information_types": sorted({kind for row in candidates for kind in row["types"]}),
                "fact_part": fact_part.name,
            })

        for (mag_id, contig_index), agg in sorted(contig_agg.items()):
            meta = contig_meta[(mag_id, contig_index)]
            medium, high = contig_answers(meta, agg)
            identifier = example_id("contig", mag_id, contig_index)
            contig_writer.add({
                "source_name": meta["source_name"], "split": index[mag_id]["split"],
                "level": "contig", "example_id": identifier,
                "context_json": context(mag_id, contig_index=contig_index, include_uct=True),
                **prompt_fields("contig", identifier),
                "answer_medium": medium, "answer_high": high,
                "information_types": sorted(agg.information_types), "fact_part": fact_part.name,
            })

        contigs_by_mag: dict[str, list[tuple[int, Aggregate]]] = defaultdict(list)
        for (mag_id, contig_index), aggregate in contig_agg.items():
            contigs_by_mag[mag_id].append((contig_index, aggregate))
        for mag_id, choices in sorted(contigs_by_mag.items()):
            selected = sorted(
                choices, key=lambda item: (item[1].informative, item[1].n_genes, -item[0]), reverse=True
            )[:4]
            if len(selected) < 2:
                continue
            medium, high = multi_contig_answers(selected)
            meta = mag_meta[mag_id]
            combined = merged_aggregate([aggregate for _, aggregate in selected])
            identifier = example_id("multi_contig", mag_id)
            multi_contig_writer.add({
                "source_name": meta["source_name"], "split": index[mag_id]["split"],
                "level": "multi_contig", "example_id": identifier,
                "context_json": context_contigs(mag_id, [contig_index for contig_index, _ in selected]),
                **prompt_fields("multi_contig", identifier),
                "answer_medium": medium, "answer_high": high,
                "information_types": sorted(combined.information_types), "fact_part": fact_part.name,
            })
        for mag_id, agg in sorted(mag_agg.items()):
            meta = mag_meta[mag_id]
            medium, high = mag_answers(meta, agg)
            identifier = example_id("mag", mag_id)
            mag_writer.add({
                "source_name": meta["source_name"], "split": index[mag_id]["split"],
                "level": "mag", "example_id": identifier,
                "context_json": context(mag_id, include_mag=True),
                **prompt_fields("mag", identifier),
                "answer_medium": medium, "answer_high": high,
                "information_types": sorted(agg.information_types), "fact_part": fact_part.name,
            })
            summary_writer.add({
                "source_name": meta["source_name"], "split": index[mag_id]["split"],
                "mag_id": mag_id, "context_json": context(mag_id, include_mag=True),
                "taxonomy": list(meta.get("taxonomy", []) or []),
                "environment": environment_phrase(meta),
                "n_genes": int(meta.get("n_genes") or agg.n_genes),
                "n_contigs": int(meta.get("n_contigs") or 0),
                "informative_genes": agg.informative,
                "information_types": sorted(agg.information_types),
                "categories": top_names(agg.categories, 12),
                "pathways": top_names(agg.pathways, 12),
                "representative_roles": agg.representative[:12],
            })
        writers = (gene_writer, multi_gene_writer, contig_writer, multi_contig_writer,
                   mag_writer, summary_writer)
        for writer in writers:
            writer.close()
        report = {
            "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "fact_part": str(fact_part),
            "counts": {
                "gene": gene_writer.rows, "multi_gene": multi_gene_writer.rows,
                "contig": contig_writer.rows, "multi_contig": multi_contig_writer.rows,
                "mag": mag_writer.rows, "mag_summary": summary_writer.rows,
            },
            "mags_without_embedding_context": len(missing_mags),
        }
        (stage / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (stage / "COMPLETE").write_text("")
        os.replace(stage, final)
        print(json.dumps({**report, "path": str(final)}, sort_keys=True))
    except BaseException:
        for writer in (gene_writer, multi_gene_writer, contig_writer, multi_contig_writer,
                       mag_writer, summary_writer):
            try:
                writer.close()
            except BaseException:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fact-part", type=Path, required=True)
    result.add_argument("--context-index", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--vocab", type=Path, default=HERE / "vocab.json.gz")
    return result


def main() -> None:
    render_part(parser().parse_args())


if __name__ == "__main__":
    main()
