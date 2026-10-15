#!/usr/bin/env python3
"""Create a read-only JSON/HTML profile of the hierarchical text corpus."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


LEVELS = ("gene", "multi_gene", "contig", "multi_contig", "mag", "multi_mag")


def add_values(path: Path, column: str, counter: Counter) -> None:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=[column], batch_size=100000):
        counter.update(value for value in batch.column(0).to_pylist() if value is not None)


def distribution(counter: Counter) -> dict:
    total = sum(counter.values())
    ordered = sorted(counter.items())
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "histogram": {}}

    def value_at(position: int):
        seen = 0
        for value, count in ordered:
            seen += count
            if position < seen:
                return value
        raise AssertionError("weighted position exceeds distribution")

    left = value_at((total - 1) // 2)
    right = value_at(total // 2)
    return {
        "count": total,
        "mean": sum(value * count for value, count in counter.items()) / total,
        "median": (left + right) / 2,
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "histogram": {str(key): value for key, value in ordered},
    }


def build_profile(facts_root: Path, examples_root: Path, max_parts: int | None = None) -> dict:
    fact_parts = sorted(facts_root.glob("*/*"))
    example_parts = sorted(examples_root.glob("*/*"))
    if max_parts:
        fact_parts = fact_parts[:max_parts]
        example_parts = example_parts[:max_parts]

    genes_per_mag = Counter()
    contigs_per_mag = Counter()
    genes_per_contig = Counter()
    source_mags = Counter()
    for part in fact_parts:
        mags = part / "mags.parquet"
        contigs = part / "contigs.parquet"
        if mags.exists():
            table = pq.read_table(mags, columns=["source_name", "n_genes", "n_contigs"])
            for source, genes, n_contigs in zip(*(column.to_pylist() for column in table.columns)):
                source_mags[source] += 1
                genes_per_mag[int(genes)] += 1
                contigs_per_mag[int(n_contigs)] += 1
        if contigs.exists():
            add_values(contigs, "n_genes", genes_per_contig)

    combinations = Counter()
    source_text_genes = Counter()
    for part in example_parts:
        path = part / "gene_examples.parquet"
        if not path.exists():
            continue
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["source_name", "information_types"], batch_size=100000):
            sources = batch.column(0).to_pylist()
            types = batch.column(1).to_pylist()
            for source, values in zip(sources, types):
                combinations["+".join(sorted(values)) or "none"] += 1
                source_text_genes[source] += 1

    levels = Counter()
    for report_path in examples_root.glob("*/*/report.json"):
        report = json.loads(report_path.read_text())
        levels.update({key: int(value) for key, value in report.get("counts", {}).items() if key in LEVELS})
    cross_report = examples_root / "multi_mag" / "report.json"
    if cross_report.exists():
        levels["multi_mag"] += sum(json.loads(cross_report.read_text()).get("examples", {}).values())

    total_genes = sum(value * count for value, count in genes_per_mag.items())
    text_genes = sum(combinations.values())
    return {
        "scope": {"fact_parts": len(fact_parts), "example_parts": len(example_parts),
                  "max_parts": max_parts},
        "counts": {"genes": total_genes, "genes_with_text": text_genes,
                   "text_coverage": text_genes / max(1, total_genes),
                   "examples_by_level": dict(levels), "mags_by_source": dict(source_mags),
                   "text_genes_by_source": dict(source_text_genes)},
        "information_type_combinations": dict(combinations.most_common()),
        "genes_per_mag": distribution(genes_per_mag),
        "genes_per_contig": distribution(genes_per_contig),
        "contigs_per_mag": distribution(contigs_per_mag),
    }


def write_html(path: Path, profile: dict) -> None:
    payload = json.dumps(profile).replace("</", "<\\/")
    title = "Hierarchical MAG–text corpus profile"
    document = f"""<!doctype html><meta charset=utf-8><title>{title}</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}td,th{{padding:.35rem .7rem;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}select{{font:inherit;padding:.3rem}}</style>
<h1>{title}</h1><p>This file is self-contained and read-only.</p>
<div id=summary></div><h2>Distribution</h2><select id=choice>
<option>genes_per_mag</option><option>genes_per_contig</option><option>contigs_per_mag</option>
<option>information_type_combinations</option></select><div id=table></div>
<script>const p={payload};
const c=p.counts; document.getElementById('summary').innerHTML=`<p><b>${{c.genes.toLocaleString()}}</b> genes; <b>${{c.genes_with_text.toLocaleString()}}</b> with text (${{(100*c.text_coverage).toFixed(2)}}%).</p><pre>${{JSON.stringify(c.examples_by_level,null,2)}}</pre>`;
function draw(){{let k=choice.value,d=p[k].histogram||p[k],rows=Object.entries(d).sort((a,b)=>b[1]-a[1]).slice(0,100);table.innerHTML='<table><tr><th>value</th><th>count</th></tr>'+rows.map(x=>`<tr><td>${{x[0]}}</td><td>${{x[1].toLocaleString()}}</td></tr>`).join('')+'</table>'}}choice.onchange=draw;draw();</script>"""
    path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts-root", type=Path, required=True)
    parser.add_argument("--examples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output path without an extension")
    parser.add_argument("--max-parts", type=int, help="Profile a deterministic prefix for a quick pilot")
    args = parser.parse_args()
    profile = build_profile(args.facts_root, args.examples_root, args.max_parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    write_html(args.output.with_suffix(".html"), profile)


if __name__ == "__main__":
    main()
