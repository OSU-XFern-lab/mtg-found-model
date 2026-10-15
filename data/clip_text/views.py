#!/usr/bin/env python3
"""Build several text descriptions, or *views*, of one gene.

``vocab.py`` translates database identifiers into words. This file chooses the
facts to describe. ``textify.py`` turns those facts into sentences.

Database identifiers remain in the structured data but never enter captions. This
prevents the model from learning an identifier-to-protein lookup table.

Try it with: ``python views.py --demo PATH_TO_EXTRACTED_BIN``
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

import textify as T
from vocab import _clean_prose

HERE = Path(__file__).resolve().parent

# Column positions in JGI's tab-separated annotation files.
HIT_FILES = {
    "ko":   {"suffix": "ko.txt",   "accession": 2, "start": None, "evalue": 9},
    "ec":   {"suffix": "ec.txt",   "accession": 2, "start": None, "evalue": 9},
    "cog":  {"suffix": "cog.txt",  "accession": 1, "start": 4,    "evalue": 8},
    "pfam": {"suffix": "pfam.txt", "accession": 1, "start": 3,    "evalue": 7},
    "tigr": {"suffix": "tigr.txt", "accession": 1, "start": 3,    "evalue": 7},
}

# More specific facts come first if a caption reaches its length limit.
VIEW_PRIORITY = ["product", "function", "enzyme", "domains", "orthology", "family",
                 "taxonomic", "environmental", "unannotated"]

GENERIC_LABEL = re.compile(
    r"(?:^|\b)(?:conserved\s+)?hypothetical protein(?:\b|$)|"
    r"(?:^|\b)uncharacteri[sz]ed protein(?:\b|$)|"
    r"protein (?:of )?unknown function|function unknown|"
    r"^unknown protein$|^predicted protein$|^unnamed protein product$",
    re.IGNORECASE,
)


def informative_label(text: str) -> bool:
    """Return true when a label says more than merely "unknown protein"."""
    value = _clean_prose(text or "").strip()
    return bool(value) and not bool(GENERIC_LABEL.search(value))


def load_index() -> dict:
    with gzip.open(HERE / "vocab.json.gz", "rt") as file:
        return json.load(file)


def normalise(kind: str, acc: str) -> str:
    """Remove prefixes that JGI adds but the reference vocabulary does not use."""
    accession = acc.strip()
    if kind in ("ko", "ec"):
        accession = accession.split(":", 1)[-1]
    elif kind == "pfam" and accession.lower().startswith("pfam"):
        accession = "PF" + accession[4:]
    return accession


def load_gene_annotations(bin_dir: Path) -> dict[str, dict]:
    """Read one extracted JGI bin and group all annotations by gene ID."""
    bin_id = next(bin_dir.glob("*.faa")).stem
    genes: dict[str, dict] = {}

    def gene_record(gene_id: str) -> dict:
        return genes.setdefault(
            gene_id,
            {"product": "", "source": "", **{kind: [] for kind in HIT_FILES}},
        )

    product_file = bin_dir / f"{bin_id}.gene_product.txt"
    if product_file.exists():
        for line in product_file.read_text(errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                entry = gene_record(parts[0])
                # Product names can contain identifiers such as "(TIGR01777 family)".
                entry["product"] = _clean_prose(parts[1].strip())
                entry["source"] = parts[2].strip() if len(parts) > 2 else ""

    for kind, columns in HIT_FILES.items():
        path = bin_dir / f"{bin_id}.{columns['suffix']}"
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) <= columns["evalue"]:
                continue
            accession = normalise(kind, parts[columns["accession"]])
            if not accession:
                continue
            start_column = columns["start"]
            has_numeric_start = start_column is not None and parts[start_column].isdigit()
            start = int(parts[start_column]) if has_numeric_start else 0
            evalue = parts[columns["evalue"]]
            gene_record(parts[0])[kind].append((accession, start, evalue))
    return genes


def _entry(index: dict, kind: str, accession: str) -> dict:
    """Return one vocabulary entry, or an empty dictionary when it is unknown."""
    return index.get(kind, {}).get(accession) or {}


def build_views(gene: dict, index: dict, rng: random.Random,
                taxonomy: str = "", environment: str = "") -> dict[str, str]:
    """Return ``{view name: sentence}`` for facts supported by this gene's annotations."""
    views: dict[str, str] = {}
    product = _clean_prose(gene.get("product", "") or "")
    if not informative_label(product):
        product = ""

    if product:
        views["product"] = T.render(rng, "product", product)

    # Do not repeat a KO fact when KO also supplied the product name.
    ko_hits = gene.get("ko", [])
    product_from_ko = gene.get("source", "").startswith("KO:")
    if ko_hits and not (product and product_from_ko):
        accession, _, evalue = ko_hits[0]
        entry = _entry(index, "ko", accession)
        if informative_label(entry.get("name", "")):
            text = T.hedge(rng, T.confidence(evalue)) + entry["name"]
            views["function"] = T.render(rng, "function", text)

    for accession, _, _ in gene.get("ec", [])[:1]:
        entry = _entry(index, "ec", accession)
        if informative_label(entry.get("name", "")):
            views["enzyme"] = T.render(rng, "enzyme", entry["name"])
            synonyms = entry.get("synonyms") or []
            if synonyms and rng.random() < 0.5:
                views["enzyme"] += f" Also known as {', '.join(synonyms[:2])}."

    for accession, _, _ in gene.get("cog", [])[:1]:
        entry = _entry(index, "cog", accession)
        if informative_label(entry.get("name", "")):
            text = T.render(rng, "orthology", entry["name"])
            categories = [c for c in entry.get("categories", []) if c]
            if categories and rng.random() < 0.7:
                text += " " + T.render(rng, "category", categories[0])
            if entry.get("pathway") and rng.random() < 0.5:
                text += " " + T.render(rng, "pathway", entry["pathway"])
            views["orthology"] = text

    # Domain order carries information about protein architecture.
    domains: list[str] = []
    for accession, start, _ in sorted(gene.get("pfam", []), key=lambda hit: hit[1]):
        name = _entry(index, "pfam", accession).get("name")
        if informative_label(name or "") and name not in domains:
            domains.append(name)
    if domains:
        views["domains"] = T.domain_chain(rng, domains)

    for accession, _, _ in gene.get("tigr", [])[:1]:
        entry = _entry(index, "tigr", accession)
        if entry.get("description"):
            views["family"] = T.render(rng, "family", entry["description"])
        elif informative_label(entry.get("name", "")):
            views["family"] = T.as_sentence(f"a member of the {entry['name']} family")

    if product:
        anchor = product
    elif "function" in views:
        anchor = views["function"]
    else:
        anchor = views.get("orthology", "")
    anchor = T.strip_terminal_period(anchor)
    if not views:
        tail = f" from {taxonomy}" if taxonomy else ""
        views["unannotated"] = T.render(rng, "unannotated", "", tail=tail)
        anchor = "A protein of unknown function"
    if taxonomy and anchor:
        views["taxonomic"] = T.render(rng, "taxonomic", anchor, taxon=taxonomy)
    if environment and anchor:
        views["environmental"] = T.render(rng, "environmental", anchor, env=environment)
    return views


def compose(views: dict[str, str], rng: random.Random,
            max_parts: int = 3, max_chars: int = 900) -> str:
    """Sample and join a few views to make one caption."""
    main_views = [name for name in views if name not in ("taxonomic", "environmental")]
    if not main_views:
        main_views = list(views)

    number_to_choose = min(len(main_views), rng.randint(1, max_parts))
    chosen = rng.sample(main_views, number_to_choose)

    # Include context sometimes, but not often enough for it to dominate the text.
    for context in ("taxonomic", "environmental"):
        if context in views and rng.random() < 0.25:
            chosen.append(context)
    chosen.sort(key=lambda name: VIEW_PRIORITY.index(name) if name in VIEW_PRIORITY else 99)
    selected_text = [views[name] for name in chosen]
    return " ".join(T.truncate(selected_text, max_chars))


def _fact_hits(row: dict, kind: str) -> list[tuple[str, int, float | None]]:
    """Convert one Parquet fact row to the compact hit tuples used above."""
    return [
        (str(hit.get("accession", "")), int(hit.get("query_start") or 0), hit.get("evalue"))
        for hit in row.get(kind, []) or []
        if hit.get("accession")
    ]


def build_views_from_fact(row: dict, index: dict, rng: random.Random,
                          taxonomy: str = "", environment: str = "") -> dict[str, str]:
    """Build gene views from one row emitted by ``facts.py``."""
    gene = {
        "product": row.get("product", "") or "",
        "source": row.get("product_source", "") or "",
        **{kind: _fact_hits(row, kind) for kind in HIT_FILES},
    }
    return build_views(gene, index, rng, taxonomy=taxonomy, environment=environment)


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _first_sentences(text: str, count: int) -> str:
    sentences = SENTENCE_END.split(text.strip())
    return " ".join(sentences[:count]).strip()


def compose_richness(views: dict[str, str], richness: str) -> str:
    """Deterministically compose medium or high text from supported views.

    The fact set is stable; only wording templates are seeded elsewhere. Medium
    keeps the strongest few non-redundant facts. High includes every supported
    view, while limiting a family paragraph so it cannot hide all other facts.
    """
    if richness not in {"medium", "high"}:
        raise ValueError("richness must be 'medium' or 'high'")
    ordered = [name for name in VIEW_PRIORITY if name in views]
    if richness == "medium":
        main = [name for name in ordered if name not in {"taxonomic", "environmental", "unannotated"}]
        chosen = main[:3]
        if not chosen and "unannotated" in views:
            chosen = ["unannotated"]
        for context in ("taxonomic", "environmental"):
            if context in views and len(chosen) < 4:
                chosen.append(context)
                break
        parts = [_first_sentences(views[name], 1) if name == "family" else views[name]
                 for name in chosen]
        return " ".join(T.truncate(parts, 600))

    parts = []
    for name in ordered:
        text = _first_sentences(views[name], 2) if name == "family" and len(ordered) > 1 else views[name]
        if text and text not in parts:
            parts.append(text)
    return " ".join(T.truncate(parts, 1600))


def stable_rng(*parts: object) -> random.Random:
    """Return the same RNG for the same biological address and build seed."""
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], "little"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--genes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taxonomy", default="a Rhodoferax ferrireducens genome")
    parser.add_argument("--environment", default="a freshwater sediment metagenome")
    args = parser.parse_args()

    index = load_index()
    genes = load_gene_annotations(args.demo)
    rng = random.Random(args.seed)

    def annotation_count(item: tuple[str, dict]) -> int:
        _, gene = item
        return sum(len(gene[kind]) for kind in HIT_FILES)

    ranked = sorted(genes.items(), key=annotation_count, reverse=True)
    shown = 0
    for gene_id, gene in ranked:
        views = build_views(gene, index, rng, args.taxonomy, args.environment)
        if len(views) < 3:
            continue
        print("=" * 96)
        print(f"GENE {gene_id}")
        for name, text in views.items():
            preview = text if len(text) < 200 else text[:197] + "..."
            print(f"  [{name:14s}] {preview}")
        print("  four independently sampled captions:")
        for epoch in range(4):
            print(f"    - {compose(views, rng)[:200]}")
        shown += 1
        if shown >= args.genes:
            break


if __name__ == "__main__":
    main()
