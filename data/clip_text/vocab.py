#!/usr/bin/env python3
"""Build the identifier-to-English vocabulary used by the caption pipeline.

JGI annotations contain identifiers such as ``K02867`` and ``PF00466``. Public
reference files give those identifiers human-readable names and descriptions. This
file parses the references and stores one lookup table in ``vocab.json.gz``.

Identifiers are useful metadata, but captions contain only the English they unlock.
Otherwise the model could memorise identifiers instead of learning biology.

Build:  ``python vocab.py --build``
Lookup: ``python vocab.py --lookup K02867 COG0244 PF00466``
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
INDEX_PATH = HERE / "vocab.json.gz"


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


# Reference descriptions contain citations and identifiers that must not enter captions.
_XREF = re.compile(
    r"\(\s*(?:SP|GB|EMBL|PIR|PDB|GenBank|UniProt|TIGR|PF|EC|PMID|PMC|DOI)\s*:\s*[^)]*\)"
    r"|\[[^\]]*(?:PMID|GOC|ISBN)[^\]]*\]"
    r"|\(\s*[A-Z][A-Z.\s]{3,}\d+\s*:\s*[\d\-]+\s*\(\d{4}\)\s*\)"
    r"|\(\s*(?:see\s+|cf\.?\s+|also\s+)?(?:TIGR\d{4,5}|PF\d{5}|COG\d{4}|K\d{5})"
    r"(?:\s*(?:,|and|;)\s*(?:TIGR\d{4,5}|PF\d{5}|COG\d{4}|K\d{5}))*\s*\)",
    re.IGNORECASE,
)
_BARE_ACC = re.compile(
    r"\b(?:TIGR\d{4,5}|PF\d{5}|COG\d{4}|K\d{5})\b"
    r"|\b(?:PMID|PMC|DOI)\s*:\s*[\w./-]+(?:\s*\(\d{4}\))?",
    re.IGNORECASE,
)
_ORPHAN_PUNCT = re.compile(r"\s+([.,;:])")


def _clean_prose(text: str) -> str:
    """Remove citations and identifiers, then repair leftover punctuation."""
    text = _XREF.sub("", text)
    text = _BARE_ACC.sub("", text)
    text = _ORPHAN_PUNCT.sub(r"\1", text)
    # A product such as ``uncharacterized protein (TIGR01777 family)`` leaves
    # ``( family)`` after its accession is removed.  Remove that empty shell.
    text = re.sub(r"\(\s*(?:family\s*)?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s*\.", ".", text)
    return _clean_text(text)


def load_kegg_ko() -> dict[str, dict]:
    """Read KEGG Orthology names."""
    entries: dict[str, dict] = {}
    path = REF / "kegg_ko.tsv"
    if not path.exists():
        return entries
    for line in path.read_text(errors="ignore").splitlines():
        accession, _, rest = line.partition("\t")
        if not accession or not rest:
            continue
        rest = re.sub(r"\s*\[EC:[^\]]*\]", "", rest).strip()
        symbol, _, name = rest.partition("; ")
        if not name:
            symbol, name = "", rest
        entries[accession.strip()] = {
            "name": _clean_text(name),
            "symbol": _clean_text(symbol),
            "description": "",
        }
    return entries


def load_kegg_enzyme() -> dict[str, dict]:
    """Read enzyme names. The first name is accepted; the rest are synonyms."""
    entries: dict[str, dict] = {}
    path = REF / "kegg_enzyme.tsv"
    if not path.exists():
        return entries
    for line in path.read_text(errors="ignore").splitlines():
        accession, _, rest = line.partition("\t")
        if not accession or not rest:
            continue
        names = [n.strip() for n in rest.split(";") if n.strip()]
        if not names:
            continue
        entries[accession.strip()] = {
            "name": _clean_text(names[0]),
            "synonyms": [_clean_text(name) for name in names[1:6]],
            "description": "",
        }
    return entries


def load_cog() -> tuple[dict[str, dict], dict[str, str]]:
    """Read COG names, pathways, and one-letter functional categories."""
    categories: dict[str, str] = {}
    category_path = REF / "cog-24.fun.tab"
    if category_path.exists():
        for line in category_path.read_text(errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and len(parts[0]) == 1:
                categories[parts[0]] = _clean_text(parts[3])

    entries: dict[str, dict] = {}
    definitions_path = REF / "cog-24.def.tab"
    if definitions_path.exists():
        for line in definitions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            accession = parts[0].strip()
            category_letters = parts[1].strip()
            name = _clean_text(parts[2])
            entries[accession] = {
                "name": name,
                "symbol": _clean_text(parts[3]) if len(parts) > 3 else "",
                "pathway": _clean_text(parts[4]) if len(parts) > 4 else "",
                "categories": [
                    categories[letter]
                    for letter in category_letters
                    if letter in categories
                ],
                "description": "",
            }
    return entries, categories


def load_pfam() -> dict[str, dict]:
    """Read Pfam's repeated AC (accession), ID, and DE (description) records."""
    entries: dict[str, dict] = {}
    path = REF / "Pfam-A.hmm.dat.gz"
    if not path.exists():
        return entries
    accession = identifier = description = ""
    with gzip.open(path, "rt", errors="ignore") as file:
        for line in file:
            if line.startswith("#=GF AC"):
                accession = line.split(maxsplit=2)[-1].strip().split(".")[0]
            elif line.startswith("#=GF ID"):
                identifier = line.split(maxsplit=2)[-1].strip()
            elif line.startswith("#=GF DE"):
                description = line.split(maxsplit=2)[-1].strip()
            elif line.startswith("//"):
                if accession:
                    entries[accession] = {
                        "name": _clean_text(description) or identifier,
                        "symbol": identifier,
                        "description": "",
                    }
                accession = identifier = description = ""
    return entries


def load_pgap() -> dict[str, dict]:
    """Read TIGRFAM/NCBIFAM names and descriptive paragraphs from NCBI PGAP."""
    entries: dict[str, dict] = {}
    path = REF / "hmm_PGAP.tsv"
    if not path.exists():
        return entries
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return entries
    header = [h.lstrip("#") for h in lines[0].split("\t")]
    columns = {name: position for position, name in enumerate(header)}
    # The prose column has changed names between releases, so find long text near the end.
    for line in lines[1:]:
        parts = line.split("\t")
        accession_column = columns.get("source_identifier", 1)
        accession = parts[accession_column].strip() if len(parts) > 1 else ""
        if not accession:
            continue
        product_column = columns.get("product_name", 99)
        symbol_column = columns.get("gene_symbol", 99)
        product = parts[product_column].strip() if product_column < len(parts) else ""
        symbol = parts[symbol_column].strip() if symbol_column < len(parts) else ""
        prose_candidates = [
            value.strip()
            for value in parts[20:]
            if len(value.strip()) > 60 and " " in value
        ]
        entries[accession] = {
            "name": _clean_text(product),
            "symbol": _clean_text(symbol),
            "description": _clean_prose(max(prose_candidates, key=len)) if prose_candidates else "",
        }
    return entries


def load_go() -> dict[str, dict]:
    """Read active Gene Ontology terms and their definitions."""
    entries: dict[str, dict] = {}
    path = REF / "go-basic.obo"
    if not path.exists():
        return entries
    current: dict[str, str] = {}
    for raw in path.read_text(errors="ignore").splitlines() + ["[Term]"]:
        line = raw.strip()
        if line in ("[Term]", "[Typedef]"):
            if current.get("id") and current.get("name") and current.get("is_obsolete") != "true":
                definition = current.get("def", "")
                definition_match = re.match(r'"(.*?)"(?:\s*\[|$)', definition)
                entries[current["id"]] = {
                    "name": _clean_text(current["name"]),
                    "namespace": current.get("namespace", ""),
                    "description": _clean_prose(definition_match.group(1)) if definition_match else "",
                }
            current = {}
        elif ": " in line:
            key, _, value = line.partition(": ")
            if key in ("id", "name", "namespace", "def", "is_obsolete"):
                current.setdefault(key, value)
    return entries


def build() -> dict:
    """Parse every reference file and return the complete vocabulary."""
    index = {
        "ko": load_kegg_ko(),
        "ec": load_kegg_enzyme(),
        "pfam": load_pfam(),
        "tigr": load_pgap(),
        "go": load_go(),
    }
    cog, cog_categories = load_cog()
    index["cog"] = cog
    index["cog_categories"] = cog_categories
    return index


def describe(index: dict, kind: str, acc: str) -> dict | None:
    """Look up one identifier, returning ``None`` when it is unknown."""
    accession = acc
    table = index.get(kind if kind != "product" else "ko", {})
    entry = table.get(accession)
    if entry is None and kind == "pfam" and not accession.startswith("PF"):
        entry = table.get("PF" + accession.lstrip("pfamPFAM"))
    return entry


def guess_kind(accession: str) -> str:
    """Choose the vocabulary table that normally contains an identifier."""
    if accession.startswith("GO:"):
        return "go"
    if re.fullmatch(r"K\d{5}", accession):
        return "ko"
    if accession.startswith("COG"):
        return "cog"
    if accession.upper().startswith("PF"):
        return "pfam"
    if accession.startswith("TIGR"):
        return "tigr"
    return "ec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--lookup", nargs="*", help="identifiers to resolve, any vocabulary")
    args = parser.parse_args()

    if args.build or not INDEX_PATH.exists():
        index = build()
        with gzip.open(INDEX_PATH, "wt") as fh:
            json.dump(index, fh)
        for kind, table in index.items():
            print(f"  {kind:16s} {len(table):>7,} entries")
        description_counts = {
            kind: sum(1 for entry in table.values() if entry.get("description"))
            for kind, table in index.items()
            if kind != "cog_categories"
        }
        print("  entries carrying a prose description:", description_counts)
        print(f"-> {INDEX_PATH} ({INDEX_PATH.stat().st_size/1e6:.1f} MB)")
    else:
        with gzip.open(INDEX_PATH, "rt") as fh:
            index = json.load(fh)

    for accession in args.lookup or []:
        kind = guess_kind(accession)
        entry = describe(index, kind, accession)
        print(f"\n[{kind}] {accession}")
        if entry:
            for key, value in entry.items():
                if value:
                    print(f"    {key}: {value}")
        else:
            print("    (not found)")


if __name__ == "__main__":
    main()
