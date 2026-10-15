#!/usr/bin/env python3
"""Build deterministic multi-MAG language examples from compact MAG summaries.

Run this once after every per-archive rendering job is complete.  MAGs are
paired only within the same train/validation split, and each MAG is used at
most once.  This keeps the release useful without creating a quadratic number
of comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import render as R


def context(rows: list[dict]) -> str:
    mags = [
        {"mag_id": row["mag_id"], "include_mag": True, "contigs": []}
        for row in rows
    ]
    return json.dumps({"mags": mags}, separators=(",", ":"), sort_keys=True)


def _identity(row: dict, position: str) -> str:
    lineage = "; ".join(row.get("taxonomy", []) or [])
    environment = row.get("environment", "") or ""
    phrase = f"Genome {position}"
    if lineage:
        phrase += " is classified as " + lineage
    if environment:
        phrase += " and was recovered from " + environment
    phrase += (
        f". It contains {int(row['n_genes']):,} genes across "
        f"{int(row['n_contigs']):,} contigs, with {int(row['informative_genes']):,} "
        "genes carrying functional or domain annotations"
    )
    return R.sentence(phrase)


def answers(rows: list[dict]) -> tuple[str, str]:
    ordinals = ("first", "second", "third", "fourth")
    medium = [_identity(row, ordinals[index]) for index, row in enumerate(rows)]
    category_sets = [set(row.get("categories", []) or []) for row in rows]
    shared = set.intersection(*category_sets) if category_sets else set()
    if shared:
        medium.append(R.sentence(
            "Shared broad functional evidence includes " + ", ".join(sorted(shared)[:6])))

    high = list(medium)
    for index, row in enumerate(rows):
        distinctive = [name for name in row.get("categories", []) or [] if name not in shared]
        if distinctive:
            high.append(R.sentence(
                f"Evidence more prominent in the {ordinals[index]} MAG includes " +
                ", ".join(distinctive[:5])))
        pathways = row.get("pathways", []) or []
        if pathways:
            high.append(R.sentence(
                f"Named pathway or system context for the {ordinals[index]} MAG includes " +
                ", ".join(pathways[:5])))
    high.append(
        "This comparison concerns encoded potential from computational annotations; it does not demonstrate phenotype or ecological interaction."
    )
    return " ".join(medium), " ".join(high)


def pair_key(row: dict) -> bytes:
    return hashlib.sha256(f"multi-mag-v1\x1f{row['mag_id']}".encode()).digest()


def build(args: argparse.Namespace) -> None:
    final = args.output.resolve()
    if (final / "COMPLETE").is_file():
        print(json.dumps({"state": "already_complete", "path": str(final)}))
        return
    if final.exists():
        raise RuntimeError(f"refusing to overwrite incomplete output: {final}")
    paths = sorted(args.summary_root.glob("*/*/mag_summaries.parquet"))
    if not paths:
        raise RuntimeError(f"no MAG summaries found below {args.summary_root}")

    by_split: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            by_split[row["split"]].append(row)

    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    writer = R.BufferedWriter(stage / "multi_mag_examples.parquet")
    split_counts = {}
    try:
        for split, split_rows in sorted(by_split.items()):
            split_rows.sort(key=pair_key)
            count = 0
            for offset in range(0, len(split_rows) - 1, 2):
                rows = split_rows[offset:offset + 2]
                medium, high = answers(rows)
                identifier = R.example_id("multi_mag", *(row["mag_id"] for row in rows))
                writer.add({
                    "source_name": "MULTI_SOURCE",
                    "split": split,
                    "level": "multi_mag",
                    "example_id": identifier,
                    "context_json": context(rows),
                    **R.prompt_fields("multi_mag", identifier),
                    "answer_medium": medium,
                    "answer_high": high,
                    "information_types": sorted({
                        kind for row in rows for kind in (row.get("information_types", []) or [])
                    }),
                    "fact_part": "cross_mag",
                })
                count += 1
                if args.max_examples_per_split and count >= args.max_examples_per_split:
                    break
            split_counts[split] = count
        writer.close()
        report = {
            "schema_name": R.SCHEMA_NAME,
            "schema_version": R.SCHEMA_VERSION,
            "summary_parts": len(paths),
            "input_mags": {split: len(rows) for split, rows in sorted(by_split.items())},
            "examples": split_counts,
        }
        (stage / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (stage / "COMPLETE").write_text("")
        os.replace(stage, final)
        print(json.dumps({**report, "path": str(final)}, sort_keys=True))
    except BaseException:
        try:
            writer.close()
        except BaseException:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples-per-split", type=int)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
