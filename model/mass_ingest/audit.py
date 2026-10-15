"""Fail closed at every handoff between mass-ingestion stages."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mass_ingest.common import load_plan, require_path


def reports(root: Path, pattern: str) -> list[dict]:
    paths = sorted(root.glob(pattern))
    return [json.loads(path.read_text()) for path in paths]


def require_equal(label: str, observed: int, expected: int) -> None:
    if observed != expected:
        raise RuntimeError(f"{label}: observed {observed}, expected {expected}")


def audit_collect() -> dict:
    root = require_path("MASS_ROOT"); plan = load_plan(root)
    values = reports(root, "collect/task-*/report.json")
    require_equal("collection reports", len(values), len(plan["collect_tasks"]))
    require_equal("collected genes", sum(value["genes"] for value in values), plan["counts"]["genes"])
    return {"stage": "collect", "tasks": len(values), "genes": sum(x["genes"] for x in values)}


def audit_dedup() -> dict:
    root = require_path("MASS_ROOT"); plan = load_plan(root)
    values = reports(root, "proteins/bucket-*/report.json")
    require_equal("dedup reports", len(values), plan["bucket_count"])
    require_equal("dedup occurrences", sum(value["occurrences"] for value in values), plan["counts"]["genes"])
    return {"stage": "dedup", "buckets": len(values),
            "occurrences": sum(x["occurrences"] for x in values),
            "unique_proteins": sum(x["unique_proteins"] for x in values)}


def audit_embed() -> dict:
    root = require_path("MASS_ROOT"); plan = load_plan(root)
    proteins = reports(root, "proteins/bucket-*/report.json")
    values = reports(root, "embeddings/bucket-*/report.json")
    require_equal("embedding reports", len(values), plan["bucket_count"])
    require_equal("embedded proteins", sum(value["proteins"] for value in values),
                  sum(value["unique_proteins"] for value in proteins))
    return {"stage": "embed", "buckets": len(values), "proteins": sum(x["proteins"] for x in values)}


def audit_vq() -> dict:
    root = require_path("MASS_ROOT"); plan = load_plan(root)
    embeddings = reports(root, "embeddings/bucket-*/report.json")
    values = reports(root, "vq/bucket-*/report.json")
    require_equal("VQ reports", len(values), plan["bucket_count"])
    require_equal("VQ proteins", sum(value["proteins"] for value in values),
                  sum(value["proteins"] for value in embeddings))
    return {"stage": "vq", "buckets": len(values), "proteins": sum(x["proteins"] for x in values)}


def audit_build() -> dict:
    root = require_path("MASS_ROOT"); plan = load_plan(root)
    values = reports(root, "build_reports/task-*.json")
    require_equal("build reports", len(values), len(plan["build_tasks"]))
    require_equal("built MAGs", sum(value["n_mags"] for value in values), plan["counts"]["mags"])
    require_equal("built genes", sum(value["n_genes"] for value in values), plan["counts"]["genes"])
    return {"stage": "build", "shards": len(values), "mags": sum(x["n_mags"] for x in values),
            "genes": sum(x["n_genes"] for x in values)}


AUDITS = {"collect": audit_collect, "dedup": audit_dedup, "embed": audit_embed,
          "vq": audit_vq, "build": audit_build}


def main(stage: str | None = None) -> None:
    stage = stage or os.environ.get("AUDIT_STAGE", "")
    if stage not in AUDITS:
        raise RuntimeError("audit stage must be collect, dedup, embed, vq, or build")
    print(json.dumps(AUDITS[stage](), indent=2))


if __name__ == "__main__":
    main()
