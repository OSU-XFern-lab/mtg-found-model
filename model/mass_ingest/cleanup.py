"""Reclaim reproducible ingestion staging only after publication is proven."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from data_sources import source_paths
from mag_shards.manifest import load_manifest
from mass_ingest.common import atomic_json, load_plan, require_path


def source_counts(root: Path, plan: dict) -> dict[str, dict[str, int]]:
    result = {name: {"mags": 0, "genes": 0} for name in plan["sources"]}
    for entry in plan["build_tasks"]:
        report = json.loads((root / "build_reports" / f"task-{entry['task']:04d}.json").read_text())
        result[entry["source"]]["mags"] += int(report["n_mags"])
        result[entry["source"]]["genes"] += int(report["n_genes"])
    return result


def verify_publication() -> dict:
    root = require_path("MASS_ROOT")
    parent_path = require_path("PARENT_MANIFEST")
    plan = load_plan(root)
    parent = load_manifest(parent_path)
    publication = json.loads((root / "publication_report.json").read_text())
    expected = source_counts(root, plan)

    for label, value in publication.items():
        manifest = load_manifest(Path(value["manifest"]))
        manifest.validate()
        if manifest.sha256 != value["sha256"]:
            raise RuntimeError(f"{label}: published manifest digest changed")
        if manifest.parent_sha256 != parent.sha256:
            raise RuntimeError(f"{label}: published manifest has the wrong parent")
        for ref in manifest.shards:
            shard = manifest.resolve_shard(ref)
            if not (shard / "COMPLETE").is_file() or not (shard / "shard.json").is_file():
                raise RuntimeError(f"{label}: incomplete referenced shard {shard}")
        if label == "combined":
            added_mags = sum(value["mags"] for value in expected.values())
            added_genes = sum(value["genes"] for value in expected.values())
        else:
            added_mags = expected[label]["mags"]
            added_genes = expected[label]["genes"]
        if manifest.totals["n_mags"] - parent.totals["n_mags"] != added_mags:
            raise RuntimeError(f"{label}: published MAG delta is wrong")
        if manifest.totals["n_genes"] - parent.totals["n_genes"] != added_genes:
            raise RuntimeError(f"{label}: published gene delta is wrong")

    prepared = {}
    for source, counts in expected.items():
        per_mag, manifest_path = source_paths(source)
        report_path = manifest_path.parent / "preparation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("errors"):
                raise RuntimeError(f"{source}: preparation has errors; refusing to remove per-MAG inputs")
            if int(report["mags_processed"]) != counts["mags"] or int(report["genes"]) != counts["genes"]:
                raise RuntimeError(f"{source}: preparation and published counts disagree")
        prepared[source] = {
            "per_mag": str(per_mag),
            "manifest": str(manifest_path),
            **counts,
        }

    return {
        "version": 1,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "publication": publication,
        "prepared_sources": prepared,
        "ingestion_counts": plan["counts"],
    }


def allowed_targets(root: Path, intent: dict) -> list[Path]:
    targets = [root / "proteins", root / "embeddings", root / "vq"]
    targets.extend(Path(value["per_mag"]) for value in intent["prepared_sources"].values())
    for path in targets:
        resolved = path.resolve()
        if path.is_symlink() or resolved == Path("/") or len(resolved.parts) < 6:
            raise RuntimeError(f"unsafe cleanup target: {path}")
        if path.name not in {"proteins", "embeddings", "vq", "per_mag"}:
            raise RuntimeError(f"unexpected cleanup target: {path}")
    return targets


def main() -> None:
    root = require_path("MASS_ROOT")
    intent_path = root / "cleanup_intent.json"
    if os.environ.get("APPLY_CLEANUP") != "1":
        raise RuntimeError("set APPLY_CLEANUP=1 after publication")
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        # A resumed cleanup still proves that published manifests have not
        # changed, while relying on the durable pre-deletion audit for staging.
        for label, value in intent["publication"].items():
            if load_manifest(Path(value["manifest"])).sha256 != value["sha256"]:
                raise RuntimeError(f"{label}: published manifest changed during cleanup")
    else:
        intent = verify_publication()
        atomic_json(intent_path, intent)

    removed = []
    for path in allowed_targets(root, intent):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    result = {
        **intent,
        "cleanup_completed_utc": datetime.now(timezone.utc).isoformat(),
        "removed": removed,
        "retained": {
            "raw_downloads": True,
            "source_manifests_and_reports": True,
            "final_mag_shards": True,
            "mass_plan_build_and_publication_reports": True,
        },
    }
    atomic_json(root / "cleanup_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
