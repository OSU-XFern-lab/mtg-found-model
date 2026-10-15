"""Plan, run, and inspect restartable CPU annotation batches."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
spec = importlib.util.spec_from_file_location("annotator", ROOT / "2_annotate_motus_ko_go_ec.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def save_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def complete(directory):
    return (directory / "_ANNOTATION_DONE").exists() and (directory / "ko_go_ec.tsv").exists()


def plan(args):
    if args.plan.exists():
        raise SystemExit("Choose a new plan path; existing plans are immutable.")
    done = {p.name for p in (OUTPUTS / "annotations/per_mag").iterdir() if complete(p)}
    with (OUTPUTS / "manifests/mag_manifest.tsv").open() as handle:
        metadata = {r["genome"]: r for r in csv.DictReader(handle, delimiter="\t")}
    pending = {}
    for folder in os.scandir(OUTPUTS / "downloads"):
        if not folder.is_dir():
            continue
        for entry in os.scandir(folder.path):
            if not entry.name.endswith(".fa.gz"):
                continue
            mag = entry.name[:-6]
            if mag in done:
                continue
            if mag in pending:
                raise ValueError(f"Duplicate raw MAG ID: {mag}")
            meta = metadata.get(mag, {})
            pending[mag] = {"mag": mag, "path": entry.path,
                            "bases": int(float(meta.get("genome_size") or 0))}
    rows = sorted(pending.values(), key=lambda r: (r["bases"], r["mag"]))
    if args.pilot:
        count = min(args.tasks * 3, len(rows))
        rows = [rows[min(len(rows)-1, int((i + 0.5) * len(rows) / count))] for i in range(count)]
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "mag", "path", "bases"], delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"task": i % args.tasks, **row})
    save_json(args.plan.with_suffix(".json"), {
        "created": time.time(), "pending_total": len(pending), "selected": len(rows),
        "tasks": args.tasks, "pilot": args.pilot,
        "sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
    })
    print(args.plan.with_suffix(".json").read_text())


def annotate(row, args, reports):
    mag = row["mag"]
    directory = OUTPUTS / "annotations/per_mag" / mag
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".annotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(directory):
            return {"mag": mag, "status": "existing"}
        if shutil.disk_usage(OUTPUTS).free < args.min_free_gb * 1024**3:
            raise RuntimeError("Free-space safety floor reached; stopping this worker.")
        start = time.time()
        record = {"mag": mag, "task": args.task, "cpus": args.cpus,
                  "job": os.environ.get("SLURM_JOB_ID"), "started": start,
                  "status": "running", "bases": int(row["bases"])}
        report = reports / f"{mag}.json"
        save_json(report, record)
        proteins = directory / f"{mag}_genes.faa"
        try:
            with (directory / "annotation.log").open("a") as log:
                def run(command):
                    print(" ".join(map(str, command)), file=log, flush=True)
                    subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)

                if not (directory / "_PRODIGAL_DONE").exists():
                    fasta = directory / f"{mag}.fa"
                    legacy.decompress_fasta_gz(Path(row["path"]), fasta)
                    temporary = directory / "genes.faa.partial"
                    run(["prodigal", "-i", fasta, "-a", temporary, "-o", os.devnull, "-p", "meta"])
                    temporary.replace(proteins)
                    (directory / "_PRODIGAL_DONE").write_text("done\n")
                    fasta.unlink()
                with proteins.open() as handle:
                    genes = sum(line.startswith(">") for line in handle)
                if not genes:
                    raise ValueError("Prodigal predicted no genes")
                prefix = f"{mag}_emapper"
                run(["emapper.py", "-i", proteins, "--output", prefix,
                     "--output_dir", directory, "--cpu", args.cpus,
                     "--data_dir", args.database, "--override"])
                annotations = directory / f"{prefix}.emapper.annotations"
                with annotations.open() as handle:
                    if not any(line.startswith("#query") for line in handle):
                        raise ValueError("Missing EggNOG annotation header")
                rows = legacy.parse_eggnog_ko_go_ec(annotations)
                temporary = directory / "ko_go_ec.tsv.partial"
                legacy.write_tsv(temporary, rows, ["gene_id", "KO", "GO", "EC"])
                temporary.replace(directory / "ko_go_ec.tsv")
                record.update(status="complete", genes=genes, genes_with_ko_go_ec=len(rows),
                              seconds=time.time()-start,
                              output_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()))
                save_json(directory / "provenance.json", {
                    "source_name": "MOTUS", "mag_id": mag, "raw_path": row["path"],
                    "raw_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest(),
                    "database": str(args.database), "plan": str(args.plan),
                    "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                    "eggnog_mapper": "2.1.13", "prodigal_mode": "meta",
                    "cpus": args.cpus, "completed": time.time(),
                })
                save_json(report, record)
                (directory / "_ANNOTATION_DONE").write_text("done\n")
        except Exception as error:
            record.update(status="failed", error=str(error), seconds=time.time()-start)
            save_json(report, record)
        return record


def worker(args):
    metadata = json.loads(args.plan.with_suffix(".json").read_text())
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Plan checksum mismatch")
    if args.cpus > int(os.environ.get("SLURM_CPUS_PER_TASK", args.cpus)):
        raise ValueError("Requested threads exceed Slurm allocation")
    legacy.validate_eggnog_data_dir(args.database)
    for tool in ["prodigal", "emapper.py", "diamond"]:
        legacy.resolve_tool(tool)
    reports = args.plan.parent / (args.plan.stem + "_reports")
    reports.mkdir(exist_ok=True)
    with args.plan.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if int(r["task"]) == args.task]
    if not rows:
        raise ValueError("No MAGs assigned to this task")
    failed = 0
    for row in rows:
        result = annotate(row, args, reports)
        print(json.dumps(result), flush=True)
        failed += result["status"] == "failed"
    if failed:
        raise SystemExit(f"{failed} MAGs failed; inspect reports and rerun the same task")


def status(args):
    metadata = json.loads(args.plan.with_suffix(".json").read_text())
    reports = args.plan.parent / (args.plan.stem + "_reports")
    records = [json.loads(p.read_text()) for p in reports.glob("*.json")]
    counts = {s: sum(r["status"] == s for r in records) for s in ["running", "complete", "failed"]}
    print(json.dumps({**metadata, "reports": counts}, indent=2))
    for cpus in sorted({r["cpus"] for r in records}):
        finished = [r for r in records if r["cpus"] == cpus and r["status"] == "complete"]
        if finished:
            seconds = sum(r["seconds"] for r in finished)
            print(json.dumps({"cpus": cpus, "completed": len(finished),
                              "mean_seconds": seconds / len(finished),
                              "genes": sum(r["genes"] for r in finished),
                              "output_bytes": sum(r["output_bytes"] for r in finished)}))
    print("Running reports may be stale after preemption; check squeue as well.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "worker", "status"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=9)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--database", type=Path, default=OUTPUTS / "eggnog_db")
    parser.add_argument("--min-free-gb", type=float, default=100)
    args = parser.parse_args()
    if args.tasks < 1 or args.cpus < 1:
        parser.error("tasks and cpus must be positive")
    args.plan = args.plan.resolve()
    {"plan": plan, "worker": worker, "status": status}[args.action](args)
