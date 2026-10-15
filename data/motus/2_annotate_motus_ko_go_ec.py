from __future__ import annotations

import csv
import gzip
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional


def resolve_tool(tool_name: str) -> str:
    if "/" in tool_name:
        if not Path(tool_name).exists():
            raise FileNotFoundError(f"Tool not found: {tool_name}")
        return tool_name
    resolved = shutil.which(tool_name)
    if not resolved:
        raise FileNotFoundError(f"Tool not found in PATH: {tool_name}")
    return resolved


def run_command(command: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def decompress_fasta_gz(in_path: Path, out_path: Path) -> None:
    with gzip.open(in_path, "rb") as src:
        with open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def find_header(line: str) -> List[str]:
    # eggnog annotation header line starts with #query
    return line.strip().lstrip("#").split("\t")


def normalize_value(value: str) -> str:
    text = (value or "").strip()
    if text in {"-", ""}:
        return ""
    return text


def parse_eggnog_ko_go_ec(annotations_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header: List[str] = []

    with open(annotations_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#query"):
                header = find_header(line)
                continue
            if line.startswith("#"):
                continue
            if not header:
                continue

            values = line.rstrip("\n").split("\t")
            if len(values) < len(header):
                values += [""] * (len(header) - len(values))
            record = dict(zip(header, values))

            ko = normalize_value(record.get("KEGG_ko", ""))
            go = normalize_value(record.get("GOs", ""))
            ec = normalize_value(record.get("EC", ""))
            if not (ko or go or ec):
                continue

            rows.append(
                {
                    "gene_id": record.get("query", ""),
                    "KO": ko,
                    "GO": go,
                    "EC": ec,
                }
            )
    return rows


def resolve_eggnog_data_dir(project_dir: Path) -> Path:
    env_path = os.environ.get("EGGNOG_DATA_DIR", "").strip()
    if env_path:
        return Path(env_path).expanduser()

    local_default = project_dir / "outputs" / "eggnog_db"
    if local_default.exists():
        return local_default

    raise FileNotFoundError(f"EggNOG data dir not found: {project_dir / 'outputs' / 'eggnog_db'}")


def validate_eggnog_data_dir(eggnog_data_dir: Path) -> None:
    if not eggnog_data_dir.exists():
        raise FileNotFoundError(f"EggNOG data dir not found: {eggnog_data_dir}")

    required_files = ["eggnog.db", "eggnog.taxa.db", "eggnog_proteins.dmnd"]
    missing_files = [name for name in required_files if not (eggnog_data_dir / name).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            "EggNOG data dir is missing required files "
            f"({missing}) in: {eggnog_data_dir}. "
            f"Run: download_eggnog_data.py -y --data_dir {eggnog_data_dir}"
        )


def write_tsv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def annotate_mag(
    fasta_gz: Path,
    annotations_root: Path,
    prodigal_bin: str,
    emapper_bin: str,
    eggnog_data_dir: Path,
    emapper_cpu_per_mag: int,
) -> Dict[str, str]:
    mag_name = fasta_gz.name.replace(".fa.gz", "")
    mag_dir = annotations_root / "per_mag" / mag_name
    done_flag = mag_dir / "_ANNOTATION_DONE"
    out_tsv = mag_dir / "ko_go_ec.tsv"

    if done_flag.exists() and out_tsv.exists():
        return {"mag": mag_name, "status": "skipped_existing", "error": ""}

    mag_dir.mkdir(parents=True, exist_ok=True)
    input_fa = mag_dir / f"{mag_name}.fa"
    proteins_faa = mag_dir / f"{mag_name}_genes.faa"
    genes_fna = mag_dir / f"{mag_name}_genes.fna"
    prodigal_gbk = mag_dir / f"{mag_name}_prodigal.gbk"
    emapper_prefix = f"{mag_name}_emapper"
    annotations_path = mag_dir / f"{emapper_prefix}.emapper.annotations"

    try:
        decompress_fasta_gz(fasta_gz, input_fa)

        run_command(
            [
                prodigal_bin,
                "-i",
                str(input_fa),
                "-a",
                str(proteins_faa),
                "-d",
                str(genes_fna),
                "-o",
                str(prodigal_gbk),
                "-p",
                "meta",
            ]
        )

        run_command(
            [
                emapper_bin,
                "-i",
                str(proteins_faa),
                "--output",
                emapper_prefix,
                "--output_dir",
                str(mag_dir),
                "--cpu",
                str(emapper_cpu_per_mag),
                "--data_dir",
                str(eggnog_data_dir),
            ]
        )

        ko_go_ec_rows = parse_eggnog_ko_go_ec(annotations_path)
        write_tsv(out_tsv, ko_go_ec_rows, ["gene_id", "KO", "GO", "EC"])
        done_flag.write_text("done\n", encoding="utf-8")

        if input_fa.exists():
            input_fa.unlink()
        return {"mag": mag_name, "status": "annotated", "error": ""}
    except Exception as exc:
        return {"mag": mag_name, "status": "failed", "error": str(exc)}


def rebuild_merged_table(annotations_root: Path) -> None:
    merged_path = annotations_root / "merged_ko_go_ec.tsv"
    per_mag_dir = annotations_root / "per_mag"
    all_rows: List[Dict[str, str]] = []

    if per_mag_dir.exists():
        for mag_dir in sorted(per_mag_dir.iterdir()):
            if not mag_dir.is_dir():
                continue
            mag = mag_dir.name
            per_mag_file = mag_dir / "ko_go_ec.tsv"
            if not per_mag_file.exists():
                continue

            with open(per_mag_file, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    all_rows.append(
                        {
                            "mag": mag,
                            "gene_id": row.get("gene_id", ""),
                            "KO": row.get("KO", ""),
                            "GO": row.get("GO", ""),
                            "EC": row.get("EC", ""),
                        }
                    )

    write_tsv(merged_path, all_rows, ["mag", "gene_id", "KO", "GO", "EC"])


def append_failures(failures_path: Path, failures: List[Dict[str, str]]) -> None:
    if not failures:
        return
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failures_path.exists()
    with open(failures_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mag", "status", "error"], delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    total_workers = env_int("TOTAL_WORKERS", os.cpu_count() or 1)
    download_workers = env_int("DOWNLOAD_WORKERS", max(1, total_workers // 4))
    emapper_cpu_per_mag = env_int("EMAPPER_CPU_PER_MAG", 2)
    annotation_cpu_budget = max(1, total_workers - download_workers)
    default_annotate_workers = max(1, annotation_cpu_budget // emapper_cpu_per_mag)
    annotate_workers = env_int("ANNOTATE_WORKERS", default_annotate_workers)
    poll_seconds = 10
    idle_stop_seconds = 120

    prodigal_bin = "prodigal"
    emapper_bin = "emapper.py"

    project_dir = Path(__file__).resolve().parent
    eggnog_data_dir = resolve_eggnog_data_dir(project_dir)
    outputs_dir = project_dir / "outputs"
    downloads_dir = outputs_dir / "downloads"
    completion_flag = downloads_dir / "_DOWNLOAD_COMPLETE"
    annotations_root = outputs_dir / "annotations"
    failures_path = annotations_root / "annotation_failures.tsv"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    print(
        "Worker config | "
        f"total_workers={total_workers} "
        f"download_workers={download_workers} "
        f"annotation_cpu_budget={annotation_cpu_budget} "
        f"annotate_workers={annotate_workers} "
        f"emapper_cpu_per_mag={emapper_cpu_per_mag}"
    )

    prodigal_bin = resolve_tool(prodigal_bin)
    emapper_bin = resolve_tool(emapper_bin)
    validate_eggnog_data_dir(eggnog_data_dir)

    print("Watching downloads folder and annotating MAGs as they arrive ...")
    seen = set()
    in_progress = {}
    last_activity_time = time.time()

    with ThreadPoolExecutor(max_workers=annotate_workers) as executor:
        while True:
            all_fasta_gz = sorted(downloads_dir.rglob("*.fa.gz"))
            pending_paths = []
            for fasta_gz in all_fasta_gz:
                mag_name = fasta_gz.name.replace(".fa.gz", "")
                done_flag = annotations_root / "per_mag" / mag_name / "_ANNOTATION_DONE"
                if done_flag.exists():
                    seen.add(str(fasta_gz))
                    continue
                if str(fasta_gz) not in seen and str(fasta_gz) not in in_progress.values():
                    pending_paths.append(fasta_gz)

            for fasta_gz in pending_paths:
                future = executor.submit(
                    annotate_mag,
                    fasta_gz=fasta_gz,
                    annotations_root=annotations_root,
                    prodigal_bin=prodigal_bin,
                    emapper_bin=emapper_bin,
                    eggnog_data_dir=eggnog_data_dir,
                    emapper_cpu_per_mag=emapper_cpu_per_mag,
                )
                in_progress[future] = str(fasta_gz)
                last_activity_time = time.time()
                print(f"queued annotation: {fasta_gz.name}")

            finished = []
            failures = []
            for future, source_path in list(in_progress.items()):
                if future.done():
                    finished.append((future, source_path))

            for future, source_path in finished:
                result = future.result()
                in_progress.pop(future, None)
                seen.add(source_path)
                if result["status"] == "failed":
                    failures.append(result)
                print(f"{result['status']:>16} | {result['mag']}")
                last_activity_time = time.time()

            if failures:
                append_failures(failures_path, failures)
            if finished:
                rebuild_merged_table(annotations_root)

            nothing_running = not in_progress and not pending_paths
            done_downloading = completion_flag.exists()
            idle_time = time.time() - last_activity_time
            if nothing_running and done_downloading and idle_time >= idle_stop_seconds:
                break

            time.sleep(poll_seconds)

    rebuild_merged_table(annotations_root)
    print("Annotation watcher finished.")


if __name__ == "__main__":
    main()
