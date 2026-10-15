from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests


API_BASE_URLS = [
    "https://motus-db.org",
]

MOTUS_TABLE_PATH = "/api/search-table-cols"
MAGS_TABLE_PATH = "/api/genome-cols"
MOTUS_DOWNLOAD_BASE = "https://sunagawalab.ethz.ch/share/MOTUS/database/4.0/data/genomes"


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def post_json(path: str, payload: Dict, timeout_seconds: int, max_retries: int) -> Dict:
    last_error: Exception | None = None
    for base_url in API_BASE_URLS:
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    f"{base_url}{path}",
                    json=payload,
                    timeout=(15, timeout_seconds),
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(min(2 * attempt, 10))

    raise RuntimeError(
        "Could not reach mOTUs API after retries. "
        "Check network/DNS on this node and try from a node with internet egress."
    ) from last_error


def fetch_motus(max_motus: int, timeout_seconds: int, api_retries: int) -> List[Dict]:
    selected: List[Dict] = []
    seen = set()
    start = 0
    page_size = 200

    while len(selected) < max_motus:
        payload = {
            "repo_id": "motus",
            "global_search_value": "",
            "search_value_pair": [],
            "start": start,
            "length": page_size,
            "window": 0,
            "fetchall": 0,
            "order": [{"column": 1, "dir": "asc"}],
        }
        data = post_json(MOTUS_TABLE_PATH, payload, timeout_seconds, api_retries).get("data", [])
        if not data:
            break

        added_this_page = 0
        for row in data:
            motu = str(row.get("motu", "")).strip()
            if not motu or motu in seen:
                continue
            seen.add(motu)
            selected.append(row)
            added_this_page += 1
            if len(selected) >= max_motus:
                break

        if added_this_page == 0:
            break
        print(f"Fetched mOTUs so far: {len(selected)}")
        start += page_size

    return selected[:max_motus]


def fetch_mag_rows_for_motu(motu: str, timeout_seconds: int, api_retries: int) -> List[Dict]:
    payload = {
        "repo_id": "motus",
        "global_search_value": "",
        "search_value_pair": [{"key": "motu", "value": motu}],
        "start": 0,
        "length": 50,
        "window": 0,
        "fetchall": 1,
        "order": [{"column": 1, "dir": "asc"}],
        "sources": [],
        "range_filters": [],
    }
    return post_json(MAGS_TABLE_PATH, payload, timeout_seconds, api_retries).get("data", [])


def build_download_jobs(
    motus: List[Dict],
    max_mags_per_motu: int,
    timeout_seconds: int,
    api_retries: int,
    collect_workers: int,
) -> Tuple[List[Dict], List[Dict]]:
    def collect_jobs_for_motu(index_and_row: Tuple[int, Dict]) -> Tuple[int, str, List[Dict], str]:
        index, motu_row = index_and_row
        motu = str(motu_row.get("motu", "")).strip()
        if not motu:
            return index, "", [], ""

        print(f"Collecting MAGs for {motu} ...")
        try:
            rows = fetch_mag_rows_for_motu(motu, timeout_seconds, api_retries)
        except Exception as exc:
            return index, motu, [], str(exc)
        motu_jobs: List[Dict] = []
        picked = 0
        for row in rows:
            if picked >= max_mags_per_motu:
                break
            if str(row.get("mag", "")).upper() != "TRUE":
                continue

            location = str(row.get("location", "")).strip().lstrip("/")
            if not location.endswith(".fa.gz"):
                continue

            genome = str(row.get("genome", "")).strip() or Path(location).stem
            download_url = f"{MOTUS_DOWNLOAD_BASE}/{location}"
            out_dir = Path(__file__).resolve().parent / "outputs" / "downloads" / motu
            out_path = out_dir / f"{genome}.fa.gz"

            motu_jobs.append(
                {
                    "motu": motu,
                    "domain": str(row.get("domain", "")),
                    "phylum": str(row.get("phylum", "")),
                    "class_": str(row.get("class_", "")),
                    "order": str(row.get("order", "")),
                    "family": str(row.get("family", "")),
                    "genus": str(row.get("genus", "")),
                    "species": str(row.get("species", "")),
                    "genome": genome,
                    "study": str(row.get("study", "")),
                    "qscore": str(row.get("qscore", "")),
                    "completeness": str(row.get("completeness", "")),
                    "contamination": str(row.get("contamination", "")),
                    "n50": str(row.get("n50", "")),
                    "no_scaffolds": str(row.get("no_scaffolds", "")),
                    "gc_content": str(row.get("gc_content", "")),
                    "genome_size": str(row.get("genome_size", "")),
                    "motu_status": str(row.get("motu_status", "")),
                    "gunc_css": str(row.get("gunc_css", "")),
                    "location": location,
                    "download_url": download_url,
                    "local_path": str(out_path),
                    "status": "pending",
                    "error": "",
                }
            )
            picked += 1

        return index, motu, motu_jobs, ""

    by_index: Dict[int, List[Dict]] = {}
    collection_failures: List[Dict] = []
    total = len(motus)
    collected = 0
    with ThreadPoolExecutor(max_workers=collect_workers) as executor:
        futures = [
            executor.submit(collect_jobs_for_motu, (idx, motu_row))
            for idx, motu_row in enumerate(motus)
        ]
        for future in as_completed(futures):
            idx, motu, motu_jobs, error = future.result()
            by_index[idx] = motu_jobs
            collected += 1
            if motu:
                if error:
                    collection_failures.append({"motu": motu, "error": error})
                    print(f"Failed MAG collection for {motu} ({collected}/{total}) | {error}")
                else:
                    print(f"Collected MAGs for {motu}: {len(motu_jobs)} jobs ({collected}/{total})")

    jobs: List[Dict] = []
    for idx in range(total):
        jobs.extend(by_index.get(idx, []))
    return jobs, collection_failures


def download_one(job: Dict, timeout_seconds: int, download_retries: int) -> Dict:
    out_path = Path(job["local_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        job["status"] = "skipped_existing"
        return job

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, download_retries + 1):
        try:
            with requests.get(job["download_url"], stream=True, timeout=(15, timeout_seconds)) as response:
                response.raise_for_status()
                with open(part_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(part_path, out_path)
            job["status"] = "downloaded"
            return job
        except Exception as exc:
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            if attempt == download_retries:
                job["status"] = "failed"
                job["error"] = str(exc)
                return job
            time.sleep(min(2 * attempt, 10))
    return job


def write_tsv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def main() -> None:
    max_motus = 5000
    max_mags_per_motu = 20
    total_workers = env_int("TOTAL_WORKERS", os.cpu_count() or 1)
    collect_workers = env_int("COLLECT_WORKERS", total_workers)
    download_workers = env_int("DOWNLOAD_WORKERS", max(1, total_workers // 4))
    request_timeout_seconds = 120
    api_retries = 4
    download_retries = 3

    project_dir = Path(__file__).resolve().parent
    outputs_dir = project_dir / "outputs"
    downloads_dir = outputs_dir / "downloads"
    manifests_dir = outputs_dir / "manifests"
    completion_flag = downloads_dir / "_DOWNLOAD_COMPLETE"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    completion_flag.unlink(missing_ok=True)
    print(
        "Worker config | "
        f"total_workers={total_workers} "
        f"collect_workers={collect_workers} "
        f"download_workers={download_workers}"
    )

    print(f"Fetching up to {max_motus} mOTUs ...")
    try:
        motus = fetch_motus(
            max_motus=max_motus,
            timeout_seconds=request_timeout_seconds,
            api_retries=api_retries,
        )
    except Exception as exc:
        raise RuntimeError(
            "Network error while reaching mOTUs API. "
            "If this node has restricted internet access, run from a login/egress node."
        ) from exc
    print(f"Selected mOTUs: {len(motus)}")

    motu_rows = []
    for row in motus:
        motu_rows.append(
            {
                "motu": str(row.get("motu", "")),
                "domain": str(row.get("domain", "")),
                "phylum": str(row.get("phylum", "")),
                "class_": str(row.get("class_", "")),
                "order": str(row.get("order", "")),
                "family": str(row.get("family", "")),
                "genus": str(row.get("genus", "")),
                "species": str(row.get("species", "")),
                "representative": str(row.get("representative", "")),
            }
        )

    write_tsv(
        manifests_dir / "motu_manifest.tsv",
        motu_rows,
        ["motu", "domain", "phylum", "class_", "order", "family", "genus", "species", "representative"],
    )

    jobs, collection_failures = build_download_jobs(
        motus=motus,
        max_mags_per_motu=max_mags_per_motu,
        timeout_seconds=request_timeout_seconds,
        api_retries=api_retries,
        collect_workers=collect_workers,
    )

    if collection_failures:
        write_tsv(
            manifests_dir / "mag_collection_failures.tsv",
            sorted(collection_failures, key=lambda r: r.get("motu", "")),
            ["motu", "error"],
        )
        print(f"MAG collection failures: {len(collection_failures)} (see manifests/mag_collection_failures.tsv)")

    print(f"MAG download jobs: {len(jobs)}")

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [
            executor.submit(download_one, job, request_timeout_seconds, download_retries)
            for job in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']:>16} | {result['motu']} | {result['genome']}")

    write_tsv(
        manifests_dir / "mag_manifest.tsv",
        sorted(results, key=lambda r: (r.get("motu", ""), r.get("genome", ""))),
        [
            "motu",
            "domain",
            "phylum",
            "class_",
            "order",
            "family",
            "genus",
            "species",
            "genome",
            "study",
            "qscore",
            "completeness",
            "contamination",
            "n50",
            "no_scaffolds",
            "gc_content",
            "genome_size",
            "motu_status",
            "gunc_css",
            "location",
            "download_url",
            "local_path",
            "status",
            "error",
        ],
    )

    completion_flag.write_text("done\n", encoding="utf-8")
    print(f"Done. Completion flag: {completion_flag}")


if __name__ == "__main__":
    main()
