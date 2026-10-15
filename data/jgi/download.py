"""Restore and download JGI files for the collection in the current directory.

Run Script 1 first. This script uses the payloads created by Script 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .portal import (
    DOWNLOAD_URL,
    RESTORE_URL,
    JGIDataPortalClient,
    clean_text,
    ensure_dir,
    load_dotenv_from_project,
    payload_file_count,
    read_json,
    write_csv,
    write_json,
)


PROJECT_DIR = Path.cwd().resolve()
OUTPUT_DIR = PROJECT_DIR / "outputs"

PAYLOAD_CHUNKS_DIR = OUTPUT_DIR / "jgi_payload_chunks"
DOWNLOAD_DIR = OUTPUT_DIR / "jgi_downloads"
ORGANIZED_DIR = OUTPUT_DIR / "jgi_downloads_by_sample"
STATUS_DIR = OUTPUT_DIR / "jgi_restore_status"
DOWNLOAD_RESULTS_CSV = OUTPUT_DIR / "jgi_download_results.csv"

DOWNLOAD_PAYLOAD_PATH = OUTPUT_DIR / "jgi_download_payload.json"
RESTORE_REQUESTS_CSV = OUTPUT_DIR / "jgi_restore_requests.csv"
RESTORE_REQUESTS_JSON = OUTPUT_DIR / "jgi_restore_requests.json"
DOWNLOAD_PLAN_PATH = OUTPUT_DIR / "jgi_selected_download_plan.csv"
SAMPLE_SUMMARY_PATH = OUTPUT_DIR / "jgi_sample_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore, download, and organize JGI files.")
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=["plan", "restore", "status", "download", "organize", "all"],
    )
    parser.add_argument("--poll", action="store_true", help="Keep checking restore status until ready or timed out.")
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--max-poll-minutes", type=int, default=5)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of simultaneous ZIP downloads (default: 8).",
    )
    parser.add_argument("--force-organize", action="store_true")
    return parser.parse_args()


def load_payloads() -> list[dict[str, Any]]:
    index_path = PAYLOAD_CHUNKS_DIR / "payload_chunks.csv"
    if index_path.exists():
        import pandas as pd

        chunks = pd.read_csv(index_path)
        payloads = [read_json(Path(path)) for path in chunks["payload_path"].dropna().astype(str)]
        if payloads:
            return payloads

    if DOWNLOAD_PAYLOAD_PATH.exists():
        return [read_json(DOWNLOAD_PAYLOAD_PATH)]

    raise FileNotFoundError("Missing download payloads. Run Script 1 first.")


def worker_count() -> int:
    return max(1, os.cpu_count() or 1)
    # return 2


def print_plan(payloads: list[dict[str, Any]]) -> None:
    print("Download/restore plan:")
    for number, payload in enumerate(payloads, start=1):
        payload_text = json.dumps(payload, separators=(",", ":"))
        print(
            f"  part={number} "
            f"datasets={len(payload.get('ids', {}))} "
            f"files={payload_file_count(payload)} "
            f"payload_chars={len(payload_text)}"
        )
    print(f"Restore endpoint:  {RESTORE_URL}")
    print(f"Download endpoint: {DOWNLOAD_URL}")


def save_restore_rows(rows: list[dict[str, Any]]) -> None:
    write_csv(RESTORE_REQUESTS_CSV, rows)
    write_json(RESTORE_REQUESTS_JSON, rows)


# A JGI session token lasts well under the ~2 h a full 4,187-part campaign needs, and it can
# only be reissued from the browser ("Copy My Session Token" -- the signon.jgi.doe.gov password
# flow no longer hands out a usable session). So `restore` resumes: parts that already hold a
# request_status_url are skipped, and an auth failure stops cleanly instead of unwinding, which
# would otherwise re-request every part on the next run.
AUTH_FAILURE_MARKERS = ("HTTP 401", "HTTP 424")
TRANSIENT_MARKERS = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")
RESTORE_MAX_RETRIES = 5
DOWNLOAD_MAX_RETRIES = 6
DOWNLOAD_RETRY_DELAY_SECONDS = 30
# Unpaced, the loop ran ~1.6 req/s; a little spacing makes edge rate-limiting less likely.
RESTORE_REQUEST_DELAY_SECONDS = 0.5

# A 404 is NOT transient and NOT about the endpoint: it means one dataset in the payload is no
# longer in JGI's catalog, and it fails the whole multi-dataset request. Observed at part 2017,
# where IMG_AP-1490186 404s while the other two datasets in the same chunk return 200. So on a
# 404 we split the chunk and submit per dataset, keeping the good ones and recording the dead.
UNAVAILABLE_DATASETS_CSV = OUTPUT_DIR / "jgi_unavailable_datasets.csv"


def load_completed_restore_parts() -> dict[int, dict[str, Any]]:
    """Parts already requested in an earlier run, keyed by part number."""
    try:
        previous = load_restore_rows()
    except FileNotFoundError:
        return {}
    completed: dict[int, dict[str, Any]] = {}
    for row in previous:
        try:
            part = int(row.get("part") or 0)
        except (TypeError, ValueError):
            continue
        if part and clean_text(row.get("request_status_url")):
            completed[part] = row
    return completed


class AuthExpired(RuntimeError):
    """The JGI session token stopped being accepted mid-campaign."""


def post_with_retry(client: JGIDataPortalClient, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one payload, retrying genuinely transient failures. Raises on 404 and auth loss."""
    for attempt in range(1, RESTORE_MAX_RETRIES + 1):
        try:
            return client.post_json(RESTORE_URL, payload)
        except RuntimeError as error:
            message = str(error)
            if any(marker in message for marker in AUTH_FAILURE_MARKERS):
                raise AuthExpired(message) from error
            if attempt == RESTORE_MAX_RETRIES or not any(
                marker in message for marker in TRANSIENT_MARKERS
            ):
                raise
            delay = 10 * attempt
            print(f"  transient failure, retrying in {delay}s ({attempt}/{RESTORE_MAX_RETRIES})")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def post_restore_payload(
    client: JGIDataPortalClient, payload: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Submit a chunk; on a 404 fall back to one request per dataset.

    Returns (successful [(dataset, response)], datasets that 404'd). `dataset` is "" when the
    whole chunk went through in a single request.
    """
    try:
        return [("", post_with_retry(client, payload))], []
    except AuthExpired:
        raise
    except RuntimeError as error:
        if "HTTP 404" not in str(error):
            raise

    datasets = payload.get("ids", {})
    if len(datasets) <= 1:
        return [], sorted(datasets)

    print(f"  chunk 404 -- splitting across {len(datasets)} dataset(s)")
    responses: list[tuple[str, dict[str, Any]]] = []
    dead: list[str] = []
    for dataset, entry in datasets.items():
        single = {
            "api_version": payload.get("api_version", "2"),
            "ids": {dataset: entry},
            "send_mail": payload.get("send_mail", False),
        }
        try:
            responses.append((dataset, post_with_retry(client, single)))
        except AuthExpired:
            raise
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise
            dead.append(dataset)
        time.sleep(RESTORE_REQUEST_DELAY_SECONDS)
    return responses, dead


def record_unavailable_datasets(part: int, datasets: list[str]) -> None:
    """Append 404'd datasets so the coverage loss is auditable, not silent."""
    existing: list[dict[str, Any]] = []
    if UNAVAILABLE_DATASETS_CSV.exists():
        existing = read_csv_rows(UNAVAILABLE_DATASETS_CSV)
    known = {row.get("organism_id") for row in existing}
    for dataset in datasets:
        if dataset not in known:
            existing.append({"organism_id": dataset, "part": part, "reason": "HTTP 404"})
    write_csv(UNAVAILABLE_DATASETS_CSV, existing)


def restore_files(client: JGIDataPortalClient, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ensure_dir(STATUS_DIR)
    completed = load_completed_restore_parts()
    rows: list[dict[str, Any]] = [completed[part] for part in sorted(completed)]
    if completed:
        print(f"[restore] resuming; {len(completed):,}/{len(payloads):,} part(s) already requested")

    for number, payload in enumerate(payloads, start=1):
        if number in completed:
            continue
        print(f"[restore {number}/{len(payloads)}] files={payload_file_count(payload)}")

        try:
            responses, dead = post_restore_payload(client, payload)
        except AuthExpired as error:
            save_restore_rows(rows)
            print(
                f"\n[restore] stopped at part {number}: the JGI session token is no longer "
                f"accepted.\n[restore] {len(rows):,} part(s) are saved in "
                f"{RESTORE_REQUESTS_CSV}.\n[restore] Refresh JGI_SESSION_TOKEN in .env "
                f"(files.jgi.doe.gov -> 'Copy My Session Token'), then rerun "
                f"'restore' -- it will resume from part {number}.\n[restore] {error}",
                flush=True,
            )
            return rows
        except RuntimeError:
            save_restore_rows(rows)
            raise

        if dead:
            record_unavailable_datasets(number, dead)
            print(f"  {len(dead)} dataset(s) unavailable (404), skipped: {', '.join(dead)}")

        for index, (dataset, response) in enumerate(responses):
            suffix = "" if len(responses) == 1 else f"_{index:02d}"
            response_path = STATUS_DIR / f"restore_part{number:03d}{suffix}_response.json"
            write_json(response_path, response)
            status_url = clean_text(
                response.get("request_status_url")
                or response.get("status_url")
                or response.get("url")
                or response.get("raw_text")
            )
            rows.append(
                {
                    "part": number,
                    "dataset": dataset,
                    "file_count": payload_file_count(payload) if len(responses) == 1 else "",
                    "request_status_url": status_url,
                    "status": "",
                    "expiration_date": "",
                    "restore_response_path": str(response_path),
                    "last_status_path": "",
                    "last_checked_at": "",
                }
            )
            print(f"  status_url={status_url or 'not found'}")

        save_restore_rows(rows)
        time.sleep(RESTORE_REQUEST_DELAY_SECONDS)

    return rows


def load_restore_rows() -> list[dict[str, Any]]:
    if RESTORE_REQUESTS_JSON.exists():
        return read_json(RESTORE_REQUESTS_JSON)
    if RESTORE_REQUESTS_CSV.exists():
        return read_csv_rows(RESTORE_REQUESTS_CSV)
    raise FileNotFoundError("No restore request table found. Run 'restore' first.")


def check_restore_status(
    client: JGIDataPortalClient,
    rows: list[dict[str, Any]],
    poll: bool,
    poll_interval_seconds: int,
    max_poll_minutes: int,
) -> list[dict[str, Any]]:
    ensure_dir(STATUS_DIR)

    if poll:
        return wait_for_all_restore_statuses(
            client,
            rows,
            poll_interval_seconds,
            max_poll_minutes,
        )

    refresh_rows_in_parallel(rows)
    save_restore_rows(rows)
    return rows


def refresh_rows_in_parallel(rows: list[dict[str, Any]]) -> None:
    targets = [
        row
        for row in rows
        if clean_text(row.get("request_status_url"))
    ]
    workers = worker_count()
    print(f"Checking {len(targets)} restore request(s) with {workers} workers...")
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(refresh_one_status, JGIDataPortalClient.from_env(required_auth=True), row)
            for row in targets
        ]
        for job in futures.as_completed(jobs):
            job.result()


def refresh_one_status(client: JGIDataPortalClient, row: dict[str, Any]) -> str:
    part = int(row["part"])
    try:
        status = client.get_status(clean_text(row.get("request_status_url")))
    except Exception as error:
        row["status_check_error"] = str(error)
        row["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"  part {part}: status check failed: {error}", flush=True)
        return ""

    status_path = STATUS_DIR / f"restore_part{part:03d}_status.json"
    write_json(status_path, status)

    row["status"] = clean_text(status.get("status"))
    row["expiration_date"] = clean_text(status.get("expiration_date"))
    row["globus_download_url"] = clean_text(status.get("globus_download_url"))
    row["last_status_path"] = str(status_path)
    row["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["status_check_error"] = ""
    print(f"  part {part}: status={row['status'] or 'unknown'} expiration={row['expiration_date'] or '-'}", flush=True)
    return clean_text(row["status"]).upper()


def wait_for_all_restore_statuses(
    client: JGIDataPortalClient,
    rows: list[dict[str, Any]],
    poll_interval_seconds: int,
    max_poll_minutes: int,
) -> list[dict[str, Any]]:
    deadline = time.time() + max_poll_minutes * 60
    round_number = 1
    terminal_statuses = {"READY", "EXPIRED", "FAILED", "ERROR"}

    while time.time() <= deadline:
        print(f"[poll round {round_number}] checking all restore requests...", flush=True)
        targets = [
            row
            for row in rows
            if clean_text(row.get("request_status_url"))
            and clean_text(row.get("status")).upper() not in terminal_statuses
        ]
        with futures.ThreadPoolExecutor(max_workers=worker_count()) as pool:
            jobs = [
                pool.submit(refresh_one_status, JGIDataPortalClient.from_env(required_auth=True), row)
                for row in targets
            ]
            statuses = [job.result() for job in futures.as_completed(jobs)]
        unfinished = sum(1 for status in statuses if status not in terminal_statuses)

        save_restore_rows(rows)

        if unfinished == 0:
            return rows

        round_number += 1
        print(f"  {unfinished} request(s) still not terminal. Waiting {poll_interval_seconds} seconds...", flush=True)
        time.sleep(poll_interval_seconds)

    print("Polling timed out. Saved the latest statuses; rerun status --poll later.", flush=True)
    return rows


def ready_part_numbers() -> set[int]:
    try:
        rows = load_restore_rows()
    except FileNotFoundError:
        return set()
    return {
        int(row["part"])
        for row in rows
        if clean_text(row.get("status")).upper() == "READY"
    }


def every_restore_is_ready(rows: list[dict[str, Any]]) -> bool:
    statuses = [clean_text(row.get("status")).upper() for row in rows if clean_text(row.get("request_status_url"))]
    return bool(statuses) and all(status == "READY" for status in statuses)


def simple_download_payload(payload: dict[str, Any]) -> dict[str, Any]:
    ids = {}
    for organism_id, value in payload.get("ids", {}).items():
        if isinstance(value, dict):
            ids[organism_id] = value.get("file_ids", [])
        else:
            ids[organism_id] = value
    return {"ids": ids, "api_version": "2"}


def remove_unavailable_datasets(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove datasets that JGI reported as unavailable during restore."""
    if not UNAVAILABLE_DATASETS_CSV.exists():
        return payload

    unavailable = {
        clean_text(row.get("organism_id"))
        for row in read_csv_rows(UNAVAILABLE_DATASETS_CSV)
        if clean_text(row.get("organism_id"))
    }
    cleaned = dict(payload)
    cleaned["ids"] = {
        organism_id: value
        for organism_id, value in payload.get("ids", {}).items()
        if organism_id not in unavailable
    }
    return cleaned


def ready_payloads(payloads: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    try:
        rows = load_restore_rows()
    except FileNotFoundError:
        print("No restore requests found. Run this first:")
        print("  python -m jgi.download restore")
        return []

    statuses_by_part: dict[int, list[str]] = {}
    for row in rows:
        part = int(row["part"])
        statuses_by_part.setdefault(part, []).append(clean_text(row.get("status")).upper())
    ready_parts = {
        part
        for part, statuses in statuses_by_part.items()
        if statuses and all(status == "READY" for status in statuses)
    }
    if not ready_parts:
        print("No restore requests are READY yet.")
        print("Run a short poll, then download whatever becomes ready:")
        print("  python -m jgi.download status --poll --max-poll-minutes 5")
        return []

    skipped = len(payloads) - len(ready_parts)
    if skipped > 0:
        print(f"Downloading {len(ready_parts)} READY part(s); skipping {skipped} not-ready part(s).")
    ready = []
    removed_dataset_count = 0
    for number, payload in enumerate(payloads, start=1):
        if number not in ready_parts:
            continue
        cleaned = remove_unavailable_datasets(payload)
        removed_dataset_count += len(payload.get("ids", {})) - len(cleaned.get("ids", {}))
        if cleaned.get("ids"):
            ready.append((number, cleaned))
    if removed_dataset_count:
        print(f"Excluded {removed_dataset_count} dataset(s) that JGI reported as unavailable.")
    return ready


def download_files(
    client: JGIDataPortalClient,
    payloads: list[tuple[int, dict[str, Any]]],
    force: bool,
    workers: int,
) -> None:
    ensure_dir(DOWNLOAD_DIR)
    workers = max(1, workers)
    print(f"Downloading with {workers} workers...")

    with tqdm(
        total=len(payloads),
        desc="Downloading",
        unit="chunk",
        dynamic_ncols=True,
    ) as progress:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [
                pool.submit(download_one_part, number, payload, force)
                for number, payload in payloads
            ]
            results = []
            failed_count = 0
            for job in futures.as_completed(jobs):
                result = job.result()
                results.append(result)
                if result["status"] in {"downloaded", "already_exists"}:
                    progress.update(1)
                else:
                    failed_count += 1
                progress.set_postfix(failed=failed_count)
                if result["error"]:
                    print(f"[part {result['part']}] {result['error']}")
                write_csv(DOWNLOAD_RESULTS_CSV, results)
                if any(marker in result["error"] for marker in AUTH_FAILURE_MARKERS):
                    for pending_job in jobs:
                        pending_job.cancel()
                    print("Download stopped because the JGI session token expired. Refresh it and rerun.")
                    break

    write_csv(DOWNLOAD_RESULTS_CSV, sorted(results, key=lambda row: int(row["part"])))
    failed = [row for row in results if row["status"] == "failed"]
    if failed:
        print(f"Download finished with {len(failed)} failed READY part(s). See {DOWNLOAD_RESULTS_CSV}")


def download_one_part(
    number: int,
    payload: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    zip_path = DOWNLOAD_DIR / f"jgi_download_part{number:03d}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0 and zipfile.is_zipfile(zip_path) and not force:
        size = zip_path.stat().st_size
        return {
            "part": number,
            "status": "already_exists",
            "zip_path": str(zip_path),
            "bytes_written": size,
            "total_bytes": size,
            "error": "",
        }

    try:
        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                load_dotenv_from_project(PROJECT_DIR)
                client = JGIDataPortalClient.from_env(required_auth=True)
                bytes_written = client.stream_download_zip(
                    simple_download_payload(payload),
                    zip_path,
                )
                if not zipfile.is_zipfile(zip_path):
                    zip_path.unlink(missing_ok=True)
                    raise RuntimeError("JGI response was not a valid ZIP file")
                break
            except Exception as error:
                message = str(error)
                retryable = "HTTP 401" not in message and "HTTP 409" not in message
                if not retryable or attempt == DOWNLOAD_MAX_RETRIES:
                    raise
                delay = min(120, DOWNLOAD_RETRY_DELAY_SECONDS * attempt)
                print(
                    f"[part {number}] temporary JGI failure; retrying in {delay}s "
                    f"({attempt}/{DOWNLOAD_MAX_RETRIES})",
                    flush=True,
                )
                time.sleep(delay)
        return {
            "part": number,
            "status": "downloaded",
            "zip_path": str(zip_path),
            "bytes_written": bytes_written,
            "total_bytes": bytes_written,
            "error": "",
        }
    except Exception as error:
        return {
            "part": number,
            "status": "failed",
            "zip_path": str(zip_path),
            "bytes_written": 0,
            "total_bytes": 0,
            "error": str(error),
        }


def read_download_plan() -> list[dict[str, Any]]:
    if not DOWNLOAD_PLAN_PATH.exists():
        raise FileNotFoundError(f"Missing download plan: {DOWNLOAD_PLAN_PATH}. Run Script 1 first.")
    return read_csv_rows(DOWNLOAD_PLAN_PATH)


def read_sample_summaries() -> dict[str, dict[str, Any]]:
    if not SAMPLE_SUMMARY_PATH.exists():
        return {}
    rows = read_csv_rows(SAMPLE_SUMMARY_PATH)
    return {clean_text(row.get("query_value")): row for row in rows}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def make_destination_maps(
    plan_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[tuple[str, str], str]]:
    by_file_id: dict[str, str] = {}
    by_file_name: dict[str, list[str]] = {}
    by_sample_and_file_name: dict[tuple[str, str], str] = {}

    for row in plan_rows:
        sample_id = clean_text(row.get("query_value"))
        file_id = clean_text(row.get("file_id"))
        file_name = clean_text(row.get("file_name")).lower()
        destination = clean_text(row.get("suggested_local_path"))

        if file_id:
            by_file_id[file_id] = destination
        if file_name:
            by_file_name.setdefault(file_name, []).append(destination)
            if sample_id:
                by_sample_and_file_name[(sample_id, file_name)] = destination

    return by_file_id, by_file_name, by_sample_and_file_name


def choose_destination(member_name: str, by_file_id: dict[str, str], by_file_name: dict[str, list[str]]) -> str:
    for file_id, destination in by_file_id.items():
        if file_id in member_name:
            return destination

    file_name = Path(member_name).name.lower()
    destinations = by_file_name.get(file_name, [])
    return destinations[0] if len(destinations) == 1 else ""


def sample_id_from_destination(destination: str) -> str:
    clean_destination = clean_text(destination)
    if not clean_destination:
        return ""
    return Path(clean_destination).parts[0]


def infer_member_sample_ids(
    members: list[zipfile.ZipInfo],
    by_file_id: dict[str, str],
    by_file_name: dict[str, list[str]],
) -> dict[str, str]:
    folder_to_sample_id: dict[str, str] = {}

    for member in members:
        if member.is_dir():
            continue

        member_name = member.filename.replace("\\", "/")
        top_folder = Path(member_name).parts[0] if "/" in member_name else ""
        if not top_folder or top_folder in folder_to_sample_id:
            continue

        destination = choose_destination(member_name, by_file_id, by_file_name)
        sample_id = sample_id_from_destination(destination)
        if sample_id:
            folder_to_sample_id[top_folder] = sample_id

    return folder_to_sample_id


def choose_destination_with_sample_context(
    member_name: str,
    by_file_id: dict[str, str],
    by_file_name: dict[str, list[str]],
    by_sample_and_file_name: dict[tuple[str, str], str],
    folder_to_sample_id: dict[str, str],
) -> str:
    destination = choose_destination(member_name, by_file_id, by_file_name)
    if destination:
        return destination

    member_path = Path(member_name.replace("\\", "/"))
    if len(member_path.parts) < 2:
        return ""

    sample_id = folder_to_sample_id.get(member_path.parts[0], "")
    if not sample_id:
        return ""

    return by_sample_and_file_name.get((sample_id, member_path.name.lower()), "")


def unused_path(path: Path) -> Path:
    if not path.exists():
        return path

    for number in range(2, 10000):
        candidate = path.with_name(f"{path.stem}__{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not make a unique path for {path}")


def extract_one_file(zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, out_path: Path, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        out_path = unused_path(out_path)

    with zip_file.open(member) as source, out_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def sample_ids_from_plan(plan_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({clean_text(row.get("query_value")) for row in plan_rows if clean_text(row.get("query_value"))})


def write_sample_summary_files(plan_rows: list[dict[str, Any]]) -> None:
    summaries = read_sample_summaries()
    sample_ids = sample_ids_from_plan(plan_rows)
    print(f"Writing sample summaries for {len(sample_ids)} sample(s)...")

    for index, sample_id in enumerate(sample_ids, start=1):
        rows = [row for row in plan_rows if clean_text(row.get("query_value")) == sample_id]
        sample_level_rows = [
            row
            for row in rows
            if clean_text(row.get("selection_reason")) == "sample_level_file"
        ]
        bin_files = [
            clean_text(row.get("file_name"))
            for row in rows
            if clean_text(row.get("selection_reason")) == "bin_artifact"
        ]
        summary = {
            "sample_id": sample_id,
            "selected_file_count": len(rows),
            "sample_level_file_count": len(sample_level_rows),
            "bin_artifact_count": len(bin_files),
            "sample_level_files": [
                {
                    "file_name": clean_text(row.get("file_name")),
                    "file_status": clean_text(row.get("file_status")),
                    "file_id": clean_text(row.get("file_id")),
                    "source_path": clean_text(row.get("file_file_path")),
                    "organized_path": clean_text(row.get("suggested_local_path")),
                }
                for row in sample_level_rows
            ],
            "bin_files": bin_files,
            "sample_metadata": summaries.get(sample_id, {}),
        }
        sample_dir = ORGANIZED_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if index == 1 or index == len(sample_ids) or index % 250 == 0:
            print(
                f"  [sample summaries] {index}/{len(sample_ids)} written; "
                f"latest={sample_id}",
                flush=True,
            )


def organize_one_zip(
    zip_path: Path,
    by_file_id: dict[str, str],
    by_file_name: dict[str, list[str]],
    by_sample_and_file_name: dict[tuple[str, str], str],
    force: bool,
) -> dict[str, Any]:
    organized = 0
    unmatched = 0
    member_files = 0

    with zipfile.ZipFile(zip_path) as zip_file:
        members = zip_file.infolist()
        folder_to_sample_id = infer_member_sample_ids(members, by_file_id, by_file_name)

        for member in members:
            if member.is_dir():
                continue
            member_files += 1

            member_name = member.filename.replace("\\", "/")
            destination = choose_destination_with_sample_context(
                member_name,
                by_file_id,
                by_file_name,
                by_sample_and_file_name,
                folder_to_sample_id,
            )
            if destination:
                out_path = ORGANIZED_DIR / destination
                organized += 1
            else:
                out_path = ORGANIZED_DIR / "_unmatched" / zip_path.stem / Path(member_name).name
                unmatched += 1

            extract_one_file(zip_file, member, out_path, force)

    zip_path.unlink()
    return {
        "zip_path": str(zip_path),
        "organized": organized,
        "unmatched": unmatched,
        "member_files": member_files,
    }


def organize_downloads(force: bool) -> None:
    plan_rows = read_download_plan()
    by_file_id, by_file_name, by_sample_and_file_name = make_destination_maps(plan_rows)
    zip_paths = sorted(DOWNLOAD_DIR.glob("*.zip"))
    print(f"Loaded {len(plan_rows):,} planned file row(s).")
    write_sample_summary_files(plan_rows)

    if not zip_paths:
        print(f"No ZIP files found in {DOWNLOAD_DIR}. Sample summaries were still written.")
        return

    organized = 0
    unmatched = 0
    processed_zips = 0
    extracted_files = 0
    workers = worker_count()
    print(f"Organizing {len(zip_paths)} ZIP file(s) with {workers} workers...")
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(organize_one_zip, zip_path, by_file_id, by_file_name, by_sample_and_file_name, force)
            for zip_path in zip_paths
        ]
        for job in futures.as_completed(jobs):
            try:
                result = job.result()
            except Exception as error:
                print(f"[organize failed] {error}")
                continue
            processed_zips += 1
            organized += int(result["organized"])
            unmatched += int(result["unmatched"])
            extracted_files += int(result["member_files"])
            print(
                f"[organized {processed_zips}/{len(zip_paths)}] {Path(result['zip_path']).name}; "
                f"files={result['member_files']:,}, matched={result['organized']:,}, "
                f"unmatched={result['unmatched']:,}; running_total={extracted_files:,}",
                flush=True,
            )

    print(f"Organized files: {organized:,} -> {ORGANIZED_DIR}")
    if unmatched:
        print(f"Unmatched files:  {unmatched:,} -> {ORGANIZED_DIR / '_unmatched'}")
        # Delete the "_unmatched" subfolder in the ORGANIZED_DIR directory, if it exists.
        unmatched_dir = ORGANIZED_DIR / "_unmatched"
        if unmatched_dir.exists() and unmatched_dir.is_dir():
            shutil.rmtree(unmatched_dir)
            print(f"Deleted unmatched folder: {unmatched_dir}")


def main() -> None:
    args = parse_args()
    load_dotenv_from_project(PROJECT_DIR)
    ensure_dir(OUTPUT_DIR)

    try:
        payloads = load_payloads()
    except FileNotFoundError as error:
        print(error)
        return

    if args.action == "plan":
        print_plan(payloads)
        return

    if args.action == "organize":
        organize_downloads(force=args.force_organize)
        return

    client = JGIDataPortalClient.from_env(required_auth=True)

    if args.action == "restore":
        restore_files(client, payloads)
        return

    if args.action == "status":
        rows = load_restore_rows()
        check_restore_status(client, rows, args.poll, args.poll_interval_seconds, args.max_poll_minutes)
        return

    if args.action == "download":
        payloads_to_download = ready_payloads(payloads)
        if not payloads_to_download:
            return
        download_files(
            client,
            payloads_to_download,
            force=args.force_download,
            workers=args.download_workers,
        )
        return

    if args.action == "all":
        rows = restore_files(client, payloads)
        rows = check_restore_status(client, rows, True, args.poll_interval_seconds, args.max_poll_minutes)
        payloads_to_download = ready_payloads(payloads)
        if not payloads_to_download:
            print("No files were ready before the polling limit. Rerun 'status --poll' later.")
            return
        download_files(
            client,
            payloads_to_download,
            force=args.force_download,
            workers=args.download_workers,
        )
        organize_downloads(force=args.force_organize)


if __name__ == "__main__":
    main()
