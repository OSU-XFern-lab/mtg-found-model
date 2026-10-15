"""Search JGI and prepare download payloads for the collection in the current directory.

This script reads inputs/jgi_query_config.json, searches the JGI Data Portal,
keeps the files we care about, and writes files that Script 2 can download.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .portal import (
    JGIDataPortalClient,
    build_current_payload,
    build_manifest_rows,
    build_search_params,
    clean_text,
    endpoint_url,
    ensure_dir,
    extract_query_values,
    load_dotenv_from_project,
    payload_file_count,
    selected_manifest_rows,
    split_rows_for_payload_limit,
    url_with_params,
    write_csv,
    write_json,
)


PROJECT_DIR = Path.cwd().resolve()
INPUT_DIR = PROJECT_DIR / "inputs"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CONFIG_PATH = INPUT_DIR / "jgi_query_config.json"
RAW_DIR = OUTPUT_DIR / "jgi_search_raw"
IMG_METADATA_RAW_DIR = OUTPUT_DIR / "img_metadata_raw"
PLAN_PATH = OUTPUT_DIR / "jgi_selected_download_plan.csv"
SAMPLE_SUMMARY_PATH = OUTPUT_DIR / "jgi_sample_summary.csv"
RESTORE_PAYLOAD_PATH = OUTPUT_DIR / "jgi_restore_payload.json"
DOWNLOAD_PAYLOAD_PATH = OUTPUT_DIR / "jgi_download_payload.json"
PAYLOAD_CHUNKS_DIR = OUTPUT_DIR / "jgi_payload_chunks"
SEARCH_URLS_PATH = OUTPUT_DIR / "jgi_search_urls.txt"
FACETS_PATH = OUTPUT_DIR / "jgi_search_facets.json"
FULL_MANIFEST_PATH = OUTPUT_DIR / "jgi_file_manifest.csv"
WIDE_SELECTED_PATH = OUTPUT_DIR / "jgi_file_manifest_selected.csv"
SIGNON_URL = "https://signon.jgi.doe.gov/signon/create"
IMG_DETAIL_URL = "https://img.jgi.doe.gov/cgi-bin/mer/main.cgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search JGI and create download payloads.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--limit-query-values", type=int, help="Pilot mode: search only the first N samples.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached JGI search responses.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_sample_ids(config: dict[str, Any], limit: int | None) -> list[str]:
    sample_ids = extract_query_values(config, PROJECT_DIR)
    if not sample_ids:
        sample_ids = [clean_text(config.get("search", {}).get("q")) or "search"]
    return sample_ids[:limit] if limit else sample_ids


def page_numbers(config: dict[str, Any]) -> list[int]:
    max_pages = int(config.get("search", {}).get("max_pages", 1) or 1)
    return list(range(1, max_pages + 1))


def worker_count(config: dict[str, Any]) -> int:
    value = config.get("search", {}).get("max_workers", "auto")
    if clean_text(value).lower() == "auto":
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


def img_metadata_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("img_metadata", {}).get("enabled", True))


def login_to_img() -> dict[str, str]:
    user = os.getenv("JGI_USERNAME", "").strip().strip('"')
    password = os.getenv("JGI_PASSWORD", "").strip().strip('"')
    if not user or not password:
        print("Skipping IMG metadata scrape: JGI_USERNAME/JGI_PASSWORD are missing.")
        return {}

    session = requests.Session()
    response = session.post(SIGNON_URL, data={"login": user, "password": password}, timeout=120)
    response.raise_for_status()
    return session.cookies.get_dict()


def fetch_img_html(sample_id: str, cookies: dict[str, str], force_refresh: bool) -> str:
    ensure_dir(IMG_METADATA_RAW_DIR)
    path = IMG_METADATA_RAW_DIR / f"{sample_id}.html"
    if path.exists() and not force_refresh:
        return path.read_text(encoding="utf-8", errors="ignore")

    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value)

    response = session.get(
        IMG_DETAIL_URL,
        params={"section": "TaxonDetail", "page": "taxonDetail", "taxon_oid": sample_id},
        timeout=120,
    )
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def clean_key(value: str) -> str:
    text = " ".join(clean_text(value).split())
    text = text.replace("/", " ").replace("%", "percent")
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.lower().split("_") if part)


def unique_key(data: dict[str, Any], key: str) -> str:
    if key not in data:
        return key
    number = 2
    while f"{key}_{number}" in data:
        number += 1
    return f"{key}_{number}"


def parse_img_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    data: dict[str, Any] = {}
    section = "general"

    for tr in soup.find_all("tr"):
        cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in tr.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell != ""]
        if len(cells) < 2:
            continue

        label, value = cells[0], cells[1]
        if label and not value:
            section = clean_key(label) or section
            continue
        if not label and value:
            section = clean_key(value) or section
            continue

        key = clean_key(label)
        if section not in {"general", ""}:
            key = f"{section}_{key}"
        data[unique_key(data, key)] = value

        if len(cells) >= 3 and cells[2]:
            data[unique_key(data, f"{key}_percent")] = cells[2]

    return data


def scrape_one_img_metadata(sample_id: str, cookies: dict[str, str], force_refresh: bool) -> dict[str, Any]:
    try:
        html = fetch_img_html(sample_id, cookies, force_refresh)
        metadata = parse_img_metadata(html)
        metadata["img_metadata_status"] = "ok"
        return metadata
    except Exception as error:
        return {"img_metadata_status": "failed", "img_metadata_error": str(error)}


def scrape_img_metadata(
    sample_ids: list[str],
    config: dict[str, Any],
    force_refresh: bool,
) -> dict[str, dict[str, Any]]:
    if not img_metadata_enabled(config):
        return {}

    cookies = login_to_img()
    if not cookies:
        return {}

    workers = worker_count(config)
    print(f"Scraping IMG metadata for {len(sample_ids):,} samples with {workers} workers...")
    metadata_by_sample: dict[str, dict[str, Any]] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(scrape_one_img_metadata, sample_id, cookies, force_refresh): sample_id
            for sample_id in sample_ids
        }
        for index, job in enumerate(futures.as_completed(jobs), start=1):
            sample_id = jobs[job]
            metadata_by_sample[sample_id] = job.result()
            print(f"[IMG {index}/{len(sample_ids)}] {sample_id}: {metadata_by_sample[sample_id].get('img_metadata_status')}")
    return metadata_by_sample


def cache_path(sample_id: str, page: int, search_url: str) -> Path:
    safe_sample_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in sample_id)
    digest = hashlib.md5(search_url.encode("utf-8")).hexdigest()[:10]
    return RAW_DIR / f"{safe_sample_id}_page{page:03d}_{digest}.json"


def should_stop(payload: dict[str, Any], params: dict[str, str]) -> bool:
    organisms = payload.get("organisms")
    if not organisms:
        return True
    if "next_page" in payload:
        return not bool(payload.get("next_page"))
    return len(organisms) < int(params.get("x", 100))


def search_one_page(
    config: dict[str, Any],
    endpoint: str,
    sample_id: str,
    page: int,
    force_refresh: bool,
) -> dict[str, Any]:
    params = build_search_params(config, query_value=sample_id)
    params["p"] = str(page)
    search_url = url_with_params(endpoint, params)
    raw_path = cache_path(sample_id, page, search_url)

    if raw_path.exists() and not force_refresh:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        from_cache = True
    else:
        client = JGIDataPortalClient.from_env(required_auth=False)
        payload = client.get_json(endpoint, params=params)
        write_json(raw_path, payload)
        from_cache = False

    return {
        "sample_id": sample_id,
        "page": page,
        "params": params,
        "url": search_url,
        "payload": payload,
        "rows": build_manifest_rows(payload, query_value=sample_id),
        "from_cache": from_cache,
    }


def search_all_samples(
    config: dict[str, Any],
    sample_ids: list[str],
    force_refresh: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    endpoint = endpoint_url(config)
    pages = page_numbers(config)
    max_workers = worker_count(config)

    all_rows: list[dict[str, Any]] = []
    search_urls: list[str] = []
    facets: dict[str, Any] = {}

    if len(pages) == 1 and max_workers > 1:
        print(f"Searching {len(sample_ids):,} samples with {max_workers} workers...")
        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            jobs = {
                pool.submit(search_one_page, config, endpoint, sample_id, 1, force_refresh): sample_id
                for sample_id in sample_ids
            }
            for index, job in enumerate(futures.as_completed(jobs), start=1):
                result = job.result()
                all_rows.extend(result["rows"])
                search_urls.append(result["url"])
                if not facets and isinstance(result["payload"].get("facets"), dict):
                    facets = result["payload"]["facets"]
                cache_note = " cache" if result["from_cache"] else ""
                print(f"[{index}/{len(sample_ids)}] {result['sample_id']}{cache_note}: {len(result['rows'])} files")
        return all_rows, search_urls, facets

    sleep_seconds = float(config.get("search", {}).get("sleep_seconds", 0) or 0)
    for index, sample_id in enumerate(sample_ids, start=1):
        for page in pages:
            result = search_one_page(config, endpoint, sample_id, page, force_refresh)
            all_rows.extend(result["rows"])
            search_urls.append(result["url"])
            if not facets and isinstance(result["payload"].get("facets"), dict):
                facets = result["payload"]["facets"]
            print(f"[{index}/{len(sample_ids)} page {page}] {sample_id}: {len(result['rows'])} files")
            if should_stop(result["payload"], result["params"]):
                break
            if sleep_seconds:
                time.sleep(sleep_seconds)

    return all_rows, search_urls, facets


def write_download_plan(selected_rows: list[dict[str, Any]]) -> None:
    useful_columns = [
        "query_value",
        "selection_reason",
        "file_name",
        "file_status",
        "file_id",
        "file_file_size",
        "file_file_type",
        "file_file_group",
        "file_file_path",
        "suggested_local_path",
        "organism_id",
        "organism_name",
    ]
    rows = [{col: row.get(col, "") for col in useful_columns} for row in selected_rows]
    write_csv(PLAN_PATH, rows)


def write_sample_summary(
    rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    img_metadata: dict[str, dict[str, Any]],
) -> None:
    selected_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_sample.setdefault(clean_text(row.get("query_value")), []).append(row)

    summary_rows = []
    seen_samples = set()
    for row in rows:
        sample_id = clean_text(row.get("query_value"))
        if not sample_id or sample_id in seen_samples:
            continue
        seen_samples.add(sample_id)

        selected = selected_by_sample.get(sample_id, [])
        summary = {
            "query_value": sample_id,
            "organism_id": row.get("organism_id", ""),
            "organism_name": row.get("organism_name", ""),
            "organism_agg": row.get("organism_agg", ""),
            "organism_agg_id": row.get("organism_agg_id", ""),
            "selected_file_count": len(selected),
            "bin_artifact_count": sum(1 for item in selected if item.get("selection_reason") == "bin_artifact"),
            "sample_level_file_count": sum(1 for item in selected if item.get("selection_reason") == "sample_level_file"),
        }
        summary.update({f"img_{key}": value for key, value in img_metadata.get(sample_id, {}).items()})

        for key, value in row.items():
            if key.startswith("organism_") or key.startswith("top_hit_metadata_"):
                summary.setdefault(key, value)

        summary_rows.append(summary)

    write_csv(SAMPLE_SUMMARY_PATH, summary_rows)


def write_payload_chunks(selected_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    ensure_dir(PAYLOAD_CHUNKS_DIR)
    send_mail = bool(config.get("payload", {}).get("send_mail", False))
    max_chars = int(config.get("payload", {}).get("max_payload_chars", 3900) or 3900)
    chunks = split_rows_for_payload_limit(selected_rows, send_mail=send_mail, max_chars=max_chars)

    summary_rows = []
    for number, rows in enumerate(chunks, start=1):
        payload = build_current_payload(rows, send_mail=send_mail)
        path = PAYLOAD_CHUNKS_DIR / f"jgi_payload_part{number:03d}.json"
        write_json(path, payload)
        summary_rows.append(
            {
                "chunk": number,
                "payload_path": str(path),
                "dataset_count": len(payload.get("ids", {})),
                "file_count": payload_file_count(payload),
                "payload_chars": len(json.dumps(payload, separators=(",", ":"))),
            }
        )
    write_csv(PAYLOAD_CHUNKS_DIR / "payload_chunks.csv", summary_rows)


def write_optional_wide_csvs(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    outputs = config.get("outputs", {})
    if outputs.get("write_full_manifest", False):
        write_csv(FULL_MANIFEST_PATH, rows)
    if outputs.get("write_wide_selected_manifest", False):
        write_csv(WIDE_SELECTED_PATH, selected_rows)


def main() -> None:
    args = parse_args()
    load_dotenv_from_project(PROJECT_DIR)
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(IMG_METADATA_RAW_DIR)

    try:
        config = load_config(args.config)
    except FileNotFoundError as error:
        print(error)
        return

    sample_ids = get_sample_ids(config, args.limit_query_values)
    rows, search_urls, facets = search_all_samples(config, sample_ids, args.force_refresh)
    selected_rows = selected_manifest_rows(rows, config, PROJECT_DIR)
    img_metadata = scrape_img_metadata(sample_ids, config, args.force_refresh)

    send_mail = bool(config.get("payload", {}).get("send_mail", False))
    restore_payload = build_current_payload(selected_rows, send_mail=send_mail)
    download_payload = build_current_payload(selected_rows, send_mail=False)

    write_download_plan(selected_rows)
    write_sample_summary(rows, selected_rows, img_metadata)
    write_json(RESTORE_PAYLOAD_PATH, restore_payload)
    write_json(DOWNLOAD_PAYLOAD_PATH, download_payload)
    write_payload_chunks(selected_rows, config)
    write_optional_wide_csvs(rows, selected_rows, config)
    SEARCH_URLS_PATH.write_text("\n".join(search_urls) + "\n", encoding="utf-8")
    write_json(FACETS_PATH, facets)

    print("\nScript 1 complete.")
    print(f"Samples searched:        {len(sample_ids):,}")
    print(f"Files found:             {len(rows):,}")
    print(f"Files selected:          {len(selected_rows):,}")
    print(f"Download plan:           {PLAN_PATH}")
    print(f"Sample summary:          {SAMPLE_SUMMARY_PATH}")
    print(f"Script 2 payload chunks: {PAYLOAD_CHUNKS_DIR}")


if __name__ == "__main__":
    main()
