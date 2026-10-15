from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the scripts usable without python-dotenv.
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False


SEARCH_HOST = "https://files.jgi.doe.gov"
DOWNLOAD_HOST = "https://files-download.jgi.doe.gov"
SEARCH_ENDPOINTS = {
    "search": f"{SEARCH_HOST}/search/",
    "img_file_list": f"{SEARCH_HOST}/img_file_list/",
    "mycocosm_file_list": f"{SEARCH_HOST}/mycocosm_file_list/",
    "phytozome_file_list": f"{SEARCH_HOST}/phytozome_file_list/",
}
RESTORE_URL = f"{SEARCH_HOST}/request_archived_files/"
DOWNLOAD_URL = f"{DOWNLOAD_HOST}/download_files/"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    try:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not write {path}. Close this CSV if it is open in Excel, "
            "a preview pane, or another process, then rerun the command."
        ) from exc


def load_dotenv_from_project(project_dir: Path) -> None:
    load_dotenv(project_dir / ".env", override=True)


def authorization_header_from_env(required: bool = False) -> Dict[str, str]:
    token = (
        os.getenv("JGI_SESSION_TOKEN")
        or os.getenv("JGI_AUTHORIZATION")
        or os.getenv("JGI_DATA_PORTAL_TOKEN")
        or ""
    ).strip().strip('"').strip("'")
    if required and not token:
        raise EnvironmentError(
            "Set JGI_SESSION_TOKEN in .env or your environment. "
            "In the JGI Data Portal, log in and use the avatar menu's "
            "'Copy My Session Token' action, then paste that value as JGI_SESSION_TOKEN."
        )
    return {"Authorization": token} if token else {}


def jgi_http_error_message(status_code: int, action: str, preview: str = "") -> str:
    if status_code == 401:
        return (
            f"JGI {action} failed with HTTP 401. "
            "Authentication credentials were missing, expired, or invalid. "
            "Refresh JGI_SESSION_TOKEN in .env or your shell, then rerun the command. "
            f"Response preview: {preview!r}"
        )
    if status_code == 409:
        return (
            f"JGI {action} failed with HTTP 409. "
            "This usually means the files are not restored/READY yet, "
            "or JGI rejected the selected file set. "
            f"Response preview: {preview!r}"
        )
    return (
        f"JGI {action} failed with HTTP {status_code}. "
        f"Response preview: {preview!r}"
    )


def parse_search_url(url: str) -> Dict[str, str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in params.items()}


def encode_filter_param(filters: Mapping[str, Any] | str | None) -> str:
    if not filters:
        return ""
    if isinstance(filters, str):
        return filters
    return json.dumps(filters, separators=(",", ":"))


def normalize_search_params(params: Mapping[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in params.items():
        if value is None or value == "":
            continue
        if key in {"ff", "of"} and isinstance(value, Mapping):
            normalized[key] = encode_filter_param(value)
        elif isinstance(value, (list, tuple, dict)):
            normalized[key] = json.dumps(value, separators=(",", ":"))
        else:
            normalized[key] = str(value)
    normalized.setdefault("api_version", "2")
    return normalized


def build_search_params(config: Mapping[str, Any], query_value: Optional[str] = None) -> Dict[str, str]:
    search = config.get("search", {})
    params: Dict[str, Any] = {}
    search_url = clean_text(search.get("search_url"))
    if search_url:
        params.update(parse_search_url(search_url))

    params.update(search.get("params", {}))
    if clean_text(search.get("q")):
        params["q"] = clean_text(search.get("q"))
    if clean_text(search.get("field")):
        params["f"] = clean_text(search.get("field"))
    if query_value is not None:
        params["q"] = query_value
        if clean_text(search.get("query_field")):
            params["f"] = clean_text(search.get("query_field"))
    if search.get("file_filters"):
        params["ff"] = search["file_filters"]
    if search.get("organism_filters"):
        params["of"] = search["organism_filters"]

    params.setdefault("p", "1")
    params.setdefault("x", str(search.get("datasets_per_page", 100)))
    return normalize_search_params(params)


def endpoint_url(config: Mapping[str, Any]) -> str:
    name = clean_text(config.get("search", {}).get("endpoint")) or "search"
    if name.startswith("http://") or name.startswith("https://"):
        return name
    if name not in SEARCH_ENDPOINTS:
        raise ValueError(f"Unknown JGI search endpoint {name!r}. Valid: {', '.join(SEARCH_ENDPOINTS)}")
    return SEARCH_ENDPOINTS[name]


def url_with_params(url: str, params: Mapping[str, str]) -> str:
    return f"{url}?{urlencode(params)}"


def extract_query_values(config: Mapping[str, Any], project_dir: Path) -> List[str]:
    source = config.get("query_values", {})
    explicit = [clean_text(v) for v in source.get("values", []) if clean_text(v)]
    if explicit:
        return list(dict.fromkeys(explicit))

    path_text = clean_text(source.get("path"))
    if not path_text:
        single_q = clean_text(config.get("search", {}).get("q"))
        return [single_q] if single_q else []

    path = Path(path_text)
    if not path.is_absolute():
        path = project_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Query values path does not exist: {path}")

    regex = clean_text(source.get("regex"))
    column = clean_text(source.get("column"))
    values: List[str] = []

    if column:
        import pandas as pd

        table = pd.read_csv(path, sep=None, engine="python", dtype=str)
        if column not in table.columns:
            raise ValueError(f"Column {column!r} was not found in {path}. Available columns: {list(table.columns)}")
        values = [clean_text(v) for v in table[column].tolist()]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if regex:
            compiled = re.compile(regex)
            for match in compiled.finditer(text):
                if "value" in match.groupdict():
                    values.append(clean_text(match.group("value")))
                elif match.groups():
                    values.append(clean_text(match.group(1)))
                else:
                    values.append(clean_text(match.group(0)))
        else:
            values = [clean_text(line) for line in text.splitlines()]

    return list(dict.fromkeys(v for v in values if v))


def extract_values_from_path(
    path_text: str,
    project_dir: Path,
    regex: str = "",
    column: str = "",
) -> List[str]:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Value-list path does not exist: {path}")

    values: List[str] = []
    if column:
        import pandas as pd

        table = pd.read_csv(path, sep=None, engine="python", dtype=str)
        if column not in table.columns:
            raise ValueError(f"Column {column!r} was not found in {path}. Available columns: {list(table.columns)}")
        values = [clean_text(v) for v in table[column].tolist()]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if regex:
            compiled = re.compile(regex)
            for match in compiled.finditer(text):
                if "value" in match.groupdict():
                    values.append(clean_text(match.group("value")))
                elif match.groups():
                    values.append(clean_text(match.group(1)))
                else:
                    values.append(clean_text(match.group(0)))
        else:
            values = [clean_text(line) for line in text.splitlines()]

    return list(dict.fromkeys(v for v in values if v))


def first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def flatten_dict(data: Mapping[str, Any], prefix: str = "", max_depth: int = 2) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_").lower()
        out_key = f"{prefix}{safe_key}" if prefix else safe_key
        if isinstance(value, Mapping) and max_depth > 0:
            flat.update(flatten_dict(value, prefix=f"{out_key}_", max_depth=max_depth - 1))
        elif isinstance(value, (list, tuple)):
            flat[out_key] = json.dumps(value, ensure_ascii=True)
        else:
            flat[out_key] = value
    return flat


def iter_organisms(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    organisms = payload.get("organisms")
    return organisms if isinstance(organisms, list) else []


def top_hit_id(top_hit: Any) -> str:
    if isinstance(top_hit, Mapping):
        return clean_text(first_present(top_hit, ["_id", "id", "file_id"]))
    return clean_text(top_hit)


def build_manifest_rows(payload: Mapping[str, Any], query_value: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for organism in iter_organisms(payload):
        organism_id = clean_text(organism.get("id"))
        agg = clean_text(organism.get("agg"))
        agg_id = clean_text(organism.get("agg_id"))
        top_hit = organism.get("top_hit", {})
        top_hit_file_id = top_hit_id(top_hit)
        mycocosm_portal_id = clean_text(
            first_present(organism, ["mycocosm_portal_id", "mycocosmPortalId"])
            or (top_hit.get("mycocosm_portal_id") if isinstance(top_hit, Mapping) else "")
        )

        files = organism.get("files", [])
        if not isinstance(files, list):
            continue
        if not top_hit_file_id and files and isinstance(files[0], Mapping):
            # Some current search responses omit organisms[].top_hit even though
            # restore/download payload examples still include it. The first file
            # in the grouped dataset is the safest local fallback.
            top_hit_file_id = clean_text(first_present(files[0], ["_id", "id", "file_id"]))
        organism_flat = flatten_dict(
            {k: v for k, v in organism.items() if k not in {"files", "top_hit"}},
            prefix="organism_",
            max_depth=1,
        )
        top_hit_flat = flatten_dict(top_hit, prefix="top_hit_", max_depth=1) if isinstance(top_hit, Mapping) else {}

        for file_obj in files:
            if not isinstance(file_obj, Mapping):
                continue
            metadata = file_obj.get("metadata", {})
            metadata_flat = flatten_dict(metadata, prefix="metadata_", max_depth=2) if isinstance(metadata, Mapping) else {}
            file_id = clean_text(first_present(file_obj, ["_id", "id", "file_id"]))
            file_name = clean_text(first_present(file_obj, ["file_name", "filename", "name"]) or metadata_flat.get("metadata_file_name"))
            row: Dict[str, Any] = {
                "query_value": query_value,
                "organism_id": organism_id,
                "organism_agg": agg,
                "organism_agg_id": agg_id,
                "top_hit_id": top_hit_file_id,
                "mycocosm_portal_id": mycocosm_portal_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_status": clean_text(file_obj.get("file_status")),
            }
            row.update(organism_flat)
            row.update(top_hit_flat)
            row.update(flatten_dict({k: v for k, v in file_obj.items() if k != "metadata"}, prefix="file_", max_depth=1))
            row.update(metadata_flat)
            rows.append(row)
    return rows


def allowed_file_names(local_filters: Mapping[str, Any], project_dir: Path) -> set[str]:
    values = {clean_text(v) for v in local_filters.get("file_names", []) if clean_text(v)}
    path_text = clean_text(local_filters.get("file_names_path"))
    if path_text:
        values.update(
            extract_values_from_path(
                path_text=path_text,
                project_dir=project_dir,
                regex=clean_text(local_filters.get("file_names_regex")),
                column=clean_text(local_filters.get("file_names_column")),
            )
        )
    return values


def file_name_allowed(file_name: str, allowed_names: set[str]) -> bool:
    if not allowed_names:
        return True
    name = clean_text(file_name)
    lowered_allowed = {v.lower() for v in allowed_names}
    lowered_name = name.lower()
    if lowered_name in lowered_allowed:
        return True
    for allowed in lowered_allowed:
        # Accept exact bin archives and common extension-bearing variants like
        # 3300047297_s99.fna without matching names such as 3300047297_s990.
        if lowered_name.startswith(f"{allowed}."):
            return True
    return False


def regex_matches_any(value: str, patterns: Sequence[str]) -> bool:
    text = clean_text(value)
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns if clean_text(pattern))


def list_from_config(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if clean_text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [clean_text(item) for item in value if clean_text(item)]
    return [clean_text(value)] if clean_text(value) else []


def selection_reason(
    row: Mapping[str, Any],
    local_filters: Mapping[str, Any],
    file_name_allowlist: Optional[set[str]] = None,
) -> str:
    file_name = clean_text(row.get("file_name"))
    if file_name_allowlist and file_name_allowed(file_name, file_name_allowlist):
        return "bin_artifact"

    if bool(local_filters.get("include_all_sample_bins", False)):
        sample_id = re.escape(clean_text(row.get("query_value")))
        if sample_id and re.match(rf"^{sample_id}_s\d+\..+", file_name, flags=re.IGNORECASE):
            return "bin_artifact"

    sample_patterns = list_from_config(local_filters.get("sample_file_regex"))
    if regex_matches_any(file_name, sample_patterns):
        return "sample_level_file"
    return ""


def safe_path_component(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "_"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = " ".join(text.split()).strip(" .")
    return text[:180] or "_"


def suggested_local_path(row: Mapping[str, Any], reason: str) -> str:
    sample_id = clean_text(row.get("query_value")) or clean_text(row.get("organism_agg_id")) or clean_text(row.get("organism_id"))
    file_name = clean_text(row.get("file_name")) or clean_text(row.get("file_id"))
    folder = "bins" if reason == "bin_artifact" else "sample_level"
    return f"{safe_path_component(sample_id)}/{folder}/{safe_path_component(file_name)}"


def status_priority(status: Any) -> int:
    normalized = clean_text(status).upper()
    priorities = {
        "READY": 50,
        "RESTORED": 50,
        "BACKUP_COMPLETE": 40,
        "PURGED": 20,
    }
    return priorities.get(normalized, 10)


def parse_size_for_priority(value: Any) -> int:
    try:
        return int(float(clean_text(value)))
    except ValueError:
        return 0


def dedupe_selected_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (clean_text(row.get("query_value")), clean_text(row.get("file_name")).lower())
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        row_score = (
            status_priority(row.get("file_status")),
            parse_size_for_priority(row.get("file_file_size") or row.get("file_size")),
            clean_text(row.get("file_id")),
        )
        current_score = (
            status_priority(current.get("file_status")),
            parse_size_for_priority(current.get("file_file_size") or current.get("file_size")),
            clean_text(current.get("file_id")),
        )
        if row_score > current_score:
            best[key] = row
    return list(best.values())


def row_matches_local_filters(
    row: Mapping[str, Any],
    local_filters: Mapping[str, Any],
    file_name_allowlist: Optional[set[str]] = None,
) -> bool:
    if not selection_reason(row, local_filters, file_name_allowlist):
        return False

    include = clean_text(local_filters.get("file_name_regex"))
    if include and not re.search(include, clean_text(row.get("file_name")), flags=re.IGNORECASE):
        return False

    exclude = clean_text(local_filters.get("exclude_file_name_regex"))
    if exclude and re.search(exclude, clean_text(row.get("file_name")), flags=re.IGNORECASE):
        return False

    statuses = {clean_text(v).upper() for v in local_filters.get("file_status", []) if clean_text(v)}
    if statuses and clean_text(row.get("file_status")).upper() not in statuses:
        return False

    for key, expected in local_filters.get("metadata_equals", {}).items():
        if clean_text(row.get(key)) != clean_text(expected):
            return False

    for key, expected in local_filters.get("metadata_contains", {}).items():
        if clean_text(expected).lower() not in clean_text(row.get(key)).lower():
            return False

    return True


def selected_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    project_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    filters = config.get("local_filters", {})
    allowlist = allowed_file_names(filters, project_dir or Path(".").resolve())
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if clean_text(row.get("file_id")) and row_matches_local_filters(row, filters, allowlist):
            selected_row = dict(row)
            reason = selection_reason(selected_row, filters, allowlist)
            selected_row["selection_reason"] = reason
            selected_row["suggested_local_path"] = suggested_local_path(selected_row, reason)
            selected.append(selected_row)
    if bool(filters.get("dedupe_by_query_and_file_name", False)):
        selected = dedupe_selected_rows(selected)
    return selected


def build_current_payload(rows: Sequence[Mapping[str, Any]], send_mail: bool = False) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        organism_id = clean_text(row.get("organism_id"))
        file_id = clean_text(row.get("file_id"))
        if not organism_id or not file_id:
            continue
        entry = grouped.setdefault(
            organism_id,
            {
                "file_ids": [],
                "top_hit": clean_text(row.get("top_hit_id")),
            },
        )
        if file_id not in entry["file_ids"]:
            entry["file_ids"].append(file_id)
        mycocosm_portal_id = clean_text(row.get("mycocosm_portal_id"))
        if mycocosm_portal_id:
            entry["mycocosm_portal_id"] = mycocosm_portal_id
        if not clean_text(entry.get("top_hit")) and clean_text(row.get("top_hit_id")):
            entry["top_hit"] = clean_text(row.get("top_hit_id"))
    return {"ids": grouped, "send_mail": bool(send_mail), "api_version": "2"}


def build_legacy_download_payload(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids: Dict[str, List[str]] = {}
    for row in rows:
        organism_id = clean_text(row.get("organism_id"))
        file_id = clean_text(row.get("file_id"))
        if organism_id and file_id:
            ids.setdefault(organism_id, [])
            if file_id not in ids[organism_id]:
                ids[organism_id].append(file_id)
    return {"ids": ids, "api_version": "2"}


def payload_file_count(payload: Mapping[str, Any]) -> int:
    total = 0
    ids = payload.get("ids", {})
    if not isinstance(ids, Mapping):
        return 0
    for value in ids.values():
        if isinstance(value, Mapping):
            total += len(value.get("file_ids", []))
        elif isinstance(value, list):
            total += len(value)
    return total


def split_rows_for_payload_limit(
    rows: Sequence[Mapping[str, Any]],
    send_mail: bool = False,
    max_chars: int = 3900,
) -> List[List[Mapping[str, Any]]]:
    chunks: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    for row in rows:
        trial = current + [row]
        trial_payload = build_current_payload(trial, send_mail=send_mail)
        trial_size = len(json.dumps(trial_payload, separators=(",", ":")))
        if current and trial_size > max_chars:
            chunks.append(current)
            current = [row]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks


@dataclass
class JGIDataPortalClient:
    authorization: Optional[str] = None
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        self.session = requests.Session()
        if self.authorization:
            self.session.headers.update({"Authorization": self.authorization})
        self.session.headers.update({"Accept": "application/json"})

    @classmethod
    def from_env(cls, required_auth: bool = False, timeout_seconds: int = 120) -> "JGIDataPortalClient":
        header = authorization_header_from_env(required=required_auth)
        return cls(authorization=header.get("Authorization"), timeout_seconds=timeout_seconds)

    def get_json(self, url: str, params: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            preview = response.text[:500] if response.text else ""
            raise RuntimeError(jgi_http_error_message(response.status_code, "POST", preview)) from exc
        if not response.text.strip():
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}

    def get_status(self, status_url: str) -> Dict[str, Any]:
        return self.get_json(status_url)

    def stream_download_zip(
        self,
        payload: Mapping[str, Any],
        out_path: Path,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> int:
        ensure_dir(out_path.parent)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        bytes_written = 0
        with self.session.post(
            DOWNLOAD_URL,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            stream=True,
            timeout=self.timeout_seconds,
        ) as response:
            total_bytes = int(response.headers.get("Content-Length", "0") or 0) or None
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                preview = response.text[:1000] if response.text else ""
                raise RuntimeError(
                    jgi_http_error_message(response.status_code, "download", preview)
                ) from exc
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bytes_written += len(chunk)
                        if progress_callback:
                            progress_callback(bytes_written, total_bytes)
        if bytes_written == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("Download endpoint returned a zero-byte response.")
        tmp_path.replace(out_path)
        return bytes_written


def poll_status_until_ready(
    client: JGIDataPortalClient,
    status_url: str,
    poll_interval_seconds: int,
    max_minutes: int,
) -> Dict[str, Any]:
    deadline = time.time() + max_minutes * 60
    last: Dict[str, Any] = {}
    while time.time() <= deadline:
        last = client.get_status(status_url)
        status = clean_text(last.get("status")).upper()
        if status in {"READY", "EXPIRED", "FAILED", "ERROR"}:
            return last
        time.sleep(poll_interval_seconds)
    return last
