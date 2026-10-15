"""Small shared helpers for the mass-ingestion stages."""

from __future__ import annotations

import json
import os
import shutil
import re
from pathlib import Path

from mag_shards.manifest import atomic_write_json, sha256_file


def require_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name}")
    return Path(value)


def atomic_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def load_plan(root: Path) -> dict:
    return json.loads((root / "plan.json").read_text())


def fresh_staging(destination: Path) -> Path:
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)


def task_id() -> int:
    # A worker may process several logical tasks.  Its explicit TASK_ID must
    # therefore override the fixed Slurm array index.
    raw = os.environ.get("TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID"))
    if raw is None:
        raise RuntimeError("set TASK_ID or run as a Slurm array task")
    return int(raw)


def bucket_for_hash(value: bytes, bucket_count: int) -> int:
    if bucket_count <= 0 or bucket_count & (bucket_count - 1):
        raise ValueError("bucket_count must be a power of two")
    return int.from_bytes(value[:4], "big") & (bucket_count - 1)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not result:
        raise ValueError("name contains no safe filename characters")
    return result.lower()


file_sha256 = sha256_file
