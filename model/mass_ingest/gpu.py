"""GPU stages: ESM-C embedding followed by frozen VQ assignment."""

from __future__ import annotations

import gzip
import json
import os
import signal
from pathlib import Path

import numpy as np

from mag_shards.esmc_inference import ESMCEmbedder
from mag_shards.ingest import load_frozen_vqvae
from mag_shards.schema import ESMC_DIM
from mass_ingest.common import atomic_json, file_sha256, load_plan, require_path, task_id
from tokenization import NUM_SPECIAL, mode_dir


STOP_REQUESTED = False


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGUSR1, request_stop)


def read_sequences(root: Path, bucket: int) -> list[str]:
    path = root / "proteins" / f"bucket-{bucket:04d}" / "sequences.txt.gz"
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle]


def embed() -> None:
    import torch

    root = require_path("MASS_ROOT")
    bucket = task_id()
    source = root / "proteins" / f"bucket-{bucket:04d}"
    destination = root / "embeddings" / f"bucket-{bucket:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETE").exists():
        print(f"embedding bucket {bucket} already complete")
        return
    sequences = read_sequences(root, bucket)
    count = len(sequences)
    output = np.memmap(destination / "esmc.f16", dtype="<f2", mode="r+" if (
        destination / "esmc.f16").exists() else "w+", shape=(count, ESMC_DIM))
    state_path = destination / "state.json"
    completed = json.loads(state_path.read_text()).get("completed", 0) if state_path.exists() else 0
    chunk_size = int(os.environ.get("EMBED_CHUNK", "4096"))
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if memory_gib < 12:
        token_budget = 8192
    elif memory_gib < 20:
        token_budget = 16384
    elif memory_gib < 36:
        token_budget = 32768
    else:
        token_budget = 131072
    print(f"[embed] bucket={bucket} gpu_memory_gib={memory_gib:.1f} token_budget={token_budget}", flush=True)
    embedder = ESMCEmbedder(token_budget=token_budget)
    try:
        while completed < count:
            end = min(count, completed + chunk_size)
            output[completed:end] = embedder(sequences[completed:end])
            output.flush()
            completed = end
            atomic_json(state_path, {"bucket": bucket, "completed": completed, "total": count})
            print(f"[embed] bucket={bucket} {completed}/{count}", flush=True)
            if STOP_REQUESTED:
                print("[embed] preemption warning received; checkpoint is current", flush=True)
    finally:
        embedder.close()
    report = {
        "bucket": bucket,
        "proteins": count,
        "embedding_dim": ESMC_DIM,
        "dtype": "float16",
        "sha256": file_sha256(destination / "esmc.f16"),
    }
    atomic_json(destination / "report.json", report)
    (destination / "COMPLETE").write_text("complete\n")
    print(json.dumps(report))


class VQAssigner:
    def __init__(self):
        import torch

        self.torch = torch
        self.device = "cuda"
        self.kmeans = torch.from_numpy(np.load(mode_dir("vq_kmeans") / "codebook.npy").astype(
            np.float32)).to(self.device)
        self.kmeans_norm = self.kmeans.pow(2).sum(1)
        self.vae = load_frozen_vqvae(ESMC_DIM, self.device)

    def kmeans_ids(self, values: np.ndarray) -> np.ndarray:
        output = []
        with self.torch.no_grad():
            for vector_start in range(0, len(values), 1024):
                vectors = self.torch.from_numpy(np.asarray(
                    values[vector_start:vector_start + 1024], dtype=np.float32)).to(self.device)
                vector_norm = vectors.pow(2).sum(1)
                best = self.torch.full((len(vectors),), float("inf"), device=self.device)
                best_id = self.torch.zeros(len(vectors), dtype=self.torch.long, device=self.device)
                for code_start in range(0, len(self.kmeans), 4096):
                    block = self.kmeans[code_start:code_start + 4096]
                    distance = (vector_norm[:, None] - 2 * vectors @ block.T
                                + self.kmeans_norm[code_start:code_start + len(block)])
                    value, local = distance.min(1)
                    update = value < best
                    best = self.torch.where(update, value, best)
                    best_id = self.torch.where(update, local + code_start, best_id)
                output.append(best_id.cpu().numpy())
        return (np.concatenate(output) + NUM_SPECIAL).astype(np.uint32)

    def vae_ids(self, values: np.ndarray) -> np.ndarray:
        result = []
        with self.torch.no_grad():
            for start in range(0, len(values), 2048):
                tensor = self.torch.from_numpy(np.asarray(values[start:start + 2048], dtype=np.float32)).to(
                    self.device)
                result.append(self.vae.nearest(self.vae.enc(tensor)).cpu().numpy())
        return (np.concatenate(result) + NUM_SPECIAL).astype(np.uint32)


def assign_vq() -> None:
    root = require_path("MASS_ROOT")
    bucket = task_id()
    embedding_dir = root / "embeddings" / f"bucket-{bucket:04d}"
    if not (embedding_dir / "COMPLETE").exists():
        raise RuntimeError(f"embedding bucket {bucket} is incomplete")
    report = json.loads((embedding_dir / "report.json").read_text())
    count = int(report["proteins"])
    destination = root / "vq" / f"bucket-{bucket:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETE").exists():
        print(f"VQ bucket {bucket} already complete")
        return
    embeddings = np.memmap(embedding_dir / "esmc.f16", dtype="<f2", mode="r", shape=(count, ESMC_DIM))
    kmeans = np.memmap(destination / "vq_kmeans.u32", dtype="<u4", mode="r+" if (
        destination / "vq_kmeans.u32").exists() else "w+", shape=(count,))
    vae = np.memmap(destination / "vq_vae.u32", dtype="<u4", mode="r+" if (
        destination / "vq_vae.u32").exists() else "w+", shape=(count,))
    state_path = destination / "state.json"
    completed = json.loads(state_path.read_text()).get("completed", 0) if state_path.exists() else 0
    chunk_size = int(os.environ.get("VQ_CHUNK", "8192"))
    assigner = VQAssigner()
    while completed < count:
        end = min(count, completed + chunk_size)
        values = np.asarray(embeddings[completed:end], dtype=np.float16)
        kmeans[completed:end] = assigner.kmeans_ids(values)
        vae[completed:end] = assigner.vae_ids(values)
        kmeans.flush(); vae.flush()
        completed = end
        atomic_json(state_path, {"bucket": bucket, "completed": completed, "total": count})
        print(f"[vq] bucket={bucket} {completed}/{count}", flush=True)
        if STOP_REQUESTED:
            print("[vq] preemption warning received; checkpoint is current", flush=True)
    result = {
        "bucket": bucket,
        "proteins": count,
        "vq_kmeans_sha256": file_sha256(destination / "vq_kmeans.u32"),
        "vq_vae_sha256": file_sha256(destination / "vq_vae.u32"),
    }
    atomic_json(destination / "report.json", result)
    (destination / "COMPLETE").write_text("complete\n")
    print(json.dumps(result))


def main(stage: str | None = None) -> None:
    stage = stage or os.environ.get("MODE", "")
    actions = {"embed": embed, "vq": assign_vq}
    if stage not in actions:
        raise RuntimeError("stage must be embed or vq")
    actions[stage]()


if __name__ == "__main__":
    main()
