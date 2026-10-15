"""Distributed initialization, signals, RNG state, and atomic checkpoints."""

from __future__ import annotations

import os
import random
import signal
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def distributed_init(timeout_seconds: int = 600):
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        rank = int(os.environ["SLURM_PROCID"]); local = int(os.environ["SLURM_LOCALID"])
        world = int(os.environ["SLURM_NTASKS"])
        os.environ.setdefault("MASTER_PORT", "29500")
        if "MASTER_ADDR" not in os.environ:
            nodes = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]]).split()
            os.environ["MASTER_ADDR"] = nodes[0].decode()
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local))
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
    else:
        return 0, 0, 1
    device_index = local if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(device_index)
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=timedelta(seconds=int(timeout_seconds)))
    return rank, device_index, world


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def reduce_sum_count(total: float, count: int, device: str):
    value = torch.tensor([float(total), float(count)], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value[0]), int(value[1])


def any_rank_true(value: bool, device: str) -> bool:
    flag = torch.tensor([int(value)], dtype=torch.uint8, device=device)
    if is_distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


class StopController:
    def __init__(self):
        self.requested = False
        self.signal_name = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame):
        self.requested = True
        self.signal_name = signal.Signals(signum).name


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state) -> None:
    if not state:
        return
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device for model/optimizer
    # efficiency, which also moves these serialized byte tensors to CUDA.
    # PyTorch RNG restoration APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].detach().cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([
            value.detach().cpu().to(torch.uint8) for value in state["torch_cuda"]])


def collect_rng_states(rank: int, world: int):
    local = capture_rng_state()
    if world == 1:
        return [local]
    gathered = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def atomic_torch_save(value, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def hardware_summary(world: int):
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    memory = (torch.cuda.get_device_properties(0).total_memory
              if torch.cuda.is_available() else 0)
    return {"world_size": world, "nodes": nodes, "gpus_per_node": world // max(1, nodes),
            "gpu_name": gpu, "gpu_memory_bytes": memory,
            "nodelist": os.environ.get("SLURM_NODELIST", "")}
