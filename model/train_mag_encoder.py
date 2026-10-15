"""Train the MAG encoder from one immutable shard manifest.

Storage, collation, sampling, model construction, losses, and distributed
runtime live in focused modules.  This file owns only run orchestration.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mag_shards.collate import move_batch_to_device
from losses import patch_disc_metrics
from optimizers import MuonAdamW
from mag_shards.manifest import atomic_write_bytes, atomic_write_json
from mag_shards.reader import ManifestDataset
from split_plans import load_split_plan, materialize_split_plan
from train_config import TrainConfig
from train_runtime import (StopController, any_rank_true, atomic_torch_save,
                           collect_rng_states, distributed_init,
                           hardware_summary, is_distributed, reduce_sum_count,
                           restore_rng_state, seed_everything, unwrap)
from training import (build_loader, build_model, calibrate_batch_tokens,
                      compute_losses, learning_rate, vocabulary_sizes)

HARDWARE_PROFILE = os.environ.get("TRAIN_PROFILE", "default")
FORCE_NO_RESUME = os.environ.get("TRAIN_NO_RESUME", "0") == "1"
STOP_FILE = os.environ.get("TRAIN_STOP_FILE")
STOP_ACK_FILE = os.environ.get("TRAIN_STOP_ACK_FILE")


def _load_experiment(hardware_profile: str) -> tuple[TrainConfig, Path]:
    selected = os.environ.get("TRAIN_EXPERIMENT")
    if not selected:
        raise RuntimeError("TRAIN_EXPERIMENT must name a file in model/experiments")
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("training experiment does not exist: {}".format(path))
    module_name = "mag_experiment_{}".format(
        hashlib.sha256(str(path).encode()).hexdigest()[:12]
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load training experiment: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_config = getattr(module, "get_config", None)
    if get_config is None:
        raise AttributeError("experiment must define get_config(profile): {}".format(path))
    return get_config(hardware_profile), path


def _amp_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[name]


def _batch_plan_signature(config: TrainConfig, world: int, per_device_tokens: int):
    return {
        "automatic": config.hardware.auto_batch_tokens,
        "per_device_batch_tokens": per_device_tokens,
        "max_per_device_batch_tokens": config.hardware.max_per_device_batch_tokens,
        "effective_global_batch_tokens": per_device_tokens * world,
        "max_batch_mags": config.batch.max_batch_mags,
        "per_mag_token_budget": config.batch.per_mag_token_budget,
        "resolved_max_mag_tokens": config.batch.resolved_max_mag_tokens,
        "max_contig_genes": config.batch.max_contig_genes,
        "max_contigs_per_mag": config.batch.max_contigs_per_mag,
        "contig_dropout_range": config.batch.contig_dropout_range,
        "contig_dropout_apply_prob": config.batch.contig_dropout_apply_prob,
        "bucket_window": config.batch.bucket_window,
        "seed": config.runtime.seed,
        "world_size": world,
    }


def _global_mean(value: float, device: str):
    total, count = reduce_sum_count(value, 1, device)
    return total / max(1, count)


def _global_max(value: float, device: str):
    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


@torch.no_grad()
def _validate(model, loader, sampler, config, device, max_batches=None):
    sampler.set_epoch(0, 0)
    model.eval(); total = 0.0; count = 0
    amp_dtype = _amp_dtype(config.optim.amp_dtype)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        autocast = (torch.autocast("cuda", dtype=amp_dtype)
                    if amp_dtype is not None and device != "cpu" else contextlib.nullcontext())
        with autocast:
            loss, _ = compute_losses(model(batch), batch, config)
        total += float(loss); count += 1
    total, count = reduce_sum_count(total, count, device)
    model.train()
    return total / max(1, count), count


def _checkpoint_payload(model, optimizer, scaler, config, train_data, split_plan, data_summaries,
                        run_identity, batch_plan,
                        rank, world,
                        step, epoch, next_global_batch, max_steps, best_val, best_step, best_epoch,
                        gpu_seconds, rng_states):
    return {
        "format_version": 3,
        "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "step": int(step), "epoch": int(epoch),
        "next_global_batch": int(next_global_batch), "max_steps": int(max_steps),
        "best_val": float(best_val), "best_step": int(best_step),
        "best_epoch": int(best_epoch),
        "mode": config.mode, "loss_weights": dict(config.objectives.weights),
        "head_spec": config.input_spec.to_dict(), "config": config.to_dict(),
        "scientific_sha256": config.scientific_sha256,
        "manifest_path": str(train_data.manifest_path.resolve()),
        "manifest_sha256": train_data.manifest_sha256,
        "split_plan_path": str(split_plan.path.resolve()),
        "split_plan_sha256": split_plan.sha256,
        "assignment_table_sha256": split_plan.table_sha256,
        "data_summaries": data_summaries,
        "run_identity": run_identity,
        "batch_plan": batch_plan,
        "world_size": int(world), "gpu_seconds": float(gpu_seconds),
        "rng_states": rng_states,
    }


def _save_checkpoint(path, model, optimizer, scaler, config, train_data, split_plan, data_summaries,
                     run_identity, batch_plan,
                     rank, world,
                     step, epoch, next_global_batch, max_steps, best_val, best_step, best_epoch,
                     gpu_seconds):
    rng_states = collect_rng_states(rank, world)
    if rank == 0:
        payload = _checkpoint_payload(
            model, optimizer, scaler, config, train_data, split_plan, data_summaries,
            run_identity, batch_plan,
            rank, world, step, epoch,
            next_global_batch, max_steps, best_val, best_step, best_epoch, gpu_seconds, rng_states)
        atomic_torch_save(payload, path)
    if is_distributed():
        dist.barrier()


def _stop_file_requested(stop_file: str | None) -> bool:
    return bool(stop_file) and Path(stop_file).is_file()


def _early_stopping_due(config: TrainConfig, step: int, completed_epochs: int,
                        best_step: int, best_epoch: int, max_steps: int) -> bool:
    fraction = config.optim.early_stopping_patience_fraction
    if fraction is None:
        return False
    if config.optim.target_epochs is not None:
        patience = max(1, math.ceil(config.optim.target_epochs * fraction))
        return completed_epochs - best_epoch >= patience
    patience = max(1, math.ceil(max_steps * fraction))
    return step - best_step >= patience


def _identity_key(config, manifest_sha, split_plan_sha):
    value = "{}:{}:{}".format(config.scientific_sha256, manifest_sha, split_plan_sha)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _wandb_start(config, identity_path, manifest_sha, split_plan, data_summaries,
                 nparams, max_steps, hardware, batch_plan, force_new=False):
    if config.runtime.wandb_mode == "disabled":
        name = "local-{}".format(config.scientific_sha256[:12])
        return None, {"id": None, "name": name, "mode": "disabled"}
    import wandb
    values = config.to_dict()
    values.update(
        manifest_sha256=manifest_sha,
        split_plan_sha256=split_plan.sha256,
        assignment_table_sha256=split_plan.table_sha256,
        data_summaries=data_summaries,
        nparams=nparams,
        max_steps=max_steps,
        batch_plan=batch_plan,
        effective_global_batch_tokens=batch_plan["effective_global_batch_tokens"],
        resolved_max_mag_tokens=config.batch.resolved_max_mag_tokens,
        **hardware,
    )
    prior = None
    if config.runtime.resume and not force_new and identity_path.exists():
        candidate = json.loads(identity_path.read_text())
        checkpoint = Path(config.runtime.checkpoint_root) / candidate["name"] / "last.pt"
        if checkpoint.exists():
            prior = candidate
    kwargs = {"id": prior["id"], "resume": "must"} if prior else {}
    run = wandb.init(
        project=config.runtime.wandb_project,
        entity=config.runtime.wandb_entity,
        mode=config.runtime.wandb_mode,
        config=values,
        **kwargs,
    )
    identity = {"id": run.id, "name": run.name, "mode": config.runtime.wandb_mode}
    atomic_write_json(identity_path, identity)
    wandb.define_metric("step"); wandb.define_metric("*", step_metric="step")
    return run, identity


def main(config: TrainConfig | None = None, hardware_profile=HARDWARE_PROFILE,
         force_no_resume=FORCE_NO_RESUME):
    experiment_path = None
    if config is None:
        config, experiment_path = _load_experiment(hardware_profile)
    config.validate()
    rank, local_rank, world = distributed_init(config.runtime.process_group_timeout_seconds)
    device = "cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu"
    main_process = rank == 0
    seed_everything(config.runtime.seed, rank)
    manifest_path = Path(config.data.manifest)
    from mag_shards.manifest import load_manifest
    manifest_sha = load_manifest(manifest_path).sha256
    if main_process:
        split_path = materialize_split_plan(
            manifest_path, config.data.split,
            Path(config.runtime.checkpoint_root) / "split_plans",
            config.data.root_overrides,
        )
        split_path_value = str(split_path.resolve())
    else:
        split_path_value = None
    if is_distributed():
        values = [split_path_value]
        dist.broadcast_object_list(values, src=0)
        split_path_value = values[0]
    split_plan = load_split_plan(Path(split_path_value), manifest_sha)
    catalog = ManifestDataset(
        manifest_path, config.input_spec, root_overrides=config.data.root_overrides,
        assignments=split_plan.assignments)
    train_data = catalog.select(config.data.train.labels, config.data.train.sources)
    val_data = (catalog.select(config.data.validation.labels, config.data.validation.sources)
                if config.runtime.validate else None)
    test_data = catalog.select(config.data.test.labels, config.data.test.sources)
    data_summaries = {
        "train": train_data.summary(),
        "validation": val_data.summary() if val_data is not None else None,
        "test": test_data.summary(),
    }
    token_size, ko_size, ec_size, tax_sizes = vocabulary_sizes(train_data, config)
    model = build_model(config, token_size, ko_size, ec_size, tax_sizes, device)

    hardware = hardware_summary(world)
    identity_key = _identity_key(config, train_data.manifest_sha256, split_plan.sha256)
    identity_path = Path(config.runtime.checkpoint_root) / ".wandb_runs" / "{}.json".format(identity_key)
    calibration_path = identity_path.with_name("{}.batch_plan.json".format(identity_key))
    calibration_key = {
        "calibration_version": 2,
        "gpu_name": hardware["gpu_name"],
        "gpu_memory_bytes": hardware["gpu_memory_bytes"],
        "world_size": world,
        "memory_fraction": config.hardware.gpu_memory_fraction,
        "search_step": config.hardware.batch_token_search_step,
        "initial_tokens": config.hardware.per_device_batch_tokens,
        "maximum_tokens": config.hardware.max_per_device_batch_tokens,
    }
    cached = None
    if main_process and config.hardware.auto_batch_tokens and calibration_path.exists():
        candidate = json.loads(calibration_path.read_text())
        if candidate.get("calibration_key") == calibration_key:
            cached = candidate
    if is_distributed():
        values = [cached]
        dist.broadcast_object_list(values, src=0)
        cached = values[0]
    if config.hardware.auto_batch_tokens:
        if cached is not None:
            safe_per_device_tokens = int(cached["safe_per_device_batch_tokens"])
            calibration_peak = int(cached.get("calibration_peak_bytes", 0))
            if main_process:
                print("[batch-calibration] reusing {} safe tokens per GPU".format(
                    safe_per_device_tokens), flush=True)
        else:
            safe_per_device_tokens, calibration_peak = calibrate_batch_tokens(
                model, train_data, config, rank, world, device)
            if main_process:
                atomic_write_json(calibration_path, {
                    "calibration_key": calibration_key,
                    "safe_per_device_batch_tokens": safe_per_device_tokens,
                    "calibration_peak_bytes": calibration_peak,
                })
    else:
        safe_per_device_tokens = config.hardware.per_device_batch_tokens
        calibration_peak = 0
    per_device_tokens = safe_per_device_tokens
    effective_global_tokens = per_device_tokens * world
    batch_plan = _batch_plan_signature(config, world, per_device_tokens)
    batch_plan.update(
        safe_per_device_batch_tokens=safe_per_device_tokens,
        calibration_peak_bytes=calibration_peak,
        gpu_memory_fraction=config.hardware.gpu_memory_fraction,
    )
    # Probing performs real backward passes; restore the training RNG stream.
    seed_everything(config.runtime.seed, rank)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
    optimizer = (MuonAdamW(model, config.optim.lr, config.optim.weight_decay)
                 if config.optim.optimizer == "muon" else
                 torch.optim.AdamW(model.parameters(), lr=config.optim.lr,
                                   weight_decay=config.optim.weight_decay))
    amp_dtype = _amp_dtype(config.optim.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device != "cpu"))
    train_loader, train_sampler = build_loader(
        train_data, config, rank, world, train=True,
        per_device_batch_tokens=per_device_tokens)
    if val_data is not None:
        val_loader, val_sampler = build_loader(
            val_data, config, rank, world, train=False,
            per_device_batch_tokens=per_device_tokens)
    else:
        val_loader = val_sampler = None

    optimizer_steps_per_epoch = len(train_sampler)
    if optimizer_steps_per_epoch <= 0:
        raise RuntimeError("dataset/profile produces fewer batches than one optimizer update")
    max_steps = (optimizer_steps_per_epoch * config.optim.target_epochs
                 if config.optim.target_epochs is not None else config.optim.max_steps)
    nparams = sum(parameter.numel() for parameter in unwrap(model).parameters())
    if main_process:
        wandb_run, run_identity = _wandb_start(
            config, identity_path, train_data.manifest_sha256, split_plan,
            data_summaries, nparams, max_steps, hardware, batch_plan,
            force_new=force_no_resume,
        )
    else:
        wandb_run = None
        run_identity = None
    if is_distributed():
        values = [run_identity]
        dist.broadcast_object_list(values, src=0)
        run_identity = values[0]
    run_name = run_identity["name"]
    run_dir = Path(config.runtime.checkpoint_root) / run_name
    if main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        if experiment_path is not None:
            experiment_bytes = experiment_path.read_bytes()
            snapshot_path = run_dir / "experiment.py"
            if snapshot_path.exists() and snapshot_path.read_bytes() != experiment_bytes:
                raise RuntimeError(
                    "pinned experiment differs from existing run snapshot: {}".format(snapshot_path)
                )
            atomic_write_bytes(snapshot_path, experiment_bytes)
            atomic_write_json(run_dir / "experiment_source.json", {
                "sha256": hashlib.sha256(experiment_bytes).hexdigest(),
                "source_path": str(experiment_path),
            })
    last_path = run_dir / "last.pt"; best_path = run_dir / "best.pt"

    step = epoch = next_global_batch = best_step = best_epoch = 0
    best_val = float("inf"); prior_gpu_seconds = 0.0
    do_resume = config.runtime.resume and not force_no_resume and last_path.exists()
    if do_resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("manifest_sha256") != train_data.manifest_sha256:
            raise RuntimeError("checkpoint manifest differs from configured immutable dataset")
        if checkpoint.get("scientific_sha256") != config.scientific_sha256:
            raise RuntimeError("checkpoint scientific configuration differs; use an explicit warm start")
        unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint.get("scaler", {}))
        step = int(checkpoint["step"]); epoch = int(checkpoint.get("epoch", 0))
        next_global_batch = int(checkpoint.get("next_global_batch", 0))
        best_val = float(checkpoint.get("best_val", best_val)); best_step = int(checkpoint.get("best_step", step))
        best_epoch = int(checkpoint.get("best_epoch", epoch))
        prior_gpu_seconds = float(checkpoint.get("gpu_seconds", 0.0))
        max_steps = int(checkpoint.get("max_steps", max_steps))
        states = checkpoint.get("rng_states") or []
        if rank < len(states):
            restore_rng_state(states[rank])
        if checkpoint.get("batch_plan") != batch_plan and next_global_batch:
            if main_process:
                print("[resume] hardware batch plan changed; advancing to the next epoch boundary", flush=True)
            epoch += 1; next_global_batch = 0
        if main_process:
            print("[resume] step={} epoch={} next_global_batch={} prior_world={} current_world={}".format(
                step, epoch, next_global_batch, checkpoint.get("world_size"), world), flush=True)

    if main_process:
        print("[run] {} | params={:,} manifest={} split_plan={} train={} val={} test={} gpus={} per_gpu_tokens={} global_tokens={} max_mag_tokens={} steps_per_epoch={} target_epochs={} max_steps={}".format(
            run_name, nparams, train_data.manifest_sha256, split_plan.sha256,
            len(train_data), len(val_data) if val_data is not None else 0, len(test_data),
            world, per_device_tokens, effective_global_tokens,
            config.batch.resolved_max_mag_tokens, optimizer_steps_per_epoch,
            config.optim.target_epochs, max_steps), flush=True)
        print("[data] {}".format(json.dumps(data_summaries, sort_keys=True)), flush=True)
        print("[columns] {}".format(config.input_spec.required_files), flush=True)
        (run_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        atomic_write_json(run_dir / "resolved_batch_plan.json", batch_plan)
        atomic_write_json(run_dir / "run_identity.json", run_identity)
    stop = StopController(); wall_start = time.time(); log_start = time.time()
    interval_loss = 0.0
    interval_batches = interval_genes = interval_tokens = interval_padded = 0
    interval_parts = {}
    model.train(); optimizer.zero_grad(set_to_none=True)
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    last_validation_step = -1

    while step < max_steps:
        train_sampler.set_epoch(epoch, next_global_batch)
        epoch_batches = len(train_sampler)
        processed_batches = 0
        data_ready = time.time()
        for host_batch in train_loader:
            processed_batches += 1
            data_wait = time.time() - data_ready
            batch = move_batch_to_device(host_batch, device)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, max_steps, config.optim.warmup_steps, config.optim.lr)
            autocast = (torch.autocast("cuda", dtype=amp_dtype)
                        if amp_dtype is not None and device != "cpu" else contextlib.nullcontext())
            diagnostics = {}
            with autocast:
                loss, parts = compute_losses(model(batch), batch, config, diagnostics=diagnostics)
            scaler.scale(loss).backward()
            next_global_batch += world
            interval_loss += float(loss.detach()); interval_batches += 1
            interval_genes += int(host_batch["n_genes"])
            interval_tokens += int(host_batch["token_type"].ge(0).sum())
            interval_padded += int(host_batch["token_type"].numel())
            interval_parts["data_wait_seconds"] = interval_parts.get("data_wait_seconds", 0.0) + data_wait
            for name, value in parts.items():
                interval_parts[name] = interval_parts.get(name, 0.0) + float(value.detach())
            if config.optim.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            step += 1
            step_metrics = {}
            if diagnostics:
                if is_distributed():
                    dist.all_reduce(diagnostics["sums"])
                step_metrics = patch_disc_metrics(diagnostics["sums"])
                if main_process:
                    with (run_dir / "patch_disc_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, **step_metrics}) + "\n")
                    if wandb_run and step % config.runtime.log_every != 0:
                        wandb_run.log({"step": step, **step_metrics})
            if _stop_file_requested(STOP_FILE):
                stop.requested = True
                stop.signal_name = stop.signal_name or "STOP_FILE"
            should_stop = any_rank_true(stop.requested or step >= max_steps, device)

            if step % config.runtime.log_every == 0:
                elapsed = max(1e-9, time.time() - log_start)
                mean_loss = _global_mean(interval_loss / max(1, interval_batches), device)
                genes, _ = reduce_sum_count(interval_genes, 0, device)
                tokens, _ = reduce_sum_count(interval_tokens, 0, device)
                padded, _ = reduce_sum_count(interval_padded, 0, device)
                metrics = {"train/loss": mean_loss, "train/genes_per_second": genes / elapsed,
                           "train/data_wait_fraction": interval_parts.get("data_wait_seconds", 0.0) / elapsed,
                           "train/observed_tokens_per_update": tokens / max(1, interval_batches),
                           "train/effective_tokens_per_update": effective_global_tokens,
                           "train/padding_efficiency": tokens / max(1, padded),
                           "lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "step": step}
                metrics.update(step_metrics)
                if device != "cpu":
                    metrics["train/peak_cuda_allocated_gib"] = _global_max(
                        torch.cuda.max_memory_allocated(device) / 2 ** 30, device)
                    metrics["train/peak_cuda_reserved_gib"] = _global_max(
                        torch.cuda.max_memory_reserved(device) / 2 ** 30, device)
                for name in parts:
                    metrics["train/{}".format(name)] = _global_mean(
                        interval_parts.get(name, 0.0) / max(1, interval_batches), device)
                if main_process:
                    print("[{}] loss={:.4f} genes/s={:.0f} data_wait={:.1%} lr={:.2e}".format(
                        step, mean_loss, metrics["train/genes_per_second"],
                        metrics["train/data_wait_fraction"], metrics["lr"]), flush=True)
                    if wandb_run:
                        wandb_run.log(metrics)
                interval_loss = 0.0
                interval_batches = interval_genes = interval_tokens = interval_padded = 0
                interval_parts = {}
                log_start = time.time()
                if device != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)

            improved = False
            early_stopped = False
            if (config.runtime.validate and config.runtime.val_every > 0
                    and step % config.runtime.val_every == 0):
                value, val_batches = _validate(
                    unwrap(model), val_loader, val_sampler, config, device,
                    config.runtime.max_val_batches
                )
                last_validation_step = step
                improved = value < best_val
                if improved:
                    best_val = value; best_step = step
                    best_epoch = epoch
                if config.optim.target_epochs is None:
                    early_stopped = _early_stopping_due(
                        config, step, epoch, best_step, best_epoch, max_steps)
                    should_stop = should_stop or early_stopped
                if main_process:
                    print("[val] step={} loss={:.4f} batches={} mags={} digest={} best={:.4f}{}".format(
                        step, value, val_batches, data_summaries["validation"]["mags"],
                        data_summaries["validation"]["digest"], best_val,
                        " *" if improved else ""), flush=True)
                    if wandb_run:
                        wandb_run.log({"val/loss": value, "val/best": best_val, "step": step})
            checkpoint_due = improved or should_stop
            if checkpoint_due:
                gpu_seconds = prior_gpu_seconds + world * (time.time() - wall_start)
                _save_checkpoint(last_path, model, optimizer, scaler, config, train_data,
                                 split_plan, data_summaries, run_identity, batch_plan,
                                 rank, world,
                                 step, epoch, next_global_batch, max_steps, best_val, best_step,
                                 best_epoch, gpu_seconds)
                if improved:
                    _save_checkpoint(best_path, model, optimizer, scaler, config, train_data,
                                     split_plan, data_summaries, run_identity, batch_plan,
                                     rank, world,
                                     step, epoch, next_global_batch, max_steps, best_val, best_step,
                                     best_epoch, gpu_seconds)
            if early_stopped and main_process:
                print("[early-stop] no validation improvement within the step patience", flush=True)
            if should_stop:
                break
            data_ready = time.time()

        epoch_complete = processed_batches >= epoch_batches
        if epoch_complete:
            completed_epoch = epoch + 1
            epoch += 1; next_global_batch = 0
            improved = False
            validated_epoch = False
            if config.runtime.validate and last_validation_step != step:
                validated_epoch = True
                value, val_batches = _validate(
                    unwrap(model), val_loader, val_sampler, config, device,
                    config.runtime.max_val_batches
                )
                last_validation_step = step
                improved = value < best_val
                if improved:
                    best_val = value; best_step = step
                    best_epoch = completed_epoch
                if main_process:
                    print("[val] epoch={} step={} loss={:.4f} batches={} mags={} digest={} best={:.4f}{}".format(
                        completed_epoch, step, value, val_batches,
                        data_summaries["validation"]["mags"],
                        data_summaries["validation"]["digest"], best_val,
                        " *" if improved else ""), flush=True)
                    if wandb_run:
                        wandb_run.log({"val/loss": value, "val/best": best_val,
                                       "epoch": completed_epoch, "step": step})
            early_stopped = (validated_epoch and _early_stopping_due(
                config, step, completed_epoch, best_step, best_epoch, max_steps))
            gpu_seconds = prior_gpu_seconds + world * (time.time() - wall_start)
            _save_checkpoint(last_path, model, optimizer, scaler, config, train_data,
                             split_plan, data_summaries, run_identity, batch_plan,
                             rank, world,
                             step, epoch, next_global_batch, max_steps,
                             best_val, best_step, best_epoch, gpu_seconds)
            if improved:
                _save_checkpoint(best_path, model, optimizer, scaler, config, train_data,
                                 split_plan, data_summaries, run_identity, batch_plan,
                                 rank, world,
                                 step, epoch, next_global_batch, max_steps,
                                 best_val, best_step, best_epoch, gpu_seconds)
            if early_stopped and main_process:
                unit = "epochs" if config.optim.target_epochs is not None else "steps"
                print("[early-stop] no validation improvement within the {} patience".format(unit),
                      flush=True)

        if step >= max_steps or early_stopped or any_rank_true(stop.requested, device):
            break
        if not epoch_complete:
            break

    gpu_seconds = prior_gpu_seconds + world * (time.time() - wall_start)
    _save_checkpoint(last_path, model, optimizer, scaler, config, train_data,
                     split_plan, data_summaries, run_identity, batch_plan,
                     rank, world,
                     step, epoch, next_global_batch, max_steps, best_val, best_step, best_epoch,
                     gpu_seconds)
    if main_process and stop.signal_name == "STOP_FILE" and STOP_ACK_FILE:
        atomic_write_json(Path(STOP_ACK_FILE), {
            "checkpoint": str(last_path.resolve()),
            "checkpoint_bytes": last_path.stat().st_size,
            "epoch": int(epoch),
            "next_global_batch": int(next_global_batch),
            "scientific_sha256": config.scientific_sha256,
            "step": int(step),
        })
    if is_distributed():
        dist.barrier()
    if main_process:
        print("[done] step={} epoch={} best={:.4f}@{} gpu_hours={:.2f} signal={}".format(
            step, epoch, best_val, best_step, gpu_seconds / 3600, stop.signal_name), flush=True)
        if wandb_run:
            wandb_run.summary["compute/total_gpu_hours"] = gpu_seconds / 3600
            wandb_run.finish()
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
