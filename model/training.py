"""Reusable shard-backed model/data/loss construction for train and probe."""

from __future__ import annotations

import contextlib
import math
from typing import Mapping, Optional

import torch
from torch.utils.data import DataLoader

from losses import combine, ec_loss, gene_identity_loss, ko_loss, mag_taxonomy_loss, patch_disc_loss
from mag_shards.collate import CollateConfig, ShardCollator, move_batch_to_device
from mag_shards.reader import ManifestDataset
from mag_shards.sampler import TokenBudgetBatchSampler
from mag_shards.schema import ALL_TOKEN_MODES, InputSpec
from model import MAGEncoder
from tokenization import NUM_SPECIAL
from train_config import TrainConfig


def vocabulary_sizes(dataset: ManifestDataset, config: TrainConfig):
    manifest = dataset.manifest
    token_size = NUM_SPECIAL
    if config.mode in ALL_TOKEN_MODES:
        try:
            token_size = int(manifest.representations["tokens"][config.mode]["size"])
        except KeyError:
            reason = manifest.unavailable_token_modes.get(config.mode, "not present in manifest")
            raise ValueError("token mode {} unavailable: {}".format(config.mode, reason))
    vocabs = manifest.vocabularies
    ko_size = len(vocabs.get("ko", {})); ec_size = len(vocabs.get("ec", {}))
    tax_sizes = {rank: len(vocabs.get("taxonomy", {}).get(rank, {}))
                 for rank in config.objectives.taxonomy_ranks}
    return token_size, ko_size, ec_size, tax_sizes


def build_model(config: TrainConfig, token_size: int, ko_size: int, ec_size: int,
                tax_sizes: Mapping[str, int], device: str):
    spec = config.input_spec
    gene_sizes = {config.mode: token_size} if spec.active("gene_id") else {}
    model = MAGEncoder(
        vocab_size=token_size, mode=config.mode, num_layers=config.model.n_layers,
        d_model=config.model.d_model, num_heads=config.model.n_heads, ff_mult=config.model.ff_mult,
        use_checkpoint=config.model.use_checkpoint, dropout=config.model.dropout,
        max_local_pos=config.model.max_local_pos, num_ko=max(1, ko_size), num_ec=max(1, ec_size),
        num_tax_per_rank=tax_sizes, gene_id_target_sizes=gene_sizes,
        use_patch_disc=spec.active("patch_disc"), use_ko_head=spec.active("ko"),
        use_ec_head=spec.active("ec"),
        taxonomy_head_ranks=config.objectives.taxonomy_ranks if spec.active("mag_tax") else ())
    if model.embed.mask_vec is not None:
        model.embed.mask_vec.requires_grad_(config.model.trainable_mask)
    return model.to(device)


def build_loader(dataset: ManifestDataset, config: TrainConfig, rank: int, world: int, train: bool,
                 spec: Optional[InputSpec] = None, mask_gene: Optional[bool] = None,
                 per_device_batch_tokens: Optional[int] = None,
                 num_workers: Optional[int] = None):
    spec = spec or config.input_spec
    sampler = TokenBudgetBatchSampler(
        dataset.locators, dataset.manifest_sha256,
        max_batch_tokens=(config.hardware.per_device_batch_tokens
                          if per_device_batch_tokens is None else per_device_batch_tokens),
        per_mag_token_budget=config.batch.resolved_max_mag_tokens,
        rank=rank, world_size=world, max_batch_mags=config.batch.max_batch_mags,
        bucket_window=config.batch.bucket_window, seed=config.runtime.seed,
        shuffle=train, drop_last_group=train,
        max_contig_genes=config.batch.max_contig_genes,
        max_contigs_per_mag=config.batch.max_contigs_per_mag)
    _, ko_size, ec_size, _ = vocabulary_sizes(dataset, config)
    collate = ShardCollator(
        spec,
        CollateConfig(
            per_mag_token_budget=config.batch.per_mag_token_budget,
            max_contig_genes=config.batch.max_contig_genes,
            max_contigs_per_mag=config.batch.max_contigs_per_mag,
            contig_dropout_range=config.batch.contig_dropout_range,
            contig_dropout_apply_prob=(config.batch.contig_dropout_apply_prob if train else 0.0),
            mask_rate_min=config.objectives.mask_rate_min,
            mask_rate_max=config.objectives.mask_rate_max,
            mask_gene=config.objectives.mask_gene if mask_gene is None else bool(mask_gene),
            mask_ko=config.objectives.mask_ko,
            mask_ec=config.objectives.mask_ec,
            train=train,
            eval_seed=config.runtime.seed),
        num_ko=ko_size, num_ec=ec_size)
    workers = config.hardware.num_workers if num_workers is None else int(num_workers)
    kwargs = dict(dataset=dataset, batch_sampler=sampler, num_workers=workers,
                  collate_fn=collate, pin_memory=config.hardware.pin_memory,
                  persistent_workers=workers > 0)
    if workers > 0:
        kwargs["prefetch_factor"] = config.hardware.prefetch_factor
    return DataLoader(**kwargs), sampler


def calibrate_batch_tokens(model, dataset: ManifestDataset, config: TrainConfig,
                           rank: int, world: int, device: str) -> tuple[int, int]:
    """Find a DDP-safe per-device token budget with real forward/backward probes."""
    if device == "cpu" or not torch.cuda.is_available():
        return config.hardware.per_device_batch_tokens, 0

    step = config.hardware.batch_token_search_step
    candidate = max(step, (config.hardware.per_device_batch_tokens // step) * step)
    maximum = config.hardware.max_per_device_batch_tokens
    if maximum is not None:
        candidate = min(candidate, maximum)
    fraction = config.hardware.gpu_memory_fraction
    total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
    # Calibration happens before AdamW and DDP are constructed.  Reserve the
    # memory they subsequently make persistent: two FP32 Adam moments plus a
    # DDP gradient bucket, each approximately one parameter tensor in size.
    parameter_bytes = sum(
        value.numel() * value.element_size()
        for value in model.parameters() if value.requires_grad
    )
    runtime_reserve = 3 * parameter_bytes
    target_memory = int(total_memory * fraction) - runtime_reserve
    if target_memory <= 0:
        raise RuntimeError("model runtime reserve exceeds the GPU memory target")
    safe = 0
    unsafe = None
    peak_at_safe = 0

    def probe(token_budget: int):
        loader, sampler = build_loader(
            dataset, config, rank, world, train=True,
            per_device_batch_tokens=token_budget, num_workers=0)
        sampler.set_epoch(0, 0)
        iterator = iter(loader)
        host_batch = batch = loss = None
        local_ok = True
        peak = 0
        try:
            host_batch = next(iterator)
            batch = move_batch_to_device(host_batch, device)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                         "none": None}[config.optim.amp_dtype]
            autocast = (torch.autocast("cuda", dtype=amp_dtype)
                        if amp_dtype is not None else contextlib.nullcontext())
            with autocast:
                loss, _ = compute_losses(model(batch), batch, config)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            local_ok = peak <= target_memory
        except (torch.OutOfMemoryError, StopIteration):
            local_ok = False
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            local_ok = False
        finally:
            model.zero_grad(set_to_none=True)
            loss = batch = host_batch = iterator = loader = None
            torch.cuda.empty_cache()

        result = torch.tensor([int(local_ok), int(peak)], dtype=torch.int64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result[:1], op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(result[1:], op=torch.distributed.ReduceOp.MAX)
        return bool(result[0].item()), int(result[1].item())

    for _ in range(12):
        fits, peak = probe(candidate)
        if rank == 0:
            print("[batch-calibration] tokens={} fits={} peak={:.1f}/{:.1f} GiB "
                  "runtime_reserve={:.1f} GiB".format(
                      candidate, fits, peak / 2 ** 30, target_memory / 2 ** 30,
                      runtime_reserve / 2 ** 30), flush=True)
        if fits:
            safe = candidate
            peak_at_safe = peak
            if maximum is not None and candidate >= maximum:
                break
            if unsafe is None:
                candidate = candidate * 2 if maximum is None else min(candidate * 2, maximum)
            else:
                next_candidate = ((safe + unsafe) // 2 // step) * step
                if next_candidate <= safe:
                    break
                candidate = next_candidate
        else:
            unsafe = candidate
            if safe == 0:
                candidate = (candidate // 2 // step) * step
                if candidate < step:
                    break
            else:
                next_candidate = ((safe + unsafe) // 2 // step) * step
                if next_candidate <= safe:
                    break
                candidate = next_candidate
    if safe == 0:
        raise RuntimeError(
            "automatic batch calibration could not fit the minimum {}-token batch".format(step))
    return safe, peak_at_safe


def compute_losses(model_output: Mapping[str, object], batch: Mapping[str, object], config: TrainConfig,
                   diagnostics: dict = None):
    weights = config.objectives.weights; parts = {}
    if float(weights.get("gene_id", 0.0)) > 0:
        parts["gene_id"] = gene_identity_loss(
            model_output["gene_id_logits"][config.mode], batch["gene_id_tgt"], batch["mlm_gene"])
    if float(weights.get("ko", 0.0)) > 0:
        where = batch["mlm_ko"] if config.objectives.mask_ko else batch["ko_has"]
        parts["ko"] = ko_loss(model_output["ko_logits"], batch["ko_tgt"], where)
    if float(weights.get("ec", 0.0)) > 0:
        where = batch["mlm_ec"] if config.objectives.mask_ec else batch["ec_has"]
        parts["ec"] = ec_loss(model_output["ec_logits"], batch["ec_tgt"], where)
    if float(weights.get("mag_tax", 0.0)) > 0:
        values = [mag_taxonomy_loss(model_output["mag_tax_logits"][rank], batch["tax_label"][rank])
                  for rank in config.objectives.taxonomy_ranks]
        parts["mag_tax"] = sum(values) / max(1, len(values))
    if float(weights.get("patch_disc", 0.0)) > 0:
        parts["patch_disc"] = patch_disc_loss(
            model_output["esmc_pred"], batch["esmc_target"], batch["mlm_gene"],
            batch["contig_id"], batch["token_type"], diagnostics=diagnostics)
    return combine(weights, parts), parts


def learning_rate(step: int, max_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * peak_lr * (1 + math.cos(math.pi * min(1.0, progress)))
