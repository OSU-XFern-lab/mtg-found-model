"""Evaluate a frozen shard-trained MAG encoder.

Probe tasks request their own data columns from the checkpoint-pinned manifest;
they do not import or mutate the training entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mag_shards.collate import move_batch_to_device
from mag_shards.reader import ManifestDataset
from mag_shards.schema import ALL_TOKEN_MODES, InputSpec
from split_plans import load_split_plan, materialize_split_plan
from train_config import train_config_from_dict
from training import build_loader, build_model, vocabulary_sizes

# ---------------------------------------------------------------------------
# Probe configuration. Edit these values for direct runs.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (Path(os.environ["PROBE_CHECKPOINT"])
                   if os.environ.get("PROBE_CHECKPOINT") else None)
MANIFEST_OVERRIDE = (Path(os.environ["PROBE_MANIFEST"])
                     if os.environ.get("PROBE_MANIFEST") else None)
TASKS = tuple(os.environ.get(
    "PROBE_TASKS", "gene_id,ko,ec,mag_tax,patch_disc").split(","))
FIT_MAGS = int(os.environ.get("PROBE_FIT_MAGS", "2000"))
EVAL_MAGS = int(os.environ.get("PROBE_EVAL_MAGS", "1000"))
PROBE_TYPE = os.environ.get("PROBE_TYPE", "linear")              # linear | mlp
PROBE_HIDDEN = int(os.environ.get("PROBE_HIDDEN", "512"))
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", "30"))
PROBE_LR = float(os.environ.get("PROBE_LR", "1e-3"))
DEVICE = os.environ.get("PROBE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _probe_spec(mode, tasks, taxonomy_ranks):
    tasks = set(tasks)
    if mode not in ALL_TOKEN_MODES:
        tasks.discard("gene_id")
    weights = {name: 0.0 for name in ("gene_id", "ko", "ec", "mag_tax", "patch_disc")}
    for name in tasks:
        if name == "gene_id" and mode not in ALL_TOKEN_MODES:
            continue
        weights[name] = 1.0
    return InputSpec(mode, weights, probe_tasks=tuple(tasks), taxonomy_ranks=taxonomy_ranks)


def _load(checkpoint_path, manifest_override, tasks, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = train_config_from_dict(checkpoint["config"], checkpoint["config"].get("profile", "default"))
    manifest_path = Path(manifest_override or checkpoint["manifest_path"])
    spec = _probe_spec(config.mode, tasks, config.objectives.taxonomy_ranks)
    dataset = ManifestDataset(manifest_path, spec, root_overrides=config.data.root_overrides)
    if dataset.manifest_sha256 != checkpoint["manifest_sha256"]:
        raise RuntimeError("probe manifest differs from checkpoint-pinned dataset")
    token_size, ko_size, ec_size, tax_sizes = vocabulary_sizes(dataset, config)
    model = build_model(config, token_size, ko_size, ec_size, tax_sizes, device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    split_plan_path = checkpoint.get("split_plan_path")
    if split_plan_path is None:
        split_plan_path = materialize_split_plan(
            manifest_path, config.data.split,
            Path(config.runtime.checkpoint_root) / "split_plans",
        )
    split_plan_path = Path(split_plan_path)
    split_plan = load_split_plan(split_plan_path, dataset.manifest_sha256)
    return model, checkpoint, config, manifest_path, spec, split_plan


@torch.no_grad()
def _collect(model, loader, cap, device, want):
    gene_hidden = []; mag_hidden = []; result = {"tax": {}}
    ko_target = []; ko_has = []; ec_target = []; ec_has = []
    seen = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device); output = model(batch)
        gene = batch["token_type"].eq(1)
        gene_hidden.append(output["hidden"][gene].float().cpu())
        mag_hidden.append(output["mag_repr"].float().cpu())
        if "ko" in want:
            ko_target.append(batch["ko_tgt"][gene].cpu()); ko_has.append(batch["ko_has"][gene].cpu())
        if "ec" in want:
            ec_target.append(batch["ec_tgt"][gene].cpu()); ec_has.append(batch["ec_has"][gene].cpu())
        if "mag_tax" in want:
            for rank, labels in batch["tax_label"].items():
                result["tax"].setdefault(rank, []).append(labels.cpu())
        seen += batch["token_type"].shape[0]
        if seen >= cap:
            break
    cat = lambda values: torch.cat(values) if values else torch.empty(0)
    result.update(gene=cat(gene_hidden), mag=cat(mag_hidden), ko_tgt=cat(ko_target),
                  ko_has=cat(ko_has), ec_tgt=cat(ec_target), ec_has=cat(ec_has),
                  tax={rank: cat(values) for rank, values in result["tax"].items()})
    return result


def _head(d_in, d_out):
    if PROBE_TYPE == "mlp":
        return nn.Sequential(nn.Linear(d_in, PROBE_HIDDEN), nn.GELU(), nn.Linear(PROBE_HIDDEN, d_out))
    return nn.Linear(d_in, d_out)


def _fit(head, features, targets, loss_function):
    head = head.to(DEVICE); features = features.to(DEVICE); targets = targets.to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad(); loss = loss_function(head(features), targets); loss.backward(); optimizer.step()
    return head.eval()


def _average_precision(scores, labels):
    try:
        from sklearn.metrics import average_precision_score
        keep = labels.sum(0) > 0
        return float(average_precision_score(labels[:, keep], scores[:, keep], average="micro"))
    except Exception:
        return float("nan")


def _probe_multilabel(name, fit, evaluation):
    fit_mask = fit[name + "_has"]; eval_mask = evaluation[name + "_has"]
    if fit_mask.numel() == 0 or eval_mask.numel() == 0 or not fit_mask.any() or not eval_mask.any():
        return "{}: no annotated genes".format(name)
    x_fit = fit["gene"][fit_mask]; y_fit = fit[name + "_tgt"][fit_mask]
    x_eval = evaluation["gene"][eval_mask]; y_eval = evaluation[name + "_tgt"][eval_mask]
    head = _fit(_head(x_fit.shape[1], y_fit.shape[1]), x_fit, y_fit, nn.BCEWithLogitsLoss())
    with torch.no_grad():
        logits = head(x_eval.to(DEVICE)).cpu()
    prediction = logits.gt(0).float()
    micro_f1 = (2 * (prediction * y_eval).sum() / (prediction.sum() + y_eval.sum()).clamp(min=1)).item()
    return "{}: micro-F1={:.4f} micro-AP={:.4f} (n={})".format(
        name, micro_f1, _average_precision(logits.numpy(), y_eval.numpy()), len(x_eval))


def _probe_ko(fit, evaluation, classes):
    fit_mask = fit["ko_has"]; eval_mask = evaluation["ko_has"]
    if not fit_mask.any() or not eval_mask.any():
        return "ko: no unambiguous annotated genes"
    x_fit = fit["gene"][fit_mask]; y_fit = fit["ko_tgt"][fit_mask]
    x_eval = evaluation["gene"][eval_mask]; y_eval = evaluation["ko_tgt"][eval_mask]
    head = _fit(_head(x_fit.shape[1], classes), x_fit, y_fit, nn.CrossEntropyLoss())
    with torch.no_grad():
        prediction = head(x_eval.to(DEVICE)).argmax(1).cpu()
    return "ko: multiclass accuracy={:.4f} (n={})".format(
        (prediction == y_eval).float().mean().item(), len(y_eval))


def _probe_taxonomy(fit, evaluation):
    lines = []
    for rank in sorted(set(fit["tax"]) & set(evaluation["tax"])):
        y_fit = fit["tax"][rank]; y_eval = evaluation["tax"][rank]
        fit_mask = y_fit.ne(-1); eval_mask = y_eval.ne(-1)
        if not fit_mask.any() or not eval_mask.any():
            continue
        x_fit = fit["mag"][fit_mask]; x_eval = evaluation["mag"][eval_mask]
        y_fit = y_fit[fit_mask]; y_eval = y_eval[eval_mask]
        classes = int(max(y_fit.max(), y_eval.max())) + 1
        head = _fit(_head(x_fit.shape[1], classes), x_fit, y_fit, nn.CrossEntropyLoss())
        with torch.no_grad():
            prediction = head(x_eval.to(DEVICE)).argmax(1).cpu()
        lines.append("  {}: acc={:.4f} (n={})".format(rank, (prediction == y_eval).float().mean(), len(x_eval)))
    return "mag_tax:\n" + "\n".join(lines) if lines else "mag_tax: no labels"


@torch.no_grad()
def _intrinsic_gene_id(model, loader, mode, device):
    hits = total = 0
    for host in loader:
        batch = move_batch_to_device(host, device); output = model(batch); mask = batch["mlm_gene"]
        if "gene_id_logits" not in output or not mask.any():
            continue
        target = batch["gene_id_tgt"][mask]; valid = target.ne(-100)
        prediction = output["gene_id_logits"][mode][mask].argmax(-1)
        hits += int((prediction[valid] == target[valid]).sum()); total += int(valid.sum())
    return "gene_id: masked top-1 acc={:.4f} (n={})".format(hits / max(1, total), total)


@torch.no_grad()
def _intrinsic_patch(model, loader, device):
    hits = total = 0
    for host in loader:
        batch = move_batch_to_device(host, device); output = model(batch)
        if "esmc_pred" not in output:
            continue
        predicted = F.normalize(output["esmc_pred"], dim=-1)
        target = F.normalize(batch["esmc_target"].to(predicted.dtype), dim=-1)
        for b in range(predicted.shape[0]):
            for contig in torch.unique(batch["contig_id"][b][batch["token_type"][b].eq(1)]):
                genes = (batch["token_type"][b].eq(1) & batch["contig_id"][b].eq(contig)).nonzero().squeeze(-1)
                if genes.numel() < 2:
                    continue
                query = batch["mlm_gene"][b, genes]
                if not query.any():
                    continue
                prediction = (predicted[b, genes] @ target[b, genes].T).argmax(-1)
                identity = torch.arange(genes.numel(), device=device)
                hits += int((prediction[query] == identity[query]).sum()); total += int(query.sum())
    return "patch_disc: same-contig retrieval acc={:.4f} (n={})".format(hits / max(1, total), total)


def main():
    if CHECKPOINT_PATH is None:
        raise ValueError("set CHECKPOINT_PATH at the top of probe.py or PROBE_CHECKPOINT")
    allowed = {"gene_id", "ko", "ec", "mag_tax", "patch_disc"}
    if set(TASKS) - allowed or PROBE_TYPE not in ("linear", "mlp"):
        raise ValueError("invalid probe configuration")
    model, checkpoint, config, manifest_path, spec, split_plan = _load(
        CHECKPOINT_PATH, MANIFEST_OVERRIDE, TASKS, DEVICE)
    train_data = ManifestDataset(
        manifest_path, spec, splits=config.data.train.labels,
        sources=config.data.train.sources, root_overrides=config.data.root_overrides,
        assignments=split_plan.assignments)
    val_data = ManifestDataset(
        manifest_path, spec, splits=config.data.validation.labels,
        sources=config.data.validation.sources, root_overrides=config.data.root_overrides,
        assignments=split_plan.assignments)
    fit_loader, _ = build_loader(train_data, config, 0, 1, False, spec=spec, mask_gene=False)
    eval_loader, _ = build_loader(val_data, config, 0, 1, False, spec=spec, mask_gene=False)
    probe_tasks = set(TASKS) & {"ko", "ec", "mag_tax"}
    results = []
    if probe_tasks:
        fit = _collect(model, fit_loader, FIT_MAGS, DEVICE, probe_tasks)
        evaluation = _collect(model, eval_loader, EVAL_MAGS, DEVICE, probe_tasks)
        if "ko" in probe_tasks:
            results.append(_probe_ko(fit, evaluation, fit_loader.collate_fn.num_ko))
        if "ec" in probe_tasks:
            results.append(_probe_multilabel("ec", fit, evaluation))
        if "mag_tax" in probe_tasks:
            results.append(_probe_taxonomy(fit, evaluation))
    intrinsic = set(TASKS) & {"gene_id", "patch_disc"}
    if intrinsic:
        intrinsic_loader, _ = build_loader(val_data, config, 0, 1, False, spec=spec, mask_gene=True)
        trained = {name for name, value in checkpoint["loss_weights"].items() if float(value) > 0}
        if "gene_id" in intrinsic and "gene_id" in trained and config.mode in ALL_TOKEN_MODES:
            results.append(_intrinsic_gene_id(model, intrinsic_loader, config.mode, DEVICE))
        if "patch_disc" in intrinsic and "patch_disc" in trained:
            results.append(_intrinsic_patch(model, intrinsic_loader, DEVICE))
    print("\n=== probe results ===")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
