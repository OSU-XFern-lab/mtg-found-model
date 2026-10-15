"""Loss terms for the MAG encoder.

All losses return a scalar tensor. Missing signal yields 0.0 so they are
safe to weight-sum unconditionally.
"""

from __future__ import annotations

import torch
import math
import torch.nn.functional as F

from mag_shards.schema import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Gene identity MLM
# ---------------------------------------------------------------------------

def gene_identity_loss(logits: torch.Tensor, targets: torch.Tensor, mlm_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on masked gene positions only."""
    if logits is None:
        return torch.zeros((), device=targets.device)
    if not mlm_mask.any():
        return logits.sum() * 0.0
    logits_sel = logits[mlm_mask]                     # (N, V)
    targets_sel = targets[mlm_mask]                   # (N,)
    valid = targets_sel.ne(IGNORE_INDEX)
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits_sel[valid], targets_sel[valid])


# ---------------------------------------------------------------------------
# KO multiclass and EC multilabel
# ---------------------------------------------------------------------------

def _multilabel_loss(logits: torch.Tensor, targets: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    if not where.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[where], targets[where])


def ko_loss(logits: torch.Tensor, targets: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    return gene_identity_loss(logits, targets, where)


def ec_loss(logits: torch.Tensor, targets: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    # EC targets are already constructed from top-1 and top-2 levels.
    return _multilabel_loss(logits, targets, where)


# ---------------------------------------------------------------------------
# MAG-level taxonomy
# ---------------------------------------------------------------------------

def mag_taxonomy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-1)
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


# ---------------------------------------------------------------------------
# Patch discrimination (ESM-C target mode)
# ---------------------------------------------------------------------------

def patch_disc_loss(
    z_hat: torch.Tensor,          # (B, L, D) predicted ESM-C latents
    z_target: torch.Tensor,       # (B, L, D) target ESM-C latents
    mlm_gene: torch.Tensor,       # (B, L) bool, True at masked gene positions
    contig_id: torch.Tensor,      # (B, L) int, same value within a contig, -1 for PAD/UCT
    token_type: torch.Tensor,     # (B, L) int, 1 = GENE
    tau: float = 0.05,
    diagnostics: dict = None,
) -> torch.Tensor:
    """InfoNCE per masked gene: positive is its own target, negatives are
    all other GENE target latents in the same contig (same sample).
    """
    is_gene = token_type.eq(1)
    zh = F.normalize(z_hat[is_gene], dim=-1)
    zt = F.normalize(z_target[is_gene].to(z_hat.dtype), dim=-1)
    valid_gene = mlm_gene[is_gene]

    # Gene rows are contiguous within each contig.  Group equally sized
    # contigs so their InfoNCE matrices are evaluated as batched GEMMs instead
    # of launching several small GPU kernels for every individual contig.
    batch_id = torch.arange(z_hat.size(0), device=z_hat.device)[:, None]
    keys = batch_id * (z_hat.size(1) + 1) + contig_id
    _, lengths = torch.unique_consecutive(keys[is_gene], return_counts=True)
    starts = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
    total = zh.sum() * 0.0
    count = lengths.new_zeros(())
    if diagnostics is not None:
        stats = torch.zeros(12 + 2 * zh.shape[-1], device=zh.device, dtype=torch.float64)
        with torch.no_grad():
            masked = zh[valid_gene].detach().float()
            stats[9:12] = torch.tensor([len(masked), len(zh), len(masked)], device=zh.device)
            stats[12:12 + zh.shape[-1]] = masked.sum(0)
            stats[12 + zh.shape[-1]:] = masked.square().sum(0)
    for length in torch.unique(lengths).tolist():
        if length < 2:
            continue
        group_starts = starts[lengths.eq(length)]
        indices = group_starts[:, None] + torch.arange(length, device=z_hat.device)
        predicted = zh[indices]
        target = zt[indices]
        valid = valid_gene[indices]
        sim = torch.bmm(predicted, target.transpose(1, 2)) / tau
        diagonal = F.log_softmax(sim, dim=-1).diagonal(dim1=1, dim2=2)
        total = total - diagonal[valid].sum()
        count = count + valid.sum()
        if diagnostics is not None:
            with torch.no_grad():
                scores = sim.detach().float()[valid]
                labels = torch.arange(length, device=zh.device).expand_as(valid)[valid]
                positive = scores.gather(1, labels[:, None]).squeeze(1)
                logp = scores.log_softmax(-1)
                probability = logp.exp()
                top = scores.topk(min(5, length), dim=-1).indices
                negative = scores.scatter(1, labels[:, None], -torch.inf).max(-1).values
                values = torch.stack((
                    top[:, 0].eq(labels).float(), top.eq(labels[:, None]).any(-1).float(),
                    positive * tau, negative * tau, (positive - negative) * tau,
                    logp.gather(1, labels[:, None]).squeeze(1).exp(),
                    -(probability * logp).sum(-1) / math.log(length),
                    torch.full_like(positive, length)), dim=1)
                stats[:8] += values.double().sum(0)
                stats[8] += len(labels)
    if diagnostics is not None:
        diagnostics["sums"] = stats
    return total / count.clamp(min=1).to(total.dtype)


def patch_disc_metrics(stats: torch.Tensor) -> dict:
    """Finalize additive statistics AFTER summing across DDP ranks."""
    names = ("top1", "top5", "positive_cosine", "hardest_negative_cosine", "margin",
             "target_probability", "normalized_entropy", "candidate_count")
    values = (stats[:8] / stats[8]).tolist() if stats[8] > 0 else [float("nan")] * 8
    result = dict(zip(names, values))
    result["mask_ratio"] = float(stats[9] / stats[10]) if stats[10] else float("nan")
    n = stats[11]
    sums, squares = stats[12:].chunk(2)
    result["prediction_feature_std"] = (float(((squares - sums.square() / n) / (n - 1))
                                              .clamp_min(0).sqrt().mean())
                                        if n > 1 else float("nan"))
    return {"patch_disc/" + key: value for key, value in result.items()}


# ---------------------------------------------------------------------------
# Weighted combined loss
# ---------------------------------------------------------------------------

def combine(weights: dict, parts: dict) -> torch.Tensor:
    """Weight-sum per-task losses. Missing weights default to 1.0."""
    total = None
    for name, value in parts.items():
        w = float(weights.get(name, 1.0))
        if w == 0.0:
            continue
        term = w * value
        total = term if total is None else total + term
    if total is None:
        any_val = next(iter(parts.values()))
        return any_val * 0.0
    return total
