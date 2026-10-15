"""Flat sparse transformer encoder for MAGs.

Design notes
------------
A MAG is flattened as

    [UCT_1, g_1..g_n, UCT_2, g_1..g_m, ...]

Attention is allowed between two tokens iff
    (a) they live in the same contig, or
    (b) both are UCT tokens.

We implement the sparsity with an additive mask over dense attention. This is
not the optimized kernel (documented in the PRD) but has the exact semantics
and is easy to verify.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from tokenization import DISCRETE_MODES, NUM_SPECIAL

flex_attention = torch.compile(flex_attention)


def build_block_mask(token_type: torch.Tensor, contig_id: torch.Tensor):
    """Block-sparse mask: same-contig OR both-UCT (self always allowed)."""
    B, L = token_type.shape

    if token_type.device.type == "cpu":
        # FlexAttention is CUDA-only.  A dense boolean mask keeps CPU smoke
        # tests and small debugging runs semantically equivalent.
        same = contig_id.unsqueeze(2) == contig_id.unsqueeze(1)
        both_uct = token_type.eq(0).unsqueeze(2) & token_type.eq(0).unsqueeze(1)
        valid = token_type.ne(-1)
        allowed = valid.unsqueeze(2) & valid.unsqueeze(1) & (same | both_uct)
        identity = torch.eye(L, dtype=torch.bool, device=token_type.device).unsqueeze(0)
        return (allowed | identity).unsqueeze(1)

    def mod(b, h, q, k):
        same = contig_id[b, q] == contig_id[b, k]
        both_uct = (token_type[b, q] == 0) & (token_type[b, k] == 0)
        valid = (token_type[b, q] != -1) & (token_type[b, k] != -1)
        return (valid & (same | both_uct)) | (q == k)

    return create_block_mask(mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=token_type.device)


# ---------------------------------------------------------------------------
# Embedding builder
# ---------------------------------------------------------------------------

class EmbeddingBuilder(nn.Module):
    """Produce per-token embeddings for all modes, plus UCT init."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        mode: str,
        max_local_pos: int,
        esmc_dim: int = 1152,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.pos_emb = nn.Embedding(max_local_pos + 2, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0=UCT, 1=GENE
        if mode in DISCRETE_MODES:
            self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.esm_proj = None
            self.mask_vec = None
        elif mode == "esmc":
            # tok_emb is used only for specials (PAD/MASK/UCT/UNK); real
            # genes use esm_proj over their precomputed ESM-C vector.
            self.tok_emb = nn.Embedding(NUM_SPECIAL, d_model, padding_idx=0)
            self.esm_proj = nn.Linear(esmc_dim, d_model)
            self.mask_vec = nn.Parameter(torch.zeros(d_model))
        else:
            raise ValueError(mode)

        # BERT-[CLS]-style learned vectors for UCT and (esmc only) UNK gene.
        self.uct_vec = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.uct_vec, mean=0.0, std=0.02)
        # Not sure of this, or if I should just use the ESM-C model to compute the embedding for the unseen protein
        self.unk_vec = nn.Parameter(torch.empty(d_model)) if mode == "esmc" else None
        if self.unk_vec is not None:
            nn.init.normal_(self.unk_vec, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, L)
        token_type: torch.Tensor,        # (B, L)
        contig_id: torch.Tensor,         # (B, L)   (unused)
        local_pos: torch.Tensor,         # (B, L)
        esmc_vecs: Optional[torch.Tensor] = None,  # (B, L, esmc_dim) or None
        mlm_gene: Optional[torch.Tensor] = None,   # (B, L) bool
        gene_is_unk: Optional[torch.Tensor] = None,  # (B, L) bool, ESM-C unseen
    ) -> torch.Tensor:
        x = self.tok_emb(token_ids.clamp(min=0))
        if self.mode == "esmc":
            assert esmc_vecs is not None, "esmc mode requires esmc_vecs input"
            # CUDA autocast handles FP16/BF16 inputs against FP32 weights.
            # Validation, probing, and CPU debugging may run without autocast;
            # make those call paths dtype-safe without forcing an FP32 copy in
            # the normal mixed-precision training path.
            if esmc_vecs.dtype != self.esm_proj.weight.dtype and not torch.is_autocast_enabled():
                esmc_vecs = esmc_vecs.to(self.esm_proj.weight.dtype)
            proj = self.esm_proj(esmc_vecs)
            gene_mask = (token_type == 1).unsqueeze(-1)
            x = torch.where(gene_mask, proj, x)
            # Replace genes with no precomputed ESM-C vector by the learned
            # UNK embedding so we never feed an all-zero row through.
            if gene_is_unk is not None and self.unk_vec is not None:
                x = torch.where(gene_is_unk.unsqueeze(-1), self.unk_vec.expand_as(x), x)
            if mlm_gene is not None:
                x = torch.where(mlm_gene.unsqueeze(-1), self.mask_vec.expand_as(x), x)

        # Replace UCT content with the learned [CLS]-style vector before
        # type/pos embeddings, so UCT slots still get positional signal.
        uct_rows = (token_type == 0).unsqueeze(-1)
        if uct_rows.any():
            x = torch.where(uct_rows, self.uct_vec.expand_as(x), x)

        safe_type = token_type.clamp(min=0)              # PADs get a dummy type.
        x = x + self.type_emb(safe_type)
        x = x + self.pos_emb(local_pos.clamp(min=0))
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _attn(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]    # (B, H, L, Hd)
        if torch.is_tensor(block_mask):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        else:
            out = flex_attention(q, k, v, block_mask=block_mask)
        return self.o(out.transpose(1, 2).reshape(B, L, D))

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        x = x + self.drop(self._attn(self.ln1(x), block_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class FlatSparseEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ff_mult: int,
        use_checkpoint: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, block_mask, use_reentrant=False)
            else:
                x = block(x, block_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class MAGEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        mode: str,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ff_mult: int,
        use_checkpoint: bool,
        max_local_pos: int,
        num_ko: int,
        num_ec: int,
        num_tax_per_rank: dict[str, int],
        dropout: float = 0.0,
        esmc_dim: int = 1152,
        gene_id_target_sizes: dict[str, int] | None = None,
        use_patch_disc: bool = False,
        use_ko_head: bool = True,
        use_ec_head: bool = True,
        taxonomy_head_ranks: Optional[Sequence[str]] = None,
    ):
        """One head per discrete gene-target mode (so train/val/test can use
        different vocabs). `gene_id_target_sizes` maps target-mode name to
        its vocab size, e.g. {"vq_kmeans": 65540, "vq_vae": 65540}.
        `use_patch_disc=True` adds the ESM-C InfoNCE projector.
        """
        super().__init__()
        self.mode = mode
        self.esmc_dim = esmc_dim
        self.embed = EmbeddingBuilder(
            d_model=d_model,
            vocab_size=vocab_size,
            mode=mode,
            max_local_pos=max_local_pos,
            esmc_dim=esmc_dim,
        )
        self.encoder = FlatSparseEncoder(num_layers, d_model, num_heads, ff_mult, use_checkpoint, dropout)

        sizes = gene_id_target_sizes or {}
        self.gene_id_heads = nn.ModuleDict(
            {m: nn.Linear(d_model, n) for m, n in sizes.items() if n > 0}
        )
        self.esmc_pred = nn.Linear(d_model, esmc_dim) if use_patch_disc else None

        self.ko_head = nn.Linear(d_model, num_ko) if use_ko_head else None
        self.ec_head = nn.Linear(d_model, num_ec) if use_ec_head else None
        active_taxonomy = set(num_tax_per_rank) if taxonomy_head_ranks is None else set(taxonomy_head_ranks)
        self.mag_tax_heads = nn.ModuleDict(
            {rank: nn.Linear(d_model, max(1, n)) for rank, n in num_tax_per_rank.items()
             if rank in active_taxonomy}
        )

    def forward(self, batch: dict) -> dict:
        x = self.embed(
            token_ids=batch["token_ids"],
            token_type=batch["token_type"],
            contig_id=batch["contig_id"],
            local_pos=batch["local_pos"],
            esmc_vecs=batch.get("esmc_vecs"),
            mlm_gene=batch.get("mlm_gene") if self.mode == "esmc" else None,
            gene_is_unk=batch.get("gene_is_unk") if self.mode == "esmc" else None,
        )
        block_mask = build_block_mask(batch["token_type"], batch["contig_id"])
        h = self.encoder(x, block_mask)

        # `where` (not multiply): NaNs at PAD rows must not poison the sum.
        uct_rows = (batch["token_type"] == 0).unsqueeze(-1)                     # (B, L, 1)
        uct_sum = torch.where(uct_rows, h, torch.zeros_like(h)).sum(dim=1)
        uct_cnt = uct_rows.float().sum(dim=1).clamp(min=1.0)
        mag_repr = uct_sum / uct_cnt                         # (B, D)

        out = {"hidden": h, "mag_repr": mag_repr}
        if self.ko_head is not None:
            out["ko_logits"] = self.ko_head(h)
        if self.ec_head is not None:
            out["ec_logits"] = self.ec_head(h)
        if len(self.mag_tax_heads) > 0:
            out["mag_tax_logits"] = {rank: head(mag_repr) for rank, head in self.mag_tax_heads.items()}
        if len(self.gene_id_heads) > 0:
            out["gene_id_logits"] = {m: head(h) for m, head in self.gene_id_heads.items()}
        if self.esmc_pred is not None:
            out["esmc_pred"] = self.esmc_pred(h)
        return out
