"""Objective-aware collation from compact shard examples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from tokenization import SPECIAL
from .reader import MagExample
from .schema import ESMC_DIM, IGNORE_INDEX, InputSpec, MISSING_U32


@dataclass(frozen=True)
class CollateConfig:
    per_mag_token_budget: Optional[int] = 16384
    max_contig_genes: int = 512
    max_contigs_per_mag: Optional[int] = None
    contig_dropout_range: Tuple[float, float] = (0.0, 0.0)
    contig_dropout_apply_prob: float = 0.0
    mask_rate_min: float = 0.15
    mask_rate_max: float = 0.15
    mask_gene: bool = True
    mask_ko: bool = False
    mask_ec: bool = False
    train: bool = True
    eval_seed: int = 0
    esmc_host_dtype: torch.dtype = torch.float16


@dataclass
class _Selected:
    example: MagExample
    ranges: List[Tuple[int, int]]
    seed: int

    @property
    def length(self) -> int:
        return sum(1 + end - start for start, end in self.ranges)


class ShardCollator:
    def __init__(self, spec: InputSpec, config: CollateConfig, num_ko: int = 0, num_ec: int = 0):
        self.spec = spec
        self.config = config
        self.num_ko = int(num_ko)
        self.num_ec = int(num_ec)
        if spec.needs_ko and self.num_ko <= 0:
            raise ValueError("active KO objective requires a non-empty vocabulary")
        if spec.needs_ec and self.num_ec <= 0:
            raise ValueError("active EC objective requires a non-empty vocabulary")

    def __call__(self, examples: Sequence[MagExample]) -> Dict[str, object]:
        selected = [self._select(example) for example in examples]
        batch_size = len(selected)
        length = max((x.length for x in selected), default=1)
        token_ids = torch.full((batch_size, length), SPECIAL["PAD"], dtype=torch.long)
        token_type = torch.full((batch_size, length), -1, dtype=torch.long)
        contig_id = torch.full((batch_size, length), -1, dtype=torch.long)
        local_pos = torch.zeros((batch_size, length), dtype=torch.long)
        valid_mask = torch.zeros((batch_size, length), dtype=torch.bool)
        gene_is_unk = torch.zeros((batch_size, length), dtype=torch.bool)

        gene_id_tgt = (torch.full((batch_size, length), IGNORE_INDEX, dtype=torch.long)
                       if self.spec.active("gene_id") else None)
        ko_tgt = (torch.full((batch_size, length), IGNORE_INDEX, dtype=torch.long)
                  if self.spec.needs_ko else None)
        ko_has = torch.zeros((batch_size, length), dtype=torch.bool) if self.spec.needs_ko else None
        ec_tgt = (torch.zeros((batch_size, length, self.num_ec), dtype=torch.float32)
                  if self.spec.needs_ec else None)
        ec_has = torch.zeros((batch_size, length), dtype=torch.bool) if self.spec.needs_ec else None
        tax_label = ({rank: torch.full((batch_size,), -1, dtype=torch.long)
                      for rank in self.spec.taxonomy_ranks} if self.spec.needs_taxonomy else None)
        esmc = (torch.zeros((batch_size, length, ESMC_DIM), dtype=self.config.esmc_host_dtype)
                if self.spec.needs_esmc else None)

        batch_seeds = []
        for b, item in enumerate(selected):
            ex = item.example; cursor = 0; batch_seeds.append(item.seed)
            for new_cid, (start, end) in enumerate(item.ranges):
                token_ids[b, cursor] = SPECIAL["UCT"]
                token_type[b, cursor] = 0
                contig_id[b, cursor] = new_cid
                valid_mask[b, cursor] = True
                cursor += 1
                n = end - start
                rows = slice(cursor, cursor + n)
                token_type[b, rows] = 1
                contig_id[b, rows] = new_cid
                local_pos[b, rows] = torch.arange(1, n + 1, dtype=torch.long)
                valid_mask[b, rows] = True
                if ex.tokens is not None:
                    vals = np.asarray(ex.tokens[start:end], dtype=np.uint32)
                    ok = vals != MISSING_U32
                    encoded = np.where(ok, vals, np.uint32(SPECIAL["UNK"])).astype(np.int64)
                    token_ids[b, rows] = torch.from_numpy(encoded)
                    gene_is_unk[b, rows] = torch.from_numpy(~ok)
                    if gene_id_tgt is not None:
                        targets = np.where(ok, vals.astype(np.int64), IGNORE_INDEX)
                        gene_id_tgt[b, rows] = torch.from_numpy(targets)
                elif self.spec.mode == "esmc":
                    valid = np.ones(n, dtype=bool) if ex.esmc_valid is None else np.asarray(ex.esmc_valid[start:end])
                    gene_is_unk[b, rows] = torch.from_numpy(~valid)
                if esmc is not None and ex.esmc is not None:
                    # Memmap slices are read-only; copy into the owned batch
                    # tensor explicitly to avoid exposing a non-writable view.
                    esmc[b, rows] = torch.tensor(np.asarray(ex.esmc[start:end]), dtype=self.config.esmc_host_dtype)
                if ko_tgt is not None:
                    for local, gene in enumerate(range(start, end)):
                        begin, finish = int(ex.ko_indptr[gene]), int(ex.ko_indptr[gene + 1])
                        labels = np.unique(ex.ko_ids[begin:finish])
                        # Duplicate copies of one label are unambiguous; multiple KOs are not.
                        if len(labels) == 1 and 0 <= int(labels[0]) < self.num_ko:
                            ko_tgt[b, cursor + local] = int(labels[0])
                            ko_has[b, cursor + local] = True
                if ec_tgt is not None:
                    self._fill_sparse(ec_tgt[b], ec_has[b], ex.ec_indptr, ex.ec_ids, start, end, cursor)
                cursor += n
            if tax_label is not None:
                for rank in self.spec.taxonomy_ranks:
                    tax_label[rank][b] = int(ex.taxonomy.get(rank, -1))

        gene_positions = token_type.eq(1)
        mlm_gene = torch.zeros((batch_size, length), dtype=torch.bool)
        mlm_ko = torch.zeros_like(mlm_gene) if self.spec.needs_ko else None
        mlm_ec = torch.zeros_like(mlm_gene) if self.spec.needs_ec else None
        for b, seed in enumerate(batch_seeds):
            rng = np.random.RandomState(seed & 0xFFFFFFFF)
            rate = rng.uniform(self.config.mask_rate_min, self.config.mask_rate_max)
            if self.config.mask_gene:
                mask = torch.from_numpy(rng.random_sample(length) < rate) & gene_positions[b]
                if self.spec.needs_esmc_target:
                    mask &= ~gene_is_unk[b]
                elif gene_id_tgt is not None:
                    mask &= gene_id_tgt[b].ne(IGNORE_INDEX)
                mlm_gene[b] = mask
            if mlm_ko is not None and self.config.mask_ko:
                mlm_ko[b] = torch.from_numpy(rng.random_sample(length) < rate) & ko_has[b]
            if mlm_ec is not None and self.config.mask_ec:
                mlm_ec[b] = torch.from_numpy(rng.random_sample(length) < rate) & ec_has[b]
        token_ids.masked_fill_(mlm_gene, SPECIAL["MASK"])

        batch = {"token_ids": token_ids, "token_type": token_type, "contig_id": contig_id,
                 "local_pos": local_pos, "valid_mask": valid_mask, "mlm_gene": mlm_gene,
                 "gene_is_unk": gene_is_unk,
                 "mag_id": [x.example.mag_id for x in selected],
                 "source_name": [x.example.source_name for x in selected],
                 "n_genes": int(gene_positions.sum())}
        if gene_id_tgt is not None:
            batch["gene_id_tgt"] = gene_id_tgt
        if ko_tgt is not None:
            batch.update(ko_tgt=ko_tgt, ko_has=ko_has, mlm_ko=mlm_ko)
        if ec_tgt is not None:
            batch.update(ec_tgt=ec_tgt, ec_has=ec_has, mlm_ec=mlm_ec)
        if tax_label is not None:
            batch["tax_label"] = tax_label
        if esmc is not None:
            if self.spec.needs_esmc_input:
                batch["esmc_vecs"] = esmc
            if self.spec.needs_esmc_target:
                batch["esmc_target"] = esmc
        return batch

    def _select(self, ex: MagExample) -> _Selected:
        seed = ex.sample_seed if self.config.train else _stable_seed(self.config.eval_seed, ex.mag_id)
        rng = np.random.RandomState(seed & 0xFFFFFFFF)
        offsets = np.asarray(ex.contig_gene_offsets, dtype=np.int64)
        ranges = [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)
                  if offsets[i + 1] > offsets[i]]
        if self.config.train:
            rng.shuffle(ranges)
        if self.config.max_contigs_per_mag is not None:
            ranges = ranges[:self.config.max_contigs_per_mag]
        truncated = []
        for start, end in ranges:
            if end - start > self.config.max_contig_genes:
                if self.config.train:
                    anchor = int(rng.randint(end - start))
                    start += max(0, min(anchor - self.config.max_contig_genes // 2,
                                        end - start - self.config.max_contig_genes))
                end = start + self.config.max_contig_genes
            truncated.append((start, end))
        # Budgeted selection already varies oversized MAGs. Only thin fitting MAGs.
        genes = sum(end - start for start, end in truncated)
        fits = (self.config.per_mag_token_budget is None or
                genes + len(truncated) <= self.config.per_mag_token_budget)
        if (self.config.train and fits and len(truncated) > 3
                and rng.random_sample() < self.config.contig_dropout_apply_prob):
            rate = rng.uniform(*self.config.contig_dropout_range)
            kept = [x for x in truncated if rng.random_sample() >= rate]
            if len(kept) >= 3 and sum(end - start for start, end in kept) >= max(2, genes / 2):
                truncated = kept
        if self.config.per_mag_token_budget is None:
            packed = truncated
        else:
            packed = []; used = 0
            for start, end in truncated:
                cost = 1 + end - start
                if used + cost <= self.config.per_mag_token_budget:
                    packed.append((start, end)); used += cost
            if not packed and truncated:
                start, end = truncated[0]
                packed = [(start, min(end, start + max(1, self.config.per_mag_token_budget - 1)))]
        return _Selected(ex, packed, seed)

    @staticmethod
    def _fill_sparse(target: torch.Tensor, has: torch.Tensor, indptr: np.ndarray, ids: np.ndarray,
                     start: int, end: int, output_start: int) -> None:
        for local, gene in enumerate(range(start, end)):
            begin, finish = int(indptr[gene]), int(indptr[gene + 1])
            if finish <= begin:
                continue
            labels = torch.as_tensor(np.asarray(ids[begin:finish], dtype=np.int64), dtype=torch.long)
            target[output_start + local, labels] = 1.0
            has[output_start + local] = True


def _stable_seed(seed: int, mag_id: str) -> int:
    digest = hashlib.sha256("{}:{}".format(seed, mag_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def move_batch_to_device(batch: Mapping[str, object], device: str) -> Dict[str, object]:
    """Nonblocking transfer that preserves shared tensor identity.

    `esmc_vecs` and `esmc_target` may be the same host tensor.  Caching by
    object identity ensures only one H2D copy is made.
    """
    moved = {}

    def move(value):
        if torch.is_tensor(value):
            key = id(value)
            if key not in moved:
                moved[key] = value.to(device, non_blocking=True)
            return moved[key]
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    return {key: move(value) for key, value in batch.items()}
