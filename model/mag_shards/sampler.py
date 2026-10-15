"""Deterministic shard-aware token-budget batching for DDP."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .reader import MagLocator


def _seed64(*parts: object) -> int:
    text = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


class TokenBudgetBatchSampler:
    """Yield disjoint per-rank batches while preserving shard locality.

    Items are `(dataset_index, sample_seed)` tuples.  The seed makes contig
    selection and masking reproducible without depending on DataLoader worker
    RNG state.
    """

    def __init__(self, locators: Sequence[MagLocator], manifest_sha256: str, max_batch_tokens: int,
                 per_mag_token_budget: int, rank: int = 0, world_size: int = 1,
                 max_batch_mags: Optional[int] = None, bucket_window: int = 256,
                 seed: int = 0, shuffle: bool = True, drop_last_group: bool = True,
                 max_contig_genes: Optional[int] = None,
                 max_contigs_per_mag: Optional[int] = None):
        if max_batch_tokens <= 0 or per_mag_token_budget <= 0:
            raise ValueError("token budgets must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.locators = locators
        self.manifest_sha256 = manifest_sha256
        self.max_batch_tokens = int(max_batch_tokens)
        self.per_mag_token_budget = int(per_mag_token_budget)
        self.rank = int(rank); self.world_size = int(world_size)
        self.max_batch_mags = int(max_batch_mags) if max_batch_mags else None
        self.max_contig_genes = int(max_contig_genes) if max_contig_genes else None
        self.max_contigs_per_mag = (int(max_contigs_per_mag)
                                    if max_contigs_per_mag else None)
        self.bucket_window = max(1, int(bucket_window))
        self.seed = int(seed); self.shuffle = bool(shuffle)
        self.drop_last_group = bool(drop_last_group)
        self.epoch = 0; self.start_global_batch = 0
        self._cache = None

    def set_epoch(self, epoch: int, start_global_batch: int = 0) -> None:
        self.epoch = int(epoch); self.start_global_batch = int(start_global_batch); self._cache = None

    def _cost(self, index: int) -> int:
        loc = self.locators[index]
        n_contigs = min(loc.n_contigs, self.max_contigs_per_mag or loc.n_contigs)
        max_genes = loc.n_genes
        if self.max_contig_genes is not None:
            max_genes = min(max_genes, n_contigs * self.max_contig_genes)
        return min(self.per_mag_token_budget, max_genes + n_contigs)

    def _global_batches(self) -> List[List[int]]:
        by_shard = defaultdict(list)
        for i, loc in enumerate(self.locators):
            by_shard[loc.shard_index].append(i)
        rng = random.Random(_seed64(self.manifest_sha256, self.seed, self.epoch, "shards"))
        shard_ids = sorted(by_shard)
        if self.shuffle:
            rng.shuffle(shard_ids)
        batches = []
        for shard_id in shard_ids:
            indices = by_shard[shard_id]
            if self.shuffle:
                rng.shuffle(indices)
            ordered = []
            for start in range(0, len(indices), self.bucket_window):
                window = indices[start:start + self.bucket_window]
                window.sort(key=self._cost)
                if self.shuffle and rng.random() < 0.5:
                    window.reverse()
                ordered.extend(window)
            batch = []; longest = 0
            for index in ordered:
                item_cost = self._cost(index)
                # Collation produces a dense B x max_length rectangle.  Bound
                # that actual GPU shape, not the sum of unpadded MAG lengths;
                # otherwise a later mix of unequal MAGs can exceed the shape
                # used by automatic memory calibration.
                padded_cost = (len(batch) + 1) * max(longest, item_cost)
                full_by_tokens = bool(batch) and padded_cost > self.max_batch_tokens
                full_by_mags = self.max_batch_mags is not None and len(batch) >= self.max_batch_mags
                if full_by_tokens or full_by_mags:
                    batches.append(batch); batch = []; longest = 0
                batch.append(index); longest = max(longest, item_cost)
            if batch:
                batches.append(batch)
        return batches

    def _rank_batches(self):
        if self._cache is not None:
            return self._cache
        batches = self._global_batches()
        remaining = max(0, len(batches) - self.start_global_batch)
        n_groups = remaining // self.world_size if self.drop_last_group else int(math.ceil(remaining / self.world_size))
        result = []
        for group in range(n_groups):
            group_start = self.start_global_batch + group * self.world_size
            pos = group_start + self.rank
            if pos >= len(batches):
                if self.drop_last_group:
                    break
                continue
            batch = []
            for within, index in enumerate(batches[pos]):
                sample_seed = _seed64(self.manifest_sha256, self.seed, self.epoch, group_start, self.rank, within,
                                      self.locators[index].mag_id)
                batch.append((index, sample_seed))
            result.append(batch)
        self._cache = result
        return result

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        yield from self._rank_batches()

    def __len__(self) -> int:
        return len(self._rank_batches())
