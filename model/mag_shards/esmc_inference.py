"""Exact ESM-C embedding used by incremental shard ingestion."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .schema import ESMC_DIM


class ESMCEmbedder:
    """Lazy, reusable ESM-C model for an entire ingestion run."""

    def __init__(self, model_name="esmc_600m", max_residues=1022,
                 token_budget=131072, max_sequences=1024):
        self.model_name = model_name; self.max_residues = int(max_residues)
        self.token_budget = int(token_budget); self.max_sequences = int(max_sequences)
        self.model = self.tokenizer = None

    def _ensure(self):
        if self.model is not None:
            return
        import torch
        from esm.models.esmc import ESMC
        if not torch.cuda.is_available():
            raise RuntimeError("ESM-C incremental embedding requires a CUDA GPU")
        self.model = ESMC.from_pretrained(self.model_name).to("cuda").eval()
        self.tokenizer = self.model.tokenizer

    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        if not sequences:
            return np.empty((0, ESMC_DIM), dtype=np.float16)
        import torch
        self._ensure(); order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        output = np.empty((len(sequences), ESMC_DIM), dtype=np.float16); cursor = 0
        with torch.no_grad():
            while cursor < len(order):
                n = 1
                while cursor + n < len(order) and n < self.max_sequences:
                    candidate_length = min(len(sequences[order[cursor + n]]), self.max_residues) + 2
                    if (n + 1) * candidate_length > self.token_budget:
                        break
                    n += 1
                while True:
                    chosen = order[cursor:cursor + n]
                    chunk = [sequences[i][:self.max_residues] for i in chosen]
                    encoded = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt",
                                             max_length=self.max_residues + 2)
                    ids = encoded["input_ids"].to("cuda")
                    try:
                        hidden = self.model(sequence_tokens=ids).embeddings
                        break
                    except torch.cuda.OutOfMemoryError:
                        del ids
                        torch.cuda.empty_cache()
                        if n == 1:
                            raise
                        n = max(1, n // 2)
                mask = encoded["attention_mask"].to("cuda").unsqueeze(-1)
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                output[np.asarray(chosen)] = pooled.cpu().numpy().astype(np.float16)
                del hidden, pooled, ids, mask
                cursor += n
        return output

    def close(self):
        if self.model is None:
            return
        import torch
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        torch.cuda.empty_cache()


def embed_sequences(sequences: Sequence[str], model_name: str = "esmc_600m", max_residues: int = 1022,
                    token_budget: int = 131072, max_sequences: int = 1024) -> np.ndarray:
    """One-shot convenience wrapper; ingestion uses reusable `ESMCEmbedder`."""
    embedder = ESMCEmbedder(model_name, max_residues, token_budget, max_sequences)
    try:
        return embedder(sequences)
    finally:
        embedder.close()
