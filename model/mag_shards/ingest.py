"""Append canonical MAG records as immutable shards and publish a child manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from tokenization import NUM_SPECIAL, mode_dir

from .esmc_inference import ESMCEmbedder, embed_sequences
from .manifest import DatasetManifest, ShardRef, load_manifest
from .policies import deterministic_split, ec_top2_labels
from .protein_index import ProteinIndex
from .schema import ESMC_DIM, MISSING_U32, SchemaError, TAXONOMY_RANKS, token_column
from .validation import validate_shard
from .writer import ShardPayload, ShardWriter


_VQVAE_CACHE = {}


def _record_digest(record, hashes: np.ndarray) -> str:
    digest = hashlib.sha256(); cursor = 0
    for contig in record["contigs"]:
        for gene in contig["genes"]:
            digest.update(str(gene["gene_id"]).encode()); digest.update(b"\0")
            digest.update(hashes[cursor].tobytes()); cursor += 1
    return digest.hexdigest()


def _protein_hashes(record):
    sequences = []; hashes = []
    for contig in record["contigs"]:
        for gene in contig["genes"]:
            sequence = gene.get("aa_sequence") or ""
            if not sequence:
                raise SchemaError("{} contains a gene without amino-acid sequence".format(record["mag_id"]))
            sequences.append(sequence); hashes.append(np.frombuffer(hashlib.md5(sequence.encode()).digest(), dtype=np.uint8))
    return np.asarray(hashes, dtype=np.uint8), sequences


class RepresentationResolver:
    def __init__(self, manifest: DatasetManifest, root_overrides: Optional[Mapping[str, str]] = None,
                 embedder: Callable[[Sequence[str]], np.ndarray] = embed_sequences,
                 token_assigner: Optional[Callable[[str, np.ndarray, Sequence[str], np.ndarray], np.ndarray]] = None):
        self.manifest = manifest; self.root_overrides = dict(root_overrides or {})
        self.index = ProteinIndex(manifest, root_overrides)
        self.embedder = embedder; self.token_assigner = token_assigner or assign_discrete
        self._payloads = {}
        self.token_modes = tuple(manifest.representations.get("tokens", {}))

    def _payload(self, shard_index):
        payload = self._payloads.get(shard_index)
        if payload is not None:
            return payload
        ref = self.manifest.shards[shard_index]
        path = self.manifest.resolve_shard(ref, self.root_overrides)
        with (path / "shard.json").open() as handle:
            meta = json.load(handle)
        n_genes = int(meta["n_genes"])
        payload = {"esmc": np.memmap(path / "esmc.f16", dtype="<f2", mode="r", shape=(n_genes, ESMC_DIM)),
                   "tokens": {}}
        for mode in self.token_modes:
            column = path / token_column(mode)
            if column.exists():
                payload["tokens"][mode] = np.memmap(column, dtype="<u4", mode="r")
        self._payloads[shard_index] = payload
        return payload

    def resolve(self, hashes: np.ndarray, sequences: Sequence[str]):
        shard_ids, rows = self.index.lookup(hashes)
        vectors = np.empty((len(hashes), ESMC_DIM), dtype=np.float16)
        tokens = {mode: np.full(len(hashes), MISSING_U32, dtype=np.uint32) for mode in self.token_modes}
        known = shard_ids >= 0
        for shard_index in np.unique(shard_ids[known]):
            positions = np.flatnonzero(shard_ids == shard_index)
            payload = self._payload(int(shard_index)); source_rows = rows[positions]
            order = np.argsort(source_rows)
            sorted_positions = positions[order]; sorted_rows = source_rows[order]
            vectors[sorted_positions] = payload["esmc"][sorted_rows]
            for mode, values in payload["tokens"].items():
                tokens[mode][sorted_positions] = values[sorted_rows]
        new_positions = np.flatnonzero(~known)
        if len(new_positions):
            new_sequences = [sequences[i] for i in new_positions]
            new_vectors = np.asarray(self.embedder(new_sequences), dtype=np.float16)
            if new_vectors.shape != (len(new_positions), ESMC_DIM):
                raise SchemaError("embedder returned unexpected shape {}".format(new_vectors.shape))
            vectors[new_positions] = new_vectors
            for mode in self.token_modes:
                assigned = np.asarray(self.token_assigner(mode, new_vectors, new_sequences, hashes[new_positions]),
                                      dtype=np.uint32)
                if assigned.shape != (len(new_positions),):
                    raise SchemaError("{} assigner returned unexpected shape".format(mode))
                tokens[mode][new_positions] = assigned
        return vectors, tokens, known


def assign_discrete(mode: str, vectors: np.ndarray, sequences: Sequence[str], hashes: np.ndarray) -> np.ndarray:
    if mode == "vq_kmeans":
        codebook_path = mode_dir(mode) / "codebook.npy"
        if not codebook_path.exists():
            return np.full(len(vectors), MISSING_U32, dtype=np.uint32)
        return (_nearest_chunked(vectors, np.load(codebook_path)) + NUM_SPECIAL).astype(np.uint32)
    if mode == "vq_vae":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_frozen_vqvae(ESMC_DIM, device)
            output = []
            with torch.no_grad():
                for start in range(0, len(vectors), 2048):
                    value = torch.from_numpy(np.asarray(vectors[start:start + 2048], dtype=np.float32)).to(device)
                    output.append(model.nearest(model.enc(value)).cpu().numpy())
            return (np.concatenate(output) + NUM_SPECIAL).astype(np.uint32)
        except (FileNotFoundError, ImportError, RuntimeError, KeyError, ValueError):
            return np.full(len(vectors), MISSING_U32, dtype=np.uint32)
    raise SchemaError("unsupported token mode {}".format(mode))


def load_frozen_vqvae(input_dim: int, device: str):
    """Reconstruct the frozen encoder directly from its checkpoint.

    Ingestion intentionally does not import the 1,000-line offline token
    builder.  Shapes are derived from the artifact so this also works with
    future codebook sizes and latent dimensions.
    """
    import torch
    from torch import nn

    checkpoint = mode_dir("vq_vae") / "vqvae.pt"
    key = (str(checkpoint.resolve()), int(checkpoint.stat().st_mtime_ns), str(device))
    if key in _VQVAE_CACHE:
        return _VQVAE_CACHE[key]
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    codebook = state["codebook"]
    latent_dim = int(codebook.shape[1]); codebook_size = int(codebook.shape[0])
    artifact_input_dim = int(state["enc.0.weight"].shape[1])
    if artifact_input_dim != int(input_dim):
        raise ValueError("VQ-VAE expects {}-D input, got {}".format(artifact_input_dim, input_dim))

    class FrozenVQVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.GELU(),
                                     nn.Linear(latent_dim, latent_dim))
            self.dec = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                     nn.Linear(latent_dim, input_dim))
            self.register_buffer("codebook", torch.zeros(codebook_size, latent_dim))
            self.register_buffer("cluster_size", torch.zeros(codebook_size))
            self.register_buffer("embed_sum", torch.zeros(codebook_size, latent_dim))

        def nearest(self, values):
            code_norm = self.codebook.pow(2).sum(1)
            best = torch.full((len(values),), float("inf"), device=values.device)
            ids = torch.zeros(len(values), dtype=torch.long, device=values.device)
            for start in range(0, codebook_size, 4096):
                block = self.codebook[start:start + 4096]
                distance = (values.pow(2).sum(1, keepdim=True) - 2 * values @ block.T
                            + code_norm[start:start + len(block)])
                score, local = distance.min(1); update = score < best
                best = torch.where(update, score, best)
                ids = torch.where(update, local + start, ids)
            return ids

    model = FrozenVQVAE().to(device)
    model.load_state_dict(state); model.eval()
    _VQVAE_CACHE.clear(); _VQVAE_CACHE[key] = model
    return model


def _nearest_chunked(vectors, codebook, vector_chunk=2048, code_chunk=4096):
    try:
        import torch
        if torch.cuda.is_available():
            codes = torch.from_numpy(np.asarray(codebook, dtype=np.float32)).to("cuda")
            code_norm = codes.pow(2).sum(1); output = []
            with torch.no_grad():
                for start in range(0, len(vectors), vector_chunk):
                    x = torch.from_numpy(np.asarray(vectors[start:start + vector_chunk], dtype=np.float32)).to("cuda")
                    x_norm = x.pow(2).sum(1); best = torch.full((len(x),), float("inf"), device="cuda")
                    best_id = torch.zeros(len(x), dtype=torch.long, device="cuda")
                    for code_start in range(0, len(codes), code_chunk):
                        block = codes[code_start:code_start + code_chunk]
                        distance = x_norm[:, None] - 2 * x @ block.T + code_norm[code_start:code_start + len(block)]
                        value, local = distance.min(1); update = value < best
                        best = torch.where(update, value, best)
                        best_id = torch.where(update, local + code_start, best_id)
                    output.append(best_id.cpu().numpy())
            return np.concatenate(output) if output else np.empty(0, dtype=np.int64)
    except ModuleNotFoundError:
        pass
    result = np.empty(len(vectors), dtype=np.int64)
    codebook = np.asarray(codebook, dtype=np.float32); c2 = (codebook * codebook).sum(1)
    for start in range(0, len(vectors), vector_chunk):
        x = np.asarray(vectors[start:start + vector_chunk], dtype=np.float32)
        best = np.full(len(x), np.inf, dtype=np.float32); best_id = np.zeros(len(x), dtype=np.int64)
        x2 = (x * x).sum(1)
        for code_start in range(0, len(codebook), code_chunk):
            block = codebook[code_start:code_start + code_chunk]
            distance = x2[:, None] - 2 * x @ block.T + c2[code_start:code_start + len(block)]
            local = distance.argmin(1); value = distance[np.arange(len(x)), local]
            update = value < best; best[update] = value[update]; best_id[update] = local[update] + code_start
        result[start:start + len(x)] = best_id
    return result


def manifest_mag_digests(manifest: DatasetManifest, root_overrides=None):
    result = {}
    for ref in manifest.shards:
        path = manifest.resolve_shard(ref, root_overrides)
        with (path / "mags.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line); result[row["mag_id"]] = row.get("source_record_sha256")
    return result


def append_records(parent_manifest_path: Path, records: Iterable[Mapping[str, object]], output_manifest: Path,
                   root_alias: str, root_path: Path, relative_dataset_dir: str,
                   target_shard_bytes: int = 4 * 1024 ** 3, split_override: Optional[str] = None,
                   root_overrides: Optional[Mapping[str, str]] = None,
                   embedder: Callable[[Sequence[str]], np.ndarray] = embed_sequences,
                   token_assigner=None, dataset_id: Optional[str] = None,
                   source_metadata: Optional[Mapping[str, Mapping[str, object]]] = None):
    parent = load_manifest(parent_manifest_path)
    existing = manifest_mag_digests(parent, root_overrides)
    owned_embedder = None
    if embedder is embed_sequences:
        owned_embedder = ESMCEmbedder(); embedder = owned_embedder
    resolver = RepresentationResolver(parent, root_overrides, embedder, token_assigner)
    pending = []; pending_bytes = 0; new_refs = []; skipped = 0
    missing_tokens = {mode: 0 for mode in resolver.token_modes}
    next_number = len(parent.shards)
    writer = ShardWriter(root_path, relative_dataset_dir,
                         builder={"parent_manifest_sha256": parent.sha256, "operation": "incremental_ingest"})

    def flush():
        nonlocal pending, pending_bytes, next_number
        if not pending:
            return
        shard_id = "shard-{:06d}".format(next_number)
        payload = build_shard_payload(parent, pending, resolver, split_override)
        for mode, values in payload.tokens.items():
            missing_tokens[mode] += int(np.count_nonzero(values == MISSING_U32))
        path = writer.write(shard_id, payload)
        report = validate_shard(path, verify_hashes=False); meta = json.loads((path / "shard.json").read_text())
        new_refs.append(ShardRef(
            shard_id=shard_id, root=root_alias,
            path="{}/{}".format(relative_dataset_dir.rstrip("/"), shard_id),
            metadata_sha256=report["metadata_sha256"], n_mags=report["n_mags"],
            n_contigs=report["n_contigs"], n_genes=report["n_genes"],
            bytes_by_column={k: int(v["bytes"]) for k, v in meta["columns"].items()}))
        next_number += 1; pending = []; pending_bytes = 0

    seen_this_run = set()
    for record in records:
        hashes, _ = _protein_hashes(record); digest = _record_digest(record, hashes)
        if record["mag_id"] in seen_this_run:
            raise SchemaError("duplicate MAG ID in ingestion input: {}".format(record["mag_id"]))
        seen_this_run.add(record["mag_id"])
        if record["mag_id"] in existing:
            prior = existing[record["mag_id"]]
            if prior is None:
                raise SchemaError("existing MAG {} lacks a source digest; cannot prove idempotence".format(record["mag_id"]))
            if prior != digest:
                raise SchemaError("MAG ID {} already exists with different content".format(record["mag_id"]))
            skipped += 1; continue
        n_genes = len(hashes); cost = n_genes * ESMC_DIM * 2
        if pending and pending_bytes + cost > target_shard_bytes:
            flush()
        pending.append(record); pending_bytes += cost
    flush()
    if owned_embedder is not None:
        owned_embedder.close()
    if not new_refs:
        return parent, {"added_mags": 0, "skipped_mags": skipped, "new_shards": 0}
    roots = dict(parent.storage_roots); roots[root_alias] = str(root_path)
    sources = dict(parent.sources)
    for ref in new_refs:
        path = Path(roots[ref.root]) / ref.path
        with (path / "mags.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line); source = row["source_name"]
                source_meta = dict(sources.get(source, {})); source_meta["n_mags"] = int(source_meta.get("n_mags", 0)) + 1
                source_meta.update(dict((source_metadata or {}).get(source, {})))
                sources[source] = source_meta
    split_counts = dict(parent.split.get("counts", {}))
    for ref in new_refs:
        path = Path(roots[ref.root]) / ref.path
        with (path / "mags.jsonl").open() as handle:
            for line in handle:
                split = json.loads(line)["split"]; split_counts[split] = split_counts.get(split, 0) + 1
    representations = dict(parent.representations)
    token_metadata = {mode: dict(metadata)
                      for mode, metadata in parent.representations.get("tokens", {}).items()}
    for mode, count in missing_tokens.items():
        if count:
            token_metadata[mode]["missing_occurrences"] = (
                int(token_metadata[mode].get("missing_occurrences", 0)) + count)
            token_metadata[mode]["ingestion_warning"] = (
                "some novel proteins lacked a compatible frozen assignment artifact")
    representations["tokens"] = token_metadata
    child = DatasetManifest(
        dataset_id=dataset_id or "{}-g{}".format(parent.dataset_id, parent.generation + 1),
        generation=parent.generation + 1, parent_sha256=parent.sha256,
        storage_roots=roots, shards=list(parent.shards) + new_refs,
        totals={"n_mags": parent.totals["n_mags"] + sum(x.n_mags for x in new_refs),
                "n_contigs": parent.totals["n_contigs"] + sum(x.n_contigs for x in new_refs),
                "n_genes": parent.totals["n_genes"] + sum(x.n_genes for x in new_refs)},
        sources=sources, representations=representations, vocabularies=parent.vocabularies,
        split={**parent.split, "counts": split_counts}, created_utc=datetime.now(timezone.utc).isoformat(),
        builder={"operation": "incremental_ingest", "parent_manifest": str(parent_manifest_path),
                 "parent_sha256": parent.sha256}, validation={"new_shards_validated": len(new_refs)},
        unavailable_token_modes=parent.unavailable_token_modes,
        indexes={"protein": {"type": "per_shard_sorted_v1", "hash": "md5-aa-raw-16bytes",
                             "segments": [ref.shard_id for ref in list(parent.shards) + new_refs]}})
    child.publish(output_manifest)
    return child, {"added_mags": sum(x.n_mags for x in new_refs), "skipped_mags": skipped,
                   "new_shards": len(new_refs), "missing_tokens": missing_tokens}


def build_shard_payload(parent, records, resolver, split_override=None):
    """Convert canonical MAG records into one aligned shard payload."""
    all_hashes = []; all_sequences = []; record_hashes = []
    for record in records:
        hashes, sequences = _protein_hashes(record)
        record_hashes.append(hashes); all_hashes.append(hashes); all_sequences.extend(sequences)
    occurrence_hashes = np.concatenate(all_hashes)
    keys = np.ascontiguousarray(occurrence_hashes).view("S16").reshape(-1)
    unique_keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    unique_hashes = np.ascontiguousarray(unique_keys).view(np.uint8).reshape(-1, 16)
    unique_sequences = [all_sequences[int(i)] for i in first]
    unique_vectors, unique_tokens, _ = resolver.resolve(unique_hashes, unique_sequences)
    vectors = unique_vectors[inverse]
    tokens = {mode: values[inverse] for mode, values in unique_tokens.items()}
    mag_gene = [0]; mag_contig = [0]; contig_gene = [0]; mags = []
    ko_values = []; ko_indptr = [0]; ec_values = []; ec_indptr = [0]
    ko_vocab = parent.vocabularies.get("ko", {}); ec_vocab = parent.vocabularies.get("ec", {})
    tax_vocab = parent.vocabularies.get("taxonomy", {}); cursor = 0
    for record, hashes in zip(records, record_hashes):
        for contig in record["contigs"]:
            contig_gene.append(contig_gene[-1] + len(contig["genes"]))
            for gene in contig["genes"]:
                ko_values.extend(ko_vocab[label] for label in gene.get("ko_ids", []) if label in ko_vocab)
                ko_indptr.append(len(ko_values))
                labels = []
                for raw in gene.get("ec_ids", []):
                    labels.extend(ec_top2_labels(raw))
                for label in dict.fromkeys(labels):
                    if label in ec_vocab:
                        ec_values.append(ec_vocab[label])
                ec_indptr.append(len(ec_values))
        source = record["source_name"]
        if split_override:
            split = split_override
        elif parent.split.get("mode") == "registry":
            raise SchemaError("registry-split datasets require an explicit split for new MAGs")
        else:
            split = deterministic_split(record["mag_id"], source, parent.split["mode"],
                                        int(parent.split.get("seed", 0)), float(parent.split.get("val_frac", 0.05)),
                                        {}, parent.split.get("source_splits", {}))
        taxonomy = {}
        for rank in TAXONOMY_RANKS:
            mapping = tax_vocab.get(rank, {}); label = record.get("taxonomy", {}).get(rank) or "UNKNOWN"
            taxonomy[rank] = mapping.get(label, mapping.get("UNKNOWN", -1))
        n = len(hashes); n_contigs = len(record["contigs"])
        mags.append({"mag_id": record["mag_id"], "source_name": source, "split": split,
                     "n_genes": n, "n_contigs": n_contigs, "taxonomy": taxonomy,
                     "source_record_sha256": _record_digest(record, hashes)})
        cursor += n; mag_gene.append(cursor); mag_contig.append(mag_contig[-1] + n_contigs)
    return ShardPayload(
        mags=mags, mag_gene_offsets=np.asarray(mag_gene, dtype=np.uint64),
        mag_contig_offsets=np.asarray(mag_contig, dtype=np.uint64),
        contig_gene_offsets=np.asarray(contig_gene, dtype=np.uint64),
        esmc=np.asarray(vectors, dtype=np.float16), esmc_valid=np.ones(len(vectors), dtype=bool),
        tokens={mode: np.asarray(values, dtype=np.uint32) for mode, values in tokens.items()},
        protein_hash=occurrence_hashes,
        ko_indptr=np.asarray(ko_indptr, dtype=np.uint64), ko_ids=np.asarray(ko_values, dtype=np.uint32),
        ec_indptr=np.asarray(ec_indptr, dtype=np.uint64), ec_ids=np.asarray(ec_values, dtype=np.uint32))
