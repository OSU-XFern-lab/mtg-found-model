from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mag_shards.ingest import append_records
from mag_shards.manifest import DatasetManifest, ShardRef
from mag_shards.reader import ManifestDataset
from mag_shards.schema import ALL_TOKEN_MODES, ESMC_DIM, InputSpec, SchemaError
from mag_shards.validation import validate_dataset, validate_shard
from mag_shards.writer import ShardPayload, ShardWriter


def _hash(sequence):
    return np.frombuffer(hashlib.md5(sequence.encode()).digest(), dtype=np.uint8)


def _record(mag_id, source, genes):
    return {"mag_id": mag_id, "source_name": source,
            "taxonomy": {"domain": "Bacteria", "phylum": None, "class": None,
                         "order": None, "family": None, "genus": None, "species": None},
            "contigs": [{"contig_id": mag_id + ":0", "genes": [
                {"gene_id": mag_id + ":" + str(i), "aa_sequence": sequence,
                 "ko_ids": ["K1"] if i == 0 else [], "ec_ids": ["1.2.3.4"] if i == 1 else []}
                for i, sequence in enumerate(genes)], "metadata": {}}], "metadata": {}}


class IncrementalIngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        sequences = ["AAA", "BBB"]; hashes = np.vstack([_hash(x) for x in sequences])
        esmc = np.vstack((np.full((1, ESMC_DIM), 1, np.float16),
                          np.full((1, ESMC_DIM), 2, np.float16)))
        tokens = {mode: np.asarray([10 + i, 20 + i], np.uint32) for i, mode in enumerate(ALL_TOKEN_MODES)}
        record = _record("m1", "S1", sequences)
        digest = hashlib.sha256()
        for gene, value in zip(record["contigs"][0]["genes"], hashes):
            digest.update(gene["gene_id"].encode()); digest.update(b"\0"); digest.update(value.tobytes())
        payload = ShardPayload(
            mags=[{"mag_id": "m1", "source_name": "S1", "split": "train", "n_genes": 2,
                   "n_contigs": 1, "taxonomy": {"domain": 0}, "source_record_sha256": digest.hexdigest()}],
            mag_gene_offsets=np.asarray([0, 2], np.uint64),
            mag_contig_offsets=np.asarray([0, 1], np.uint64),
            contig_gene_offsets=np.asarray([0, 2], np.uint64), esmc=esmc, tokens=tokens,
            protein_hash=hashes, ko_indptr=np.asarray([0, 1, 1], np.uint64),
            ko_ids=np.asarray([0], np.uint32), ec_indptr=np.asarray([0, 0, 2], np.uint64),
            ec_ids=np.asarray([0, 1], np.uint32))
        shard = ShardWriter(self.root, "parent").write("shard-000000", payload)
        report = validate_shard(shard, True); meta = json.loads((shard / "shard.json").read_text())
        ref = ShardRef("shard-000000", "root", "parent/shard-000000", report["metadata_sha256"],
                       1, 1, 2, {k: v["bytes"] for k, v in meta["columns"].items()})
        token_meta = {mode: {"size": 100, "special": {"PAD": 0, "MASK": 1, "UCT": 2, "UNK": 3}}
                      for mode in ALL_TOKEN_MODES}
        tax = {rank: ({"Bacteria": 0, "UNKNOWN": 1} if rank == "domain" else {"UNKNOWN": 0})
               for rank in ("domain", "phylum", "class", "order", "family", "genus", "species")}
        manifest = DatasetManifest(
            dataset_id="parent", generation=1, storage_roots={"root": str(self.root)}, shards=[ref],
            totals={"n_mags": 1, "n_contigs": 1, "n_genes": 2}, sources={"S1": {"n_mags": 1}},
            representations={"esmc": {"dim": ESMC_DIM}, "tokens": token_meta},
            vocabularies={"ko": {"K1": 0}, "ec": {"1": 0, "1.2": 1}, "taxonomy": tax},
            split={"mode": "random_mag", "seed": 0, "val_frac": 0.0, "counts": {"train": 1}},
            created_utc=datetime.now(timezone.utc).isoformat(), builder={"test": True},
            indexes={"protein": {"type": "per_shard_sorted_v1", "segments": ["shard-000000"]}})
        self.parent = self.root / "parent.json"; manifest.publish(self.parent)

    def tearDown(self):
        self.tmp.cleanup()

    def test_known_and_new_protein_append(self):
        record = _record("m2", "S2", ["AAA", "CCC"])
        calls = []
        def embedder(sequences):
            calls.extend(sequences)
            return np.full((len(sequences), ESMC_DIM), 9, dtype=np.float16)
        def assigner(mode, vectors, sequences, hashes):
            return np.full(len(sequences), 70 + ALL_TOKEN_MODES.index(mode), dtype=np.uint32)
        output = self.root / "child.json"
        child, report = append_records(
            self.parent, [record], output, "root", self.root, "child-data",
            target_shard_bytes=10 * ESMC_DIM, embedder=embedder, token_assigner=assigner,
            dataset_id="child")
        self.assertEqual(report["added_mags"], 1); self.assertEqual(calls, ["CCC"])
        self.assertEqual(report["missing_tokens"], {mode: 0 for mode in ALL_TOKEN_MODES})
        validate_dataset(output, verify_hashes=True)
        spec = InputSpec("vq_kmeans", {"gene_id": 1.0, "patch_disc": 1.0})
        dataset = ManifestDataset(output, spec, sources=["S2"]); example = dataset[0]
        self.assertTrue(np.all(example.esmc[0] == 1))
        self.assertTrue(np.all(example.esmc[1] == 9))
        mode_index = ALL_TOKEN_MODES.index("vq_kmeans")
        self.assertEqual(example.tokens.tolist(), [10 + mode_index, 70 + mode_index])

        _, second = append_records(output, [record], self.root / "unused.json", "root", self.root,
                                   "unused", embedder=embedder, token_assigner=assigner)
        self.assertEqual(second["added_mags"], 0); self.assertEqual(second["skipped_mags"], 1)

    def test_conflicting_mag_id_is_rejected(self):
        conflict = _record("m1", "S1", ["AAA", "DIFFERENT"])
        with self.assertRaises(SchemaError):
            append_records(self.parent, [conflict], self.root / "bad.json", "root", self.root,
                           "bad", embedder=lambda x: np.zeros((len(x), ESMC_DIM), np.float16),
                           token_assigner=lambda *x: np.zeros(len(x[1]), np.uint32))

if __name__ == "__main__":
    unittest.main()
