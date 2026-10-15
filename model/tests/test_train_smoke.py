from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mag_shards.manifest import DatasetManifest, ShardRef
from mag_shards.schema import ESMC_DIM
from mag_shards.validation import validate_shard
from mag_shards.writer import ShardPayload, ShardWriter
from split_plans import compile_stored_labels
from train_config import BatchConfig, train_config_from_dict

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
class TrainSmokeTest(unittest.TestCase):
    def test_mag_limit_resolves_independent_ceilings(self):
        structural = BatchConfig(
            per_mag_token_budget=None, max_contig_genes=512, max_contigs_per_mag=32)
        explicit = BatchConfig(
            per_mag_token_budget=12000, max_contig_genes=512, max_contigs_per_mag=None)
        combined = BatchConfig(
            per_mag_token_budget=20000, max_contig_genes=512, max_contigs_per_mag=32)
        self.assertEqual(structural.resolved_max_mag_tokens, 16416)
        self.assertEqual(explicit.resolved_max_mag_tokens, 12000)
        self.assertEqual(combined.resolved_max_mag_tokens, 16416)

    def test_two_step_all_objective_train(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name); n_mags = 6; genes_per = 3; n_genes = n_mags * genes_per
            mags = []
            for i in range(n_mags):
                mags.append({"mag_id": "m{}".format(i), "source_name": "S", "split": "train" if i < 4 else "val",
                             "n_genes": genes_per, "n_contigs": 1, "taxonomy": {"domain": i % 2},
                             "source_record_sha256": hashlib.sha256(str(i).encode()).hexdigest()})
            gene_offsets = np.arange(0, n_genes + 1, genes_per, dtype=np.uint64)
            hashes = np.vstack([np.frombuffer(hashlib.md5(str(i).encode()).digest(), dtype=np.uint8)
                                for i in range(n_genes)])
            payload = ShardPayload(
                mags=mags, mag_gene_offsets=gene_offsets,
                mag_contig_offsets=np.arange(n_mags + 1, dtype=np.uint64),
                contig_gene_offsets=gene_offsets.copy(),
                esmc=np.random.RandomState(0).normal(size=(n_genes, ESMC_DIM)).astype(np.float16),
                tokens={"vq_kmeans": (np.arange(n_genes, dtype=np.uint32) % 12) + 4},
                protein_hash=hashes,
                ko_indptr=np.arange(n_genes + 1, dtype=np.uint64),
                ko_ids=np.arange(n_genes, dtype=np.uint32) % 2,
                ec_indptr=np.arange(n_genes + 1, dtype=np.uint64),
                ec_ids=np.arange(n_genes, dtype=np.uint32) % 2)
            shard = ShardWriter(root, "data").write("shard-000000", payload)
            report = validate_shard(shard, True); meta = json.loads((shard / "shard.json").read_text())
            ref = ShardRef("shard-000000", "root", "data/shard-000000", report["metadata_sha256"],
                           n_mags, n_mags, n_genes, {k: v["bytes"] for k, v in meta["columns"].items()})
            manifest = DatasetManifest(
                dataset_id="smoke", generation=1, storage_roots={"root": str(root)}, shards=[ref],
                totals={"n_mags": n_mags, "n_contigs": n_mags, "n_genes": n_genes},
                sources={"S": {"n_mags": n_mags}},
                representations={"esmc": {"dim": ESMC_DIM},
                                 "tokens": {"vq_kmeans": {"size": 16}}},
                vocabularies={"ko": {"K0": 0, "K1": 1}, "ec": {"1": 0, "2": 1},
                              "taxonomy": {"domain": {"A": 0, "B": 1}}},
                split={"mode": "random_mag", "counts": {"train": 4, "val": 2}},
                created_utc=datetime.now(timezone.utc).isoformat(), builder={"test": True},
                indexes={"protein": {"type": "per_shard_sorted_v1", "segments": ["shard-000000"]}})
            manifest_path = root / "manifest.json"; manifest.publish(manifest_path)
            split_plan_path = compile_stored_labels(manifest_path, root / "split.json")
            config = {
                "mode": "vq_kmeans",
                "data": {
                    "manifest": str(manifest_path), "split_plan": str(split_plan_path),
                    "train": {"sources": ["S"], "labels": ["train"]},
                    "validation": {"sources": ["S"], "labels": ["val"]},
                    "test": {"sources": [], "labels": ["test"]},
                },
                "objectives": {"weights": {"gene_id": 1, "ko": 1, "ec": 1, "mag_tax": 1, "patch_disc": 1},
                               "production_approved": ["gene_id", "ko", "ec", "mag_tax", "patch_disc"],
                               "taxonomy_ranks": ["domain"], "mask_gene": True,
                               "mask_rate_min": 1.0, "mask_rate_max": 1.0},
                "model": {"d_model": 32, "n_layers": 1, "n_heads": 4, "ff_mult": 2,
                          "dropout": 0.0, "use_checkpoint": False, "max_local_pos": 8},
                "batch": {"per_mag_token_budget": 8,
                          "max_contig_genes": 8, "bucket_window": 8},
                "hardware": {"per_device_batch_tokens": 8, "num_workers": 0,
                             "prefetch_factor": 2, "pin_memory": False},
                "optim": {"lr": 0.001, "weight_decay": 0.0, "max_steps": 2, "target_epochs": None,
                          "warmup_steps": 1, "grad_clip": 1.0, "amp_dtype": "none"},
                "runtime": {"checkpoint_root": str(root / "checkpoints"), "resume": False, "validate": True,
                            "max_val_batches": 2, "log_every": 1, "val_every": 0,
                            "seed": 0, "wandb_mode": "disabled", "process_group_timeout_seconds": 60,
                            "wandb_project": "test"}}
            import train_mag_encoder
            base_config = train_config_from_dict(config)
            train_mag_encoder.main(base_config, force_no_resume=True)
            checkpoints = list((root / "checkpoints").glob("*/last.pt"))
            self.assertEqual(len(checkpoints), 1)
            self.assertTrue(checkpoints[0].with_name("best.pt").is_file())
            records = [json.loads(line) for line in checkpoints[0].with_name(
                "patch_disc_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["step"] for record in records], [1, 2])
            self.assertTrue(all(len(record) == 11 for record in records))
            state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["manifest_sha256"], manifest.sha256)
            self.assertIn("head_spec", state)

            stop_file = root / "stop.requested"
            ack_file = root / "stop.ack.json"
            stop_config = replace(
                base_config,
                optim=replace(base_config.optim, max_steps=100),
                runtime=replace(
                    base_config.runtime,
                    checkpoint_root=str(root / "stop-checkpoints"),
                    resume=True,
                    validate=False,
                ),
            )
            old_stop = train_mag_encoder.STOP_FILE
            old_ack = train_mag_encoder.STOP_ACK_FILE
            try:
                train_mag_encoder.STOP_FILE = str(stop_file)
                train_mag_encoder.STOP_ACK_FILE = str(ack_file)
                stop_file.touch()
                train_mag_encoder.main(stop_config, force_no_resume=True)
                stopped = list((root / "stop-checkpoints").glob("*/last.pt"))
                self.assertEqual(len(stopped), 1)
                first = torch.load(stopped[0], map_location="cpu", weights_only=False)
                acknowledgement = json.loads(ack_file.read_text())
                self.assertEqual(first["step"], 1)
                self.assertEqual(acknowledgement["step"], 1)
                self.assertGreater(acknowledgement["checkpoint_bytes"], 0)

                stop_file.unlink(); ack_file.unlink()
                stop_file.touch()
                train_mag_encoder.main(stop_config, force_no_resume=False)
                resumed = torch.load(stopped[0], map_location="cpu", weights_only=False)
                self.assertEqual(resumed["step"], 2)
                self.assertEqual(json.loads(ack_file.read_text())["step"], 2)
            finally:
                train_mag_encoder.STOP_FILE = old_stop
                train_mag_encoder.STOP_ACK_FILE = old_ack


if __name__ == "__main__":
    unittest.main()
