from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mag_shards.manifest import DatasetManifest, ShardRef, sha256_file
from mag_shards.reader import ManifestDataset
from mag_shards.protein_index import ProteinIndex
from mag_shards.sampler import TokenBudgetBatchSampler
from mag_shards.schema import ALL_TOKEN_MODES, ESMC_DIM, InputSpec, MISSING_U32, SchemaError
from mag_shards.validation import validate_dataset, validate_shard
from mag_shards.writer import ShardPayload, ShardWriter

try:
    import torch
    from mag_shards.collate import CollateConfig, ShardCollator, move_batch_to_device
    HAS_TORCH = True
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    HAS_TORCH = False


class ShardRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_rel = "datasets/test-v1"
        mags = [
            {"mag_id": "a", "source_name": "MOTUS", "split": "train", "n_genes": 3,
             "n_contigs": 2, "taxonomy": {"domain": 0}},
            {"mag_id": "b", "source_name": "IMG_JGI", "split": "val", "n_genes": 2,
             "n_contigs": 1, "taxonomy": {"domain": 1}},
        ]
        esmc = np.arange(5 * ESMC_DIM, dtype=np.float16).reshape(5, ESMC_DIM)
        tokens = {
            mode: (np.arange(5, dtype=np.uint32) + 4 + i * 10)
            for i, mode in enumerate(ALL_TOKEN_MODES)
        }
        hashes = np.arange(5 * 16, dtype=np.uint8).reshape(5, 16)
        payload = ShardPayload(
            mags=mags,
            mag_gene_offsets=np.asarray([0, 3, 5], dtype=np.uint64),
            mag_contig_offsets=np.asarray([0, 2, 3], dtype=np.uint64),
            contig_gene_offsets=np.asarray([0, 2, 3, 5], dtype=np.uint64),
            esmc=esmc,
            esmc_valid=np.asarray([True, False, True, True, True]),
            tokens=tokens,
            protein_hash=hashes,
            ko_indptr=np.asarray([0, 1, 1, 3, 3, 4], dtype=np.uint64),
            ko_ids=np.asarray([2, 3, 4, 1], dtype=np.uint32),
            ec_indptr=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint64),
            ec_ids=np.asarray([7, 8], dtype=np.uint32),
        )
        self.shard_path = ShardWriter(self.root, self.dataset_rel, builder={"test": True}).write(
            "shard-000000", payload)
        report = validate_shard(self.shard_path, verify_hashes=True)
        with (self.shard_path / "shard.json").open() as handle:
            meta = json.load(handle)
        ref = ShardRef(
            shard_id="shard-000000", root="test", path="{}/shard-000000".format(self.dataset_rel),
            metadata_sha256=report["metadata_sha256"], n_mags=2, n_contigs=3, n_genes=5,
            bytes_by_column={k: int(v["bytes"]) for k, v in meta["columns"].items()},
        )
        manifest = DatasetManifest(
            dataset_id="test-v1", generation=1, storage_roots={"test": str(self.root)}, shards=[ref],
            totals={"n_mags": 2, "n_contigs": 3, "n_genes": 5},
            sources={"MOTUS": {}, "IMG_JGI": {}},
            representations={"esmc": {"dim": ESMC_DIM},
                             "tokens": {mode: {"size": int(tokens[mode].max()) + 1}
                                        for mode in ALL_TOKEN_MODES}},
            vocabularies={"taxonomy": {"domain": {"d__A": 0, "d__B": 1}},
                          "ko": {"K1": 0}, "ec": {"1.1": 0}},
            split={"mode": "registry", "counts": {"train": 1, "val": 1}},
            created_utc=datetime.now(timezone.utc).isoformat(), builder={"test": True},
            indexes={"protein": {"type": "per_shard_sorted_v1", "segments": ["shard-000000"]}})
        self.manifest_path = self.root / "manifest.json"
        manifest.publish(self.manifest_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_retired_sequence_cluster_modes_are_rejected(self):
        for mode in ("cluster97", "clusterX"):
            with self.assertRaises(SchemaError):
                InputSpec(mode, {"gene_id": 1.0})

    def test_validation_and_continuous_read(self):
        report = validate_dataset(self.manifest_path, verify_hashes=True)
        self.assertEqual(report["totals"]["n_genes"], 5)
        spec = InputSpec("esmc", {"patch_disc": 1.0, "mag_tax": 1.0})
        ds = ManifestDataset(self.manifest_path, spec, splits=["train"])
        self.assertEqual(len(ds), 1)
        ex = ds[0]
        self.assertEqual(ex.mag_id, "a")
        self.assertEqual(ex.contig_gene_offsets.tolist(), [0, 2, 3])
        self.assertEqual(ex.esmc.shape, (3, ESMC_DIM))
        self.assertEqual(ex.esmc_valid.tolist(), [True, False, True])
        handle = ds._handles[0]
        self.assertIsNone(handle.tokens)
        self.assertIsNone(handle.ko_indptr)

    def test_filtered_views_reuse_one_verified_catalog(self):
        catalog = ManifestDataset(self.manifest_path, InputSpec("esmc", {"mag_tax": 1.0}))
        train = catalog.select(("train",), ("MOTUS",))
        validation = catalog.select(("val",), ("IMG_JGI",))
        self.assertEqual([item.mag_id for item in train.locators], ["a"])
        self.assertEqual([item.mag_id for item in validation.locators], ["b"])
        self.assertIs(train.manifest, catalog.manifest)
        self.assertIs(train._shard_paths, catalog._shard_paths)
        self.assertEqual(train._handles, {})

    def test_every_discrete_mode_and_active_annotations(self):
        for mode in ALL_TOKEN_MODES:
            spec = InputSpec(mode, {"gene_id": 1.0, "ko": 1.0, "ec": 1.0})
            ds = ManifestDataset(self.manifest_path, spec, sources=["IMG_JGI"])
            ex = ds[0]
            self.assertEqual(ex.mag_id, "b")
            self.assertEqual(len(ex.tokens), 2)
            self.assertIsNone(ex.esmc)
            self.assertEqual(ex.ko_indptr.tolist(), [0, 0, 1])
            self.assertEqual(ex.ko_ids.tolist(), [1])
            self.assertEqual(ex.ec_indptr.tolist(), [0, 1, 1])

    def test_missing_requested_column_fails_before_training(self):
        mode = ALL_TOKEN_MODES[0]
        token_path = self.shard_path / "token.{}.u32".format(mode)
        token_path.rename(token_path.with_suffix(".missing"))
        with self.assertRaises(SchemaError):
            ManifestDataset(self.manifest_path, InputSpec(mode, {"gene_id": 1.0}))

    @unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
    def test_objective_aware_collation_and_shared_esmc(self):
        spec = InputSpec("esmc", {"mag_tax": 1.0, "patch_disc": 1.0})
        ds = ManifestDataset(self.manifest_path, spec)
        examples = [ds[(0, 123)], ds[(1, 456)]]
        batch = ShardCollator(spec, CollateConfig(train=True, mask_rate_min=1.0, mask_rate_max=1.0))(
            examples)
        self.assertIn("tax_label", batch)
        self.assertNotIn("ko_tgt", batch)
        self.assertNotIn("ec_tgt", batch)
        self.assertNotIn("gene_id_tgt", batch)
        self.assertIs(batch["esmc_vecs"], batch["esmc_target"])
        moved = move_batch_to_device(batch, "cpu")
        self.assertIs(moved["esmc_vecs"], moved["esmc_target"])
        # The invalid ESM row is never selected for patch discrimination.
        self.assertFalse(bool(moved["mlm_gene"][0, 2]))

    @unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
    def test_structural_caps_and_optional_stochastic_dropout(self):
        spec = InputSpec("esmc", {"patch_disc": 1.0})
        example = ManifestDataset(self.manifest_path, spec, splits=["train"])[(0, 123)]
        untouched = ShardCollator(
            spec,
            CollateConfig(
                per_mag_token_budget=None,
                max_contig_genes=1,
                max_contigs_per_mag=2,
                contig_dropout_range=(1.0, 1.0),
                contig_dropout_apply_prob=0.0,
                train=True,
            ),
        )([example])
        dropped = ShardCollator(
            spec,
            CollateConfig(
                per_mag_token_budget=None,
                max_contig_genes=1,
                max_contigs_per_mag=2,
                contig_dropout_range=(1.0, 1.0),
                contig_dropout_apply_prob=1.0,
                train=True,
            ),
        )([example])
        self.assertEqual(int(untouched["token_type"].eq(0).sum()), 2)
        self.assertEqual(int(untouched["token_type"].eq(1).sum()), 2)
        # Few-contig MAGs are protected, even with dropout explicitly enabled.
        self.assertEqual(int(dropped["token_type"].eq(0).sum()), 2)
        self.assertEqual(int(dropped["token_type"].eq(1).sum()), 2)

    @unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
    def test_explicit_token_budget_packs_only_whole_contigs(self):
        spec = InputSpec("esmc", {"patch_disc": 1.0})
        example = ManifestDataset(self.manifest_path, spec, splits=["train"])[(0, 123)]
        batch = ShardCollator(
            spec,
            CollateConfig(
                per_mag_token_budget=2,
                max_contig_genes=2,
                max_contigs_per_mag=None,
                train=False,
            ),
        )([example])
        # The two-gene contig costs three tokens and is skipped; the later
        # one-gene contig fits as one UCT plus one gene.
        self.assertEqual(int(batch["token_type"].eq(0).sum()), 1)
        self.assertEqual(int(batch["token_type"].eq(1).sum()), 1)

    @unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
    def test_discrete_collation_builds_only_requested_targets(self):
        mode = ALL_TOKEN_MODES[-1]
        spec = InputSpec(mode, {"gene_id": 1.0, "ko": 1.0})
        ds = ManifestDataset(self.manifest_path, spec)
        batch = ShardCollator(spec, CollateConfig(train=False, mask_rate_min=1.0, mask_rate_max=1.0),
                              num_ko=10)([ds[0]])
        self.assertIn("gene_id_tgt", batch)
        self.assertIn("ko_tgt", batch)
        self.assertNotIn("ec_tgt", batch)
        self.assertNotIn("esmc_vecs", batch)
        self.assertEqual(batch["ko_tgt"].shape, batch["token_type"].shape)
        self.assertEqual(batch["ko_tgt"].dtype, torch.long)
        # First gene has KO 2, second has none, third has two distinct KOs.
        self.assertEqual(batch["ko_tgt"][batch["token_type"].eq(1)].tolist(), [2, -100, -100])
        self.assertEqual(int(batch["ko_has"].sum()), 1)

    @unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
    def test_ko_duplicate_label_is_not_ambiguous_and_ec_unchanged(self):
        from dataclasses import replace
        spec = InputSpec("esmc", {"ko": 1.0, "ec": 1.0})
        example = ManifestDataset(self.manifest_path, spec)[0]
        example = replace(example, ko_ids=np.asarray([2, 3, 3], dtype=np.uint32))
        batch = ShardCollator(spec, CollateConfig(train=False, mask_ko=True,
                              mask_rate_min=1.0, mask_rate_max=1.0),
                              num_ko=10, num_ec=10)([example])
        genes = batch["token_type"].eq(1)
        self.assertEqual(batch["ko_tgt"][genes].tolist(), [2, -100, 3])
        self.assertEqual(int(batch["mlm_ko"].sum()), 2)
        self.assertEqual(tuple(batch["ec_tgt"].shape), (*batch["token_type"].shape, 10))
        self.assertEqual(batch["ec_tgt"][genes][:, 7].tolist(), [0.0, 1.0, 0.0])

    def test_ddp_sampler_is_disjoint_and_reproducible(self):
        spec = InputSpec("esmc", {"mag_tax": 1.0})
        ds = ManifestDataset(self.manifest_path, spec)
        s0 = TokenBudgetBatchSampler(ds.locators, ds.manifest_sha256, 100, 100,
                                     rank=0, world_size=2, seed=4, shuffle=False)
        s1 = TokenBudgetBatchSampler(ds.locators, ds.manifest_sha256, 100, 100,
                                     rank=1, world_size=2, seed=4, shuffle=False)
        a = list(s0); b = list(s1)
        self.assertEqual(a, list(s0))
        ai = {x[0] for batch in a for x in batch}; bi = {x[0] for batch in b for x in batch}
        self.assertFalse(ai & bi)

    def test_sampler_bounds_the_padded_gpu_rectangle(self):
        spec = InputSpec("esmc", {"mag_tax": 1.0})
        ds = ManifestDataset(self.manifest_path, spec)
        sampler = TokenBudgetBatchSampler(
            ds.locators, ds.manifest_sha256,
            max_batch_tokens=8, per_mag_token_budget=100,
            shuffle=False, drop_last_group=False,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            costs = [sampler._cost(index) for index, _ in batch]
            self.assertLessEqual(len(costs) * max(costs), 8)

    def test_validation_sampler_covers_every_mag_once(self):
        spec = InputSpec("esmc", {"mag_tax": 1.0})
        ds = ManifestDataset(self.manifest_path, spec)
        samplers = [
            TokenBudgetBatchSampler(ds.locators, ds.manifest_sha256, 1, 100,
                                    rank=rank, world_size=3, shuffle=False,
                                    drop_last_group=False)
            for rank in range(3)
        ]
        indices = [item[0] for sampler in samplers for batch in sampler for item in batch]
        self.assertEqual(sorted(indices), list(range(len(ds))))

    def test_protein_index_lookup(self):
        from mag_shards.manifest import load_manifest
        manifest = load_manifest(self.manifest_path)
        index = ProteinIndex(manifest)
        known = np.vstack((np.arange(16, dtype=np.uint8), np.arange(48, 64, dtype=np.uint8)))
        unknown = np.full((1, 16), 255, dtype=np.uint8)
        shards, rows = index.lookup(np.vstack((known, unknown)))
        self.assertEqual(shards.tolist(), [0, 0, -1])
        self.assertEqual(rows[:2].tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
