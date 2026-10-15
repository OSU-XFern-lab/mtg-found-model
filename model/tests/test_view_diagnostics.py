"""Small CPU checks for MAG views and additive patch-discrimination metrics."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from losses import patch_disc_loss, patch_disc_metrics
from mag_shards.collate import CollateConfig, ShardCollator
from mag_shards.schema import InputSpec


class ViewDiagnosticsTest(unittest.TestCase):
    def select(self, lengths, config, seed=0):
        ex = SimpleNamespace(sample_seed=seed, mag_id="test",
                             contig_gene_offsets=np.r_[0, np.cumsum(lengths)])
        return ShardCollator(InputSpec("esmc", {"patch_disc": 1}), config)._select(ex).ranges

    def test_anchored_crop_and_deterministic_evaluation(self):
        config = CollateConfig(max_contig_genes=512)
        for seed in range(20):
            anchor = np.random.RandomState(seed).randint(2000)
            start = max(0, min(anchor - 256, 1488))
            self.assertEqual(self.select([2000], config, seed), [(start, start + 512)])
        self.assertEqual(self.select([2000], replace(config, train=False), 1),
                         self.select([2000], replace(config, train=False), 2))

    def test_dropout_only_for_fitting_well_supported_mags(self):
        config = CollateConfig(per_mag_token_budget=1000, contig_dropout_apply_prob=1,
                               contig_dropout_range=(0.1, 0.1))
        observed_drop = False
        for seed in range(30):
            kept = self.select([10] * 20, config, seed)
            self.assertGreaterEqual(len(kept), 10)
            observed_drop |= len(kept) < 20
            over = replace(config, per_mag_token_budget=55)
            self.assertEqual(self.select([10] * 20, over, seed), self.select(
                [10] * 20, replace(over, contig_dropout_apply_prob=0), seed))
            self.assertEqual(len(self.select([10] * 3, config, seed)), 3)
            self.assertEqual(len(self.select([10] * 20, replace(config, train=False), seed)), 20)
        self.assertTrue(observed_drop)

    def test_metrics_and_loss_gradient_unchanged(self):
        target = torch.eye(4).unsqueeze(0)
        pred = target.clone().requires_grad_()
        mask = torch.ones(1, 4, dtype=torch.bool)
        ids = torch.zeros(1, 4, dtype=torch.long)
        types = torch.ones_like(ids)
        plain = patch_disc_loss(pred, target, mask, ids, types)
        expected_grad, = torch.autograd.grad(plain, pred)
        diagnostics = {}
        monitored = patch_disc_loss(pred, target, mask, ids, types, diagnostics=diagnostics)
        actual_grad, = torch.autograd.grad(monitored, pred)
        torch.testing.assert_close(monitored, plain)
        torch.testing.assert_close(actual_grad, expected_grad)
        metrics = patch_disc_metrics(diagnostics["sums"])
        self.assertEqual(len(metrics), 10)
        for name in ("top1", "top5", "positive_cosine", "margin", "mask_ratio"):
            self.assertAlmostEqual(metrics["patch_disc/" + name], 1.0)
        self.assertEqual(metrics["patch_disc/candidate_count"], 4)
        self.assertAlmostEqual(metrics["patch_disc/prediction_feature_std"], 0.5)
        self.assertFalse(diagnostics["sums"].requires_grad)

    def test_empty_mask_and_singletons(self):
        for masked in (False, True):
            diagnostics = {}
            z = torch.ones(1, 1, 4, requires_grad=True)
            loss = patch_disc_loss(z, z.detach(), torch.full((1, 1), masked),
                                   torch.zeros(1, 1, dtype=torch.long),
                                   torch.ones(1, 1, dtype=torch.long), diagnostics=diagnostics)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            metrics = patch_disc_metrics(diagnostics["sums"])
            self.assertTrue(np.isnan(metrics["patch_disc/top1"]))
            self.assertEqual(metrics["patch_disc/mask_ratio"], float(masked))

    def test_additive_metrics_match_unequal_rank_partition(self):
        torch.manual_seed(7)
        prediction = torch.randn(3, 6, 8)
        target = torch.randn_like(prediction)
        mask = torch.rand(3, 6) < 0.5
        ids = torch.zeros(3, 6, dtype=torch.long)
        types = torch.ones_like(ids)
        all_stats = {}
        patch_disc_loss(prediction, target, mask, ids, types, diagnostics=all_stats)
        pieces = []
        for rows in (slice(0, 1), slice(1, 3)):
            stats = {}
            patch_disc_loss(prediction[rows], target[rows], mask[rows], ids[rows],
                            types[rows], diagnostics=stats)
            pieces.append(stats["sums"])
        expected = patch_disc_metrics(all_stats["sums"])
        actual = patch_disc_metrics(sum(pieces))
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key], places=6)
