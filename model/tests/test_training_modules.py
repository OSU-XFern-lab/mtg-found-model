from __future__ import annotations

import tempfile
import unittest
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is only installed in the training environment")
class TrainingModuleTest(unittest.TestCase):
    def test_ko_multiclass_ignores_missing_and_ambiguous_targets(self):
        from losses import ko_loss
        logits = torch.randn(1, 3, 5, requires_grad=True)
        targets = torch.tensor([[2, -100, -100]])
        where = torch.ones_like(targets, dtype=torch.bool)
        loss = ko_loss(logits, targets, where)
        self.assertTrue(torch.allclose(loss, F.cross_entropy(logits[:, 0], targets[:, 0])))
        loss.backward()
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0.0)
        empty = ko_loss(logits, torch.full_like(targets, -100), where)
        self.assertEqual(float(empty.detach()), 0.0)
        self.assertTrue(empty.requires_grad)

    def test_removed_sampling_option_loads_only_when_disabled(self):
        from train_config import base_train_config, train_config_from_dict
        value = base_train_config().to_dict()
        value["objectives"]["weights"]["patch_disc"] = 1.0
        value["objectives"]["production_approved"] = ["patch_disc"]
        self.assertNotIn("sample_mag_ratio", value["batch"])
        value["batch"]["sample_mag_ratio"] = None
        restored = train_config_from_dict(value)
        self.assertFalse(hasattr(restored.batch, "sample_mag_ratio"))
        value["batch"]["sample_mag_ratio"] = [0.5, 1.0]
        with self.assertRaisesRegex(ValueError, "sample_mag_ratio"):
            train_config_from_dict(value)

    def test_ko_probe_uses_integer_classes(self):
        import probe
        data = {"gene": torch.randn(3, 4), "ko_tgt": torch.tensor([0, -100, 2]),
                "ko_has": torch.tensor([True, False, True])}
        with patch.object(probe, "DEVICE", "cpu"), patch.object(probe, "PROBE_EPOCHS", 1):
            result = probe._probe_ko(data, data, 3)
        self.assertIn("multiclass accuracy=", result)
        self.assertIn("n=2", result)

    def test_wandb_generates_and_resumes_the_run_nickname(self):
        from train_config import base_train_config
        from train_mag_encoder import _wandb_start

        with tempfile.TemporaryDirectory() as tmp_name:
            config = base_train_config()
            config = replace(
                config,
                runtime=replace(config.runtime, checkpoint_root=tmp_name, wandb_mode="online"),
            )
            calls = []
            run = SimpleNamespace(id="abc123", name="bright-tree-7")
            fake_wandb = SimpleNamespace(
                init=lambda **kwargs: calls.append(kwargs) or run,
                define_metric=lambda *args, **kwargs: None,
            )
            split = SimpleNamespace(sha256="split-sha", table_sha256="table-sha")
            identity_path = Path(tmp_name) / ".wandb_runs" / "identity.json"
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                _, identity = _wandb_start(
                    config, identity_path, "manifest-sha", split, {}, 10, 20, {},
                    {"effective_global_batch_tokens": 8})
                self.assertEqual(identity["name"], "bright-tree-7")
                self.assertNotIn("name", calls[0])
                self.assertNotIn("id", calls[0])

                checkpoint = Path(tmp_name) / identity["name"] / "last.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                _wandb_start(
                    config, identity_path, "manifest-sha", split, {}, 10, 20, {},
                    {"effective_global_batch_tokens": 8})
                self.assertEqual(calls[1]["id"], "abc123")
                self.assertEqual(calls[1]["resume"], "must")

    def test_hardware_profiles_are_machine_specific(self):
        from train_config import base_train_config

        a40 = base_train_config("a40").hardware
        h100_40g = base_train_config("h100_40g").hardware
        h100_80g = base_train_config("h100_80g").hardware
        h200 = base_train_config("h200").hardware
        self.assertNotEqual(a40.per_device_batch_tokens,
                            h200.per_device_batch_tokens)
        self.assertIsNone(a40.max_per_device_batch_tokens)
        self.assertLess(h100_40g.per_device_batch_tokens,
                        h100_80g.per_device_batch_tokens)
        self.assertEqual(h100_40g.max_per_device_batch_tokens, 131072)
        self.assertEqual(h100_80g.num_workers, 6)
        self.assertEqual(h200.max_per_device_batch_tokens, 131072)

    def test_dense_production_objectives_require_opt_in(self):
        from train_config import train_config_from_dict
        value = {
            "mode": "vq_kmeans",
            "data": {
                "manifest": "manifest.json", "split_plan": "split.json",
                "train": {}, "validation": {}, "test": {},
            },
            "objectives": {"weights": {"gene_id": 1.0}},
            "batch": {"global_batch_tokens": 8},
            "hardware": {"per_device_batch_tokens": 8},
            "runtime": {"checkpoint_root": "checkpoints", "wandb_mode": "disabled"},
        }
        with self.assertRaisesRegex(ValueError, "production_approved"):
            train_config_from_dict(value)

    def test_rng_state_round_trip(self):
        from train_runtime import capture_rng_state, restore_rng_state
        torch.manual_seed(123); state = capture_rng_state()
        expected = torch.rand(4)
        torch.manual_seed(999)
        restore_rng_state(state)
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_patch_discrimination_matches_global_reference(self):
        from losses import patch_disc_loss
        torch.manual_seed(3)
        predicted = torch.randn(2, 9, 7, requires_grad=True)
        target = torch.randn(2, 9, 7)
        token_type = torch.tensor([[0, 1, 1, 1, 0, 1, 1, -1, -1],
                                   [0, 1, 1, 0, 1, 1, 1, 1, -1]])
        contig = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1, -1],
                               [0, 0, 0, 1, 1, 1, 1, 1, -1]])
        mask = torch.tensor([[False, True, False, True, False, True, False, False, False],
                             [False, True, False, False, True, False, True, False, False]])

        def reference(z_hat):
            zh = F.normalize(z_hat, dim=-1); zt = F.normalize(target.to(z_hat.dtype), dim=-1)
            total = zh.new_zeros(()); count = 0
            for b in range(z_hat.shape[0]):
                genes = (token_type[b] == 1).nonzero(as_tuple=False).squeeze(-1)
                same = contig[b, genes, None] == contig[b, None, genes]
                valid = mask[b, genes] & (same.sum(-1) >= 2)
                sim = (zh[b, genes] @ zt[b, genes].T / 0.05).masked_fill(~same, float("-inf"))
                total = total - F.log_softmax(sim, dim=-1).diagonal()[valid].sum()
                count += int(valid.sum())
            return total / max(1, count)

        expected = reference(predicted)
        actual = patch_disc_loss(predicted, target, mask, contig, token_type)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        expected.backward(retain_graph=True); expected_grad = predicted.grad.clone(); predicted.grad.zero_()
        actual.backward()
        self.assertTrue(torch.allclose(predicted.grad, expected_grad, atol=1e-6, rtol=1e-6))

    def test_empty_active_losses_remain_connected_to_the_model(self):
        from losses import mag_taxonomy_loss, patch_disc_loss
        taxonomy_logits = torch.randn(3, 4, requires_grad=True)
        tax_loss = mag_taxonomy_loss(taxonomy_logits, torch.full((3,), -1))
        tax_loss.backward()
        self.assertIsNotNone(taxonomy_logits.grad)

        predicted = torch.randn(1, 4, 5, requires_grad=True)
        patch_loss = patch_disc_loss(
            predicted, torch.randn_like(predicted),
            torch.zeros((1, 4), dtype=torch.bool),
            torch.tensor([[0, 0, 0, -1]]),
            torch.tensor([[0, 1, 1, -1]]),
        )
        patch_loss.backward()
        self.assertIsNotNone(predicted.grad)

    def test_inactive_heads_are_not_constructed(self):
        from model import MAGEncoder
        model = MAGEncoder(vocab_size=4, mode="esmc", num_layers=1, d_model=32, num_heads=4,
                           ff_mult=2, use_checkpoint=False, max_local_pos=8,
                           num_ko=10, num_ec=7, num_tax_per_rank={"domain": 3, "phylum": 4},
                           use_patch_disc=True, use_ko_head=False, use_ec_head=False,
                           taxonomy_head_ranks=("domain",))
        self.assertIsNone(model.ko_head); self.assertIsNone(model.ec_head)
        self.assertEqual(list(model.mag_tax_heads), ["domain"])
        self.assertIsNotNone(model.esmc_pred)

    def test_esmc_fp16_input_is_safe_without_autocast(self):
        from model import EmbeddingBuilder
        module = EmbeddingBuilder(d_model=16, vocab_size=4, mode="esmc", max_local_pos=8)
        token_ids = torch.tensor([[2, 0, 0]])
        token_type = torch.tensor([[0, 1, 1]])
        contig_id = torch.tensor([[0, 0, 0]])
        local_pos = torch.tensor([[0, 1, 2]])
        vectors = torch.randn(1, 3, 1152, dtype=torch.float16)
        output = module(token_ids, token_type, contig_id, local_pos, vectors)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (1, 3, 16))


if __name__ == "__main__":
    unittest.main()
