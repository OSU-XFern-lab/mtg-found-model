"""Muon/AdamW checkpoint continuation must match uninterrupted updates."""
import copy
import unittest
import torch
from optimizers import MuonAdamW


class MuonResumeTest(unittest.TestCase):
    def test_fixed_zero_mask_excluded_from_optimizer(self):
        from dataclasses import replace
        from experiments.pretrain_300m_patch_disc import get_config
        from training import build_model
        from train_config import train_config_from_dict
        config = get_config()
        self.assertFalse(config.model.trainable_mask)
        self.assertFalse(train_config_from_dict(config.to_dict()).model.trainable_mask)
        config = replace(config, model=replace(config.model, d_model=16, n_layers=1, n_heads=2))
        model = build_model(config, 4, 0, 0, {}, "cpu")
        mask = model.embed.mask_vec
        self.assertFalse(mask.requires_grad)
        self.assertEqual(torch.count_nonzero(mask).item(), 0)
        optimizer = MuonAdamW(model, 3e-4, 0.01)
        self.assertFalse(any(p is mask for g in optimizer.param_groups for p in g["params"]))
        # Other trainable embeddings can update without moving the fixed mask.
        model.embed.type_emb.weight.square().sum().backward()
        optimizer.step()
        self.assertIsNone(mask.grad)
        self.assertEqual(torch.count_nonzero(mask).item(), 0)
        other = build_model(config, 4, 0, 0, {}, "cpu")
        other.load_state_dict(model.state_dict())
        self.assertFalse(other.embed.mask_vec.requires_grad)
        self.assertEqual(torch.count_nonzero(other.embed.mask_vec).item(), 0)

    def test_checkpoint_continuation(self):
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(8, 8)
        model.head = torch.nn.Linear(8, 2)
        optimizer = MuonAdamW(model, 3e-4, 0.01)
        x = torch.randn(4, 8)

        def update(m, opt):
            opt.zero_grad()
            m.head(m.encoder(x)).square().mean().backward()
            opt.step()

        update(model, optimizer)
        resumed = copy.deepcopy(model)
        other = MuonAdamW(resumed, 3e-4, 0.01)
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertIs(other.param_groups[0], other.muon.param_groups[0])
        update(model, optimizer)
        update(resumed, other)
        for expected, actual in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
