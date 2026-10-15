"""Small starting recipe for a user's own ESM-C dataset; scale after validation."""
import os
from dataclasses import replace
from train_config import base_train_config, SplitConfig


def get_config(profile='default'):
    base = base_train_config(profile)
    return replace(
        base,
        data=replace(base.data, manifest=os.environ.get('MAG_MANIFEST', base.data.manifest),
                     split=SplitConfig(strategy='random_mag', seed=42,
                                       train_fraction=0.8, validation_fraction=0.1, test_fraction=0.1)),
        objectives=replace(base.objectives, weights={'patch_disc': 1.0},
                           production_approved=('patch_disc',)),
        model=replace(base.model, d_model=128, n_layers=2, n_heads=4,
                      max_local_pos=128, trainable_mask=False),
        batch=replace(base.batch, per_mag_token_budget=1024, max_contig_genes=128),
        hardware=replace(base.hardware, auto_batch_tokens=False, per_device_batch_tokens=2048,
                         num_workers=0),
        optim=replace(base.optim, target_epochs=5, warmup_steps=10),
        runtime=replace(base.runtime, checkpoint_root=os.environ.get('CHECKPOINT_ROOT', '../runs/custom_esmc'),
                        wandb_mode='disabled', val_every=0),
    )
