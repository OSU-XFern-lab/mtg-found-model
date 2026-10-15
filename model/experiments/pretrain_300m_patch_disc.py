"""Full 300M pretraining: ESM-C input, patch discrimination only."""

from dataclasses import replace
import os

from train_config import DatasetSelection, SplitConfig, base_train_config


MODE = "esmc"
OBJECTIVES = {
    "gene_id": 0.0,
    "ko": 0.0,
    "ec": 0.0,
    "mag_tax": 0.0,
    "patch_disc": 1.0,
}
CHECKPOINT_ROOT = "../runs/pretrain_300m_patch_disc"
TARGET_EPOCHS = 100


def get_config(profile: str = "default"):
    base = base_train_config(profile)
    return replace(
        base,
        mode=MODE,
        data=replace(
            base.data,
            manifest=os.environ.get("MAG_MANIFEST", "../work/dataset/manifest.json"),
            split=SplitConfig(strategy="stored_labels"),
            train=DatasetSelection((), ("train",)),
            validation=DatasetSelection((), ("val",)),
            test=DatasetSelection((), ("test",)),
        ),
        objectives=replace(
            base.objectives,
            weights=OBJECTIVES,
            production_approved=("patch_disc",),
        ),
        model=replace(base.model, d_model=1024, n_layers=24, n_heads=8, trainable_mask=False),
        batch=replace(
            base.batch,
            per_mag_token_budget=16384,
            max_contig_genes=512,
            max_contigs_per_mag=None,
            contig_dropout_apply_prob=0.25,
            contig_dropout_range=(0.10, 0.10),
        ),
        optim=replace(
            base.optim,
            optimizer="muon",
            lr=3e-4,
            target_epochs=TARGET_EPOCHS,
            early_stopping_patience_fraction=0.15,
        ),
        runtime=replace(
            base.runtime,
            checkpoint_root=CHECKPOINT_ROOT,
            val_every=0,
        ),
    )
