"""Typed configuration for MAG-encoder training."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from mag_shards.schema import InputSpec, TAXONOMY_RANKS, VALID_SPLITS


MODEL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSelection:
    sources: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    plan: Optional[str] = None
    seed: int = 0
    train_fraction: float = 0.90
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    stratify_by_source: bool = True
    source_labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    manifest: str
    split: SplitConfig
    train: DatasetSelection
    validation: DatasetSelection
    test: DatasetSelection = DatasetSelection()
    root_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveConfig:
    weights: Mapping[str, float]
    production_approved: Tuple[str, ...] = ()
    taxonomy_ranks: Tuple[str, ...] = TAXONOMY_RANKS
    mask_gene: bool = True
    mask_ko: bool = False
    mask_ec: bool = False
    mask_rate_min: float = 0.15
    mask_rate_max: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    trainable_mask: bool = True
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    use_checkpoint: bool = True
    max_local_pos: int = 512


@dataclass(frozen=True)
class BatchConfig:
    max_batch_mags: Optional[int] = None
    per_mag_token_budget: Optional[int] = 16384
    max_contig_genes: int = 512
    max_contigs_per_mag: Optional[int] = None
    contig_dropout_range: Tuple[float, float] = (0.0, 0.0)
    contig_dropout_apply_prob: float = 0.0
    bucket_window: int = 256

    @property
    def resolved_max_mag_tokens(self) -> int:
        """Finite upper bound used to construct device-sized batches."""
        limits = []
        if self.per_mag_token_budget is not None:
            limits.append(int(self.per_mag_token_budget))
        if self.max_contigs_per_mag is not None:
            limits.append(
                int(self.max_contigs_per_mag) * (int(self.max_contig_genes) + 1)
            )
        if not limits:
            raise ValueError("set per_mag_token_budget or max_contigs_per_mag")
        return min(limits)


@dataclass(frozen=True)
class HardwareConfig:
    # Calibration starts here and doubles until it finds the current GPU's
    # safe limit. This value is exact only when auto_batch_tokens is False.
    per_device_batch_tokens: int = 32768
    max_per_device_batch_tokens: Optional[int] = None
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    auto_batch_tokens: bool = True
    gpu_memory_fraction: float = 0.90
    batch_token_search_step: int = 4096


HARDWARE_PROFILES = {
    "default": HardwareConfig(),
    "a40": HardwareConfig(per_device_batch_tokens=24576),
    "l40s": HardwareConfig(per_device_batch_tokens=32768),
    # 131K saturated A100 compute in the real-data benchmark; larger batches
    # consumed more memory without improving throughput.
    "a100": HardwareConfig(per_device_batch_tokens=131072,
                           max_per_device_batch_tokens=131072),
    "h100": HardwareConfig(per_device_batch_tokens=131072,
                           max_per_device_batch_tokens=131072,
                           num_workers=6),
    "h100_40g": HardwareConfig(per_device_batch_tokens=98304,
                               max_per_device_batch_tokens=131072,
                               num_workers=6),
    "h100_80g": HardwareConfig(per_device_batch_tokens=131072,
                               max_per_device_batch_tokens=131072,
                               num_workers=6),
    "h200": HardwareConfig(per_device_batch_tokens=131072,
                           max_per_device_batch_tokens=131072,
                           num_workers=6),
}


@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    target_epochs: Optional[int] = 100
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    early_stopping_patience_fraction: Optional[float] = None


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_root: str
    resume: bool = True
    validate: bool = True
    max_val_batches: Optional[int] = None
    log_every: int = 50
    val_every: int = 2000
    seed: int = 0
    wandb_project: str = "mag-encoder"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "disabled"
    process_group_timeout_seconds: int = 600


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    data: DataConfig
    objectives: ObjectiveConfig
    model: ModelConfig
    batch: BatchConfig
    optim: OptimConfig
    runtime: RuntimeConfig
    hardware: HardwareConfig
    profile: str = "default"

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(self.mode, self.objectives.weights, taxonomy_ranks=self.objectives.taxonomy_ranks)

    def validate(self) -> None:
        self.input_spec
        strategies = {"existing", "stored_labels", "source_holdout", "random_mag"}
        if self.data.split.strategy not in strategies:
            raise ValueError("split strategy must be one of {}".format(sorted(strategies)))
        if self.data.split.strategy == "existing" and not self.data.split.plan:
            raise ValueError("existing split strategy requires a plan path")
        if self.data.split.strategy == "source_holdout":
            if not self.data.split.source_labels:
                raise ValueError("source_holdout requires source_labels")
            bad = set(self.data.split.source_labels.values()) - set(VALID_SPLITS)
            if bad:
                raise ValueError("source_holdout has invalid labels: {}".format(sorted(bad)))
        if self.data.split.strategy == "random_mag":
            fractions = (self.data.split.train_fraction, self.data.split.validation_fraction,
                         self.data.split.test_fraction)
            if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
                raise ValueError("random split fractions must be nonnegative and sum to one")
            bad = set(self.data.split.source_labels.values()) - set(VALID_SPLITS)
            if bad:
                raise ValueError("random_mag has invalid fixed source labels: {}".format(sorted(bad)))
        for name, selection in (("train", self.data.train),
                                ("validation", self.data.validation), ("test", self.data.test)):
            bad = set(selection.labels) - set(VALID_SPLITS)
            if bad:
                raise ValueError("{} selection has invalid labels: {}".format(name, sorted(bad)))
        if not any(float(v) > 0 for v in self.objectives.weights.values()):
            raise ValueError("at least one training objective must be active")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.batch.per_mag_token_budget is None
                and self.batch.max_contigs_per_mag is None):
            raise ValueError(
                "set per_mag_token_budget, max_contigs_per_mag, or both"
            )
        if self.batch.per_mag_token_budget is not None and self.batch.per_mag_token_budget <= 1:
            raise ValueError("per_mag_token_budget must exceed one token or be None")
        if self.batch.max_contig_genes <= 0:
            raise ValueError("max_contig_genes must be positive")
        if self.batch.max_contigs_per_mag is not None and self.batch.max_contigs_per_mag <= 0:
            raise ValueError("max_contigs_per_mag must be positive or None")
        drop_lo, drop_hi = self.batch.contig_dropout_range
        if not 0.0 <= drop_lo <= drop_hi <= 1.0:
            raise ValueError("contig_dropout_range must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= self.batch.contig_dropout_apply_prob <= 1.0:
            raise ValueError("contig_dropout_apply_prob must be between zero and one")
        if self.hardware.per_device_batch_tokens <= 0:
            raise ValueError("per-device token budget must be positive")
        if (self.hardware.max_per_device_batch_tokens is not None
                and self.hardware.max_per_device_batch_tokens <= 0):
            raise ValueError("maximum per-device token budget must be positive or None")
        if not 0.0 < self.hardware.gpu_memory_fraction < 1.0:
            raise ValueError("gpu_memory_fraction must be between zero and one")
        if self.hardware.batch_token_search_step <= 0:
            raise ValueError("batch_token_search_step must be positive")
        if self.batch.max_contig_genes > self.model.max_local_pos:
            raise ValueError("max_contig_genes cannot exceed model.max_local_pos")
        if self.optim.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if (self.optim.early_stopping_patience_fraction is not None
                and not 0.0 < self.optim.early_stopping_patience_fraction <= 1.0):
            raise ValueError("early_stopping_patience_fraction must be in (0, 1] or None")
        if self.optim.early_stopping_patience_fraction is not None and not self.runtime.validate:
            raise ValueError("early stopping requires validation")
        if self.optim.amp_dtype not in ("bfloat16", "float16", "none"):
            raise ValueError("amp_dtype must be bfloat16, float16, or none")
        if self.optim.optimizer not in ("adamw", "muon"):
            raise ValueError("optimizer must be adamw or muon")
        if self.optim.optimizer == "muon" and self.optim.amp_dtype == "float16":
            raise ValueError("Muon recipe supports bfloat16 or full precision")
        if self.runtime.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("invalid W&B mode")
        active = {name for name, weight in self.objectives.weights.items() if float(weight) > 0}
        approved = set(self.objectives.production_approved)
        if active != approved:
            raise ValueError(
                "active objectives must exactly match production_approved; active={} approved={}".format(
                    sorted(active), sorted(approved))
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def scientific_sha256(self) -> str:
        value = {
            "mode": self.mode,
            "data": asdict(self.data),
            "objectives": asdict(self.objectives),
            "model": asdict(self.model),
            "batch": asdict(self.batch),
            "optim": asdict(self.optim),
            "seed": self.runtime.seed,
        }
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def base_train_config(profile: str = "default") -> TrainConfig:
    if profile not in HARDWARE_PROFILES:
        raise ValueError("unknown hardware profile {!r}".format(profile))
    config = TrainConfig(
        mode="esmc",
        data=DataConfig(
            manifest=str(MODEL_DIR.parent / "work" / "dataset" / "manifest.json"),
            split=SplitConfig(strategy="random_mag", seed=42),
            train=DatasetSelection((), ("train",)),
            validation=DatasetSelection((), ("val",)),
            test=DatasetSelection((), ("test",)),
        ),
        objectives=ObjectiveConfig(weights={}),
        model=ModelConfig(),
        batch=BatchConfig(),
        optim=OptimConfig(),
        runtime=RuntimeConfig(
            checkpoint_root=str(MODEL_DIR.parent / "runs" / "mag_encoder"),
        ),
        hardware=HARDWARE_PROFILES[profile],
        profile=profile,
    )
    return config


def _selection(value: Mapping[str, object], default_label: str) -> DatasetSelection:
    return DatasetSelection(
        sources=tuple(value.get("sources", ())),
        labels=tuple(value.get("labels", (default_label,))),
    )


def train_config_from_dict(value: Mapping[str, object], profile: Optional[str] = None) -> TrainConfig:
    raw = copy.deepcopy(dict(value))
    data = dict(raw["data"])
    objectives = dict(raw["objectives"])
    batch = dict(raw.get("batch", {}))
    # Read checkpoints/configs written before batch size became fully
    # hardware-calibrated.
    batch.pop("global_batch_tokens", None)
    # Preserve old checkpoints with the disabled option; never silently change an active recipe.
    if batch.pop("sample_mag_ratio", None) is not None:
        raise ValueError("sample_mag_ratio is no longer supported")
    dropout_range = batch.get("contig_dropout_range")
    if dropout_range is not None:
        batch["contig_dropout_range"] = tuple(dropout_range)
    # Read checkpoints/configs from the former fixed-probability augmentation.
    old_drop_prob = batch.pop("contig_drop_prob", None)
    if old_drop_prob is not None and dropout_range is None:
        batch["contig_dropout_range"] = (float(old_drop_prob), float(old_drop_prob))
        batch["contig_dropout_apply_prob"] = 1.0 if float(old_drop_prob) > 0 else 0.0
    selected_profile = profile or str(raw.get("profile", "default"))
    hardware_value = raw.get("hardware")
    if hardware_value:
        hardware_dict = dict(hardware_value)
        hardware_dict.pop("expected_gpus", None)
        hardware = HardwareConfig(**hardware_dict)
    else:
        hardware = HARDWARE_PROFILES.get(selected_profile, HardwareConfig())
    split_value = dict(data.get("split", {}))
    if not split_value and data.get("split_plan"):
        split_value = {"strategy": "existing", "plan": data["split_plan"]}
    runtime = dict(raw["runtime"])
    # Accept checkpoints/configs written before rolling checkpoints became
    # epoch-based only.
    runtime.pop("checkpoint_every", None)
    config = TrainConfig(
        mode=str(raw["mode"]),
        data=DataConfig(
            manifest=str(data["manifest"]),
            split=SplitConfig(
                strategy=str(split_value["strategy"]),
                plan=str(split_value["plan"]) if split_value.get("plan") else None,
                seed=int(split_value.get("seed", 0)),
                train_fraction=float(split_value.get("train_fraction", 0.90)),
                validation_fraction=float(split_value.get("validation_fraction", 0.05)),
                test_fraction=float(split_value.get("test_fraction", 0.05)),
                stratify_by_source=bool(split_value.get("stratify_by_source", True)),
                source_labels=dict(split_value.get("source_labels", {})),
            ),
            train=_selection(dict(data.get("train", {})), "train"),
            validation=_selection(dict(data.get("validation", {})), "val"),
            test=_selection(dict(data.get("test", {})), "test"),
            root_overrides=dict(data.get("root_overrides", {})),
        ),
        objectives=ObjectiveConfig(
            weights={k: float(v) for k, v in dict(objectives["weights"]).items()},
            production_approved=tuple(objectives.get("production_approved", ())),
            taxonomy_ranks=tuple(objectives.get("taxonomy_ranks", TAXONOMY_RANKS)),
            mask_gene=bool(objectives.get("mask_gene", True)),
            mask_ko=bool(objectives.get("mask_ko", False)),
            mask_ec=bool(objectives.get("mask_ec", False)),
            mask_rate_min=float(objectives.get("mask_rate_min", 0.15)),
            mask_rate_max=float(objectives.get("mask_rate_max", 0.15)),
        ),
        model=ModelConfig(**dict(raw.get("model", {}))),
        batch=BatchConfig(**batch),
        optim=OptimConfig(**dict(raw.get("optim", {}))),
        runtime=RuntimeConfig(**runtime),
        hardware=hardware,
        profile=selected_profile,
    )
    config.validate()
    return config
