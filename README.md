# MTG foundation model

Train a hierarchical encoder on your own metagenome-assembled genomes (MAGs).
The model represents genes, contigs, and MAGs using protein embeddings and
masked prediction objectives. This repository contains training and processing
code, loss functions, optimizer implementations, and tests.

**Start with assembled, binned MAGs or their predicted proteins.** Read trimming,
assembly, and binning are upstream steps and are not implemented here. This is
an encoder training workflow, not a complete conversational language model.

No biological datasets, dataset manifests, learned codebooks, model weights,
checkpoints, credentials, or experimental results are distributed here. You
create manifests locally when building your own dataset; training needs those
local files, but they stay out of Git.

## Install

Use Python 3.10 or newer in a dedicated environment. Install a CUDA-compatible
PyTorch build for your machine, then install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# For embedding new protein sequences with ESM-C:
python -m pip install -r requirements-ingest.txt
# For optional text-corpus utilities and the complete test suite:
python -m pip install -r requirements-optional.txt
```

Training uses PyTorch FlexAttention and the Muon recipe uses `torch.optim.Muon`.
The requirements conservatively select PyTorch 2.12 or newer, matching the
2.12.0 environment used for CPU validation. The requirements are compatibility bounds, not a locked
reproduction environment. ESM-C ingestion requires an NVIDIA CUDA GPU, the
`esm` package, and access to its `esmc_600m` weights. Those weights are downloaded
separately and have their own access and usage terms. Nucleotide input also
requires the `prodigal` executable on `PATH`. The CPU tests do not download weights.

## Prepare your MAGs

Give each MAG a unique directory name. Supply one `.faa` protein FASTA per MAG,
or one `.fna` nucleotide FASTA for Prodigal gene calling:

```text
my_mags/
  mag_001/
    proteins.faa
    ko_go_ec.tsv       # optional
    taxonomy.tsv       # optional
  mag_002/
    contigs.fna
```

Protein identifiers must end in a numeric gene position, e.g. `contigA_1`,
`contigA_2`, `contigB_1`. The prefix identifies the contig and the final integer
orders its genes. Identifiers must be unique within a MAG. Arbitrary protein IDs
can incorrectly turn each gene into its own contig, making patch discrimination
uninformative. If you supply both FASTA types, proteins are preferred. Supply
exactly one file of each type; the converter selects the first sorted match.

Optional `ko_go_ec.tsv` is tab-separated with columns `gene_id`, `KO`, `EC`;
multiple annotations are comma-separated. Gene IDs must match the FASTA.
Optional `taxonomy.tsv` has no header: each line is a rank, a tab, and a value
(e.g. `domain<TAB>Bacteria`). Accepted ranks are domain, phylum, class, order,
family, genus, and species. Annotation tools and reference databases are not
bundled. Missing annotations are allowed for the default patch objective.

## Build your first dataset

From the repository root, with your environment activated:

```bash
cd model
python build_dataset.py \
  --input /absolute/path/to/my_mags \
  --output ../work/dataset \
  --source CUSTOM
```

Use a new output directory for each build. The builder converts inputs, discovers
annotation vocabularies, embeds proteins with ESM-C, writes checksummed shards,
and publishes `work/dataset/manifest.json`. It refuses to overwrite an existing
output directory. A failed build may leave partial output: inspect it and retry
with a new directory. `--shard-genes` controls the approximate in-memory shard
size (default 100,000 genes); one MAG is never divided between shards.

The ESM-C implementation truncates proteins to 1,022 residues and mean-pools
non-padding positions, including the tokenizer's special tokens. It produces
1,152-dimensional FP16 embeddings. Identical proteins are reused across shards.
These details affect reproducibility; keep them consistent when adding data.

## Train

Run this from `model/` after building your dataset:

```bash
export MAG_MANIFEST="$(realpath ../work/dataset/manifest.json)"
export TRAIN_EXPERIMENT=experiments/custom_esmc.py
python train_mag_encoder.py
# Or use multiple GPUs on one node:
# torchrun --standalone --nproc_per_node=4 train_mag_encoder.py
```

`custom_esmc.py` is a small two-layer starting configuration. It uses masked
patch discrimination, five epochs, and a seeded 80/10/10 MAG split. Use enough
MAGs for nonempty train and validation sets (at least 10 for this recipe), and
inspect the resolved counts. Closely related MAGs can leak biological signal
across random splits; use source or curated similarity/taxonomy holdouts for
scientific evaluation. The trainer validates on the validation set; it does not
automatically evaluate the reserved test set.

Copy and edit the recipe to change model size, objectives, token budgets,
optimizer, split strategy, and training duration. `train_config.py` defines the
configuration fields and validates them. `pretrain_300m_patch_disc.py` preserves
the larger Muon training recipe, with portable paths; it requires substantially
more GPU memory. Its split policy uses stored labels instead of the starter's
80/10/10 split. Check the resolved configuration before a large run.

Checkpoints and metrics are written below `runs/`. `last.pt` supports resume and
`best.pt` tracks validation improvement. W&B is disabled by default; enable it
in the recipe if desired. Set `TRAIN_NO_RESUME=1` to request a fresh run. For
scheduler use, wrap the same Python or `torchrun` command in your site's job
script with its account, partition, environment, and GPU request.

## Losses and representations

The weighted total is assembled in `model/training.py`; implementations are in
`model/losses.py`.

| Objective | Supervision | Loss |
|---|---|---|
| `patch_disc` | Masked genes' ESM-C embeddings | InfoNCE: the gene's own target is positive; other genes in its contig are negatives |
| `gene_id` | Discrete VQ gene tokens | Cross-entropy on masked valid genes |
| `ko` | One unambiguous KO assignment per gene | Multiclass cross-entropy; missing/ambiguous labels are ignored |
| `ec` | EC top-level and second-level labels | Multilabel binary cross-entropy |
| `mag_tax` | MAG taxonomy by rank | Cross-entropy with missing labels ignored |

Every enabled objective must have a positive weight and appear in
`production_approved` in the recipe. That field is an explicit configuration
gate, not evidence that an objective has been validated on your data. Optional
supervised heads require suitable annotations and vocabularies.

The first-dataset builder supports **ESM-C input**. The encoder, readers, and
larger recipes also support `vq_kmeans` and `vq_vae`. Those modes require your own
compatible frozen codebooks/weights and tokenized shards. The code needed to
train new VQ codebooks was not present in this source snapshot; the VQ recipes
are not standalone from raw MAGs. See [workflow details](docs/workflow.md).

## Code map

- `model/model.py`: hierarchical encoder and prediction heads.
- `model/losses.py`, `model/optimizers.py`: objectives, AdamW/Muon support.
- `model/train_mag_encoder.py`, `training.py`, `train_runtime.py`: training, batching, distributed execution, and checkpoints.
- `model/build_dataset.py`, `ingest_mags.py`, `data_sources.py`: new dataset creation and incremental conversion/ingestion.
- `model/mag_shards/`: immutable storage, validation, readers, and collation; [format](model/mag_shards/README.md).
- `model/split_plans.py`: reproducible data partitions.
- `model/mass_ingest/`: advanced staged ingestion into an existing dataset.
- `data/`: source code for optional downloading, annotation, taxonomy, and text-corpus preparation; this directory contains no datasets.
- `model/tests/`: generated-fixture unit and integration tests.

## Tests

```bash
cd model
python -m unittest discover -s tests -v
```

Tests generate temporary fixtures. They cover shard integrity, initial and
incremental ingestion, splits, losses, optimizer resume, and a small CPU training
run with checkpoint continuation. Real ESM-C inference and distributed GPU
training require separate hardware validation.

## License

A project license has not yet been selected by the maintainers. No license
permission is implied by publication. Third-party dependencies and model
weights retain their own terms.
