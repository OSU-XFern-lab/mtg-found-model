# Workflow and extension points

## Data flow

```text
Binned nucleotide MAGs -- Prodigal --> proteins ordered within contigs
                                          |
Optional KO/EC/taxonomy -------------------+
                                          v
                         ESM-C embeddings + immutable shards
                                          |
                         local manifest + reproducible split plan
                                          v
                         masked training -> validation -> checkpoints
```

`build_dataset.py` is the entry point for a new ESM-C dataset. It makes two
passes over canonical records: the first discovers annotation vocabularies;
the second writes shards. Protein embedding reuse is exact-sequence based.
The canonical files and manifest are generated locally and are ignored by Git.
Data paths are recorded as absolute storage roots. If moving a dataset, set
`data.root_overrides` in your training recipe to map `local` to the new root.

## Adding MAGs

`ingest_mags.py` converts a new source and appends it to an existing immutable
manifest. Run from `model/`, using new source/output names:

```bash
MAG_INGEST_ACTION=convert SOURCE=NEW_SOURCE INPUT_ROOT=/path/to/new_mags \
  python ingest_mags.py
MAG_INGEST_ACTION=append SOURCE=NEW_SOURCE \
  PARENT_MANIFEST=/path/to/dataset/manifest.json \
  OUTPUT_MANIFEST=/path/to/new-release/manifest.json \
  SHARD_ROOT_PATH=/path/to/new-release \
  python ingest_mags.py
```

Generic conversion writes canonical files beneath
`data/NEW_SOURCE/per_mag/`. Append preserves the parent's annotation vocabularies;
unseen labels are not automatically added. Build a fresh dataset if you need
new supervision classes. Child manifests retain references to parent shards,
so keep those storage roots available. The bulk workflow in
[model/mass_ingest](../model/mass_ingest/README.md) is for advanced existing
corpora and requires extra setup, including VQ artifacts for its VQ stage.

## Discrete representations

`model/tokenization.py` defines IDs 0–3 as PAD, MASK, UCT, UNK. Frozen VQ-k-means
assignment expects `token/vq_kmeans/codebook.npy`; VQ-VAE assignment loads its
compatible frozen network via `mag_shards.ingest.load_frozen_vqvae`. Inspect that
loader before supplying your own checkpoint. Neither artifacts nor a VQ fitting
pipeline are distributed in this release. A manifest's representation metadata
must agree with the shard columns and vocabulary sizes. Missing discrete tokens
are not a substitute for a trained codebook.

## Custom objectives and evaluation

Implement a scalar loss in `losses.py`, wire its head and computation into
`model.py` and `training.py`, and extend configuration/schema validation as
needed. Preserve masking and missing-label behavior. Add an independent numeric
or gradient check before using a new objective at scale.

Training masks genes and retains contig boundaries. Crop size, MAG token budget,
and contig dropout are recipe options. Batch tokens and GPU count affect the
effective optimizer batch. Retune learning rate when changing those settings.

Use `split_plans.py` to select random MAG splits, source holdouts, stored labels,
or existing curated assignment plans. Keep train/validation/test assignment
independent of augmentation seeds. `probe.py` provides evaluation helpers; it is
not an end-user prediction CLI. Checkpoint deployment and a pretrained model
service are outside this source release.

## Optional preprocessing utilities

The Python sources in `data/jgi`, `data/motus`, `data/tax_tree`, and
`data/clip_text` preserve the original acquisition/annotation and text-building
utilities. They are optional research utilities with source-specific settings;
review their module-level paths, inputs, external programs, and CLI help before
running them. They are not needed for training on your own prepared MAGs.
Downloaded databases, authentication files, source manifests, and generated
text corpora are intentionally absent. Access to external data services must be
arranged separately.
