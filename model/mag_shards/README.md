# MAG shard format

## Overview

Each immutable shard contains whole MAGs. All per-gene arrays use the same gene
order, allowing training to read bounded sequential ranges rather than gather
rows from a terabyte-scale matrix. See the
[model operating guide](../README.md) for training and ingestion instructions.

The code has four small layers:

- `schema.py` and `manifest.py` define the versioned contract;
- `writer.py` and `validation.py` publish and verify immutable shards;
- `reader.py`, `sampler.py`, and `collate.py` feed training;
- `ingest.py` reuses or creates representations for new canonical MAG records.

Training verifies the complete manifest once, then creates lightweight
train/validation/test views with `ManifestDataset.select()`. Payload columns are
memory-mapped only when a selected objective needs them.

## Structural files

- `mags.jsonl`: MAG identity, source, split, taxonomy IDs, counts, and source digest.
- `mag_gene_offsets.u64`: half-open gene interval for each MAG.
- `mag_contig_offsets.u64`: half-open contig-table interval for each MAG.
- `contig_gene_offsets.u64`: half-open gene interval for each contig.

Contig IDs and within-contig positions are derived from these offsets, so a
redundant per-gene contig column is unnecessary.

## Representations

- `esmc.f16`: row-major `[n_genes, 1152]` FP16 ESM-C vectors.
- `esmc_valid.bit`: optional validity bitmap, omitted when every row is valid.
- `token.<mode>.u32`: one aligned column per discrete representation;
  `UINT32_MAX` denotes a missing assignment.

A discrete-only training configuration does not open `esmc.f16` unless an
active objective requires it.

## Labels and protein index

- `ko_indptr.u64` / `ko_ids.u32`: sparse KO targets.
- `ec_indptr.u64` / `ec_ids.u32`: sparse top-level EC targets.
- `protein_hash.u8`: occurrence-aligned 16-byte MD5 protein hashes.
- `protein_index_hash.u8` / `protein_index_row.u32`: sorted unique hash-to-row lookup.

Raw protein sequences and gene-name strings remain in the canonical source
layer; training does not read them from shards.

## Validation and publication

`shard.json` records the schema, shapes, dtypes, byte sizes, checksums, counts,
token ranges, sources, and split counts. `COMPLETE` is written last. The staging
directory is atomically renamed, and a published shard is never modified.

The dataset manifest resolves shards through storage-root aliases, records
representation and vocabulary provenance, and receives a canonical SHA-256.
Training stores that digest in every checkpoint. `schema.py`, `manifest.py`,
`writer.py`, and `validation.py` are the executable format contract.
`mass_ingest/` only orchestrates parallel work; it uses the same public payload
builder and writer as incremental ingestion.

## Tests

From `model/`, run:

```bash
python -m unittest -v tests.test_mag_shards tests.test_shard_ingest
```
