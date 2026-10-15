# Bulk MAG ingestion

This package appends large prepared sources to an immutable parent dataset.
Source names, output names, shard numbering, and storage locations come from
the run configuration; no JGI collection is embedded in this workflow.

## The five responsibilities

1. `prepare.py` plans balanced tasks, collects proteins, and deduplicates them
   into stable hash buckets.
2. `gpu.py` embeds each unique protein once and assigns both frozen VQ modes.
3. `build.py` reconstructs canonical MAG records, writes validated shards, and
   publishes a child manifest.
4. `audit.py` proves counts at every completed handoff.
5. `cleanup.py` removes reproducible staging only after publication is proven.

`common.py` contains only small runtime helpers. The immutable format, payload
builder, hashing, atomic writes, and validation remain in `mag_shards/` so they
have one implementation.

## Configuration

Every run needs `MASS_ROOT` and `PARENT_MANIFEST`. Planning also needs the
comma-separated `SOURCES`; building needs `SHARD_ROOT_PATH`.
`data_sources.source_paths()` resolves each source to its canonical `per_mag/`
directory and manifest.

Optional settings include `DATASET_ID`, `DATASET_BASE`, `OUTPUT_MANIFEST`,
`SHARD_ROOT_ALIAS`, `BUCKET_COUNT`, `COLLECT_TASKS`, `BUILD_TARGET_GENES`, and
`SOURCE_WORKERS`. The generated `plan.json` pins the parent digest and resolved
work counts. `FIRST_SHARD` defaults to the parent shard count.

## Commands

From `model/`:

```bash
python -m mass_ingest plan
python -m mass_ingest collect --task 0
python -m mass_ingest audit collect
python -m mass_ingest dedup --task 0
python -m mass_ingest audit dedup
python -m mass_ingest embed --task 0
python -m mass_ingest audit embed
python -m mass_ingest vq --task 0
python -m mass_ingest audit vq
python -m mass_ingest build --task 0
python -m mass_ingest audit build
python -m mass_ingest publish
APPLY_CLEANUP=1 python -m mass_ingest cleanup
```

The numbered commands illustrate one task. A scheduler runs the appropriate
task or bucket range and then runs the matching audit once the array finishes.
`TASK_ID` and the Slurm array index remain supported alternatives to `--task`.

Every stage is resumable. A `COMPLETE` marker is written only after its output
and report are durable. Cleanup is fail-closed: it rechecks the parent and child
digests, shard markers, source counts, gene counts, and preparation reports
before removing protein, embedding, VQ, or canonical per-MAG staging.
