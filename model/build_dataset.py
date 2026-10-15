"""Build a first ESM-C shard dataset from user-supplied MAG directories."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import data_sources as ds
from ingest_mags import convert_input
from mag_shards.esmc_inference import ESMCEmbedder
from mag_shards.ingest import RepresentationResolver, build_shard_payload
from mag_shards.manifest import DatasetManifest, ShardRef
from mag_shards.policies import ec_top2_labels
from mag_shards.schema import ESMC_DIM
from mag_shards.validation import validate_shard
from mag_shards.writer import ShardWriter


def build_dataset(input_root, output_root, *, source='CUSTOM', shard_genes=100000,
                  prefer='faa', embedder=None):
    """Two-pass vocabulary discovery and bounded shard writing; no parent needed.

    ``embedder`` is injectable for CPU tests. Real runs always use ESM-C.
    A single MAG is never split across shards and may exceed shard_genes.
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise ValueError('input must be a directory containing one subdirectory per MAG')
    if shard_genes <= 0 or not source or '/' in source or '\\' in source:
        raise ValueError('use a positive shard gene count and a simple source name')
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError('output must be outside the input tree')
    output_root.mkdir(parents=True, exist_ok=False)
    previous = ds.SOURCE_LAYOUTS.get(source)
    ds.SOURCE_LAYOUTS[source] = (output_root/'canonical/per_mag', output_root/'canonical/taxonomy.tsv')
    owned = embedder is None
    embedder = embedder or ESMCEmbedder()
    try:
        convert_input(input_root, source, prefer=prefer)
        def records():
            # Serial iteration keeps shard order reproducible across machines.
            per_mag, taxonomy = ds.source_paths(source)
            tax = ds.load_taxonomy(taxonomy)
            for directory in sorted(per_mag.iterdir()):
                record = ds.build_record((directory.name, directory, source, tax.get(directory.name, {}), True))
                if record is not None:
                    yield record
        ko, ec = set(), set()
        taxa = {rank: set() for rank in ds.TAX_KEYS}
        count = 0
        for record in records():
            count += 1
            for rank, value in record['taxonomy'].items():
                if value:
                    taxa[rank].add(value)
            for contig in record['contigs']:
                for gene in contig['genes']:
                    ko.update(gene['ko_ids'])
                    for label in gene['ec_ids']:
                        ec.update(ec_top2_labels(label))
        if not count:
            raise ValueError('no nonempty MAGs found')
        vocab = lambda values: {value: i for i, value in enumerate(sorted(values))}
        manifest = DatasetManifest(
            dataset_id=source, generation=1, storage_roots={'local': str(output_root)},
            shards=[], totals={}, sources={source: {'n_mags': count}},
            representations={'esmc': {'dim': ESMC_DIM}, 'tokens': {}},
            vocabularies={'ko': vocab(ko), 'ec': vocab(ec),
                          'taxonomy': {rank: vocab(values) for rank, values in taxa.items()}},
            split={'mode': 'random_mag', 'seed': 42, 'val_frac': 0.1},
            created_utc=datetime.now(timezone.utc).isoformat(),
            builder={'operation': 'build_dataset', 'embedding': 'esmc_600m'},
            indexes={'protein': {'type': 'per_shard_sorted_v1', 'segments': []}})
        resolver = RepresentationResolver(manifest, embedder=embedder)
        writer = ShardWriter(output_root, 'shards')
        counts = Counter()
        def flush(batch):
            payload = build_shard_payload(manifest, batch, resolver)
            shard_id = 'shard-{:06d}'.format(len(manifest.shards))
            path = writer.write(shard_id, payload)
            report = validate_shard(path, True)
            meta = json.loads((path/'shard.json').read_text())
            manifest.shards.append(ShardRef(
                shard_id, 'local', str(path.relative_to(output_root)), report['metadata_sha256'],
                len(payload.mags), int(meta['n_contigs']), int(meta['n_genes']),
                {key: value['bytes'] for key, value in meta['columns'].items()}))
            counts.update(mag['split'] for mag in payload.mags)
            # Refresh lookup so subsequent shards reuse previously embedded proteins.
            resolver.index.paths.append(path)
        pending, genes = [], 0
        for record in records():
            pending.append(record)
            genes += sum(len(c['genes']) for c in record['contigs'])
            if genes >= shard_genes:
                flush(pending); pending, genes = [], 0
        if pending:
            flush(pending)
        manifest.totals = {key: sum(getattr(s, key) for s in manifest.shards)
                           for key in ('n_mags', 'n_contigs', 'n_genes')}
        manifest.split['counts'] = dict(counts)
        manifest.indexes['protein']['segments'] = [s.shard_id for s in manifest.shards]
        manifest.publish(output_root/'manifest.json')
        return output_root/'manifest.json'
    finally:
        if owned:
            embedder.close()
        if previous is None:
            ds.SOURCE_LAYOUTS.pop(source, None)
        else:
            ds.SOURCE_LAYOUTS[source] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', default='CUSTOM')
    parser.add_argument('--shard-genes', type=int, default=100000)
    parser.add_argument('--prefer', choices=('faa', 'fna'), default='faa')
    args = parser.parse_args()
    print(build_dataset(args.input, args.output, source=args.source,
                        shard_genes=args.shard_genes, prefer=args.prefer))


if __name__ == '__main__':
    main()
