"""Exercise first-dataset ingestion with generated proteins and a fake embedder."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from build_dataset import build_dataset
from mag_shards.manifest import load_manifest
from mag_shards.validation import validate_shard
from mag_shards.schema import ESMC_DIM


class BuildDatasetTest(unittest.TestCase):
    def test_first_dataset_reuses_proteins_and_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a', 'b'):
                mag = root/'input'/name
                mag.mkdir(parents=True)
                (mag/'proteins.faa').write_text('>contig_1\nMAAA\n>contig_2\nMKKK\n')
                (mag/'ko_go_ec.tsv').write_text('gene_id\tKO\tEC\ncontig_1\tK00001\t1.2.3.4\n')
                (mag/'taxonomy.tsv').write_text('domain\tBacteria\n')
            calls = []
            def embed(sequences):
                calls.extend(sequences)
                return np.ones((len(sequences), ESMC_DIM), dtype=np.float16)
            path = build_dataset(root/'input', root/'output', shard_genes=1, embedder=embed)
            manifest = load_manifest(path)
            self.assertEqual(manifest.totals, {'n_mags': 2, 'n_contigs': 2, 'n_genes': 4})
            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest.vocabularies['ko'], {'K00001': 0})
            self.assertIn('1.2', manifest.vocabularies['ec'])
            for ref in manifest.shards:
                validate_shard(manifest.resolve_shard(ref), True)
            with self.assertRaises(FileExistsError):
                build_dataset(root/'input', root/'output', embedder=embed)

    def test_empty_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'input').mkdir()
            with self.assertRaisesRegex(ValueError, 'no nonempty MAGs'):
                build_dataset(root/'input', root/'output', embedder=lambda x: None)
