from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mag_shards.manifest import DatasetManifest, ShardRef
from split_plans import (compile_random_mag, load_split_plan,
                         materialize_split_plan)
from train_config import SplitConfig


class SplitPlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shard = self.root / "dataset" / "shard-000000"
        shard.mkdir(parents=True)
        rows = []
        for index in range(20):
            rows.append({
                "mag_id": "m{:02d}".format(index),
                "source_name": "A" if index < 10 else "B",
                "split": "train" if index % 4 else "val",
                "n_genes": 1,
                "n_contigs": 1,
            })
        (shard / "mags.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        ref = ShardRef("shard-000000", "root", "dataset/shard-000000", "unused", 20, 20, 20)
        manifest = DatasetManifest(
            dataset_id="split-test", generation=1, storage_roots={"root": str(self.root)},
            shards=[ref], totals={"n_mags": 20, "n_contigs": 20, "n_genes": 20},
            sources={"A": {}, "B": {}}, representations={}, vocabularies={},
            split={"mode": "stored", "counts": {"train": 15, "val": 5}},
            created_utc=datetime.now(timezone.utc).isoformat(), builder={"test": True},
        )
        self.manifest_path = self.root / "manifest.json"
        manifest.publish(self.manifest_path)
        self.manifest_sha = manifest.sha256

    def tearDown(self):
        self.temporary.cleanup()

    def test_seeded_random_is_exact_stratified_and_reproducible(self):
        first = compile_random_mag(self.manifest_path, self.root / "first.json", 42, .6, .2, .2, True)
        second = compile_random_mag(self.manifest_path, self.root / "second.json", 42, .6, .2, .2, True)
        one = load_split_plan(first, self.manifest_sha)
        two = load_split_plan(second, self.manifest_sha)
        self.assertEqual(one.assignments, two.assignments)
        self.assertEqual(one.table_sha256, two.table_sha256)
        self.assertEqual(Counter(value.label for value in one.assignments.values()),
                         {"train": 12, "val": 4, "test": 4})
        for source in ("A", "B"):
            labels = Counter(value.label for value in one.assignments.values()
                             if value.source_name == source)
            self.assertEqual(labels, {"train": 6, "val": 2, "test": 2})

    def test_each_strategy_materializes_a_pinned_plan(self):
        configurations = (
            SplitConfig(strategy="stored_labels"),
            SplitConfig(strategy="source_holdout", source_labels={"A": "train", "B": "test"}),
            SplitConfig(strategy="random_mag", seed=7, train_fraction=.8,
                        validation_fraction=.1, test_fraction=.1),
        )
        for config in configurations:
            path = materialize_split_plan(self.manifest_path, config, self.root / "generated")
            plan = load_split_plan(path, self.manifest_sha)
            self.assertEqual(len(plan.assignments), 20)
            self.assertEqual(materialize_split_plan(
                self.manifest_path, config, self.root / "generated"), path)

    def test_random_strategy_can_pin_a_complete_source(self):
        config = SplitConfig(
            strategy="random_mag", seed=9, train_fraction=.8,
            validation_fraction=.2, test_fraction=0,
            source_labels={"B": "test"},
        )
        path = materialize_split_plan(self.manifest_path, config, self.root / "hybrid")
        plan = load_split_plan(path, self.manifest_sha)
        source_b = [value.label for value in plan.assignments.values() if value.source_name == "B"]
        source_a = [value.label for value in plan.assignments.values() if value.source_name == "A"]
        self.assertEqual(Counter(source_b), {"test": 10})
        self.assertEqual(Counter(source_a), {"train": 8, "val": 2})

    def test_existing_strategy_reuses_a_versioned_plan(self):
        path = compile_random_mag(self.manifest_path, self.root / "fixed.json", 1, .8, .1, .1)
        resolved = materialize_split_plan(
            self.manifest_path, SplitConfig(strategy="existing", plan=str(path)),
            self.root / "not-used",
        )
        self.assertEqual(resolved, path)


if __name__ == "__main__":
    unittest.main()
