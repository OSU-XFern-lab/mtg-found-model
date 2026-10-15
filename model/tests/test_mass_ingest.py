from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mass_ingest.common import bucket_for_hash, task_id
from mass_ingest.prepare import balanced_groups


class MassIngestHelpersTest(unittest.TestCase):
    def test_explicit_task_overrides_slurm_worker(self):
        with patch.dict(os.environ, {"TASK_ID": "129", "SLURM_ARRAY_TASK_ID": "1"}, clear=False):
            self.assertEqual(task_id(), 129)

    def test_hash_bucketing_is_stable(self):
        digest = bytes.fromhex("12345678" + "00" * 12)
        self.assertEqual(bucket_for_hash(digest, 512), int.from_bytes(digest[:4], "big") & 511)

    def test_balanced_groups_assign_every_mag_once(self):
        rows = [
            {"source": "S", "genome": "m{}".format(i), "n_genes": size}
            for i, size in enumerate((10, 9, 8, 2, 1))
        ]
        groups = balanced_groups(rows, 2)
        observed = [item["mag_id"] for group in groups for item in group]
        self.assertEqual(sorted(observed), ["m0", "m1", "m2", "m3", "m4"])
        self.assertLessEqual(abs(sum(rows[int(x[1:])]["n_genes"] for x in
                                     [item["mag_id"] for item in groups[0]]) -
                                 sum(rows[int(x[1:])]["n_genes"] for x in
                                     [item["mag_id"] for item in groups[1]])), 10)


if __name__ == "__main__":
    unittest.main()
