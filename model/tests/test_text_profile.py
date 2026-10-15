from __future__ import annotations

import unittest
from collections import Counter
import importlib.util
from pathlib import Path

CLIP_TEXT = Path(__file__).resolve().parents[2] / "data" / "clip_text"
SPEC = importlib.util.spec_from_file_location("clip_text_profile", CLIP_TEXT / "profile.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
distribution = MODULE.distribution


class TextProfileTest(unittest.TestCase):
    def test_weighted_statistics_do_not_expand_observations(self):
        result = distribution(Counter({1: 10_000_000, 3: 10_000_000}))
        self.assertEqual(result["count"], 20_000_000)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["median"], 2)
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 3)


if __name__ == "__main__":
    unittest.main()
