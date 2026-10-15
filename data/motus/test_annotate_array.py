import argparse
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("array_runner", Path(__file__).with_name("annotate_array.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AnnotationTests(unittest.TestCase):
    def test_completed_mag_is_preserved_and_partial_run_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "mag.fa.gz"
            with gzip.open(raw, "wt") as handle:
                handle.write(">contig\nATGAAA\n")
            plan = root / "plan.tsv"
            plan.write_text("test plan\n")
            reports = root / "reports"
            reports.mkdir()
            args = argparse.Namespace(task=0, cpus=2, min_free_gb=0, database=root, plan=plan)
            row = {"mag": "mag", "path": str(raw), "bases": "6"}
            calls = []

            def command(argv, **kwargs):
                calls.append(argv[0])
                if argv[0] == "prodigal":
                    Path(argv[argv.index("-a") + 1]).write_text(">contig_1\nMK\n")
                else:
                    directory = Path(argv[argv.index("--output_dir") + 1])
                    (directory / "mag_emapper.emapper.annotations").write_text(
                        "#query\tKEGG_ko\tGOs\tEC\ncontig_1\tko:K00001\t-\t-\n")

            with patch.object(runner, "OUTPUTS", root), patch.object(runner.subprocess, "run", side_effect=command):
                result = runner.annotate(row, args, reports)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["genes"], 1)
                self.assertEqual(result["genes_with_ko_go_ec"], 1)
                self.assertEqual(runner.annotate(row, args, reports)["status"], "existing")
                self.assertEqual(calls, ["prodigal", "emapper.py"])
                (root / "annotations/per_mag/mag/_ANNOTATION_DONE").unlink()
                result = runner.annotate(row, args, reports)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(calls, ["prodigal", "emapper.py", "emapper.py"])
                (root / "annotations/per_mag/mag/_ANNOTATION_DONE").unlink()
                with patch.object(runner.subprocess, "run", side_effect=RuntimeError("interrupted")):
                    result = runner.annotate(row, args, reports)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(runner.complete(root / "annotations/per_mag/mag"))


if __name__ == "__main__":
    unittest.main()
