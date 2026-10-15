from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

CLIP_TEXT = Path(__file__).resolve().parents[2] / "data" / "clip_text"
sys.path.append(str(CLIP_TEXT))

from facts import mag_id, normalise_accession, parse_hits
from render import best_role, info_types
from render_cross import answers as multi_mag_answers
from views import compose_richness, informative_label
from vocab import _clean_prose


class TextFactsTest(unittest.TestCase):
    def test_source_specific_mag_ids(self):
        self.assertEqual(mag_id("IMG_JGI", "3300_s1"), "3300_s1")
        self.assertEqual(mag_id("JGI_IMG_ENG", "3300_s1"), "JGI_IMG_ENG__3300_s1")

    def test_accession_normalisation(self):
        self.assertEqual(normalise_accession("ko", "KO:K00001"), "K00001")
        self.assertEqual(normalise_accession("ec", "EC:1.2.3.4"), "1.2.3.4")
        self.assertEqual(normalise_accession("pfam", "pfam00001"), "PF00001")

    def test_hit_evidence_is_preserved(self):
        text = "gene\tYes\tKO:K01874\t57.61\t11\t582\t9\t585\t2.2e-243\t783\t572\n"
        hit = parse_hits("ko", text)["gene"][0]
        self.assertEqual(hit["accession"], "K01874")
        self.assertEqual(hit["query_start"], 11)
        self.assertAlmostEqual(hit["percent_identity"], 57.61, places=2)
        self.assertAlmostEqual(hit["evalue"], 2.2e-243)

    def test_richness_keeps_high_at_least_as_detailed(self):
        views = {
            "product": "A transporter.",
            "domains": "Contains an ATP-binding domain.",
            "family": "This family transports ions. It is widespread in bacteria. A third detail.",
            "taxonomic": "A transporter from a bacterial genome.",
        }
        medium = compose_richness(views, "medium")
        high = compose_richness(views, "high")
        self.assertIn("A transporter", medium)
        self.assertGreaterEqual(len(high), len(medium))
        self.assertNotIn("A third detail", high)

    def test_accession_cleanup_removes_empty_family_shell(self):
        self.assertEqual(
            _clean_prose("uncharacterized protein (TIGR01777 family)"),
            "uncharacterized protein",
        )

    def test_generic_product_is_not_informative(self):
        self.assertFalse(informative_label("uncharacterized protein"))
        gene = {"product": "uncharacterized protein", "ko": [], "ec": [],
                "cog": [], "pfam": [], "tigr": []}
        self.assertEqual(info_types(gene, {}), [])
        self.assertEqual(best_role(gene, {}), "")

    def test_resolved_domain_makes_generic_product_informative(self):
        gene = {"product": "uncharacterized protein", "ko": [], "ec": [],
                "cog": [], "pfam": [{"accession": "PF00001"}], "tigr": []}
        vocab = {"pfam": {"PF00001": {"name": "ABC transporter domain"}}}
        self.assertEqual(info_types(gene, vocab), ["pfam"])
        self.assertEqual(best_role(gene, vocab), "ABC transporter domain")

    def test_multi_mag_answers_compare_without_claiming_phenotype(self):
        rows = [
            {"taxonomy": ["Bacteria", "Firmicutes"], "environment": "soil",
             "n_genes": 100, "n_contigs": 2, "informative_genes": 80,
             "categories": ["Translation", "Motility"], "pathways": ["Ribosome"]},
            {"taxonomy": ["Bacteria", "Proteobacteria"], "environment": "water",
             "n_genes": 120, "n_contigs": 3, "informative_genes": 90,
             "categories": ["Translation", "Transport"], "pathways": ["Secretion"]},
        ]
        medium, high = multi_mag_answers(rows)
        self.assertIn("Shared broad functional evidence includes Translation", medium)
        self.assertIn("does not demonstrate phenotype", high)


if __name__ == "__main__":
    unittest.main()
