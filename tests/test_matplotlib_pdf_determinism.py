import unittest
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfFile

ROOT = Path(__file__).resolve().parents[1]
from analysis.matplotlib_pdf_determinism import make_pdf_font_subsets_deterministic


class MatplotlibPdfDeterminismTests(unittest.TestCase):
    def test_dict_values_use_value_based_subset_prefix(self):
        make_pdf_font_subsets_deterministic()
        glyph_map = {10: 18, 11: 28, 12: 59, 13: 61}

        actual = PdfFile._get_subset_prefix(glyph_map.values())
        expected = PdfFile._get_subset_prefix(frozenset(glyph_map.values()))

        self.assertEqual(expected, actual)

    def test_installation_is_idempotent(self):
        make_pdf_font_subsets_deterministic()
        installed = PdfFile._get_subset_prefix

        make_pdf_font_subsets_deterministic()

        self.assertIs(installed, PdfFile._get_subset_prefix)


if __name__ == "__main__":
    unittest.main()
