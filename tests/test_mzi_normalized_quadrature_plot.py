import copy
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.transforms import Bbox
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config
from analysis.mzi_normalized_quadrature_plot import (
    _tick_label,
    NormalizedQuadrature,
    PlotInputError,
    centered_column_crop,
    conservative_upper_bound,
    dataset_confidence_title,
    ellipse_geometry,
    full_restoration_label,
    normalize_inputs,
    pooled_bound_label,
    render_plot,
    standard_quantum_label,
)


class NormalizedQuadraturePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.payload = json.loads(
            (
                ROOT / "results/analysis/mzi-pooled-analysis.json"
            ).read_text(encoding="utf-8")
        )

    def test_normalizes_centers_and_covariances_in_configured_order(self):
        confidence, pooled_upper_bound, inputs = normalize_inputs(
            self.payload,
            self.config,
        )

        self.assertEqual(2, self.payload["schema_version"])
        self.assertEqual(0.95, confidence)
        self.assertEqual(0.16, pooled_upper_bound)
        self.assertEqual(
            [dataset["id"] for dataset in self.config["datasets"]],
            [item.dataset_id for item in inputs],
        )
        first_record = self.payload["inputs"][0]
        normalization = first_record["normalization_Rinfty_cps"]
        np.testing.assert_allclose(
            np.asarray(first_record["q_hat_cps"]) / normalization,
            inputs[0].center,
        )
        np.testing.assert_allclose(
            np.asarray(first_record["covariance_cps2"]) / normalization**2,
            inputs[0].covariance,
        )

    def test_legend_labels_use_display_ids(self):
        confidence, _, inputs = normalize_inputs(self.payload, self.config)

        self.assertEqual(
            [f"D{index}" for index in range(1, 7)],
            [item.label for item in inputs],
        )
        self.assertEqual(
            "Per-dataset 95\\% joint\nconfidence regions",
            dataset_confidence_title(confidence),
        )

    def test_pooled_upper_bound_is_rounded_up_conservatively(self):
        raw_upper_bound = self.payload["summary"]["eta_upper"]

        displayed_upper_bound = conservative_upper_bound(raw_upper_bound, 2)

        self.assertEqual(0.16, displayed_upper_bound)
        self.assertGreaterEqual(displayed_upper_bound, raw_upper_bound)
        self.assertLess(displayed_upper_bound - raw_upper_bound, 0.01)
        self.assertEqual(
            "Pooled 95\\% upper bound,\n$\\eta_\\infty<0.16$",
            pooled_bound_label(0.95, displayed_upper_bound),
        )
        self.assertEqual(
            "Standard quantum theory,\n$\\eta_\\infty=0$",
            standard_quantum_label(),
        )
        self.assertEqual(
            "Full fringe restoration,\n$\\eta_\\infty=1$",
            full_restoration_label(),
        )

    def test_rejects_invalid_pooled_upper_bound(self):
        for invalid in (-0.01, 1.01):
            with self.subTest(invalid=invalid):
                payload = copy.deepcopy(self.payload)
                payload["summary"]["eta_upper"] = invalid

                with self.assertRaisesRegex(
                    PlotInputError,
                    "must lie between 0 and 1",
                ):
                    normalize_inputs(payload, self.config)

    def test_ellipse_geometry_uses_two_dimensional_confidence_cutoff(self):
        item = NormalizedQuadrature(
            dataset_id="test",
            label="test",
            center=np.zeros(2),
            covariance=np.diag([0.01, 0.04]),
        )

        major, minor, angle = ellipse_geometry(item, 0.95)

        cutoff = chi2.ppf(0.95, df=2)
        self.assertAlmostEqual(2.0 * np.sqrt(0.04 * cutoff), major)
        self.assertAlmostEqual(2.0 * np.sqrt(0.01 * cutoff), minor)
        self.assertAlmostEqual(90.0, abs(angle))

    def test_half_ticks_are_unlabeled(self):
        self.assertEqual("", _tick_label(-0.5, None))
        self.assertEqual("", _tick_label(0.5, None))
        self.assertEqual(r"$0$", _tick_label(0.0, None))
        self.assertEqual(r"$1$", _tick_label(1.0, None))

    def test_rejects_inputs_that_do_not_match_configured_datasets(self):
        payload = copy.deepcopy(self.payload)
        payload["inputs"].pop()

        with self.assertRaisesRegex(
            PlotInputError,
            "do not match configured datasets",
        ):
            normalize_inputs(payload, self.config)

    def test_rejects_nonpositive_normalization(self):
        payload = copy.deepcopy(self.payload)
        payload["inputs"][0]["normalization_Rinfty_cps"] = 0.0

        with self.assertRaisesRegex(PlotInputError, "must be positive"):
            normalize_inputs(payload, self.config)

    def test_crop_centers_content_in_exact_column_width(self):
        content = Bbox.from_extents(-0.1, 0.2, 2.9, 1.8)

        crop = centered_column_crop(content)

        self.assertAlmostEqual(3.4, crop.width)
        self.assertAlmostEqual(content.x0 - crop.x0, crop.x1 - content.x1)
        self.assertAlmostEqual(content.y0 - 0.5 / 72.0, crop.y0)
        self.assertAlmostEqual(content.y1 + 0.5 / 72.0, crop.y1)

    def test_crop_rejects_content_wider_than_column(self):
        with self.assertRaisesRegex(PlotInputError, "wider"):
            centered_column_crop(Bbox.from_extents(0.0, 0.0, 3.5, 1.0))

    def test_rendered_products_have_exact_column_width(self):
        confidence, pooled_upper_bound, inputs = normalize_inputs(
            self.payload,
            self.config,
        )
        with tempfile.TemporaryDirectory() as temporary:
            out_base = Path(temporary) / "normalized"

            pdf_path, png_path = render_plot(
                confidence,
                pooled_upper_bound,
                inputs,
                out_base,
            )

            media_box = re.search(
                rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
                pdf_path.read_bytes(),
            )
            self.assertIsNotNone(media_box)
            self.assertAlmostEqual(244.8, float(media_box.group(1)), places=5)
            self.assertGreater(float(media_box.group(2)), 0.0)
            with png_path.open("rb") as handle:
                header = handle.read(24)
            self.assertEqual(b"\x89PNG\r\n\x1a\n", header[:8])
            width, height = struct.unpack(">II", header[16:24])
            self.assertEqual(680, width)
            self.assertGreater(height, 0)


if __name__ == "__main__":
    unittest.main()
