import copy
import json
import unittest
from pathlib import Path

import numpy as np

from analysis.mzi_null_plot import PlotInputError, _parse_plot_inputs


ROOT = Path(__file__).resolve().parents[1]


class NullPlotInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sources = sorted(
            (ROOT / "results/datasets").glob(
                "*/analysis/mzi-null-bounds.json"
            )
        )
        if not sources:
            raise RuntimeError("no packaged mzi-null-bounds.json inputs found")
        cls.summary = json.loads(sources[0].read_text(encoding="utf-8"))[
            "summary"
        ]

    def test_accepts_current_scientific_json(self):
        inputs = _parse_plot_inputs(self.summary)

        self.assertEqual(self.summary["confidence"], inputs.confidence)
        self.assertEqual(self.summary["Rinfty_cps"], inputs.normalization)
        self.assertEqual(
            self.summary["full_restoration_mahalanobis_distance"],
            inputs.full_restoration_distance,
        )
        np.testing.assert_array_equal(
            [
                self.summary["full_restoration_nearest_C_cps"],
                self.summary["full_restoration_nearest_S_cps"],
            ],
            inputs.full_restoration_nearest_q,
        )

    def test_missing_confidence_is_rejected(self):
        summary = copy.deepcopy(self.summary)
        summary.pop("confidence")

        with self.assertRaisesRegex(
            PlotInputError,
            r"summary\.confidence is required",
        ):
            _parse_plot_inputs(summary)

    def test_invalid_confidence_is_rejected(self):
        cases = (
            (True, "must be numeric"),
            ("0.95", "must be numeric"),
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (0.5, "must lie between 0.5 and 1"),
            (1.0, "must lie between 0.5 and 1"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                summary = copy.deepcopy(self.summary)
                summary["confidence"] = value

                with self.assertRaisesRegex(
                    PlotInputError,
                    rf"summary\.confidence {message}",
                ):
                    _parse_plot_inputs(summary)

    def test_rejects_nonpositive_rinfinity(self):
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                summary = copy.deepcopy(self.summary)
                summary["Rinfty_cps"] = value

                with self.assertRaisesRegex(
                    PlotInputError,
                    r"summary\.Rinfty_cps must be positive",
                ):
                    _parse_plot_inputs(summary)

    def test_missing_covariance_is_rejected(self):
        summary = copy.deepcopy(self.summary)
        summary.pop("CLSL_cov_cps2")

        with self.assertRaisesRegex(
            PlotInputError,
            r"summary\.CLSL_cov_cps2 is required",
        ):
            _parse_plot_inputs(summary)

    def test_rejects_nonnumeric_and_nonfinite_covariance(self):
        for value, message in (
            (True, "must be numeric"),
            ("0.0", "must be numeric"),
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
        ):
            with self.subTest(value=value):
                summary = copy.deepcopy(self.summary)
                summary["CLSL_cov_cps2"] = value

                with self.assertRaisesRegex(
                    PlotInputError,
                    rf"summary\.CLSL_cov_cps2 {message}",
                ):
                    _parse_plot_inputs(summary)

    def test_zero_off_diagonal_covariance_is_valid(self):
        summary = copy.deepcopy(self.summary)
        summary["CLSL_cov_cps2"] = 0.0

        inputs = _parse_plot_inputs(summary)

        self.assertEqual(0.0, inputs.covariance[0, 1])
        self.assertEqual(0.0, inputs.covariance[1, 0])

    def test_rejects_nonpositive_uncertainties(self):
        for field in ("CL_sigma_cps", "SL_sigma_cps"):
            with self.subTest(field=field):
                summary = copy.deepcopy(self.summary)
                summary[field] = 0.0

                with self.assertRaisesRegex(
                    PlotInputError,
                    rf"summary\.{field} must be positive",
                ):
                    _parse_plot_inputs(summary)

    def test_rejects_non_positive_definite_covariance(self):
        summary = copy.deepcopy(self.summary)
        summary["CLSL_cov_cps2"] = (
            2.0 * summary["CL_sigma_cps"] * summary["SL_sigma_cps"]
        )

        with self.assertRaisesRegex(
            PlotInputError,
            "positive-definite covariance matrix",
        ):
            _parse_plot_inputs(summary)

    def test_missing_geometric_results_are_rejected(self):
        fields = (
            "full_restoration_mahalanobis_distance",
            "full_restoration_nearest_C_cps",
            "full_restoration_nearest_S_cps",
        )
        for field in fields:
            with self.subTest(field=field):
                summary = copy.deepcopy(self.summary)
                summary.pop(field)

                with self.assertRaisesRegex(
                    PlotInputError,
                    rf"summary\.{field} is required",
                ):
                    _parse_plot_inputs(summary)

    def test_nonfinite_geometric_results_are_rejected(self):
        fields = (
            "full_restoration_mahalanobis_distance",
            "full_restoration_nearest_C_cps",
            "full_restoration_nearest_S_cps",
        )
        for field in fields:
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    summary = copy.deepcopy(self.summary)
                    summary[field] = value

                    with self.assertRaisesRegex(
                        PlotInputError,
                        rf"summary\.{field} must be finite",
                    ):
                        _parse_plot_inputs(summary)

    def test_rejects_negative_geometric_distance(self):
        summary = copy.deepcopy(self.summary)
        summary["full_restoration_mahalanobis_distance"] = -0.1

        with self.assertRaisesRegex(
            PlotInputError,
            "full_restoration_mahalanobis_distance must be nonnegative",
        ):
            _parse_plot_inputs(summary)


if __name__ == "__main__":
    unittest.main()
