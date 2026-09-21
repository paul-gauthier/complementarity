import json
import unittest
from pathlib import Path

import numpy as np

from analysis.artifact_config import dataset_runs, load_config
from analysis.mzi_analysis import fit_scan, scan_from_records
from analysis.mzi_io import load_dark, load_plan_length, load_points, split_by_plan_length

ROOT = Path(__file__).resolve().parents[1]


class ConfiguredFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    @staticmethod
    def load_passes(run):
        run_dir = ROOT / run["path"]
        return load_dark(run_dir), split_by_plan_length(
            load_points(run_dir), load_plan_length(run_dir)
        )

    def assert_finite_fit(self, fit):
        self.assertIsNotNone(fit)
        for field in ("A1", "A2", "V", "x0", "P", "total_fringe"):
            self.assertTrue(np.isfinite(getattr(fit, field)), field)
        self.assertGreater(fit.P, 0.0)
        for field in (
            "A1_sigma", "A2_sigma", "A_sig_sigma", "V_sigma",
            "total_fringe_sigma", "Fringe1_sigma", "Fringe2_sigma",
        ):
            value = getattr(fit, field)
            self.assertTrue(np.isfinite(value), field)
            self.assertGreaterEqual(value, 0.0, field)

    def test_every_configured_pass_has_finite_fixed_period_fits(self):
        for dataset in self.config["datasets"]:
            bounds_path = (
                ROOT / "results/datasets" / dataset["id"]
                / "analysis/mzi-null-bounds.json"
            )
            bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
            period = float(bounds["summary"]["period_V"])
            for run in dataset_runs(self.config, dataset):
                dark, passes = self.load_passes(run)
                for pass_number, records in enumerate(passes, start=1):
                    for channel, fit_field in (("C", "coincidence"), ("I", "singles")):
                        with self.subTest(
                            dataset=dataset["display_id"], run=run["id"],
                            pass_number=pass_number, channel=channel,
                        ):
                            data, duration = scan_from_records(records, dark)
                            fit = getattr(
                                fit_scan(data, channel, period, dark, duration), fit_field
                            )
                            self.assert_finite_fit(fit)
                            self.assertEqual(period, fit.P)
                            self.assertTrue(np.isnan(fit.P_sigma))

    def test_every_phase_reference_pass_has_a_finite_free_period_seed(self):
        options = self.config["null_bounds"]
        condition_runs = {
            "Erase": "run-1-erase",
            "Launch": "run-1-launch",
            "Preserve": "run-2-preserve",
        }
        channel = options["phase_channel"]
        fit_field = {"C": "coincidence", "I": "singles"}[channel]
        for dataset in self.config["datasets"]:
            runs = {run["id"]: run for run in dataset_runs(self.config, dataset)}
            run = runs[condition_runs[options["phase_reference"]]]
            dark, passes = self.load_passes(run)
            for pass_number, records in enumerate(passes, start=1):
                with self.subTest(
                    dataset=dataset["display_id"], run=run["id"],
                    pass_number=pass_number, channel=channel,
                ):
                    data, duration = scan_from_records(records, dark)
                    fit = getattr(fit_scan(data, channel, None, dark, duration), fit_field)
                    self.assert_finite_fit(fit)
                    self.assertTrue(np.isfinite(fit.P_sigma))
                    self.assertGreaterEqual(fit.P_sigma, 0.0)


if __name__ == "__main__":
    unittest.main()
