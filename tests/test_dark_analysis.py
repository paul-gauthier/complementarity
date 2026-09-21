import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from analysis.dark_analysis import (
    COINCIDENCE_WINDOW_S,
    DarkModel,
    accidentals_counts,
    corrected_coincidence_rate,
)
from analysis.darks_accidentals import (
    _plan_details,
    main,
    parse_args,
    run_specs_from_config,
)


class DarkAnalysisTests(unittest.TestCase):
    def test_window_and_configured_runs(self):
        self.assertAlmostEqual(25e-9, COINCIDENCE_WINDOW_S)
        specs = run_specs_from_config("2026-03-08-12-21-56--2026-03-08-13-28-33")
        self.assertEqual(4, len(specs))
        self.assertEqual({"Erase", "Launch", "Preserve"}, {item.condition for item in specs})

    def test_cli_selects_dataset_by_id(self):
        dataset_id = "2026-03-08-12-21-56--2026-03-08-13-28-33"
        self.assertEqual(
            dataset_id,
            parse_args(["--dataset-id", dataset_id]).dataset_id,
        )

    def test_dark_model_rejects_invalid_inputs(self):
        valid = {"Ns": 0, "Ni": 0, "Nc": 0, "Ni2": 0, "Nc2": 0, "T": 1.0}
        for field, value in (("T", 0), ("T", float("nan")), ("Ns", -1), ("Ni", True)):
            with self.subTest(field=field, value=value):
                values = dict(valid)
                values[field] = value
                with self.assertRaises(ValueError):
                    DarkModel.from_counts(**values)
        with self.assertRaisesRegex(ValueError, "w must be finite and positive"):
            DarkModel.from_counts(**valid, w=0)

    def test_accidentals_and_channel_corrections(self):
        self.assertEqual((6.0, 15.0), accidentals_counts(4, 6, 2.0, w=0.5))
        dark = DarkModel.from_counts(
            Ns=2, Ni=4, Nc=3, T=2.0, Ni2=6, Nc2=5, w=0.5,
        )
        for channel, expected in (("I", (4.0, 47 / 16)), ("I2", (3.5, 67 / 16))):
            with self.subTest(channel=channel):
                self.assertEqual(
                    expected,
                    corrected_coincidence_rate(20, 4.0, 2.0, 3.0, dark, channel),
                )

    def test_numerical_helpers_require_positive_finite_exposure(self):
        dark = DarkModel.from_counts(Ns=0, Ni=0, Nc=0, T=1.0, Ni2=0, Nc2=0)
        for duration in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "dur must be finite and positive"):
                    accidentals_counts(4, 6, duration)
                with self.assertRaisesRegex(ValueError, "T_tot must be finite and positive"):
                    corrected_coincidence_rate(20, duration, 2.0, 3.0, dark)

    def test_corrected_rate_requires_known_channel(self):
        dark = DarkModel.from_counts(Ns=0, Ni=0, Nc=0, T=1.0, Ni2=0, Nc2=0)
        with self.assertRaisesRegex(ValueError, "channel must be I or I2"):
            corrected_coincidence_rate(20, 4.0, 2.0, 3.0, dark, channel="II")

    def test_plan_requires_canonical_executed_voltages(self):
        details = _plan_details(
            {"deltas_exec_V": [0.1, 0.2]},
            run_label="test run",
            record_count=4,
        )
        self.assertEqual(
            {
                "planned_voltage_min_V": 0.1,
                "planned_voltage_max_V": 0.2,
                "points_per_scan": 2,
                "scan_count": 2,
            },
            details,
        )

        malformed_plans = (
            {"deltas_V": [0.1]},
            {"deltas_exec_V": []},
            {"deltas_exec_V": ["invalid"]},
            {"deltas_exec_V": ["0.1"]},
            {"deltas_exec_V": [True]},
            {"deltas_exec_V": [float("inf")]},
        )
        for plan in malformed_plans:
            with self.subTest(plan=plan):
                with self.assertRaisesRegex(ValueError, "deltas_exec_V"):
                    _plan_details(plan, run_label="test run", record_count=1)

    def test_report_declares_and_validates_the_canonical_window(self):
        dataset_id = "2026-03-08-12-21-56--2026-03-08-13-28-33"
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--dataset-id", dataset_id, "--outdir", temporary])
            payload = json.loads(
                (Path(temporary) / "darks-accidentals.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(5, payload["schema_version"])
        self.assertEqual(
            {
                "file": "dark.json",
                "field": "w_s",
                "location": "top-level",
                "unit": "s",
            },
            payload["coincidence_window"]["input"],
        )
        self.assertEqual(
            {
                "required_for_every_run": True,
                "matched_canonical_width": True,
            },
            payload["coincidence_window"]["validation"],
        )


if __name__ == "__main__":
    unittest.main()
