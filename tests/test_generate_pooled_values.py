import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.generate_pooled_values import render_macros


EXPECTED_VALUES = {
    "ResultPooledEtaEstimateLetter": "0.09",
    "ResultPooledEtaInfinityBoundLetter": "0.16",
    "ResultPooledNullPValueLetter": "0.79",
    "ResultPooledGoodnessOfFitPValueLetter": "0.95",
    "ResultPooledLeaveOneOutMinimum": "0.154",
    "ResultPooledLeaveOneOutMaximum": "0.168",
    "ResultPooledLaunchOpticsTransmissionLetter": "0.99",
    "ResultPooledAtmosphericTransmissionMinimumLetter": "0.86",
    "ResultPooledAtmosphericTransmissionMaximumLetter": "0.92",
    "ResultPooledMilkyWayTransmissionMinimumLetter": "0.84",
    "ResultPooledMilkyWayTransmissionMaximumLetter": "0.94",
    "ResultPooledCommonIgmTransmission": "0.9839",
    "ResultPooledCommonIgmTransmissionLetter": "0.98",
    "ResultPooledTInfinityMinimumLetter": "0.73",
    "ResultPooledTInfinityMaximumLetter": "0.82",
    "ResultPooledTInfinityMinimumPercentLetter": "73",
    "ResultPooledTInfinityMaximumPercentLetter": "82",
    "ResultPooledTFinitePathMinimumLetter": "0.75",
    "ResultPooledTFinitePathMaximumLetter": "0.83",
    "ResultPooledEtaFinitePathBoundLetter": "0.16",
    "ResultPooledEtaLaunchInfinityBoundLetter": "0.078",
    "ResultPooledEtaLaunchFinitePathBoundLetter": "0.077",
    "ResultPooledLaunchOpticsLossPercentLetter": "1.5",
    "ResultPooledAtmosphericLossMinimumPercentLetter": "8.5",
    "ResultPooledAtmosphericLossMaximumPercentLetter": "14",
    "ResultPooledMilkyWayLossMinimumPercentLetter": "5.6",
    "ResultPooledMilkyWayLossMaximumPercentLetter": "16",
    "ResultPooledIgmLossPercentLetter": "1.6",
    "ResultPooledTInfinityLossMinimumPercentLetter": "18",
    "ResultPooledTInfinityLossMaximumPercentLetter": "27",
    "ResultPooledTFinitePathLossMinimumPercentLetter": "17",
    "ResultPooledTFinitePathLossMaximumPercentLetter": "25",
}


def parsed_macros(rendered: str) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\([^}]+)\}\{([^}]+)\}", rendered))


class GeneratePooledValuesTests(unittest.TestCase):
    def test_renderer_requires_schema_version_2(self):
        source = ROOT / "results" / "analysis" / "mzi-pooled-analysis.json"
        for version in (None, "2", 3):
            with self.subTest(version=version):
                payload = json.loads(source.read_text(encoding="utf-8"))
                if version is None:
                    payload.pop("schema_version")
                else:
                    payload["schema_version"] = version
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pooled.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "unsupported pooled schema version"
                    ):
                        render_macros(path)

    def test_reviewed_pooled_values_are_rendered_as_macros(self):
        rendered = render_macros(
            ROOT / "results" / "analysis" / "mzi-pooled-analysis.json"
        )
        self.assertEqual(EXPECTED_VALUES, parsed_macros(rendered))
        self.assertIn("% Source SHA-256:", rendered)

    def test_every_displayed_upper_bound_is_conservative(self):
        source = ROOT / "results" / "analysis" / "mzi-pooled-analysis.json"
        pooled = json.loads(source.read_text(encoding="utf-8"))
        values = parsed_macros(render_macros(source))
        comparisons = {
            "ResultPooledEtaInfinityBoundLetter": (
                pooled["summary"]["eta_upper"],
                2,
            ),
            "ResultPooledEtaFinitePathBoundLetter": (
                pooled["summary"]["eta_upper_finite_path"],
                2,
            ),
            "ResultPooledEtaLaunchInfinityBoundLetter": (
                pooled["summary"]["eta_upper_launch_infinity"],
                3,
            ),
            "ResultPooledEtaLaunchFinitePathBoundLetter": (
                pooled["summary"]["eta_upper_launch_finite_path"],
                3,
            ),
        }
        for macro, (unrounded, decimals) in comparisons.items():
            displayed = float(values[macro])
            self.assertGreaterEqual(displayed, unrounded, macro)
            self.assertLess(
                displayed - unrounded,
                10.0 ** (-decimals) + 1e-15,
                macro,
            )


if __name__ == "__main__":
    unittest.main()
