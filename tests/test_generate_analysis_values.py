import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.generate_analysis_values import build_macros, parse_args


class GenerateAnalysisValuesTests(unittest.TestCase):
    def test_cli_requires_explicit_input_and_output_paths(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args([])

        self.assertEqual(2, raised.exception.code)

    def test_analysis_exports_expected_macro_inventory(self):
        dataset_id = "2026-03-08-12-21-56--2026-03-08-13-28-33"
        dataset = ROOT / "results" / "datasets" / dataset_id

        def load(relative: str):
            return json.loads((dataset / relative).read_text(encoding="utf-8"))

        macros = build_macros(
            load("analysis/mzi-null-bounds.json"),
            load("analysis/mzi-alt-joint.json"),
            load("conditions/conditions.json"),
        )
        expected = {
            "ResultPairedScans",
            "ResultPairedScansAsText",
            "ResultPairedPoints",
            "ResultPairedDegreesFreedom",
            "ResultReducedChiSquared",
            "ResultFringePeriod",
            "ResultFringePeriodSigma",
            "ResultCAddLetter",
            "ResultCAddSigmaLetter",
            "ResultSAddLetter",
            "ResultSAddSigmaLetter",
            "ResultCAddSupplement",
            "ResultCAddSigmaSupplement",
            "ResultSAddSupplement",
            "ResultSAddSigmaSupplement",
            "ResultCAddSAddCovariance",
            "ResultAAddEstimateLetter",
            "ResultAAddEstimateSupplement",
            "ResultGaussianPValueLetter",
            "ResultGaussianPValueSupplement",
            "ResultPermutationPValueLetter",
            "ResultPermutationPValueSupplement",
            "ResultAAddBoundLetter",
            "ResultAAddBoundSupplement",
            "ResultFullRestorationGap",
            "ResultRInfinityLetter",
            "ResultRInfinitySupplement",
            "ResultRFinitePathLetter",
            "ResultEtaInfinityBoundLetter",
            "ResultEtaInfinityBoundSupplement",
            "ResultTInfinityExact",
            "ResultRpmMeasured",
            "ResultPreserveCoincidenceAmplitudeMeasured",
            "ResultPreserveCoincidenceVisibilityMeasured",
            "ResultEraseCoincidenceAmplitudeMeasured",
            "ResultEraseCoincidenceVisibilityMeasured",
            "ResultLaunchSinglesAmplitudeMeasured",
            "ResultEtaFinitePathBoundLetter",
            "ResultFullRestorationSigmaLetter",
            "ResultRestorationToLaunchSinglesRatio",
        }
        self.assertEqual(expected, set(macros))


if __name__ == "__main__":
    unittest.main()
