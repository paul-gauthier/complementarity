import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config
from analysis.generate_dataset_summary import render_tables


class DatasetSummaryTests(unittest.TestCase):
    def test_summary_uses_semantic_roles_and_selected_atmosphere(self):
        transmission, analysis = render_tables(load_config(), ROOT / "results")

        self.assertIn(
            r"D3"
            r" & $0.8982^{\mathrm{A}}$"
            r" & $0.9384$"
            r" & $0.9594$"
            r" & $0.9384$"
            r" & $0.8305$"
            r" & $0.8171$",
            transmission,
        )
        self.assertIn(r"& $T_{\mathrm{fin}}$", transmission)
        self.assertIn(r"& $T_\infty$ \\", transmission)
        self.assertIn(r"& $\langle T_{\mathrm{MW}}\rangle$", transmission)
        self.assertIn(r"& $\min(T_{\mathrm{MW}})$", transmission)
        self.assertIn(r"D6 &", transmission)
        self.assertIn(r"& \multicolumn{2}{c}{Model independent}", analysis)
        self.assertIn(r"& \multicolumn{3}{c}{Infinite future}", analysis)
        self.assertIn(r"& \multicolumn{3}{c}{Finite path} \\", analysis)
        self.assertIn(
            "$R_{\\mathrm{pm}}$\n"
            "& $A_{\\mathrm{add},95}$\n"
            "& $R_\\infty$",
            analysis,
        )
        self.assertIn(r"& $R_{\mathrm{fin}}$", analysis)
        self.assertIn(r"& $\eta_{\mathrm{fin},95}$", analysis)
        self.assertEqual(1, analysis.count(r"& $Z_\infty$"))
        self.assertEqual(1, analysis.count(r"& $Z_{\mathrm{fin}}$"))
        expected_analysis_rows = (
            r"D1 & $61.3\pm1.4$ & $15$ & $46$ & $0.33$ & $9.0$ & $47$ & $0.32$ & $9.2$ \\",
            r"D2 & $61.7\pm1.3$ & $14$ & $45$ & $0.29$ & $9.3$ & $46$ & $0.29$ & $9.5$ \\",
            r"D3 & $70.4\pm1.3$ & $17$ & $58$ & $0.30$ & $12$ & $58$ & $0.29$ & $12$ \\",
            r"D4 & $49.0\pm1.2$ & $12$ & $39$ & $0.31$ & $9.2$ & $39$ & $0.31$ & $9.4$ \\",
            r"D5 & $75.6\pm1.4$ & $19$ & $56$ & $0.34$ & $10$ & $56$ & $0.34$ & $10$ \\",
            r"D6 & $63.4\pm1.6$ & $18$ & $49$ & $0.37$ & $9.0$ & $49$ & $0.36$ & $9.2$ \\",
        )
        for row in expected_analysis_rows:
            self.assertIn(row, analysis)
        for index in range(1, 7):
            self.assertEqual(1, transmission.count(f"D{index} &"))
            self.assertEqual(1, analysis.count(f"D{index} &"))
        for rendered in (transmission, analysis):
            self.assertIn(r"\begin{tabular*}{\linewidth}", rendered)
            self.assertNotIn(r"\textwidth", rendered)
            self.assertNotIn(r"\caption", rendered)
            self.assertNotIn(r"\begin{table", rendered)


if __name__ == "__main__":
    unittest.main()
