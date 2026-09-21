import unittest

from analysis.artifact_config import load_config
from analysis.generate_results_readmes import render_dataset_readme, render_global_readme


class ResultsReadmeTests(unittest.TestCase):
    def test_global_readme_displays_summary_and_links_every_dataset(self):
        config = load_config()
        rendered = render_global_readme(config)

        self.assertIn(
            "![Normalized quadrature confidence regions]"
            "(figures/mzi-normalized-quadratures.png)",
            rendered,
        )
        self.assertIn(
            "![Dataset transmissions](tables/dataset-transmission.png)",
            rendered,
        )
        self.assertIn(
            "![Dataset analysis results](tables/dataset-analysis.png)",
            rendered,
        )
        self.assertIn(
            "(analysis/mzi-pooled-analysis.json)",
            rendered,
        )
        self.assertIn(
            "(analysis/igm-transmission.json)",
            rendered,
        )
        for dataset in config["datasets"]:
            self.assertIn(f"(datasets/{dataset['id']}/)", rendered)

    def test_dataset_readme_displays_all_figures_and_tables(self):
        dataset = load_config()["datasets"][0]
        rendered = render_dataset_readme(dataset)

        self.assertEqual(2, rendered.count("](figures/"))
        self.assertEqual(5, rendered.count("](tables/"))
        self.assertIn(f"Launch run: `{dataset['launch_run']}`", rendered)
        self.assertIn(f"Preserve run: `{dataset['preserve_run']}`", rendered)


if __name__ == "__main__":
    unittest.main()
