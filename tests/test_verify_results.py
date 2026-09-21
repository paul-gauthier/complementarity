import copy
import json
import unittest
from pathlib import Path

from analysis.artifact_config import load_config
from scripts.verify_results import verify_dark_analysis

ROOT = Path(__file__).resolve().parents[1]


class DarkAnalysisVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {
            dataset["display_id"]: json.loads(
                (
                    ROOT / "results/datasets" / dataset["id"]
                    / "analysis/darks-accidentals.json"
                ).read_text(encoding="utf-8")
            )
            for dataset in load_config()["datasets"]
        }

    def test_every_configured_dark_analysis_satisfies_the_published_schema(self):
        for dataset, record in self.records.items():
            with self.subTest(dataset=dataset):
                self.assertEqual([], verify_dark_analysis(record))

    def test_record_fields_must_match_the_schema_exactly(self):
        template = next(iter(self.records.values()))
        for path in ((), ("coincidence_window",), ("runs", 0), ("runs", 0, "plan")):
            original = template
            for key in path:
                original = original[key]
            for field in (*original, "unexpected_field"):
                with self.subTest(path=path, field=field):
                    record = copy.deepcopy(template)
                    target = record
                    for key in path:
                        target = target[key]
                    if field in original:
                        target.pop(field)
                    else:
                        target[field] = 42
                    self.assertTrue(verify_dark_analysis(record))

    def test_record_containers_have_required_types(self):
        for invalid in (None, [], "invalid"):
            with self.subTest(root=invalid):
                self.assertEqual(
                    ["dark analysis must be an object"], verify_dark_analysis(invalid)
                )
        for path in (("coincidence_window",), ("runs",), ("runs", 0), ("runs", 0, "plan")):
            with self.subTest(path=path):
                record = copy.deepcopy(next(iter(self.records.values())))
                target = record
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = None
                self.assertTrue(verify_dark_analysis(record))

    def test_dark_analysis_requires_four_runs(self):
        for count in (0, 3, 5):
            with self.subTest(count=count):
                record = copy.deepcopy(next(iter(self.records.values())))
                record["runs"] = [record["runs"][0]] * count
                self.assertIn(
                    "dark analysis must contain four runs", verify_dark_analysis(record)
                )

    def test_schema_and_window_declarations_are_required(self):
        for path, value, error in (
            (("schema_version",), "5", "schema_version must be 5"),
            (("coincidence_window", "input", "field"), None, "noncanonical coincidence-window input"),
            (("coincidence_window", "validation", "required_for_every_run"), False, "validation is incomplete"),
            (("coincidence_window", "validation", "matched_canonical_width"), False, "validation is incomplete"),
        ):
            with self.subTest(path=path):
                record = copy.deepcopy(next(iter(self.records.values())))
                target = record
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertTrue(any(error in item for item in verify_dark_analysis(record)))


if __name__ == "__main__":
    unittest.main()
