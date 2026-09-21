import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import dataset_runs, load_config
from analysis.mzi_io import (
    load_dark,
    load_plan,
    load_plan_length,
    load_points,
    plan_length,
    split_by_plan_length,
    validate_constant_point_duration,
    validate_mzi_run,
)


class MziIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        dataset = next(
            item for item in config["datasets"] if item["display_id"] == "D5"
        )
        runs = {item["id"]: item for item in dataset_runs(config, dataset)}
        cls.run_dir = ROOT / runs["run-1-erase"]["path"]
        cls.points = load_points(cls.run_dir)

    def test_load_and_plan_split(self):
        length = plan_length(load_plan(self.run_dir))
        passes = split_by_plan_length(self.points, length)
        self.assertEqual(9, len(passes))
        self.assertTrue(all(len(group) == length for group in passes))

    def test_duration_uses_canonical_field(self):
        self.assertEqual(10.0, validate_constant_point_duration(self.points))

    def test_malformed_duration(self):
        with self.assertRaises(ValueError):
            validate_constant_point_duration([{"duration_s": 1}, {"duration_s": 2}])

    def test_incomplete_pass_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact multiple"):
            split_by_plan_length(self.points[:-1], plan_length(load_plan(self.run_dir)))

    def test_run_loader_rejects_missing_or_invalid_canonical_fields(self):
        dark = {
            "Ns": 1,
            "Ni": 2,
            "Ni2": 3,
            "Nc": 4,
            "Nc2": 5,
            "T_s": 10.0,
            "w_s": 25e-9,
        }
        point = {
            "duration_s": 1.0,
            "target_delta_V": 0.1,
            "actual_delta_V": 0.1,
            "counts": {"N_s": 1, "N_i": 2, "N_i2": 3, "N_c": 4, "N_c2": 5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "dark.json").write_text(json.dumps(dark), encoding="utf-8")
            (run_dir / "plan.json").write_text(
                json.dumps({"deltas_exec_V": [0.1]}), encoding="utf-8"
            )
            (run_dir / "points.jsonl").write_text(
                json.dumps(point) + "\n", encoding="utf-8"
            )
            validate_mzi_run(run_dir)
            self.assertEqual(1, load_plan_length(run_dir))

            for field in dark:
                with self.subTest(dark_field=field):
                    malformed = dict(dark)
                    malformed.pop(field)
                    (run_dir / "dark.json").write_text(
                        json.dumps(malformed), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "missing required"):
                        load_dark(run_dir)
            for field, value in (("Ns", None), ("Nc2", -1), ("T_s", 0), ("w_s", 1e-9)):
                with self.subTest(dark_field=field, invalid_value=value):
                    malformed = dict(dark)
                    malformed[field] = value
                    (run_dir / "dark.json").write_text(
                        json.dumps(malformed), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        load_dark(run_dir)
            (run_dir / "dark.json").write_text(json.dumps(dark), encoding="utf-8")

            for field in ("duration_s", "target_delta_V", "actual_delta_V"):
                with self.subTest(point_field=field):
                    malformed_point = json.loads(json.dumps(point))
                    malformed_point.pop(field)
                    (run_dir / "points.jsonl").write_text(
                        json.dumps(malformed_point) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        load_points(run_dir)
            for path, value in (
                (("duration_s",), None),
                (("target_delta_V",), "invalid"),
                (("actual_delta_V",), float("inf")),
                (("counts", "N_s"), True),
                (("counts", "N_c2"), -1),
            ):
                with self.subTest(point_path=path, invalid_value=value):
                    malformed_point = json.loads(json.dumps(point))
                    target = malformed_point
                    for part in path[:-1]:
                        target = target[part]
                    target[path[-1]] = value
                    (run_dir / "points.jsonl").write_text(
                        json.dumps(malformed_point) + "\n", encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        load_points(run_dir)
            for field in point["counts"]:
                with self.subTest(count_field=field):
                    malformed_point = json.loads(json.dumps(point))
                    malformed_point["counts"].pop(field)
                    (run_dir / "points.jsonl").write_text(
                        json.dumps(malformed_point) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        load_points(run_dir)

            (run_dir / "plan.json").write_text(
                json.dumps({"deltas_V": [0.1]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "deltas_exec_V"):
                load_plan_length(run_dir)


if __name__ == "__main__":
    unittest.main()
