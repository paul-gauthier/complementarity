import copy
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import (
    ConfigError,
    canonical_outputs,
    load_config,
    dataset_runs,
    validate_raw_inputs,
)


class ArtifactConfigTests(unittest.TestCase):
    def test_canonical_config_and_inputs(self):
        config = load_config()
        validate_raw_inputs(config)
        self.assertEqual(10, config["schema_version"])
        self.assertEqual(6, len(config["datasets"]))
        self.assertEqual(
            [f"D{index}" for index in range(1, 7)],
            [dataset["display_id"] for dataset in config["datasets"]],
        )
        self.assertEqual(12, len(config["acquisitions"]))
        self.assertTrue(all(len(dataset_runs(config, dataset)) == 4 for dataset in config["datasets"]))
        outputs = canonical_outputs(config)
        self.assertEqual(143, len(outputs))
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertIn("README.md", outputs)
        self.assertIn("analysis/mzi-pooled-analysis.json", outputs)
        self.assertIn("analysis/igm-transmission.json", outputs)
        self.assertIn(
            "manuscript-values/pooled-analysis-values.tex", outputs
        )
        self.assertIn(
            "manuscript-values/igm-transmission-values.tex", outputs
        )
        self.assertIn("tables/dataset-transmission.tex", outputs)
        self.assertIn("tables/dataset-transmission.png", outputs)
        self.assertIn("tables/dataset-analysis.tex", outputs)
        self.assertIn("tables/dataset-analysis.png", outputs)
        for dataset in config["datasets"]:
            self.assertIn(f"datasets/{dataset['id']}/README.md", outputs)
            conditions = f"datasets/{dataset['id']}/conditions"
            self.assertIn(f"{conditions}/conditions.json", outputs)
            self.assertIn(f"{conditions}/conditions.txt", outputs)

    def test_config_requires_schema_version_10(self):
        for version in (None, "10", 11):
            with self.subTest(version=version):
                config = copy.deepcopy(load_config())
                if version is None:
                    config.pop("schema_version")
                else:
                    config["schema_version"] = version
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "artifact.json"
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ConfigError, r"schema_version must be 10"
                    ):
                        load_config(path)

    def test_paths_are_repository_relative(self):
        config = load_config()
        self.assertTrue(all(not Path(path).is_absolute() for path in config["acquisitions"].values()))
        for profile in config["conditions_profiles"].values():
            self.assertTrue(
                all(
                    not Path(value).is_absolute()
                    for value in profile["sources"].values()
                )
            )

    def test_display_ids_must_be_unique(self):
        config = copy.deepcopy(load_config())
        config["datasets"][1]["display_id"] = "D1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigError, r"duplicate dataset display_id"
            ):
                load_config(path)

    def test_dataset_acquisition_roles_match_the_experiment(self):
        config = load_config()
        self.assertEqual(
            {
                "D1": ("2026-03-05-13-29-15", "2026-03-05-14-20-45"),
                "D2": ("2026-03-06-15-06-58", "2026-03-06-16-03-22"),
                "D3": ("2026-03-06-21-27-03", "2026-03-06-22-25-50"),
                "D4": ("2026-03-08-06-52-49", "2026-03-08-07-59-00"),
                "D5": ("2026-03-08-12-21-56", "2026-03-08-13-28-33"),
                "D6": ("2026-03-08-15-45-35", "2026-03-08-14-44-08"),
            },
            {
                dataset["display_id"]: (
                    dataset["launch_run"], dataset["preserve_run"]
                )
                for dataset in config["datasets"]
            },
        )

    def test_every_dataset_has_its_own_launch_conditions_profile(self):
        config = load_config()
        profile_ids = [dataset["conditions_profile"] for dataset in config["datasets"]]
        self.assertEqual(len(profile_ids), len(set(profile_ids)))
        self.assertEqual(set(profile_ids), set(config["conditions_profiles"]))
        for dataset in config["datasets"]:
            profile = config["conditions_profiles"][dataset["conditions_profile"]]
            self.assertEqual(dataset["launch_run"], profile["target_timestamp"])

    def test_nighttime_dataset_uses_aeronet_lunar(self):
        config = load_config()
        dataset = next(
            dataset
            for dataset in config["datasets"]
            if dataset["id"] == "2026-03-06-21-27-03--2026-03-06-22-25-50"
        )
        profile = config["conditions_profiles"][dataset["conditions_profile"]]
        self.assertEqual("lunar", profile["aeronet_variant"])
        self.assertEqual("aeronet", profile["atmosphere_source"])
        self.assertNotIn("goes", profile["pull_prefixes"])
        self.assertFalse(any(name.startswith("goes_") for name in profile["sources"]))

    def test_conditions_sources_follow_selected_atmosphere(self):
        config = load_config()
        for profile_id, profile in config["conditions_profiles"].items():
            with self.subTest(profile=profile_id):
                expected_prefixes = {"metar", "aeronet", "irsa"}
                expected_sources = {
                    "metar", "aeronet", "irsa_json", "irsa_csv", "irsa_ecsv"
                }
                if profile["atmosphere_source"] == "goes":
                    expected_prefixes.add("goes")
                    expected_sources.update({"goes_aod", "goes_cod", "goes_tpw"})
                self.assertEqual(expected_prefixes, set(profile["pull_prefixes"]))
                self.assertEqual(expected_sources, set(profile["sources"]))
                for source_name in ("goes_aod", "goes_cod", "goes_tpw"):
                    if source_name in profile["sources"]:
                        self.assertIsInstance(profile["sources"][source_name], str)
                        self.assertTrue(profile["sources"][source_name])

    def test_conditions_source_keys_must_match_selected_atmosphere(self):
        config = copy.deepcopy(load_config())
        aeronet_profile = next(
            profile
            for profile in config["conditions_profiles"].values()
            if profile["atmosphere_source"] == "aeronet"
        )
        aeronet_profile["pull_prefixes"]["goes"] = "test-pull"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "pull_prefixes must contain exactly"):
                load_config(path)

        config = copy.deepcopy(load_config())
        goes_profile = next(
            profile
            for profile in config["conditions_profiles"].values()
            if profile["atmosphere_source"] == "goes"
        )
        goes_profile["sources"].pop("goes_tpw")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "sources must contain exactly"):
                load_config(path)

    def test_goes_source_paths_are_strictly_validated(self):
        def goes_config():
            config = copy.deepcopy(load_config())
            profile = next(
                profile
                for profile in config["conditions_profiles"].values()
                if profile["atmosphere_source"] == "goes"
            )
            return config, profile

        cases = (
            (
                "non-string",
                lambda profile: profile["sources"].__setitem__(
                    "goes_aod", 42
                ),
                "non-empty repository-relative path",
            ),
            (
                "empty",
                lambda profile: profile["sources"].__setitem__("goes_aod", ""),
                "non-empty repository-relative path",
            ),
            (
                "wrong product",
                lambda profile: profile["sources"].__setitem__(
                    "goes_aod", profile["sources"]["goes_cod"]
                ),
                "ABI-L2-AODC",
            ),
            (
                "unsafe path",
                lambda profile: profile["sources"].__setitem__(
                    "goes_aod", "../ABI-L2-AODC.nc"
                ),
                "safe repository-relative path",
            ),
        )
        for label, mutation, error in cases:
            with self.subTest(label=label):
                config, profile = goes_config()
                mutation(profile)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "artifact.json"
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, error):
                        load_config(path)

    def test_minimum_milky_way_transmission_drives_analysis(self):
        config = load_config()
        self.assertEqual(
            "minimum", config["transmission_analysis"]["milky_way_basis"]
        )

    def test_igm_model_is_flat_and_positive(self):
        config = copy.deepcopy(load_config())
        config["igm_transmission"]["omega_lambda"] = 0.7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"flat model"):
                load_config(path)

        config = copy.deepcopy(load_config())
        config["igm_transmission"]["galaxy_radius_hinv_pc"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"finite and positive"):
                load_config(path)

    def test_igm_references_are_complete_nonempty_strings(self):
        config = copy.deepcopy(load_config())
        config["igm_transmission"]["references"].pop("cosmology")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigError, r"references must contain exactly"
            ):
                load_config(path)

        config = copy.deepcopy(load_config())
        config["igm_transmission"]["references"]["cosmology"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"must be a non-empty string"):
                load_config(path)

    def test_pooled_analysis_calibration_is_explicit(self):
        pooled = load_config()["pooled_analysis"]
        self.assertEqual(0.95, pooled["confidence"])
        self.assertEqual(1.0, pooled["eta_max"])
        self.assertGreaterEqual(pooled["coverage_simulations"], 100)
        self.assertEqual(0.0, pooled["coverage_eta_grid"][0])
        self.assertEqual(
            pooled["eta_max"], pooled["coverage_eta_grid"][-1]
        )

    def test_pooled_analysis_rejects_nonphysical_eta_max(self):
        config = copy.deepcopy(load_config())
        config["pooled_analysis"]["eta_max"] = 1.5
        config["pooled_analysis"]["coverage_eta_grid"][-1] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigError, r"pooled_analysis\.eta_max must equal 1\.0"
            ):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
