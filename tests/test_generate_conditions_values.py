import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config
from analysis.generate_conditions_values import (
    ConditionsError,
    build_macros,
    integer,
    parse_args,
    transmissions,
    validate,
)


class GenerateConditionsValuesTests(unittest.TestCase):
    def test_cli_requires_explicit_input_and_output_paths(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args([])

        self.assertEqual(2, raised.exception.code)

    def load_conditions(self, dataset_id: str) -> dict:
        source = (
            ROOT
            / "results"
            / "datasets"
            / dataset_id
            / "conditions"
            / "conditions.json"
        )
        return json.loads(source.read_text(encoding="utf-8"))

    def test_configured_conditions_have_complete_normalized_records(self):
        section_fields = {
            "metar": {
                "pressure_hpa", "selected_report", "source", "source_file",
                "source_pull_prefix", "station",
            },
            "milky_way": {"T", "dec_deg", "ra_deg"},
            "irsa": {
                "csv_file", "dec_deg", "extinction_column", "extinction_mag",
                "filter_name", "integration", "matched_index", "matched_wavelength_um",
                "ra_deg", "source", "source_file", "source_pull_prefix", "table_file",
                "target_utc", "transmission_fraction", "transmission_percent",
            },
            "aeronet": {
                "angstrom_alpha", "angstrom_source", "data_level", "pwv_cm", "pwv_mm",
                "record_utc", "source", "source_file", "source_pull_prefix", "tau675",
                "tau870", "variant",
            },
        }
        integration_fields = {
            "avg_T", "avg_sample", "cadence_minutes", "end_utc", "max_T",
            "max_sample", "min_T", "min_sample", "minutes", "samples", "start_utc",
        }
        goes_fields = {
            "aod", "cod", "lat", "lon", "pressure_hpa", "satellite", "source",
            "source_pull_prefix", "spatial_selection", "target_utc", "tpw",
        }
        config = load_config()
        for dataset in config["datasets"]:
            with self.subTest(dataset=dataset["display_id"]):
                data = self.load_conditions(dataset["id"])
                profile = config["conditions_profiles"][dataset["conditions_profile"]]
                validate(data, atmosphere_source=profile["atmosphere_source"])
                expected_sections = set(section_fields) | {"target", "libradtran"}
                if profile["atmosphere_source"] == "goes":
                    expected_sections.add("goes")
                    self.assertEqual(goes_fields, set(data["goes"]))
                self.assertEqual(expected_sections, set(data))
                for section, fields in section_fields.items():
                    self.assertEqual(fields, set(data[section]), section)
                self.assertEqual(integration_fields, set(data["irsa"]["integration"]))
                self.assertEqual(profile["aeronet_variant"], data["aeronet"]["variant"])
                for record in data["libradtran"]:
                    self.assertEqual({"inputs", "result", "tag"}, set(record))

    def test_aeronet_variant_is_required_and_matches_selected_record(self):
        dataset_id = "2026-03-06-21-27-03--2026-03-06-22-25-50"
        data = self.load_conditions(dataset_id)
        data["aeronet"].pop("variant")
        with self.assertRaisesRegex(ConditionsError, "aeronet.variant"):
            validate(data, atmosphere_source="aeronet")

        data = self.load_conditions(dataset_id)
        data["aeronet"]["variant"] = "solar"
        with self.assertRaisesRegex(ConditionsError, "does not match"):
            validate(data, atmosphere_source="aeronet")

    def test_metar_observation_time_controls_exported_macros(self):
        dataset_id = "2026-03-08-12-21-56--2026-03-08-13-28-33"
        source = (
            ROOT
            / "results"
            / "datasets"
            / dataset_id
            / "conditions"
            / "conditions.json"
        )
        data = json.loads(source.read_text(encoding="utf-8"))

        macros = build_macros(data)

        self.assertEqual("18{:}53{:}00", macros["ConditionMetarObservationUTC"])
        self.assertEqual("29", macros["ConditionMetarObservationLeadMinutes"])

    def test_integer_rounding_renders_zero_without_a_sign(self):
        for value in (-0.49, -0.0, 0.0, 0.49):
            with self.subTest(value=value):
                self.assertEqual("0", integer(value))
        self.assertEqual("-1", integer(-0.5))
        self.assertEqual("1", integer(0.5))

    def test_aeronet_primary_source_controls_atmospheric_macro(self):
        dataset_id = "2026-03-06-21-27-03--2026-03-06-22-25-50"
        source = ROOT / "results" / "datasets" / dataset_id / "conditions" / "conditions.json"
        data = json.loads(source.read_text(encoding="utf-8"))

        macros = build_macros(data, atmosphere_source="aeronet")

        self.assertEqual("0.8982", macros["ConditionAtmosphericTransmission"])

    def test_selected_goes_requires_usable_aod_exponent_and_tpw(self):
        dataset_id = "2026-03-05-13-29-15--2026-03-05-14-20-45"
        for section, field in (
            ("aod", "usable"),
            ("aod", "alpha_usable"),
            ("tpw", "usable"),
        ):
            with self.subTest(section=section, field=field):
                data = self.load_conditions(dataset_id)
                data["goes"][section][field] = False
                with self.assertRaisesRegex(
                    ConditionsError,
                    rf"goes\.{section}\.{field} must be true",
                ):
                    transmissions(data, atmosphere_source="goes")

    def test_aeronet_selection_contains_no_goes_data(self):
        dataset_id = "2026-03-06-21-27-03--2026-03-06-22-25-50"
        data = self.load_conditions(dataset_id)

        self.assertNotIn("goes", data)
        self.assertEqual(
            ["AERONET-lunar"],
            [record["tag"] for record in data["libradtran"]],
        )
        validate(data, atmosphere_source="aeronet")
        result = transmissions(data, atmosphere_source="aeronet")
        macros = build_macros(data, atmosphere_source="aeronet")

        self.assertAlmostEqual(0.8982, result.atmosphere, places=4)
        self.assertFalse(any(name.startswith("ConditionGoes") for name in macros))

    def test_aeronet_selection_rejects_goes_data(self):
        dataset_id = "2026-03-06-21-27-03--2026-03-06-22-25-50"
        data = self.load_conditions(dataset_id)
        data["goes"] = {}

        with self.assertRaisesRegex(ConditionsError, "must not contain GOES"):
            validate(data, atmosphere_source="aeronet")

    def test_aeronet_selection_rejects_goes_record(self):
        dataset_id = "2026-03-06-21-27-03--2026-03-06-22-25-50"
        data = self.load_conditions(dataset_id)
        data["libradtran"].append({"tag": "GOES-18"})

        with self.assertRaisesRegex(
            ConditionsError,
            "expected libRadtran records",
        ):
            validate(data, atmosphere_source="aeronet")

    def test_band_macros_use_selected_atmospheric_record(self):
        cases = (
            (
                "goes", "2026-03-05-13-29-15--2026-03-05-14-20-45",
                "GOES-18", 782, 867,
            ),
            (
                "aeronet", "2026-03-06-21-27-03--2026-03-06-22-25-50",
                "AERONET-lunar", 781, 866,
            ),
        )
        for atmosphere_source, dataset_id, tag, minimum, maximum in cases:
            with self.subTest(atmosphere_source=atmosphere_source):
                data = self.load_conditions(dataset_id)
                record = next(
                    item for item in data["libradtran"] if item["tag"] == tag
                )
                record["result"]["lambda_min_nm"] = float(minimum)
                record["result"]["lambda_max_nm"] = float(maximum)

                macros = build_macros(data, atmosphere_source=atmosphere_source)

                self.assertEqual(str(minimum), macros["ConditionIdlerBandMinimum"])
                self.assertEqual(str(maximum), macros["ConditionIdlerBandMaximum"])
                self.assertEqual(f"{minimum}--{maximum}", macros["ConditionIdlerBand"])

    def test_milky_way_basis_is_selectable_and_defaults_to_minimum(self):
        dataset_id = "2026-03-05-13-29-15--2026-03-05-14-20-45"
        source = ROOT / "results" / "datasets" / dataset_id / "conditions" / "conditions.json"
        data = json.loads(source.read_text(encoding="utf-8"))

        minimum = transmissions(data, "goes", "minimum")
        point = transmissions(data, "goes", "point")
        integrated = transmissions(data, "goes", "integrated")

        self.assertEqual(minimum, transmissions(data, "goes"))
        self.assertAlmostEqual(0.8449, minimum.milky_way, places=4)
        self.assertAlmostEqual(0.8203, point.infinity, places=4)
        self.assertAlmostEqual(0.7974, integrated.infinity, places=4)
        self.assertAlmostEqual(0.7474, minimum.infinity, places=4)

    def test_goes_conditions_export_expected_macro_inventory(self):
        dataset_id = "2026-03-08-12-21-56--2026-03-08-13-28-33"
        source = (
            ROOT / "results" / "datasets" / dataset_id
            / "conditions" / "conditions.json"
        )
        data = json.loads(source.read_text(encoding="utf-8"))
        expected = {
            "ConditionLaunchDate",
            "ConditionLaunchUTC",
            "ConditionLaunchWindowMinutes",
            "ConditionMetarStation",
            "ConditionMetarObservationUTC",
            "ConditionMetarObservationLeadMinutes",
            "ConditionMetarPressure",
            "ConditionMetarTemperature",
            "ConditionMetarSky",
            "ConditionGoesSatellite",
            "ConditionGoesFileStartUTC",
            "ConditionGoesLeadSeconds",
            "ConditionGoesTPWDQF",
            "ConditionGoesAODDQF",
            "ConditionGoesCODDQF",
            "ConditionGoesTPW",
            "ConditionGoesAOD",
            "ConditionGoesAngstromExponent",
            "ConditionGoesAngstromCoefficient",
            "ConditionIdlerBandMinimum",
            "ConditionIdlerBandMaximum",
            "ConditionIdlerBand",
            "ConditionLaunchMirrorTransmission",
            "ConditionLaunchMirrorTransmissionPercent",
            "ConditionLaunchWindowSurfaceReflectance",
            "ConditionLaunchWindowTransmission",
            "ConditionLaunchOpticsTransmission",
            "ConditionAtmosphericTransmission",
            "ConditionAeronetRecordUTC",
            "ConditionAeronetLeadMinutes",
            "ConditionAeronetAtmosphericTransmission",
            "ConditionZenithRA",
            "ConditionZenithDec",
            "ConditionIrsaBand",
            "ConditionIrsaWavelength",
            "ConditionIrsaExtinctionColumn",
            "ConditionMilkyWayExtinction",
            "ConditionMilkyWayOpticalDepth",
            "ConditionMilkyWayTransmission",
        }
        self.assertEqual(expected, set(build_macros(data)))


if __name__ == "__main__":
    unittest.main()
