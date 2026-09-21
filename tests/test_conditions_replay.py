import io
import json
import shutil
import socket
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config, resolve_repository_path
from analysis.conditions import (
    _aeronet_metrics_to_lr_inputs,
    _compute_aeronet_metrics,
    _goes_summary_to_lr_inputs,
    _load_saved_irsa_for_target,
    _parse_aeronet_csv,
    _parse_target_dirname,
    _required_metar_number,
    _render_aeronet_report,
    _render_irsa_report,
    _select_closest_metar,
    replay_archived,
)
from analysis.conditions_sources import network_disabled
from analysis.run_libradtran_transmission import (
    AtmosInputs,
    decide_clear_sky,
    parse_args as parse_libradtran_args,
)


IRSA_INTEGRATION_FIELDS = {
    "minutes", "cadence_minutes", "start_utc", "end_utc", "samples",
    "min_T", "avg_T", "max_T", "min_sample", "avg_sample", "max_sample",
}


class ConditionsReplayTests(unittest.TestCase):
    def irsa_fixture(self):
        profile = next(iter(load_config()["conditions_profiles"].values()))
        source = resolve_repository_path(profile["sources"]["irsa_json"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        return payload, profile

    def write_irsa_fixture(self, payload, profile, directory):
        prefix = profile["pull_prefixes"]["irsa"]
        source = resolve_repository_path(profile["sources"]["irsa_json"])
        for table in source.parent.glob(f"{prefix}-IRSA_zenith*.ecsv"):
            shutil.copyfile(table, directory / table.name)
        path = directory / source.name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def load_irsa_fixture(self, payload, profile):
        prefix = profile["pull_prefixes"]["irsa"]
        with tempfile.TemporaryDirectory(prefix="irsa-loader-test-") as name:
            directory = Path(name)
            path = self.write_irsa_fixture(payload, profile, directory)
            loaded = _load_saved_irsa_for_target(
                directory,
                prefix,
                _parse_target_dirname(profile["target_timestamp"]),
                profile["integration_minutes"],
                5,
            )
        return loaded, path

    def assert_irsa_rejected(self, mutation, expected_field, *, table_mutation=None):
        payload, profile = self.irsa_fixture()
        mutation(payload)
        prefix = profile["pull_prefixes"]["irsa"]
        with tempfile.TemporaryDirectory(prefix="irsa-loader-test-") as name:
            directory = Path(name)
            path = self.write_irsa_fixture(payload, profile, directory)
            if table_mutation is not None:
                table_mutation(directory, payload)
            with self.assertRaises(ValueError) as caught:
                _load_saved_irsa_for_target(
                    directory,
                    prefix,
                    _parse_target_dirname(profile["target_timestamp"]),
                    profile["integration_minutes"],
                    5,
                )
        message = str(caught.exception)
        self.assertIn(str(path), message)
        self.assertIn(expected_field, message)

    def test_all_configured_irsa_payloads_pass_strict_validation(self):
        config = load_config()
        for profile_id, profile in config["conditions_profiles"].items():
            with self.subTest(profile=profile_id):
                source = resolve_repository_path(profile["sources"]["irsa_json"])
                loaded = _load_saved_irsa_for_target(
                    source.parent,
                    profile["pull_prefixes"]["irsa"],
                    _parse_target_dirname(profile["target_timestamp"]),
                    profile["integration_minutes"],
                    5,
                )
                self.assertEqual(
                    loaded["mw_T"], loaded["integration"]["avg_T"]
                )
                self.assertEqual(
                    {"ra_deg", "dec_deg", "point_result", "integration", "mw_T", "source_path"},
                    set(loaded),
                )
                self.assertEqual(IRSA_INTEGRATION_FIELDS, set(loaded["integration"]))

    def test_irsa_report_requires_and_publishes_complete_results(self):
        payload, profile = self.irsa_fixture()
        loaded, _ = self.load_irsa_fixture(payload, profile)
        target_utc = _parse_target_dirname(profile["target_timestamp"])

        report, published = _render_irsa_report(
            target_utc=target_utc,
            ra_deg=loaded["ra_deg"],
            dec_deg=loaded["dec_deg"],
            result=loaded["point_result"],
            integration_summary=loaded["integration"],
        )

        self.assertIn(f"Samples: {len(loaded['integration']['samples'])}", report)
        for section in (
            "Sky coord:",
            "Matched band:",
            "Extinction column:",
            "Transmission:",
            "Integrated window:",
            "Window start (UTC):",
            "Window end (UTC):",
            "Transmission summary:",
            "Min T:",
            "Avg T:",
            "Max T:",
        ):
            self.assertIn(section, report)
        self.assertEqual(
            {
                "ra_deg", "dec_deg", "target_utc", "filter_name", "matched_index",
                "matched_wavelength_um", "extinction_column", "extinction_mag",
                "transmission_fraction", "transmission_percent", "table_file",
                "csv_file", "integration",
            },
            set(published),
        )
        self.assertEqual(IRSA_INTEGRATION_FIELDS, set(published["integration"]))
        self.assertEqual(
            len(loaded["integration"]["samples"]),
            len(published["integration"]["samples"]),
        )

    def test_irsa_loader_rejects_missing_current_schema_fields(self):
        cases = (
            (
                "integration start",
                lambda payload: payload["integration"].pop("start_utc"),
                "integration.start_utc",
            ),
            (
                "attempt count",
                lambda payload: payload["integration"].pop("n_attempted"),
                "integration.n_attempted",
            ),
            (
                "summary value",
                lambda payload: payload["integration"].pop("avg_T"),
                "integration.avg_T",
            ),
            (
                "summary sample",
                lambda payload: payload["integration"].pop("avg_sample"),
                "integration.avg_sample",
            ),
            (
                "sample timestamp",
                lambda payload: payload["integration"]["samples"][0].pop(
                    "time_utc"
                ),
                "integration.samples[0].time_utc",
            ),
            (
                "sample T",
                lambda payload: payload["integration"]["samples"][0].pop("T"),
                "integration.samples[0].T",
            ),
            (
                "sample transmission",
                lambda payload: payload["integration"]["samples"][0].pop(
                    "transmission_fraction"
                ),
                "integration.samples[0].transmission_fraction",
            ),
        )
        for label, mutation, expected_field in cases:
            with self.subTest(label=label):
                self.assert_irsa_rejected(mutation, expected_field)

    def test_irsa_loader_rejects_invalid_types_ranges_and_availability(self):
        cases = (
            (
                "top-level unavailable",
                lambda payload: payload.__setitem__("available", False),
                "available",
            ),
            (
                "unavailable",
                lambda payload: payload["integration"].__setitem__(
                    "available", False
                ),
                "integration.available",
            ),
            (
                "boolean count",
                lambda payload: payload["integration"].__setitem__(
                    "n_success", True
                ),
                "integration.n_success",
            ),
            (
                "non-object sample",
                lambda payload: payload["integration"]["samples"].__setitem__(
                    0, "invalid"
                ),
                "integration.samples[0]",
            ),
            (
                "non-finite T",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "T", float("nan")
                ),
                "integration.samples[0].T",
            ),
            (
                "invalid right ascension",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "ra_deg", 360.0
                ),
                "integration.samples[0].ra_deg",
            ),
            (
                "invalid transmission percent",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "transmission_percent", 101.0
                ),
                "integration.samples[0].transmission_percent",
            ),
            (
                "naive timestamp",
                lambda payload: payload["integration"].__setitem__(
                    "start_utc", "2026-03-05T21:29:15"
                ),
                "integration.start_utc",
            ),
        )
        for label, mutation, expected_field in cases:
            with self.subTest(label=label):
                self.assert_irsa_rejected(mutation, expected_field)

    def test_irsa_loader_rejects_internal_inconsistencies(self):
        cases = (
            (
                "counts",
                lambda payload: payload["integration"].__setitem__(
                    "n_success", payload["integration"]["n_success"] - 1
                ),
                "integration counts",
            ),
            (
                "window end",
                lambda payload: payload["integration"].__setitem__(
                    "end_utc", payload["integration"]["start_utc"]
                ),
                "integration.end_utc",
            ),
            (
                "sample cadence",
                lambda payload: payload["integration"]["samples"][1].__setitem__(
                    "time_utc", payload["integration"]["samples"][0]["time_utc"]
                ),
                "integration.samples[1].time_utc",
            ),
            (
                "sample transmission",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "T", payload["integration"]["samples"][0]["T"] - 0.01
                ),
                "integration.samples[0].T",
            ),
            (
                "percent",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "transmission_percent",
                    payload["integration"]["samples"][0]["transmission_percent"]
                    - 1.0,
                ),
                "integration.samples[0].transmission_percent",
            ),
            (
                "point metadata",
                lambda payload: payload["integration"]["samples"][0].__setitem__(
                    "extinction_mag",
                    payload["integration"]["samples"][0]["extinction_mag"]
                    + 0.01,
                ),
                "integration.samples[0]",
            ),
            (
                "summary value",
                lambda payload: payload["integration"].__setitem__(
                    "min_T", payload["integration"]["min_T"] + 0.01
                ),
                "integration.min_T",
            ),
            (
                "summary selection",
                lambda payload: payload["integration"].__setitem__(
                    "avg_sample", deepcopy(payload["integration"]["min_sample"])
                ),
                "integration.avg_sample",
            ),
        )
        for label, mutation, expected_field in cases:
            with self.subTest(label=label):
                self.assert_irsa_rejected(mutation, expected_field)

    def test_irsa_loader_reports_malformed_json_with_source_path(self):
        _, profile = self.irsa_fixture()
        prefix = profile["pull_prefixes"]["irsa"]
        with tempfile.TemporaryDirectory(prefix="irsa-loader-test-") as name:
            directory = Path(name)
            path = directory / f"{prefix}-IRSA_zenith.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                _load_saved_irsa_for_target(
                    directory,
                    prefix,
                    _parse_target_dirname(profile["target_timestamp"]),
                    profile["integration_minutes"],
                    5,
                )

        self.assertIn(str(path), str(caught.exception))

    def test_irsa_loader_rejects_results_inconsistent_with_archived_bands(self):
        for key, value in (
            ("filter_name", "CTIO I"),
            ("matched_index", 4),
            ("matched_wavelength_um", 0.8007),
            ("extinction_column", "A_SFD"),
            ("extinction_mag", 5.082),
        ):
            with self.subTest(field=key):
                def mutation(payload):
                    integration = payload["integration"]
                    results = [payload, *integration["samples"]]
                    results.extend(integration[name] for name in (
                        "min_sample", "avg_sample", "max_sample",
                    ))
                    for result in results:
                        result[key] = value

                self.assert_irsa_rejected(mutation, f"$.{key}")

    def test_irsa_loader_checks_extinction_to_transmission_relation(self):
        def mutation(payload):
            integration = payload["integration"]
            results = [payload, *integration["samples"]]
            results.extend(integration[name] for name in (
                "min_sample", "avg_sample", "max_sample",
            ))
            for result in results:
                result["transmission_fraction"] *= 0.5
                result["transmission_percent"] *= 0.5
                if "T" in result:
                    result["T"] *= 0.5
            for key in ("min_T", "avg_T", "max_T"):
                integration[key] *= 0.5

        self.assert_irsa_rejected(mutation, "$.transmission_fraction")

    def test_irsa_loader_requires_every_point_and_sample_table(self):
        payload, _ = self.irsa_fixture()
        table_files = [payload["table_file"]]
        table_files.extend(
            sample["table_file"] for sample in payload["integration"]["samples"]
        )
        for table_file in table_files:
            with self.subTest(table=table_file):
                self.assert_irsa_rejected(
                    lambda payload: None,
                    table_file,
                    table_mutation=lambda directory, payload: (directory / table_file).unlink(),
                )

    def test_irsa_loader_rejects_malformed_and_inconsistent_tables(self):
        for label, mutate_text in (
            ("empty", lambda text: ""),
            ("missing column", lambda text: text.replace(" A_SandF ", " ")),
            ("short row", lambda text: text.replace('"DSS-II i" 0.8111', '"DSS-II i"')),
            (
                "nonfinite wavelength",
                lambda text: text.replace('"DSS-II i" 0.8111', '"DSS-II i" nan'),
            ),
            (
                "invalid number",
                lambda text: text.replace('"DSS-II i" 0.8111', '"DSS-II i" invalid'),
            ),
            ("unclosed quote", lambda text: text + '\n"unclosed'),
            (
                "changed extinction",
                lambda text: text.replace(
                    '"DSS-II i" 0.8111 1.487 0.082', '"DSS-II i" 0.8111 1.487 0.182'
                ),
            ),
            (
                "nearest band",
                lambda text: text.replace('"CTIO I" 0.8007', '"CTIO I" 0.8100'),
            ),
        ):
            with self.subTest(case=label):
                def table_mutation(directory, payload):
                    table = directory / payload["integration"]["samples"][0]["table_file"]
                    table.write_text(
                        mutate_text(table.read_text(encoding="utf-8")), encoding="utf-8"
                    )

                self.assert_irsa_rejected(
                    lambda payload: None,
                    "integration.samples[0]",
                    table_mutation=table_mutation,
                )

    def test_irsa_loader_uses_validated_stored_summary_value(self):
        payload, profile = self.irsa_fixture()
        payload["integration"]["avg_T"] += 5e-13
        payload["extra_metadata"] = {"allowed": True}

        loaded, _ = self.load_irsa_fixture(payload, profile)

        self.assertEqual(loaded["mw_T"], payload["integration"]["avg_T"])

    def usable_goes_summary(self):
        return SimpleNamespace(
            satellite=18,
            aod=SimpleNamespace(value=0.05, usable=True),
            ae1=SimpleNamespace(value=0.4, usable=True),
            cod=SimpleNamespace(value=None, usable=False),
            tpw=SimpleNamespace(value=6.0, usable=True),
        )

    def test_clear_sky_requires_clear_metar_and_invalid_goes_cod(self):
        cases = (
            ("CLR", "invalid", True),
            ("SKC", "INVALID", True),
            ("BKN", "invalid", False),
            ("CLR", "valid", False),
            ("BKN", "valid", False),
        )

        for metar_sky, cod_status, expected in cases:
            with self.subTest(metar_sky=metar_sky, cod_status=cod_status):
                inputs = AtmosInputs(
                    angstrom_alpha=0.0,
                    pressure_hpa=1013.0,
                    aod_ref=0.0,
                    aod_ref_wavelength_um=0.55,
                    tpw_mm=0.0,
                    metar_temp_c=20.0,
                    metar_sky=metar_sky,
                    cod_status=cod_status,
                )
                if expected:
                    clear, _ = decide_clear_sky(inputs)
                    self.assertIs(clear, True)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "no quantitative cloud model is implemented"
                    ):
                        decide_clear_sky(inputs)

    def test_clear_sky_report_explains_the_cloud_optical_depth_assumption(self):
        inputs = AtmosInputs(
            angstrom_alpha=0.0,
            pressure_hpa=1013.0,
            aod_ref=0.0,
            aod_ref_wavelength_um=0.55,
            tpw_mm=0.0,
            metar_temp_c=20.0,
            metar_sky="CLR",
            cod_status="invalid",
            source_label="AERONET lunar",
        )

        _, notes = decide_clear_sky(inputs)

        self.assertIn(
            "No cloud optical depth was supplied; none was applied.", notes
        )

    def test_metar_selection_uses_observation_time(self):
        target = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        report_time_choice = {
            "id": "report-time-choice",
            "obsTime": target.timestamp() - 3600,
            "reportTime": target.isoformat(),
        }
        observation_time_choice = {
            "id": "observation-time-choice",
            "obsTime": target.timestamp() + 60,
            "reportTime": "2026-03-08T18:00:00+00:00",
        }

        selected = _select_closest_metar(
            [report_time_choice, observation_time_choice],
            target,
        )

        self.assertIs(selected, observation_time_choice)

    def test_metar_selection_skips_invalid_observation_times(self):
        target = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        valid = {"id": "valid", "obsTime": target.timestamp() + 60}
        metars = [
            {"id": "missing", "reportTime": target.isoformat()},
            {"id": "boolean", "obsTime": True},
            {"id": "nonfinite", "obsTime": float("nan")},
            valid,
        ]

        self.assertIs(_select_closest_metar(metars, target), valid)

    def test_metar_selection_rejects_all_invalid_observation_times(self):
        target = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        metars = [
            {"obsTime": "invalid", "reportTime": target.isoformat()},
            {"obsTime": float("inf")},
        ]

        with self.assertRaisesRegex(ValueError, "valid observation time"):
            _select_closest_metar(metars, target)

    def test_scientific_metar_fields_are_required(self):
        for field, description in (("altim", "pressure"), ("temp", "temperature")):
            for value in (None, True, float("nan")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, description):
                        _required_metar_number({field: value}, field, description)
        with self.assertRaisesRegex(ValueError, "outside"):
            _required_metar_number({"altim": 700.0}, "altim", "pressure")

    def test_atmos_inputs_require_measured_pressure_and_temperature(self):
        baseline = {
            "angstrom_alpha": 0.0,
            "pressure_hpa": 1013.0,
            "aod_ref": 0.0,
            "aod_ref_wavelength_um": 0.55,
            "tpw_mm": 0.0,
            "metar_sky": "CLR",
            "cod_status": "invalid",
            "metar_temp_c": 20.0,
        }
        for field, value in (
            ("pressure_hpa", float("nan")),
            ("metar_temp_c", None),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    AtmosInputs(**(baseline | {field: value}))

    def test_atmos_inputs_require_all_environmental_fields(self):
        baseline = {
            "angstrom_alpha": 0.0,
            "pressure_hpa": 1013.0,
            "aod_ref": 0.0,
            "aod_ref_wavelength_um": 0.55,
            "tpw_mm": 0.0,
            "metar_sky": "CLR",
            "cod_status": "invalid",
            "metar_temp_c": 20.0,
        }
        for field in (
            "aod_ref",
            "aod_ref_wavelength_um",
            "tpw_mm",
            "metar_sky",
            "cod_status",
        ):
            kwargs = dict(baseline)
            kwargs.pop(field)
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    AtmosInputs(**kwargs)

    def test_atmos_inputs_reject_nonphysical_environmental_values(self):
        baseline = {
            "angstrom_alpha": 0.0,
            "pressure_hpa": 1013.0,
            "aod_ref": 0.0,
            "aod_ref_wavelength_um": 0.55,
            "tpw_mm": 0.0,
            "metar_sky": "CLR",
            "cod_status": "invalid",
            "metar_temp_c": 20.0,
        }
        invalid = (
            ("angstrom_alpha", float("nan")),
            ("aod_ref", -0.01),
            ("aod_ref_wavelength_um", 0.0),
            ("tpw_mm", -0.01),
            ("metar_sky", " "),
            ("cod_status", "unknown"),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    AtmosInputs(**(baseline | {field: value}))

    def test_libradtran_cli_requires_environmental_inputs(self):
        required = (
            ("--pressure-hpa", "1013"),
            ("--angstrom-alpha", "0.4"),
            ("--metar-temp-c", "20"),
            ("--aod-ref", "0.05"),
            ("--aod-ref-wavelength-um", "0.55"),
            ("--tpw-mm", "6"),
            ("--metar-sky", "CLR"),
            ("--cod-status", "invalid"),
        )
        complete = [item for pair in required for item in pair]
        parsed = parse_libradtran_args(complete)
        self.assertEqual(0.05, parsed.aod_ref)
        for missing_flag, _ in required[-5:]:
            argv = []
            for flag, value in required:
                if flag != missing_flag:
                    argv.extend((flag, value))
            with self.subTest(missing=missing_flag):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_libradtran_args(argv)

    def test_network_is_rejected(self):
        with network_disabled():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("example.invalid", 443))

    def test_aeronet_replay_skips_goes_and_propagates_libradtran_failure(self):
        target = datetime(2026, 3, 5, 21, 29, 15, tzinfo=timezone.utc)
        metar = {
            "altim": 1013.0,
            "temp": 20.0,
            "dewp": 10.0,
            "cover": "CLR",
            "obsTime": target.timestamp(),
        }
        irsa = {
            "ra_deg": 120.0,
            "dec_deg": 30.0,
            "point_result": {},
            "integration": {},
            "mw_T": 0.9,
            "source_path": Path("irsa.json"),
        }
        aeronet_source = {
            "row": {},
            "source_path": Path("aeronet.csv"),
        }
        aeronet_metrics = {"tau870": 0.05, "alpha": 0.4, "pwv_mm": 6.0}

        with tempfile.TemporaryDirectory(prefix="conditions-replay-test-") as name:
            output = Path(name) / "output"
            with (
                patch(
                    "analysis.conditions._load_saved_metar_for_target",
                    return_value=(metar, Path("metar.json")),
                ),
                patch(
                    "analysis.conditions._load_saved_irsa_for_target",
                    return_value=irsa,
                ),
                patch(
                    "analysis.conditions._render_irsa_report",
                    return_value=("IRSA\n", {}),
                ),
                patch(
                    "analysis.conditions._load_saved_aeronet_variant",
                    return_value=aeronet_source,
                ),
                patch(
                    "analysis.conditions._compute_aeronet_metrics",
                    return_value=aeronet_metrics,
                ),
                patch(
                    "analysis.conditions._render_aeronet_report",
                    return_value="AERONET\n",
                ),
                patch("analysis.conditions.analyze_goes") as analyze_saved_goes,
                patch(
                    "analysis.conditions.lr_estimate",
                    side_effect=RuntimeError("uvspec failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "uvspec failed"):
                    replay_archived(
                        target_dir="2026-03-05-13-29-15",
                        input_dir=Path(name),
                        output_dir=output,
                        integration_minutes=49,
                        metar_pull_prefix="metar",
                        aeronet_variant="solar",
                        aeronet_pull_prefix="aeronet",
                        irsa_pull_prefix="irsa",
                        atmosphere_source="aeronet",
                        goes_pull_prefix=None,
                        goes_sources=None,
                    )

            analyze_saved_goes.assert_not_called()
            self.assertFalse((output / "conditions.json").exists())
            self.assertFalse((output / "conditions.txt").exists())

    def test_goes_inputs_require_usable_aod_exponent_and_tpw(self):
        cases = (
            ("AOD", "aod"),
            ("Angstrom exponent", "ae1"),
            ("TPW", "tpw"),
        )
        for description, attribute in cases:
            with self.subTest(description=description):
                summary = deepcopy(self.usable_goes_summary())
                getattr(summary, attribute).usable = False
                with self.assertRaisesRegex(ValueError, f"GOES {description}.*unusable"):
                    _goes_summary_to_lr_inputs(
                        summary,
                        pressure_hpa=1013.0,
                        metar_sky="CLR",
                        metar_temp_c=20.0,
                        metar_rh_pct=50.0,
                    )

    def test_measured_zero_goes_aod_and_tpw_remain_valid(self):
        summary = self.usable_goes_summary()
        summary.aod.value = 0.0
        summary.tpw.value = 0.0
        inputs = _goes_summary_to_lr_inputs(
            summary,
            pressure_hpa=1013.0,
            metar_sky="CLR",
            metar_temp_c=20.0,
            metar_rh_pct=50.0,
        )
        self.assertIsNotNone(inputs)
        self.assertEqual(inputs.aod_ref, 0.0)
        self.assertEqual(inputs.angstrom_alpha, 0.4)
        self.assertEqual(inputs.tpw_mm, 0.0)

    def test_usable_goes_inputs_must_be_finite_numeric(self):
        cases = (
            ("AOD", "aod", None),
            ("Angstrom exponent", "ae1", True),
            ("TPW", "tpw", float("nan")),
        )
        for description, attribute, value in cases:
            with self.subTest(description=description):
                summary = deepcopy(self.usable_goes_summary())
                getattr(summary, attribute).value = value
                with self.assertRaisesRegex(
                    ValueError,
                    rf"GOES {description} is marked usable but is not finite numeric",
                ):
                    _goes_summary_to_lr_inputs(
                        summary,
                        pressure_hpa=1013.0,
                        metar_sky="CLR",
                        metar_temp_c=20.0,
                        metar_rh_pct=50.0,
                    )

    def test_aeronet_requires_an_observed_exponent(self):
        with self.assertRaisesRegex(ValueError, "lack a finite Angstrom exponent"):
            _aeronet_metrics_to_lr_inputs(
                {"tau870": 0.05, "alpha": None, "pwv_mm": 6.0},
                pressure_hpa=1013.0,
                metar_sky="CLR",
                metar_temp_c=20.0,
                metar_rh_pct=50.0,
                source_label="AERONET solar",
            )

    def test_aeronet_data_quality_level_is_preserved_and_reported(self):
        payload = "\n".join(
            [
                "AERONET Data Download (Version 3 Direct Moon)",
                "Version 3: Lunar AOD Level 1.0",
                "The following data are unscreened and may not have final calibration applied.",
                (
                    "Date(dd:mm:yyyy),Time(hh:mm:ss),AOD_675nm,AOD_870nm,"
                    "440-870_Angstrom_Exponent,Precipitable_Water(cm),"
                    "Data_Quality_Level"
                ),
                "07:03:2026,06:25:46,0.066275,0.058586,0.520038,0.871904,lev10",
            ]
        )

        rows = _parse_aeronet_csv(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["_dt_utc"],
            datetime(2026, 3, 7, 6, 25, 46, tzinfo=timezone.utc),
        )
        self.assertEqual(rows[0]["_AOD_675"], 0.066275)
        self.assertEqual(rows[0]["_AOD_870"], 0.058586)
        self.assertEqual(rows[0]["_AE_440_870"], 0.520038)
        self.assertEqual(rows[0]["_PWV_cm"], 0.871904)
        self.assertEqual(rows[0]["_Data_Level"], "lev10")

        metrics = _compute_aeronet_metrics(rows[0])
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["data_level"], "lev10")

        report = _render_aeronet_report(
            "AERONET Lunar AOD",
            "UCSB",
            metrics,
            datetime(2026, 3, 7, 5, 27, 3, tzinfo=timezone.utc),
        )
        self.assertIn(
            "Data quality level: Level 1.0 "
            "(lev10; unscreened; final calibration may not be applied)",
            report,
        )

    def test_aeronet_parser_accepts_reordered_fields_and_extra_columns(self):
        payload = "\n".join(
            [
                "AERONET Data Download (Version 3 Direct Sun)",
                (
                    "Unused,Data_Quality_Level,AOD_870nm,Date(dd:mm:yyyy),"
                    "Precipitable_Water(cm),440-870_Angstrom_Exponent,"
                    "Time(hh:mm:ss),AOD_675nm"
                ),
                "ignored,lev15,0.058586,07:03:2026,0.871904,0.520038,06:25:46,0.066275",
            ]
        )

        rows = _parse_aeronet_csv(payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Unused"], "ignored")
        self.assertEqual(rows[0]["_Data_Level"], "lev15")

    def test_aeronet_parser_rejects_invalid_required_values(self):
        columns = [
            "Date(dd:mm:yyyy)",
            "Time(hh:mm:ss)",
            "AOD_675nm",
            "AOD_870nm",
            "440-870_Angstrom_Exponent",
            "Precipitable_Water(cm)",
            "Data_Quality_Level",
        ]
        valid_values = [
            "07:03:2026",
            "06:25:46",
            "0.066275",
            "0.058586",
            "0.520038",
            "0.871904",
            "lev10",
        ]
        invalid_values = (
            ("Date(dd:mm:yyyy)", ""),
            ("Time(hh:mm:ss)", ""),
            ("AOD_675nm", "N/A"),
            ("AOD_870nm", "0"),
            ("440-870_Angstrom_Exponent", "not-a-number"),
            ("Precipitable_Water(cm)", "nan"),
            ("Data_Quality_Level", ""),
        )

        for column, invalid_value in invalid_values:
            with self.subTest(column=column, invalid_value=invalid_value):
                values = list(valid_values)
                values[columns.index(column)] = invalid_value
                payload = "\n".join([",".join(columns), ",".join(values)])
                with self.assertRaises(ValueError) as caught:
                    _parse_aeronet_csv(payload)
                self.assertIn(column, str(caught.exception))

    def test_aeronet_parser_rejects_missing_required_columns(self):
        columns = [
            "Date(dd:mm:yyyy)",
            "Time(hh:mm:ss)",
            "AOD_675nm",
            "AOD_870nm",
            "440-870_Angstrom_Exponent",
            "Precipitable_Water(cm)",
            "Data_Quality_Level",
        ]
        values = [
            "07:03:2026",
            "06:25:46",
            "0.066275",
            "0.058586",
            "0.520038",
            "0.871904",
            "lev10",
        ]

        for missing_column in columns:
            with self.subTest(missing_column=missing_column):
                keep = [column != missing_column for column in columns]
                payload = "\n".join(
                    [
                        ",".join(
                            column for column, include in zip(columns, keep) if include
                        ),
                        ",".join(value for value, include in zip(values, keep) if include),
                    ]
                )
                with self.assertRaises(ValueError) as caught:
                    _parse_aeronet_csv(payload)
                self.assertIn(missing_column, str(caught.exception))

    def test_all_archived_aeronet_files_use_supported_columns(self):
        paths = sorted(ROOT.glob("data/raw/conditions/*/aeronet/*.csv"))
        self.assertEqual(len(paths), 6)

        for path in paths:
            with self.subTest(path=path):
                rows = _parse_aeronet_csv(path.read_text(encoding="utf-8"))
                self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
