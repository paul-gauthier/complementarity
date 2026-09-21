import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from analysis import goes
from analysis.artifact_config import load_config, resolve_repository_path
from analysis.conditions import _parse_target_dirname
from analysis.conditions_report import normalize_provenance
from analysis.conditions_sources import _stage_sources


class GoesTests(unittest.TestCase):
    def profile_sources(self, profile):
        return goes.GoesSourceFiles.from_mapping(
            {
                source_name: resolve_repository_path(
                    profile["sources"][source_name]
                )
                for source_name in ("goes_aod", "goes_cod", "goes_tpw")
            }
        )

    def profile_target(self, profile):
        return _parse_target_dirname(profile["target_timestamp"])

    def test_scan_parser_preserves_fractional_start_end_and_midpoint(self):
        path = Path(
            "pull-OR_ABI-L2-AODC-M6_G18_"
            "s20260642131175_e20260642133548_c20260642137109.nc"
        )
        scan = goes.parse_goes_scan(path, goes.AOD_PRODUCT)

        self.assertEqual(
            datetime(2026, 3, 5, 21, 31, 17, 500000, tzinfo=timezone.utc),
            scan.start_utc,
        )
        self.assertEqual(
            datetime(2026, 3, 5, 21, 33, 54, 800000, tzinfo=timezone.utc),
            scan.end_utc,
        )
        self.assertEqual(scan.start_utc + (scan.end_utc - scan.start_utc) / 2, scan.midpoint_utc)

    def test_scan_parser_rejects_invalid_metadata_product_and_satellite(self):
        cases = (
            (
                "missing.nc",
                goes.AOD_PRODUCT,
                "scan metadata",
            ),
            (
                "OR_ABI-L2-CODC-M6_G18_s20260642131175_e20260642133548_c20260642137109.nc",
                goes.AOD_PRODUCT,
                "expected ABI-L2-AODC",
            ),
            (
                "OR_ABI-L2-AODC-M6_G17_s20260642131175_e20260642133548_c20260642137109.nc",
                goes.AOD_PRODUCT,
                "expected GOES-18",
            ),
            (
                "OR_ABI-L2-AODC-M6_G18_s20263672131175_e20263672133548_c20263672137109.nc",
                goes.AOD_PRODUCT,
                "invalid GOES timestamp",
            ),
        )
        for filename, product, message in cases:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, message):
                    goes.parse_goes_scan(Path(filename), product)

    def test_source_files_are_scalar_paths(self):
        valid = {
            "goes_aod": "aod.nc",
            "goes_cod": "cod.nc",
            "goes_tpw": "tpw.nc",
        }
        sources = goes.GoesSourceFiles.from_mapping(valid)
        self.assertEqual(Path("aod.nc"), sources.aod)
        self.assertEqual(Path("cod.nc"), sources.cod)
        self.assertEqual(Path("tpw.nc"), sources.tpw)

    def test_staging_returns_exactly_one_path_per_goes_source(self):
        config = load_config()
        profile = next(
            profile
            for profile in config["conditions_profiles"].values()
            if profile["atmosphere_source"] == "goes"
        )

        with tempfile.TemporaryDirectory(prefix="goes-stage-test-") as name:
            source_by_name, staged_by_role = _stage_sources(profile, Path(name))
            for source_name in ("goes_aod", "goes_cod", "goes_tpw"):
                with self.subTest(source=source_name):
                    staged = staged_by_role[source_name]
                    self.assertIsInstance(staged, Path)
                    self.assertTrue(staged.is_symlink())
                    self.assertIn(staged.name, source_by_name)

    def test_screening_requires_product_specific_primary_flag(self):
        for spec in (
            goes.AOD_MEASUREMENT,
            goes.AE1_MEASUREMENT,
            goes.COD_MEASUREMENT,
            goes.TPW_MEASUREMENT,
        ):
            flag = spec.primary_quality_flag
            with self.subTest(kind=spec.key, condition="missing"):
                with self.assertRaisesRegex(ValueError, flag):
                    goes.screen_measurement(spec, 1.0, {})
            with self.subTest(kind=spec.key, condition="unreadable"):
                with self.assertRaisesRegex(ValueError, "unreadable"):
                    goes.screen_measurement(spec, 1.0, {flag: None})

    def test_quality_codes_zero_and_one_are_accepted(self):
        for spec in (
            goes.AOD_MEASUREMENT,
            goes.AE1_MEASUREMENT,
        ):
            flag = spec.primary_quality_flag
            for quality in (0, 1):
                with self.subTest(kind=spec.key, quality=quality):
                    measurement = goes.screen_measurement(
                        spec,
                        1.0,
                        {flag: quality},
                    )
                    self.assertTrue(measurement.usable)
                    self.assertEqual(1.0, measurement.value)

    def test_ae1_reads_its_declared_quality_flag(self):
        dataset = goes.xr.Dataset(
            {
                "AE1": (("y", "x"), [[0.4]]),
                "DQF": (("y", "x"), [[1]]),
                "AE_DQF": (("y", "x"), [[2]]),
            }
        )
        quality_flags = dict(goes.read_quality_flags(dataset, "AE1", 0, 0))

        self.assertEqual({"AE_DQF": 2}, quality_flags)
        measurement = goes.screen_measurement(
            goes.AE1_MEASUREMENT,
            0.4,
            quality_flags,
        )
        self.assertFalse(measurement.usable)
        self.assertIsNone(measurement.value)
        self.assertEqual(
            "rejected by AE_DQF=2: missing co-located AOD/AE2 context",
            measurement.reason,
        )

        for quality in (0, 1):
            with self.subTest(quality=quality):
                accepted = goes.screen_measurement(
                    goes.AE1_MEASUREMENT,
                    0.4,
                    {"AE_DQF": quality, "DQF": 3},
                )
                self.assertTrue(accepted.usable)
                self.assertEqual(0.4, accepted.value)

    def test_ae1_accepts_low_quality_only_for_the_low_aod_exception(self):
        def context(
            *,
            aod: float | None = 0.1,
            aod_dqf: int | None = 1,
            ae2: float | None = 0.5,
        ) -> goes.AerosolPixelContext:
            return goes.AerosolPixelContext(
                aod=aod,
                aod_dqf=aod_dqf,
                ae2=ae2,
            )

        for ae1 in (-1.0, 0.4, 3.0):
            with self.subTest(condition="accepted AE1 boundary", ae1=ae1):
                accepted = goes.screen_measurement(
                    goes.AE1_MEASUREMENT,
                    ae1,
                    {"AE_DQF": 2},
                    aerosol_context=context(),
                )
                self.assertTrue(accepted.usable)
                self.assertEqual(ae1, accepted.value)
                self.assertIn("low-AOD exception", accepted.qf_reason)

        for aod in (0.0, 0.199999):
            with self.subTest(condition="accepted AOD boundary", aod=aod):
                accepted = goes.screen_measurement(
                    goes.AE1_MEASUREMENT,
                    0.4,
                    {"AE_DQF": 2},
                    aerosol_context=context(aod=aod),
                )
                self.assertTrue(accepted.usable)

        for ae2 in (-1.0, 3.0):
            with self.subTest(condition="accepted AE2 boundary", ae2=ae2):
                accepted = goes.screen_measurement(
                    goes.AE1_MEASUREMENT,
                    0.4,
                    {"AE_DQF": 2},
                    aerosol_context=context(ae2=ae2),
                )
                self.assertTrue(accepted.usable)

        rejected_cases = (
            ("AOD DQF", 0.4, context(aod_dqf=2)),
            ("negative AOD", 0.4, context(aod=-0.001)),
            ("AOD threshold", 0.4, context(aod=0.2)),
            ("missing AOD", 0.4, context(aod=None)),
            ("low AE1", -1.001, context()),
            ("high AE1", 3.001, context()),
            ("missing AE2", 0.4, context(ae2=None)),
            ("low AE2", 0.4, context(ae2=-1.001)),
            ("high AE2", 0.4, context(ae2=3.001)),
        )
        for name, ae1, aerosol_context in rejected_cases:
            with self.subTest(condition=name):
                rejected = goes.screen_measurement(
                    goes.AE1_MEASUREMENT,
                    ae1,
                    {"AE_DQF": 2},
                    aerosol_context=aerosol_context,
                )
                self.assertFalse(rejected.usable)
                self.assertIsNone(rejected.value)
                self.assertIn("rejected", rejected.qf_reason)

        no_retrieval = goes.screen_measurement(
            goes.AE1_MEASUREMENT,
            0.4,
            {"AE_DQF": 3},
            aerosol_context=context(),
        )
        self.assertFalse(no_retrieval.usable)

    def test_tpw_requires_all_three_quality_flags_to_be_zero(self):
        good_flags = {
            "DQF_Overall": 0,
            "DQF_Retrieval": 0,
            "DQF_SkinTemp": 0,
        }
        accepted = goes.screen_measurement(
            goes.TPW_MEASUREMENT,
            5.0,
            good_flags,
        )
        self.assertTrue(accepted.usable)
        self.assertEqual(5.0, accepted.value)

        for name, rejected_value in (
            ("DQF_Overall", 1),
            ("DQF_Retrieval", 2),
            ("DQF_SkinTemp", 2),
        ):
            with self.subTest(flag=name):
                flags = {**good_flags, name: rejected_value}
                rejected = goes.screen_measurement(
                    goes.TPW_MEASUREMENT,
                    5.0,
                    flags,
                )
                self.assertFalse(rejected.usable)
                self.assertIsNone(rejected.value)
                self.assertIn(f"{name}={rejected_value}", rejected.qf_reason)

        for missing_name in ("DQF_Retrieval", "DQF_SkinTemp"):
            with self.subTest(missing=missing_name):
                flags = {**good_flags}
                del flags[missing_name]
                with self.assertRaisesRegex(ValueError, missing_name):
                    goes.screen_measurement(
                        goes.TPW_MEASUREMENT,
                        5.0,
                        flags,
                    )

    def test_unaccepted_quality_code_marks_measurement_unusable(self):
        aod = goes.screen_measurement(goes.AOD_MEASUREMENT, 1.0, {"DQF": 2})
        tpw = goes.screen_measurement(
            goes.TPW_MEASUREMENT,
            1.0,
            {
                "DQF_Overall": 2,
                "DQF_Retrieval": 0,
                "DQF_SkinTemp": 0,
            },
        )
        self.assertFalse(aod.usable)
        self.assertFalse(tpw.usable)
        self.assertIsNone(aod.value)
        self.assertIsNone(tpw.value)

    def test_positive_cod_retrieval_is_usable(self):
        for dqf in (0, 1, 22, 133, 150, 1023):
            with self.subTest(dqf=dqf, value="missing"):
                missing = goes.screen_measurement(
                    goes.COD_MEASUREMENT,
                    None,
                    {"DQF": dqf},
                )
                self.assertFalse(missing.usable)
                self.assertIsNone(missing.value)

            with self.subTest(dqf=dqf, value="zero"):
                zero = goes.screen_measurement(
                    goes.COD_MEASUREMENT,
                    0.0,
                    {"DQF": dqf},
                )
                self.assertFalse(zero.usable)
                self.assertIsNone(zero.value)

            with self.subTest(dqf=dqf, value="positive"):
                positive = goes.screen_measurement(
                    goes.COD_MEASUREMENT,
                    0.01,
                    {"DQF": dqf},
                )
                self.assertTrue(positive.usable)
                self.assertEqual(0.01, positive.value)

    def test_cod_screening_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "COD must be nonnegative"):
            goes.screen_measurement(
                goes.COD_MEASUREMENT,
                -0.01,
                {"DQF": 0},
                iy=4,
                ix=5,
            )

    def test_any_positive_cod_pixel_triggers_the_clear_sky_veto(self):
        def pixel_measurement(
            iy: int,
            value: float | None,
            dqf: int,
        ) -> goes.PixelMeasurement:
            return goes.PixelMeasurement(
                pixel=goes.GoesPixel(iy=iy, ix=0),
                screened=goes.screen_measurement(
                    goes.COD_MEASUREMENT,
                    value,
                    {"DQF": dqf},
                    iy=iy,
                    ix=0,
                ),
            )

        no_positive = (
            pixel_measurement(0, None, 1),
            pixel_measurement(1, 0.0, 22),
            pixel_measurement(2, 0.0, 150),
        )
        accepted = goes.PIXEL_REDUCER.reduce(
            goes.COD_MEASUREMENT,
            no_positive,
        )
        self.assertFalse(accepted.usable)
        self.assertIsNone(accepted.value)

        with_positive = no_positive + (pixel_measurement(3, 0.25, 150),)
        vetoed = goes.PIXEL_REDUCER.reduce(
            goes.COD_MEASUREMENT,
            with_positive,
        )
        self.assertTrue(vetoed.usable)
        self.assertEqual(0.25, vetoed.value)
        self.assertEqual(goes.GoesPixel(iy=3, ix=0), vetoed.selected.pixel)

    def test_open_uses_only_the_pinned_backend(self):
        dataset = object()
        with patch("analysis.goes.xr.open_dataset", return_value=dataset) as opened:
            self.assertIs(dataset, goes.open_local_dataset(Path("input.nc")))
        opened.assert_called_once_with(Path("input.nc"), engine="h5netcdf")

    def test_open_failure_propagates(self):
        with patch(
            "analysis.goes.xr.open_dataset",
            side_effect=OSError("invalid netCDF"),
        ) as opened:
            with self.assertRaisesRegex(OSError, "invalid netCDF"):
                goes.open_local_dataset(Path("missing.nc"))
        opened.assert_called_once_with(Path("missing.nc"), engine="h5netcdf")

    def test_variable_selection_requires_canonical_expected_field(self):
        missing = SimpleNamespace(data_vars={"Auxiliary": object(), "DQF": object()})
        lowercase = SimpleNamespace(data_vars={"aod": object(), "DQF": object()})
        canonical = SimpleNamespace(data_vars={"AOD": object(), "DQF": object()})

        with self.assertRaisesRegex(ValueError, "expected 'AOD'.*'Auxiliary'.*'DQF'"):
            goes.require_variable(missing, "AOD")
        with self.assertRaisesRegex(ValueError, "expected 'AOD'"):
            goes.require_variable(lowercase, "AOD")
        self.assertEqual("AOD", goes.require_variable(canonical, "AOD"))

    def test_circle_selection_matches_configured_grid_footprints(self):
        config = load_config()
        # Pixel footprints intersecting the 5 km circle around Henley Hall.
        aod_cod_pixels = tuple(
            goes.GoesPixel(iy, ix)
            for iy, columns in (
                (569, range(1991, 1995)),
                (570, range(1990, 1996)),
                (571, range(1990, 1996)),
                (572, range(1991, 1996)),
                (573, range(1992, 1995)),
            )
            for ix in columns
        )
        expected_pixels = {
            "goes_aod": aod_cod_pixels,
            "goes_cod": aod_cod_pixels,
            "goes_tpw": (
                goes.GoesPixel(113, 398),
                goes.GoesPixel(114, 398),
                goes.GoesPixel(114, 399),
            ),
        }
        for profile_id, profile in config["conditions_profiles"].items():
            if profile["atmosphere_source"] != "goes":
                continue
            for source_name, expected in expected_pixels.items():
                with self.subTest(profile=profile_id, source=source_name):
                    path = resolve_repository_path(profile["sources"][source_name])
                    with goes.open_local_dataset(path) as dataset:
                        selected = goes.PIXEL_SELECTOR.select(
                            dataset,
                            goes.HENLEY_HALL_SITE,
                        )
                    self.assertCountEqual(expected, selected)

    def test_spatial_reduction_uses_product_specific_quality_flags(self):
        config = load_config()
        profile = config["conditions_profiles"]["2026-03-05-13-29-15"]
        target = self.profile_target(profile)
        summary = goes.analyze_goes(self.profile_sources(profile))

        self.assertTrue(summary.aod.usable)
        self.assertIsNotNone(summary.aod.value)
        self.assertIsNotNone(summary.aod.selected)
        self.assertEqual(1, summary.aod.selected.screened.dqf)
        self.assertTrue(
            all(
                not item.screened.usable or item.screened.dqf in (0, 1)
                for item in summary.aod.inputs
            )
        )

        aod_values = [
            item.screened.value
            for item in summary.aod.inputs
            if item.screened.usable
        ]
        self.assertEqual(summary.aod.value, max(aod_values))

        self.assertTrue(summary.ae1.usable)
        self.assertEqual(-0.17387109994888306, summary.ae1.value)
        self.assertEqual(
            goes.GoesPixel(573, 1993),
            summary.ae1.selected.pixel,
        )
        self.assertEqual(
            {2, 3},
            {item.screened.dqf for item in summary.ae1.inputs},
        )
        self.assertEqual(
            3,
            sum(item.screened.usable for item in summary.ae1.inputs),
        )
        self.assertTrue(
            all(
                item.screened.dqf == 2
                for item in summary.ae1.inputs
                if item.screened.usable
            )
        )

        self.assertFalse(summary.cod.usable)
        self.assertIsNone(summary.cod.value)
        self.assertIsNone(summary.cod.selected)
        self.assertFalse(any(item.screened.usable for item in summary.cod.inputs))

        self.assertEqual(5.148791313171387, summary.tpw.value)
        self.assertEqual(goes.GoesPixel(113, 398), summary.tpw.selected.pixel)

        tpw_values = [
            item.screened.value
            for item in summary.tpw.inputs
            if item.screened.usable
        ]
        self.assertEqual(summary.tpw.value, max(tpw_values))

        self.assertTrue(summary.tpw.usable)
        self.assertEqual(0, summary.tpw.selected.screened.dqf)
        self.assertTrue(
            all(
                not item.screened.usable
                or item.screened.qf
                == {
                    "DQF_Overall": 0,
                    "DQF_Retrieval": 0,
                    "DQF_SkinTemp": 0,
                }
                for item in summary.tpw.inputs
            )
        )

        report, payload = goes.render_report_and_payload(
            summary,
            target_utc=target,
            pressure_hpa=1013.0,
            source_pull_prefix=profile["pull_prefixes"]["goes"],
        )
        self.assertEqual(
            "2026-03-05T21:31:17.500000+00:00",
            payload["aod"]["start_utc"],
        )
        self.assertIn("AOD file start (UTC): 2026-03-05 21:31:17.5 UTC", report)
        self.assertIn("AOD file start (PT): 2026-03-05 13:31:17.5 PST", report)
        exact_delta_report, _ = goes.render_report_and_payload(
            summary,
            target_utc=(
                summary.aod_record.scan.start_utc - timedelta(seconds=18.36)
            ),
            pressure_hpa=1013.0,
            source_pull_prefix=profile["pull_prefixes"]["goes"],
        )
        self.assertIn("AOD delta from target: +0.01 hours", exact_delta_report)
        self.assertEqual(
            {"method": "footprint-circle", "radius_m": 5_000.0},
            payload["spatial_selection"],
        )
        self.assertEqual(24, payload["aod"]["pixel_count"])
        self.assertEqual(3, payload["aod"]["usable_pixel_count"])
        self.assertEqual([0, 1], payload["aod"]["accepted_quality_values"])
        self.assertEqual([0, 1], payload["aod"]["alpha_accepted_quality_values"])
        self.assertEqual("AE_DQF", payload["aod"]["alpha_qf_primary"])
        self.assertTrue(payload["aod"]["alpha_usable"])
        self.assertEqual(2, payload["aod"]["alpha_dqf"])
        self.assertEqual(3, payload["aod"]["alpha_usable_pixel_count"])
        self.assertEqual({"2": 13, "3": 11}, payload["aod"]["alpha_quality_counts"])
        self.assertEqual(
            "ae-low-aod-exception",
            payload["aod"]["alpha_screening_rule"],
        )
        self.assertEqual(
            {
                "AOD": 0.04563146457076073,
                "AOD_DQF": 1,
                "AE2": -0.1193004846572876,
            },
            payload["aod"]["alpha_screening_context"],
        )
        self.assertIn("low-AOD exception", payload["aod"]["alpha_reason"])
        self.assertIsNotNone(payload["aod"]["selected_pixel"])
        self.assertEqual(3, payload["tpw"]["pixel_count"])
        self.assertEqual(1, payload["tpw"]["usable_pixel_count"])
        self.assertEqual([0], payload["tpw"]["accepted_quality_values"])
        self.assertEqual(0, payload["cod"]["usable_pixel_count"])
        self.assertIsNone(payload["cod"]["selected_pixel"])
        self.assertIsNone(payload["cod"]["dqf"])
        self.assertEqual([134], payload["cod"]["dqf_values"])
        self.assertTrue(
            all(isinstance(key, str) for key in payload["aod"]["quality_counts"])
        )
        self.assertEqual(payload, json.loads(json.dumps(payload)))

    def test_saved_profiles_use_the_ae_low_aod_exception(self):
        config = load_config()
        expected_ae1 = {
            "2026-03-05-13-29-15": -0.17387109994888306,
            "2026-03-06-15-06-58": -0.07828092575073242,
            "2026-03-08-06-52-49": -0.19138985872268677,
            "2026-03-08-12-21-56": -0.19138985872268677,
            "2026-03-08-15-45-35": -0.19138985872268677,
        }
        for profile_id, profile in config["conditions_profiles"].items():
            if profile["atmosphere_source"] != "goes":
                continue
            with self.subTest(profile=profile_id):
                summary = goes.analyze_goes(self.profile_sources(profile))
                self.assertTrue(summary.aod.usable)
                self.assertIsNotNone(summary.aod.value)
                self.assertIsNotNone(summary.aod.selected)
                self.assertEqual(1, summary.aod.selected.screened.dqf)

                self.assertTrue(summary.ae1.usable)
                self.assertEqual(expected_ae1[profile_id], summary.ae1.value)
                self.assertIsNotNone(summary.ae1.selected)
                self.assertEqual(2, summary.ae1.selected.screened.dqf)
                self.assertIn(
                    "low-AOD exception",
                    summary.ae1.selected.screened.qf_reason,
                )
                self.assertEqual(
                    {2, 3},
                    {item.screened.dqf for item in summary.ae1.inputs},
                )
                self.assertEqual(
                    3,
                    sum(item.screened.usable for item in summary.ae1.inputs),
                )

                self.assertTrue(summary.tpw.usable)
                self.assertIsNotNone(summary.tpw.selected)
                self.assertEqual(
                    {
                        "DQF_Overall": 0,
                        "DQF_Retrieval": 0,
                        "DQF_SkinTemp": 0,
                    },
                    summary.tpw.selected.screened.qf,
                )
                expected_tpw_count = (
                    2 if profile_id == "2026-03-08-06-52-49" else 1
                )
                self.assertEqual(
                    expected_tpw_count,
                    sum(item.screened.usable for item in summary.tpw.inputs),
                )

    def test_current_profiles_reproduce_reviewed_goes_payloads_and_reports(self):
        config = load_config()
        for dataset in config["datasets"]:
            profile = config["conditions_profiles"][dataset["conditions_profile"]]
            if profile["atmosphere_source"] != "goes":
                continue
            with self.subTest(dataset=dataset["display_id"]):
                reviewed_base = ROOT / "results/datasets" / dataset["id"] / "conditions"
                reviewed = json.loads((reviewed_base / "conditions.json").read_text())
                summary = goes.analyze_goes(self.profile_sources(profile))
                report, payload = goes.render_report_and_payload(
                    summary,
                    target_utc=self.profile_target(profile),
                    pressure_hpa=reviewed["metar"]["pressure_hpa"],
                    source_pull_prefix=profile["pull_prefixes"]["goes"],
                )
                source_by_name = {
                    Path(relative).name: relative
                    for relative in (
                        profile["sources"]["goes_aod"],
                        profile["sources"]["goes_cod"],
                        profile["sources"]["goes_tpw"],
                    )
                }
                self.assertEqual(
                    reviewed["goes"],
                    normalize_provenance(payload, source_by_name),
                )

                reviewed_text = (reviewed_base / "conditions.txt").read_text()
                reviewed_goes = (
                    "GOES ABI"
                    + reviewed_text.split("GOES ABI", 1)[1].split(
                        "\n\nAtmospheric transmission estimate", 1
                    )[0]
                    + "\n"
                )
                self.assertEqual(reviewed_goes, report)

    def test_analyze_goes_keeps_one_record_per_product(self):
        config = load_config()
        profile = config["conditions_profiles"]["2026-03-08-06-52-49"]
        sources = self.profile_sources(profile)
        summary = goes.analyze_goes(sources)

        self.assertEqual(sources.aod, summary.aod_record.scan.path)
        self.assertEqual(sources.cod, summary.cod_record.scan.path)
        self.assertEqual(sources.tpw, summary.tpw_record.scan.path)


if __name__ == "__main__":
    unittest.main()
