import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config
from analysis.generate_igm_values import render_macros
from analysis.igm_transmission import (
    C_KM_S,
    _ccm_extinction_ratio,
    _relative_dust_opacity,
    build_record,
    calculate_igm_transmission,
)
from analysis.transmission_values import calculate_transmissions
from scripts.verify_results import json_equivalent


class IGMTransmissionTests(unittest.TestCase):
    def setUp(self):
        self.settings = copy.deepcopy(load_config()["igm_transmission"])
        self.result = calculate_igm_transmission(self.settings)

    def test_reviewed_component_budget(self):
        expected = {
            "density_kernel_integral": 0.3647676871,
            "dust_kernel_integral": 0.2272008017,
            "dust_visual_opacity_gpc_inverse": 0.0145426427,
            "dust_launch_to_visual_ratio": 0.5833090193,
            "dust_opacity_gpc_inverse": 0.0084828546,
            "dust_optical_depth": 0.0085726026,
            "electron_optical_depth": 0.0008325685,
            "galaxy_optical_depth": 0.0068709523,
            "nominal_optical_depth": 0.0162761233,
            "nominal_transmission": 0.9838556171,
            "gray_dust_optical_depth": 0.0137631927,
            "gray_total_optical_depth": 0.0214667134,
            "gray_transmission": 0.9787620565,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(value, getattr(self.result, name), places=10)

    def test_launch_normalization_and_direct_visual_integral(self):
        # Cover both CCM branches and more than the adopted R_V, so the launch
        # conversion cannot accidentally be hard-coded to the 810 nm example.
        for wavelength_um, r_v in ((0.810, 3.1), (0.550, 2.1), (1.25, 5.0)):
            with self.subTest(wavelength_um=wavelength_um, r_v=r_v):
                settings = copy.deepcopy(self.settings)
                settings["idler_wavelength_nm"] = 1000.0 * wavelength_um
                settings["dust_r_v"] = r_v
                result = calculate_igm_transmission(settings)
                visual_opacity = (
                    0.4 * math.log(10.0) * settings["dust_visual_extinction_mag"]
                    / settings["dust_normalization_distance_gpc"]
                )
                self.assertEqual(
                    1.0, _relative_dust_opacity(wavelength_um, wavelength_um, r_v)
                )
                self.assertAlmostEqual(
                    visual_opacity * _ccm_extinction_ratio(wavelength_um, r_v),
                    result.dust_opacity_gpc_inverse,
                    places=14,
                )
                direct_integral = quad(
                    lambda a: _ccm_extinction_ratio(wavelength_um * a, r_v)
                    / (a**4 * math.sqrt(
                        settings["omega_m"] * a**-3 + settings["omega_lambda"]
                    )),
                    1.0, math.inf, epsabs=1e-12, epsrel=1e-12,
                )[0]
                direct_tau = (
                    visual_opacity * C_KM_S
                    / settings["hubble_constant_km_s_mpc"] / 1000.0
                    * direct_integral
                )
                self.assertAlmostEqual(direct_tau, result.dust_optical_depth, places=13)

    def test_fixed_wavelength_check_reuses_launch_opacity_and_density(self):
        with patch("analysis.igm_transmission._relative_dust_opacity", return_value=1.0):
            fixed = calculate_igm_transmission(self.settings)
        self.assertEqual(self.result.dust_opacity_gpc_inverse, fixed.dust_opacity_gpc_inverse)
        self.assertAlmostEqual(
            fixed.dust_optical_depth, self.result.gray_dust_optical_depth, places=14
        )
        self.assertAlmostEqual(
            fixed.nominal_transmission, self.result.gray_transmission, places=14
        )
        self.assertEqual(self.result.electron_optical_depth, fixed.electron_optical_depth)
        self.assertEqual(self.result.galaxy_optical_depth, fixed.galaxy_optical_depth)

    def test_igm_change_does_not_change_finite_path(self):
        inputs = dict(
            atmosphere=0.9, aeronet_atmosphere=0.91,
            milky_way_point=0.89, milky_way_integrated=0.9, milky_way_minimum=0.88,
        )
        current = calculate_transmissions(**inputs)
        with patch("analysis.transmission_values.IGM_TRANSMISSION", 0.9778489942701414):
            previous = calculate_transmissions(**inputs)
        self.assertEqual(previous.finite_path, current.finite_path)
        self.assertGreater(current.infinity, previous.infinity)
        self.assertAlmostEqual(
            current.infinity / current.finite_path,
            self.result.nominal_transmission,
            places=14,
        )

    def test_craig_h_scaling_cancels(self):
        optical_depths = []
        for h0 in (50.0, 67.4, 80.0):
            settings = copy.deepcopy(self.settings)
            settings["hubble_constant_km_s_mpc"] = h0
            optical_depths.append(
                calculate_igm_transmission(settings).galaxy_optical_depth
            )
        self.assertTrue(
            all(
                math.isclose(value, optical_depths[0], rel_tol=1e-14, abs_tol=1e-14)
                for value in optical_depths[1:]
            )
        )

    def test_json_and_tex_interfaces(self):
        record = build_record(self.settings, self.result)
        self.assertEqual(1, record["schema_version"])
        self.assertEqual(self.settings["references"], record["references"])
        self.assertEqual(
            "visual_opacity_converted_to_launch_wavelength_with_ccm",
            record["model"]["dust_normalization"],
        )
        self.assertEqual(
            "same_launch_opacity_with_wavelength_factor_fixed_at_one",
            record["model"]["gray_dust"],
        )
        self.assertEqual(
            "craig_1996_opaque_fixed_radius_constant_comoving_population",
            record["model"]["galaxy"],
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "igm.json"
            source.write_text(json.dumps(record), encoding="utf-8")
            rendered = render_macros(source)
        self.assertIn(
            r"\newcommand{\IgmGalaxyOpticalDepth}{6.87\times10^{-3}}",
            rendered,
        )
        self.assertIn(
            r"\newcommand{\IgmDensityKernelIntegral}{0.365}", rendered
        )
        self.assertIn(r"\newcommand{\IgmDustKernelIntegral}{0.227}", rendered)
        self.assertIn(r"\newcommand{\IgmVisualDustOpacity}{1.45\times10^{-2}}", rendered)
        self.assertIn(r"\newcommand{\IgmLaunchToVisualExtinctionRatio}{0.583}", rendered)
        self.assertIn(r"\newcommand{\IgmDustOpacity}{8.48\times10^{-3}}", rendered)
        self.assertIn(r"\newcommand{\IgmNominalTransmission}{0.9839}", rendered)
        self.assertIn(r"\newcommand{\IgmGrayTransmission}{0.9788}", rendered)
        self.assertIn(
            r"\newcommand{\IgmWavelengthDependenceTransmissionDifferencePoints}{0.51}",
            rendered,
        )

    def test_json_invariant_allows_float_roundoff_but_not_real_changes(self):
        expected = {"nested": [1.0, {"value": 0.9778489943}], "label": "igm"}
        rounded = copy.deepcopy(expected)
        rounded["nested"][1]["value"] = math.nextafter(
            expected["nested"][1]["value"],
            math.inf,
        )
        changed = copy.deepcopy(expected)
        changed["nested"][1]["value"] += 1e-8

        self.assertTrue(json_equivalent(expected, rounded))
        self.assertFalse(json_equivalent(expected, changed))


if __name__ == "__main__":
    unittest.main()
