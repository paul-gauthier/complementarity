import json
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import load_config
from analysis.mzi_pooled_analysis import (
    DatasetQuadrature,
    PooledAnalysisError,
    build_phase_configurations,
    fit_pooled,
    fit_pooled_batch,
    infer_normalization,
    leave_one_out,
    load_dataset_quadratures,
    normalization_view,
    profile_dataset,
    profile_joint,
    profile_joint_batch,
    simulate_common_effect_gof,
    simulate_null_test,
    simulate_upper_limit_coverage,
    validate_common_reparameterization,
)


class PooledAnalysisTests(unittest.TestCase):
    def test_reviewed_robustness_quantities_use_unrounded_identities(self):
        pooled = json.loads(
            (
                ROOT
                / "results"
                / "analysis"
                / "mzi-pooled-analysis.json"
            ).read_text(encoding="utf-8")
        )
        summary = pooled["summary"]
        periods = [record["period_V"] for record in pooled["inputs"]]
        period_sigmas = [
            record["period_sigma_V"] for record in pooled["inputs"]
        ]
        self.assertEqual(summary["period_V_min"], min(periods))
        self.assertEqual(summary["period_V_max"], max(periods))
        self.assertEqual(
            summary["period_sigma_V_min"], min(period_sigmas)
        )
        self.assertEqual(
            summary["period_sigma_V_max"], max(period_sigmas)
        )
        self.assertEqual(
            (
                "q=-2*log(lambda)="
                "deviance_constrained-deviance_minimum"
            ),
            pooled["method"]["likelihood_ratio_statistic"],
        )
        f_critical = summary["critical_common_transmission_scale"]
        common_igm = summary["common_igm_transmission"]
        self.assertEqual(summary["eta_upper"], f_critical)
        self.assertAlmostEqual(
            summary["nominal_to_critical_scale_ratio"],
            1.0 / f_critical,
            places=13,
        )
        self.assertAlmostEqual(
            summary["additional_common_optical_depth"],
            -math.log(f_critical),
            places=13,
        )
        self.assertAlmostEqual(
            summary["critical_igm_transmission"],
            common_igm * f_critical,
            places=13,
        )
        self.assertAlmostEqual(
            summary["critical_igm_optical_depth"],
            -math.log(summary["critical_igm_transmission"]),
            places=13,
        )
        self.assertAlmostEqual(
            summary["nominal_igm_optical_depth"],
            -math.log(common_igm),
            places=13,
        )
        self.assertAlmostEqual(
            summary[
                "critical_to_nominal_igm_optical_depth_ratio"
            ],
            summary["critical_igm_optical_depth"]
            / summary["nominal_igm_optical_depth"],
            places=13,
        )
        for inference in pooled["normalizations"].values():
            self.assertEqual(
                summary["profile_cutoff"],
                inference["profile_cutoff"],
            )

    def test_loads_all_configured_disjoint_datasets(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        self.assertEqual(6, len(datasets))
        self.assertEqual(6, len({dataset.dataset_id for dataset in datasets}))
        self.assertTrue(all(dataset.normalization > 0.0 for dataset in datasets))
        for dataset in datasets:
            self.assertIsNotNone(dataset.transmission)
            self.assertIsNotNone(dataset.period_V)
            self.assertIsNotNone(dataset.period_sigma_V)
            self.assertGreater(dataset.period_V, 0.0)
            self.assertGreaterEqual(dataset.period_sigma_V, 0.0)
            self.assertIsNotNone(dataset.normalization_Rpm_cps)
            self.assertIsNotNone(
                dataset.normalization_Rfinite_path_cps
            )
            transmission = dataset.transmission
            self.assertAlmostEqual(
                dataset.normalization_Rfinite_path_cps,
                transmission.finite_path
                * dataset.normalization_Rpm_cps,
                places=10,
            )
            common_igm = (
                transmission.infinity
                / transmission.finite_path
            )
            self.assertAlmostEqual(
                dataset.normalization,
                common_igm
                * dataset.normalization_Rfinite_path_cps,
                places=10,
            )

    def test_physical_normalizations_share_one_reparameterized_profile(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        cutoff = 2.705543454095413
        infinity = infer_normalization(
            datasets, name="infinity", profile_cutoff=cutoff
        )
        finite_path = infer_normalization(
            normalization_view(
                datasets,
                name="finite_path",
                finite_path=True,
            ),
            name="finite_path",
            profile_cutoff=cutoff,
        )
        common_igm = (
            datasets[0].transmission.infinity
            / datasets[0].transmission.finite_path
        )
        validate_common_reparameterization(
            infinity, finite_path, common_igm
        )
        self.assertAlmostEqual(
            finite_path.fit.eta_hat,
            common_igm * infinity.fit.eta_hat,
            places=9,
        )
        self.assertAlmostEqual(
            finite_path.eta_upper,
            common_igm * infinity.eta_upper,
            places=9,
        )
        self.assertTrue(
            math.isfinite(finite_path.fit.full_restoration_q)
        )
        self.assertGreater(finite_path.fit.full_restoration_q, 0.0)
        self.assertNotAlmostEqual(
            infinity.fit.full_restoration_q,
            finite_path.fit.full_restoration_q,
            places=3,
        )

    def test_common_normalization_rescaling_is_exact_before_boundary(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        cutoff = 2.705543454095413
        nominal = infer_normalization(
            datasets, name="nominal", profile_cutoff=cutoff
        )
        scale = 0.5
        scaled_datasets = normalization_view(
            datasets, name="common_scale", multiplier=scale
        )
        scaled = infer_normalization(
            scaled_datasets, name="common_scale", profile_cutoff=cutoff
        )

        self.assertAlmostEqual(
            profile_joint(0.0, scaled_datasets).deviance,
            profile_joint(0.0, datasets).deviance,
            places=12,
        )
        self.assertAlmostEqual(
            scaled.fit.deviance_min,
            nominal.fit.deviance_min,
            places=12,
        )
        self.assertAlmostEqual(
            scaled.fit.eta_hat,
            nominal.fit.eta_hat / scale,
            places=8,
        )
        self.assertAlmostEqual(
            scaled.eta_upper,
            nominal.eta_upper / scale,
            places=9,
        )
        self.assertFalse(nominal.upper_limit_truncated)
        self.assertFalse(scaled.upper_limit_truncated)
        self.assertLess(scaled.eta_upper, 1.0)

    def test_launched_floor_is_explicitly_profiled_for_both_bases(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        cutoff = 2.705543454095413
        infinity = infer_normalization(
            datasets, name="infinity", profile_cutoff=cutoff
        )
        finite_path = infer_normalization(
            normalization_view(
                datasets,
                name="finite_path",
                finite_path=True,
            ),
            name="finite_path",
            profile_cutoff=cutoff,
        )
        launch_infinity = infer_normalization(
            normalization_view(
                datasets, name="launch_infinity", multiplier=2.0
            ),
            name="launch_infinity",
            profile_cutoff=cutoff,
        )
        launch_finite_path = infer_normalization(
            normalization_view(
                datasets,
                name="launch_finite_path",
                finite_path=True,
                multiplier=2.0,
            ),
            name="launch_finite_path",
            profile_cutoff=cutoff,
        )
        self.assertAlmostEqual(
            launch_infinity.eta_upper,
            infinity.eta_upper / 2.0,
            places=9,
        )
        self.assertAlmostEqual(
            launch_finite_path.eta_upper,
            finite_path.eta_upper / 2.0,
            places=9,
        )
        for inference in (launch_infinity, launch_finite_path):
            self.assertTrue(
                math.isfinite(inference.fit.full_restoration_q)
            )
            self.assertEqual(
                len(datasets),
                len(inference.fit.full_restoration_phases),
            )

    def test_rejects_non_positive_definite_covariance(self):
        with self.assertRaises(PooledAnalysisError):
            DatasetQuadrature(
                dataset_id="bad",
                q_hat=np.array([1.0, 2.0]),
                covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
                normalization=3.0,
            )

    def test_isotropic_profile_matches_radial_distance(self):
        dataset = DatasetQuadrature(
            dataset_id="isotropic",
            q_hat=np.array([3.0, 4.0]),
            covariance=np.eye(2) * 4.0,
            normalization=10.0,
        )
        deviance, phase = profile_dataset(0.2, dataset)
        self.assertAlmostEqual((5.0 - 2.0) ** 2 / 4.0, deviance, places=11)
        self.assertAlmostEqual(math.atan2(4.0, 3.0), phase, places=8)

    def test_isotropic_joint_fit_matches_weighted_solution(self):
        datasets = [
            DatasetQuadrature(
                dataset_id="one",
                q_hat=np.array([3.0, 4.0]),
                covariance=np.eye(2) * 4.0,
                normalization=10.0,
            ),
            DatasetQuadrature(
                dataset_id="two",
                q_hat=np.array([0.0, 3.0]),
                covariance=np.eye(2),
                normalization=20.0,
            ),
        ]
        expected = (10.0 * 5.0 / 4.0 + 20.0 * 3.0) / (
            10.0**2 / 4.0 + 20.0**2
        )
        fit = fit_pooled(datasets)
        self.assertAlmostEqual(expected, fit.eta_hat, places=9)
        expected_deviance = sum(
            (np.linalg.norm(dataset.q_hat) - expected * dataset.normalization) ** 2
            / dataset.covariance[0, 0]
            for dataset in datasets
        )
        self.assertAlmostEqual(expected_deviance, fit.deviance_min, places=9)

    def test_fit_is_invariant_under_common_quadrature_rotation(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        reference = fit_pooled(datasets)
        angle = 0.731
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        rotated = [
            DatasetQuadrature(
                dataset_id=dataset.dataset_id,
                q_hat=rotation @ dataset.q_hat,
                covariance=rotation @ dataset.covariance @ rotation.T,
                normalization=dataset.normalization,
            )
            for dataset in datasets
        ]
        result = fit_pooled(rotated)
        self.assertAlmostEqual(reference.eta_hat, result.eta_hat, places=10)
        self.assertAlmostEqual(
            reference.eta_upper_nominal, result.eta_upper_nominal, places=10
        )
        self.assertAlmostEqual(
            reference.full_restoration_q,
            result.full_restoration_q,
            places=8,
        )

    def test_leave_one_out_reports_every_dataset(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        records = leave_one_out(datasets)
        self.assertEqual(
            {dataset.dataset_id for dataset in datasets},
            {record["omitted_dataset_id"] for record in records},
        )
        self.assertTrue(
            all(0.0 < record["eta_upper_nominal"] < 1.0 for record in records)
        )

    def test_rejects_nonphysical_eta_max(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        with self.assertRaisesRegex(
            PooledAnalysisError,
            r"eta_max must equal the physical upper bound 1\.0",
        ):
            fit_pooled(datasets, eta_max=1.5)

    def test_batch_optimizer_matches_scalar_profiles(self):
        datasets = load_dataset_quadratures(load_config(), ROOT / "results")
        rng = np.random.default_rng(8317)
        observations = [
            rng.multivariate_normal(dataset.q_hat, dataset.covariance, size=8)
            for dataset in datasets
        ]
        eta_batch, deviance_batch, _ = fit_pooled_batch(datasets, observations)
        batch_at_fit = profile_joint_batch(eta_batch, datasets, observations)
        np.testing.assert_allclose(deviance_batch, batch_at_fit, atol=1e-10)
        for index in range(8):
            scalar_observations = [value[index] for value in observations]
            scalar = fit_pooled(datasets, observations=scalar_observations)
            self.assertAlmostEqual(
                scalar.eta_hat, eta_batch[index], places=8
            )
            self.assertAlmostEqual(
                scalar.deviance_min, deviance_batch[index], places=8
            )

    def test_coverage_simulation_is_deterministic(self):
        datasets = [
            DatasetQuadrature(
                dataset_id="one",
                q_hat=np.array([1.0, 0.0]),
                covariance=np.eye(2),
                normalization=10.0,
            ),
            DatasetQuadrature(
                dataset_id="two",
                q_hat=np.array([0.0, 1.0]),
                covariance=np.eye(2) * 1.5,
                normalization=12.0,
            ),
        ]
        fit = fit_pooled(datasets)
        configurations = build_phase_configurations(
            datasets, fit, random_count=0, seed=9
        )
        arguments = dict(
            confidence=0.95,
            nominal_cutoff=fit.nominal_cutoff,
            eta_grid=[0.0, 0.2],
            phase_configurations=configurations,
            simulations=100,
            seed=1729,
        )
        first = simulate_upper_limit_coverage(datasets, **arguments)
        second = simulate_upper_limit_coverage(datasets, **arguments)
        self.assertEqual(first, second)
        self.assertEqual(8, first["n_scenarios"])
        self.assertGreaterEqual(
            first["recommended_cutoff"], fit.nominal_cutoff
        )
        gof_first = simulate_common_effect_gof(
            datasets, fit, simulations=100, seed=2718
        )
        gof_second = simulate_common_effect_gof(
            datasets, fit, simulations=100, seed=2718
        )
        self.assertEqual(gof_first, gof_second)
        self.assertTrue(0.0 < gof_first["pvalue"] <= 1.0)
        null_first = simulate_null_test(
            datasets, fit, simulations=100, seed=31415
        )
        null_second = simulate_null_test(
            datasets, fit, simulations=100, seed=31415
        )
        self.assertEqual(null_first, null_second)
        self.assertTrue(0.0 < null_first["pvalue"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
