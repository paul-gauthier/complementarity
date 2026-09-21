import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.mzi_analysis import (
    _require_finite_fit_parameters,
    _require_successful_optimizer_result,
    fit_idler_sinusoid_poisson_joint,
    fit_weighted_sinusoid,
)
from analysis.mzi_joint import (
    fit_joint_coincidences,
    fit_joint_singles,
    fringe_height_and_sigma,
    validate_total_fringe_values,
)
from analysis.mzi_null_bounds import (
    ConditionBundle,
    _bootstrap_blocks,
    _build_actual_voltage_design,
    _ellipse_radius_upper,
    _get_rate_array,
    _get_rate_var_array,
    _joint_fit_condition,
    _make_paired_blocks,
    _parse_normalization_transmissions,
    _permutation_pvalue,
    _quadrature_null_pvalue,
    _solve_wls,
    main as mzi_null_bounds_main,
    parse_args as parse_null_bounds_args,
)


class OptimizerFailureTests(unittest.TestCase):
    @staticmethod
    def _scan():
        period = 2.0
        x = np.linspace(0.0, 3.0, 9)
        phase = 2.0 * np.pi * (x - 0.2) / period
        y1 = 100.0 * (1.0 + 0.3 * np.cos(phase))
        y2 = 80.0 * (1.0 - 0.3 * np.cos(phase))
        variance = np.full_like(x, 25.0)
        counts1 = np.rint(10.0 * (5.0 + y1)).astype(float)
        counts2 = np.rint(10.0 * (6.0 + y2)).astype(float)
        data_pass = SimpleNamespace(
            xs=x,
            rc1=y1,
            rc2=y2,
            var_rc1=variance,
            var_rc2=variance,
            n_i1=counts1,
            n_i2=counts2,
            ri1=counts1 / 10.0 - 5.0,
            ri2=counts2 / 10.0 - 6.0,
        )
        dark = SimpleNamespace(T=20.0, Ni=100.0, Ni2=120.0, r_i=5.0, r_i2=6.0)
        return period, x, y1, y2, variance, counts1, counts2, data_pass, dark

    @staticmethod
    def _failed_result(size):
        return SimpleNamespace(
            success=False,
            status=2,
            message="synthetic failure",
            x=np.zeros(size),
            fun=1.0,
        )

    def test_standalone_coincidence_optimizer_failure_raises(self):
        period, x, y1, y2, variance, *_ = self._scan()
        with patch(
            "analysis.mzi_analysis.minimize",
            return_value=self._failed_result(4),
        ):
            with self.assertRaisesRegex(RuntimeError, "weighted coincidence.*failed"):
                fit_weighted_sinusoid(
                    x,
                    y1,
                    y2,
                    variance,
                    variance,
                    P_fixed=period,
                )

    def test_standalone_singles_optimizer_failure_raises(self):
        period, x, _, _, _, counts1, counts2, _, dark = self._scan()
        with patch(
            "analysis.mzi_analysis.minimize",
            return_value=self._failed_result(6),
        ):
            with self.assertRaisesRegex(RuntimeError, "idler sinusoid.*failed"):
                fit_idler_sinusoid_poisson_joint(
                    x,
                    counts1,
                    counts2,
                    10.0,
                    dark.Ni,
                    dark.Ni2,
                    dark.T,
                    P_fixed=period,
                )

    def test_joint_coincidence_optimizer_failure_raises(self):
        period, *_, data_pass, _ = self._scan()
        with patch(
            "analysis.mzi_joint.minimize",
            return_value=self._failed_result(4),
        ):
            with self.assertRaisesRegex(RuntimeError, "joint coincidence.*failed"):
                fit_joint_coincidences([data_pass], P_fixed=period)

    def test_joint_singles_optimizer_failure_raises(self):
        period, *_, data_pass, dark = self._scan()
        with patch(
            "analysis.mzi_joint.minimize",
            return_value=self._failed_result(6),
        ):
            with self.assertRaisesRegex(RuntimeError, "joint singles.*failed"):
                fit_joint_singles([data_pass], [10.0], dark, P_fixed=period)

    def test_singular_coincidence_covariances_raise(self):
        period, x, y1, y2, variance, *_, data_pass, _ = self._scan()
        singular_x = np.zeros_like(x)
        successful_result = SimpleNamespace(
            success=True,
            status=0,
            message="synthetic success",
            x=np.zeros(4),
            fun=1.0,
        )

        with patch("analysis.mzi_analysis.minimize", return_value=successful_result):
            with self.assertRaisesRegex(ValueError, "covariance is not identifiable"):
                fit_weighted_sinusoid(
                    singular_x,
                    y1,
                    y2,
                    variance,
                    variance,
                    P_fixed=period,
                )

        singular_pass = SimpleNamespace(**vars(data_pass))
        singular_pass.xs = singular_x
        with patch("analysis.mzi_joint.minimize", return_value=successful_result):
            with self.assertRaisesRegex(ValueError, "covariance is not identifiable"):
                fit_joint_coincidences([singular_pass], P_fixed=period)

    def test_all_fitters_reject_malformed_covariance_or_uncertainty(self):
        period, x, y1, y2, variance, counts1, counts2, data_pass, dark = self._scan()

        malformed_standalone = np.eye(4)
        malformed_standalone[0, 0] = np.nan
        with patch(
            "analysis.mzi_analysis._covariance_from_weighted_jacobian",
            return_value=malformed_standalone,
        ):
            with self.assertRaisesRegex(ValueError, "variance must be finite"):
                fit_weighted_sinusoid(
                    x,
                    y1,
                    y2,
                    variance,
                    variance,
                    P_fixed=period,
                )

        malformed_singles = np.eye(6)
        malformed_singles[0, 0] = np.nan
        with patch(
            "analysis.mzi_analysis._poisson_singles_expected_fisher_covariance",
            return_value=malformed_singles,
        ):
            with self.assertRaisesRegex(ValueError, "variance must be finite"):
                fit_idler_sinusoid_poisson_joint(
                    x,
                    counts1,
                    counts2,
                    10.0,
                    dark.Ni,
                    dark.Ni2,
                    dark.T,
                    P_fixed=period,
                )

        malformed_joint = np.eye(4)
        malformed_joint[0, 0] = np.nan
        with patch(
            "analysis.mzi_joint._covariance_from_weighted_jacobian",
            return_value=malformed_joint,
        ):
            with self.assertRaisesRegex(ValueError, "variance must be finite"):
                fit_joint_coincidences([data_pass], P_fixed=period)

        with patch(
            "analysis.mzi_joint.np.linalg.eigh",
            return_value=(np.ones(6), np.full((6, 6), np.nan)),
        ):
            with self.assertRaisesRegex(ValueError, "covariance is not identifiable"):
                fit_joint_singles([data_pass], [10.0], dark, P_fixed=period)

    def test_report_helpers_reject_missing_uncertainties(self):
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_total_fringe_values(
                3.0,
                np.nan,
                1.0,
                2.0,
                context="test report",
            )

        fit = SimpleNamespace(
            A1=10.0,
            A2=8.0,
            V=0.2,
            Fringe1_sigma=np.nan,
            Fringe2_sigma=1.0,
        )
        with self.assertRaisesRegex(ValueError, "Fringe1_sigma"):
            fringe_height_and_sigma(fit, channel=1)

    def test_optimizer_result_rejects_wrong_shape_nonfinite_parameters_and_objective(self):
        cases = (
            (
                SimpleNamespace(success=True, x=np.zeros(2), fun=1.0),
                "expected 3 parameters",
            ),
            (
                SimpleNamespace(success=True, x=np.array([0.0, np.nan, 0.0]), fun=1.0),
                "non-finite parameters",
            ),
            (
                SimpleNamespace(success=True, x=np.zeros(3), fun=np.inf),
                "non-finite objective",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    _require_successful_optimizer_result(
                        result,
                        expected_size=3,
                        context="test fit",
                    )

    def test_all_four_optimizers_reject_malformed_success_results(self):
        period, x, y1, y2, variance, counts1, counts2, data_pass, dark = self._scan()
        fits = (
            (
                "analysis.mzi_analysis.minimize",
                4,
                lambda: fit_weighted_sinusoid(
                    x, y1, y2, variance, variance, P_fixed=period
                ),
            ),
            (
                "analysis.mzi_analysis.minimize",
                6,
                lambda: fit_idler_sinusoid_poisson_joint(
                    x,
                    counts1,
                    counts2,
                    10.0,
                    dark.Ni,
                    dark.Ni2,
                    dark.T,
                    P_fixed=period,
                ),
            ),
            (
                "analysis.mzi_joint.minimize",
                4,
                lambda: fit_joint_coincidences([data_pass], P_fixed=period),
            ),
            (
                "analysis.mzi_joint.minimize",
                6,
                lambda: fit_joint_singles(
                    [data_pass], [10.0], dark, P_fixed=period
                ),
            ),
        )
        for patch_target, expected_size, call_fit in fits:
            malformed_results = (
                (
                    "expected.*parameters",
                    SimpleNamespace(
                        success=True,
                        status=0,
                        message="reported success",
                        x=np.zeros(expected_size - 1),
                        fun=1.0,
                    ),
                ),
                (
                    "non-finite parameters",
                    SimpleNamespace(
                        success=True,
                        status=0,
                        message="reported success",
                        x=np.full(expected_size, np.nan),
                        fun=1.0,
                    ),
                ),
                (
                    "non-finite objective",
                    SimpleNamespace(
                        success=True,
                        status=0,
                        message="reported success",
                        x=np.zeros(expected_size),
                        fun=np.inf,
                    ),
                ),
            )
            for message, result in malformed_results:
                with self.subTest(
                    optimizer=patch_target,
                    expected_size=expected_size,
                    failure=message,
                ):
                    with patch(patch_target, return_value=result):
                        with self.assertRaisesRegex(RuntimeError, message):
                            call_fit()

    def test_optimizer_result_requires_scalar_objective_and_reports_status(self):
        result = SimpleNamespace(
            success=True,
            status=7,
            message="synthetic status",
            x=np.zeros(3),
            fun=np.array([1.0]),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"non-scalar objective.*status 7.*synthetic status",
        ):
            _require_successful_optimizer_result(
                result,
                expected_size=3,
                context="test fit",
            )

    def test_transformed_optimizer_parameters_must_be_finite_and_physical(self):
        with self.assertRaisesRegex(RuntimeError, "A1, V"):
            _require_finite_fit_parameters(
                {"A1": np.inf, "V": 1.0},
                positive=("A1",),
                unit_interval=("V",),
                context="test fit",
            )

    def test_fit_array_mismatches_raise(self):
        period, x, y1, y2, variance, *_, data_pass, dark = self._scan()
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            fit_weighted_sinusoid(
                x,
                y1[:-1],
                y2,
                variance,
                variance,
                P_fixed=period,
            )
        with self.assertRaisesRegex(ValueError, "passes and durations"):
            fit_joint_singles([data_pass], [], dark, P_fixed=period)

    def test_malformed_period_initialization_and_dark_inputs_raise(self):
        period, x, y1, y2, variance, counts1, counts2, data_pass, dark = self._scan()
        with self.assertRaisesRegex(ValueError, "fixed period"):
            fit_weighted_sinusoid(
                x,
                y1,
                y2,
                variance,
                variance,
                P_fixed=np.nan,
            )
        with self.assertRaisesRegex(ValueError, "fixed period"):
            fit_idler_sinusoid_poisson_joint(
                x,
                counts1,
                counts2,
                10.0,
                dark.Ni,
                dark.Ni2,
                dark.T,
                P_fixed=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "initial A1"):
            fit_joint_coincidences(
                [data_pass],
                P_fixed=period,
                init={"A1": np.nan},
            )
        invalid_dark = SimpleNamespace(
            T=0.0,
            Ni=dark.Ni,
            Ni2=dark.Ni2,
            r_i=dark.r_i,
            r_i2=dark.r_i2,
        )
        with self.assertRaisesRegex(ValueError, "dark duration"):
            fit_joint_singles(
                [data_pass],
                [10.0],
                invalid_dark,
                P_fixed=period,
            )

    def test_null_bound_rates_and_variances_use_canonical_fields(self):
        _, _, _, _, _, counts1, counts2, data_pass, dark = self._scan()

        np.testing.assert_allclose(_get_rate_array(data_pass, 1), data_pass.ri1)
        np.testing.assert_allclose(_get_rate_array(data_pass, 2), data_pass.ri2)
        np.testing.assert_allclose(
            _get_rate_var_array(data_pass, 10.0, dark, 1),
            counts1 / 10.0**2 + dark.Ni / dark.T**2,
        )
        np.testing.assert_allclose(
            _get_rate_var_array(data_pass, 10.0, dark, 2),
            counts2 / 10.0**2 + dark.Ni2 / dark.T**2,
        )

    def test_null_bound_rates_require_scan_and_dark_measurements(self):
        missing_measurements = SimpleNamespace()
        with self.assertRaisesRegex(AttributeError, "canonical field ri1"):
            _get_rate_array(missing_measurements, 1)
        with self.assertRaisesRegex(AttributeError, "canonical field n_i1"):
            _get_rate_var_array(missing_measurements, 1.0, SimpleNamespace(Ni=1, T=1.0), 1)

        canonical = SimpleNamespace(n_i1=np.ones(3))
        with self.assertRaisesRegex(AttributeError, "canonical fields Ni and T"):
            _get_rate_var_array(canonical, 1.0, SimpleNamespace(T=1.0), 1)

    def test_null_bound_variances_require_positive_derived_variances(self):
        canonical = SimpleNamespace(n_i1=np.ones(3))
        with self.assertRaisesRegex(ValueError, "dark duration"):
            _get_rate_var_array(
                canonical,
                1.0,
                SimpleNamespace(Ni=1, T=0.0),
                1,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _get_rate_var_array(
                SimpleNamespace(n_i1=np.zeros(3)),
                1.0,
                SimpleNamespace(Ni=0, T=1.0),
                1,
            )

    def test_paired_builder_rejects_unequal_canonical_arrays(self):
        _, _, _, _, _, _, _, data_pass, dark = self._scan()
        data_pass.ri1 = data_pass.ri1[:-1]
        bundle = ConditionBundle(
            name="condition",
            path=None,
            dark=dark,
            records=[],
            pass_records=[],
            pass_indices=[0],
            data_passes=[data_pass],
            T_list=[10.0],
        )
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            _make_paired_blocks(
                bundle,
                bundle,
                phase_x0_list=[0.0],
                phase_pass_indices=[0],
                period=2.0,
            )


class NullBoundsFailureTests(unittest.TestCase):
    def test_null_bound_cli_requires_transmissions_and_output(self):
        required = (
            ("--condition", "Launch:/tmp/launch"),
            ("--Tinf", "0.75"),
            ("--normalization-T", "0.75,0.70"),
            ("--output", "/tmp/mzi-null-bounds.json"),
        )
        complete = [item for pair in required for item in pair]
        parsed = parse_null_bounds_args(complete)
        self.assertEqual(0.75, parsed.Tinf)
        for missing_flag, _ in required[1:]:
            argv = []
            for flag, value in required:
                if flag != missing_flag:
                    argv.extend((flag, value))
            with self.subTest(missing=missing_flag):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_null_bounds_args(argv)

    def test_null_bound_cli_rejects_invalid_tinf_before_loading_inputs(self):
        for value in ("0", "-0.1", "nan"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Tinf.*finite and positive"):
                    mzi_null_bounds_main(
                        [
                            "--condition",
                            "Launch:/does/not/exist",
                            "--Tinf",
                            value,
                            "--normalization-T",
                            "0.75",
                            "--output",
                            "/tmp/mzi-null-bounds.json",
                        ]
                    )

    def test_null_bound_cli_rejects_invalid_rpm_before_loading_inputs(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "rpm.*finite and positive"):
                    mzi_null_bounds_main(
                        [
                            "--condition",
                            "Launch:/does/not/exist",
                            "--Tinf",
                            "0.75",
                            f"--rpm={value}",
                            "--normalization-T",
                            "0.75",
                            "--output",
                            "/tmp/mzi-null-bounds.json",
                        ]
                    )

    def test_normalization_transmissions_are_strict(self):
        np.testing.assert_allclose(
            _parse_normalization_transmissions("0.75,0.70"),
            [0.75, 0.70],
        )
        for value in ("", "0.75,,0.70", "not-a-number", "nan", "0"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parse_normalization_transmissions(value)

    def test_wls_rejects_nonfinite_rows_and_rank_deficiency(self):
        X = np.column_stack((np.ones(5), np.arange(5, dtype=float)))
        with self.assertRaisesRegex(ValueError, "observations must be finite"):
            _solve_wls(
                X,
                np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
                np.ones(5),
                covariance_scale="none",
            )

        deficient = np.column_stack((np.ones(5), np.ones(5)))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            _solve_wls(
                deficient,
                np.arange(5, dtype=float),
                np.ones(5),
                covariance_scale="none",
            )

    def test_wls_requires_positive_residual_degrees_of_freedom(self):
        with self.assertRaisesRegex(ValueError, "positive residual degrees"):
            _solve_wls(
                np.eye(2),
                np.array([1.0, 2.0]),
                np.ones(2),
                covariance_scale="none",
            )

    def test_wls_valid_fit_is_full_rank_with_identifiable_covariance(self):
        x = np.arange(6, dtype=float)
        fit = _solve_wls(
            np.column_stack((np.ones_like(x), x)),
            1.0 + 2.0 * x,
            np.ones_like(x),
            covariance_scale="none",
        )
        self.assertEqual(2, fit.rank)
        self.assertTrue(np.all(np.linalg.eigvalsh(fit.cov_beta) > 0.0))

    def test_covariance_consumers_reject_invalid_covariance(self):
        cases = (
            ("singular", np.array([[1.0, 1.0], [1.0, 1.0]])),
            ("indefinite", np.array([[1.0, 2.0], [2.0, 1.0]])),
            ("nonfinite", np.array([[1.0, 0.0], [0.0, np.nan]])),
            ("asymmetric", np.array([[1.0, 0.5], [0.0, 1.0]])),
        )
        for name, covariance in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _ellipse_radius_upper(np.zeros(2), covariance, 0.95)
                with self.assertRaises(ValueError):
                    _quadrature_null_pvalue(np.zeros(2), covariance)

    def test_block_builder_rejects_mismatched_arrays(self):
        block = {
            "theta_launch": np.array([0.0, 1.0]),
            "theta_control": np.array([0.0]),
            "theta_known_launch": np.array([0.0, 1.0]),
            "z_launch": np.array([1.0, 2.0]),
            "z_control": np.array([1.0, 2.0]),
            "var_launch": np.ones(2),
            "var_control": np.ones(2),
        }
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            _build_actual_voltage_design([block])

        block["theta_control"] = np.array([0.0, 1.0])
        block["z_launch"] = np.array([1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "arrays must be finite"):
            _build_actual_voltage_design([block])

    def test_paired_builder_rejects_phase_and_bundle_mismatches(self):
        bundle = ConditionBundle(
            name="condition",
            path=None,
            dark=None,
            records=[],
            pass_records=[],
            pass_indices=[0],
            data_passes=[object()],
            T_list=[],
        )
        with self.assertRaisesRegex(ValueError, "phase-reference.*inconsistent lengths"):
            _make_paired_blocks(
                bundle,
                bundle,
                phase_x0_list=[],
                phase_pass_indices=[0],
                period=2.0,
            )
        with self.assertRaisesRegex(ValueError, "data passes.*inconsistent lengths"):
            _make_paired_blocks(
                bundle,
                bundle,
                phase_x0_list=[0.0],
                phase_pass_indices=[0],
                period=2.0,
            )

        bundle.T_list = [np.nan]
        with self.assertRaisesRegex(ValueError, "durations must be finite"):
            _make_paired_blocks(
                bundle,
                bundle,
                phase_x0_list=[0.0],
                phase_pass_indices=[0],
                period=2.0,
            )

        bundle.T_list = [1.0]
        with self.assertRaisesRegex(ValueError, "phase values must be finite"):
            _make_paired_blocks(
                bundle,
                bundle,
                phase_x0_list=[np.nan],
                phase_pass_indices=[0],
                period=2.0,
            )

    def test_seed_fit_failure_propagates(self):
        bundle = ConditionBundle(
            name="condition",
            path=None,
            dark=object(),
            records=[],
            pass_records=[],
            pass_indices=[0],
            data_passes=[object()],
            T_list=[1.0],
        )
        with patch(
            "analysis.mzi_null_bounds.fit_scan",
            side_effect=RuntimeError("seed failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "seed failure"):
                _joint_fit_condition(bundle, "C", period=2.0)


class ExactResamplingTests(unittest.TestCase):
    @staticmethod
    def _bootstrap_result(value):
        return SimpleNamespace(
            B_hat=float(value),
            q_hat=np.array([float(value), -float(value)]),
        )

    def test_bootstrap_propagates_failed_draw(self):
        with patch(
            "analysis.mzi_null_bounds._fit_blocks",
            side_effect=[self._bootstrap_result(1.0), ValueError("draw failure")],
        ):
            with self.assertRaisesRegex(ValueError, "draw failure"):
                _bootstrap_blocks(
                    [{}, {}],
                    n_bootstrap=2,
                    cl=0.95,
                    covariance_scale="none",
                    seed=1,
                )

    def test_bootstrap_completes_exact_requested_count(self):
        with patch(
            "analysis.mzi_null_bounds._fit_blocks",
            side_effect=[self._bootstrap_result(v) for v in (1.0, 2.0, 3.0)],
        ):
            result = _bootstrap_blocks(
                [{}, {}],
                n_bootstrap=3,
                cl=0.95,
                covariance_scale="none",
                seed=1,
            )
        self.assertEqual(3.0, result["bootstrap_n_success"])

    @staticmethod
    def _design(include_launch_terms):
        y = np.arange(1.0, 6.0)
        var = np.ones(5)
        if include_launch_terms:
            X = np.column_stack((np.ones(5), np.arange(5.0), np.arange(5.0) ** 2))
            launch_offset = 1
        else:
            X = np.ones((5, 1))
            launch_offset = None
        return X, y, var, 0, 0, 0, launch_offset

    @staticmethod
    def _linear_fit(beta):
        y = np.arange(1.0, 6.0)
        return SimpleNamespace(
            beta=np.asarray(beta, dtype=float),
            y=y,
            y_hat=np.zeros_like(y),
            resid=y.copy(),
        )

    def test_permutation_propagates_failed_draw(self):
        with (
            patch(
                "analysis.mzi_null_bounds._build_actual_voltage_design",
                side_effect=lambda blocks, include_launch_terms, **kwargs: self._design(
                    include_launch_terms
                ),
            ),
            patch(
                "analysis.mzi_null_bounds._solve_wls",
                side_effect=[self._linear_fit([0.0]), ValueError("draw failure")],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "draw failure"):
                _permutation_pvalue(
                    [{}, {}],
                    observed_B=1.0,
                    n_permutations=1,
                    covariance_scale="none",
                    seed=1,
                )

    def test_permutation_completes_exact_requested_count(self):
        fits = [self._linear_fit([0.0])] + [
            self._linear_fit([0.0, 1.0, 0.0]) for _ in range(3)
        ]
        with (
            patch(
                "analysis.mzi_null_bounds._build_actual_voltage_design",
                side_effect=lambda blocks, include_launch_terms, **kwargs: self._design(
                    include_launch_terms
                ),
            ),
            patch("analysis.mzi_null_bounds._solve_wls", side_effect=fits),
        ):
            result = _permutation_pvalue(
                [{}, {}],
                observed_B=0.5,
                n_permutations=3,
                covariance_scale="none",
                seed=1,
            )
        self.assertEqual(3.0, result["permutation_n_success"])

    def test_negative_resample_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "bootstrap count"):
            _bootstrap_blocks(
                [{}, {}],
                n_bootstrap=-1,
                cl=0.95,
                covariance_scale="none",
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "permutation count"):
            _permutation_pvalue(
                [{}],
                observed_B=1.0,
                n_permutations=-1,
                covariance_scale="none",
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
