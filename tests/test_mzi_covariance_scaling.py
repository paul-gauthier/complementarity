import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.mzi_analysis import (
    DarkModel,
    MZIScanData,
    _conservative_covariance_scale,
    _poisson_singles_expected_fisher_covariance,
    _standard_deviation_from_variance,
    fit_idler_sinusoid_poisson_joint,
    fit_scan,
    fit_weighted_sinusoid,
    validate_positive_variances,
)
from analysis.mzi_joint import fit_joint_coincidences


class CoincidenceCovarianceScalingTests(unittest.TestCase):
    @staticmethod
    def _synthetic_scan():
        period = 2.0
        x = np.linspace(0.0, 3.0, 11)
        phase = 2.0 * np.pi * (x - 0.2) / period
        residual = np.array(
            [1.0, -0.5, 0.25, -1.0, 0.5, -0.25, 1.0, -0.5, 0.25, -1.0, 0.5]
        )
        y1 = 100.0 * (1.0 + 0.3 * np.cos(phase)) + residual
        y2 = 80.0 * (1.0 - 0.3 * np.cos(phase)) - residual
        return period, x, y1, y2

    def test_conservative_scale_has_unit_floor(self):
        self.assertEqual(1.0, _conservative_covariance_scale(5.0, 10))
        self.assertEqual(1.0, _conservative_covariance_scale(0.0, 10))

    def test_conservative_scale_rejects_invalid_inputs(self):
        for weighted_sse in (float("nan"), float("inf"), -1.0):
            with self.subTest(weighted_sse=weighted_sse):
                with self.assertRaisesRegex(ValueError, "weighted SSE"):
                    _conservative_covariance_scale(weighted_sse, 10)
        for df in (0, -1):
            with self.subTest(df=df):
                with self.assertRaisesRegex(ValueError, "positive residual degrees"):
                    _conservative_covariance_scale(5.0, df)

    def test_standard_deviation_rejects_invalid_variance(self):
        for variance in (float("nan"), float("inf"), -1.0):
            with self.subTest(variance=variance):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    _standard_deviation_from_variance(
                        variance,
                        context="test uncertainty",
                    )

    def test_conservative_scale_preserves_overdispersion(self):
        self.assertEqual(2.5, _conservative_covariance_scale(25.0, 10))

    def test_variances_must_be_finite_and_positive(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                values = np.array([1.0, invalid, 2.0])
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and positive; invalid indices: 1",
                ):
                    validate_positive_variances(values)

    def test_valid_variances_are_returned_unchanged(self):
        values = np.array([1e-15, 1.0, 100.0])
        np.testing.assert_array_equal(validate_positive_variances(values), values)

    def test_standalone_fit_rejects_invalid_variance_with_channel_context(self):
        period, x, y1, y2 = self._synthetic_scan()
        variance = np.full_like(x, 100.0)
        variance[3] = float("nan")
        with self.assertRaisesRegex(
            ValueError,
            "exit 1 coincidence variances.*invalid indices: 3",
        ):
            fit_weighted_sinusoid(
                x,
                y1,
                y2,
                variance,
                np.full_like(x, 100.0),
                P_fixed=period,
            )

    def test_standalone_uncertainty_scales_with_input_standard_deviation(self):
        period, x, y1, y2 = self._synthetic_scan()

        fit_low_variance = fit_weighted_sinusoid(
            x,
            y1,
            y2,
            np.full_like(x, 100.0),
            np.full_like(x, 100.0),
            P_fixed=period,
        )
        fit_high_variance = fit_weighted_sinusoid(
            x,
            y1,
            y2,
            np.full_like(x, 400.0),
            np.full_like(x, 400.0),
            P_fixed=period,
        )

        self.assertTrue(math.isfinite(fit_low_variance.total_fringe_sigma))
        self.assertAlmostEqual(
            2.0,
            fit_high_variance.total_fringe_sigma / fit_low_variance.total_fringe_sigma,
            places=4,
        )

    def test_joint_fit_respects_variance_scale(self):
        period, x, y1, y2 = self._synthetic_scan()
        fits = []
        init = {
            "A1": 100.0,
            "A2": 80.0,
            "V": 0.3,
            "P": period,
            "x0_list": [0.2],
        }
        for variance in (100.0, 400.0):
            data_pass = SimpleNamespace(
                xs=x,
                rc1=y1,
                rc2=y2,
                var_rc1=np.full_like(x, variance),
                var_rc2=np.full_like(x, variance),
            )
            fits.append(
                fit_joint_coincidences(
                    [data_pass],
                    P_fixed=period,
                    init=init,
                )
            )

        self.assertAlmostEqual(
            2.0,
            fits[1].total_fringe_sigma / fits[0].total_fringe_sigma,
            places=4,
        )
    def test_joint_fit_rejects_invalid_variance(self):
        period, x, y1, y2 = self._synthetic_scan()
        invalid_variance = np.full_like(x, 100.0)
        invalid_variance[4] = 0.0
        data_pass = SimpleNamespace(
            xs=x,
            rc1=y1,
            rc2=y2,
            var_rc1=invalid_variance,
            var_rc2=np.full_like(x, 100.0),
        )
        init = {
            "A1": 100.0,
            "A2": 80.0,
            "V": 0.3,
            "P": period,
            "x0_list": [0.2],
        }

        with self.assertRaisesRegex(
            ValueError,
            "joint coincidence pass 1 exit 1 variances.*invalid indices: 4",
        ):
            fit_joint_coincidences([data_pass], P_fixed=period, init=init)

    def test_fit_scan_routes_fixed_and_fitted_periods(self):
        data = MZIScanData(
            xs=[0.0, 1.0, 2.0, 3.0],
            rc1=[1.0] * 4,
            rc2=[1.0] * 4,
            var_rc1=[1.0] * 4,
            var_rc2=[1.0] * 4,
            n_i1=[1] * 4,
            n_i2=[1] * 4,
        )
        dark = DarkModel.from_counts(Ns=0, Ni=0, Nc=0, T=1.0, Ni2=0, Nc2=0)

        for period, coincidence_period, singles_period in (
            (2.0, 2.0, 2.0),
            (None, None, 2.1),
        ):
            with self.subTest(period=period):
                coincidence = SimpleNamespace(P=2.1)
                singles = object()
                with (
                    patch(
                        "analysis.mzi_analysis.fit_weighted_sinusoid",
                        return_value=coincidence,
                    ) as fit_coincidence,
                    patch(
                        "analysis.mzi_analysis.fit_idler_sinusoid_poisson_joint",
                        return_value=singles,
                    ) as fit_singles,
                ):
                    result = fit_scan(data, "IC", period, dark, 1.0)

                self.assertIs(result.coincidence, coincidence)
                self.assertIs(result.singles, singles)
                self.assertEqual(
                    fit_coincidence.call_args.kwargs["P_fixed"],
                    coincidence_period,
                )
                self.assertEqual(
                    fit_singles.call_args.kwargs["P_fixed"],
                    singles_period,
                )

    def test_fit_scan_rejects_invalid_requests(self):
        data = MZIScanData(
            xs=[0.0, 1.0, 2.0, 3.0],
            rc1=[1.0] * 4,
            rc2=[1.0] * 4,
            var_rc1=[1.0] * 4,
            var_rc2=[1.0] * 4,
            n_i1=[1] * 4,
            n_i2=[1] * 4,
        )
        dark = DarkModel.from_counts(Ns=0, Ni=0, Nc=0, T=1.0, Ni2=0, Nc2=0)

        for period in (float("nan"), 0.0, -1.0):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ValueError, "fixed period"):
                    fit_scan(data, "C", period, dark, 1.0)

        with self.assertRaisesRegex(ValueError, "channel must be"):
            fit_scan(data, "unknown", None, dark, 1.0)

        too_short = MZIScanData(
            xs=[0.0, 1.0, 2.0],
            rc1=[1.0] * 3,
            rc2=[1.0] * 3,
            var_rc1=[1.0] * 3,
            var_rc2=[1.0] * 3,
        )
        with self.assertRaisesRegex(ValueError, "at least four points"):
            fit_scan(too_short, "C", None, dark, 1.0)

        missing_dark_field = SimpleNamespace(Ni2=0, T=1.0)
        with self.assertRaises(AttributeError):
            fit_scan(data, "I", 2.0, missing_dark_field, 1.0)


class PoissonExpectedFisherTests(unittest.TestCase):
    @staticmethod
    def _synthetic_counts():
        period = 2.05
        x = np.linspace(0.0, 3.0, 11)
        phase = 2.0 * np.pi * (x - 0.2) / period
        duration = 20.0
        dark_duration = 100.0
        dark_rate1 = 5.0
        dark_rate2 = 6.0
        counts1 = np.rint(
            duration * (dark_rate1 + 100.0 * (1.0 + 0.3 * np.cos(phase)))
        )
        counts2 = np.rint(
            duration * (dark_rate2 + 80.0 * (1.0 - 0.3 * np.cos(phase)))
        )
        return (
            period,
            x,
            counts1,
            counts2,
            duration,
            dark_rate1 * dark_duration,
            dark_rate2 * dark_duration,
            dark_duration,
        )

    def test_fixed_and_free_period_fits_have_finite_expected_fisher_uncertainties(self):
        period, x, counts1, counts2, duration, dark1, dark2, dark_duration = (
            self._synthetic_counts()
        )
        for fixed_period in (period, None):
            with self.subTest(fixed_period=fixed_period):
                fit = fit_idler_sinusoid_poisson_joint(
                    x,
                    counts1,
                    counts2,
                    duration,
                    dark1,
                    dark2,
                    dark_duration,
                    P_fixed=fixed_period,
                )
                for field in (
                    "A1_sigma",
                    "A2_sigma",
                    "A_sig_sigma",
                    "V_sigma",
                    "total_fringe_sigma",
                    "Fringe1_sigma",
                    "Fringe2_sigma",
                ):
                    self.assertTrue(np.isfinite(getattr(fit, field)), field)
                if fixed_period is None:
                    self.assertTrue(np.isfinite(fit.P_sigma))
                else:
                    self.assertTrue(np.isnan(fit.P_sigma))

    def test_nonidentifiable_expected_fisher_information_raises(self):
        with self.assertRaisesRegex(ValueError, "covariance is not identifiable"):
            _poisson_singles_expected_fisher_covariance(
                np.zeros(11),
                10.0,
                100.0,
                A1=100.0,
                dark_rate1=5.0,
                A2=80.0,
                dark_rate2=6.0,
                V=0.3,
                x0=0.0,
                P=2.0,
                fit_period=False,
            )


if __name__ == "__main__":
    unittest.main()
