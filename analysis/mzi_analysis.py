#!/usr/bin/env python

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from .dark_analysis import DarkModel, accidentals_counts, corrected_coincidence_rate

DEFAULT_PERIOD = 2.05  # volts

_dp = 0.3
P_FIT_MIN = DEFAULT_PERIOD - _dp
P_FIT_MAX = DEFAULT_PERIOD + _dp


@dataclass
class PointResult:
    actual_delta: float
    r_s_corr: float
    r_i1_corr: float
    r_i2_corr: float
    rc1_corr: float
    rc2_corr: float
    var_rc1: float
    var_rc2: float
    n_i1: int
    n_i2: int


@dataclass
class MZIScanData:
    xs: List[float] = field(default_factory=list)
    rc1: List[float] = field(default_factory=list)
    rc2: List[float] = field(default_factory=list)
    var_rc1: List[float] = field(default_factory=list)
    var_rc2: List[float] = field(default_factory=list)
    ri1: List[float] = field(default_factory=list)
    ri2: List[float] = field(default_factory=list)
    n_i1: List[int] = field(default_factory=list)
    n_i2: List[int] = field(default_factory=list)

    def append(self, point: PointResult) -> None:
        self.xs.append(point.actual_delta)
        self.rc1.append(point.rc1_corr)
        self.rc2.append(point.rc2_corr)
        self.var_rc1.append(point.var_rc1)
        self.var_rc2.append(point.var_rc2)
        self.ri1.append(point.r_i1_corr)
        self.ri2.append(point.r_i2_corr)
        self.n_i1.append(point.n_i1)
        self.n_i2.append(point.n_i2)


@dataclass(kw_only=True)
class FitResult:
    x_fit: np.ndarray
    y_fit1: np.ndarray
    y_fit2: np.ndarray
    V: float
    x0: float
    V_sigma: float
    P: float
    A1: float
    A2: float
    A1_sigma: float
    A2_sigma: float
    A_sig_sigma: float
    P_sigma: float
    total_fringe: float
    total_fringe_sigma: float
    Fringe1_sigma: float
    Fringe2_sigma: float


@dataclass(kw_only=True)
class ScanFits:
    coincidence: Optional[FitResult]
    singles: Optional[FitResult]


def _require_successful_optimizer_result(
    result: Any,
    *,
    expected_size: int,
    context: str,
) -> np.ndarray:
    """Return a validated optimizer parameter vector or raise."""
    status = getattr(result, "status", "unknown")
    message = getattr(result, "message", "no optimizer message")
    detail = f"(status {status}): {message}"
    if not bool(getattr(result, "success", False)):
        raise RuntimeError(f"{context} optimizer failed {detail}")

    try:
        params = np.asarray(result.x, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{context} optimizer returned an invalid parameter vector {detail}"
        ) from exc
    if params.ndim != 1 or params.size != int(expected_size):
        raise RuntimeError(
            f"{context} optimizer returned {params.size} parameters with shape "
            f"{params.shape}; expected {int(expected_size)} parameters {detail}"
        )
    if not np.all(np.isfinite(params)):
        raise RuntimeError(
            f"{context} optimizer returned non-finite parameters {detail}"
        )

    try:
        objective_array = np.asarray(result.fun, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{context} optimizer returned an invalid objective {detail}"
        ) from exc
    if objective_array.ndim != 0:
        raise RuntimeError(
            f"{context} optimizer returned a non-scalar objective with shape "
            f"{objective_array.shape} {detail}"
        )
    objective = float(objective_array)
    if not np.isfinite(objective):
        raise RuntimeError(
            f"{context} optimizer returned a non-finite objective {detail}"
        )
    return params


def _require_finite_fit_parameters(
    values: Dict[str, float],
    *,
    positive: Tuple[str, ...],
    unit_interval: Tuple[str, ...],
    context: str,
) -> None:
    invalid = [name for name, value in values.items() if not np.isfinite(value)]
    invalid.extend(name for name in positive if not (values[name] > 0.0))
    invalid.extend(
        name for name in unit_interval if not (0.0 < values[name] < 1.0)
    )
    if invalid:
        names = ", ".join(dict.fromkeys(invalid))
        raise RuntimeError(
            f"{context} optimizer produced invalid transformed parameters: {names}"
        )


def _validate_optional_fixed_period(value: Any, *, context: str) -> Optional[float]:
    if value is None:
        return None
    try:
        period = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} fixed period must be finite and positive") from exc
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"{context} fixed period must be finite and positive")
    return period


def _poisson_singles_expected_fisher_covariance(
    x: np.ndarray,
    duration_s: float,
    dark_duration_s: float,
    *,
    A1: float,
    dark_rate1: float,
    A2: float,
    dark_rate2: float,
    V: float,
    x0: float,
    P: float,
    fit_period: bool,
) -> np.ndarray:
    """Return inverse expected Fisher information for a Poisson singles fit."""
    x = np.asarray(x, dtype=float)
    m = 7 if fit_period else 6
    idx_a1, idx_d1, idx_a2, idx_d2, idx_u, idx_x0 = range(6)
    idx_g = 6 if fit_period else None
    information = np.zeros((m, m), dtype=float)

    wx = 2.0 * np.pi * (x - x0) / P
    c = np.cos(wx)
    s = np.sin(wx)
    visibility_derivative = V * (1.0 - V)
    phase_derivative = 2.0 * np.pi / P

    lam1 = duration_s * (dark_rate1 + A1 * (1.0 + V * c))
    jac1 = np.zeros((x.size, m), dtype=float)
    jac1[:, idx_a1] = duration_s * A1 * (1.0 + V * c)
    jac1[:, idx_d1] = duration_s * dark_rate1
    jac1[:, idx_u] = duration_s * A1 * visibility_derivative * c
    jac1[:, idx_x0] = duration_s * A1 * V * phase_derivative * s
    if idx_g is not None:
        jac1[:, idx_g] = duration_s * A1 * V * wx * s
    information += jac1.T @ (jac1 / lam1[:, np.newaxis])

    lam2 = duration_s * (dark_rate2 + A2 * (1.0 - V * c))
    jac2 = np.zeros((x.size, m), dtype=float)
    jac2[:, idx_a2] = duration_s * A2 * (1.0 - V * c)
    jac2[:, idx_d2] = duration_s * dark_rate2
    jac2[:, idx_u] = -duration_s * A2 * visibility_derivative * c
    jac2[:, idx_x0] = -duration_s * A2 * V * phase_derivative * s
    if idx_g is not None:
        jac2[:, idx_g] = -duration_s * A2 * V * wx * s
    information += jac2.T @ (jac2 / lam2[:, np.newaxis])

    information[idx_d1, idx_d1] += dark_duration_s * dark_rate1
    information[idx_d2, idx_d2] += dark_duration_s * dark_rate2
    information = 0.5 * (information + information.T)

    if not np.all(np.isfinite(information)):
        raise ValueError(
            "idler sinusoid Poisson fit covariance is not identifiable "
            "(expected Fisher information is nonfinite)"
        )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "idler sinusoid Poisson fit covariance is not identifiable "
            "(expected Fisher information decomposition failed)"
        ) from exc
    tolerance = (
        max(float(eigenvalues[-1]), 1.0)
        * np.finfo(float).eps
        * max(m, 1)
        * 100.0
    )
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= tolerance:
        raise ValueError(
            "idler sinusoid Poisson fit covariance is not identifiable "
            f"(minimum Fisher eigenvalue {float(eigenvalues[0]):.6g}, "
            f"tolerance {tolerance:.6g})"
        )
    covariance = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "idler sinusoid Poisson fit covariance is not identifiable "
            "(inverse expected Fisher information is nonfinite)"
        )
    return covariance


def _covariance_from_weighted_jacobian(
    weighted_jacobian: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Return covariance for a finite, full-rank weighted Jacobian or raise."""
    jacobian = np.asarray(weighted_jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] <= 0 or jacobian.shape[1] <= 0:
        raise ValueError(f"{context} weighted Jacobian must be a nonempty matrix")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError(f"{context} weighted Jacobian must be finite")

    expected_rank = int(jacobian.shape[1])
    rank = int(np.linalg.matrix_rank(jacobian))
    if rank != expected_rank:
        raise ValueError(
            f"{context} covariance is not identifiable "
            f"(Jacobian rank {rank}, expected {expected_rank})"
        )

    information = jacobian.T @ jacobian
    if not np.all(np.isfinite(information)):
        raise ValueError(f"{context} information matrix must be finite")
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context} covariance is not identifiable") from exc
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{context} covariance must be finite")

    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context} covariance decomposition failed") from exc
    tolerance = (
        max(float(eigenvalues[-1]), 1.0)
        * np.finfo(float).eps
        * max(covariance.shape[0], 1)
        * 100.0
    )
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= tolerance:
        raise ValueError(
            f"{context} covariance is not positive definite "
            f"(minimum eigenvalue {float(eigenvalues[0]):.6g}, "
            f"tolerance {tolerance:.6g})"
        )
    return covariance


def _standard_deviation_from_variance(value: Any, *, context: str) -> float:
    """Return sqrt(value) for a finite nonnegative variance or raise."""
    variance = float(value)
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError(f"{context} variance must be finite and nonnegative")
    return math.sqrt(variance)


def _conservative_covariance_scale(weighted_sse: float, df: int) -> float:
    """Return max(1, reduced chi-square) for propagated counting variances."""
    if int(df) <= 0:
        raise ValueError(
            f"covariance scaling requires positive residual degrees of freedom; got {int(df)}"
        )
    weighted_sse = float(weighted_sse)
    if not np.isfinite(weighted_sse) or weighted_sse < 0.0:
        raise ValueError("weighted SSE must be finite and nonnegative")
    reduced_chi2 = weighted_sse / float(df)
    return max(1.0, reduced_chi2)


def full_restoration_circle_distance(
    q_hat: np.ndarray,
    cov_q: np.ndarray,
    radius: float,
) -> Tuple[float, float, np.ndarray]:
    """Return the minimum Mahalanobis distance from q_hat to a circle."""
    q_hat = np.asarray(q_hat, dtype=float).reshape(2)
    cov_q = np.asarray(cov_q, dtype=float).reshape(2, 2)
    radius = float(radius)

    if not np.all(np.isfinite(q_hat)):
        raise ValueError("The fitted quadratures must be finite")
    if not np.all(np.isfinite(cov_q)):
        raise ValueError("The quadrature covariance must be finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("The full-restoration radius must be finite and positive")

    cov_q = 0.5 * (cov_q + cov_q.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_q)
    tolerance = max(float(np.max(eigenvalues)), 1.0) * np.finfo(float).eps * 100.0
    if float(np.min(eigenvalues)) <= tolerance:
        raise ValueError("The quadrature covariance must be positive definite")

    precision = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

    def squared_distance(phase: float) -> float:
        q = radius * np.array([math.cos(phase), math.sin(phase)], dtype=float)
        delta = q - q_hat
        return float(delta @ precision @ delta)

    n_grid = 4096
    step = 2.0 * np.pi / n_grid
    phases = np.arange(n_grid, dtype=float) * step
    circle = radius * np.column_stack((np.cos(phases), np.sin(phases)))
    deltas = circle - q_hat
    distances_squared = np.einsum("ni,ij,nj->n", deltas, precision, deltas)

    previous = np.roll(distances_squared, 1)
    following = np.roll(distances_squared, -1)
    candidate_indices = np.flatnonzero(
        (distances_squared <= previous)
        & (distances_squared <= following)
        & ((distances_squared < previous) | (distances_squared < following))
    )
    if candidate_indices.size == 0:
        candidate_indices = np.asarray([int(np.argmin(distances_squared))])

    best_phase = float(phases[int(candidate_indices[0])])
    best_squared_distance = squared_distance(best_phase)
    for index in candidate_indices:
        center = float(phases[int(index)])
        result = minimize_scalar(
            squared_distance,
            bounds=(center - step, center + step),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if result.success and float(result.fun) < best_squared_distance:
            best_phase = float(result.x)
            best_squared_distance = float(result.fun)

    best_phase %= 2.0 * np.pi
    nearest_q = radius * np.array(
        [math.cos(best_phase), math.sin(best_phase)],
        dtype=float,
    )
    return math.sqrt(max(0.0, best_squared_distance)), best_phase, nearest_q


def validate_positive_variances(
    values,
    *,
    expected_size: Optional[int] = None,
    context: str = "variances",
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{context} must be a one-dimensional array")
    if expected_size is not None and result.size != int(expected_size):
        raise ValueError(
            f"{context} length {result.size} does not match the expected "
            f"length {int(expected_size)}"
        )
    if result.size == 0:
        raise ValueError(f"{context} must not be empty")

    invalid = np.flatnonzero(~np.isfinite(result) | (result <= 0.0))
    if invalid.size:
        indices = ", ".join(str(int(index)) for index in invalid[:8])
        if invalid.size > 8:
            indices += ", ..."
        raise ValueError(
            f"{context} must be finite and positive; invalid indices: {indices}"
        )
    return result


def _assert_constant_point_duration(recs, label: str) -> None:
    """
    Reject records with missing, invalid, or non-constant per-point durations.
    Idler singles fits assume a single per-point duration T.
    """
    if not recs:
        raise ValueError(f"{label} must contain at least one point")
    vals = []
    for index, record in enumerate(recs):
        value = record.get("duration_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} point {index} duration_s must be numeric")
        v = float(value)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{label} point {index} duration_s must be finite and positive")
        vals.append(v)
    v0 = vals[0]
    if any(not np.isclose(v, v0, rtol=1e-9, atol=1e-12) for v in vals[1:]):
        uniq = sorted(set(round(v, 6) for v in vals))
        raise ValueError(
            f"non-constant per-point durations in {label}: {uniq}. "
            "Idler singles fits assume a single per-point duration T. "
            "Split and fit per pass (e.g., using mzi_plot_alternating_by_pass.py), "
            "or regenerate runs with constant durations."
        )


def point_from_counts(
    counts: Dict[str, int], acq_dur: float, dark: DarkModel, actual_delta: float
) -> PointResult:
    parsed: Dict[str, int] = {}
    for name in ("N_s", "N_i", "N_i2", "N_c", "N_c2"):
        if name not in counts:
            raise ValueError(f"counts is missing required field {name}")
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"counts.{name} must be a non-negative integer")
        parsed[name] = value
    if isinstance(acq_dur, bool) or not isinstance(acq_dur, (int, float)):
        raise ValueError("acq_dur must be numeric")
    acq_dur = float(acq_dur)
    if not np.isfinite(acq_dur) or acq_dur <= 0:
        raise ValueError("acq_dur must be finite and positive")
    if isinstance(actual_delta, bool) or not isinstance(actual_delta, (int, float)):
        raise ValueError("actual_delta must be numeric")
    actual_delta = float(actual_delta)
    if not np.isfinite(actual_delta):
        raise ValueError("actual_delta must be finite")

    n_s = parsed["N_s"]
    n_i1 = parsed["N_i"]
    n_i2 = parsed["N_i2"]
    n_c1 = parsed["N_c"]
    n_c2 = parsed["N_c2"]

    n_acc1, var_acc1 = accidentals_counts(n_s, n_i1, float(acq_dur))
    n_acc2, var_acc2 = accidentals_counts(n_s, n_i2, float(acq_dur))

    R_c1_corr, var_R_c1_corr = corrected_coincidence_rate(
        n_c1, acq_dur, n_acc1, var_acc1, dark, channel="I"
    )
    R_c2_corr, var_R_c2_corr = corrected_coincidence_rate(
        n_c2, acq_dur, n_acc2, var_acc2, dark, channel="I2"
    )

    r_s_corr = max(n_s / float(acq_dur) - dark.r_s, 0.0)
    r_i1_corr = max(n_i1 / float(acq_dur) - dark.r_i, 0.0)
    r_i2_corr = max(n_i2 / float(acq_dur) - dark.r_i2, 0.0)

    return PointResult(
        actual_delta=float(actual_delta),
        r_s_corr=float(r_s_corr),
        r_i1_corr=float(r_i1_corr),
        r_i2_corr=float(r_i2_corr),
        rc1_corr=float(R_c1_corr),
        rc2_corr=float(R_c2_corr),
        var_rc1=float(var_R_c1_corr),
        var_rc2=float(var_R_c2_corr),
        n_i1=int(n_i1),
        n_i2=int(n_i2),
    )


def scan_from_records(records: List[Dict[str, Any]], dark: DarkModel) -> Tuple[MZIScanData, float]:
    data = MZIScanData()
    acq_dur = float(records[0]["duration_s"]) if records else 0.0
    for rec in records:
        pr = point_from_counts(
            rec["counts"], float(rec["duration_s"]), dark, float(rec["actual_delta_V"])
        )
        data.append(pr)
    return data, acq_dur


def fit_idler_sinusoid_poisson_joint(
    xs_V,
    counts_i1,
    counts_i2,
    duration_s,
    dark_counts_i1,
    dark_counts_i2,
    dark_duration_s,
    P_fixed=None,
) -> FitResult:
    P_fixed = _validate_optional_fixed_period(P_fixed, context="idler fit")
    x = np.asarray(xs_V, dtype=float)
    N1 = np.asarray(counts_i1, dtype=float)
    N2 = np.asarray(counts_i2, dtype=float)

    if x.ndim != 1 or N1.ndim != 1 or N2.ndim != 1:
        raise ValueError("idler fit coordinates and counts must be one-dimensional")
    if x.size < 4:
        raise ValueError("idler fit requires at least four points")
    if N1.size != x.size or N2.size != x.size:
        raise ValueError("idler fit coordinates and count arrays have inconsistent lengths")
    if not (np.isfinite(x).all() and np.isfinite(N1).all() and np.isfinite(N2).all()):
        raise ValueError("idler fit coordinates and counts must be finite")
    if np.any(N1 < 0.0) or np.any(N2 < 0.0):
        raise ValueError("idler fit counts must be nonnegative")

    T = float(duration_s)
    dark_T = float(dark_duration_s)
    dark_N1 = float(dark_counts_i1)
    dark_N2 = float(dark_counts_i2)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("idler fit duration_s must be finite and positive")
    if not np.isfinite(dark_T) or dark_T <= 0.0:
        raise ValueError("idler fit dark_duration_s must be finite and positive")
    if not np.isfinite(dark_N1) or not np.isfinite(dark_N2):
        raise ValueError("idler fit dark counts must be finite")
    if dark_N1 < 0.0 or dark_N2 < 0.0:
        raise ValueError("idler fit dark counts must be nonnegative")

    r1_dark0 = max(dark_N1 / dark_T, 1e-12)
    r2_dark0 = max(dark_N2 / dark_T, 1e-12)
    rates1 = N1 / T
    rates2 = N2 / T

    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        P0 = float(P_fixed)
        w = 2.0 * math.pi / P0
        X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
        coeffs1, _, _, _ = np.linalg.lstsq(X, rates1, rcond=None)
        coeffs2, _, _, _ = np.linalg.lstsq(X, rates2, rcond=None)
        A1_tot0, C1_0, S1_0 = coeffs1
        A2_tot0, C2_0, S2_0 = coeffs2
    else:
        P_grid = np.linspace(P_FIT_MIN, P_FIT_MAX, 31)
        best = None
        for P0 in P_grid:
            w = 2.0 * math.pi / P0
            X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
            c1, _, _, _ = np.linalg.lstsq(X, rates1, rcond=None)
            c2, _, _, _ = np.linalg.lstsq(X, rates2, rcond=None)
            yhat1 = X @ c1
            yhat2 = X @ c2
            sse = float(np.sum((rates1 - yhat1) ** 2) + np.sum((rates2 - yhat2) ** 2))
            if (best is None) or (sse < best[0]):
                best = (sse, c1, c2, P0)
        _, (A1_tot0, C1_0, S1_0), (A2_tot0, C2_0, S2_0), P0 = best

    B1_0 = float(math.hypot(C1_0, S1_0))
    B2_0 = float(math.hypot(C2_0, S2_0))
    w0 = 2.0 * math.pi / P0
    x0_0 = float(math.atan2(S1_0, C1_0) / w0)

    A1_sig0 = max(float(A1_tot0) - r1_dark0, 1e-12)
    A2_sig0 = max(float(A2_tot0) - r2_dark0, 1e-12)
    v1 = B1_0 / A1_sig0 if A1_sig0 > 0 else 0.0
    v2 = B2_0 / A2_sig0 if A2_sig0 > 0 else 0.0
    V0 = float(np.clip(np.median([v1, v2]), 0.0, 0.95))

    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        p0 = [
            math.log(A1_sig0),
            math.log(r1_dark0),
            math.log(A2_sig0),
            math.log(r2_dark0),
            logit(V0),
            x0_0,
        ]
    else:
        p0 = [
            math.log(A1_sig0),
            math.log(r1_dark0),
            math.log(A2_sig0),
            math.log(r2_dark0),
            logit(V0),
            x0_0,
            math.log(P0),
        ]

    def nll(p):
        if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
            a1, d1, a2, d2, u, x0 = p
            P = float(P_fixed)
        else:
            a1, d1, a2, d2, u, x0, g = p
            P = float(np.exp(g))
        A1 = float(np.exp(a1))
        r1_dark = float(np.exp(d1))
        A2 = float(np.exp(a2))
        r2_dark = float(np.exp(d2))
        V = float(expit(u))
        wx = 2.0 * math.pi * (x - x0) / P
        rate1 = r1_dark + A1 * (1.0 + V * np.cos(wx))
        rate2 = r2_dark + A2 * (1.0 - V * np.cos(wx))
        lam1 = T * rate1
        lam2 = T * rate2
        if (
            np.any(lam1 <= 0)
            or np.any(lam2 <= 0)
            or not (np.all(np.isfinite(lam1)) and np.all(np.isfinite(lam2)))
        ):
            return np.inf
        nll_counts = np.sum(lam1 - N1 * np.log(lam1)) + np.sum(lam2 - N2 * np.log(lam2))
        lam_dark1 = dark_T * r1_dark
        lam_dark2 = dark_T * r2_dark
        if (
            lam_dark1 <= 0
            or lam_dark2 <= 0
            or not (np.isfinite(lam_dark1) and np.isfinite(lam_dark2))
        ):
            return np.inf
        nll_dark = lam_dark1 - dark_N1 * math.log(lam_dark1)
        nll_dark += lam_dark2 - dark_N2 * math.log(lam_dark2)
        return nll_counts + nll_dark

    bounds = (
        None
        if (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0)
        else [
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (math.log(P_FIT_MIN), math.log(P_FIT_MAX)),
        ]
    )

    res = (
        minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxls": 100},
        )
        if bounds is not None
        else minimize(nll, p0, method="L-BFGS-B", options={"maxls": 100})
    )
    params = _require_successful_optimizer_result(
        res,
        expected_size=len(p0),
        context="idler sinusoid Poisson fit",
    )
    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        a1, d1, a2, d2, u, x0 = params
        P = float(P_fixed)
    else:
        a1, d1, a2, d2, u, x0, g = params
        P = float(np.exp(g))
    A1 = float(np.exp(a1))
    r1_dark = float(np.exp(d1))
    A2 = float(np.exp(a2))
    r2_dark = float(np.exp(d2))
    V = float(expit(u))
    _require_finite_fit_parameters(
        {
            "A1": A1,
            "dark_rate1": r1_dark,
            "A2": A2,
            "dark_rate2": r2_dark,
            "V": V,
            "x0": float(x0),
            "P": P,
        },
        positive=("A1", "dark_rate1", "A2", "dark_rate2", "P"),
        unit_interval=("V",),
        context="idler sinusoid Poisson fit",
    )

    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    wx_fit = 2.0 * math.pi * (x_fit - x0) / P
    y1_fit_corr = A1 * (1.0 + V * np.cos(wx_fit))
    y2_fit_corr = A2 * (1.0 - V * np.cos(wx_fit))

    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        idx_a1, idx_a2, idx_u = 0, 2, 4
        idx_g = None
    else:
        idx_a1, idx_a2, idx_u, idx_g = 0, 2, 4, 6

    Cov = _poisson_singles_expected_fisher_covariance(
        x,
        T,
        dark_T,
        A1=A1,
        dark_rate1=r1_dark,
        A2=A2,
        dark_rate2=r2_dark,
        V=V,
        x0=float(x0),
        P=P,
        fit_period=idx_g is not None,
    )

    var_a1 = float(Cov[idx_a1, idx_a1])
    var_a2 = float(Cov[idx_a2, idx_a2])
    cov_a1a2 = float(Cov[idx_a1, idx_a2])
    var_u = float(Cov[idx_u, idx_u])

    V_sigma = V * (1.0 - V) * _standard_deviation_from_variance(
        var_u,
        context="idler sinusoid Poisson fit visibility",
    )
    A1_sigma = A1 * _standard_deviation_from_variance(
        var_a1,
        context="idler sinusoid Poisson fit exit 1 amplitude",
    )
    A2_sigma = A2 * _standard_deviation_from_variance(
        var_a2,
        context="idler sinusoid Poisson fit exit 2 amplitude",
    )

    g1 = 0.5 * A1
    g2 = 0.5 * A2
    var_A_sig = g1 * g1 * var_a1 + g2 * g2 * var_a2 + 2.0 * g1 * g2 * cov_a1a2
    A_sig_sigma = _standard_deviation_from_variance(
        var_A_sig,
        context="idler sinusoid Poisson fit mean amplitude",
    )

    if idx_g is not None:
        var_g = float(Cov[idx_g, idx_g])
        P_sigma = P * _standard_deviation_from_variance(
            var_g,
            context="idler sinusoid Poisson fit period",
        )
    else:
        P_sigma = float("nan")

    total_fringe = (A1 + A2) * V
    grad = np.zeros(Cov.shape[0], dtype=float)
    grad[idx_a1] = A1 * V
    grad[idx_a2] = A2 * V
    grad[idx_u] = (A1 + A2) * V * (1.0 - V)
    total_fringe_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="idler sinusoid Poisson fit total fringe",
    )

    grad = np.zeros(Cov.shape[0], dtype=float)
    grad[idx_a1] = V * A1
    grad[idx_u] = A1 * V * (1.0 - V)
    Fringe1_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="idler sinusoid Poisson fit exit 1 fringe",
    )

    grad[:] = 0.0
    grad[idx_a2] = V * A2
    grad[idx_u] = A2 * V * (1.0 - V)
    Fringe2_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="idler sinusoid Poisson fit exit 2 fringe",
    )

    return FitResult(
        x_fit=x_fit,
        y_fit1=y1_fit_corr,
        y_fit2=y2_fit_corr,
        V=float(V),
        x0=float(x0),
        V_sigma=float(V_sigma),
        P=float(P),
        A1=float(A1),
        A2=float(A2),
        A1_sigma=float(A1_sigma),
        A2_sigma=float(A2_sigma),
        A_sig_sigma=float(A_sig_sigma),
        P_sigma=float(P_sigma),
        total_fringe=float(total_fringe),
        total_fringe_sigma=float(total_fringe_sigma),
        Fringe1_sigma=float(Fringe1_sigma),
        Fringe2_sigma=float(Fringe2_sigma),
    )


def fit_weighted_sinusoid(
    xs_V, y1_vals, y2_vals, var1_vals, var2_vals, P_fixed=None
) -> FitResult:
    P_fixed = _validate_optional_fixed_period(P_fixed, context="coincidence fit")
    x = np.asarray(xs_V, dtype=float)
    y1 = np.asarray(y1_vals, dtype=float)
    y2 = np.asarray(y2_vals, dtype=float)
    var1 = np.asarray(var1_vals, dtype=float)
    var2 = np.asarray(var2_vals, dtype=float)

    if x.ndim != 1 or y1.ndim != 1 or y2.ndim != 1:
        raise ValueError("coincidence fit coordinates and values must be one-dimensional")
    if x.size < 4:
        raise ValueError("coincidence fit requires at least four points")
    if y1.size != x.size or y2.size != x.size:
        raise ValueError(
            "coincidence fit coordinates and measurement arrays have inconsistent lengths"
        )
    if not (np.isfinite(x).all() and np.isfinite(y1).all() and np.isfinite(y2).all()):
        raise ValueError("coincidence fit coordinates and measurements must be finite")

    var1 = validate_positive_variances(
        var1,
        expected_size=x.size,
        context="exit 1 coincidence variances",
    )
    var2 = validate_positive_variances(
        var2,
        expected_size=x.size,
        context="exit 2 coincidence variances",
    )
    w1 = 1.0 / var1
    w2 = 1.0 / var2
    sw1 = np.sqrt(w1)
    sw2 = np.sqrt(w2)

    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        P0 = float(P_fixed)
        w = 2.0 * math.pi / P0
        X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
        Xw1 = X * sw1[:, None]
        Xw2 = X * sw2[:, None]
        y1w = y1 * sw1
        y2w = y2 * sw2
        c1, _, _, _ = np.linalg.lstsq(Xw1, y1w, rcond=None)
        c2, _, _, _ = np.linalg.lstsq(Xw2, y2w, rcond=None)
        A1_0, C1_0, S1_0 = c1
        A2_0, C2_0, S2_0 = c2
    else:
        P_grid = np.linspace(P_FIT_MIN, P_FIT_MAX, 31)
        best = None
        for P0 in P_grid:
            w = 2.0 * math.pi / P0
            X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
            Xw1 = X * sw1[:, None]
            Xw2 = X * sw2[:, None]
            y1w = y1 * sw1
            y2w = y2 * sw2
            c1, _, _, _ = np.linalg.lstsq(Xw1, y1w, rcond=None)
            c2, _, _, _ = np.linalg.lstsq(Xw2, y2w, rcond=None)
            yhat1 = X @ c1
            yhat2 = X @ c2
            sse = float(np.sum(w1 * (y1 - yhat1) ** 2) + np.sum(w2 * (y2 - yhat2) ** 2))
            if (best is None) or (sse < best[0]):
                best = (sse, c1, c2, P0)
        _, (A1_0, C1_0, S1_0), (A2_0, C2_0, S2_0), P0 = best

    B1_0 = float(math.hypot(C1_0, S1_0))
    B2_0 = float(math.hypot(C2_0, S2_0))
    w0 = 2.0 * math.pi / P0
    x0_0 = float(math.atan2(S1_0, C1_0) / w0)

    A1_sig0 = max(float(A1_0), 1e-12)
    A2_sig0 = max(float(A2_0), 1e-12)
    v1 = B1_0 / A1_sig0 if A1_sig0 > 0 else 0.0
    v2 = B2_0 / A2_sig0 if A2_sig0 > 0 else 0.0
    V0 = float(np.clip(np.median([v1, v2]), 0.0, 0.95))

    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        p0 = [math.log(A1_sig0), math.log(A2_sig0), logit(V0), x0_0]
    else:
        p0 = [math.log(A1_sig0), math.log(A2_sig0), logit(V0), x0_0, math.log(P0)]

    def nll(p):
        if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
            a1, a2, u, x0 = p
            P = float(P_fixed)
        else:
            a1, a2, u, x0, g = p
            P = float(np.exp(g))
        A1 = float(np.exp(a1))
        A2 = float(np.exp(a2))
        V = float(expit(u))
        wx = 2.0 * math.pi * (x - x0) / P
        m1 = A1 * (1.0 + V * np.cos(wx))
        m2 = A2 * (1.0 - V * np.cos(wx))
        return 0.5 * float(np.sum(w1 * (y1 - m1) ** 2) + np.sum(w2 * (y2 - m2) ** 2))

    bounds = (
        None
        if (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0)
        else [
            (None, None),
            (None, None),
            (None, None),
            (None, None),
            (math.log(P_FIT_MIN), math.log(P_FIT_MAX)),
        ]
    )

    res = (
        minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxls": 100},
        )
        if bounds is not None
        else minimize(nll, p0, method="L-BFGS-B", options={"maxls": 100})
    )
    params = _require_successful_optimizer_result(
        res,
        expected_size=len(p0),
        context="weighted coincidence sinusoid fit",
    )
    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        a1, a2, u, x0 = params
        P = float(P_fixed)
    else:
        a1, a2, u, x0, g = params
        P = float(np.exp(g))
    A1 = float(np.exp(a1))
    A2 = float(np.exp(a2))
    V = float(expit(u))
    _require_finite_fit_parameters(
        {"A1": A1, "A2": A2, "V": V, "x0": float(x0), "P": P},
        positive=("A1", "A2", "P"),
        unit_interval=("V",),
        context="weighted coincidence sinusoid fit",
    )

    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    wx_fit = 2.0 * math.pi * (x_fit - x0) / P
    y1_fit = A1 * (1.0 + V * np.cos(wx_fit))
    y2_fit = A2 * (1.0 - V * np.cos(wx_fit))

    n = x.size
    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0:
        m = 4
        idx_a1, idx_a2, idx_u, idx_x0 = 0, 1, 2, 3
        idx_g = None
    else:
        m = 5
        idx_a1, idx_a2, idx_u, idx_x0, idx_g = 0, 1, 2, 3, 4

    Jw = np.zeros((2 * n, m), dtype=float)
    two_pi_over_P = 2.0 * math.pi / P
    for i in range(n):
        wx = 2.0 * math.pi * (x[i] - x0) / P
        c = math.cos(wx)
        s = math.sin(wx)

        r1 = np.zeros(m, dtype=float)
        r1[idx_a1] = A1 * (1.0 + V * c)
        r1[idx_u] = A1 * (V * (1.0 - V)) * c
        r1[idx_x0] = A1 * V * two_pi_over_P * s
        if idx_g is not None:
            r1[idx_g] = A1 * V * (2.0 * math.pi * (x[i] - x0) / P) * s
        Jw[2 * i, :] = r1 * sw1[i]

        r2 = np.zeros(m, dtype=float)
        r2[idx_a2] = A2 * (1.0 - V * c)
        r2[idx_u] = -A2 * (V * (1.0 - V)) * c
        r2[idx_x0] = -A2 * V * two_pi_over_P * s
        if idx_g is not None:
            r2[idx_g] = -A2 * V * (2.0 * math.pi * (x[i] - x0) / P) * s
        Jw[2 * i + 1, :] = r2 * sw2[i]

    Cov = _covariance_from_weighted_jacobian(
        Jw,
        context="weighted coincidence sinusoid fit",
    )

    wx = 2.0 * math.pi * (x - x0) / P
    m1 = A1 * (1.0 + V * np.cos(wx))
    m2 = A2 * (1.0 - V * np.cos(wx))
    r1w = (y1 - m1) * sw1
    r2w = (y2 - m2) * sw2
    SSE_w = float(np.sum(r1w * r1w) + np.sum(r2w * r2w))
    df = int(2 * n - m)
    # Propagated counting variances set the uncertainty floor; residual
    # overdispersion may enlarge, but never deflate, the covariance.
    Cov = _conservative_covariance_scale(SSE_w, df) * Cov

    var_a1 = float(Cov[idx_a1, idx_a1])
    var_a2 = float(Cov[idx_a2, idx_a2])
    cov_a1a2 = float(Cov[idx_a1, idx_a2])
    var_u = float(Cov[idx_u, idx_u])

    V_sigma = V * (1.0 - V) * _standard_deviation_from_variance(
        var_u,
        context="weighted coincidence sinusoid fit visibility",
    )
    A1_sigma = A1 * _standard_deviation_from_variance(
        var_a1,
        context="weighted coincidence sinusoid fit exit 1 amplitude",
    )
    A2_sigma = A2 * _standard_deviation_from_variance(
        var_a2,
        context="weighted coincidence sinusoid fit exit 2 amplitude",
    )

    g1 = 0.5 * A1
    g2 = 0.5 * A2
    var_A_sig = g1 * g1 * var_a1 + g2 * g2 * var_a2 + 2.0 * g1 * g2 * cov_a1a2
    A_sig_sigma = _standard_deviation_from_variance(
        var_A_sig,
        context="weighted coincidence sinusoid fit mean amplitude",
    )

    if idx_g is not None:
        var_g = float(Cov[idx_g, idx_g])
        P_sigma = P * _standard_deviation_from_variance(
            var_g,
            context="weighted coincidence sinusoid fit period",
        )
    else:
        P_sigma = float("nan")

    total_fringe = (A1 + A2) * V
    grad = np.zeros(Cov.shape[0], dtype=float)
    grad[idx_a1] = A1 * V
    grad[idx_a2] = A2 * V
    grad[idx_u] = (A1 + A2) * V * (1.0 - V)
    total_fringe_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="weighted coincidence sinusoid fit total fringe",
    )

    grad = np.zeros(Cov.shape[0], dtype=float)
    grad[idx_a1] = V * A1
    grad[idx_u] = A1 * V * (1.0 - V)
    Fringe1_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="weighted coincidence sinusoid fit exit 1 fringe",
    )

    grad[:] = 0.0
    grad[idx_a2] = V * A2
    grad[idx_u] = A2 * V * (1.0 - V)
    Fringe2_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="weighted coincidence sinusoid fit exit 2 fringe",
    )

    return FitResult(
        x_fit=x_fit,
        y_fit1=y1_fit,
        y_fit2=y2_fit,
        V=float(V),
        x0=float(x0),
        V_sigma=float(V_sigma),
        P=float(P),
        A1=float(A1),
        A2=float(A2),
        A1_sigma=float(A1_sigma),
        A2_sigma=float(A2_sigma),
        A_sig_sigma=float(A_sig_sigma),
        P_sigma=float(P_sigma),
        total_fringe=float(total_fringe),
        total_fringe_sigma=float(total_fringe_sigma),
        Fringe1_sigma=float(Fringe1_sigma),
        Fringe2_sigma=float(Fringe2_sigma),
    )


def fit_scan(
    data: MZIScanData,
    channel: str,
    period: Optional[float],
    dark: DarkModel,
    acq_dur: float,
) -> ScanFits:
    channel = str(channel).upper()
    if channel not in {"C", "I", "IC"}:
        raise ValueError("scan fit channel must be C, I, or IC")
    period_for_fit = _validate_optional_fixed_period(period, context="scan fit")
    fit_c = None
    fit_i = None

    if channel in ("C", "IC"):
        fit_c = fit_weighted_sinusoid(
            data.xs,
            data.rc1,
            data.rc2,
            data.var_rc1,
            data.var_rc2,
            P_fixed=period_for_fit,
        )

    P_for_I = period_for_fit
    if P_for_I is None and fit_c is not None:
        P_for_I = float(fit_c.P)

    if channel in ("I", "IC"):
        fit_i = fit_idler_sinusoid_poisson_joint(
            data.xs,
            data.n_i1,
            data.n_i2,
            acq_dur,
            dark_counts_i1=dark.Ni,
            dark_counts_i2=dark.Ni2,
            dark_duration_s=dark.T,
            P_fixed=P_for_I,
        )
    return ScanFits(coincidence=fit_c, singles=fit_i)


def _eval_idler_model(x, A1, A2, V, x0, P):
    x = np.asarray(x, dtype=float)
    wx = 2.0 * np.pi * (x - float(x0)) / float(P)
    y1 = float(A1) * (1.0 + float(V) * np.cos(wx))
    y2 = float(A2) * (1.0 - float(V) * np.cos(wx))
    return y1, y2
