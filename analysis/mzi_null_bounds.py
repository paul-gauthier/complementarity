#!/usr/bin/env python
"""
Compute launch-null uncertainty bounds from the same MZI records used by
mzi_plot_alternating_by_pass.py.

Primary output:
  B95_known_phase_cps       one-sided 95% upper bound for a launch-only
                            contrast fringe with the phase predicted by the
                            MZI phase reference.
  B95_any_phase_cps         95% upper bound for an arbitrary-phase launch-only
                            contrast fringe, using the 2D confidence ellipse.
  eta95_*                   B95 / (T_inf * R_pm), conditional on supplied or
                            fitted T_inf and R_pm.
  Tinf_required_for_eta1_*  B95 / R_pm, the survival probability below which
                            full restoration would evade the bound.

The estimator uses index-aligned launch/control records but keeps their actual
voltages in the fit. For each pass k, with z = R1 - R2,

  z_control,k,i = a_control,k + C0 cos(theta_control,k,i) + S0 sin(theta_control,k,i)

  z_launch,k,i = a_launch,k
                 + (C0 + CL) cos(theta_launch,k,i)
                 + (S0 + SL) sin(theta_launch,k,i)

where theta is defined by a phase-reference joint fit, normally the
high-visibility erased coincidence data. The nuisance quadratures C0,S0 absorb
the ordinary common residual singles fringe, including the small leakage from
launch/control voltage mismatches. The launch-only amplitude is
B_launch = sqrt(CL^2 + SL^2). If the alternative predicts the ordinary MZI
phase, use the known-phase bound on CL. If the phase could be arbitrary, use
the 2D bound on B_launch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, norm

from .mzi_analysis import (
    fit_scan,
    full_restoration_circle_distance,
    scan_from_records,
    validate_positive_variances,
)
from .mzi_joint import (
    JointFitResult,
    aligned_record_truncation,
    fit_joint_coincidences,
    fit_joint_singles,
    fringe_height_and_sigma,
    validate_total_fringe_values,
)
from .mzi_io import (
    assert_constant_point_duration,
    load_dark,
    load_plan_length,
    load_points,
    parse_condition_specs,
    split_by_plan_length,
)

DEFAULT_CL = 0.95


@dataclass
class ConditionBundle:
    name: str
    path: Path
    dark: Any
    records: List[Dict[str, Any]]
    pass_records: List[List[Dict[str, Any]]]
    pass_indices: List[int]
    data_passes: List[Any]
    T_list: List[float]


@dataclass
class JointFitSummary:
    condition: str
    channel: str
    fit: JointFitResult
    fringe1: float
    fringe1_sigma: float
    fringe2: float
    fringe2_sigma: float
    total_fringe: float
    total_fringe_sigma: float


@dataclass
class PairedFitResult:
    q_hat: np.ndarray
    cov_q: np.ndarray
    beta: np.ndarray
    cov_beta: np.ndarray
    chi2: float
    red_chi2: float
    df: int
    n_point_pairs: int
    n_observations: int
    n_passes: int
    B_hat: float
    B95_known_phase: float
    B95_any_phase: float
    p_null_any_phase: float
    q_common_hat: np.ndarray
    cov_common: np.ndarray
    blocks: List[Dict[str, np.ndarray]]


@dataclass
class _LinearFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    chi2: float
    red_chi2: float
    df: int
    rank: int
    y_hat: np.ndarray
    resid: np.ndarray
    X: np.ndarray
    y: np.ndarray
    var: np.ndarray


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric diagnostics to JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _sqrt_nonnegative(v: float) -> float:
    return float(math.sqrt(max(0.0, float(v)))) if np.isfinite(v) else float("nan")


def _phase_values_by_pass(
    pass_indices: Sequence[int],
    phase_values: Sequence[float],
    *,
    context: str,
) -> Dict[int, float]:
    if len(pass_indices) != len(phase_values):
        raise ValueError(
            f"{context} pass indices and phase values have inconsistent lengths"
        )
    result: Dict[int, float] = {}
    for index, value in zip(pass_indices, phase_values, strict=True):
        pass_index = int(index)
        if pass_index in result:
            raise ValueError(f"{context} contains duplicate pass index {pass_index}")
        phase = float(value)
        if not np.isfinite(phase):
            raise ValueError(f"{context} phase values must be finite")
        result[pass_index] = phase
    return result


def _validate_positive_definite_covariance(
    covariance: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] <= 0:
        raise ValueError(f"{context} must be a nonempty square matrix")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{context} must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{context} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{context} eigenvalues must be finite")
    tolerance = (
        max(float(eigenvalues[-1]), 1.0)
        * np.finfo(float).eps
        * max(cov.shape[0], 1)
        * 100.0
    )
    if float(eigenvalues[0]) <= tolerance:
        raise ValueError(
            f"{context} is not identifiable "
            f"(minimum eigenvalue {float(eigenvalues[0]):.6g}, "
            f"tolerance {tolerance:.6g})"
        )
    return cov


def _fmt(v: Any, nd: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _channel_label(channel: str) -> str:
    ch = str(channel).upper()
    if ch == "I":
        return "Signal singles"
    if ch == "C":
        return "Signal+idler coincidences"
    return ch


def _load_condition_bundles(
    conditions: List[Tuple[str, Path]],
    *,
    max_passes: Optional[int],
    min_points: int,
    use_first_secs: Optional[float],
) -> Dict[str, ConditionBundle]:
    dark_by_cond: Dict[str, Any] = {}
    recs_by_cond: Dict[str, List[Dict[str, Any]]] = {}

    for cond_name, cond_dir in conditions:
        dark_by_cond[cond_name] = load_dark(cond_dir)
        recs_by_cond[cond_name] = load_points(cond_dir)

    if use_first_secs is not None and float(use_first_secs) > 0.0:
        recs_by_cond = aligned_record_truncation(recs_by_cond, float(use_first_secs))

    out: Dict[str, ConditionBundle] = {}
    for cond_name, cond_dir in conditions:
        recs = recs_by_cond.get(cond_name, [])
        pass_records = split_by_plan_length(recs, load_plan_length(cond_dir))

        if max_passes is not None and int(max_passes) > 0:
            pass_records = pass_records[: int(max_passes)]

        data_passes: List[Any] = []
        T_list: List[float] = []
        pass_indices: List[int] = []
        for pidx, recs_pass in enumerate(pass_records):
            assert_constant_point_duration(recs_pass, f"{cond_name} pass {pidx + 1}")
            data_pass, T_pass = scan_from_records(recs_pass, dark_by_cond[cond_name])
            n_points = len(data_pass.xs)
            if n_points < int(min_points):
                raise ValueError(
                    f"{cond_name} pass {pidx + 1} has {n_points} points; "
                    f"at least {int(min_points)} are required"
                )
            data_passes.append(data_pass)
            T_list.append(float(T_pass))
            pass_indices.append(int(pidx))

        out[cond_name] = ConditionBundle(
            name=cond_name,
            path=cond_dir,
            dark=dark_by_cond[cond_name],
            records=recs,
            pass_records=pass_records,
            pass_indices=pass_indices,
            data_passes=data_passes,
            T_list=T_list,
        )

    return out


def _joint_fit_condition(
    bundle: ConditionBundle,
    channel: str,
    *,
    period: Optional[float],
) -> JointFitResult:
    """Run the same style of joint fit used by mzi_plot_alternating_by_pass.py."""
    ch = str(channel).upper()
    if ch not in {"I", "C"}:
        raise ValueError("channel must be I or C")
    if not bundle.data_passes:
        raise ValueError(f"{bundle.name} has no data passes to fit")
    if not (
        len(bundle.data_passes) == len(bundle.T_list) == len(bundle.pass_indices)
    ):
        raise ValueError(
            f"{bundle.name} data passes, durations, and pass indices have inconsistent lengths"
        )

    init_A1: List[float] = []
    init_A2: List[float] = []
    init_V: List[float] = []
    init_P: List[float] = []
    init_x0: List[float] = []

    for pass_number, (dp, T_pass) in enumerate(
        zip(bundle.data_passes, bundle.T_list, strict=True),
        start=1,
    ):
        try:
            fits = fit_scan(
                dp,
                channel=ch,
                period=period,
                dark=bundle.dark,
                acq_dur=float(T_pass),
            )
        except (RuntimeError, ValueError) as exc:
            raise type(exc)(
                f"{bundle.name} {ch} pass {pass_number} independent seed fit failed: {exc}"
            ) from exc
        fit = fits.singles if ch == "I" else fits.coincidence
        values = np.asarray([fit.A1, fit.A2, fit.V, fit.P, fit.x0], dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"independent seed fit returned non-finite parameters for "
                f"{bundle.name} {ch} pass {pass_number}"
            )
        init_A1.append(float(fit.A1))
        init_A2.append(float(fit.A2))
        init_V.append(float(fit.V))
        init_P.append(float(fit.P))
        init_x0.append(float(fit.x0))

    x0_list: List[float] = []
    for x0 in init_x0:
        x0_list.append(float(x0))

    init = {
        "A1": float(np.median(init_A1)) if init_A1 else np.nan,
        "A2": float(np.median(init_A2)) if init_A2 else np.nan,
        "V": float(np.median(init_V)) if init_V else 0.5,
        "P": (
            float(period)
            if period is not None and np.isfinite(float(period))
            else (float(np.median(init_P)) if init_P else np.nan)
        ),
        "x0_list": x0_list,
    }

    if ch == "I":
        return fit_joint_singles(
            bundle.data_passes,
            bundle.T_list,
            bundle.dark,
            P_fixed=period,
            init=init,
        )
    return fit_joint_coincidences(bundle.data_passes, P_fixed=period, init=init)


def _summarize_joint_fit(
    condition: str, channel: str, fit: JointFitResult
) -> JointFitSummary:
    H1, H1s = fringe_height_and_sigma(fit, channel=1)
    H2, H2s = fringe_height_and_sigma(fit, channel=2)
    total, sig = validate_total_fringe_values(
        fit.total_fringe,
        fit.total_fringe_sigma,
        H1,
        H2,
        context=f"{condition} {str(channel).upper()} joint fit",
    )
    return JointFitSummary(
        condition=condition,
        channel=str(channel).upper(),
        fit=fit,
        fringe1=float(H1),
        fringe1_sigma=float(H1s),
        fringe2=float(H2),
        fringe2_sigma=float(H2s),
        total_fringe=float(total),
        total_fringe_sigma=float(sig),
    )


def _get_rate_array(dp: Any, exit_index: int) -> np.ndarray:
    """Return the canonical dark-subtracted singles rate for an MZI exit."""
    idx = int(exit_index)
    if idx not in {1, 2}:
        raise ValueError("exit_index must be 1 or 2")
    field = "ri1" if idx == 1 else "ri2"
    try:
        return np.asarray(getattr(dp, field), dtype=float)
    except AttributeError as exc:
        raise AttributeError(f"scan data is missing canonical field {field}") from exc


def _get_rate_var_array(dp: Any, T: float, dark: Any, exit_index: int) -> np.ndarray:
    """Return the canonical Poisson singles-rate variance for an MZI exit."""
    idx = int(exit_index)
    if idx not in {1, 2}:
        raise ValueError("exit_index must be 1 or 2")
    duration = float(T)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("point duration must be finite and positive")

    count_field = "n_i1" if idx == 1 else "n_i2"
    dark_count_field = "Ni" if idx == 1 else "Ni2"
    try:
        counts = np.asarray(getattr(dp, count_field), dtype=float)
    except AttributeError as exc:
        raise AttributeError(
            f"scan data is missing canonical field {count_field}"
        ) from exc
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(
            f"canonical field {count_field} must contain finite nonnegative counts"
        )

    try:
        dark_counts = float(getattr(dark, dark_count_field))
        dark_duration = float(getattr(dark, "T"))
    except AttributeError as exc:
        raise AttributeError(
            f"dark data must contain canonical fields {dark_count_field} and T"
        ) from exc
    if not np.isfinite(dark_counts) or dark_counts < 0.0:
        raise ValueError(
            f"dark field {dark_count_field} must be finite and nonnegative"
        )
    if not np.isfinite(dark_duration) or dark_duration <= 0.0:
        raise ValueError("dark duration must be finite and positive")

    var = counts / (duration**2) + dark_counts / (dark_duration**2)

    return validate_positive_variances(
        var,
        context=f"derived exit {idx} singles-rate variances",
    )


def _make_paired_blocks(
    launch: ConditionBundle,
    control: ConditionBundle,
    *,
    phase_x0_list: Sequence[float],
    phase_pass_indices: Sequence[int],
    known_phase_x0_list: Optional[Sequence[float]] = None,
    known_phase_pass_indices: Optional[Sequence[int]] = None,
    period: float,
    max_pairs: Optional[int] = None,
    x_match_tol: float = 1e-3,
) -> List[Dict[str, np.ndarray]]:
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError("A finite positive period is required for paired quadrature fitting")

    x_tol = float(x_match_tol)
    if not (np.isfinite(x_tol) and x_tol >= 0.0):
        raise ValueError("--x-match-tol must be finite and nonnegative")

    if known_phase_x0_list is None:
        known_phase_x0_list = phase_x0_list
    if known_phase_pass_indices is None:
        known_phase_pass_indices = phase_pass_indices

    phase_x0_by_pass = _phase_values_by_pass(
        phase_pass_indices,
        phase_x0_list,
        context="phase-reference fit",
    )
    known_phase_x0_by_pass = _phase_values_by_pass(
        known_phase_pass_indices,
        known_phase_x0_list,
        context="known-phase-reference fit",
    )

    for bundle, label in ((launch, "launch"), (control, "control")):
        if not (
            len(bundle.data_passes) == len(bundle.T_list) == len(bundle.pass_indices)
        ):
            raise ValueError(
                f"{label} data passes, durations, and pass indices have inconsistent lengths"
            )
        if not bundle.data_passes:
            raise ValueError(f"{label} condition has no data passes")
        if len(set(map(int, bundle.pass_indices))) != len(bundle.pass_indices):
            raise ValueError(f"{label} condition contains duplicate pass indices")
        durations = np.asarray(bundle.T_list, dtype=float)
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError(f"{label} pass durations must be finite and positive")

    launch_by_pass = {
        int(pass_idx): idx for idx, pass_idx in enumerate(launch.pass_indices)
    }
    control_by_pass = {
        int(pass_idx): idx for idx, pass_idx in enumerate(control.pass_indices)
    }
    expected_passes = set(launch_by_pass)
    pass_sets = {
        "control": set(control_by_pass),
        "phase reference": set(phase_x0_by_pass),
        "known-phase reference": set(known_phase_x0_by_pass),
    }
    for label, indices in pass_sets.items():
        if indices != expected_passes:
            missing = sorted(expected_passes - indices)
            extra = sorted(indices - expected_passes)
            raise ValueError(
                f"{label} pass indices do not match launch pass indices "
                f"(missing={missing}, extra={extra})"
            )

    common_pass_indices = [int(pass_idx) for pass_idx in launch.pass_indices]
    if max_pairs is not None and int(max_pairs) > 0:
        common_pass_indices = common_pass_indices[: int(max_pairs)]
    n_passes = len(common_pass_indices)
    if n_passes <= 0:
        raise ValueError("No aligned passes are available for paired launch-control fitting")

    blocks: List[Dict[str, np.ndarray]] = []
    for pass_idx in common_pass_indices:
        launch_idx = launch_by_pass[pass_idx]
        control_idx = control_by_pass[pass_idx]
        dp_l = launch.data_passes[launch_idx]
        dp_c = control.data_passes[control_idx]
        T_l = float(launch.T_list[launch_idx])
        T_c = float(control.T_list[control_idx])

        x_l0 = np.asarray(getattr(dp_l, "xs", []), dtype=float)
        x_c0 = np.asarray(getattr(dp_c, "xs", []), dtype=float)

        l1_0 = _get_rate_array(dp_l, 1)
        l2_0 = _get_rate_array(dp_l, 2)
        c1_0 = _get_rate_array(dp_c, 1)
        c2_0 = _get_rate_array(dp_c, 2)

        vl1_0 = _get_rate_var_array(dp_l, T_l, launch.dark, 1)
        vl2_0 = _get_rate_var_array(dp_l, T_l, launch.dark, 2)
        vc1_0 = _get_rate_var_array(dp_c, T_c, control.dark, 1)
        vc2_0 = _get_rate_var_array(dp_c, T_c, control.dark, 2)

        point_arrays = {
            "launch coordinates": x_l0,
            "control coordinates": x_c0,
            "launch exit 1 rates": l1_0,
            "launch exit 2 rates": l2_0,
            "control exit 1 rates": c1_0,
            "control exit 2 rates": c2_0,
            "launch exit 1 variances": vl1_0,
            "launch exit 2 variances": vl2_0,
            "control exit 1 variances": vc1_0,
            "control exit 2 variances": vc2_0,
        }
        if any(values.ndim != 1 for values in point_arrays.values()):
            raise ValueError(f"pass {pass_idx + 1} point arrays must be one-dimensional")
        point_lengths = {name: values.size for name, values in point_arrays.items()}
        if len(set(point_lengths.values())) != 1:
            raise ValueError(
                f"pass {pass_idx + 1} point arrays have inconsistent lengths: "
                f"{point_lengths}"
            )
        n = x_l0.size
        if n <= 0:
            raise ValueError(f"pass {pass_idx + 1} point arrays must not be empty")
        if not all(np.all(np.isfinite(values)) for values in point_arrays.values()):
            raise ValueError(f"pass {pass_idx + 1} point arrays must be finite")

        x_l = x_l0
        x_c = x_c0

        dx = np.abs(x_l - x_c)
        max_dx = float(np.max(dx))
        median_dx = float(np.median(dx))
        frac_period = max_dx / float(period)
        if max_dx > x_tol:
            raise ValueError(
                f"launch/control x grids differ in pass {pass_idx + 1}: n={n}, "
                f"max(|dx|)={max_dx:.6g} V, median(|dx|)={median_dx:.6g} V, "
                f"max(|dx|)/period={frac_period:.6g}; "
                f"exceeds --x-match-tol={x_tol:.6g} V"
            )
        if max_dx > 1e-6:
            print(
                (
                    f"Grid diagnostic: pass {pass_idx + 1}, n={n}, "
                    f"max(|dx|)={max_dx:.6g} V, median(|dx|)={median_dx:.6g} V, "
                    f"max(|dx|)/period={frac_period:.6g}; "
                    "fitting actual launch/control voltages."
                ),
                file=sys.stderr,
            )

        l1 = np.asarray(l1_0, dtype=float)
        l2 = np.asarray(l2_0, dtype=float)
        c1 = np.asarray(c1_0, dtype=float)
        c2 = np.asarray(c2_0, dtype=float)

        vl1 = np.asarray(vl1_0, dtype=float)
        vl2 = np.asarray(vl2_0, dtype=float)
        vc1 = np.asarray(vc1_0, dtype=float)
        vc2 = np.asarray(vc2_0, dtype=float)

        z_l = l1 - l2
        z_c = c1 - c2
        var_l = validate_positive_variances(
            vl1 + vl2,
            expected_size=n,
            context=f"pass {pass_idx + 1} launch contrast variances",
        )
        var_c = validate_positive_variances(
            vc1 + vc2,
            expected_size=n,
            context=f"pass {pass_idx + 1} control contrast variances",
        )

        x0 = phase_x0_by_pass[pass_idx]
        known_x0 = known_phase_x0_by_pass[pass_idx]
        theta_l = 2.0 * np.pi * (x_l - x0) / float(period)
        theta_c = 2.0 * np.pi * (x_c - x0) / float(period)
        theta_known_l = 2.0 * np.pi * (x_l - known_x0) / float(period)

        blocks.append(
            {
                "pass_index": np.full(n, pass_idx, dtype=int),
                "x_launch": x_l,
                "x_control": x_c,
                "dx": x_l - x_c,
                "theta_launch": theta_l,
                "theta_control": theta_c,
                "theta_known_launch": theta_known_l,
                "z_launch": z_l,
                "z_control": z_c,
                "var_launch": var_l,
                "var_control": var_c,
            }
        )

    if not blocks:
        raise ValueError("No paired points are available")
    return blocks


def _replace_known_phase_in_blocks(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    known_phase_x0_list: Sequence[float],
    known_phase_pass_indices: Sequence[int],
    period: float,
) -> List[Dict[str, np.ndarray]]:
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError("A finite positive period is required for known-phase block construction")

    known_phase_x0_by_pass = _phase_values_by_pass(
        known_phase_pass_indices,
        known_phase_x0_list,
        context="known-phase-reference fit",
    )

    out: List[Dict[str, np.ndarray]] = []
    block_pass_indices: List[int] = []
    for block in blocks:
        pass_indices = np.asarray(block.get("pass_index", []), dtype=int)
        if pass_indices.ndim != 1 or pass_indices.size <= 0:
            raise ValueError("paired block pass_index must be a nonempty one-dimensional array")
        pass_idx = int(pass_indices[0])
        if not np.all(pass_indices == pass_idx):
            raise ValueError("paired block contains more than one pass index")
        block_pass_indices.append(pass_idx)
        known_x0 = known_phase_x0_by_pass.get(pass_idx)
        if known_x0 is None:
            raise ValueError(
                f"known-phase reference is missing paired pass index {pass_idx}"
            )
        x_launch = np.asarray(block.get("x_launch", []), dtype=float)
        if x_launch.ndim != 1 or x_launch.size != pass_indices.size:
            raise ValueError(
                f"paired pass {pass_idx + 1} launch coordinates and pass indices "
                "have inconsistent lengths"
            )
        if not np.all(np.isfinite(x_launch)):
            raise ValueError(f"paired pass {pass_idx + 1} launch coordinates must be finite")
        block_known = dict(block)
        block_known["theta_known_launch"] = 2.0 * np.pi * (x_launch - known_x0) / float(period)
        out.append(block_known)

    if not out:
        raise ValueError("No known-phase-aligned paired blocks are available")
    if len(set(block_pass_indices)) != len(block_pass_indices):
        raise ValueError("paired blocks contain duplicate pass indices")
    return out


def _build_actual_voltage_design(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    include_launch_terms: bool = True,
    launch_phase_key: str = "theta_known_launch",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, Optional[int]]:
    """
    Build stacked WLS design for

      z_control = a_control,k + C0 cos(theta_control) + S0 sin(theta_control)

      z_launch = a_launch,k + C0 cos(theta_launch) + S0 sin(theta_launch)
                            + CL cos(theta_launch_term) + SL sin(theta_launch_term)

    where theta_launch_term is selected by launch_phase_key, e.g.
    "theta_launch" for arbitrary/reference-phase bounds or
    "theta_known_launch" for known-phase bounds.

    Parameters are ordered as:

      [a_control,0 ... a_control,K-1,
       a_launch,0 ... a_launch,K-1,
       C0, S0, CL, SL]

    If include_launch_terms=False, the final CL, SL columns are omitted.
    """
    K = len(blocks)
    if K <= 0:
        raise ValueError("At least one block is required")

    common_offset = 2 * K
    launch_offset = common_offset + 2 if include_launch_terms else None
    n_params = common_offset + 2 + (2 if include_launch_terms else 0)

    launch_term_key = str(launch_phase_key)
    if include_launch_terms and not launch_term_key:
        raise ValueError("launch_phase_key must be non-empty when launch terms are included")

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    var_parts: List[np.ndarray] = []
    n_pairs_total = 0

    required = [
        "theta_launch",
        "theta_control",
        "z_launch",
        "z_control",
        "var_launch",
        "var_control",
    ]

    for j, block in enumerate(blocks):
        for key in required:
            if key not in block:
                raise KeyError(
                    f"Block is missing '{key}'. Rebuild blocks with actual-voltage "
                    "_make_paired_blocks()."
                )

        theta_l = np.asarray(block["theta_launch"], dtype=float)
        theta_c = np.asarray(block["theta_control"], dtype=float)
        if include_launch_terms:
            if launch_term_key not in block:
                raise KeyError(
                    f"Block is missing '{launch_term_key}'. Rebuild blocks with actual-voltage "
                    "_make_paired_blocks()."
                )
            theta_launch_term = np.asarray(block[launch_term_key], dtype=float)
        else:
            theta_launch_term = theta_l
        z_l = np.asarray(block["z_launch"], dtype=float)
        z_c = np.asarray(block["z_control"], dtype=float)
        var_l = validate_positive_variances(
            np.asarray(block["var_launch"], dtype=float),
            context=f"block {j + 1} launch variances",
        )
        var_c = validate_positive_variances(
            np.asarray(block["var_control"], dtype=float),
            context=f"block {j + 1} control variances",
        )

        arrays = {
            "theta_launch": theta_l,
            "theta_control": theta_c,
            launch_term_key if include_launch_terms else "theta_launch_term": theta_launch_term,
            "z_launch": z_l,
            "z_control": z_c,
            "var_launch": var_l,
            "var_control": var_c,
        }
        if any(values.ndim != 1 for values in arrays.values()):
            raise ValueError(f"block {j + 1} arrays must be one-dimensional")
        lengths = {name: values.size for name, values in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"block {j + 1} arrays have inconsistent lengths: {lengths}")
        n_f = theta_l.size
        if n_f <= 0:
            raise ValueError(f"block {j + 1} arrays must not be empty")
        if not all(np.all(np.isfinite(values)) for values in arrays.values()):
            raise ValueError(f"block {j + 1} arrays must be finite")

        n_pairs_total += int(n_f)

        cos_l = np.cos(theta_l)
        sin_l = np.sin(theta_l)
        cos_launch_term = np.cos(theta_launch_term)
        sin_launch_term = np.sin(theta_launch_term)
        cos_c = np.cos(theta_c)
        sin_c = np.sin(theta_c)

        Xc = np.zeros((n_f, n_params), dtype=float)
        Xc[:, j] = 1.0
        Xc[:, common_offset] = cos_c
        Xc[:, common_offset + 1] = sin_c

        Xl = np.zeros((n_f, n_params), dtype=float)
        Xl[:, K + j] = 1.0
        Xl[:, common_offset] = cos_l
        Xl[:, common_offset + 1] = sin_l

        if include_launch_terms:
            if launch_offset is None:
                raise RuntimeError("Internal error: launch offset was not set")
            Xl[:, launch_offset] = cos_launch_term
            Xl[:, launch_offset + 1] = sin_launch_term

        X_parts.extend([Xc, Xl])
        y_parts.extend([z_c, z_l])
        var_parts.extend([var_c, var_l])

    if not X_parts:
        raise ValueError("No finite actual-voltage observations are available for WLS fit")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    var = validate_positive_variances(
        np.concatenate(var_parts),
        context="combined launch-control variances",
    )

    return X, y, var, n_pairs_total, K, common_offset, launch_offset


def _solve_wls(
    X: np.ndarray,
    y: np.ndarray,
    var: np.ndarray,
    *,
    covariance_scale: str,
) -> _LinearFit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    var = validate_positive_variances(
        np.asarray(var, dtype=float),
        context="WLS variances",
    )

    if X.ndim != 2:
        raise ValueError("WLS design matrix must be 2D")
    if y.ndim != 1:
        raise ValueError("WLS observations must be one-dimensional")
    if X.shape[0] != y.size or y.size != var.size:
        raise ValueError("WLS design, observations, and variances have inconsistent lengths")
    if y.size <= 0 or X.shape[1] <= 0:
        raise ValueError("WLS design and observations must not be empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("WLS design matrix must be finite")
    if not np.all(np.isfinite(y)):
        raise ValueError("WLS observations must be finite")

    w = 1.0 / var
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    M = Xw.T @ Xw
    rhs = Xw.T @ yw
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(rhs)):
        raise ValueError("WLS normal equations must be finite")

    rank = int(np.linalg.matrix_rank(Xw))
    expected_rank = int(X.shape[1])
    if rank != expected_rank:
        raise ValueError(
            f"WLS design matrix is rank deficient (rank {rank}, expected {expected_rank})"
        )
    df = int(y.size - expected_rank)
    if df <= 0:
        raise ValueError(
            f"WLS fit requires positive residual degrees of freedom; got {df}"
        )

    try:
        cov_inverse = np.linalg.inv(M)
    except np.linalg.LinAlgError as exc:
        raise ValueError("WLS covariance is not identifiable") from exc
    cov_unscaled = _validate_positive_definite_covariance(
        cov_inverse,
        context="WLS covariance",
    )

    beta = cov_inverse @ rhs
    y_hat = X @ beta
    resid = y - y_hat
    if not (
        np.all(np.isfinite(beta))
        and np.all(np.isfinite(y_hat))
        and np.all(np.isfinite(resid))
    ):
        raise ValueError("WLS fitted parameters and residuals must be finite")

    chi2_val = float(np.sum(w * resid * resid))
    red = float(chi2_val / df)
    if not np.isfinite(chi2_val) or chi2_val < 0.0:
        raise ValueError("WLS chi-square must be finite and nonnegative")
    if not np.isfinite(red) or red < 0.0:
        raise ValueError("WLS reduced chi-square must be finite and nonnegative")

    cov_mode = str(covariance_scale).lower().strip()
    if cov_mode in {"residual", "chi2", "redchi2"}:
        scale = red if np.isfinite(red) and red > 0.0 else 1.0
    elif cov_mode in {"max1", "conservative", "default"}:
        scale = max(1.0, red) if np.isfinite(red) and red > 0.0 else 1.0
    elif cov_mode in {"none", "poisson", "raw"}:
        scale = 1.0
    else:
        raise ValueError("--covariance-scale must be max1, residual, or none")

    cov_scaled = np.asarray(cov_unscaled * scale, dtype=float)
    cov_scaled = _validate_positive_definite_covariance(
        cov_scaled,
        context="scaled WLS covariance",
    )

    return _LinearFit(
        beta=np.asarray(beta, dtype=float),
        cov_beta=cov_scaled,
        chi2=chi2_val,
        red_chi2=red,
        df=df,
        rank=rank,
        y_hat=np.asarray(y_hat, dtype=float),
        resid=np.asarray(resid, dtype=float),
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        var=np.asarray(var, dtype=float),
    )


def _fit_blocks(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    cl: float,
    covariance_scale: str = "max1",
    launch_phase_key: str = "theta_known_launch",
) -> PairedFitResult:
    """
    Weighted stacked actual-voltage fit with one launch and one control intercept
    per pass, common residual-fringe nuisance terms C0/S0, and launch-only
    terms CL/SL.
    """
    X, y, var, n_pairs, K, common_offset, launch_offset = _build_actual_voltage_design(
        blocks,
        include_launch_terms=True,
        launch_phase_key=launch_phase_key,
    )
    if launch_offset is None:
        raise RuntimeError("Internal error: launch terms were not included")

    fit = _solve_wls(X, y, var, covariance_scale=covariance_scale)

    launch_end = launch_offset + 2
    common_end = common_offset + 2

    q_hat = np.asarray(fit.beta[launch_offset:launch_end], dtype=float)
    cov_q = np.asarray(
        fit.cov_beta[launch_offset:launch_end, launch_offset:launch_end],
        dtype=float,
    )

    q_common_hat = np.asarray(fit.beta[common_offset:common_end], dtype=float)
    cov_common = np.asarray(
        fit.cov_beta[common_offset:common_end, common_offset:common_end],
        dtype=float,
    )

    B_hat = float(np.linalg.norm(q_hat))
    B95_known = _known_phase_upper(q_hat[0], cov_q[0, 0], cl)
    B95_any = _ellipse_radius_upper(q_hat, cov_q, cl)
    p_null = _quadrature_null_pvalue(q_hat, cov_q)

    return PairedFitResult(
        q_hat=q_hat,
        cov_q=cov_q,
        beta=np.asarray(fit.beta, dtype=float),
        cov_beta=np.asarray(fit.cov_beta, dtype=float),
        chi2=fit.chi2,
        red_chi2=fit.red_chi2,
        df=fit.df,
        n_point_pairs=int(n_pairs),
        n_observations=int(fit.y.size),
        n_passes=K,
        B_hat=B_hat,
        B95_known_phase=float(B95_known),
        B95_any_phase=float(B95_any),
        p_null_any_phase=float(p_null),
        q_common_hat=q_common_hat,
        cov_common=cov_common,
        blocks=list(blocks),
    )


def _known_phase_upper(c_hat: float, var_c: float, cl: float) -> float:
    sigma = _sqrt_nonnegative(var_c)
    z = float(norm.ppf(float(cl)))
    return float(max(0.0, float(c_hat)) + z * sigma)


def _ellipse_radius_upper(q_hat: np.ndarray, cov_q: np.ndarray, cl: float) -> float:
    q_hat = np.asarray(q_hat, dtype=float).reshape(2)
    if not np.all(np.isfinite(q_hat)):
        raise ValueError("quadrature estimate must be finite")
    cov_q = _validate_positive_definite_covariance(
        np.asarray(cov_q, dtype=float).reshape(2, 2),
        context="quadrature covariance",
    )
    vals, vecs = np.linalg.eigh(cov_q)
    c = float(chi2.ppf(float(cl), df=2))
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"Invalid chi-square quantile for confidence={cl}")
    r = math.sqrt(c)

    # Maximize the radius by sampling the two-dimensional ellipse boundary.
    n_grid = 20000
    theta = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    delta = (vecs @ (np.sqrt(vals)[:, None] * r * unit)).T
    radius = np.linalg.norm(q_hat[None, :] + delta, axis=1)
    return float(np.max(radius))


def _quadrature_null_pvalue(q_hat: np.ndarray, cov_q: np.ndarray) -> float:
    q_hat = np.asarray(q_hat, dtype=float).reshape(2)
    if not np.all(np.isfinite(q_hat)):
        raise ValueError("quadrature estimate must be finite")
    cov_q = _validate_positive_definite_covariance(
        np.asarray(cov_q, dtype=float).reshape(2, 2),
        context="quadrature covariance",
    )
    try:
        inv = np.linalg.inv(cov_q)
    except np.linalg.LinAlgError as exc:
        raise ValueError("quadrature covariance is not identifiable") from exc
    stat = float(q_hat.T @ inv @ q_hat)
    if not np.isfinite(stat) or stat < 0.0:
        raise ValueError("quadrature null statistic must be finite and nonnegative")
    return float(chi2.sf(stat, df=2))


def _bootstrap_blocks(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    n_bootstrap: int,
    cl: float,
    covariance_scale: str,
    launch_phase_key: str = "theta_known_launch",
    seed: Optional[int],
) -> Dict[str, float]:
    requested = int(n_bootstrap)
    if requested < 0:
        raise ValueError("bootstrap count must be nonnegative")
    if requested == 0:
        return {}
    K = len(blocks)
    if K <= 1:
        raise ValueError("bootstrap requires at least two paired blocks")
    rng = np.random.default_rng(seed)
    B_vals: List[float] = []
    C_vals: List[float] = []
    S_vals: List[float] = []
    for _ in range(requested):
        idx = rng.integers(0, K, size=K)
        selected = [blocks[int(i)] for i in idx]
        res = _fit_blocks(
            selected,
            cl=cl,
            covariance_scale=covariance_scale,
            launch_phase_key=launch_phase_key,
        )
        values = np.asarray([res.B_hat, res.q_hat[0], res.q_hat[1]], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("bootstrap draw returned non-finite statistics")
        B_vals.append(float(values[0]))
        C_vals.append(float(values[1]))
        S_vals.append(float(values[2]))
    if not (len(B_vals) == len(C_vals) == len(S_vals) == requested):
        raise RuntimeError(
            f"bootstrap completed {len(B_vals)} of {requested} configured draws"
        )
    q = 100.0 * float(cl)
    return {
        "bootstrap_n_success": float(requested),
        "bootstrap_B_percentile_cps": float(np.percentile(np.asarray(B_vals), q)),
        "bootstrap_CL_percentile_cps": float(np.percentile(np.asarray(C_vals), q)),
        "bootstrap_SL_percentile_cps": float(np.percentile(np.asarray(S_vals), q)),
        "bootstrap_B_median_cps": float(np.median(np.asarray(B_vals))),
    }


def _permutation_pvalue(
    blocks: Sequence[Dict[str, np.ndarray]],
    *,
    observed_B: float,
    n_permutations: int,
    covariance_scale: str,
    launch_phase_key: str = "theta_launch",
    seed: Optional[int],
) -> Dict[str, Any]:
    requested = int(n_permutations)
    if requested < 0:
        raise ValueError("permutation count must be nonnegative")
    if requested == 0:
        return {}
    if len(blocks) <= 1:
        raise ValueError("permutation requires at least two paired blocks")

    obs = float(observed_B)
    if not np.isfinite(obs):
        raise ValueError("observed permutation statistic must be finite")

    rng = np.random.default_rng(seed)

    X_full, y, var, _, _, _, launch_offset = _build_actual_voltage_design(
        blocks,
        include_launch_terms=True,
        launch_phase_key=launch_phase_key,
    )
    X_null, y_null, var_null, _, _, _, _ = _build_actual_voltage_design(
        blocks,
        include_launch_terms=False,
    )

    if launch_offset is None:
        raise RuntimeError("Full permutation design is missing launch terms")

    if y.shape != y_null.shape or not np.array_equal(y, y_null):
        raise RuntimeError("Full/null permutation designs have inconsistent observations")
    if var.shape != var_null.shape or not np.array_equal(var, var_null):
        raise RuntimeError("Full/null permutation designs have inconsistent variances")

    null_fit = _solve_wls(X_null, y, var, covariance_scale=covariance_scale)
    if null_fit.y.size != y.size:
        raise RuntimeError("Null permutation fit changed the observation count")

    y0 = null_fit.y_hat
    resid0 = null_fit.resid
    launch_end = launch_offset + 2

    count = 0
    success = 0
    for _ in range(requested):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=resid0.size)
        y_perm = y0 + signs * resid0

        fit = _solve_wls(X_full, y_perm, var, covariance_scale=covariance_scale)

        q = np.asarray(fit.beta[launch_offset:launch_end], dtype=float)
        B = float(np.linalg.norm(q))
        if not np.isfinite(B):
            raise ValueError("permutation draw returned a non-finite statistic")
        success += 1
        if B >= obs:
            count += 1

    if success != requested:
        raise RuntimeError(
            f"permutation completed {success} of {requested} configured draws"
        )

    return {
        "permutation_n_success": float(requested),
        "permutation_p_any_phase": float((count + 1.0) / (requested + 1.0)),
        "permutation_method": "null_residual_signflip_actual_voltage",
    }


def _parse_normalization_transmissions(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            raise ValueError("--normalization-T must not contain empty values")
        try:
            value = float(part)
        except Exception as exc:
            raise ValueError(f"--normalization-T contains a non-numeric value: {part!r}") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("--normalization-T values must be finite and positive")
        out.append(value)
    if not out:
        raise ValueError("--normalization-T must contain at least one positive value")
    return out


def _build_normalization_cases(
    transmissions: Sequence[float],
    *,
    rpm: float,
    A95: float,
    q_hat: np.ndarray,
    cov_q: np.ndarray,
) -> List[Dict[str, Any]]:
    if not np.isfinite(rpm) or rpm <= 0.0:
        raise ValueError("R_pm must be finite and positive for normalization cases")
    if not np.isfinite(A95) or A95 < 0.0:
        raise ValueError("A_add,95 must be finite and nonnegative for normalization cases")

    cases: List[Dict[str, Any]] = []
    for T in transmissions:
        T_value = float(T)
        if not np.isfinite(T_value) or T_value <= 0.0:
            raise ValueError("Normalization transmissions must be finite and positive")
        for multiplicity in (1, 2):
            effective_normalization = float(multiplicity * T_value)
            restoration_radius = float(effective_normalization * rpm)
            distance, nearest_phase, nearest_q = full_restoration_circle_distance(
                q_hat,
                cov_q,
                restoration_radius,
            )
            cases.append(
                {
                    "T": T_value,
                    "m": multiplicity,
                    "effective_normalization_mT": effective_normalization,
                    "restoration_radius_cps": restoration_radius,
                    "eta95": float(A95 / restoration_radius),
                    "d_min": float(distance),
                    "nearest_phase_rad": float(nearest_phase),
                    "nearest_C_cps": float(nearest_q[0]),
                    "nearest_S_cps": float(nearest_q[1]),
                }
            )
    return cases


def _build_summary(
    *,
    args: argparse.Namespace,
    cl: float,
    period_fixed: float,
    period_sigma: float,
    launch_name: str,
    control_name: str,
    phase_name: str,
    known_phase_name: str,
    known_phase_channel: str,
    phase_delta_values: Sequence[float],
    known_phase_delta_mean: float,
    known_phase_delta_rms_about_mean: float,
    known_phase_delta_span: float,
    paired_any: PairedFitResult,
    paired_known: PairedFitResult,
    x_mismatch_max: float,
    x_mismatch_median: float,
    phase_mismatch_max: float,
    phase_mismatch_median: float,
    launch_I: JointFitSummary,
    control_I: JointFitSummary,
    scalar_excess: float,
    scalar_sigma: float,
    scalar_B95: float,
    rpm: float,
    rpm_sigma: float,
    rpm_source: str,
    Tinf: float,
    rinfinity: float,
    eta_known: float,
    eta_any: float,
    T_req_known: float,
    T_req_any: float,
    normalization_cases: List[Dict[str, Any]],
    boot: Dict[str, Any],
    perm: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "confidence": cl,
        "covariance_scale": str(args.covariance_scale),
        "period_V": float(period_fixed),
        "period_sigma_V": float(period_sigma),
        "x_match_tol_V": float(args.x_match_tol),
        "launch_condition": launch_name,
        "control_condition": control_name,
        "phase_reference_condition": phase_name,
        "phase_reference_channel": args.phase_channel,
        "known_phase_reference_condition": known_phase_name,
        "known_phase_channel": known_phase_channel,
        "known_phase_delta_n_passes": len(phase_delta_values),
        "known_phase_delta_mean_rad": known_phase_delta_mean,
        "known_phase_delta_rms_about_mean_rad": known_phase_delta_rms_about_mean,
        "known_phase_delta_span_rad": known_phase_delta_span,
        "paired_voltage_model": (
            "stacked_actual_voltage_common_phase_with_separate_any_and_known_launch_fits"
        ),
        "paired_n_passes": paired_any.n_passes,
        "paired_n_point_pairs": paired_any.n_point_pairs,
        "paired_n_observations": paired_any.n_observations,
        "paired_df": paired_any.df,
        "paired_chi2": paired_any.chi2,
        "paired_reduced_chi2": paired_any.red_chi2,
        "paired_known_n_passes": paired_known.n_passes,
        "paired_known_n_point_pairs": paired_known.n_point_pairs,
        "paired_known_n_observations": paired_known.n_observations,
        "paired_known_df": paired_known.df,
        "paired_known_chi2": paired_known.chi2,
        "paired_known_reduced_chi2": paired_known.red_chi2,
        "x_mismatch_max_V": x_mismatch_max,
        "x_mismatch_median_V": x_mismatch_median,
        "phase_mismatch_max_rad": phase_mismatch_max,
        "phase_mismatch_median_rad": phase_mismatch_median,
        "C0_hat_cps": float(paired_any.q_common_hat[0]),
        "S0_hat_cps": float(paired_any.q_common_hat[1]),
        "C0_sigma_cps": _sqrt_nonnegative(float(paired_any.cov_common[0, 0])),
        "S0_sigma_cps": _sqrt_nonnegative(float(paired_any.cov_common[1, 1])),
        "C0S0_cov_cps2": float(paired_any.cov_common[0, 1]),
        "CL_hat_cps": float(paired_any.q_hat[0]),
        "SL_hat_cps": float(paired_any.q_hat[1]),
        "CL_sigma_cps": _sqrt_nonnegative(float(paired_any.cov_q[0, 0])),
        "SL_sigma_cps": _sqrt_nonnegative(float(paired_any.cov_q[1, 1])),
        "CLSL_cov_cps2": float(paired_any.cov_q[0, 1]),
        "B_hat_cps": paired_any.B_hat,
        "known_phase_CL_hat_cps": float(paired_known.q_hat[0]),
        "known_phase_SL_hat_cps": float(paired_known.q_hat[1]),
        "known_phase_CL_sigma_cps": _sqrt_nonnegative(float(paired_known.cov_q[0, 0])),
        "known_phase_SL_sigma_cps": _sqrt_nonnegative(float(paired_known.cov_q[1, 1])),
        "known_phase_CLSL_cov_cps2": float(paired_known.cov_q[0, 1]),
        "known_phase_B_hat_cps": paired_known.B_hat,
        "B95_known_phase_cps": paired_known.B95_known_phase,
        "B95_any_phase_cps": paired_any.B95_any_phase,
        "p_null_any_phase": paired_any.p_null_any_phase,
        "launch_total_singles_fringe_cps": launch_I.total_fringe,
        "launch_total_singles_fringe_sigma_cps": launch_I.total_fringe_sigma,
        "control_total_singles_fringe_cps": control_I.total_fringe,
        "control_total_singles_fringe_sigma_cps": control_I.total_fringe_sigma,
        "scalar_total_singles_excess_cps": float(scalar_excess),
        "scalar_total_singles_excess_sigma_cps": float(scalar_sigma),
        "scalar_B95_cps": float(scalar_B95),
        "Rpm_cps": float(rpm),
        "Rpm_sigma_cps": float(rpm_sigma),
        "Rpm_source": rpm_source,
        "Tinf": Tinf,
        "Rinfty_cps": float(rinfinity),
        "eta95_known_phase": float(eta_known),
        "eta95_any_phase": float(eta_any),
        "Tinf_required_for_eta1_known_phase": float(T_req_known),
        "Tinf_required_for_eta1_any_phase": float(T_req_any),
        "normalization_cases": normalization_cases,
    }
    summary.update(boot)
    summary.update(perm)
    return summary


def _add_full_restoration_summary(
    summary: Dict[str, Any],
    *,
    full_restoration_gap: float,
    full_restoration_distance: float,
    full_restoration_phase: float,
    full_restoration_nearest_q: np.ndarray,
) -> None:
    summary.update(
        {
            "full_restoration_gap_cps": float(full_restoration_gap),
            "full_restoration_mahalanobis_distance": float(full_restoration_distance),
            "full_restoration_nearest_phase_rad": float(full_restoration_phase),
            "full_restoration_nearest_C_cps": float(full_restoration_nearest_q[0]),
            "full_restoration_nearest_S_cps": float(full_restoration_nearest_q[1]),
            "full_restoration_distance_method": (
                "minimum_mahalanobis_distance_to_full_restoration_circle"
            ),
        }
    )


def _build_joint_fit_table(
    joint_fits: Dict[Tuple[str, str], JointFitSummary],
) -> List[Dict[str, Any]]:
    joint_table: List[Dict[str, Any]] = []
    for (cond, ch), jf in sorted(joint_fits.items()):
        joint_table.append(
            {
                "condition": cond,
                "channel": ch,
                "channel_label": _channel_label(ch),
                "fringe1_cps": jf.fringe1,
                "fringe1_sigma_cps": jf.fringe1_sigma,
                "fringe2_cps": jf.fringe2,
                "fringe2_sigma_cps": jf.fringe2_sigma,
                "total_fringe_cps": jf.total_fringe,
                "total_fringe_sigma_cps": jf.total_fringe_sigma,
                "V": _finite_float(jf.fit.V),
                "V_sigma": _finite_float(jf.fit.V_sigma),
                "A1_cps": _finite_float(jf.fit.A1),
                "A2_cps": _finite_float(jf.fit.A2),
            }
        )
    return joint_table


def _build_payload(
    summary: Dict[str, Any],
    joint_table: List[Dict[str, Any]],
    normalization_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "summary": summary,
        "joint_fits": joint_table,
        "normalization_cases": normalization_cases,
    }


def _write_payload(payload: Dict[str, Any], json_path: Path) -> Path:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return json_path


def _print_terminal_report(
    *,
    launch_name: str,
    control_name: str,
    phase_name: str,
    phase_channel: str,
    known_phase_name: str,
    known_phase_channel: str,
    period_fixed: float,
    period_sigma: float,
    paired_any: PairedFitResult,
    paired_known: PairedFitResult,
    x_mismatch_max: float,
    x_mismatch_median: float,
    phase_mismatch_max: float,
    known_phase_delta_rms_about_mean: float,
    known_phase_delta_span: float,
    summary: Dict[str, Any],
    rpm_source: str,
    rpm: float,
    Tinf: float,
    rinfinity: float,
    eta_known: float,
    eta_any: float,
    T_req_known: float,
    T_req_any: float,
    launch_I: JointFitSummary,
    control_I: JointFitSummary,
    scalar_excess: float,
    scalar_sigma: float,
    scalar_B95: float,
    normalization_cases: List[Dict[str, Any]],
    boot: Dict[str, Any],
    perm: Dict[str, Any],
    json_path: Path,
) -> None:
    print("\nPaired launch-control quadrature bound")
    print("--------------------------------------")
    print(f"launch condition         {launch_name}")
    print(f"control condition        {control_name}")
    print(f"phase reference          {phase_name}:{phase_channel}")
    print(f"known phase reference    {known_phase_name}:{known_phase_channel}")
    print(f"fixed period             {_fmt(period_fixed, 6)} V")
    print(f"period uncertainty       {_fmt(period_sigma, 6)} V")
    print(f"any-phase passes / pairs {paired_any.n_passes} / {paired_any.n_point_pairs}")
    print(f"known passes / pairs     {paired_known.n_passes} / {paired_known.n_point_pairs}")
    print(f"any observations         {paired_any.n_observations}")
    print(f"known observations       {paired_known.n_observations}")
    print(f"max voltage mismatch     {_fmt(x_mismatch_max, 6)} V")
    print(f"median voltage mismatch  {_fmt(x_mismatch_median, 6)} V")
    print(f"max phase mismatch       {_fmt(phase_mismatch_max, 4)} rad")
    print(f"known-phase delta RMS    {_fmt(known_phase_delta_rms_about_mean, 4)} rad")
    print(f"known-phase delta span   {_fmt(known_phase_delta_span, 4)} rad")
    print(
        f"C0 nuisance (any)        {_fmt(paired_any.q_common_hat[0], 3)} ±"
        f" {_fmt(summary['C0_sigma_cps'], 3)} cps"
    )
    print(
        f"S0 nuisance (any)        {_fmt(paired_any.q_common_hat[1], 3)} ±"
        f" {_fmt(summary['S0_sigma_cps'], 3)} cps"
    )
    print(
        f"CL_hat any phase         {_fmt(paired_any.q_hat[0], 3)} ±"
        f" {_fmt(summary['CL_sigma_cps'], 3)} cps"
    )
    print(
        f"SL_hat any phase         {_fmt(paired_any.q_hat[1], 3)} ±"
        f" {_fmt(summary['SL_sigma_cps'], 3)} cps"
    )
    print(f"B_launch any phase       {_fmt(paired_any.B_hat, 3)} cps")
    print(
        f"CL_hat known phase       {_fmt(paired_known.q_hat[0], 3)} ±"
        f" {_fmt(summary['known_phase_CL_sigma_cps'], 3)} cps"
    )
    print(
        f"SL_hat known phase       {_fmt(paired_known.q_hat[1], 3)} ±"
        f" {_fmt(summary['known_phase_SL_sigma_cps'], 3)} cps"
    )
    print(f"B_launch known phase     {_fmt(paired_known.B_hat, 3)} cps")
    print(f"B95 known phase          {_fmt(paired_known.B95_known_phase, 3)} cps")
    print(f"B95 arbitrary phase      {_fmt(paired_any.B95_any_phase, 3)} cps")
    print(f"null p-value, 2D approx  {_fmt(paired_any.p_null_any_phase, 4)}")
    print(f"reduced chi2 any         {_fmt(paired_any.red_chi2, 3)}")
    print(f"reduced chi2 known       {_fmt(paired_known.red_chi2, 3)}")

    print("\nModel interpretation")
    print("--------------------")
    print(f"R_pm source              {rpm_source}")
    print(f"R_pm                     {_fmt(rpm, 3)} cps")
    print(f"T_inf                    {_fmt(Tinf, 4)}")
    print(f"T_inf R_pm               {_fmt(rinfinity, 3)} cps")
    print(f"eta95 known phase        {_fmt(eta_known, 4)}")
    print(f"eta95 arbitrary phase    {_fmt(eta_any, 4)}")
    print(f"T_inf for eta=1, known   {_fmt(T_req_known, 4)}")
    print(f"T_inf for eta=1, any     {_fmt(T_req_any, 4)}")

    print("\nScalar total-amplitude check")
    print("----------------------------")
    print(
        f"Launch singles total     {_fmt(launch_I.total_fringe, 3)} ± "
        f"{_fmt(launch_I.total_fringe_sigma, 3)} cps"
    )
    print(
        f"Control singles total    {_fmt(control_I.total_fringe, 3)} ± "
        f"{_fmt(control_I.total_fringe_sigma, 3)} cps"
    )
    print(f"Scalar excess            {_fmt(scalar_excess, 3)} ± {_fmt(scalar_sigma, 3)} cps")
    print(f"Scalar one-sided B95     {_fmt(scalar_B95, 3)} cps")

    if normalization_cases:
        print("\nNormalization cases")
        print("-------------------")
        print("T          m       mT    radius_cps      eta95   d_min_sigma")
        for row in normalization_cases:
            print(
                f"{_fmt(row['T'], 4).rjust(8)}   "
                f"{str(row['m']).rjust(1)}   "
                f"{_fmt(row['effective_normalization_mT'], 4).rjust(8)}   "
                f"{_fmt(row['restoration_radius_cps'], 3).rjust(10)}   "
                f"{_fmt(row['eta95'], 4).rjust(8)}   "
                f"{_fmt(row['d_min'], 2).rjust(11)}"
            )

    if boot:
        print("\nBootstrap diagnostic")
        print("--------------------")
        print(f"successful resamples     {int(boot['bootstrap_n_success'])}")
        print(f"B percentile             {_fmt(boot['bootstrap_B_percentile_cps'], 3)} cps")
        print(f"CL percentile            {_fmt(boot['bootstrap_CL_percentile_cps'], 3)} cps")

    if perm:
        print("\nPermutation diagnostic")
        print("----------------------")
        print(f"successful permutations  {int(perm['permutation_n_success'])}")
        print(f"p-value any phase        {_fmt(perm['permutation_p_any_phase'], 4)}")

    print("\nWrote")
    print(str(json_path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute launch-null uncertainty bounds from alternating MZI records."
    )
    p.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Condition spec name:path/to/subdir. Repeat for Launched, Erased/control, etc.",
    )
    p.add_argument("--launch", default="Launch", help="Condition name for launched idlers.")
    p.add_argument(
        "--control",
        default="Erase",
        help="Contemporaneous control condition for paired subtraction, normally Erase.",
    )
    p.add_argument(
        "--phase-reference",
        default=None,
        help="Condition used to define phase x0_i; defaults to --control.",
    )
    p.add_argument(
        "--phase-channel",
        choices=["C", "I"],
        default="C",
        help="Use coincidences C or singles I for the phase-reference joint fit. Default: C.",
    )
    p.add_argument(
        "--known-phase-reference",
        default=None,
        help=(
            "Condition used to define the predicted/known launch phase. Defaults to"
            " --phase-reference."
        ),
    )
    p.add_argument(
        "--known-phase-channel",
        choices=["C", "I"],
        default=None,
        help=(
            "Channel used to define the predicted/known launch phase. Defaults to --phase-channel."
        ),
    )
    p.add_argument(
        "--rpm-erased",
        default="Erase",
        help="Condition whose coincidence fringe supplies the erase term for R_pm.",
    )
    p.add_argument(
        "--rpm-unmodified",
        default="Preserve",
        help="Condition whose coincidence fringe supplies the preserve term for R_pm.",
    )
    p.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Use this path-marked rate R_pm in cps instead of fitting from coincidences.",
    )
    p.add_argument(
        "--Tinf",
        type=float,
        required=True,
        help="Nominal survival probability T_inf.",
    )
    p.add_argument(
        "--period",
        type=float,
        default=None,
        help="Fixed MZI period in V. If omitted, fit it from the phase reference and then fix it.",
    )
    p.add_argument(
        "--period-sigma",
        type=float,
        default=None,
        help=(
            "Uncertainty of the fixed MZI period in V. Required with --period; "
            "otherwise obtained from the phase-reference fit."
        ),
    )
    p.add_argument(
        "--covariance-scale",
        choices=["max1", "residual", "none"],
        default="max1",
        help=(
            "How to scale the paired WLS covariance: max1=max(1, reduced chi2), "
            "residual=reduced chi2, none=Poisson variances only. Default: max1."
        ),
    )
    p.add_argument("--max-passes", type=int, default=None, help="Limit to first N passes.")
    p.add_argument(
        "--min-points", type=int, default=6, help="Minimum points required in a pass. Default: 6."
    )
    p.add_argument(
        "--use-first-secs",
        type=float,
        default=None,
        help="Use first index-aligned records up to this cumulative duration per condition.",
    )
    p.add_argument(
        "--x-match-tol",
        type=float,
        default=0.05,
        help=(
            "Sanity limit for index-aligned launch/control x-grid mismatch in V before aborting. "
            "Default: 0.05."
        ),
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Optional pass-resampling bootstrap count. Default: 0.",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Optional null-residual sign-flip test count for null p-value. Default: 0.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for bootstrap/permutation.")
    p.add_argument(
        "--normalization-T",
        required=True,
        help="Comma-separated transmission values for m=1 and m=2 normalization cases.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the JSON output.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cl = DEFAULT_CL
    Tinf = float(args.Tinf)
    if not np.isfinite(Tinf) or Tinf <= 0.0:
        raise ValueError("--Tinf must be finite and positive")
    if args.rpm is not None and (not np.isfinite(args.rpm) or args.rpm <= 0.0):
        raise ValueError("--rpm must be finite and positive")
    normalization_transmissions = _parse_normalization_transmissions(
        args.normalization_T
    )

    conditions = parse_condition_specs(args.condition)
    condition_names = [name for name, _ in conditions]
    bundles = _load_condition_bundles(
        conditions,
        max_passes=args.max_passes,
        min_points=args.min_points,
        use_first_secs=args.use_first_secs,
    )

    launch_name = args.launch
    control_name = args.control
    phase_name = args.phase_reference or control_name
    known_phase_name = args.known_phase_reference or phase_name
    known_phase_channel = args.known_phase_channel or args.phase_channel
    for required in [launch_name, control_name, phase_name, known_phase_name]:
        if required not in bundles:
            raise ValueError(
                f"Condition '{required}' was not loaded. Available conditions:"
                f" {', '.join(condition_names)}"
            )

    # Determine the period. If absent, use the phase-reference fit with a free period,
    # then refit all summaries with that fixed period for a consistent phase reference.
    period_fixed: Optional[float]
    if args.period is not None:
        period_fixed = float(args.period)
        if not np.isfinite(period_fixed) or period_fixed <= 0.0:
            raise ValueError("--period must be finite and positive")
        phase_seed_fit = None
    else:
        phase_seed_fit = _joint_fit_condition(bundles[phase_name], args.phase_channel, period=None)
        period_fixed = _finite_float(phase_seed_fit.P)
        if not (np.isfinite(period_fixed) and period_fixed > 0.0):
            raise RuntimeError("Phase-reference period fit failed; pass --period.")

    if args.period_sigma is not None:
        period_sigma = float(args.period_sigma)
    elif phase_seed_fit is not None:
        period_sigma = _finite_float(phase_seed_fit.P_sigma)
    else:
        raise ValueError("--period-sigma is required when --period is supplied")
    if not np.isfinite(period_sigma) or period_sigma < 0.0:
        raise ValueError("Period uncertainty must be finite and nonnegative")

    # Joint fits used for phase, scalar checks, and R_pm.
    joint_fits: Dict[Tuple[str, str], JointFitSummary] = {}

    def get_joint(cond: str, ch: str) -> JointFitSummary:
        key = (cond, ch.upper())
        if key not in joint_fits:
            fit = _joint_fit_condition(bundles[cond], ch.upper(), period=period_fixed)
            joint_fits[key] = _summarize_joint_fit(cond, ch.upper(), fit)
        return joint_fits[key]

    phase_summary = get_joint(phase_name, args.phase_channel)
    phase_x0 = list(phase_summary.fit.x0_list)
    if not phase_x0:
        raise RuntimeError("Phase-reference fit did not return x0_list")

    known_phase_summary = get_joint(known_phase_name, known_phase_channel)
    known_phase_x0 = list(known_phase_summary.fit.x0_list)
    if not known_phase_x0:
        raise RuntimeError("Known-phase reference fit did not return x0_list")

    phase_x0_by_pass = _phase_values_by_pass(
        bundles[phase_name].pass_indices,
        phase_x0,
        context="phase-reference joint fit",
    )
    known_phase_x0_by_pass = _phase_values_by_pass(
        bundles[known_phase_name].pass_indices,
        known_phase_x0,
        context="known-phase-reference joint fit",
    )
    if set(phase_x0_by_pass) != set(known_phase_x0_by_pass):
        raise ValueError(
            "phase-reference and known-phase-reference pass indices do not match"
        )

    phase_delta_values: List[float] = []
    for pass_idx in bundles[phase_name].pass_indices:
        pass_idx_i = int(pass_idx)
        delta = (
            2.0
            * np.pi
            * (known_phase_x0_by_pass[pass_idx_i] - phase_x0_by_pass[pass_idx_i])
            / float(period_fixed)
        )
        if not np.isfinite(delta):
            raise ValueError("phase-reference offset difference must be finite")
        phase_delta_values.append(float(delta))

    if phase_delta_values:
        phase_delta_arr = np.unwrap(np.asarray(phase_delta_values, dtype=float))
        known_phase_delta_mean = float(np.mean(phase_delta_arr))
        known_phase_delta_rms_about_mean = float(
            np.sqrt(np.mean((phase_delta_arr - known_phase_delta_mean) ** 2))
        )
        known_phase_delta_span = float(np.max(phase_delta_arr) - np.min(phase_delta_arr))
    else:
        raise ValueError("at least one aligned phase-reference pass is required")

    launch_I = get_joint(launch_name, "I")
    control_I = get_joint(control_name, "I")

    blocks_any = _make_paired_blocks(
        bundles[launch_name],
        bundles[control_name],
        phase_x0_list=phase_x0,
        phase_pass_indices=bundles[phase_name].pass_indices,
        period=float(period_fixed),
        max_pairs=args.max_passes,
        x_match_tol=float(args.x_match_tol),
    )
    blocks_known = _replace_known_phase_in_blocks(
        blocks_any,
        known_phase_x0_list=known_phase_x0,
        known_phase_pass_indices=bundles[known_phase_name].pass_indices,
        period=float(period_fixed),
    )

    dx_chunks: List[np.ndarray] = []
    for block_number, block in enumerate(blocks_any, start=1):
        dx = np.asarray(block.get("dx", []), dtype=float)
        if dx.ndim != 1 or dx.size <= 0:
            raise ValueError(
                f"paired block {block_number} dx must be a nonempty one-dimensional array"
            )
        if not np.all(np.isfinite(dx)):
            raise ValueError(f"paired block {block_number} dx must be finite")
        dx_chunks.append(np.abs(dx))
    dx_abs = np.concatenate(dx_chunks) if dx_chunks else np.asarray([], dtype=float)

    x_mismatch_max = float(np.max(dx_abs)) if dx_abs.size else float("nan")
    x_mismatch_median = float(np.median(dx_abs)) if dx_abs.size else float("nan")
    phase_mismatch_max = (
        2.0 * np.pi * x_mismatch_max / float(period_fixed)
        if np.isfinite(x_mismatch_max) and np.isfinite(float(period_fixed))
        else float("nan")
    )
    phase_mismatch_median = (
        2.0 * np.pi * x_mismatch_median / float(period_fixed)
        if np.isfinite(x_mismatch_median) and np.isfinite(float(period_fixed))
        else float("nan")
    )

    paired_any = _fit_blocks(
        blocks_any,
        cl=cl,
        covariance_scale=args.covariance_scale,
        launch_phase_key="theta_launch",
    )
    paired_known = _fit_blocks(
        blocks_known,
        cl=cl,
        covariance_scale=args.covariance_scale,
        launch_phase_key="theta_known_launch",
    )

    boot = _bootstrap_blocks(
        blocks_any,
        n_bootstrap=int(args.bootstrap),
        cl=cl,
        covariance_scale=args.covariance_scale,
        launch_phase_key="theta_launch",
        seed=args.seed,
    )
    perm = _permutation_pvalue(
        blocks_any,
        observed_B=paired_any.B_hat,
        n_permutations=int(args.permutations),
        covariance_scale=args.covariance_scale,
        launch_phase_key="theta_launch",
        seed=None if args.seed is None else int(args.seed) + 1,
    )

    # R_pm: prefer user-supplied, otherwise compute from erased and unmodified coincidence fits.
    rpm_source = "supplied"
    rpm_sigma = float("nan")
    if args.rpm is not None:
        rpm = float(args.rpm)
    else:
        rpm_erased_name = args.rpm_erased
        rpm_unmodified_name = args.rpm_unmodified
        if rpm_erased_name not in bundles or rpm_unmodified_name not in bundles:
            raise ValueError(
                "R_pm was not supplied and the --rpm-erased/--rpm-unmodified conditions were not"
                " both loaded. Either load them or pass --rpm."
            )
        erased_C = get_joint(rpm_erased_name, "C")
        unmodified_C = get_joint(rpm_unmodified_name, "C")
        rpm = float(erased_C.total_fringe - unmodified_C.total_fringe)
        if np.isfinite(erased_C.total_fringe_sigma) and np.isfinite(
            unmodified_C.total_fringe_sigma
        ):
            rpm_sigma = math.hypot(
                erased_C.total_fringe_sigma,
                unmodified_C.total_fringe_sigma,
            )
        rpm_source = f"{rpm_erased_name}:C - {rpm_unmodified_name}:C"

    rinfinity = (
        Tinf * rpm if np.isfinite(Tinf) and np.isfinite(rpm) else float("nan")
    )
    eta_known = (
        paired_known.B95_known_phase / rinfinity
        if rinfinity > 0.0
        else float("nan")
    )
    eta_any = (
        paired_any.B95_any_phase / rinfinity
        if rinfinity > 0.0
        else float("nan")
    )
    T_req_known = paired_known.B95_known_phase / rpm if rpm > 0.0 else float("nan")
    T_req_any = paired_any.B95_any_phase / rpm if rpm > 0.0 else float("nan")

    scalar_excess = launch_I.total_fringe - control_I.total_fringe
    scalar_sigma = (
        math.hypot(
            launch_I.total_fringe_sigma,
            control_I.total_fringe_sigma,
        )
        if np.isfinite(launch_I.total_fringe_sigma)
        and np.isfinite(control_I.total_fringe_sigma)
        else float("nan")
    )
    scalar_B95 = (
        max(0.0, scalar_excess) + float(norm.ppf(cl)) * scalar_sigma
        if np.isfinite(scalar_excess) and np.isfinite(scalar_sigma)
        else float("nan")
    )

    normalization_cases = _build_normalization_cases(
        normalization_transmissions,
        rpm=rpm,
        A95=paired_any.B95_any_phase,
        q_hat=paired_any.q_hat,
        cov_q=paired_any.cov_q,
    )

    summary = _build_summary(
        args=args,
        cl=cl,
        period_fixed=period_fixed,
        period_sigma=period_sigma,
        launch_name=launch_name,
        control_name=control_name,
        phase_name=phase_name,
        known_phase_name=known_phase_name,
        known_phase_channel=known_phase_channel,
        phase_delta_values=phase_delta_values,
        known_phase_delta_mean=known_phase_delta_mean,
        known_phase_delta_rms_about_mean=known_phase_delta_rms_about_mean,
        known_phase_delta_span=known_phase_delta_span,
        paired_any=paired_any,
        paired_known=paired_known,
        x_mismatch_max=x_mismatch_max,
        x_mismatch_median=x_mismatch_median,
        phase_mismatch_max=phase_mismatch_max,
        phase_mismatch_median=phase_mismatch_median,
        launch_I=launch_I,
        control_I=control_I,
        scalar_excess=scalar_excess,
        scalar_sigma=scalar_sigma,
        scalar_B95=scalar_B95,
        rpm=rpm,
        rpm_sigma=rpm_sigma,
        rpm_source=rpm_source,
        Tinf=Tinf,
        rinfinity=rinfinity,
        eta_known=eta_known,
        eta_any=eta_any,
        T_req_known=T_req_known,
        T_req_any=T_req_any,
        normalization_cases=normalization_cases,
        boot=boot,
        perm=perm,
    )

    B_hat = _finite_float(summary.get("B_hat_cps"))
    full_restoration_gap = (
        rinfinity - B_hat
        if np.isfinite(rinfinity) and np.isfinite(B_hat)
        else float("nan")
    )
    (
        full_restoration_distance,
        full_restoration_phase,
        full_restoration_nearest_q,
    ) = full_restoration_circle_distance(
        paired_any.q_hat,
        paired_any.cov_q,
        rinfinity,
    )
    _add_full_restoration_summary(
        summary,
        full_restoration_gap=full_restoration_gap,
        full_restoration_distance=full_restoration_distance,
        full_restoration_phase=full_restoration_phase,
        full_restoration_nearest_q=full_restoration_nearest_q,
    )

    joint_table = _build_joint_fit_table(joint_fits)
    payload = _build_payload(summary, joint_table, normalization_cases)
    json_path = _write_payload(payload, args.output.expanduser().resolve())
    _print_terminal_report(
        launch_name=launch_name,
        control_name=control_name,
        phase_name=phase_name,
        phase_channel=args.phase_channel,
        known_phase_name=known_phase_name,
        known_phase_channel=known_phase_channel,
        period_fixed=period_fixed,
        period_sigma=period_sigma,
        paired_any=paired_any,
        paired_known=paired_known,
        x_mismatch_max=x_mismatch_max,
        x_mismatch_median=x_mismatch_median,
        phase_mismatch_max=phase_mismatch_max,
        known_phase_delta_rms_about_mean=known_phase_delta_rms_about_mean,
        known_phase_delta_span=known_phase_delta_span,
        summary=summary,
        rpm_source=rpm_source,
        rpm=rpm,
        Tinf=Tinf,
        rinfinity=rinfinity,
        eta_known=eta_known,
        eta_any=eta_any,
        T_req_known=T_req_known,
        T_req_any=T_req_any,
        launch_I=launch_I,
        control_I=control_I,
        scalar_excess=scalar_excess,
        scalar_sigma=scalar_sigma,
        scalar_B95=scalar_B95,
        normalization_cases=normalization_cases,
        boot=boot,
        perm=perm,
        json_path=json_path,
    )


if __name__ == "__main__":
    main()
