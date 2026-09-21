#!/usr/bin/env python

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.special import expit, logit

from .matplotlib_pdf_determinism import make_pdf_font_subsets_deterministic
from .mzi_analysis import (
    P_FIT_MAX,
    P_FIT_MIN,
    FitResult,
    _conservative_covariance_scale,
    _covariance_from_weighted_jacobian,
    _eval_idler_model,
    _require_finite_fit_parameters,
    _require_successful_optimizer_result,
    _standard_deviation_from_variance,
    _validate_optional_fixed_period,
    fit_scan,
    scan_from_records,
    validate_positive_variances,
)
from .mzi_io import (
    assert_constant_point_duration,
    load_dark,
    load_plan_length,
    load_points,
    parse_condition_specs,
    split_by_plan_length,
)
from .mzi_plot_utils import (
    _fmt_pm,
    _plot_series,
    _plot_series_coinc,
    _prepare_plot_arrays,
    _prepare_plot_arrays_coinc,
)

make_pdf_font_subsets_deterministic()

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb,amsfonts}",
        "lines.linewidth": 0.6,
        "lines.markersize": 2.0,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)

PRL_DOUBLE_COL_WIDTH = 7.05
PRL_FIG_HEIGHT = 2.65
CONDITION_PALETTE = ("#d62728", "#2ca02c", "#1f77b4")
LINE_OPACITY = 0.3
JOINT_LINEWIDTH = 1.2
X_TICKS = [0.0, 1.5, 3.0]
X_TICK_LABELS = ["0", "1.5", "3.0"]
AXIS_LABEL_FONTSIZE = 8
TICK_LABEL_FONTSIZE = 8
X_TICK_LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8

# Naming convention only. The labels used in the laboratory used
# "signal" and "idler" opposite to the standard convention adopted in
# the paper. In the acquisition data, channel I is the MZI-exit
# singles channel, so it is displayed here as "signal singles."
# Coincidences involve both photons and require no relabeling.


def _validate_joint_initialization(
    init: Optional[Dict[str, Any]],
    *,
    pass_count: int,
    context: str,
) -> None:
    if init is None:
        return
    if not isinstance(init, dict):
        raise ValueError(f"{context} initialization must be a dictionary")
    for name in ("A1", "A2", "P"):
        if name in init:
            try:
                value = float(init[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{context} initial {name} must be finite and positive"
                ) from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{context} initial {name} must be finite and positive"
                )
    if "V" in init:
        try:
            visibility = float(init["V"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{context} initial V must be finite and strictly between zero and one"
            ) from exc
        if not np.isfinite(visibility) or not (0.0 < visibility < 1.0):
            raise ValueError(
                f"{context} initial V must be finite and strictly between zero and one"
            )
    if "x0_list" in init:
        x0_values = init["x0_list"]
        if not isinstance(x0_values, (list, tuple)) or len(x0_values) != pass_count:
            raise ValueError(f"{context} initial x0_list has the wrong length")
        try:
            x0_array = np.asarray(x0_values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} initial x0_list must be finite") from exc
        if x0_array.ndim != 1 or not np.all(np.isfinite(x0_array)):
            raise ValueError(f"{context} initial x0_list must be finite")

CHANNEL_TITLE_LABEL = {
    "C": "Coincidences",
    "I": "Signal singles",
}

CONDITION_SYMBOL_LABEL = {
    "Launch": "L",
    "Erase": "E",
    "Preserve": "P",
}

CONDITION_TABLE_LABEL = {
    "Erase2": "Erase",
}

# Fringe band styling (subtle, behind data)
BAND_FACE_UPPER = "0.85"  # lighter grey for upper band (mean to peaks)
BAND_ALPHA = 0.9
Z_BAND = -100


def _deterministic_figure_metadata(fmt: str) -> Dict[str, Any]:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
    if fmt == "pdf":
        return {
            "Creator": "complementarity artifact",
            "Producer": "Matplotlib",
            "CreationDate": timestamp,
            "ModDate": timestamp,
        }
    return {"Software": "complementarity artifact"}

@dataclass(kw_only=True)
class JointFitResult:
    A1: float
    A2: float
    V: float
    P: float
    x0_list: List[float]
    A1_sigma: float
    A2_sigma: float
    V_sigma: float
    P_sigma: float
    total_fringe: float
    total_fringe_sigma: float
    Fringe1_sigma: float
    Fringe2_sigma: float


def _condition_color(idx: int):
    return CONDITION_PALETTE[int(idx) % len(CONDITION_PALETTE)]


def _parse_order_specs(
    specs: Optional[List[str]],
    valid_condition_names: List[str],
    option_name: str = "--order",
) -> List[Tuple[str, str]]:
    """
    Parse ordering entries of the form condition:channel where channel is C or I.
    Returns an ordered list of (condition, channel) tuples with channel uppercased.
    """
    if not specs:
        return []

    valid = set(valid_condition_names)
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for spec in specs:
        raw = str(spec)
        if ":" not in raw:
            raise ValueError(f"Invalid {option_name} '{raw}': expected condition:channel")
        cond_name, channel = raw.split(":", 1)
        cond_name = cond_name.strip()
        channel = channel.strip().upper()

        if not cond_name:
            raise ValueError(f"Invalid {option_name} '{raw}': empty condition name")
        if cond_name not in valid:
            raise ValueError(f"Invalid {option_name} '{raw}': unknown condition '{cond_name}'")
        if channel not in {"C", "I"}:
            raise ValueError(f"Invalid {option_name} '{raw}': channel must be C or I")

        item = (cond_name, channel)
        if item in seen:
            raise ValueError(f"Duplicate {option_name} entry '{cond_name}:{channel}'")
        seen.add(item)
        out.append(item)

    return out


def _parse_table_order_specs(
    specs: Optional[List[str]],
    valid_condition_names: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Parse table ordering entries of the form run:condition:channel where
    channel is C or I.
    """
    if not specs:
        return []

    valid = set(valid_condition_names)
    out: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for spec in specs:
        raw = str(spec)
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --table-order '{raw}': expected run:condition:channel")

        run_id, cond_name, channel = (part.strip() for part in parts)
        channel = channel.upper()

        if not run_id:
            raise ValueError(f"Invalid --table-order '{raw}': empty run identifier")
        if not cond_name:
            raise ValueError(f"Invalid --table-order '{raw}': empty condition name")
        if cond_name not in valid:
            raise ValueError(f"Invalid --table-order '{raw}': unknown condition '{cond_name}'")
        if channel not in {"C", "I"}:
            raise ValueError(f"Invalid --table-order '{raw}': channel must be C or I")

        item = (run_id, cond_name, channel)
        if item in seen:
            raise ValueError(f"Duplicate --table-order entry '{run_id}:{cond_name}:{channel}'")
        seen.add(item)
        out.append(item)

    return out


def aligned_record_truncation(
    recs_by_cond: Dict[str, List[Dict[str, Any]]], limit_s: float
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return truncated leading slices per condition that include the first M
    index-aligned records such that every condition's cumulative time remains
    <= limit_s.
    """
    try:
        limit = float(limit_s)
    except Exception:
        return recs_by_cond
    if limit <= 0.0:
        return recs_by_cond
    if not recs_by_cond:
        return recs_by_cond

    n_common = min(len(recs) for recs in recs_by_cond.values())
    cum: Dict[str, float] = {name: 0.0 for name in recs_by_cond}
    m = 0
    for i in range(n_common):
        ok = True
        for name, recs in recs_by_cond.items():
            t = float(recs[i].get("duration_s", 0.0) or 0.0)
            if (cum[name] + t) > (limit + 1e-9):
                ok = False
                break
        if not ok:
            break
        for name, recs in recs_by_cond.items():
            t = float(recs[i].get("duration_s", 0.0) or 0.0)
            cum[name] += t
        m += 1

    return {name: recs[:m] for name, recs in recs_by_cond.items()}


def _shift_x(x: Any, x_offset: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr - float(x_offset)


def _fmt_val(v: Any) -> str:
    """
    Format a value for table output with fixed 3 decimals for numeric values,
    blanks for None/NaN/inf, and strings as-is.
    """
    try:
        if v is None:
            return ""
        fv = float(v)
        if not np.isfinite(fv):
            return ""
        return f"{fv:.3f}"
    except Exception:
        return str(v)


def _json_num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def validate_total_fringe_values(
    total_fringe: Any,
    total_fringe_sigma: Any,
    fringe1: Any,
    fringe2: Any,
    *,
    context: str,
) -> Tuple[float, float]:
    try:
        total = float(total_fringe)
        sigma = float(total_fringe_sigma)
        exit1 = float(fringe1)
        exit2 = float(fringe2)
    except Exception as exc:
        raise ValueError(f"{context}: canonical total-fringe values are required") from exc

    if not np.isfinite(total):
        raise ValueError(f"{context}: total_fringe must be finite")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(
            f"{context}: total_fringe_sigma must be finite and nonnegative"
        )
    if not np.isfinite(exit1) or not np.isfinite(exit2):
        raise ValueError(f"{context}: per-exit fringes must be finite")
    if not np.isclose(total, exit1 + exit2, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{context}: total_fringe does not equal Fringe1 + Fringe2")
    return total, sigma


def _total_fringe_and_sigma(
    fit: FitResult | JointFitResult,
    *,
    context: str,
) -> Tuple[float, float]:
    return validate_total_fringe_values(
        fit.total_fringe,
        fit.total_fringe_sigma,
        fit.A1 * fit.V,
        fit.A2 * fit.V,
        context=context,
    )


def _expand_rows_for_report(
    rows: List[Dict[str, Any]],
    _conditions: List[Tuple[str, Path]],
    order: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """
    Convert internal wide rows (C.* and I.*) into narrow report rows with a Channel column.
    Emit only the provided (condition, channel) groups in that exact order.
    """
    value_fields = ["A1", "A2", "total_fringe", "Fringe1", "Fringe2", "P", "V", "x0"]
    out: List[Dict[str, Any]] = []
    groups: List[Tuple[str, str]] = [
        (str(cond_name), str(ch).strip().upper()) for cond_name, ch in order
    ]

    for cond_name, ch in groups:
        cond_rows = [r for r in rows if str(r.get("condition", "")) == cond_name]
        for r in cond_rows:
            rr: Dict[str, Any] = {
                "condition": cond_name,
                "channel": ch,
                "pass": r.get("pass", ""),
            }
            for f in value_fields:
                key = f"{ch}.{f}"
                rr[f] = r.get(key, np.nan)
                if f != "x0":
                    rr[f"{f}_sigma"] = r.get(f"{key}_sigma")
            out.append(rr)

    return out


def _expand_rows_for_table_report(
    rows: List[Dict[str, Any]],
    order: List[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    """
    Convert internal wide rows into narrow table rows in run, condition,
    and channel order.
    """
    value_fields = ["A1", "A2", "total_fringe", "Fringe1", "Fringe2", "P", "V", "x0"]
    out: List[Dict[str, Any]] = []

    for run_id, cond_name, channel in order:
        ch = str(channel).strip().upper()
        cond_rows = [r for r in rows if str(r.get("condition", "")) == cond_name]
        for row in cond_rows:
            report_row: Dict[str, Any] = {
                "run": str(run_id),
                "condition": str(cond_name),
                "channel": ch,
                "pass": row.get("pass", ""),
            }
            for field in value_fields:
                key = f"{ch}.{field}"
                report_row[field] = row.get(key, np.nan)
                if field != "x0":
                    report_row[f"{field}_sigma"] = row.get(f"{key}_sigma")
            out.append(report_row)

    return out


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def _fmt_pm_latex(v: Any, s: Any, nd: int = 3) -> str:
    return _fmt_pm(v, s, nd=nd).replace("±", r"\ensuremath{\pm}")


def _condition_display_label(name: str) -> str:
    normalized_name = str(name).strip()
    return CONDITION_TABLE_LABEL.get(normalized_name, normalized_name)


def _write_joint_latex_table(
    rows: List[Dict[str, Any]],
    conditions: List[Tuple[str, Path]],
    order: List[Tuple[str, str, str]],
    out_tex_path: Path,
) -> None:
    report_rows = _expand_rows_for_table_report(rows, order=order)
    joint_rows = [r for r in report_rows if str(r.get("pass", "")) == "joint"]

    lines: List[str] = []
    lines.append(r"\begin{tabular}{lllrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Run & Condition & Detection channel & $\bar R_1$ ($\mathrm{s}^{-1}$) & "
        r"$\bar R_2$ ($\mathrm{s}^{-1}$) & $\mathcal{V}$ & "
        r"$A$ ($\mathrm{s}^{-1}$) \\"
    )
    lines.append(r"\midrule")

    for r in joint_rows:
        run_id = _latex_escape(str(r.get("run", "")))
        cond = _latex_escape(_condition_display_label(str(r.get("condition", ""))))
        ch = str(r.get("channel", "")).upper()
        ch_label = _latex_escape(CHANNEL_TITLE_LABEL.get(ch, ch))
        total_fringe, total_fringe_sigma = validate_total_fringe_values(
            r.get("total_fringe"),
            r.get("total_fringe_sigma"),
            r.get("Fringe1"),
            r.get("Fringe2"),
            context=f"{r.get('condition', '')} {ch} joint table row",
        )
        vals = [
            _fmt_pm_latex(r.get("A1"), r.get("A1_sigma"), nd=1),
            _fmt_pm_latex(r.get("A2"), r.get("A2_sigma"), nd=1),
            _fmt_pm_latex(r.get("V"), r.get("V_sigma"), nd=3),
            _fmt_pm_latex(total_fringe, total_fringe_sigma, nd=1),
        ]
        lines.append(f"{run_id} & {cond} & {ch_label} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_by_pass_latex_table(
    rows: List[Dict[str, Any]],
    conditions: List[Tuple[str, Path]],
    order: List[Tuple[str, str, str]],
    out_tex_path: Path,
) -> None:
    report_rows = _expand_rows_for_table_report(rows, order=order)
    pass_rows = [r for r in report_rows if str(r.get("pass", "")) != "joint"]

    lines: List[str] = []
    lines.append(r"\begin{tabular}{llllrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Run & Condition & Detection channel & Scan & $\bar R_1$ ($\mathrm{s}^{-1}$) & "
        r"$\bar R_2$ ($\mathrm{s}^{-1}$) & $\mathcal{V}$ & "
        r"$A$ ($\mathrm{s}^{-1}$) \\"
    )
    lines.append(r"\midrule")

    prev_group: Optional[Tuple[str, str, str]] = None
    for r in pass_rows:
        run_raw = str(r.get("run", ""))
        cond_raw = str(r.get("condition", ""))
        ch = str(r.get("channel", "")).upper()
        counts_raw = CHANNEL_TITLE_LABEL.get(ch, ch)
        group = (run_raw, cond_raw, counts_raw)
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")

        run_id = _latex_escape(run_raw)
        cond = _latex_escape(_condition_display_label(cond_raw))
        ch_label = _latex_escape(counts_raw)
        pass_id = _latex_escape(str(r.get("pass", "")))
        total_fringe, total_fringe_sigma = validate_total_fringe_values(
            r.get("total_fringe"),
            r.get("total_fringe_sigma"),
            r.get("Fringe1"),
            r.get("Fringe2"),
            context=(
                f"{r.get('condition', '')} {ch} pass {r.get('pass', '')} table row"
            ),
        )
        vals = [
            _fmt_pm_latex(r.get("A1"), r.get("A1_sigma"), nd=1),
            _fmt_pm_latex(r.get("A2"), r.get("A2_sigma"), nd=1),
            _fmt_pm_latex(r.get("V"), r.get("V_sigma"), nd=3),
            _fmt_pm_latex(total_fringe, total_fringe_sigma, nd=1),
        ]
        lines.append(f"{run_id} & {cond} & {ch_label} & {pass_id} & " + " & ".join(vals) + r" \\")
        prev_group = group
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_joint_json(
    rows: List[Dict[str, Any]],
    conditions: List[Tuple[str, Path]],
    order: List[Tuple[str, str]],
    out_json_path: Path,
) -> None:
    report_rows = _expand_rows_for_report(rows, conditions, order=order)
    joint_rows = [r for r in report_rows if str(r.get("pass", "")) == "joint"]

    joint_fits: List[Dict[str, Any]] = []
    for r in joint_rows:
        f1 = _json_num(r.get("Fringe1"))
        f2 = _json_num(r.get("Fringe2"))
        s1 = _json_num(r.get("Fringe1_sigma"))
        s2 = _json_num(r.get("Fringe2_sigma"))
        total, total_sigma = validate_total_fringe_values(
            r.get("total_fringe"),
            r.get("total_fringe_sigma"),
            f1,
            f2,
            context=f"{r.get('condition', '')} {r.get('channel', '')} joint JSON record",
        )

        joint_fits.append(
            {
                "condition": str(r.get("condition", "")),
                "channel": str(r.get("channel", "")).upper(),
                "A1_cps": _json_num(r.get("A1")),
                "A1_sigma_cps": _json_num(r.get("A1_sigma")),
                "A2_cps": _json_num(r.get("A2")),
                "A2_sigma_cps": _json_num(r.get("A2_sigma")),
                "V": _json_num(r.get("V")),
                "V_sigma": _json_num(r.get("V_sigma")),
                "P_V": _json_num(r.get("P")),
                "P_sigma_V": _json_num(r.get("P_sigma")),
                "fringe1_cps": f1,
                "fringe1_sigma_cps": s1,
                "fringe2_cps": f2,
                "fringe2_sigma_cps": s2,
                "total_fringe_cps": _json_num(total),
                "total_fringe_sigma_cps": _json_num(total_sigma),
                "x0_V": _json_num(r.get("x0")),
            }
        )

    payload = {"joint_fits": joint_fits}
    out_json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )



def _safe_median(vals) -> float:
    try:
        arr = [float(v) for v in vals if np.isfinite(v)]
    except Exception:
        arr = []
    if not arr:
        return float("nan")
    return float(np.median(np.asarray(arr, dtype=float)))


def _joint_fringe_bands(
    joint_fit: Optional[JointFitResult], channel: int
) -> Optional[Tuple[float, float]]:
    """
    Return (mean, upper_bound) for fringe bands.
    mean = A
    upper_bound = A + A*V (peak)
    """
    if not joint_fit:
        return None
    V = joint_fit.V
    A = joint_fit.A1 if int(channel) == 1 else joint_fit.A2
    if not (np.isfinite(V) and np.isfinite(A)):
        return None
    mean = float(A)
    upper = float(A * (1.0 + V))
    return mean, upper


def fringe_height_and_sigma(
    src: FitResult | JointFitResult, channel: int
) -> Tuple[float, float]:
    """
    Compute fringe amplitude H = A*V and require its covariance-based uncertainty
    Fringe{1,2}_sigma from the fit. The analysis requires covariance-derived
    Fringe{1,2}_sigma values.
    Channel 1 uses A1, channel 2 uses A2.
    """
    channel = int(channel)
    if channel == 1:
        A = src.A1
        sH = src.Fringe1_sigma
        sigma_key = "Fringe1_sigma"
    elif channel == 2:
        A = src.A2
        sH = src.Fringe2_sigma
        sigma_key = "Fringe2_sigma"
    else:
        raise ValueError("fringe channel must be 1 or 2")
    V = src.V
    if not (np.isfinite(A) and np.isfinite(V)):
        raise ValueError("fringe amplitude and visibility must be finite")
    if not np.isfinite(sH) or sH < 0.0:
        raise ValueError(f"{sigma_key} must be finite and nonnegative")
    H = A * V
    return float(H), float(sH)


def _select_row_center(
    joint_fit: Optional[JointFitResult],
    per_pass_rows: List[Dict[str, Any]],
    *,
    joint_key: str,
    row_key: str,
) -> float:
    """
    Choose a panel center value with fallback order:
    1) joint all-pass fit value,
    2) finite mean across per-pass row values,
    3) NaN when unavailable.
    """
    if joint_fit is not None:
        v = joint_fit.A1 if joint_key == "A1" else joint_fit.A2
        if np.isfinite(v):
            return v

    vals: List[float] = []
    for r in per_pass_rows:
        try:
            v = float(r.get(row_key, np.nan))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass

    if vals:
        return float(np.mean(np.asarray(vals, dtype=float)))
    return float("nan")


def _median_value_index(vals) -> Optional[int]:
    """
    Return the index of the finite entry closest to the finite median value.
    Returns None if no finite values exist.
    """
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return None
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    idx = np.where(finite)[0]
    vals_f = arr[finite]
    med = float(np.median(vals_f))
    rel = np.abs(vals_f - med)
    return int(idx[int(np.argmin(rel))])


def _plot_joint_median_x0_overlay(
    ax1,
    ax2,
    passes,
    *,
    A1: float,
    A2: float,
    V: float,
    P: float,
    x0_list,
    color: str,
    label: str,
    linestyle: str = "-",
    x_offset: float = 0.0,
) -> None:
    """
    Plot a single joint-fit overlay using the pass whose x0 is nearest
    to the median x0 across finite entries.
    """
    k = _median_value_index(x0_list)
    if k is None:
        return
    if k < 0 or k >= len(passes):
        return
    dp = passes[k]
    xs = np.asarray(getattr(dp, "xs", []), dtype=float)
    if xs.size == 0:
        return
    x0 = float(np.asarray(x0_list, dtype=float)[k])
    xg = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    xg_plot = _shift_x(xg, x_offset)
    y1g, y2g = _eval_idler_model(xg, A1, A2, V, x0, P)
    ax1.plot(
        xg_plot,
        y1g,
        linestyle=linestyle,
        color=color,
        alpha=1.0,
        linewidth=JOINT_LINEWIDTH,
        zorder=10,
        label=label,
    )
    ax2.plot(
        xg_plot,
        y2g,
        linestyle=linestyle,
        color=color,
        alpha=1.0,
        linewidth=JOINT_LINEWIDTH,
        zorder=10,
    )


def fit_joint_coincidences(
    passes, P_fixed: Optional[float] = None, init: Optional[Dict[str, Any]] = None
) -> JointFitResult:
    """
    Joint weighted LS fit across multiple passes for coincidences with per-pass x0.
    Shared across passes: A1, A2, V, (P if not fixed). Each pass gets its own x0_k.
    Returns the joint fit and raises when the requested fit is invalid.
    """
    P_fixed = _validate_optional_fixed_period(
        P_fixed,
        context="joint coincidence fit",
    )
    if not passes:
        raise ValueError("joint coincidence fit requires at least one pass")

    for k, dp in enumerate(passes):
        x = np.asarray(getattr(dp, "xs", []), dtype=float)
        y1 = np.asarray(getattr(dp, "rc1", []), dtype=float)
        y2 = np.asarray(getattr(dp, "rc2", []), dtype=float)
        if x.ndim != 1 or y1.ndim != 1 or y2.ndim != 1:
            raise ValueError(
                f"joint coincidence pass {k + 1} arrays must be one-dimensional"
            )
        if x.size <= 0:
            raise ValueError(f"joint coincidence pass {k + 1} must not be empty")
        if y1.size != x.size or y2.size != x.size:
            raise ValueError(
                f"joint coincidence pass {k + 1} arrays have inconsistent lengths"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y1)) and np.all(np.isfinite(y2))):
            raise ValueError(
                f"joint coincidence pass {k + 1} coordinates and measurements must be finite"
            )
        validate_positive_variances(
            np.asarray(getattr(dp, "var_rc1", []), dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 1 variances",
        )
        validate_positive_variances(
            np.asarray(getattr(dp, "var_rc2", []), dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 2 variances",
        )

    K = len(passes)
    _validate_joint_initialization(
        init,
        pass_count=K,
        context="joint coincidence fit",
    )
    # Initial values
    A1_0 = (
        float(init.get("A1"))
        if (init and np.isfinite(init.get("A1", np.nan)))
        else _safe_median([np.nanmean(np.asarray(dp.rc1, dtype=float)) for dp in passes])
    )
    A2_0 = (
        float(init.get("A2"))
        if (init and np.isfinite(init.get("A2", np.nan)))
        else _safe_median([np.nanmean(np.asarray(dp.rc2, dtype=float)) for dp in passes])
    )
    V_0 = float(init.get("V")) if (init and np.isfinite(init.get("V", np.nan))) else 0.5
    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0:
        P_0 = float(P_fixed)
    else:
        P_0 = (
            float(init.get("P"))
            if (init and np.isfinite(init.get("P", np.nan)))
            else 0.5 * (float(P_FIT_MIN) + float(P_FIT_MAX))
        )
    x0_list_0 = None
    if init and "x0_list" in init:
        init_x0 = init["x0_list"]
        if not isinstance(init_x0, (list, tuple)) or len(init_x0) != K:
            raise ValueError("joint coincidence initial x0_list has the wrong length")
        x0_list_0 = [float(x) for x in init_x0]
        if not np.all(np.isfinite(x0_list_0)):
            raise ValueError("joint coincidence initial x0_list must be finite")
    if x0_list_0 is None:
        x0_list_0 = []
        for dp in passes:
            xs = np.asarray(getattr(dp, "xs", []), dtype=float)
            if xs.size:
                x0_list_0.append(float(xs[0]))
            else:
                x0_list_0.append(0.0)

    # Pack parameters: [log(A1), log(A2), u=logit(V), x0_1..x0_K, (log(P) if free)]
    def pack(A1, A2, V, x0_list, P):
        u = float(logit(np.clip(float(V), 1e-6, 1.0 - 1e-6)))
        theta = [np.log(max(float(A1), 1e-12)), np.log(max(float(A2), 1e-12)), u]
        theta.extend([float(x0) for x0 in x0_list])
        if not (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0):
            theta.append(np.log(max(float(P), 1e-12)))
        return np.asarray(theta, dtype=float)

    def unpack(theta):
        a1 = float(np.exp(theta[0]))
        a2 = float(np.exp(theta[1]))
        u = float(theta[2])
        V = float(expit(u))
        x0_end = 3 + K
        x0_list = [float(z) for z in theta[3:x0_end]]
        if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0:
            P = float(P_fixed)
        else:
            g = float(theta[3 + K])
            P = float(np.exp(g))
        return a1, a2, V, x0_list, P

    def objective(theta):
        A1, A2, V, x0_list, P = unpack(theta)
        if not (np.isfinite(P) and P > 0.0):
            return np.inf
        sse = 0.0
        for k, dp in enumerate(passes):
            x = np.asarray(dp.xs, dtype=float)
            y1 = np.asarray(dp.rc1, dtype=float)
            y2 = np.asarray(dp.rc2, dtype=float)
            v1 = validate_positive_variances(
                np.asarray(dp.var_rc1, dtype=float),
                expected_size=x.size,
                context=f"joint coincidence pass {k + 1} exit 1 variances",
            )
            v2 = validate_positive_variances(
                np.asarray(dp.var_rc2, dtype=float),
                expected_size=x.size,
                context=f"joint coincidence pass {k + 1} exit 2 variances",
            )
            w1 = 1.0 / v1
            w2 = 1.0 / v2
            wx = 2.0 * np.pi * (x - float(x0_list[k])) / float(P)
            m1 = A1 * (1.0 + V * np.cos(wx))
            m2 = A2 * (1.0 - V * np.cos(wx))
            sse += 0.5 * (np.sum(w1 * (y1 - m1) ** 2) + np.sum(w2 * (y2 - m2) ** 2))
        return float(sse)

    theta0 = pack(
        A1_0 if np.isfinite(A1_0) else 1.0, A2_0 if np.isfinite(A2_0) else 1.0, V_0, x0_list_0, P_0
    )
    bounds = None
    if not (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0):
        # bounds only for log(P)
        bounds = [(None, None)] * (3 + K) + [(np.log(float(P_FIT_MIN)), np.log(float(P_FIT_MAX)))]
    res = minimize(
        objective, theta0, method="L-BFGS-B", bounds=bounds if bounds is not None else None
    )
    params = _require_successful_optimizer_result(
        res,
        expected_size=theta0.size,
        context="joint coincidence fit",
    )
    A1, A2, V, x0_list, P = unpack(params)
    transformed = {
        "A1": A1,
        "A2": A2,
        "V": V,
        "P": P,
        **{f"x0[{index}]": value for index, value in enumerate(x0_list)},
    }
    _require_finite_fit_parameters(
        transformed,
        positive=("A1", "A2", "P"),
        unit_interval=("V",),
        context="joint coincidence fit",
    )

    # Build analytic covariance via stacked Jacobian across all passes
    m = 3 + K + (0 if (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0) else 1)
    idx_a1, idx_a2, idx_u = 0, 1, 2
    idx_x0_base = 3
    idx_g = None if m == (3 + K) else (3 + K)

    total_pts = int(sum(len(dp.xs) for dp in passes))
    Jw = np.zeros((2 * total_pts, m), dtype=float)
    row = 0
    two_pi_over_P = 2.0 * np.pi / float(P)
    for k, dp in enumerate(passes):
        x = np.asarray(dp.xs, dtype=float)
        v1 = validate_positive_variances(
            np.asarray(dp.var_rc1, dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 1 variances",
        )
        v2 = validate_positive_variances(
            np.asarray(dp.var_rc2, dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 2 variances",
        )
        sw1 = 1.0 / np.sqrt(v1)
        sw2 = 1.0 / np.sqrt(v2)
        x0k = float(x0_list[k])
        for i in range(x.size):
            wx = 2.0 * np.pi * (x[i] - x0k) / float(P)
            c = np.cos(wx)
            s = np.sin(wx)
            r1 = np.zeros(m, dtype=float)
            r1[idx_a1] = A1 * (1.0 + V * c)
            r1[idx_u] = A1 * (V * (1.0 - V)) * c
            r1[idx_x0_base + k] = A1 * V * two_pi_over_P * s
            if idx_g is not None:
                r1[idx_g] = A1 * V * (2.0 * np.pi * (x[i] - x0k) / float(P)) * s
            Jw[row, :] = r1 * sw1[i]
            row += 1

            r2 = np.zeros(m, dtype=float)
            r2[idx_a2] = A2 * (1.0 - V * c)
            r2[idx_u] = -A2 * (V * (1.0 - V)) * c
            r2[idx_x0_base + k] = -A2 * V * two_pi_over_P * s
            if idx_g is not None:
                r2[idx_g] = -A2 * V * (2.0 * np.pi * (x[i] - x0k) / float(P)) * s
            Jw[row, :] = r2 * sw2[i]
            row += 1

    Cov = _covariance_from_weighted_jacobian(
        Jw,
        context="joint coincidence fit",
    )

    # Propagated counting variances set the uncertainty floor; residual
    # overdispersion may enlarge, but never deflate, the covariance.
    SSE_w = 0.0
    for k, dp in enumerate(passes):
        x = np.asarray(dp.xs, dtype=float)
        y1 = np.asarray(dp.rc1, dtype=float)
        y2 = np.asarray(dp.rc2, dtype=float)
        v1 = validate_positive_variances(
            np.asarray(dp.var_rc1, dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 1 variances",
        )
        v2 = validate_positive_variances(
            np.asarray(dp.var_rc2, dtype=float),
            expected_size=x.size,
            context=f"joint coincidence pass {k + 1} exit 2 variances",
        )
        wx = 2.0 * np.pi * (x - float(x0_list[k])) / float(P)
        m1 = A1 * (1.0 + V * np.cos(wx))
        m2 = A2 * (1.0 - V * np.cos(wx))
        r1w = (y1 - m1) / np.sqrt(v1)
        r2w = (y2 - m2) / np.sqrt(v2)
        SSE_w += float(np.sum(r1w * r1w) + np.sum(r2w * r2w))
    df = int(2 * total_pts - m)
    Cov = _conservative_covariance_scale(SSE_w, df) * Cov

    var_a1 = float(Cov[idx_a1, idx_a1])
    var_a2 = float(Cov[idx_a2, idx_a2])
    var_u = float(Cov[idx_u, idx_u])
    A1_sigma = A1 * _standard_deviation_from_variance(
        var_a1,
        context="joint coincidence fit exit 1 amplitude",
    )
    A2_sigma = A2 * _standard_deviation_from_variance(
        var_a2,
        context="joint coincidence fit exit 2 amplitude",
    )
    V_sigma = V * (1.0 - V) * _standard_deviation_from_variance(
        var_u,
        context="joint coincidence fit visibility",
    )
    if idx_g is not None:
        P_sigma = P * _standard_deviation_from_variance(
            float(Cov[idx_g, idx_g]),
            context="joint coincidence fit period",
        )
    else:
        P_sigma = float("nan")

    total_fringe = (A1 + A2) * V
    grad = np.zeros(m, dtype=float)
    grad[idx_a1] = A1 * V
    grad[idx_a2] = A2 * V
    grad[idx_u] = (A1 + A2) * V * (1.0 - V)
    total_fringe_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint coincidence fit total fringe",
    )

    grad = np.zeros(m, dtype=float)
    grad[idx_a1] = V * A1
    grad[idx_u] = A1 * V * (1.0 - V)
    Fringe1_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint coincidence fit exit 1 fringe",
    )

    grad[:] = 0.0
    grad[idx_a2] = V * A2
    grad[idx_u] = A2 * V * (1.0 - V)
    Fringe2_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint coincidence fit exit 2 fringe",
    )

    return JointFitResult(
        A1=float(A1),
        A2=float(A2),
        V=float(V),
        P=float(P),
        x0_list=[float(z) for z in x0_list],
        A1_sigma=float(A1_sigma),
        A2_sigma=float(A2_sigma),
        V_sigma=float(V_sigma),
        P_sigma=float(P_sigma),
        total_fringe=float(total_fringe),
        total_fringe_sigma=float(total_fringe_sigma),
        Fringe1_sigma=float(Fringe1_sigma),
        Fringe2_sigma=float(Fringe2_sigma),
    )


def fit_joint_singles(
    passes, T_list, dark, P_fixed: Optional[float] = None, init: Optional[Dict[str, Any]] = None
) -> JointFitResult:
    """
    Joint Poisson MLE fit across multiple passes for idler singles with per-pass x0.
    Shared across passes: A1, A2, r1_dark, r2_dark, V, (P if not fixed).
    Each pass gets its own x0_k.
    Returns the joint fit and raises when the requested fit is invalid.
    """
    P_fixed = _validate_optional_fixed_period(P_fixed, context="joint singles fit")
    if not passes:
        raise ValueError("joint singles fit requires at least one pass")
    if len(T_list) != len(passes):
        raise ValueError("joint singles passes and durations have inconsistent lengths")

    durations = np.asarray(T_list, dtype=float)
    if durations.ndim != 1 or not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("joint singles durations must be one-dimensional, finite, and positive")
    for k, dp in enumerate(passes):
        x = np.asarray(getattr(dp, "xs", []), dtype=float)
        N1 = np.asarray(getattr(dp, "n_i1", []), dtype=float)
        N2 = np.asarray(getattr(dp, "n_i2", []), dtype=float)
        r1 = np.asarray(getattr(dp, "ri1", []), dtype=float)
        r2 = np.asarray(getattr(dp, "ri2", []), dtype=float)
        if any(values.ndim != 1 for values in (x, N1, N2, r1, r2)):
            raise ValueError(f"joint singles pass {k + 1} arrays must be one-dimensional")
        if x.size <= 0:
            raise ValueError(f"joint singles pass {k + 1} must not be empty")
        if any(values.size != x.size for values in (N1, N2, r1, r2)):
            raise ValueError(f"joint singles pass {k + 1} arrays have inconsistent lengths")
        if not all(np.all(np.isfinite(values)) for values in (x, N1, N2, r1, r2)):
            raise ValueError(f"joint singles pass {k + 1} arrays must be finite")
        if np.any(N1 < 0.0) or np.any(N2 < 0.0):
            raise ValueError(f"joint singles pass {k + 1} counts must be nonnegative")

    K = len(passes)
    _validate_joint_initialization(
        init,
        pass_count=K,
        context="joint singles fit",
    )
    # Initial values
    A1_0 = (
        float(init.get("A1"))
        if (init and np.isfinite(init.get("A1", np.nan)))
        else _safe_median([np.nanmean(np.asarray(dp.ri1, dtype=float)) for dp in passes])
    )
    A2_0 = (
        float(init.get("A2"))
        if (init and np.isfinite(init.get("A2", np.nan)))
        else _safe_median([np.nanmean(np.asarray(dp.ri2, dtype=float)) for dp in passes])
    )
    V_0 = float(init.get("V")) if (init and np.isfinite(init.get("V", np.nan))) else 0.5
    if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0:
        P_0 = float(P_fixed)
    else:
        P_0 = (
            float(init.get("P"))
            if (init and np.isfinite(init.get("P", np.nan)))
            else 0.5 * (float(P_FIT_MIN) + float(P_FIT_MAX))
        )
    x0_list_0 = None
    if init and "x0_list" in init:
        init_x0 = init["x0_list"]
        if not isinstance(init_x0, (list, tuple)) or len(init_x0) != K:
            raise ValueError("joint singles initial x0_list has the wrong length")
        x0_list_0 = [float(x) for x in init_x0]
        if not np.all(np.isfinite(x0_list_0)):
            raise ValueError("joint singles initial x0_list must be finite")
    if x0_list_0 is None:
        x0_list_0 = []
        for dp in passes:
            xs = np.asarray(getattr(dp, "xs", []), dtype=float)
            if xs.size:
                x0_list_0.append(float(xs[0]))
            else:
                x0_list_0.append(0.0)

    try:
        r1d_0 = float(dark.r_i)
        r2d_0 = float(dark.r_i2)
        Td = float(dark.T)
        Ni_d = float(dark.Ni)
        Ni2_d = float(dark.Ni2)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "joint singles dark rates, duration, and counts must be provided"
        ) from exc
    if not np.isfinite(r1d_0) or not np.isfinite(r2d_0) or r1d_0 < 0.0 or r2d_0 < 0.0:
        raise ValueError("joint singles dark rates must be finite and nonnegative")
    if not np.isfinite(Td) or Td <= 0.0:
        raise ValueError("joint singles dark duration must be finite and positive")
    if not np.isfinite(Ni_d) or not np.isfinite(Ni2_d) or Ni_d < 0.0 or Ni2_d < 0.0:
        raise ValueError("joint singles dark counts must be finite and nonnegative")

    # Pack parameters: [log(A1), log(r1_dark), log(A2), log(r2_dark), u=logit(V),
    # x0_1..x0_K, (log(P) if free)]
    def pack(A1, r1d, A2, r2d, V, x0_list, P):
        u = float(logit(np.clip(float(V), 1e-6, 1.0 - 1e-6)))
        theta = [
            np.log(max(float(A1), 1e-12)),
            np.log(max(float(r1d), 1e-12)),
            np.log(max(float(A2), 1e-12)),
            np.log(max(float(r2d), 1e-12)),
            u,
        ]
        theta.extend([float(x0) for x0 in x0_list])
        if not (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0):
            theta.append(np.log(max(float(P), 1e-12)))
        return np.asarray(theta, dtype=float)

    def unpack(theta):
        a1 = float(np.exp(theta[0]))
        d1 = float(np.exp(theta[1]))
        a2 = float(np.exp(theta[2]))
        d2 = float(np.exp(theta[3]))
        u = float(theta[4])
        V = float(expit(u))
        x0_end = 5 + K
        x0_list = [float(z) for z in theta[5:x0_end]]
        if P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0:
            P = float(P_fixed)
        else:
            g = float(theta[5 + K])
            P = float(np.exp(g))
        return a1, d1, a2, d2, V, x0_list, P

    def objective(theta):
        A1, r1d, A2, r2d, V, x0_list, P = unpack(theta)
        if not (np.isfinite(P) and P > 0.0):
            return np.inf
        nll = 0.0
        for k, dp in enumerate(passes):
            x = np.asarray(dp.xs, dtype=float)
            N1 = np.asarray(dp.n_i1, dtype=float)
            N2 = np.asarray(dp.n_i2, dtype=float)
            T = float(T_list[k])
            wx = 2.0 * np.pi * (x - float(x0_list[k])) / float(P)
            lam1 = T * (r1d + A1 * (1.0 + V * np.cos(wx)))
            lam2 = T * (r2d + A2 * (1.0 - V * np.cos(wx)))
            if (
                np.any(lam1 <= 0.0)
                or np.any(lam2 <= 0.0)
                or not np.all(np.isfinite(lam1))
                or not np.all(np.isfinite(lam2))
            ):
                return np.inf
            nll += float(np.sum(lam1 - N1 * np.log(lam1)) + np.sum(lam2 - N2 * np.log(lam2)))
        # Dark likelihood (once per channel)
        lamd1 = Td * r1d
        lamd2 = Td * r2d
        if (
            lamd1 <= 0.0
            or lamd2 <= 0.0
            or not np.isfinite(lamd1)
            or not np.isfinite(lamd2)
        ):
            return np.inf
        nll += float(lamd1 - Ni_d * np.log(lamd1) + lamd2 - Ni2_d * np.log(lamd2))
        return nll

    theta0 = pack(
        A1_0 if np.isfinite(A1_0) else 1.0,
        max(r1d_0, 1e-6),
        A2_0 if np.isfinite(A2_0) else 1.0,
        max(r2d_0, 1e-6),
        V_0,
        x0_list_0,
        P_0,
    )
    bounds = None
    if not (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0.0):
        # bounds only for log(P)
        bounds = [(None, None)] * (5 + K) + [(np.log(float(P_FIT_MIN)), np.log(float(P_FIT_MAX)))]
    res = minimize(
        objective, theta0, method="L-BFGS-B", bounds=bounds if bounds is not None else None
    )
    params = _require_successful_optimizer_result(
        res,
        expected_size=theta0.size,
        context="joint singles fit",
    )
    A1, r1d, A2, r2d, V, x0_list, P = unpack(params)
    transformed = {
        "A1": A1,
        "dark_rate1": r1d,
        "A2": A2,
        "dark_rate2": r2d,
        "V": V,
        "P": P,
        **{f"x0[{index}]": value for index, value in enumerate(x0_list)},
    }
    _require_finite_fit_parameters(
        transformed,
        positive=("A1", "dark_rate1", "A2", "dark_rate2", "P"),
        unit_interval=("V",),
        context="joint singles fit",
    )

    # Use the analytic expected Fisher information for a numerically stable
    # covariance estimate.
    idx_a1, idx_a2, idx_u = 0, 2, 4
    idx_g = None
    if not (P_fixed is not None and np.isfinite(P_fixed) and P_fixed > 0):
        idx_g = 5 + K
    m = 5 + K + (1 if idx_g is not None else 0)
    information = np.zeros((m, m), dtype=float)

    for k, dp in enumerate(passes):
        x = np.asarray(dp.xs, dtype=float)
        T = float(T_list[k])
        wx = 2.0 * np.pi * (x - float(x0_list[k])) / float(P)
        c = np.cos(wx)
        s = np.sin(wx)

        lam1 = T * (r1d + A1 * (1.0 + V * c))
        if np.any(lam1 <= 0.0) or not np.all(np.isfinite(lam1)):
            raise ValueError("joint singles expected exit 1 counts must be finite and positive")
        jac1 = np.zeros((x.size, m), dtype=float)
        jac1[:, idx_a1] = T * A1 * (1.0 + V * c)
        jac1[:, 1] = T * r1d
        jac1[:, idx_u] = T * A1 * V * (1.0 - V) * c
        jac1[:, 5 + k] = T * A1 * V * (2.0 * np.pi / P) * s
        if idx_g is not None:
            jac1[:, idx_g] = T * A1 * V * wx * s
        information += jac1.T @ (jac1 / lam1[:, np.newaxis])

        lam2 = T * (r2d + A2 * (1.0 - V * c))
        if np.any(lam2 <= 0.0) or not np.all(np.isfinite(lam2)):
            raise ValueError("joint singles expected exit 2 counts must be finite and positive")
        jac2 = np.zeros((x.size, m), dtype=float)
        jac2[:, idx_a2] = T * A2 * (1.0 - V * c)
        jac2[:, 3] = T * r2d
        jac2[:, idx_u] = -T * A2 * V * (1.0 - V) * c
        jac2[:, 5 + k] = -T * A2 * V * (2.0 * np.pi / P) * s
        if idx_g is not None:
            jac2[:, idx_g] = -T * A2 * V * wx * s
        information += jac2.T @ (jac2 / lam2[:, np.newaxis])

    # The dark observations constrain the two fitted dark-rate parameters.
    information[1, 1] += Td * r1d
    information[3, 3] += Td * r2d
    information = 0.5 * (information + information.T)
    if not np.all(np.isfinite(information)):
        raise ValueError(
            "joint singles covariance is not identifiable "
            "(expected Fisher information is nonfinite)"
        )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "joint singles covariance is not identifiable "
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
            "joint singles covariance is not identifiable "
            f"(minimum Fisher eigenvalue {float(eigenvalues[0]):.6g}, "
            f"tolerance {tolerance:.6g})"
        )
    Cov = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    if not np.all(np.isfinite(Cov)):
        raise ValueError(
            "joint singles covariance is not identifiable "
            "(inverse expected Fisher information is nonfinite)"
        )

    var_a1 = float(Cov[idx_a1, idx_a1])
    var_a2 = float(Cov[idx_a2, idx_a2])
    var_u = float(Cov[idx_u, idx_u])

    A1_sigma = A1 * _standard_deviation_from_variance(
        var_a1,
        context="joint singles fit exit 1 amplitude",
    )
    A2_sigma = A2 * _standard_deviation_from_variance(
        var_a2,
        context="joint singles fit exit 2 amplitude",
    )
    V_sigma = V * (1.0 - V) * _standard_deviation_from_variance(
        var_u,
        context="joint singles fit visibility",
    )
    if idx_g is not None:
        P_sigma = P * _standard_deviation_from_variance(
            float(Cov[idx_g, idx_g]),
            context="joint singles fit period",
        )
    else:
        P_sigma = float("nan")

    total_fringe = (A1 + A2) * V
    grad = np.zeros(m, dtype=float)
    grad[idx_a1] = A1 * V
    grad[idx_a2] = A2 * V
    grad[idx_u] = (A1 + A2) * V * (1.0 - V)
    total_fringe_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint singles fit total fringe",
    )
    # Fringe amplitude uncertainties via covariance: H = A*V
    grad = np.zeros(m, dtype=float)
    grad[idx_a1] = V * A1
    grad[idx_u] = A1 * V * (1.0 - V)
    Fringe1_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint singles fit exit 1 fringe",
    )

    grad = np.zeros(m, dtype=float)
    grad[idx_a2] = V * A2
    grad[idx_u] = A2 * V * (1.0 - V)
    Fringe2_sigma = _standard_deviation_from_variance(
        float(grad @ Cov @ grad),
        context="joint singles fit exit 2 fringe",
    )

    return JointFitResult(
        A1=float(A1),
        A2=float(A2),
        V=float(V),
        P=float(P),
        x0_list=[float(z) for z in x0_list],
        A1_sigma=float(A1_sigma),
        A2_sigma=float(A2_sigma),
        V_sigma=float(V_sigma),
        P_sigma=float(P_sigma),
        total_fringe=float(total_fringe),
        total_fringe_sigma=float(total_fringe_sigma),
        Fringe1_sigma=float(Fringe1_sigma),
        Fringe2_sigma=float(Fringe2_sigma),
    )


def _load_condition_passes(
    conditions: List[Tuple[str, Path]],
    *,
    max_passes: Optional[int],
    use_first_secs: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, List[List[Dict[str, Any]]]]]:
    dark_by_cond: Dict[str, Any] = {}
    recs_by_cond: Dict[str, List[Dict[str, Any]]] = {}
    passes_by_cond: Dict[str, List[List[Dict[str, Any]]]] = {}

    for cond_name, cond_dir in conditions:
        dark_by_cond[cond_name] = load_dark(cond_dir)
        recs_by_cond[cond_name] = load_points(cond_dir)

    # Optional time-capped truncation of aligned records across all conditions
    if use_first_secs is not None and use_first_secs > 0:
        recs_by_cond = aligned_record_truncation(recs_by_cond, use_first_secs)

    # Split into complete passes using the acquisition plan.
    for cond_name, cond_dir in conditions:
        recs = recs_by_cond.get(cond_name, [])
        passes = split_by_plan_length(recs, load_plan_length(cond_dir))

        if max_passes is not None and max_passes > 0:
            passes = passes[: int(max_passes)]

        passes_by_cond[cond_name] = passes

    return dark_by_cond, passes_by_cond


def _setup_by_pass_figure(
    col_specs: List[Tuple[str, str]],
) -> Tuple[Any, np.ndarray, Dict[Tuple[str, str], Tuple[Any, Any]]]:
    ncols = len(col_specs)
    fig, axes = plt.subplots(
        2, ncols, sharex=True, squeeze=False, figsize=(PRL_DOUBLE_COL_WIDTH, PRL_FIG_HEIGHT)
    )

    ax_by_cond_ch: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
    for col_idx, (cond_name, ch) in enumerate(col_specs):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]
        ax_by_cond_ch[(cond_name, ch)] = (ax_top, ax_bot)

        ch_label = CHANNEL_TITLE_LABEL.get(ch)
        ax_top.set_title(
            f"{cond_name} {ch_label.lower()}", fontsize=TITLE_FONTSIZE, pad=3
        )
        if col_idx == 0:
            ax_top.set_ylabel(
                r"Exit 1 rate ($\mathrm{s}^{\text{--}1}$)", fontsize=AXIS_LABEL_FONTSIZE
            )
            ax_bot.set_ylabel(
                r"Exit 2 rate ($\mathrm{s}^{\text{--}1}$)", fontsize=AXIS_LABEL_FONTSIZE
            )
        else:
            ax_top.set_ylabel("")
            ax_bot.set_ylabel("")

        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_bot.set_xticks(X_TICKS)
        ax_bot.set_xticklabels(X_TICK_LABELS)
        ax_top.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=1)
        ax_bot.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, pad=1)
        ax_bot.tick_params(axis="x", labelsize=X_TICK_LABEL_FONTSIZE, pad=1)
        ax_top.grid(True)
        ax_bot.grid(True)

    center_col = ncols // 2
    center_xlabel = r"$\Delta$ piezo stage voltage (V)"
    for col_idx in range(ncols):
        ax_bot = axes[1, col_idx]
        if col_idx == center_col:
            ax_bot.set_xlabel(center_xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=4)
        else:
            ax_bot.set_xlabel("")

    return fig, axes, ax_by_cond_ch


def _print_fit_report(
    rows: List[Dict[str, Any]],
    conditions: List[Tuple[str, Path]],
    col_specs: List[Tuple[str, str]],
    stats_by_cond: Dict[str, Dict[str, List[float]]],
) -> None:
    def _emit_group_summary(group: Optional[Tuple[str, str]]) -> None:
        if not group:
            return
        cond_name, _ch = group
        st = stats_by_cond.get(cond_name, {})
        Ts = st.get("T") or []
        Ns = st.get("N") or []
        if not Ns:
            return
        secs_pt = float(np.median(np.asarray(Ts, dtype=float))) if Ts else float("nan")
        pts_per_pass = int(round(float(np.median(np.asarray(Ns, dtype=float))))) if Ns else 0
        total_points = int(sum(int(n) for n in Ns))
        total_secs = float(sum(float(t) * float(n) for t, n in zip(Ts, Ns)))
        print(f"secs/point  {_fmt_val(secs_pt)}")
        print(f"points/pass {pts_per_pass}")
        print(f"total points {total_points}")
        print(f"total secs  {_fmt_val(total_secs)}")

    # Print per-pass fit parameter table in narrow (condition, channel) layout
    report_cols = [
        "condition",
        "channel",
        "pass",
        "x0",
        "A1",
        "A2",
        "V",
        "P",
        "Fringe1",
        "Fringe2",
        "total_fringe",
    ]
    report_rows = _expand_rows_for_report(rows, conditions, order=col_specs)

    formatted_rows: List[List[str]] = []
    for r in report_rows:
        row_cells: List[str] = []
        for k in report_cols:
            if k in ("condition", "channel", "pass"):
                row_cells.append(str(r.get(k, "")))
                continue
            if k == "x0":
                row_cells.append(_fmt_val(r.get(k)))
            else:
                row_cells.append(_fmt_pm(r.get(k), r.get(f"{k}_sigma"), nd=3))
        formatted_rows.append(row_cells)

    widths = [len(c) for c in report_cols]
    for fr in formatted_rows:
        for i, cell in enumerate(fr):
            widths[i] = max(widths[i], len(cell))

    print("  ".join(report_cols[i].ljust(widths[i]) for i in range(len(report_cols))))
    prev_group: Optional[Tuple[str, str]] = None
    for fr in formatted_rows:
        group = (fr[0], fr[1])
        if prev_group is not None and group != prev_group:
            _emit_group_summary(prev_group)
            print()
        out_cells = []
        for i, cell in enumerate(fr):
            if report_cols[i] in {"condition", "channel"}:
                out_cells.append(cell.ljust(widths[i]))
            else:
                out_cells.append(cell.rjust(widths[i]))
        print("  ".join(out_cells))
        prev_group = group

    _emit_group_summary(prev_group)


def _finalize_by_pass_figure(
    fig: Any,
    axes: np.ndarray,
    col_specs: List[Tuple[str, str]],
    target_row_center: Dict[Tuple[str, str, int], float],
    target_total_fringe: Dict[Tuple[str, str], float],
) -> None:
    ncols = len(col_specs)

    # Align y-axes globally: one shared span for all panels,
    # centered on A1 (top row) / A2 (bottom row) per panel.
    global_half_spans: List[float] = []
    for row_idx in range(2):
        for col_idx in range(ncols):
            ymin, ymax = axes[row_idx, col_idx].get_ylim()
            global_half_spans.append(0.5 * (ymax - ymin))
    shared_half = 1.03 * max(global_half_spans) if global_half_spans else 1.0

    for col_idx, (cond_name, ch) in enumerate(col_specs):
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            ymin, ymax = ax.get_ylim()
            current_center = 0.5 * (ymin + ymax)
            center = target_row_center.get((cond_name, ch, row_idx), current_center)
            if not np.isfinite(center):
                center = current_center
            ax.set_ylim(center - shared_half, center + shared_half)

    for col_idx, (cond_name, ch) in enumerate(col_specs):
        ax_bot = axes[1, col_idx]
        total_fringe = float(target_total_fringe.get((cond_name, ch), np.nan))
        if np.isfinite(total_fringe):
            cond_symbol = CONDITION_SYMBOL_LABEL.get(cond_name, cond_name)
            ch_symbol = {"C": "c", "I": "s"}.get(ch, ch.lower())
            amplitude_decimals = {"C": 1, "I": 0}[ch]
            amplitude_text = f"{total_fringe:.{amplitude_decimals}f}"
            label_txt = (
                rf"$A_{{{cond_symbol},{ch_symbol}}} = "
                rf"{amplitude_text}\ \mathrm{{s}}^{{\text{{--}}1}}$"
            )
            ax_bot.text(
                0.50,
                0.06,
                label_txt,
                transform=ax_bot.transAxes,
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONTSIZE,
                zorder=25,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#ffffff",
                    edgecolor="#cccccc",
                    linewidth=0,
                    alpha=1,
                ),
            )

    fig.tight_layout(rect=[0, 0.01, 1, 0.99], w_pad=0.05, h_pad=0.2)


def _write_by_pass_outputs(
    fig: Any,
    conditions: List[Tuple[str, Path]],
    *,
    output_dir: Optional[Path],
    outfile_name: str,
    rows: List[Dict[str, Any]],
    table_specs: List[Tuple[str, str, str]],
    col_specs: List[Tuple[str, str]],
    json_filename: str,
) -> None:
    destination = output_dir if output_dir is not None else conditions[0][1]
    destination.mkdir(parents=True, exist_ok=True)
    out_path = (destination / outfile_name).resolve()
    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=_deterministic_figure_metadata("pdf"),
    )
    print(str(out_path))

    png_path = out_path.with_suffix(".png")
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=_deterministic_figure_metadata("png"),
    )
    print(str(png_path))

    tex_path = out_path.parent / "mzi-alt-joint.tex"
    _write_joint_latex_table(rows, conditions, table_specs, tex_path)
    print(str(tex_path))

    json_path = out_path.parent / json_filename
    _write_joint_json(rows, conditions, col_specs, json_path)
    print(str(json_path))

    run_ids = list(dict.fromkeys(run_id for run_id, _, _ in table_specs))
    for run_index, run_id in enumerate(run_ids, start=1):
        run_table_specs = [spec for spec in table_specs if spec[0] == run_id]
        tex_by_pass_path = out_path.parent / f"mzi-alt-by-pass-run-{run_index}.tex"
        _write_by_pass_latex_table(
            rows,
            conditions,
            run_table_specs,
            tex_by_pass_path,
        )
        print(str(tex_by_pass_path))

    plt.close(fig)


def plot_by_pass(
    conditions: List[Tuple[str, Path]],
    outfile_name: str = "mzi-alt-by-pass.pdf",
    output_dir: Optional[Path] = None,
    max_passes: Optional[int] = None,
    min_points: int = 6,
    period: Optional[float] = None,
    use_first_secs: Optional[float] = None,
    order: Optional[List[Tuple[str, str]]] = None,
    table_order: Optional[List[Tuple[str, str, str]]] = None,
    json_filename: str = "mzi-alt-joint.json",
) -> None:
    if not conditions:
        raise ValueError("At least one condition is required")

    dark_by_cond, passes_by_cond = _load_condition_passes(
        conditions,
        max_passes=max_passes,
        use_first_secs=use_first_secs,
    )

    plotted_condition_names = {cond_name for cond_name, _ in (order or [])}
    x_vals_global: List[float] = []
    for cond_name, _ in conditions:
        if cond_name not in plotted_condition_names:
            continue
        for recs_pass in passes_by_cond.get(cond_name, []):
            for rec in recs_pass:
                xv = float(rec["target_delta_V"])
                if np.isfinite(xv):
                    x_vals_global.append(xv)
    x_min_global = float(np.min(np.asarray(x_vals_global, dtype=float))) if x_vals_global else 0.0

    # Build figure: 2 x N panels, one column per condition:channel
    col_specs = list(order or [])
    if not col_specs:
        raise ValueError("At least one --order condition:channel is required")
    table_specs = list(
        table_order or [("", cond_name, channel) for cond_name, channel in col_specs]
    )

    fig, axes, ax_by_cond_ch = _setup_by_pass_figure(col_specs)
    target_row_center: Dict[Tuple[str, str, int], float] = {}
    target_total_fringe: Dict[Tuple[str, str], float] = {}

    # Prepare per-pass parameter table collection
    rows: List[Dict[str, Any]] = []
    # Accumulate per-condition pass stats for report summaries
    stats_by_cond: Dict[str, Dict[str, List[float]]] = {}

    # Plot per-pass fits for each condition
    for cidx, (cond_name, _) in enumerate(conditions):
        passes = passes_by_cond.get(cond_name, [])
        dark = dark_by_cond.get(cond_name)
        if not passes:
            raise ValueError(f"{cond_name} has no complete passes to fit")
        base_col = _condition_color(cidx)
        plot_C = (cond_name, "C") in ax_by_cond_ch
        plot_I = (cond_name, "I") in ax_by_cond_ch
        ax_c1_col, ax_c2_col = ax_by_cond_ch.get((cond_name, "C"), (None, None))
        ax_i1, ax_i2 = ax_by_cond_ch.get((cond_name, "I"), (None, None))
        cond_rows: List[Dict[str, Any]] = []
        # Collections for joint fit
        singles_passes: List[Any] = []
        singles_T: List[float] = []
        coinc_passes: List[Any] = []
        init_I_A1: List[float] = []
        init_I_A2: List[float] = []
        init_I_V: List[float] = []
        init_I_P: List[float] = []
        init_I_x0: List[float] = []
        init_C_A1: List[float] = []
        init_C_A2: List[float] = []
        init_C_V: List[float] = []
        init_C_P: List[float] = []
        init_C_x0: List[float] = []

        for pidx, recs_pass in enumerate(passes):
            # Convert records in this pass into a scan dataset
            assert_constant_point_duration(
                recs_pass, f"{cond_name} pass {pidx+1} (idler singles)"
            )
            data_pass, T_pass = scan_from_records(recs_pass, dark)
            num_pts = len(data_pass.xs)
            if num_pts < int(min_points):
                raise ValueError(
                    f"{cond_name} pass {pidx + 1} has {num_pts} points; "
                    f"at least {int(min_points)} are required"
                )

            # Record per-pass stats for report summaries
            stats = stats_by_cond.setdefault(cond_name, {"T": [], "N": []})
            stats["T"].append(float(T_pass))
            stats["N"].append(int(num_pts))

            # Independent fits per pass (no shared period)
            fit_c = fit_scan(
                data_pass, channel="C", period=period, dark=dark, acq_dur=T_pass
            ).coincidence
            fit_i = fit_scan(
                data_pass, channel="I", period=period, dark=dark, acq_dur=T_pass
            ).singles

            # Prepare arrays for plotting
            xs_i, ri1, ri2, si1, si2 = _prepare_plot_arrays(data_pass, T_pass)
            xs_c, rc1, rc2, sc1, sc2 = _prepare_plot_arrays_coinc(data_pass)
            xs_i_plot = _shift_x(xs_i, x_min_global)
            xs_c_plot = _shift_x(xs_c, x_min_global)

            # Collect data for joint fit
            singles_passes.append(data_pass)
            singles_T.append(T_pass)
            coinc_passes.append(data_pass)

            # Seed inits from per-pass fits
            init_I_A1.append(fit_i.A1)
            init_I_A2.append(fit_i.A2)
            init_I_V.append(fit_i.V)
            init_I_P.append(fit_i.P)
            init_I_x0.append(fit_i.x0)

            init_C_A1.append(fit_c.A1)
            init_C_A2.append(fit_c.A2)
            init_C_V.append(fit_c.V)
            init_C_P.append(fit_c.P)
            init_C_x0.append(fit_c.x0)

            # Collect fit parameters for table with uncertainties
            cA1 = fit_c.A1
            cA2 = fit_c.A2
            cV = fit_c.V
            cP = fit_c.P
            cx0 = fit_c.x0
            cA1_sigma = fit_c.A1_sigma
            cA2_sigma = fit_c.A2_sigma
            cV_sigma = fit_c.V_sigma
            cP_sigma = fit_c.P_sigma
            cTotalFringe, cTotalFringeSigma = _total_fringe_and_sigma(
                fit_c,
                context=f"{cond_name} coincidence pass {pidx + 1}",
            )
            cFr1, cFr1s = fringe_height_and_sigma(fit_c, channel=1)
            cFr2, cFr2s = fringe_height_and_sigma(fit_c, channel=2)

            iA1 = fit_i.A1
            iA2 = fit_i.A2
            iV = fit_i.V
            iP = fit_i.P
            ix0 = fit_i.x0
            iA1_sigma = fit_i.A1_sigma
            iA2_sigma = fit_i.A2_sigma
            iV_sigma = fit_i.V_sigma
            iP_sigma = fit_i.P_sigma
            iTotalFringe, iTotalFringeSigma = _total_fringe_and_sigma(
                fit_i,
                context=f"{cond_name} singles pass {pidx + 1}",
            )
            iFr1, iFr1s = fringe_height_and_sigma(fit_i, channel=1)
            iFr2, iFr2s = fringe_height_and_sigma(fit_i, channel=2)

            row = {
                "condition": cond_name,
                "pass": pidx + 1,
                "C.A1": cA1,
                "C.A2": cA2,
                "C.total_fringe": cTotalFringe,
                "C.Fringe1": cFr1,
                "C.Fringe2": cFr2,
                "C.P": cP,
                "C.V": cV,
                "C.x0": cx0,
                "C.A1_sigma": cA1_sigma,
                "C.A2_sigma": cA2_sigma,
                "C.total_fringe_sigma": cTotalFringeSigma,
                "C.Fringe1_sigma": cFr1s,
                "C.Fringe2_sigma": cFr2s,
                "C.P_sigma": cP_sigma,
                "C.V_sigma": cV_sigma,
                "I.A1": iA1,
                "I.A2": iA2,
                "I.total_fringe": iTotalFringe,
                "I.Fringe1": iFr1,
                "I.Fringe2": iFr2,
                "I.P": iP,
                "I.V": iV,
                "I.x0": ix0,
                "I.A1_sigma": iA1_sigma,
                "I.A2_sigma": iA2_sigma,
                "I.total_fringe_sigma": iTotalFringeSigma,
                "I.Fringe1_sigma": iFr1s,
                "I.Fringe2_sigma": iFr2s,
                "I.P_sigma": iP_sigma,
                "I.V_sigma": iV_sigma,
            }
            rows.append(row)
            cond_rows.append(row)

            # Choose a tinted color per pass
            col = to_rgba(base_col, alpha=LINE_OPACITY)

            # Plot singles and coincidences for this pass
            if plot_I:
                _plot_series(
                    ax_i1,
                    ax_i2,
                    xs_i_plot,
                    ri1,
                    ri2,
                    si1,
                    si2,
                    None,
                    col,
                    "_nolegend_",
                )
                x_fit_raw = np.asarray(fit_i.x_fit, dtype=float)
                x_fit = _shift_x(x_fit_raw, x_min_global)
                y1_fit = np.asarray(fit_i.y_fit1, dtype=float)
                y2_fit = np.asarray(fit_i.y_fit2, dtype=float)
                if x_fit.size and y1_fit.size == x_fit.size:
                    ax_i1.plot(x_fit, y1_fit, "--", color=col, label=None)
                if x_fit.size and y2_fit.size == x_fit.size:
                    ax_i2.plot(x_fit, y2_fit, "--", color=col, label=None)
            if plot_C:
                _plot_series_coinc(
                    ax_c1_col,
                    ax_c2_col,
                    xs_c_plot,
                    rc1,
                    rc2,
                    sc1,
                    sc2,
                    None,
                    col,
                    "_nolegend_",
                )
                x_fit_raw = np.asarray(fit_c.x_fit, dtype=float)
                x_fit = _shift_x(x_fit_raw, x_min_global)
                y1_fit = np.asarray(fit_c.y_fit1, dtype=float)
                y2_fit = np.asarray(fit_c.y_fit2, dtype=float)
                if x_fit.size and y1_fit.size == x_fit.size:
                    ax_c1_col.plot(x_fit, y1_fit, "-", color=col, label=None)
                if x_fit.size and y2_fit.size == x_fit.size:
                    ax_c2_col.plot(x_fit, y2_fit, "-", color=col, label=None)

        # Joint overlays and summary row
        init_I = {
            "A1": float(np.median(init_I_A1)),
            "A2": float(np.median(init_I_A2)),
            "V": float(np.median(init_I_V)),
            "P": float(period) if period is not None else float(np.median(init_I_P)),
            "x0_list": list(init_I_x0),
        }
        joint_I = fit_joint_singles(
            singles_passes, singles_T, dark, P_fixed=period, init=init_I
        )
        if plot_I:
            _plot_joint_median_x0_overlay(
                ax_i1,
                ax_i2,
                singles_passes,
                A1=joint_I.A1,
                A2=joint_I.A2,
                V=joint_I.V,
                P=joint_I.P,
                x0_list=joint_I.x0_list,
                color=base_col,
                label=None,
                linestyle="--",
                x_offset=x_min_global,
            )

        init_C = {
            "A1": float(np.median(init_C_A1)),
            "A2": float(np.median(init_C_A2)),
            "V": float(np.median(init_C_V)),
            "P": float(period) if period is not None else float(np.median(init_C_P)),
            "x0_list": list(init_C_x0),
        }
        joint_C = fit_joint_coincidences(coinc_passes, P_fixed=period, init=init_C)
        if plot_C:
            _plot_joint_median_x0_overlay(
                ax_c1_col,
                ax_c2_col,
                coinc_passes,
                A1=joint_C.A1,
                A2=joint_C.A2,
                V=joint_C.V,
                P=joint_C.P,
                x0_list=joint_C.x0_list,
                color=base_col,
                label=None,
                x_offset=x_min_global,
            )

        target_row_center[(cond_name, "C", 0)] = _select_row_center(
            joint_C, cond_rows, joint_key="A1", row_key="C.A1"
        )
        target_row_center[(cond_name, "C", 1)] = _select_row_center(
            joint_C, cond_rows, joint_key="A2", row_key="C.A2"
        )
        target_row_center[(cond_name, "I", 0)] = _select_row_center(
            joint_I, cond_rows, joint_key="A1", row_key="I.A1"
        )
        target_row_center[(cond_name, "I", 1)] = _select_row_center(
            joint_I, cond_rows, joint_key="A2", row_key="I.A2"
        )

        for joint, cond_tag in ((joint_C, "C"), (joint_I, "I")):
            total, _ = _total_fringe_and_sigma(
                joint,
                context=f"{cond_name} {cond_tag} joint plot annotation",
            )
            target_total_fringe[(cond_name, cond_tag)] = total

        if plot_C:
            bands_c1 = _joint_fringe_bands(joint_C, channel=1)
            bands_c2 = _joint_fringe_bands(joint_C, channel=2)
            if bands_c1 is not None:
                mean, upper = bands_c1
                ax_c1_col.axhspan(
                    mean,
                    upper,
                    facecolor=BAND_FACE_UPPER,
                    alpha=BAND_ALPHA,
                    zorder=Z_BAND,
                    linewidth=0,
                )
            if bands_c2 is not None:
                mean, upper = bands_c2
                ax_c2_col.axhspan(
                    mean,
                    upper,
                    facecolor=BAND_FACE_UPPER,
                    alpha=BAND_ALPHA,
                    zorder=Z_BAND,
                    linewidth=0,
                )
        if plot_I:
            bands_i1 = _joint_fringe_bands(joint_I, channel=1)
            bands_i2 = _joint_fringe_bands(joint_I, channel=2)
            if bands_i1 is not None:
                mean, upper = bands_i1
                ax_i1.axhspan(
                    mean,
                    upper,
                    facecolor=BAND_FACE_UPPER,
                    alpha=BAND_ALPHA,
                    zorder=Z_BAND,
                    linewidth=0,
                )
            if bands_i2 is not None:
                mean, upper = bands_i2
                ax_i2.axhspan(
                    mean,
                    upper,
                    facecolor=BAND_FACE_UPPER,
                    alpha=BAND_ALPHA,
                    zorder=Z_BAND,
                    linewidth=0,
                )

        # Per-condition "joint" summary row
        totalI, totalI_sigma = _total_fringe_and_sigma(
            joint_I,
            context=f"{cond_name} singles joint row",
        )
        totalC, totalC_sigma = _total_fringe_and_sigma(
            joint_C,
            context=f"{cond_name} coincidence joint row",
        )
        x0_list_i = np.asarray(joint_I.x0_list, dtype=float)
        x0_list_c = np.asarray(joint_C.x0_list, dtype=float)
        k_i = _median_value_index(x0_list_i)
        k_c = _median_value_index(x0_list_c)
        if k_i is None or k_c is None:
            raise ValueError(f"{cond_name} joint fits must contain finite phase values")
        iFr1, iFr1s = fringe_height_and_sigma(joint_I, channel=1)
        iFr2, iFr2s = fringe_height_and_sigma(joint_I, channel=2)
        cFr1, cFr1s = fringe_height_and_sigma(joint_C, channel=1)
        cFr2, cFr2s = fringe_height_and_sigma(joint_C, channel=2)
        joint_row: Dict[str, Any] = {
            "condition": cond_name,
            "pass": "joint",
            "I.A1": joint_I.A1,
            "I.A2": joint_I.A2,
            "I.V": joint_I.V,
            "I.P": joint_I.P,
            "I.total_fringe": totalI,
            "I.x0": float(x0_list_i[k_i]),
            "I.A1_sigma": joint_I.A1_sigma,
            "I.A2_sigma": joint_I.A2_sigma,
            "I.V_sigma": joint_I.V_sigma,
            "I.P_sigma": joint_I.P_sigma,
            "I.total_fringe_sigma": totalI_sigma,
            "I.Fringe1": iFr1,
            "I.Fringe1_sigma": iFr1s,
            "I.Fringe2": iFr2,
            "I.Fringe2_sigma": iFr2s,
            "C.A1": joint_C.A1,
            "C.A2": joint_C.A2,
            "C.V": joint_C.V,
            "C.P": joint_C.P,
            "C.total_fringe": totalC,
            "C.x0": float(x0_list_c[k_c]),
            "C.A1_sigma": joint_C.A1_sigma,
            "C.A2_sigma": joint_C.A2_sigma,
            "C.V_sigma": joint_C.V_sigma,
            "C.P_sigma": joint_C.P_sigma,
            "C.total_fringe_sigma": totalC_sigma,
            "C.Fringe1": cFr1,
            "C.Fringe1_sigma": cFr1s,
            "C.Fringe2": cFr2,
            "C.Fringe2_sigma": cFr2s,
        }
        rows.append(joint_row)

    _print_fit_report(rows, conditions, col_specs, stats_by_cond)
    _finalize_by_pass_figure(
        fig,
        axes,
        col_specs,
        target_row_center,
        target_total_fringe,
    )
    _write_by_pass_outputs(
        fig,
        conditions,
        output_dir=output_dir,
        outfile_name=outfile_name,
        rows=rows,
        table_specs=table_specs,
        col_specs=col_specs,
        json_filename=json_filename,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot alternating MZI data with independent per-pass fits"
    )
    p.add_argument(
        "--condition",
        action="append",
        required=True,
        help=(
            "Condition spec in the form name:path/to/runs/subdir. "
            "Repeat --condition for multiple conditions."
        ),
    )
    p.add_argument(
        "--order",
        action="append",
        default=[],
        help=(
            "Panels to plot, left-to-right, as condition:C or condition:I. "
            "Repeat --order for each panel."
        ),
    )
    p.add_argument(
        "--table-order",
        action="append",
        default=[],
        help=(
            "Rows to include in the TeX tables, as run:condition:C or "
            "run:condition:I. Repeat for each row group. Defaults to --order "
            "with a blank run identifier."
        ),
    )
    p.add_argument(
        "--outfile",
        default="mzi-alt-by-pass.pdf",
        help="Output filename.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory. Default: first condition directory.",
    )
    p.add_argument(
        "--json",
        default="mzi-alt-joint.json",
        help="JSON output filename for joint-fit summaries.",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=None,
        help="Limit to the first N passes per condition.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=6,
        help="Minimum number of points required to fit/plot a pass (default: 6).",
    )
    p.add_argument(
        "--period",
        type=float,
        default=None,
        help="Force all fits to use this period (V). If omitted, period is fit per pass.",
    )
    p.add_argument(
        "--use-first-secs",
        type=float,
        default=None,
        help=(
            "Only use the first M index-aligned records such that cumulative times are <= N "
            "seconds for every condition."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conditions = parse_condition_specs(args.condition)
    valid_names = [name for name, _ in conditions]
    order = _parse_order_specs(args.order, valid_names)
    table_order = _parse_table_order_specs(args.table_order, valid_names)
    plot_by_pass(
        conditions=conditions,
        outfile_name=args.outfile,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        max_passes=args.max_passes,
        min_points=args.min_points,
        period=args.period,
        use_first_secs=args.use_first_secs,
        order=order,
        table_order=table_order,
        json_filename=args.json,
    )


if __name__ == "__main__":
    main()
