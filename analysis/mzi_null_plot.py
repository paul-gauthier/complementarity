#!/usr/bin/env python
"""
Render the launch-only fringe exclusion plot from the mzi-null-bounds.json
summary written by mzi_null_bounds.py.

The plot shows the best-fit launch-only quadratures (C_add, S_add) with 68%
and confidence-level joint ellipses from the full fit covariance, the
any-phase upper-limit circle, and the radius predicted by full restoration of
surviving-pair coherence (eta = 1).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox
from scipy.stats import chi2

from .matplotlib_pdf_determinism import make_pdf_font_subsets_deterministic

make_pdf_font_subsets_deterministic()

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb,amsfonts}",
        "axes.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)

PRL_SINGLE_COL_WIDTH = 3.4
BLUE = "#2b6ca3"
DARK_BLUE = "#173f5f"
AXIS_LABEL_FONTSIZE = 8
TICK_LABEL_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8
LEGEND_FONTSIZE = 8


class PlotInputError(ValueError):
    """Raised when the null-plot scientific input is incomplete or invalid."""


@dataclass(frozen=True)
class NullPlotInputs:
    qhat: np.ndarray
    covariance: np.ndarray
    upper_bound: float
    normalization: float
    confidence: float
    full_restoration_distance: float
    full_restoration_nearest_q: np.ndarray


def _required_finite_number(summary: Dict[str, Any], field: str) -> float:
    qualified_field = f"summary.{field}"
    if field not in summary:
        raise PlotInputError(f"{qualified_field} is required")
    value = summary[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotInputError(f"{qualified_field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PlotInputError(f"{qualified_field} must be finite")
    return number


def _parse_plot_inputs(summary: Dict[str, Any]) -> NullPlotInputs:
    C_hat = _required_finite_number(summary, "CL_hat_cps")
    S_hat = _required_finite_number(summary, "SL_hat_cps")
    sC = _required_finite_number(summary, "CL_sigma_cps")
    sS = _required_finite_number(summary, "SL_sigma_cps")
    if sC <= 0.0:
        raise PlotInputError("summary.CL_sigma_cps must be positive")
    if sS <= 0.0:
        raise PlotInputError("summary.SL_sigma_cps must be positive")

    A95 = _required_finite_number(summary, "B95_any_phase_cps")
    R_inf = _required_finite_number(summary, "Rinfty_cps")
    if R_inf <= 0.0:
        raise PlotInputError("summary.Rinfty_cps must be positive")

    cov_cs = _required_finite_number(summary, "CLSL_cov_cps2")
    Sigma = np.array([[sC * sC, cov_cs], [cov_cs, sS * sS]], dtype=float)
    if np.any(np.linalg.eigvalsh(Sigma) <= 0.0):
        raise PlotInputError(
            "summary.CLSL_cov_cps2 and marginal uncertainties must form "
            "a positive-definite covariance matrix"
        )

    nsig = _required_finite_number(
        summary,
        "full_restoration_mahalanobis_distance",
    )
    if nsig < 0.0:
        raise PlotInputError(
            "summary.full_restoration_mahalanobis_distance must be nonnegative"
        )
    nearest_q = np.array(
        [
            _required_finite_number(summary, "full_restoration_nearest_C_cps"),
            _required_finite_number(summary, "full_restoration_nearest_S_cps"),
        ],
        dtype=float,
    )
    confidence = _required_finite_number(summary, "confidence")
    if not 0.5 < confidence < 1.0:
        raise PlotInputError("summary.confidence must lie between 0.5 and 1")

    return NullPlotInputs(
        qhat=np.array([C_hat, S_hat], dtype=float),
        covariance=Sigma,
        upper_bound=A95,
        normalization=R_inf,
        confidence=confidence,
        full_restoration_distance=nsig,
        full_restoration_nearest_q=nearest_q,
    )


def _ceil_to_places(x: float, places: int) -> float:
    if not np.isfinite(x):
        return x
    scale = 10**places
    return math.ceil((x - 1e-12) * scale) / scale


def _fmt_upper(x: float, places: int) -> str:
    return f"{_ceil_to_places(x, places):.{places}f}"


def _tex_text_minus_tick(x: float, _pos: Any) -> str:
    if abs(x) < 1e-9:
        return r"$0$"
    v = int(round(abs(x)))
    if x < 0:
        return rf"$\text{{--}}{v}$"
    return rf"${v}$"


def _load_summary(json_path: Path) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise PlotInputError(f"No 'summary' section found in {json_path}")
    return summary


def render_null_plot(summary: Dict[str, Any], out_base: Path) -> Tuple[Path, Path]:
    inputs = _parse_plot_inputs(summary)
    qhat = inputs.qhat
    Sigma = inputs.covariance
    A95 = inputs.upper_bound
    R_inf = inputs.normalization
    cl = inputs.confidence
    nsig = inputs.full_restoration_distance
    nearest_q = inputs.full_restoration_nearest_q

    chi2_inner = float(chi2.ppf(0.68, df=2))
    chi2_outer = float(chi2.ppf(cl, df=2))
    cl_label = f"{100.0 * cl:.0f}"
    eta_ul = A95 / R_inf

    evals, evecs = np.linalg.eigh(Sigma)
    ang = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))

    def conf_ellipse(c2: float, **kw: Any) -> Ellipse:
        return Ellipse(
            qhat,
            2.0 * np.sqrt(evals[1] * c2),
            2.0 * np.sqrt(evals[0] * c2),
            angle=ang,
            **kw,
        )

    lim = max(1.25 * R_inf, 60.0)

    fig = plt.figure(figsize=(PRL_SINGLE_COL_WIDTH, 1.65))
    ax = fig.add_axes([0.08, 0.235, 0.38, 0.68])
    ax_leg = fig.add_axes([0.465, 0.15, 0.4, 0.86])
    ax_leg.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.82", lw=0.4, ls=":", zorder=0)

    ax.add_patch(Circle((0, 0), R_inf, fill=False, color="k", lw=0.9, ls=":", zorder=2))
    ax.add_patch(Circle((0, 0), A95, fill=False, color="0.25", lw=0.8, zorder=3))
    ax.add_patch(conf_ellipse(chi2_outer, fc=BLUE, ec=BLUE, alpha=0.18, lw=0.6, zorder=4))
    ax.add_patch(conf_ellipse(chi2_inner, fc=BLUE, ec=BLUE, alpha=0.40, lw=0.5, zorder=5))
    ax.plot(*qhat, "o", color=DARK_BLUE, ms=2, zorder=6)
    ax.plot(0, 0, "+", color="k", ms=5, mew=0.9, zorder=6)

    direction = nearest_q - qhat
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 0.0:
        direction /= direction_norm
        precision = np.linalg.inv(Sigma)
        directional_precision = float(direction @ precision @ direction)
        ellipse_distance = math.sqrt(chi2_outer / directional_precision)
        start = qhat + direction * (ellipse_distance + 0.02 * R_inf)
        end = nearest_q - direction * (0.026 * R_inf)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.6),
        )
        mid = 0.5 * (start + end)
        perp = np.array([direction[1], -direction[0]], dtype=float)
        txt_pos = mid + 0.06 * R_inf * perp + 0.018 * R_inf * np.array([3, 3])
        ax.text(
            txt_pos[0],
            txt_pos[1],
            rf"${nsig:.1f}\sigma$",
            fontsize=ANNOTATION_FONTSIZE,
            ha="left",
            va="center",
            color="0.2",
        )

    handles = [
        Line2D(
            [],
            [],
            marker="+",
            color="k",
            ls="none",
            ms=6,
            mew=0.9,
            label="Standard quantum theory",
        ),
        Line2D([], [], marker="o", color=DARK_BLUE, ls="none", ms=3, label="Best fit"),
        Patch(
            fc=BLUE,
            ec=BLUE,
            alpha=0.30,
            label=f"68\\% and {cl_label}\\% confidence",
        ),
        Line2D([], [], color="k", lw=0.9, ls=":", label="Full restoration, " + r"$\eta_\infty=1$"),
        Line2D(
            [],
            [],
            color="0.25",
            lw=0.8,
            label=rf"{cl_label}\% confidence upper bound,"
            + "\n"
            + rf"$\eta_{{\infty,{cl_label}}}<{_fmt_upper(eta_ul, 2)}$",
        ),
    ]
    leg = ax_leg.legend(
        handles=handles,
        loc="center left",
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=1,
        borderaxespad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_linespacing(1.25)
    leg.set_zorder(10)

    ticks = [-60, -30, 0, 30, 60]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    tick_fmt = FuncFormatter(_tex_text_minus_tick)
    ax.xaxis.set_major_formatter(tick_fmt)
    ax.yaxis.set_major_formatter(tick_fmt)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=1)
    labelpad = 3
    ax.set_xlabel(
        r"$C_{\rm add}\ (\mathrm{s}^{\text{--}1})$", fontsize=AXIS_LABEL_FONTSIZE, labelpad=labelpad
    )
    ax.set_ylabel(
        r"$S_{\rm add}\ (\mathrm{s}^{\text{--}1})$", fontsize=AXIS_LABEL_FONTSIZE, labelpad=labelpad
    )

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    # Crop tightly around visible content only while preserving the exact column width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    content_px = Bbox.union(
        [
            ax.get_tightbbox(renderer),
            leg.get_window_extent(renderer),
        ]
    )
    content_inches = content_px.transformed(fig.dpi_scale_trans.inverted())
    vertical_crop = Bbox.from_extents(
        0.0,
        content_inches.y0,
        PRL_SINGLE_COL_WIDTH,
        content_inches.y1,
    )
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
    fig.savefig(
        pdf_path,
        bbox_inches=vertical_crop,
        pad_inches=0,
        metadata={
            "Creator": "complementarity artifact",
            "Producer": "Matplotlib",
            "CreationDate": timestamp,
            "ModDate": timestamp,
        },
    )
    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches=vertical_crop,
        pad_inches=0,
        metadata={"Software": "complementarity artifact"},
    )
    plt.close(fig)
    return pdf_path, png_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the launch-null exclusion plot from mzi-null-bounds.json."
    )
    p.add_argument(
        "--json",
        required=True,
        help="Path to the mzi-null-bounds.json written by mzi_null_bounds.py.",
    )
    p.add_argument(
        "--outfile",
        default=None,
        help=(
            "Output path base; .pdf and .png are both written. "
            "Default: mzi-null-plot next to the JSON."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    json_path = Path(args.json).expanduser().resolve()
    summary = _load_summary(json_path)

    if args.outfile:
        out_base = Path(args.outfile).expanduser().resolve()
        if out_base.suffix.lower() in {".pdf", ".png"}:
            out_base = out_base.with_suffix("")
    else:
        out_base = json_path.parent / "mzi-null-plot"

    pdf_path, png_path = render_null_plot(summary, out_base)

    print(str(pdf_path))
    print(str(png_path))


if __name__ == "__main__":
    main()
