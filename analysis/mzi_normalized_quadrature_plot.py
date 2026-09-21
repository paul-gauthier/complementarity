#!/usr/bin/env python3
"""Render normalized launch-only quadrature confidence regions for all datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox
from scipy.stats import chi2

from .artifact_config import load_config
from .dataset_colors import DATASET_COLORS
from .matplotlib_pdf_determinism import make_pdf_font_subsets_deterministic

make_pdf_font_subsets_deterministic()

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.latex.preamble": (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{amsmath,amssymb,amsfonts}"
        ),
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

APS_COLUMN_WIDTH_IN = 3.4
INITIAL_HEIGHT_IN = 1.85
PLOT_SIZE_IN = 1.35
AXIS_LABEL_FONTSIZE = 8
TICK_LABEL_FONTSIZE = 8
LEGEND_FONTSIZE = 7
POOLED_DISK_COLOR = "0.55"
POOLED_DISK_ALPHA = 0.45


class PlotInputError(ValueError):
    """Raised when the pooled quadrature plot input is incomplete or invalid."""


@dataclass(frozen=True)
class NormalizedQuadrature:
    dataset_id: str
    label: str
    center: np.ndarray
    covariance: np.ndarray


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotInputError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PlotInputError(f"{field} must be finite")
    return number


def _array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PlotInputError(f"{field} must be numeric") from exc
    if result.shape != shape:
        raise PlotInputError(f"{field} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise PlotInputError(f"{field} must be finite")
    return result


def conservative_upper_bound(value: float, decimals: int) -> float:
    """Round a nonnegative upper endpoint upward to the displayed precision."""
    if not math.isfinite(value) or value < 0.0:
        raise PlotInputError("upper bounds must be finite and nonnegative")
    if decimals < 0:
        raise PlotInputError("upper-bound decimals must be nonnegative")
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_CEILING)
    return float(rounded)


def pooled_bound_label(confidence: float, upper_bound: float) -> str:
    confidence_percent = 100.0 * confidence
    return (
        rf"Pooled {confidence_percent:.0f}\% upper bound,"
        "\n"
        rf"$\eta_\infty<{upper_bound:.2f}$"
    )


def standard_quantum_label() -> str:
    return "Standard quantum theory,\n$\\eta_\\infty=0$"


def full_restoration_label() -> str:
    return "Full fringe restoration,\n$\\eta_\\infty=1$"


def dataset_confidence_title(confidence: float) -> str:
    return (
        rf"Per-dataset {100.0 * confidence:.0f}\% joint"
        "\n"
        "confidence regions"
    )


def normalize_inputs(
    payload: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float, list[NormalizedQuadrature]]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise PlotInputError("pooled input is missing its summary")
    confidence = _finite_number(summary.get("confidence"), "summary.confidence")
    if not 0.5 < confidence < 1.0:
        raise PlotInputError("summary.confidence must lie between 0.5 and 1")
    eta_upper = _finite_number(summary.get("eta_upper"), "summary.eta_upper")
    if not 0.0 <= eta_upper <= 1.0:
        raise PlotInputError("summary.eta_upper must lie between 0 and 1")
    displayed_eta_upper = conservative_upper_bound(eta_upper, 2)

    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list):
        raise PlotInputError("pooled input is missing its input records")
    by_dataset: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw_inputs):
        if not isinstance(record, dict):
            raise PlotInputError(f"inputs[{index}] must be an object")
        dataset_id = record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise PlotInputError(f"inputs[{index}].dataset_id must be a string")
        if dataset_id in by_dataset:
            raise PlotInputError(f"duplicate pooled input: {dataset_id}")
        by_dataset[dataset_id] = record

    configured_ids = [dataset["id"] for dataset in config["datasets"]]
    if set(by_dataset) != set(configured_ids):
        raise PlotInputError("pooled inputs do not match configured datasets")

    normalized: list[NormalizedQuadrature] = []
    for dataset in config["datasets"]:
        dataset_id = dataset["id"]
        record = by_dataset[dataset_id]
        q_hat = _array(record.get("q_hat_cps"), (2,), f"{dataset_id}.q_hat_cps")
        covariance = _array(
            record.get("covariance_cps2"),
            (2, 2),
            f"{dataset_id}.covariance_cps2",
        )
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise PlotInputError(f"{dataset_id}.covariance_cps2 must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise PlotInputError(
                f"{dataset_id}.covariance_cps2 must be positive definite"
            )
        normalization = _finite_number(
            record.get("normalization_Rinfty_cps"),
            f"{dataset_id}.normalization_Rinfty_cps",
        )
        if normalization <= 0.0:
            raise PlotInputError(
                f"{dataset_id}.normalization_Rinfty_cps must be positive"
            )
        normalized.append(
            NormalizedQuadrature(
                dataset_id=dataset_id,
                label=dataset["display_id"],
                center=q_hat / normalization,
                covariance=covariance / normalization**2,
            )
        )
    return confidence, displayed_eta_upper, normalized


def load_plot_inputs(
    json_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[float, float, list[NormalizedQuadrature]]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotInputError(f"cannot load {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlotInputError(f"{json_path} must contain a JSON object")
    return normalize_inputs(payload, config or load_config())


def ellipse_geometry(
    item: NormalizedQuadrature,
    confidence: float,
) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(item.covariance)
    cutoff = float(chi2.ppf(confidence, df=2))
    major = 2.0 * math.sqrt(float(eigenvalues[1]) * cutoff)
    minor = 2.0 * math.sqrt(float(eigenvalues[0]) * cutoff)
    angle = math.degrees(
        math.atan2(float(eigenvectors[1, 1]), float(eigenvectors[0, 1]))
    )
    return major, minor, angle


def _tick_label(value: float, _position: Any) -> str:
    if abs(abs(value) - 0.5) < 1e-9:
        return ""
    if abs(value) < 1e-9:
        return r"$0$"
    magnitude = f"{abs(value):g}"
    if value < 0:
        return rf"$\text{{--}}{magnitude}$"
    return rf"${magnitude}$"


def centered_column_crop(content: Bbox) -> Bbox:
    if content.width > APS_COLUMN_WIDTH_IN:
        raise PlotInputError(
            "plot content is wider than the configured APS column width"
        )
    center_x = 0.5 * (content.x0 + content.x1)
    half_width = 0.5 * APS_COLUMN_WIDTH_IN
    # Tight extents omit stroke widths and can underestimate TeX glyph bounds
    # across renderers. Keep both vertical edges clear without scaling the plot.
    vertical_padding = 0.5 / 72.0
    return Bbox.from_extents(
        center_x - half_width,
        content.y0 - vertical_padding,
        center_x + half_width,
        content.y1 + vertical_padding,
    )


def render_plot(
    confidence: float,
    pooled_upper_bound: float,
    inputs: Sequence[NormalizedQuadrature],
    out_base: Path,
) -> tuple[Path, Path]:
    if len(inputs) != len(DATASET_COLORS):
        raise PlotInputError(
            f"expected {len(DATASET_COLORS)} configured datasets, found {len(inputs)}"
        )

    fig = plt.figure(figsize=(APS_COLUMN_WIDTH_IN, INITIAL_HEIGHT_IN))
    ax = fig.add_axes(
        [
            0.32 / APS_COLUMN_WIDTH_IN,
            0.31 / INITIAL_HEIGHT_IN,
            PLOT_SIZE_IN / APS_COLUMN_WIDTH_IN,
            PLOT_SIZE_IN / INITIAL_HEIGHT_IN,
        ]
    )
    ax.set_aspect("equal")
    limit = 1.08
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.84", lw=0.4, ls=":", zorder=0)

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            color="k",
            lw=0.9,
            ls=":",
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            pooled_upper_bound,
            facecolor=POOLED_DISK_COLOR,
            edgecolor="none",
            alpha=POOLED_DISK_ALPHA,
            zorder=2,
        )
    )

    dataset_handles: list[Line2D] = []
    for item in inputs:
        color = DATASET_COLORS[item.label]
        width, height, angle = ellipse_geometry(item, confidence)
        ax.add_patch(
            Ellipse(
                item.center,
                width,
                height,
                angle=angle,
                facecolor="none",
                edgecolor=color,
                lw=1.0,
                zorder=3,
            )
        )
        ax.plot(
            *item.center,
            marker="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.25,
            ms=3.0,
            ls="none",
            zorder=5,
        )
        dataset_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color=color,
                lw=0.9,
                ms=3.0,
                label=item.label,
            )
        )

    ax.plot(0.0, 0.0, "+", color="k", ms=5.5, mew=0.9, zorder=6)
    reference_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            marker="+",
            color="k",
            ls="none",
            ms=6,
            mew=0.9,
            label=standard_quantum_label(),
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=POOLED_DISK_COLOR,
            markeredgecolor="none",
            color="none",
            alpha=POOLED_DISK_ALPHA,
            ls="none",
            ms=8.5,
            label=pooled_bound_label(confidence, pooled_upper_bound),
        ),
        Line2D(
            [],
            [],
            color="k",
            lw=0.9,
            ls=":",
            label=full_restoration_label(),
        ),
    ]
    legend_left = 1.866 / APS_COLUMN_WIDTH_IN
    reference_legends = [
        fig.legend(
            handles=[handle],
            loc="upper left",
            bbox_to_anchor=(legend_left, 1.0),
            bbox_transform=fig.transFigure,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            handlelength=1.25,
            handletextpad=0.45,
            borderpad=0.0,
            borderaxespad=0.0,
        )
        for handle in reference_handles
    ]
    # Matplotlib fills multi-column legends down each column. Interleave the
    # handles so the visible rows read D1--D3 and D4--D6.
    dataset_order = (0, 3, 1, 4, 2, 5)
    dataset_legend = fig.legend(
        handles=[dataset_handles[index] for index in dataset_order],
        loc="upper left",
        bbox_to_anchor=(legend_left, 1.0),
        bbox_transform=fig.transFigure,
        ncol=3,
        title=dataset_confidence_title(confidence),
        title_fontsize=LEGEND_FONTSIZE,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.70,
        labelspacing=0.55,
        borderpad=0.0,
        borderaxespad=0.0,
        alignment="left",
    )

    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    formatter = FuncFormatter(_tick_label)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=1)
    ax.set_xlabel(
        r"$C_{{\rm add}}^{(d)}/R_{\infty}^{(d)}$",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=2,
    )
    ax.set_ylabel(
        r"$S_{{\rm add}}^{(d)}/R_{\infty}^{(d)}$",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=2,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_chunks = [*reference_legends, dataset_legend]
    chunk_boxes = [legend.get_window_extent(renderer) for legend in legend_chunks]
    plot_box = ax.get_window_extent(renderer)
    chunk_gap = (plot_box.height - sum(box.height for box in chunk_boxes)) / (
        len(chunk_boxes) - 1
    )
    chunk_top = plot_box.y1
    for legend, box in zip(legend_chunks, chunk_boxes):
        legend.set_bbox_to_anchor(
            (legend_left, chunk_top / fig.bbox.height),
            transform=fig.transFigure,
        )
        chunk_top -= box.height + chunk_gap
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_px = ax.get_tightbbox(renderer)
    legend_px = Bbox.union(
        [legend.get_window_extent(renderer) for legend in legend_chunks]
    )
    # The legend layout box reserves unused descent below its final row.
    # Let the corrected x-axis-label extent set the lower crop instead.
    content_px = Bbox.from_extents(
        min(axes_px.x0, legend_px.x0),
        axes_px.y0,
        max(axes_px.x1, legend_px.x1),
        max(axes_px.y1, legend_px.y1),
    )
    content_inches = content_px.transformed(fig.dpi_scale_trans.inverted())
    column_crop = centered_column_crop(content_inches)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
    fig.savefig(
        pdf_path,
        bbox_inches=column_crop,
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
        bbox_inches=column_crop,
        pad_inches=0,
        metadata={"Software": "complementarity artifact"},
    )
    plt.close(fig)
    return pdf_path, png_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render normalized quadrature confidence regions for all datasets."
        )
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Pooled-analysis JSON containing the dataset quadratures.",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        help="Output path base; .pdf and .png are both written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    json_path = Path(args.json).expanduser().resolve()
    out_base = Path(args.outfile).expanduser().resolve()
    if out_base.suffix.lower() in {".pdf", ".png"}:
        out_base = out_base.with_suffix("")
    confidence, pooled_upper_bound, inputs = load_plot_inputs(json_path)
    pdf_path, png_path = render_plot(
        confidence,
        pooled_upper_bound,
        inputs,
        out_base,
    )
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
