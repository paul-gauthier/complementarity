#!/usr/bin/env python

import matplotlib.colors as mcolors
import numpy as np

chunk_marker = "."
chunk_ms = 3

# Styling constants for visual hierarchy
CHUNK_ALPHA = 0.75
Z_CHUNK = 1


def _fmt_pm(val: float, sigma: float, nd: int = 3) -> str:
    if val is None or not np.isfinite(val):
        return "nan"
    if sigma is not None and np.isfinite(sigma):
        return f"{val:.{nd}f} ± {sigma:.{nd}f}"
    return f"{val:.{nd}f}"


def _tint(color, amount: float = 0.5):
    """
    Blend the given color toward white by the specified amount in [0,1].
    Returns an RGBA tuple suitable for Matplotlib.
    """
    r, g, b, a = mcolors.to_rgba(color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return (r, g, b, a)


def _prepare_plot_arrays(data, acq_dur: float):
    xs = np.asarray(data.xs, dtype=float)
    ri1 = np.asarray(data.ri1, dtype=float)
    ri2 = np.asarray(data.ri2, dtype=float)
    n_i1 = np.asarray(data.n_i1, dtype=float)
    n_i2 = np.asarray(data.n_i2, dtype=float)

    if xs.size:
        order = np.argsort(xs)
        xs = xs[order]
        ri1 = ri1[order]
        ri2 = ri2[order]
        n_i1 = n_i1[order]
        n_i2 = n_i2[order]

    sigma_i1 = np.sqrt(np.maximum(n_i1, 0.0)) / float(acq_dur) if xs.size else None
    sigma_i2 = np.sqrt(np.maximum(n_i2, 0.0)) / float(acq_dur) if xs.size else None
    return xs, ri1, ri2, sigma_i1, sigma_i2


def _prepare_plot_arrays_coinc(data):
    xs = np.asarray(data.xs, dtype=float)
    rc1 = np.asarray(data.rc1, dtype=float)
    rc2 = np.asarray(data.rc2, dtype=float)
    var_rc1 = np.asarray(data.var_rc1, dtype=float)
    var_rc2 = np.asarray(data.var_rc2, dtype=float)

    if xs.size:
        order = np.argsort(xs)
        xs = xs[order]
        rc1 = rc1[order]
        rc2 = rc2[order]
        var_rc1 = var_rc1[order]
        var_rc2 = var_rc2[order]

    sigma_rc1 = np.sqrt(np.maximum(var_rc1, 0.0)) if xs.size else None
    sigma_rc2 = np.sqrt(np.maximum(var_rc2, 0.0)) if xs.size else None
    return xs, rc1, rc2, sigma_rc1, sigma_rc2


def _plot_series(ax1, ax2, xs, ri1, ri2, sigma_i1, sigma_i2, fit_i, color: str, label_prefix: str):
    # Points with dashed line + markers
    pt_color = _tint(color, 0.5)
    if xs.size:
        ax1.plot(
            xs,
            ri1,
            linestyle="None",
            marker=chunk_marker,
            color=pt_color,
            alpha=CHUNK_ALPHA,
            ms=(chunk_ms - 1 if chunk_ms > 1 else chunk_ms),
            zorder=Z_CHUNK,
            label=None,
        )
        ax2.plot(
            xs,
            ri2,
            linestyle="None",
            marker=chunk_marker,
            color=pt_color,
            alpha=CHUNK_ALPHA,
            ms=(chunk_ms - 1 if chunk_ms > 1 else chunk_ms),
            zorder=Z_CHUNK,
            label=None,
        )

    # Fit lines (solid)
    if fit_i is not None and getattr(fit_i, "x_fit", None) is not None:
        x_fit = np.asarray(fit_i.x_fit, dtype=float)
        y1_fit = np.asarray(fit_i.y_fit1, dtype=float)
        y2_fit = np.asarray(fit_i.y_fit2, dtype=float)
        if x_fit.size and y1_fit.size == x_fit.size:
            ax1.plot(x_fit, y1_fit, "-", color=color, label=f"{label_prefix} fit I1")
        if x_fit.size and y2_fit.size == x_fit.size:
            ax2.plot(x_fit, y2_fit, "-", color=color, label=f"{label_prefix} fit I2")


def _plot_series_coinc(
    ax_c1, ax_c2, xs, rc1, rc2, sigma_rc1, sigma_rc2, fit_c, color: str, label_prefix: str
):
    # Points with markers
    pt_color = _tint(color, 0.5)
    if xs.size:
        ax_c1.plot(
            xs,
            rc1,
            linestyle="None",
            marker=chunk_marker,
            color=pt_color,
            alpha=CHUNK_ALPHA,
            ms=(chunk_ms - 1 if chunk_ms > 1 else chunk_ms),
            zorder=Z_CHUNK,
            label=None,
        )
        ax_c2.plot(
            xs,
            rc2,
            linestyle="None",
            marker=chunk_marker,
            color=pt_color,
            alpha=CHUNK_ALPHA,
            ms=(chunk_ms - 1 if chunk_ms > 1 else chunk_ms),
            zorder=Z_CHUNK,
            label=None,
        )

    # Fit lines (solid)
    if fit_c is not None and getattr(fit_c, "x_fit", None) is not None:
        x_fit = np.asarray(fit_c.x_fit, dtype=float)
        y1_fit = np.asarray(fit_c.y_fit1, dtype=float)
        y2_fit = np.asarray(fit_c.y_fit2, dtype=float)
        if x_fit.size and y1_fit.size == x_fit.size:
            ax_c1.plot(x_fit, y1_fit, "-", color=color, label=f"{label_prefix} fit C1")
        if x_fit.size and y2_fit.size == x_fit.size:
            ax_c2.plot(x_fit, y2_fit, "-", color=color, label=f"{label_prefix} fit C2")
