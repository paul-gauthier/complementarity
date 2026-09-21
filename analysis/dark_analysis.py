#!/usr/bin/env python

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "COINCIDENCE_WINDOW_S",
    "DarkModel",
    "accidentals_counts",
    "corrected_coincidence_rate",
]

# Global coincidence window (seconds)
COINCIDENCE_WINDOW_S: float = 2.5e-8


@dataclass(frozen=True)
class DarkModel:
    """
    Represents an accumulated dark run and derived quantities.

    Fields:
      - w: coincidence window [s]
      - Ns, Ni, Nc: total dark counts (signal, idler1, coincidences1)
      - Ni2, Nc2: total dark counts for second idler/coincidence channel
      - T: total dark acquisition time [s]
      - r_s, r_i, r_c: dark rates [Hz] for signal, idler1, coincidence1
      - r_i2, r_c2: dark rates [Hz] for idler2/coincidence2
      - dark_excess: r_c - w * r_s * r_i [Hz]
      - dark_excess2: r_c2 - w * r_s * r_i2 [Hz]
      - var_dark_excess: variance of dark_excess [Hz^2]
      - var_dark_excess2: variance of dark_excess2 [Hz^2]
    """

    w: float
    Ns: int
    Ni: int
    Nc: int
    Ni2: int
    Nc2: int
    T: float
    r_s: float
    r_i: float
    r_c: float
    r_i2: float
    r_c2: float
    dark_excess: float
    var_dark_excess: float
    dark_excess2: float
    var_dark_excess2: float

    @staticmethod
    def from_counts(
        Ns: int,
        Ni: int,
        Nc: int,
        T: float,
        Ni2: int,
        Nc2: int,
        w: float = COINCIDENCE_WINDOW_S,
    ) -> "DarkModel":
        raw_counts = {"Ns": Ns, "Ni": Ni, "Nc": Nc, "Ni2": Ni2, "Nc2": Nc2}
        for name, value in raw_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(T, bool) or not isinstance(T, (int, float)):
            raise ValueError("T must be numeric")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError("w must be numeric")
        T = float(T)
        w = float(w)
        if not math.isfinite(T) or T <= 0:
            raise ValueError("T must be finite and positive")
        if not math.isfinite(w) or w <= 0:
            raise ValueError("w must be finite and positive")
        r_s = Ns / T
        r_i = Ni / T
        r_c = Nc / T
        r_i2 = Ni2 / T
        r_c2 = Nc2 / T
        # Accidentals during dark run (rates)
        r_acc = w * r_s * r_i
        r_acc2 = w * r_s * r_i2
        # Dark-excess beyond accidentals
        dark_excess = r_c - r_acc
        dark_excess2 = r_c2 - r_acc2
        # Variances:
        # var(r_c) = var(Nc/T) = Nc / T^2
        var_r_c = Nc / (T * T)
        var_r_c2 = Nc2 / (T * T)
        # var(r_s) ~= Ns / T^2, var(r_i) ~= Ni / T^2, var(r_i2) ~= Ni2 / T^2
        var_r_s = Ns / (T * T)
        var_r_i = Ni / (T * T)
        var_r_i2 = Ni2 / (T * T)
        # r_acc = w * r_s * r_i; r_acc2 = w * r_s * r_i2
        # var via linearization:
        var_r_acc = (w * w) * (r_i * r_i * var_r_s + r_s * r_s * var_r_i)
        var_r_acc2 = (w * w) * (r_i2 * r_i2 * var_r_s + r_s * r_s * var_r_i2)
        var_dark_excess = var_r_c + var_r_acc
        var_dark_excess2 = var_r_c2 + var_r_acc2
        return DarkModel(
            w=w,
            Ns=Ns,
            Ni=Ni,
            Nc=Nc,
            Ni2=Ni2,
            Nc2=Nc2,
            T=T,
            r_s=r_s,
            r_i=r_i,
            r_c=r_c,
            r_i2=r_i2,
            r_c2=r_c2,
            dark_excess=dark_excess,
            var_dark_excess=var_dark_excess,
            dark_excess2=dark_excess2,
            var_dark_excess2=var_dark_excess2,
        )


def accidentals_counts(
    n_s: int, n_i: int, dur: float, w: float = COINCIDENCE_WINDOW_S
) -> Tuple[float, float]:
    """
    Estimate accidental coincidences (counts) and variance for one acquisition window.
      n_acc = (w/dur) * n_s * n_i
      var(n_acc) = (w/dur)^2 * (n_i^2 * n_s + n_s^2 * n_i)
    """
    dur = float(dur)
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError("dur must be finite and positive")
    k = w / dur
    n_s = float(n_s)
    n_i = float(n_i)
    n_acc = k * n_s * n_i
    var_n_acc = (k * k) * ((n_i * n_i) * n_s + (n_s * n_s) * n_i)
    return n_acc, var_n_acc


def corrected_coincidence_rate(
    N_c_tot: int,
    T_tot: float,
    N_acc_tot: float,
    var_acc_tot: float,
    dark: DarkModel,
    channel: str = "I",
) -> Tuple[float, float]:
    """
    Aggregate corrected coincidence rate and its variance.

    For channel "I":
      R_c_corr = (N_c_tot - N_acc_tot - dark.dark_excess * T_tot) / T_tot
      var(R_c_corr) = [N_c_tot + var_acc_tot + (dark.var_dark_excess * T_tot^2)] / T_tot^2

    For channel "I2":
      Replace dark_excess/var_dark_excess with dark_excess2/var_dark_excess2.
    """
    T_tot = float(T_tot)
    if not math.isfinite(T_tot) or T_tot <= 0:
        raise ValueError("T_tot must be finite and positive")

    if channel == "I2":
        dex = dark.dark_excess2
        vdex = dark.var_dark_excess2
    elif channel == "I":
        dex = dark.dark_excess
        vdex = dark.var_dark_excess
    else:
        raise ValueError("channel must be I or I2")

    R_c_corr = (float(N_c_tot) - float(N_acc_tot) - dex * T_tot) / T_tot
    var_R_c_corr = (float(N_c_tot) + float(var_acc_tot) + (vdex * T_tot * T_tot)) / (T_tot * T_tot)
    return R_c_corr, var_R_c_corr
