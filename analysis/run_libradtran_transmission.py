#!/usr/bin/env python3
"""
Compute a simple, conservative atmospheric transmission estimate with libRadtran.

What it does
------------
1. Builds a top-hat filter for 782-867 nm.
2. Builds a uvspec input file for clear-sky zenith direct-beam transmittance.
3. Uses:
      - METAR surface pressure
      - Aerosol optical depth at a reference wavelength + Ångström exponent
      - Total precipitable water
   and ignores cloud optical depth numerically when COD is invalid and METAR says CLR/SKC.
4. Runs uvspec.
5. Produces:
      - a band-averaged transmission T_band over the filter
      - a point estimate near 810 nm, T_810, from the spectral output

Notes
-----
- This is the manuscript's physically based clear-sky transmission model.
- Non-clear scenes are rejected because no quantitative cloud model is implemented.
- Conservative choices here:
    * use the full rectangular 782-867 nm passband
    * report the filter-averaged direct transmittance
- Requires libRadtran installed and `uvspec` available.
- Set LIBRADTRAN_DATA_PATH if needed, e.g.
      export LIBRADTRAN_DATA_PATH=/path/to/libRadtran/data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

# Provenance: the manuscript's "Atmospheric transmission" section derives the
# launched-idler band from SPDC energy conservation. A 405 nm pump and the
# signal-side 800 +/- 40 nm SPAD interference filter map to approximately
# 782--867 nm for the paired idlers. The survival budget uses the top-hat
# average over this full band; 810 nm is the nominal degenerate wavelength and
# is retained as a point-reporting reference.
DEFAULT_LAMBDA_MIN_NM = 782.0
DEFAULT_LAMBDA_MAX_NM = 867.0
DEFAULT_LAMBDA_REF_NM = 810.0

@dataclass(kw_only=True)
class AtmosInputs:
    angstrom_alpha: float
    pressure_hpa: float
    aod_ref: float  # AOD at reference wavelength
    aod_ref_wavelength_um: float  # reference wavelength (microns)
    tpw_mm: float
    metar_sky: str
    cod_status: str  # "valid" or "invalid"
    metar_temp_c: float
    metar_rh_pct: Optional[float] = None
    source_label: str = ""  # e.g. "GOES-18", "AERONET solar"

    def __post_init__(self) -> None:
        if (
            isinstance(self.angstrom_alpha, bool)
            or not isinstance(self.angstrom_alpha, (int, float))
            or not math.isfinite(float(self.angstrom_alpha))
        ):
            raise ValueError("angstrom_alpha must be finite")
        if (
            isinstance(self.pressure_hpa, bool)
            or not isinstance(self.pressure_hpa, (int, float))
            or not math.isfinite(float(self.pressure_hpa))
            or not 800.0 <= float(self.pressure_hpa) <= 1100.0
        ):
            raise ValueError("pressure_hpa must be finite and between 800 and 1100 hPa")
        if (
            isinstance(self.metar_temp_c, bool)
            or not isinstance(self.metar_temp_c, (int, float))
            or not math.isfinite(float(self.metar_temp_c))
            or float(self.metar_temp_c) <= -273.15
        ):
            raise ValueError("metar_temp_c must be finite and above absolute zero")
        if (
            isinstance(self.aod_ref, bool)
            or not isinstance(self.aod_ref, (int, float))
            or not math.isfinite(float(self.aod_ref))
            or float(self.aod_ref) < 0.0
        ):
            raise ValueError("aod_ref must be finite and nonnegative")
        if (
            isinstance(self.aod_ref_wavelength_um, bool)
            or not isinstance(self.aod_ref_wavelength_um, (int, float))
            or not math.isfinite(float(self.aod_ref_wavelength_um))
            or float(self.aod_ref_wavelength_um) <= 0.0
        ):
            raise ValueError("aod_ref_wavelength_um must be finite and positive")
        if (
            isinstance(self.tpw_mm, bool)
            or not isinstance(self.tpw_mm, (int, float))
            or not math.isfinite(float(self.tpw_mm))
            or float(self.tpw_mm) < 0.0
        ):
            raise ValueError("tpw_mm must be finite and nonnegative")
        if not isinstance(self.metar_sky, str) or not self.metar_sky.strip():
            raise ValueError("metar_sky must be a nonempty string")
        if (
            not isinstance(self.cod_status, str)
            or self.cod_status.strip().lower() not in {"valid", "invalid"}
        ):
            raise ValueError("cod_status must be 'valid' or 'invalid'")


@dataclass
class Result:
    t_band: float
    t_810: Optional[float]
    lambda_min_nm: float
    lambda_max_nm: float
    lambda_ref_nm: float
    beta_angstrom: float
    used_clear_sky: bool
    notes: list[str]


def find_uvspec(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("UVSPEC")
    if env:
        return env
    exe = shutil.which("uvspec")
    if exe:
        return exe
    raise FileNotFoundError(
        "Could not find `uvspec`. Pass --uvspec, set UVSPEC, or add it to PATH."
    )


def find_libradtran_data(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LIBRADTRAN_DATA_PATH")
    if env:
        return env
    return "/opt/libRadtran-2.0.6/data"


def angstrom_beta(aod_ref: float, wavelength_ref_um: float, alpha: float) -> float:
    """
    libRadtran aerosol_angstrom uses:
        tau(lambda_um) = beta * lambda_um^(-alpha)
    Given tau at a reference wavelength (in microns):
        beta = aod_ref * wavelength_ref_um^alpha
    """
    return aod_ref * (wavelength_ref_um**alpha)


def write_tophat_filter(path: Path, lambda_min_nm: float, lambda_max_nm: float) -> None:
    """
    Rectangular filter. libRadtran filter_function_file expects wavelength/value pairs.
    We add zeros outside the band edges so normalization+integration is well-defined.
    """
    lines = [
        f"{lambda_min_nm - 1:.3f} 0.0",
        f"{lambda_min_nm:.3f} 1.0",
        f"{lambda_max_nm:.3f} 1.0",
        f"{lambda_max_nm + 1:.3f} 0.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_uvspec_input(
    data_path: str,
    inputs: AtmosInputs,
    lambda_min_nm: float,
    lambda_max_nm: float,
    beta: float,
    atmosphere_file: str = "afglms.dat",
    use_fine_reptran: bool = True,
) -> str:
    """
    We compute clear-sky zenith direct-beam transmittance.
    output_quantity transmittance + output_user lambda edir
    gives direct transmittance spectrum.
    """
    reptran = "reptran fine" if use_fine_reptran else "reptran medium"

    lines = [
        f"data_files_path {data_path}",
        f"atmosphere_file {Path(data_path) / 'atmmod' / atmosphere_file}",
        "source solar",
        "rte_solver disort",
        f"wavelength {lambda_min_nm:.1f} {lambda_max_nm:.1f}",
        f"mol_abs_param {reptran}",
        f"pressure {inputs.pressure_hpa:.3f}",
        f"mol_modify H2O {inputs.tpw_mm:.4f} MM",
        "aerosol_default",
        f"aerosol_angstrom {inputs.angstrom_alpha:.6f} {beta:.8f}",
        "sza 0.0",
        "phi0 0.0",
        "umu 1.0",
        "albedo 0.0",
        "output_quantity transmittance",
        "output_user lambda edir",
        "quiet",
    ]

    sur_temp_k = inputs.metar_temp_c + 273.15
    lines.insert(-1, f"sur_temperature {sur_temp_k:.2f}")

    return "\n".join(lines) + "\n"


def build_uvspec_input_integrated(
    data_path: str,
    inputs: AtmosInputs,
    filter_file: Path,
    lambda_min_nm: float,
    lambda_max_nm: float,
    beta: float,
    atmosphere_file: str = "afglms.dat",
    use_fine_reptran: bool = True,
) -> str:
    """
    Same as spectral run, but with filter convolution and integration to get one number.
    """
    reptran = "reptran fine" if use_fine_reptran else "reptran medium"

    lines = [
        f"data_files_path {data_path}",
        f"atmosphere_file {Path(data_path) / 'atmmod' / atmosphere_file}",
        "source solar",
        "rte_solver disort",
        f"wavelength {lambda_min_nm:.1f} {lambda_max_nm:.1f}",
        f"mol_abs_param {reptran}",
        f"pressure {inputs.pressure_hpa:.3f}",
        f"mol_modify H2O {inputs.tpw_mm:.4f} MM",
        "aerosol_default",
        f"aerosol_angstrom {inputs.angstrom_alpha:.6f} {beta:.8f}",
        "sza 0.0",
        "phi0 0.0",
        "umu 1.0",
        "albedo 0.0",
        "output_quantity transmittance",
        "output_user lambda edir",
        f"filter_function_file {filter_file}",
        "output_process integrate",
        "quiet",
    ]

    sur_temp_k = inputs.metar_temp_c + 273.15
    lines.insert(-1, f"sur_temperature {sur_temp_k:.2f}")

    return "\n".join(lines) + "\n"


def run_uvspec(uvspec_exe: str, uvspec_input: str, cwd: Path) -> str:
    proc = subprocess.run(
        [uvspec_exe],
        input=uvspec_input,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"uvspec failed with code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def parse_spectral_output(stdout: str) -> list[tuple[float, float]]:
    """
    Expect lines with: wavelength_nm  edir_transmittance
    Ignores comments/blank lines.
    """
    rows: list[tuple[float, float]] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        lam = float(parts[0])
        tdir = float(parts[1])
        rows.append((lam, tdir))
    if not rows:
        raise ValueError("No spectral rows parsed from uvspec output.")
    return rows


def parse_integrated_output(stdout: str) -> float:
    """
    output_process integrate usually returns a single numeric row.
    We take the last token on the first non-comment line.
    """
    for line in stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        return float(parts[-1])
    raise ValueError("No integrated value parsed from uvspec output.")


def nearest_t(rows: list[tuple[float, float]], target_nm: float) -> float:
    lam, t = min(rows, key=lambda x: abs(x[0] - target_nm))
    return t


def decide_clear_sky(inputs: AtmosInputs) -> tuple[bool, list[str]]:
    """Require the inputs to support the implemented clear-sky model."""
    notes: list[str] = []

    # Require agreement between the clear-sky report and the absence of a usable
    # cloud-optical-depth input. Neither condition is sufficient on its own.
    metar_clear = inputs.metar_sky.strip().upper() in {"CLR", "SKC"}
    cod_invalid = inputs.cod_status.strip().lower() == "invalid"

    if metar_clear:
        notes.append(f"METAR sky condition = {inputs.metar_sky}; METAR reports clear sky.")
    if cod_invalid:
        notes.append("No cloud optical depth was supplied; none was applied.")

    if not (metar_clear and cod_invalid):
        raise ValueError(
            "Cannot calculate libRadtran transmission for a non-clear scene: "
            "clear sky requires both a CLR/SKC METAR and no usable cloud-optical-depth "
            "input, and no quantitative cloud model is implemented."
        )

    notes.append("Running libRadtran in clear-sky mode (no cloud optical depth).")

    sur_temp_k = inputs.metar_temp_c + 273.15
    notes.append(
        f"Surface temperature {inputs.metar_temp_c:.1f} C ({sur_temp_k:.2f} K) "
        "applied via sur_temperature."
    )
    if inputs.metar_rh_pct is not None:
        notes.append(f"Surface RH {inputs.metar_rh_pct:.1f}% available but not used directly.")

    return True, notes


def estimate_transmission(inputs: AtmosInputs) -> tuple[Result, Optional[Path]]:
    """
    Programmatic API: compute clear-sky zenith direct-beam transmission
    over a fixed 782–867 nm rectangular band with reference 810 nm,
    using libRadtran. Paths and settings come from environment:
      - UVSPEC (executable path) or PATH
      - LIBRADTRAN_DATA_PATH
      - LIBRADTRAN_REPTRAN = fine|medium (default fine)
      - LIBRADTRAN_KEEP_WORKDIR = 1/true/yes/on to keep temp dir

    Returns (result, workdir). workdir is None unless LIBRADTRAN_KEEP_WORKDIR is enabled.
    """
    used_clear_sky, notes = decide_clear_sky(inputs)

    uvspec_exe = find_uvspec(None)
    data_path = find_libradtran_data(None)
    reptran = os.environ.get("LIBRADTRAN_REPTRAN", "fine").strip().lower()
    use_fine_reptran = reptran == "fine"
    keep = os.environ.get("LIBRADTRAN_KEEP_WORKDIR", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    lambda_min_nm = DEFAULT_LAMBDA_MIN_NM
    lambda_max_nm = DEFAULT_LAMBDA_MAX_NM
    lambda_ref_nm = DEFAULT_LAMBDA_REF_NM

    beta = angstrom_beta(inputs.aod_ref, inputs.aod_ref_wavelength_um, inputs.angstrom_alpha)
    # Working directory
    if keep:
        workdir = Path(tempfile.mkdtemp(prefix="uvspec_atm_", dir="."))
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="uvspec_atm_", dir=".")
        workdir = Path(tmp_ctx.name)

    # Build inputs and run
    filter_file = workdir / "filter_band.dat"
    write_tophat_filter(filter_file, lambda_min_nm, lambda_max_nm)

    spectral_inp = build_uvspec_input(
        data_path=data_path,
        inputs=inputs,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        beta=beta,
        use_fine_reptran=use_fine_reptran,
    )
    integrated_inp = build_uvspec_input_integrated(
        data_path=data_path,
        inputs=inputs,
        filter_file=filter_file,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        beta=beta,
        use_fine_reptran=use_fine_reptran,
    )

    (workdir / "uvspec_spectral.inp").write_text(spectral_inp, encoding="utf-8")
    (workdir / "uvspec_integrated.inp").write_text(integrated_inp, encoding="utf-8")

    spectral_out = run_uvspec(uvspec_exe, spectral_inp, cwd=workdir)
    rows = parse_spectral_output(spectral_out)
    t_810 = nearest_t(rows, lambda_ref_nm)

    integrated_out = run_uvspec(uvspec_exe, integrated_inp, cwd=workdir)
    t_band = parse_integrated_output(integrated_out)

    result = Result(
        t_band=t_band,
        t_810=t_810,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        lambda_ref_nm=lambda_ref_nm,
        beta_angstrom=beta,
        used_clear_sky=used_clear_sky,
        notes=notes,
    )

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
        return result, None
    else:
        return result, workdir


def format_text_report(
    *,
    inputs: AtmosInputs,
    result: Result,
    uvspec_exe: str,
    data_path: str,
    workdir: Optional[Path],
) -> str:
    ref_nm = inputs.aod_ref_wavelength_um * 1000.0
    title = "Atmospheric transmission estimate from libRadtran"
    if inputs.source_label:
        title += f" [{inputs.source_label}]"
    sep = "-" * len(title)
    lines = [
        title,
        sep,
        f"uvspec executable : {uvspec_exe}",
        f"data path         : {data_path}",
    ]
    if workdir is not None:
        lines.append(f"workdir           : {workdir}")
    lines.extend(
        [
            "",
            "Inputs",
            f"  Source          : {inputs.source_label or '(unspecified)'}",
            f"  Pressure        : {inputs.pressure_hpa:.3f} hPa",
            f"  AOD ({ref_nm:.0f}nm)     : {inputs.aod_ref:.5f}",
            f"  Ref wavelength  : {ref_nm:.0f} nm ({inputs.aod_ref_wavelength_um:.4f} µm)",
            f"  Angstrom alpha  : {inputs.angstrom_alpha:.5f}",
            f"  Angstrom beta   : {result.beta_angstrom:.8f}",
            f"  TPW             : {inputs.tpw_mm:.3f} mm",
            f"  METAR sky       : {inputs.metar_sky}",
            f"  COD status      : {inputs.cod_status}",
            "",
            "Band",
            f"  Filter          : rectangular {result.lambda_min_nm:.1f}-{result.lambda_max_nm:.1f} nm",
            f"  Reference       : {result.lambda_ref_nm:.1f} nm",
            "",
            "Results",
            f"  T_band          : {result.t_band:.6f}",
        ]
    )
    if result.t_810 is not None:
        lines.append(f"  T_{int(result.lambda_ref_nm)}            : {result.t_810:.6f}")
    else:
        lines.append(f"  T_{int(result.lambda_ref_nm)}            : n/a")
    lines.append("")
    lines.append("Notes")
    for n in result.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvspec", default=None, help="Path to uvspec executable")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to libRadtran data directory (or set LIBRADTRAN_DATA_PATH)",
    )
    parser.add_argument("--pressure-hpa", type=float, required=True)
    parser.add_argument(
        "--aod-ref",
        type=float,
        required=True,
        help="AOD at reference wavelength",
    )
    parser.add_argument(
        "--aod-ref-wavelength-um",
        type=float,
        required=True,
        help="Reference wavelength for AOD in microns",
    )
    parser.add_argument("--angstrom-alpha", type=float, required=True)
    parser.add_argument("--tpw-mm", type=float, required=True)
    parser.add_argument("--metar-sky", required=True)
    parser.add_argument(
        "--cod-status",
        choices=["valid", "invalid"],
        required=True,
    )
    parser.add_argument("--metar-temp-c", type=float, required=True)
    parser.add_argument("--metar-rh-pct", type=float, default=None)
    parser.add_argument("--source-label", default="")
    parser.add_argument("--lambda-min-nm", type=float, default=DEFAULT_LAMBDA_MIN_NM)
    parser.add_argument("--lambda-max-nm", type=float, default=DEFAULT_LAMBDA_MAX_NM)
    parser.add_argument("--lambda-ref-nm", type=float, default=DEFAULT_LAMBDA_REF_NM)
    parser.add_argument(
        "--reptran",
        choices=["fine", "medium"],
        default="fine",
        help="Spectral resolution for gas absorption",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep temporary work directory for inspection",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    inputs = AtmosInputs(
        pressure_hpa=args.pressure_hpa,
        aod_ref=args.aod_ref,
        aod_ref_wavelength_um=args.aod_ref_wavelength_um,
        angstrom_alpha=args.angstrom_alpha,
        tpw_mm=args.tpw_mm,
        metar_sky=args.metar_sky,
        cod_status=args.cod_status,
        metar_temp_c=args.metar_temp_c,
        metar_rh_pct=args.metar_rh_pct,
        source_label=args.source_label,
    )

    used_clear_sky, notes = decide_clear_sky(inputs)

    uvspec_exe = find_uvspec(args.uvspec)
    data_path = find_libradtran_data(args.data_path)

    lambda_min_nm = args.lambda_min_nm
    lambda_max_nm = args.lambda_max_nm
    lambda_ref_nm = args.lambda_ref_nm

    beta = angstrom_beta(inputs.aod_ref, inputs.aod_ref_wavelength_um, inputs.angstrom_alpha)
    if args.keep_workdir:
        workdir_obj = tempfile.TemporaryDirectory(prefix="uvspec_atm_", dir=".", delete=False)
    else:
        workdir_obj = tempfile.TemporaryDirectory(prefix="uvspec_atm_", dir=".")

    with workdir_obj as workdir_name:
        workdir = Path(workdir_name)

        filter_file = workdir / "filter_band.dat"
        write_tophat_filter(filter_file, lambda_min_nm, lambda_max_nm)

        spectral_inp = build_uvspec_input(
            data_path=data_path,
            inputs=inputs,
            lambda_min_nm=lambda_min_nm,
            lambda_max_nm=lambda_max_nm,
            beta=beta,
            use_fine_reptran=(args.reptran == "fine"),
        )
        integrated_inp = build_uvspec_input_integrated(
            data_path=data_path,
            inputs=inputs,
            filter_file=filter_file,
            lambda_min_nm=lambda_min_nm,
            lambda_max_nm=lambda_max_nm,
            beta=beta,
            use_fine_reptran=(args.reptran == "fine"),
        )

        (workdir / "uvspec_spectral.inp").write_text(spectral_inp, encoding="utf-8")
        (workdir / "uvspec_integrated.inp").write_text(integrated_inp, encoding="utf-8")

        spectral_out = run_uvspec(uvspec_exe, spectral_inp, cwd=workdir)
        rows = parse_spectral_output(spectral_out)
        t_810 = nearest_t(rows, lambda_ref_nm)

        integrated_out = run_uvspec(uvspec_exe, integrated_inp, cwd=workdir)
        t_band = parse_integrated_output(integrated_out)

        result = Result(
            t_band=t_band,
            t_810=t_810,
            lambda_min_nm=lambda_min_nm,
            lambda_max_nm=lambda_max_nm,
            lambda_ref_nm=lambda_ref_nm,
            beta_angstrom=beta,
            used_clear_sky=used_clear_sky,
            notes=notes,
        )

        if args.json:
            payload = {
                "inputs": asdict(inputs),
                "result": asdict(result),
                "workdir": str(workdir),
            }
            print(json.dumps(payload, indent=2))
        else:
            print(
                format_text_report(
                    inputs=inputs,
                    result=result,
                    uvspec_exe=uvspec_exe,
                    data_path=data_path,
                    workdir=workdir,
                ),
                end="",
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
