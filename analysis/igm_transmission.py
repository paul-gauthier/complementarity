#!/usr/bin/env python3
"""Compute the common future-directed intergalactic transmission model."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.integrate import quad

C_KM_S = 299792.458
MPC_CM = 3.0856775814913673e24


@dataclass(frozen=True)
class IGMTransmission:
    density_kernel_integral: float
    dust_kernel_integral: float
    effective_density_path_gpc: float
    dust_visual_opacity_gpc_inverse: float
    dust_launch_to_visual_ratio: float
    dust_opacity_gpc_inverse: float
    dust_optical_depth: float
    dust_transmission: float
    electron_optical_depth: float
    electron_transmission: float
    galaxy_number_density_mpc3: float
    galaxy_radius_mpc: float
    galaxy_cross_section_mpc2: float
    galaxy_optical_depth: float
    galaxy_transmission: float
    nominal_optical_depth: float
    nominal_transmission: float
    gray_dust_optical_depth: float
    gray_total_optical_depth: float
    gray_transmission: float


def _ccm_a_b(wavelength_um: float) -> tuple[float, float]:
    """Return Cardelli a(x), b(x), extending the near-IR law to long wavelengths."""
    x = 1.0 / wavelength_um
    if x < 1.1:
        power = x**1.61
        return 0.574 * power, -0.527 * power
    if x <= 3.3:
        y = x - 1.82
        a = (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )
        return a, b
    raise ValueError("wavelength is outside the implemented Cardelli range")


def _ccm_extinction_ratio(wavelength_um: float, r_v: float) -> float:
    """Return A(lambda)/A(V), using CCM's visual-band normalization."""
    a, b = _ccm_a_b(wavelength_um)
    return a + b / r_v


def _relative_dust_opacity(wavelength_um: float, reference_um: float, r_v: float) -> float:
    return _ccm_extinction_ratio(wavelength_um, r_v) / _ccm_extinction_ratio(
        reference_um, r_v
    )


def calculate_igm_transmission(settings: dict[str, Any]) -> IGMTransmission:
    h0 = float(settings["hubble_constant_km_s_mpc"])
    omega_m = float(settings["omega_m"])
    omega_lambda = float(settings["omega_lambda"])
    wavelength_um = float(settings["idler_wavelength_nm"]) / 1000.0
    r_v = float(settings["dust_r_v"])

    def expansion(a: float) -> float:
        return math.sqrt(omega_m * a**-3 + omega_lambda)

    def density_kernel(a: float) -> float:
        return 1.0 / (a**4 * expansion(a))

    density_integral = quad(
        density_kernel, 1.0, math.inf, epsabs=1e-12, epsrel=1e-12
    )[0]
    dust_integral = quad(
        lambda a: density_kernel(a)
        * _relative_dust_opacity(wavelength_um * a, wavelength_um, r_v),
        1.0,
        math.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]

    hubble_distance_mpc = C_KM_S / h0
    hubble_distance_gpc = hubble_distance_mpc / 1000.0
    visual_dust_opacity = (
        0.4
        * math.log(10.0)
        * float(settings["dust_visual_extinction_mag"])
        / float(settings["dust_normalization_distance_gpc"])
    )
    launch_to_visual_ratio = _ccm_extinction_ratio(wavelength_um, r_v)
    dust_opacity = visual_dust_opacity * launch_to_visual_ratio
    dust_tau = dust_opacity * hubble_distance_gpc * dust_integral
    electron_tau = (
        float(settings["electron_density_cm3"])
        * float(settings["thomson_cross_section_cm2"])
        * hubble_distance_mpc
        * MPC_CM
        * density_integral
    )

    h = h0 / 100.0
    galaxy_density = float(settings["galaxy_number_density_h3_mpc3"]) * h**3
    galaxy_radius = float(settings["galaxy_radius_hinv_pc"]) / h / 1.0e6
    galaxy_cross_section = math.pi * galaxy_radius**2
    galaxy_tau = (
        galaxy_density
        * galaxy_cross_section
        * hubble_distance_mpc
        * density_integral
    )

    nominal_tau = dust_tau + electron_tau + galaxy_tau
    # Hold the corrected launch opacity fixed across future wavelengths, while
    # retaining the same cosmological density dilution as in the CCM integral.
    gray_dust_tau = dust_opacity * hubble_distance_gpc * density_integral
    gray_total_tau = gray_dust_tau + electron_tau + galaxy_tau
    return IGMTransmission(
        density_kernel_integral=density_integral,
        dust_kernel_integral=dust_integral,
        effective_density_path_gpc=hubble_distance_gpc * density_integral,
        dust_visual_opacity_gpc_inverse=visual_dust_opacity,
        dust_launch_to_visual_ratio=launch_to_visual_ratio,
        dust_opacity_gpc_inverse=dust_opacity,
        dust_optical_depth=dust_tau,
        dust_transmission=math.exp(-dust_tau),
        electron_optical_depth=electron_tau,
        electron_transmission=math.exp(-electron_tau),
        galaxy_number_density_mpc3=galaxy_density,
        galaxy_radius_mpc=galaxy_radius,
        galaxy_cross_section_mpc2=galaxy_cross_section,
        galaxy_optical_depth=galaxy_tau,
        galaxy_transmission=math.exp(-galaxy_tau),
        nominal_optical_depth=nominal_tau,
        nominal_transmission=math.exp(-nominal_tau),
        gray_dust_optical_depth=gray_dust_tau,
        gray_total_optical_depth=gray_total_tau,
        gray_transmission=math.exp(-gray_total_tau),
    )


def build_record(settings: dict[str, Any], result: IGMTransmission) -> dict[str, Any]:
    values = asdict(result)
    references = settings["references"]
    return {
        "schema_version": 1,
        "model": {
            "cosmology": "flat_lambda_cdm_future_a_1_to_infinity",
            "dust": "cardelli_rv_with_near_ir_power_law_extension",
            "dust_normalization": "visual_opacity_converted_to_launch_wavelength_with_ccm",
            "dust_wavelength_factor": "ccm_extinction_relative_to_launch_wavelength",
            "gray_dust": "same_launch_opacity_with_wavelength_factor_fixed_at_one",
            "electron": "constant_comoving_density_thomson_scattering",
            "galaxy": "craig_1996_opaque_fixed_radius_constant_comoving_population",
        },
        "inputs": {
            key: value for key, value in settings.items() if key != "references"
        },
        "integrals": {
            "density_kernel": values.pop("density_kernel_integral"),
            "dust_kernel": values.pop("dust_kernel_integral"),
            "effective_density_path_gpc": values.pop("effective_density_path_gpc"),
        },
        "derived": values,
        "references": dict(references),
    }


def _write_json_atomically(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    from .artifact_config import DEFAULT_CONFIG, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = load_config(args.config)["igm_transmission"]
    result = calculate_igm_transmission(settings)
    _write_json_atomically(args.output, build_record(settings, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
