#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

from .transmission_values import (
    DEFAULT_MILKY_WAY_BASIS,
    MILKY_WAY_BASES,
    calculate_transmissions,
)

ROOT = Path(__file__).resolve().parent.parent
class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class ValueSpec:
    macro: str
    path: str
    formatter: Callable[[float], str]


def number(data: dict[str, Any], path: str) -> float:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AnalysisError(f"missing required field: {path}")
        value = value[part]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"required field is not numeric: {path}")
    if not math.isfinite(value):
        raise AnalysisError(f"required field is not finite: {path}")
    return float(value)


def text(data: dict[str, Any], path: str) -> str:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AnalysisError(f"missing required field: {path}")
        value = value[part]

    if not isinstance(value, str):
        raise AnalysisError(f"required field is not text: {path}")
    return value


def require_close(
    actual: float,
    expected: float,
    description: str,
    *,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AnalysisError(
            f"{description} is inconsistent: got {actual!r}, expected {expected!r}"
        )


def require_integer(data: dict[str, Any], path: str, expected: int) -> None:
    value = number(data, path)
    if value != expected:
        raise AnalysisError(f"{path} must be {expected}, got {value!r}")


def joint_fit(
    data: dict[str, Any], *, condition: str, channel: str
) -> dict[str, Any]:
    records = data.get("joint_fits")
    if not isinstance(records, list):
        raise AnalysisError("joint_fits must be an array")

    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("condition") == condition
        and record.get("channel") == channel
    ]
    if len(matches) != 1:
        raise AnalysisError(
            f"expected exactly one joint_fits record for "
            f"condition={condition!r}, channel={channel!r}; got {len(matches)}"
        )
    return matches[0]


def record_number(record: dict[str, Any], field: str, description: str) -> float:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AnalysisError(f"missing required field: {description}.{field}")
        value = value[part]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"required field is not numeric: {description}.{field}")
    if not math.isfinite(value):
        raise AnalysisError(f"required field is not finite: {description}.{field}")
    return float(value)


def conditions_libradtran_record(
    data: dict[str, Any], tag: str
) -> dict[str, Any]:
    records = data.get("libradtran")
    if not isinstance(records, list):
        raise AnalysisError("conditions libradtran must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("tag") == tag
    ]
    if len(matches) != 1:
        raise AnalysisError(
            f"expected exactly one conditions libradtran record tagged "
            f"{tag!r}; got {len(matches)}"
        )
    return matches[0]


def conditions_aeronet_record(data: dict[str, Any]) -> dict[str, Any]:
    records = data.get("libradtran")
    if not isinstance(records, list):
        raise AnalysisError("conditions libradtran must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("tag") in {"AERONET-solar", "AERONET-lunar"}
    ]
    if len(matches) != 1:
        raise AnalysisError(
            f"expected exactly one AERONET conditions record; got {len(matches)}"
        )
    return matches[0]


def condition_transmissions(
    data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
):
    atmosphere_tag = "GOES-18" if atmosphere_source == "goes" else next(
        record["tag"] for record in data.get("libradtran", [])
        if isinstance(record, dict) and str(record.get("tag", "")).startswith("AERONET-")
    )
    atmosphere_record = conditions_libradtran_record(data, atmosphere_tag)
    aeronet_record = conditions_aeronet_record(data)
    return calculate_transmissions(
        atmosphere=record_number(
            atmosphere_record, "result.t_band", f"conditions libradtran[{atmosphere_tag}]"
        ),
        aeronet_atmosphere=record_number(
            aeronet_record,
            "result.t_band",
            f"conditions libradtran[{aeronet_record['tag']}]",
        ),
        milky_way_point=number(data, "irsa.transmission_fraction"),
        milky_way_integrated=number(data, "irsa.integration.avg_T"),
        milky_way_minimum=number(data, "irsa.integration.min_T"),
        milky_way_basis=milky_way_basis,
    )


def mahalanobis_distance(
    *,
    point_c: float,
    point_s: float,
    center_c: float,
    center_s: float,
    variance_c: float,
    variance_s: float,
    covariance: float,
) -> float:
    determinant = variance_c * variance_s - covariance**2
    if determinant <= 0:
        raise AnalysisError("added-quadrature covariance matrix is not positive definite")

    delta_c = point_c - center_c
    delta_s = point_s - center_s
    squared_distance = (
        variance_s * delta_c**2
        - 2 * covariance * delta_c * delta_s
        + variance_c * delta_s**2
    ) / determinant
    return math.sqrt(max(0.0, squared_distance))


def minimum_mahalanobis_distance(
    *,
    radius: float,
    center_c: float,
    center_s: float,
    variance_c: float,
    variance_s: float,
    covariance: float,
) -> float:
    determinant = variance_c * variance_s - covariance**2
    if determinant <= 0:
        raise AnalysisError("added-quadrature covariance matrix is not positive definite")

    inverse_cc = variance_s / determinant
    inverse_ss = variance_c / determinant
    inverse_cs = -covariance / determinant

    def squared_distance(angle: float) -> float:
        delta_c = radius * math.cos(angle) - center_c
        delta_s = radius * math.sin(angle) - center_s
        return (
            inverse_cc * delta_c**2
            + 2 * inverse_cs * delta_c * delta_s
            + inverse_ss * delta_s**2
        )

    sample_count = 4096
    step = 2 * math.pi / sample_count
    sampled = [squared_distance(index * step) for index in range(sample_count)]
    candidate_indices = [
        index
        for index, result in enumerate(sampled)
        if result <= sampled[index - 1]
        and result <= sampled[(index + 1) % sample_count]
    ]

    golden_ratio = (math.sqrt(5) - 1) / 2
    minima = []
    for index in candidate_indices:
        lower = (index - 1) * step
        upper = (index + 1) * step
        left = upper - golden_ratio * (upper - lower)
        right = lower + golden_ratio * (upper - lower)
        left_value = squared_distance(left)
        right_value = squared_distance(right)
        for _ in range(80):
            if left_value <= right_value:
                upper = right
                right = left
                right_value = left_value
                left = upper - golden_ratio * (upper - lower)
                left_value = squared_distance(left)
            else:
                lower = left
                left = right
                left_value = right_value
                right = lower + golden_ratio * (upper - lower)
                right_value = squared_distance(right)
        minima.append(min(left_value, right_value))

    if not minima:
        raise AnalysisError("could not minimize the restoration Mahalanobis distance")
    return math.sqrt(max(0.0, min(minima)))


def validate_normalization_cases(
    data: dict[str, Any],
    *,
    infinity: float,
    finite_path: float,
    rpm: float,
    bound: float,
    center_c: float,
    center_s: float,
    variance_c: float,
    variance_s: float,
    covariance: float,
) -> None:
    summary_cases = data.get("summary", {}).get("normalization_cases")
    top_level_cases = data.get("normalization_cases")
    if not isinstance(summary_cases, list):
        raise AnalysisError("summary.normalization_cases must be an array")
    if not isinstance(top_level_cases, list):
        raise AnalysisError("normalization_cases must be an array")
    if summary_cases != top_level_cases:
        raise AnalysisError(
            "summary.normalization_cases and normalization_cases must agree"
        )
    if len(summary_cases) != 4:
        raise AnalysisError(
            f"expected four normalization cases, got {len(summary_cases)}"
        )

    expected_transmissions = sorted((infinity, finite_path))
    for multiplier in (1, 2):
        cases = [
            case
            for case in summary_cases
            if isinstance(case, dict)
            and record_number(case, "m", "normalization case") == multiplier
        ]
        if len(cases) != 2:
            raise AnalysisError(
                f"expected two normalization cases with m={multiplier}, "
                f"got {len(cases)}"
            )
        cases.sort(
            key=lambda case: record_number(case, "T", "normalization case")
        )

        for case, expected_transmission in zip(
            cases, expected_transmissions, strict=True
        ):
            description = (
                f"normalization case m={multiplier}, "
                f"T={expected_transmission!r}"
            )
            case_transmission = record_number(case, "T", description)
            require_close(
                case_transmission,
                expected_transmission,
                f"{description}.T",
            )

            effective_normalization = multiplier * expected_transmission
            radius = effective_normalization * rpm
            require_close(
                record_number(
                    case,
                    "effective_normalization_mT",
                    description,
                ),
                effective_normalization,
                f"{description}.effective_normalization_mT",
            )
            require_close(
                record_number(case, "restoration_radius_cps", description),
                radius,
                f"{description}.restoration_radius_cps",
            )
            require_close(
                record_number(case, "eta95", description),
                bound / radius,
                f"{description}.eta95",
            )

            phase = record_number(case, "nearest_phase_rad", description)
            nearest_c = radius * math.cos(phase)
            nearest_s = radius * math.sin(phase)
            require_close(
                record_number(case, "nearest_C_cps", description),
                nearest_c,
                f"{description}.nearest_C_cps",
            )
            require_close(
                record_number(case, "nearest_S_cps", description),
                nearest_s,
                f"{description}.nearest_S_cps",
            )
            require_close(
                record_number(case, "d_min", description),
                minimum_mahalanobis_distance(
                    radius=radius,
                    center_c=center_c,
                    center_s=center_s,
                    variance_c=variance_c,
                    variance_s=variance_s,
                    covariance=covariance,
                ),
                f"{description}.d_min",
            )


def fixed(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def upper_bound(value: float, decimals: int) -> str:
    if value < 0:
        raise AnalysisError(f"upper bound must be nonnegative, got {value!r}")
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_CEILING)
    return f"{rounded:.{decimals}f}"


def integer(value: float) -> str:
    rounded = Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return str(rounded)


def integer_as_text(value: float) -> str:
    words = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    )
    if value.is_integer() and 0 <= value < len(words):
        return words[int(value)]
    return integer(value)


def measured(value: float, sigma: float) -> str:
    if sigma <= 0:
        raise AnalysisError(f"measurement uncertainty must be positive, got {sigma!r}")

    exponent = math.floor(math.log10(sigma))
    leading_digit = int(sigma / (10**exponent))
    significant_digits = 2 if leading_digit == 1 else 1
    decimals = significant_digits - 1 - exponent

    def rounded(result: float) -> str:
        if decimals >= 0:
            return fixed(result, decimals)
        quantum = Decimal(1).scaleb(-decimals)
        rounded_result = Decimal(str(result)).quantize(
            quantum, rounding=ROUND_HALF_UP
        )
        return f"{rounded_result:.0f}"

    return rf"{rounded(value)}\pm{rounded(sigma)}"


def validate(
    data: dict[str, Any],
    joint_data: dict[str, Any],
    conditions_data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> None:
    require_close(number(data, "summary.confidence"), 0.95, "confidence")
    require_integer(data, "summary.paired_n_passes", 9)
    require_integer(data, "summary.paired_n_point_pairs", 99)
    require_integer(data, "summary.paired_n_observations", 198)
    require_integer(data, "summary.paired_df", 176)

    if text(data, "summary.covariance_scale") != "max1":
        raise AnalysisError("unexpected covariance_scale")
    if (
        text(data, "summary.paired_voltage_model")
        != "stacked_actual_voltage_common_phase_with_separate_any_and_known_launch_fits"
    ):
        raise AnalysisError("unexpected paired_voltage_model")
    if text(data, "summary.launch_condition") != "Launch":
        raise AnalysisError("unexpected launch_condition")
    if text(data, "summary.control_condition") != "Erase":
        raise AnalysisError("unexpected control_condition")
    if text(data, "summary.phase_reference_condition") != "Erase":
        raise AnalysisError("unexpected phase_reference_condition")
    if text(data, "summary.phase_reference_channel") != "C":
        raise AnalysisError("unexpected phase_reference_channel")
    if text(data, "summary.known_phase_reference_condition") != "Erase":
        raise AnalysisError("unexpected known_phase_reference_condition")
    if text(data, "summary.known_phase_channel") != "C":
        raise AnalysisError("unexpected known_phase_channel")

    period = number(data, "summary.period_V")
    if period <= 0:
        raise AnalysisError("summary.period_V must be positive")
    match_tolerance = number(data, "summary.x_match_tol_V")
    if match_tolerance < 0:
        raise AnalysisError("summary.x_match_tol_V must be nonnegative")
    if number(data, "summary.x_mismatch_max_V") > match_tolerance:
        raise AnalysisError("maximum voltage mismatch exceeds x_match_tol_V")
    require_close(
        number(data, "summary.phase_mismatch_max_rad"),
        2 * math.pi * number(data, "summary.x_mismatch_max_V") / period,
        "maximum voltage/phase mismatch",
    )
    require_close(
        number(data, "summary.phase_mismatch_median_rad"),
        2 * math.pi * number(data, "summary.x_mismatch_median_V") / period,
        "median voltage/phase mismatch",
    )

    require_integer(data, "summary.known_phase_delta_n_passes", 9)
    for path in (
        "summary.known_phase_delta_mean_rad",
        "summary.known_phase_delta_rms_about_mean_rad",
        "summary.known_phase_delta_span_rad",
    ):
        require_close(number(data, path), 0.0, path)

    for suffix in (
        "n_passes",
        "n_point_pairs",
        "n_observations",
        "df",
        "chi2",
        "reduced_chi2",
    ):
        require_close(
            number(data, f"summary.paired_known_{suffix}"),
            number(data, f"summary.paired_{suffix}"),
            f"paired known/any-phase {suffix}",
        )
    require_close(
        number(data, "summary.paired_n_observations"),
        2 * number(data, "summary.paired_n_point_pairs"),
        "paired observations/point-pairs",
    )

    chi2 = number(data, "summary.paired_chi2")
    degrees_of_freedom = number(data, "summary.paired_df")
    require_close(
        number(data, "summary.paired_reduced_chi2"),
        chi2 / degrees_of_freedom,
        "paired reduced chi-squared",
    )

    for path in (
        "summary.period_sigma_V",
        "summary.Rpm_sigma_cps",
        "summary.launch_total_singles_fringe_sigma_cps",
        "summary.control_total_singles_fringe_sigma_cps",
        "summary.scalar_total_singles_excess_sigma_cps",
        "summary.C0_sigma_cps",
        "summary.S0_sigma_cps",
        "summary.CL_sigma_cps",
        "summary.SL_sigma_cps",
        "summary.B95_known_phase_cps",
        "summary.B95_any_phase_cps",
        "summary.eta95_known_phase",
        "summary.eta95_any_phase",
        "summary.full_restoration_mahalanobis_distance",
    ):
        if number(data, path) < 0:
            raise AnalysisError(f"{path} must be nonnegative")

    if text(data, "summary.Rpm_source") != "Erase:C - Preserve:C":
        raise AnalysisError("unexpected Rpm_source")

    direct_fits = {
        ("Erase", "C"): joint_fit(joint_data, condition="Erase", channel="C"),
        ("Erase", "I"): joint_fit(joint_data, condition="Erase", channel="I"),
        ("Launch", "I"): joint_fit(joint_data, condition="Launch", channel="I"),
        ("Preserve", "C"): joint_fit(joint_data, condition="Preserve", channel="C"),
        ("Preserve", "I"): joint_fit(joint_data, condition="Preserve", channel="I"),
    }
    for (condition, channel), record in direct_fits.items():
        description = f"direct joint_fits[{condition},{channel}]"
        for field in (
            "A1_cps",
            "A1_sigma_cps",
            "A2_cps",
            "A2_sigma_cps",
            "fringe1_cps",
            "fringe1_sigma_cps",
            "fringe2_cps",
            "fringe2_sigma_cps",
            "total_fringe_cps",
            "total_fringe_sigma_cps",
            "V",
            "V_sigma",
            "P_V",
            "x0_V",
        ):
            value = record_number(record, field, description)
            if "sigma" in field and value < 0:
                raise AnalysisError(f"{description}.{field} must be nonnegative")
        for field in (
            "A1_cps",
            "A2_cps",
            "fringe1_cps",
            "fringe2_cps",
            "total_fringe_cps",
        ):
            if record_number(record, field, description) < 0:
                raise AnalysisError(f"{description}.{field} must be nonnegative")

        visibility = record_number(record, "V", description)
        if not 0 <= visibility <= 1:
            raise AnalysisError(f"{description}.V must be in [0, 1]")
        require_close(
            record_number(record, "P_V", description),
            period,
            f"{description}.P_V",
        )

        a1 = record_number(record, "A1_cps", description)
        a2 = record_number(record, "A2_cps", description)
        fringe1 = record_number(record, "fringe1_cps", description)
        fringe2 = record_number(record, "fringe2_cps", description)
        require_close(fringe1, a1 * visibility, f"{description}.fringe1_cps")
        require_close(fringe2, a2 * visibility, f"{description}.fringe2_cps")
        require_close(
            record_number(record, "total_fringe_cps", description),
            fringe1 + fringe2,
            f"{description}.total_fringe_cps",
        )

    null_fits = {
        ("Erase", "C"): joint_fit(data, condition="Erase", channel="C"),
        ("Erase", "I"): joint_fit(data, condition="Erase", channel="I"),
        ("Launch", "I"): joint_fit(data, condition="Launch", channel="I"),
        ("Preserve", "C"): joint_fit(data, condition="Preserve", channel="C"),
    }
    overlapping_fields = (
        "A1_cps",
        "A2_cps",
        "V",
        "V_sigma",
        "fringe1_cps",
        "fringe1_sigma_cps",
        "fringe2_cps",
        "fringe2_sigma_cps",
        "total_fringe_cps",
        "total_fringe_sigma_cps",
    )
    for condition_channel, null_record in null_fits.items():
        condition, channel = condition_channel
        direct_record = direct_fits[condition_channel]
        for field in overlapping_fields:
            require_close(
                record_number(
                    null_record,
                    field,
                    f"null joint_fits[{condition},{channel}]",
                ),
                record_number(
                    direct_record,
                    field,
                    f"direct joint_fits[{condition},{channel}]",
                ),
                f"joint-fit {field} for {condition}/{channel}",
            )

    erase_coincidence = record_number(
        direct_fits[("Erase", "C")],
        "total_fringe_cps",
        "joint_fits[Erase,C]",
    )
    preserve_coincidence = record_number(
        direct_fits[("Preserve", "C")],
        "total_fringe_cps",
        "joint_fits[Preserve,C]",
    )
    rpm = erase_coincidence - preserve_coincidence
    require_close(number(data, "summary.Rpm_cps"), rpm, "Rpm_cps")

    erase_coincidence_sigma = record_number(
        direct_fits[("Erase", "C")],
        "total_fringe_sigma_cps",
        "joint_fits[Erase,C]",
    )
    preserve_coincidence_sigma = record_number(
        direct_fits[("Preserve", "C")],
        "total_fringe_sigma_cps",
        "joint_fits[Preserve,C]",
    )
    rpm_sigma = math.hypot(erase_coincidence_sigma, preserve_coincidence_sigma)
    require_close(
        number(data, "summary.Rpm_sigma_cps"),
        rpm_sigma,
        "Rpm_sigma_cps",
    )

    launch_singles = record_number(
        direct_fits[("Launch", "I")],
        "total_fringe_cps",
        "joint_fits[Launch,I]",
    )
    erase_singles = record_number(
        direct_fits[("Erase", "I")],
        "total_fringe_cps",
        "joint_fits[Erase,I]",
    )
    launch_singles_sigma = record_number(
        direct_fits[("Launch", "I")],
        "total_fringe_sigma_cps",
        "joint_fits[Launch,I]",
    )
    erase_singles_sigma = record_number(
        direct_fits[("Erase", "I")],
        "total_fringe_sigma_cps",
        "joint_fits[Erase,I]",
    )
    require_close(
        number(data, "summary.launch_total_singles_fringe_cps"),
        launch_singles,
        "launch_total_singles_fringe_cps",
    )
    require_close(
        number(data, "summary.control_total_singles_fringe_cps"),
        erase_singles,
        "control_total_singles_fringe_cps",
    )
    require_close(
        number(data, "summary.launch_total_singles_fringe_sigma_cps"),
        launch_singles_sigma,
        "launch_total_singles_fringe_sigma_cps",
    )
    require_close(
        number(data, "summary.control_total_singles_fringe_sigma_cps"),
        erase_singles_sigma,
        "control_total_singles_fringe_sigma_cps",
    )
    scalar_excess = launch_singles - erase_singles
    scalar_sigma = math.hypot(launch_singles_sigma, erase_singles_sigma)
    require_close(
        number(data, "summary.scalar_total_singles_excess_cps"),
        scalar_excess,
        "scalar_total_singles_excess_cps",
    )
    require_close(
        number(data, "summary.scalar_total_singles_excess_sigma_cps"),
        scalar_sigma,
        "scalar_total_singles_excess_sigma_cps",
    )
    require_close(
        number(data, "summary.scalar_B95_cps"),
        max(0.0, scalar_excess)
        + NormalDist().inv_cdf(number(data, "summary.confidence")) * scalar_sigma,
        "scalar_B95_cps",
    )

    exact_transmissions = condition_transmissions(
        conditions_data, atmosphere_source, milky_way_basis
    )
    tinf = number(data, "summary.Tinf")
    require_close(
        tinf,
        exact_transmissions.infinity,
        "analysis/conditions infinite-future transmission",
    )
    rinfinity = number(data, "summary.Rinfty_cps")
    require_close(
        rinfinity,
        exact_transmissions.infinity * rpm,
        "Rinfty_cps from independently recomputed transmission",
    )

    any_phase_bound = number(data, "summary.B95_any_phase_cps")
    known_phase_bound = number(data, "summary.B95_known_phase_cps")
    require_close(
        number(data, "summary.eta95_any_phase"),
        any_phase_bound / rinfinity,
        "eta95_any_phase",
    )
    require_close(
        number(data, "summary.eta95_known_phase"),
        known_phase_bound / rinfinity,
        "eta95_known_phase",
    )
    require_close(
        number(data, "summary.Tinf_required_for_eta1_any_phase"),
        any_phase_bound / rpm,
        "Tinf_required_for_eta1_any_phase",
    )
    require_close(
        number(data, "summary.Tinf_required_for_eta1_known_phase"),
        known_phase_bound / rpm,
        "Tinf_required_for_eta1_known_phase",
    )

    for prefix in ("CL", "SL"):
        require_close(
            number(data, f"summary.known_phase_{prefix}_hat_cps"),
            number(data, f"summary.{prefix}_hat_cps"),
            f"known-phase {prefix} estimate",
        )
        require_close(
            number(data, f"summary.known_phase_{prefix}_sigma_cps"),
            number(data, f"summary.{prefix}_sigma_cps"),
            f"known-phase {prefix} uncertainty",
        )
    require_close(
        number(data, "summary.known_phase_CLSL_cov_cps2"),
        number(data, "summary.CLSL_cov_cps2"),
        "known-phase CL/SL covariance",
    )
    require_close(
        number(data, "summary.B_hat_cps"),
        math.hypot(
            number(data, "summary.CL_hat_cps"),
            number(data, "summary.SL_hat_cps"),
        ),
        "B_hat_cps",
    )
    require_close(
        number(data, "summary.known_phase_B_hat_cps"),
        math.hypot(
            number(data, "summary.known_phase_CL_hat_cps"),
            number(data, "summary.known_phase_SL_hat_cps"),
        ),
        "known_phase_B_hat_cps",
    )

    variance_c = number(data, "summary.CL_sigma_cps") ** 2
    variance_s = number(data, "summary.SL_sigma_cps") ** 2
    covariance = number(data, "summary.CLSL_cov_cps2")
    if variance_c <= 0 or variance_s <= 0:
        raise AnalysisError("added-quadrature variances must be positive")
    if variance_c * variance_s - covariance**2 <= 0:
        raise AnalysisError("added-quadrature covariance matrix is not positive definite")

    require_close(
        number(data, "summary.full_restoration_gap_cps"),
        rinfinity - number(data, "summary.B_hat_cps"),
        "full_restoration_gap_cps",
    )
    nearest_phase = number(data, "summary.full_restoration_nearest_phase_rad")
    nearest_c = rinfinity * math.cos(nearest_phase)
    nearest_s = rinfinity * math.sin(nearest_phase)
    require_close(
        number(data, "summary.full_restoration_nearest_C_cps"),
        nearest_c,
        "full_restoration_nearest_C_cps",
    )
    require_close(
        number(data, "summary.full_restoration_nearest_S_cps"),
        nearest_s,
        "full_restoration_nearest_S_cps",
    )
    full_restoration_distance = minimum_mahalanobis_distance(
        radius=rinfinity,
        center_c=number(data, "summary.CL_hat_cps"),
        center_s=number(data, "summary.SL_hat_cps"),
        variance_c=variance_c,
        variance_s=variance_s,
        covariance=covariance,
    )
    require_close(
        number(data, "summary.full_restoration_mahalanobis_distance"),
        full_restoration_distance,
        "full_restoration_mahalanobis_distance",
    )
    require_close(
        mahalanobis_distance(
            point_c=nearest_c,
            point_s=nearest_s,
            center_c=number(data, "summary.CL_hat_cps"),
            center_s=number(data, "summary.SL_hat_cps"),
            variance_c=variance_c,
            variance_s=variance_s,
            covariance=covariance,
        ),
        full_restoration_distance,
        "reported nearest full-restoration point",
    )
    if (
        text(data, "summary.full_restoration_distance_method")
        != "minimum_mahalanobis_distance_to_full_restoration_circle"
    ):
        raise AnalysisError("unexpected full_restoration_distance_method")
    validate_normalization_cases(
        data,
        infinity=exact_transmissions.infinity,
        finite_path=exact_transmissions.finite_path,
        rpm=rpm,
        bound=any_phase_bound,
        center_c=number(data, "summary.CL_hat_cps"),
        center_s=number(data, "summary.SL_hat_cps"),
        variance_c=variance_c,
        variance_s=variance_s,
        covariance=covariance,
    )

    for path in (
        "summary.p_null_any_phase",
        "summary.permutation_p_any_phase",
    ):
        result = number(data, path)
        if not 0 <= result <= 1:
            raise AnalysisError(f"{path} must be in [0, 1]")
    for path in (
        "summary.bootstrap_n_success",
        "summary.permutation_n_success",
    ):
        if number(data, path) <= 0:
            raise AnalysisError(f"{path} must be positive")



def build_macros(
    data: dict[str, Any],
    joint_data: dict[str, Any],
    conditions_data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> dict[str, str]:
    summary = data["summary"]
    erase_c = joint_fit(joint_data, condition="Erase", channel="C")
    preserve_c = joint_fit(joint_data, condition="Preserve", channel="C")
    launch_i = joint_fit(joint_data, condition="Launch", channel="I")
    transmissions = condition_transmissions(
        conditions_data, atmosphere_source, milky_way_basis
    )

    rpm = float(erase_c["total_fringe_cps"]) - float(
        preserve_c["total_fringe_cps"]
    )
    rpm_sigma = math.hypot(
        float(erase_c["total_fringe_sigma_cps"]),
        float(preserve_c["total_fringe_sigma_cps"]),
    )
    bound = float(summary["B95_any_phase_cps"])
    tinf = transmissions.infinity
    finite_path_transmission = transmissions.finite_path
    rinfty = tinf * rpm
    rfinite_path = finite_path_transmission * rpm
    eta_infinity = bound / rinfty
    eta_finite_path = bound / rfinite_path
    launch_singles = float(launch_i["total_fringe_cps"])

    variance_c = float(summary["CL_sigma_cps"]) ** 2
    variance_s = float(summary["SL_sigma_cps"]) ** 2
    covariance = float(summary["CLSL_cov_cps2"])
    center_c = float(summary["CL_hat_cps"])
    center_s = float(summary["SL_hat_cps"])

    def restoration_distance(transmission: float, multiplier: int = 1) -> float:
        return minimum_mahalanobis_distance(
            radius=multiplier * transmission * rpm,
            center_c=center_c,
            center_s=center_s,
            variance_c=variance_c,
            variance_s=variance_s,
            covariance=covariance,
        )

    nominal_distance = restoration_distance(tinf)

    direct_specs = (
        ValueSpec("ResultPairedScans", "summary.paired_n_passes", integer),
        ValueSpec(
            "ResultPairedScansAsText",
            "summary.paired_n_passes",
            integer_as_text,
        ),
        ValueSpec("ResultPairedPoints", "summary.paired_n_point_pairs", integer),
        ValueSpec("ResultPairedDegreesFreedom", "summary.paired_df", integer),
        ValueSpec("ResultReducedChiSquared", "summary.paired_reduced_chi2", lambda x: fixed(x, 3)),
        ValueSpec("ResultFringePeriod", "summary.period_V", lambda x: fixed(x, 3)),
        ValueSpec("ResultFringePeriodSigma", "summary.period_sigma_V", lambda x: fixed(x, 3)),
        ValueSpec("ResultCAddLetter", "summary.CL_hat_cps", lambda x: fixed(x, 1)),
        ValueSpec("ResultCAddSigmaLetter", "summary.CL_sigma_cps", lambda x: fixed(x, 1)),
        ValueSpec("ResultSAddLetter", "summary.SL_hat_cps", lambda x: fixed(x, 1)),
        ValueSpec("ResultSAddSigmaLetter", "summary.SL_sigma_cps", lambda x: fixed(x, 1)),
        ValueSpec("ResultCAddSupplement", "summary.CL_hat_cps", lambda x: fixed(x, 2)),
        ValueSpec("ResultCAddSigmaSupplement", "summary.CL_sigma_cps", lambda x: fixed(x, 2)),
        ValueSpec("ResultSAddSupplement", "summary.SL_hat_cps", lambda x: fixed(x, 2)),
        ValueSpec("ResultSAddSigmaSupplement", "summary.SL_sigma_cps", lambda x: fixed(x, 2)),
        ValueSpec("ResultCAddSAddCovariance", "summary.CLSL_cov_cps2", lambda x: fixed(x, 3)),
        ValueSpec("ResultAAddEstimateLetter", "summary.B_hat_cps", lambda x: fixed(x, 1)),
        ValueSpec("ResultAAddEstimateSupplement", "summary.B_hat_cps", lambda x: fixed(x, 3)),
        ValueSpec("ResultGaussianPValueLetter", "summary.p_null_any_phase", lambda x: fixed(x, 2)),
        ValueSpec("ResultGaussianPValueSupplement", "summary.p_null_any_phase", lambda x: fixed(x, 3)),
        ValueSpec("ResultPermutationPValueLetter", "summary.permutation_p_any_phase", lambda x: fixed(x, 2)),
        ValueSpec("ResultPermutationPValueSupplement", "summary.permutation_p_any_phase", lambda x: fixed(x, 3)),
        ValueSpec("ResultAAddBoundLetter", "summary.B95_any_phase_cps", lambda x: upper_bound(x, 1)),
        ValueSpec("ResultAAddBoundSupplement", "summary.B95_any_phase_cps", lambda x: fixed(x, 3)),
        ValueSpec("ResultFullRestorationGap", "summary.full_restoration_gap_cps", lambda x: fixed(x, 1)),
    )

    macros = {spec.macro: spec.formatter(number(data, spec.path)) for spec in direct_specs}
    macros.update(
        {
            "ResultRInfinityLetter": integer(rinfty),
            "ResultRInfinitySupplement": fixed(rinfty, 2),
            "ResultRFinitePathLetter": integer(rfinite_path),
            "ResultEtaInfinityBoundLetter": upper_bound(eta_infinity, 2),
            "ResultEtaInfinityBoundSupplement": fixed(eta_infinity, 3),
            "ResultTInfinityExact": fixed(tinf, 4),
            "ResultRpmMeasured": measured(rpm, rpm_sigma),
            "ResultPreserveCoincidenceAmplitudeMeasured": measured(
                float(preserve_c["total_fringe_cps"]),
                float(preserve_c["total_fringe_sigma_cps"]),
            ),
            "ResultPreserveCoincidenceVisibilityMeasured": rf"{fixed(float(preserve_c['V']), 3)}\pm{fixed(float(preserve_c['V_sigma']), 3)}",
            "ResultEraseCoincidenceAmplitudeMeasured": measured(
                float(erase_c["total_fringe_cps"]),
                float(erase_c["total_fringe_sigma_cps"]),
            ),
            "ResultEraseCoincidenceVisibilityMeasured": rf"{fixed(float(erase_c['V']), 3)}\pm{fixed(float(erase_c['V_sigma']), 3)}",
            "ResultLaunchSinglesAmplitudeMeasured": measured(
                float(launch_i["total_fringe_cps"]),
                float(launch_i["total_fringe_sigma_cps"]),
            ),
            "ResultEtaFinitePathBoundLetter": upper_bound(eta_finite_path, 2),
            "ResultFullRestorationSigmaLetter": fixed(nominal_distance, 1),
            "ResultRestorationToLaunchSinglesRatio": fixed(
                rinfty / launch_singles, 1
            ),
        }
    )
    return macros


def render(
    source_path: Path,
    joint_source_path: Path,
    conditions_source_path: Path,
    source_bytes: bytes,
    joint_source_bytes: bytes,
    conditions_source_bytes: bytes,
    data: dict[str, Any],
    joint_data: dict[str, Any],
    conditions_data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> str:
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    joint_source_hash = hashlib.sha256(joint_source_bytes).hexdigest()
    conditions_source_hash = hashlib.sha256(conditions_source_bytes).hexdigest()
    macros = build_macros(
        data,
        joint_data,
        conditions_data,
        atmosphere_source,
        milky_way_basis,
    )
    lines = [
        "% Generated file. Do not edit.",
        f"% Null source: {source_path.name}",
        f"% Null source SHA-256: {source_hash}",
        f"% Joint-fit source: {joint_source_path.name}",
        f"% Joint-fit source SHA-256: {joint_source_hash}",
        f"% Conditions source: {conditions_source_path.name}",
        f"% Conditions source SHA-256: {conditions_source_hash}",
    ]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    )
    return "\n".join(lines) + "\n"


def write_atomically_if_changed(path: Path, contents: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == contents:
            return
    except FileNotFoundError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semantic TeX macros from the pinned analysis JSON."
    )
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--joint-source", type=Path, required=True)
    parser.add_argument(
        "--conditions-source",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atmosphere-source", choices=("goes", "aeronet"), default="goes")
    parser.add_argument(
        "--milky-way-basis",
        choices=MILKY_WAY_BASES,
        default=DEFAULT_MILKY_WAY_BASIS,
        help="Milky Way transmission value used in T_inf and downstream analysis",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        source_bytes = args.source.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"cannot read pinned result {args.source}: {exc}") from exc

    try:
        data = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON in {args.source}: {exc}") from exc
    if not isinstance(data, dict):
        raise AnalysisError("analysis result must be a JSON object")

    try:
        joint_source_bytes = args.joint_source.read_bytes()
    except OSError as exc:
        raise AnalysisError(
            f"cannot read pinned joint-fit result {args.joint_source}: {exc}"
        ) from exc

    try:
        joint_data = json.loads(joint_source_bytes)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON in {args.joint_source}: {exc}") from exc
    if not isinstance(joint_data, dict):
        raise AnalysisError("joint-fit result must be a JSON object")

    try:
        conditions_source_bytes = args.conditions_source.read_bytes()
    except OSError as exc:
        raise AnalysisError(
            f"cannot read pinned conditions {args.conditions_source}: {exc}"
        ) from exc

    try:
        conditions_data = json.loads(conditions_source_bytes)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"invalid JSON in {args.conditions_source}: {exc}"
        ) from exc
    if not isinstance(conditions_data, dict):
        raise AnalysisError("conditions result must be a JSON object")

    validate(
        data,
        joint_data,
        conditions_data,
        args.atmosphere_source,
        args.milky_way_basis,
    )
    expected = render(
        args.source,
        args.joint_source,
        args.conditions_source,
        source_bytes,
        joint_source_bytes,
        conditions_source_bytes,
        data,
        joint_data,
        conditions_data,
        args.atmosphere_source,
        args.milky_way_basis,
    )

    if args.check:
        try:
            actual = args.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"stale generated file: {args.output} does not exist", file=sys.stderr)
            return 1
        if actual != expected:
            print(
                f"stale generated file: run {Path(__file__).relative_to(ROOT)}",
                file=sys.stderr,
            )
            return 1
        return 0

    write_atomically_if_changed(args.output, expected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
