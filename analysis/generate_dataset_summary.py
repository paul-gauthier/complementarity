#!/usr/bin/env python3
"""Generate the all-datasets TeX summary table."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .artifact_config import load_config
from .transmission_values import calculate_transmissions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = ROOT / "build"
DEFAULT_TRANSMISSION_OUTPUT = (
    DEFAULT_SOURCE_ROOT / "tables" / "dataset-transmission.tex"
)
DEFAULT_ANALYSIS_OUTPUT = DEFAULT_SOURCE_ROOT / "tables" / "dataset-analysis.tex"


class SummaryError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError(f"expected a JSON object in {path}")
    return data


def number(data: dict[str, Any], path: str, source: Path) -> float:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise SummaryError(f"{source}: missing numeric field {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise SummaryError(f"{source}: field {path} is not numeric")
    result = float(current)
    if not math.isfinite(result):
        raise SummaryError(f"{source}: field {path} is not finite")
    return result


def normalization_case(
    bounds: dict[str, Any],
    *,
    transmission: float,
    multiplier: float,
    source: Path,
) -> dict[str, Any]:
    summary = bounds.get("summary")
    if not isinstance(summary, dict):
        raise SummaryError(f"{source}: missing summary object")
    cases = summary.get("normalization_cases")
    if not isinstance(cases, list):
        raise SummaryError(f"{source}: missing normalization cases")

    matches: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise SummaryError(f"{source}: invalid normalization case")
        case_transmission = number(case, "T", source)
        case_multiplier = number(case, "m", source)
        if math.isclose(
            case_transmission,
            transmission,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) and math.isclose(
            case_multiplier,
            multiplier,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            matches.append(case)
    if len(matches) != 1:
        raise SummaryError(
            f"{source}: expected one normalization case for "
            f"T={transmission} and m={multiplier}, found {len(matches)}"
        )
    return matches[0]


def fixed(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def significant(
    value: float,
    digits: int,
    *,
    rounding: str = ROUND_HALF_UP,
) -> str:
    if digits < 1:
        raise ValueError("significant-figure count must be positive")
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("value must be finite")
    if decimal.is_zero():
        return "0"
    decimals = digits - decimal.copy_abs().adjusted() - 1
    quantum = Decimal(1).scaleb(-decimals)
    rounded = decimal.quantize(quantum, rounding=rounding)
    return f"{rounded:.{max(decimals, 0)}f}"


def measured(value: float, uncertainty: float) -> tuple[str, str]:
    decimal_uncertainty = Decimal(str(uncertainty))
    if not decimal_uncertainty.is_finite() or decimal_uncertainty <= 0:
        raise ValueError("uncertainty must be finite and positive")
    adjusted = decimal_uncertainty.adjusted()
    leading_digit = int(decimal_uncertainty.scaleb(-adjusted))
    digits = 2 if leading_digit == 1 else 1
    decimals = digits - adjusted - 1
    quantum = Decimal(1).scaleb(-decimals)
    rounded_value = Decimal(str(value)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    rounded_uncertainty = decimal_uncertainty.quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    places = max(decimals, 0)
    return f"{rounded_value:.{places}f}", f"{rounded_uncertainty:.{places}f}"


def selected_transmissions(
    *,
    config: dict[str, Any],
    dataset: dict[str, Any],
    conditions: dict[str, Any],
    source: Path,
):
    profile = config["conditions_profiles"][dataset["conditions_profile"]]
    aeronet_tag = f"AERONET-{profile['aeronet_variant']}"
    atmosphere_tag = (
        "GOES-18"
        if profile["atmosphere_source"] == "goes"
        else aeronet_tag
    )
    records = {
        record.get("tag"): record
        for record in conditions.get("libradtran", [])
        if isinstance(record, dict) and isinstance(record.get("tag"), str)
    }
    try:
        atmosphere = float(records[atmosphere_tag]["result"]["t_band"])
        aeronet_atmosphere = float(records[aeronet_tag]["result"]["t_band"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(
            f"{source}: incomplete transmission record for {atmosphere_tag}"
        ) from exc
    result = calculate_transmissions(
        atmosphere=atmosphere,
        aeronet_atmosphere=aeronet_atmosphere,
        milky_way_point=number(
            conditions, "irsa.transmission_fraction", source
        ),
        milky_way_integrated=number(
            conditions, "irsa.integration.avg_T", source
        ),
        milky_way_minimum=number(
            conditions, "irsa.integration.min_T", source
        ),
        milky_way_basis=config["transmission_analysis"]["milky_way_basis"],
    )
    source_mark = "G" if atmosphere_tag == "GOES-18" else "A"
    return result, source_mark


def render_tables(
    config: dict[str, Any], source_root: Path
) -> tuple[str, str]:
    transmission_rows: list[str] = []
    analysis_rows: list[str] = []
    for dataset in config["datasets"]:
        dataset_root = source_root / "datasets" / dataset["id"]
        conditions_path = dataset_root / "conditions" / "conditions.json"
        bounds_path = dataset_root / "analysis" / "mzi-null-bounds.json"
        conditions = load_json(conditions_path)
        bounds = load_json(bounds_path)
        transmission, source_mark = selected_transmissions(
            config=config,
            dataset=dataset,
            conditions=conditions,
            source=conditions_path,
        )
        summary_tinf = number(bounds, "summary.Tinf", bounds_path)
        if not math.isclose(
            summary_tinf,
            transmission.infinity,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SummaryError(
                f"{dataset['id']}: null-bound and conditions transmissions disagree"
            )

        rpm = number(bounds, "summary.Rpm_cps", bounds_path)
        rpm_sigma = number(bounds, "summary.Rpm_sigma_cps", bounds_path)
        rinfinity = number(bounds, "summary.Rinfty_cps", bounds_path)
        bound = number(bounds, "summary.B95_any_phase_cps", bounds_path)
        eta = number(bounds, "summary.eta95_any_phase", bounds_path)
        distance = number(
            bounds, "summary.full_restoration_mahalanobis_distance", bounds_path
        )
        finite_case = normalization_case(
            bounds,
            transmission=transmission.finite_path,
            multiplier=1.0,
            source=bounds_path,
        )
        rfinite = number(finite_case, "restoration_radius_cps", bounds_path)
        eta_finite = number(finite_case, "eta95", bounds_path)
        distance_finite = number(finite_case, "d_min", bounds_path)
        if not math.isclose(
            rinfinity,
            transmission.infinity * rpm,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SummaryError(f"{dataset['id']}: R_infinity is inconsistent")
        if not math.isclose(
            eta,
            bound / rinfinity,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SummaryError(f"{dataset['id']}: eta_infinity,95 is inconsistent")
        if not math.isclose(
            rfinite,
            transmission.finite_path * rpm,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SummaryError(f"{dataset['id']}: R_finite_path is inconsistent")
        if not math.isclose(
            eta_finite,
            bound / rfinite,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SummaryError(f"{dataset['id']}: eta_finite,95 is inconsistent")

        label = dataset["display_id"]
        transmission_rows.append(
            f"{label}"
            f" & ${fixed(transmission.atmosphere, 4)}^{{\\mathrm{{{source_mark}}}}}$"
            f" & ${fixed(transmission.milky_way_point, 4)}$"
            f" & ${fixed(transmission.milky_way_integrated, 4)}$"
            f" & ${fixed(transmission.milky_way_minimum, 4)}$"
            f" & ${fixed(transmission.finite_path, 4)}$"
            f" & ${fixed(transmission.infinity, 4)}$"
            " \\\\"
        )
        rpm_display, rpm_sigma_display = measured(rpm, rpm_sigma)
        analysis_rows.append(
            f"{label}"
            f" & ${rpm_display}\\pm{rpm_sigma_display}$"
            f" & ${significant(bound, 2, rounding=ROUND_CEILING)}$"
            f" & ${significant(rinfinity, 2)}$"
            f" & ${significant(eta, 2, rounding=ROUND_CEILING)}$"
            f" & ${significant(distance, 2)}$"
            f" & ${significant(rfinite, 2)}$"
            f" & ${significant(eta_finite, 2, rounding=ROUND_CEILING)}$"
            f" & ${significant(distance_finite, 2)}$ \\\\"
        )

    integration_minutes = {
        profile["integration_minutes"]
        for profile in config["conditions_profiles"].values()
    }
    if len(integration_minutes) != 1:
        raise SummaryError(
            "all dataset transmission profiles must use one integration window"
        )
    basis = config["transmission_analysis"]["milky_way_basis"]
    transmission_lines = [
        "% Generated file. Do not edit.",
        f"% T_inf uses the {basis} Milky Way transmission.",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}cccccc@{}}",
        r"\toprule",
        r"& \multicolumn{6}{c}{Transmission} \\",
        r"\cmidrule(l){2-7}",
        r"Dataset",
        r"& $T_{\mathrm{atm}}$",
        r"& $T_{\mathrm{MW}}(t_0)$",
        r"& $\langle T_{\mathrm{MW}}\rangle$",
        r"& $\min(T_{\mathrm{MW}})$",
        r"& $T_{\mathrm{fin}}$",
        r"& $T_\infty$ \\",
        r"\midrule",
        *transmission_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        "",
    ]
    analysis_lines = [
        "% Generated file. Do not edit.",
        f"% T_inf uses the {basis} Milky Way transmission.",
        r"\providecommand{\DatasetAnalysisTableStyle}{%",
        r"  \footnotesize\setlength{\tabcolsep}{1.5pt}%",
        r"}%",
        r"\DatasetAnalysisTableStyle{}",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}cccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Model independent}",
        r"& \multicolumn{3}{c}{Infinite future}",
        r"& \multicolumn{3}{c}{Finite path} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-9}",
        r"Dataset",
        r"& $R_{\mathrm{pm}}$",
        r"& $A_{\mathrm{add},95}$",
        r"& $R_\infty$",
        r"& $\eta_{\infty,95}$",
        r"& $Z_\infty$",
        r"& $R_{\mathrm{fin}}$",
        r"& $\eta_{\mathrm{fin},95}$",
        r"& $Z_{\mathrm{fin}}$ \\",
        r"\midrule",
        *analysis_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        "",
    ]
    return "\n".join(transmission_lines), "\n".join(analysis_lines)


def write_atomically(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate transmission and analysis TeX tables for all datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--transmission-output", type=Path, default=DEFAULT_TRANSMISSION_OUTPUT
    )
    parser.add_argument(
        "--analysis-output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT
    )
    args = parser.parse_args()
    transmission, analysis = render_tables(load_config(), args.source_root)
    write_atomically(args.transmission_output, transmission)
    write_atomically(args.analysis_output, analysis)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
