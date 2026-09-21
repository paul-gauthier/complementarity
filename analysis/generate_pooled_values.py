#!/usr/bin/env python3
"""Generate manuscript-value macros from the pooled dataset analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "build" / "analysis" / "mzi-pooled-analysis.json"
DEFAULT_OUTPUT = (
    ROOT / "build" / "manuscript-values" / "pooled-analysis-values.tex"
)


class PooledValuesError(ValueError):
    """Raised when the pooled result cannot produce manuscript values."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PooledValuesError(f"cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PooledValuesError(f"{path}: expected a JSON object")
    return data


def _number(data: dict[str, Any], path: str, source: Path) -> float:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PooledValuesError(f"{source}: missing numeric field {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise PooledValuesError(f"{source}: field {path} is not numeric")
    result = float(current)
    if not math.isfinite(result):
        raise PooledValuesError(f"{source}: field {path} is not finite")
    return result


def _fixed(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def _upper_bound(value: float, decimals: int) -> str:
    """Format a nonnegative upper endpoint without ever rounding downward."""
    if value < 0.0:
        raise PooledValuesError("upper bounds must be nonnegative")
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_CEILING)
    return f"{rounded:.{decimals}f}"


def _significant(value: float, digits: int) -> str:
    if not math.isfinite(value) or digits <= 0:
        raise PooledValuesError("significant-figure inputs must be finite")
    if value == 0.0:
        return "0"
    decimals = digits - 1 - math.floor(math.log10(abs(value)))
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    displayed_decimals = max(0, decimals)
    return f"{rounded:.{displayed_decimals}f}"


def render_macros(source: Path) -> str:
    pooled = _load_json(source)
    if pooled.get("schema_version") != 2:
        raise PooledValuesError(f"{source}: unsupported pooled schema version")
    if not isinstance(pooled.get("summary"), dict):
        raise PooledValuesError(f"{source}: missing summary object")

    def number(field: str) -> float:
        return _number(pooled, f"summary.{field}", source)

    macros = [
        ("ResultPooledEtaEstimateLetter", _fixed(number("eta_hat"), 2)),
        (
            "ResultPooledEtaInfinityBoundLetter",
            _upper_bound(number("eta_upper"), 2),
        ),
        (
            "ResultPooledNullPValueLetter",
            _fixed(number("null_monte_carlo_pvalue"), 2),
        ),
        (
            "ResultPooledGoodnessOfFitPValueLetter",
            _fixed(number("heterogeneity_monte_carlo_pvalue"), 2),
        ),
        (
            "ResultPooledLeaveOneOutMinimum",
            _fixed(number("leave_one_out_eta_upper_min"), 3),
        ),
        (
            "ResultPooledLeaveOneOutMaximum",
            _fixed(number("leave_one_out_eta_upper_max"), 3),
        ),
        (
            "ResultPooledLaunchOpticsTransmissionLetter",
            _significant(number("launch_optics_transmission"), 2),
        ),
        (
            "ResultPooledAtmosphericTransmissionMinimumLetter",
            _significant(number("atmosphere_transmission_min"), 2),
        ),
        (
            "ResultPooledAtmosphericTransmissionMaximumLetter",
            _significant(number("atmosphere_transmission_max"), 2),
        ),
        (
            "ResultPooledMilkyWayTransmissionMinimumLetter",
            _significant(number("milky_way_transmission_min"), 2),
        ),
        (
            "ResultPooledMilkyWayTransmissionMaximumLetter",
            _significant(number("milky_way_transmission_max"), 2),
        ),
        (
            "ResultPooledCommonIgmTransmission",
            _fixed(number("common_igm_transmission"), 4),
        ),
        (
            "ResultPooledCommonIgmTransmissionLetter",
            _significant(number("common_igm_transmission"), 2),
        ),
        (
            "ResultPooledTInfinityMinimumLetter",
            _significant(number("T_infinity_min"), 2),
        ),
        (
            "ResultPooledTInfinityMaximumLetter",
            _significant(number("T_infinity_max"), 2),
        ),
        (
            "ResultPooledTInfinityMinimumPercentLetter",
            _fixed(100.0 * number("T_infinity_min"), 0),
        ),
        (
            "ResultPooledTInfinityMaximumPercentLetter",
            _fixed(100.0 * number("T_infinity_max"), 0),
        ),
        (
            "ResultPooledTFinitePathMinimumLetter",
            _significant(number("T_finite_path_min"), 2),
        ),
        (
            "ResultPooledTFinitePathMaximumLetter",
            _significant(number("T_finite_path_max"), 2),
        ),
        (
            "ResultPooledEtaFinitePathBoundLetter",
            _upper_bound(number("eta_upper_finite_path"), 2),
        ),
        (
            "ResultPooledEtaLaunchInfinityBoundLetter",
            _upper_bound(number("eta_upper_launch_infinity"), 3),
        ),
        (
            "ResultPooledEtaLaunchFinitePathBoundLetter",
            _upper_bound(number("eta_upper_launch_finite_path"), 3),
        ),
        (
            "ResultPooledLaunchOpticsLossPercentLetter",
            _significant(100.0 * (1.0 - number("launch_optics_transmission")), 2),
        ),
        (
            "ResultPooledAtmosphericLossMinimumPercentLetter",
            _significant(100.0 * (1.0 - number("atmosphere_transmission_max")), 2),
        ),
        (
            "ResultPooledAtmosphericLossMaximumPercentLetter",
            _significant(100.0 * (1.0 - number("atmosphere_transmission_min")), 2),
        ),
        (
            "ResultPooledMilkyWayLossMinimumPercentLetter",
            _significant(100.0 * (1.0 - number("milky_way_transmission_max")), 2),
        ),
        (
            "ResultPooledMilkyWayLossMaximumPercentLetter",
            _significant(100.0 * (1.0 - number("milky_way_transmission_min")), 2),
        ),
        (
            "ResultPooledIgmLossPercentLetter",
            _significant(100.0 * (1.0 - number("common_igm_transmission")), 2),
        ),
        (
            "ResultPooledTInfinityLossMinimumPercentLetter",
            _significant(100.0 * (1.0 - number("T_infinity_max")), 2),
        ),
        (
            "ResultPooledTInfinityLossMaximumPercentLetter",
            _significant(100.0 * (1.0 - number("T_infinity_min")), 2),
        ),
        (
            "ResultPooledTFinitePathLossMinimumPercentLetter",
            _significant(100.0 * (1.0 - number("T_finite_path_max")), 2),
        ),
        (
            "ResultPooledTFinitePathLossMaximumPercentLetter",
            _significant(100.0 * (1.0 - number("T_finite_path_min")), 2),
        ),
    ]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    lines = [
        "% Generated file. Do not edit.",
        f"% Source: {source.name}",
        f"% Source SHA-256: {digest}",
        *[rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros],
        "",
    ]
    return "\n".join(lines)


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
        description="Generate TeX macros from the pooled dataset analysis."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_atomically(args.output, render_macros(args.source))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PooledValuesError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
