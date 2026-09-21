#!/usr/bin/env python
"""Report measured dark-run background rates and accidental-coincidence corrections."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .artifact_config import get_dataset, load_config, dataset_runs
from .dark_analysis import COINCIDENCE_WINDOW_S, DarkModel
from .mzi_io import load_dark, scan_from_records

DEFAULT_JSON_FILENAME = "darks-accidentals.json"
DEFAULT_DARK_TEX_FILENAME = "dark-rates.tex"
DEFAULT_ACCIDENTAL_TEX_FILENAME = "accidental-corrections.tex"

CHANNEL_MAPPING = {
    "N_i": "S1",
    "N_i2": "S2",
    "N_s": "I",
    "N_c": "C1",
    "N_c2": "C2",
}

@dataclass(frozen=True)
class RunSpec:
    run: str
    run_name: str
    condition: str
    directory: str
    detected_idler: bool


def run_specs_from_config(dataset_id: str | None = None) -> tuple[RunSpec, ...]:
    config = load_config()
    if dataset_id is None:
        dataset_id = config["datasets"][0]["id"]
    configured_runs = dataset_runs(config, get_dataset(config, dataset_id))
    return tuple(
        RunSpec(
            run=item["run"],
            run_name=item["run_name"],
            condition=item["condition"],
            directory=item["path"],
            detected_idler=item["detected_idler"],
        )
        for item in configured_runs
    )


@dataclass
class CrossCheck:
    compared_values: int = 0
    max_absolute_residual: float = 0.0
    max_relative_residual: float = 0.0

    def compare(
        self,
        *,
        run_label: str,
        point_index: int,
        field: str,
        expected: float,
        observed: float,
        rtol: float = 1e-12,
        atol: float = 1e-12,
    ) -> None:
        expected_f = float(expected)
        observed_f = float(observed)
        if not (np.isfinite(expected_f) and np.isfinite(observed_f)):
            raise ValueError(
                f"{run_label}, point {point_index + 1}, {field}: "
                f"non-finite comparison expected={expected_f!r}, observed={observed_f!r}"
            )

        absolute = abs(observed_f - expected_f)
        if expected_f != 0.0:
            relative = absolute / abs(expected_f)
        else:
            relative = 0.0 if absolute == 0.0 else float("inf")

        if not math.isclose(observed_f, expected_f, rel_tol=rtol, abs_tol=atol):
            raise ValueError(
                f"{run_label}, point {point_index + 1}, {field}: "
                f"expected {expected_f:.17g}, observed {observed_f:.17g}, "
                f"absolute residual {absolute:.17g}"
            )

        self.compared_values += 1
        self.max_absolute_residual = max(self.max_absolute_residual, absolute)
        self.max_relative_residual = max(self.max_relative_residual, relative)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _read_points(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}")
                records.append(record)
    except FileNotFoundError as exc:
        raise ValueError(f"Required input does not exist: {path}") from exc

    if not records:
        raise ValueError(f"No acquisition records found in {path}")
    return records


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _plan_details(
    plan_payload: Dict[str, Any],
    *,
    run_label: str,
    record_count: int,
) -> Dict[str, Any]:
    if not isinstance(plan_payload, dict):
        raise ValueError(f"{run_label}: plan.json must contain an object")

    voltages = plan_payload.get("deltas_exec_V")
    if not isinstance(voltages, list) or not voltages:
        raise ValueError(
            f"{run_label}: plan.json.deltas_exec_V must be a non-empty list"
        )

    points_per_scan = len(voltages)
    if record_count % points_per_scan != 0:
        raise ValueError(
            f"{run_label}: {record_count} records are incompatible with "
            f"the {points_per_scan}-point scan plan"
        )

    finite_voltages: List[float] = []
    for index, value in enumerate(voltages):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{run_label}: deltas_exec_V[{index}] is not numeric"
            )
        voltage = float(value)
        if not np.isfinite(voltage):
            raise ValueError(
                f"{run_label}: deltas_exec_V[{index}] is not finite"
            )
        finite_voltages.append(voltage)

    return {
        "points_per_scan": points_per_scan,
        "scan_count": record_count // points_per_scan,
        "planned_voltage_min_V": min(finite_voltages),
        "planned_voltage_max_V": max(finite_voltages),
    }


def _required_count(
    record: Dict[str, Any],
    key: str,
    *,
    run_label: str,
    point_index: int,
) -> int:
    counts = record.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(
            f"{run_label}, point {point_index + 1}: missing counts object"
        )
    if key not in counts:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: missing counts.{key}"
        )
    try:
        value = int(counts[key])
    except Exception as exc:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: counts.{key} is not an integer"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: counts.{key} is negative"
        )
    return value


def _duration(
    record: Dict[str, Any],
    *,
    run_label: str,
    point_index: int,
) -> float:
    try:
        duration = float(record["duration_s"])
    except Exception as exc:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: invalid duration_s"
        ) from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: duration_s must be positive"
        )
    return duration


def _actual_voltage(
    record: Dict[str, Any],
    *,
    run_label: str,
    point_index: int,
) -> float:
    try:
        voltage = float(record["actual_delta_V"])
    except Exception as exc:
        raise ValueError(
            f"{run_label}, point {point_index + 1}: invalid actual_delta_V"
        ) from exc
    if not np.isfinite(voltage):
        raise ValueError(
            f"{run_label}, point {point_index + 1}: actual_delta_V is not finite"
        )
    return voltage


def _rate_or_none(numerator: float, denominator: float) -> Optional[float]:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _dark_measurement(count: int, exposure_s: float) -> Dict[str, float | int]:
    return {
        "counts": int(count),
        "rate_cps": float(count / exposure_s),
        "poisson_sigma_cps": float(math.sqrt(count) / exposure_s),
    }


def _dark_summary(
    dark: DarkModel,
) -> Dict[str, Any]:
    exposure_s = float(dark.T)

    return {
        "exposure_s": exposure_s,
        "S1": _dark_measurement(dark.Ni, exposure_s),
        "S2": _dark_measurement(dark.Ni2, exposure_s),
        "I": _dark_measurement(dark.Ns, exposure_s),
        "C1": _dark_measurement(dark.Nc, exposure_s),
        "C2": _dark_measurement(dark.Nc2, exposure_s),
        "C1_residual_dark_coincidence_background_cps": float(dark.dark_excess),
        "C1_residual_dark_coincidence_background_sigma_cps": float(
            math.sqrt(max(0.0, dark.var_dark_excess))
        ),
        "C2_residual_dark_coincidence_background_cps": float(dark.dark_excess2),
        "C2_residual_dark_coincidence_background_sigma_cps": float(
            math.sqrt(max(0.0, dark.var_dark_excess2))
        ),
    }


def _analyze_records(
    records: Sequence[Dict[str, Any]],
    dark: Any,
    *,
    run_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scan_data, _ = scan_from_records(list(records), dark)
    expected_length = len(records)
    scan_fields = {
        "xs": scan_data.xs,
        "ri1": scan_data.ri1,
        "ri2": scan_data.ri2,
        "rc1": scan_data.rc1,
        "rc2": scan_data.rc2,
        "var_rc1": scan_data.var_rc1,
        "var_rc2": scan_data.var_rc2,
    }
    for field, values in scan_fields.items():
        if len(values) != expected_length:
            raise ValueError(
                f"{run_label}: scan_from_records returned {len(values)} {field} values "
                f"for {expected_length} records"
            )

    total_exposure = 0.0
    count_totals = {key: 0 for key in CHANNEL_MAPPING}
    corrected_s1_exposure = 0.0
    corrected_s2_exposure = 0.0
    corrected_c1_exposure = 0.0
    corrected_c2_exposure = 0.0
    accidental_c1_counts = 0.0
    accidental_c2_counts = 0.0
    clipped_s1_points = 0
    clipped_s2_points = 0
    check = CrossCheck()

    for index, record in enumerate(records):
        duration = _duration(record, run_label=run_label, point_index=index)
        voltage = _actual_voltage(record, run_label=run_label, point_index=index)

        counts = {
            key: _required_count(
                record,
                key,
                run_label=run_label,
                point_index=index,
            )
            for key in CHANNEL_MAPPING
        }
        for key, value in counts.items():
            count_totals[key] += value

        n_s1 = counts["N_i"]
        n_s2 = counts["N_i2"]
        n_i = counts["N_s"]
        n_c1 = counts["N_c"]
        n_c2 = counts["N_c2"]

        accidental_c1 = COINCIDENCE_WINDOW_S * n_i * n_s1 / duration
        accidental_c2 = COINCIDENCE_WINDOW_S * n_i * n_s2 / duration

        scale = COINCIDENCE_WINDOW_S / duration
        accidental_c1_variance = scale * scale * (
            n_s1 * n_s1 * n_i + n_i * n_i * n_s1
        )
        accidental_c2_variance = scale * scale * (
            n_s2 * n_s2 * n_i + n_i * n_i * n_s2
        )

        s1_unclipped = n_s1 / duration - float(dark.r_i)
        s2_unclipped = n_s2 / duration - float(dark.r_i2)
        corrected_s1 = max(s1_unclipped, 0.0)
        corrected_s2 = max(s2_unclipped, 0.0)
        if corrected_s1 != s1_unclipped:
            clipped_s1_points += 1
        if corrected_s2 != s2_unclipped:
            clipped_s2_points += 1

        corrected_c1 = (
            (n_c1 - accidental_c1) / duration - float(dark.dark_excess)
        )
        corrected_c2 = (
            (n_c2 - accidental_c2) / duration - float(dark.dark_excess2)
        )
        corrected_c1_variance = (
            n_c1 + accidental_c1_variance + float(dark.var_dark_excess) * duration**2
        ) / duration**2
        corrected_c2_variance = (
            n_c2 + accidental_c2_variance + float(dark.var_dark_excess2) * duration**2
        ) / duration**2

        check.compare(
            run_label=run_label,
            point_index=index,
            field="actual_delta_V",
            expected=voltage,
            observed=scan_data.xs[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="S1 corrected rate",
            expected=corrected_s1,
            observed=scan_data.ri1[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="S2 corrected rate",
            expected=corrected_s2,
            observed=scan_data.ri2[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="C1 corrected rate",
            expected=corrected_c1,
            observed=scan_data.rc1[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="C2 corrected rate",
            expected=corrected_c2,
            observed=scan_data.rc2[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="C1 corrected variance",
            expected=corrected_c1_variance,
            observed=scan_data.var_rc1[index],
        )
        check.compare(
            run_label=run_label,
            point_index=index,
            field="C2 corrected variance",
            expected=corrected_c2_variance,
            observed=scan_data.var_rc2[index],
        )

        total_exposure += duration
        corrected_s1_exposure += corrected_s1 * duration
        corrected_s2_exposure += corrected_s2 * duration
        corrected_c1_exposure += corrected_c1 * duration
        corrected_c2_exposure += corrected_c2 * duration
        accidental_c1_counts += accidental_c1
        accidental_c2_counts += accidental_c2

    raw_rates = {
        paper_name: float(count_totals[raw_name] / total_exposure)
        for raw_name, paper_name in CHANNEL_MAPPING.items()
    }
    accidental_rates = {
        "C1_cps": float(accidental_c1_counts / total_exposure),
        "C2_cps": float(accidental_c2_counts / total_exposure),
        "total_cps": float(
            (accidental_c1_counts + accidental_c2_counts) / total_exposure
        ),
    }
    raw_coincidence_total = raw_rates["C1"] + raw_rates["C2"]
    accidental_rates["fraction_of_raw_total"] = _rate_or_none(
        accidental_rates["total_cps"],
        raw_coincidence_total,
    )

    analysis = {
        "point_count": len(records),
        "total_exposure_s": float(total_exposure),
        "raw_counts": {
            paper_name: int(count_totals[raw_name])
            for raw_name, paper_name in CHANNEL_MAPPING.items()
        },
        "raw_rates_cps": raw_rates,
        "dark_subtracted_singles_cps": {
            "S1": float(corrected_s1_exposure / total_exposure),
            "S2": float(corrected_s2_exposure / total_exposure),
        },
        "unclipped_aggregate_dark_subtracted_singles_cps": {
            "S1": float(raw_rates["S1"] - dark.r_i),
            "S2": float(raw_rates["S2"] - dark.r_i2),
        },
        "corrected_coincidence_rates_cps": {
            "C1": float(corrected_c1_exposure / total_exposure),
            "C2": float(corrected_c2_exposure / total_exposure),
            "total": float(
                (corrected_c1_exposure + corrected_c2_exposure) / total_exposure
            ),
        },
        "accidental_rates": accidental_rates,
        "singles_zero_clipping": {
            "S1_point_count": clipped_s1_points,
            "S2_point_count": clipped_s2_points,
        },
    }
    cross_check = {
        "status": "passed",
        "compared_values": check.compared_values,
        "max_absolute_residual": float(check.max_absolute_residual),
        "max_relative_residual": float(check.max_relative_residual),
    }
    return analysis, cross_check


def _nullable_rate_budget(
    analysis: Dict[str, Any],
    *,
    detected_idler: bool,
) -> Dict[str, Any]:
    raw = analysis["raw_rates_cps"]
    corrected = analysis["corrected_coincidence_rates_cps"]
    accidentals = analysis["accidental_rates"]

    return {
        "raw_singles_cps": {
            "S1": raw["S1"],
            "S2": raw["S2"],
            "I": raw["I"] if detected_idler else None,
        },
        "dark_subtracted_singles_cps": analysis["dark_subtracted_singles_cps"],
        "raw_coincidence_rates_cps": {
            "C1": raw["C1"] if detected_idler else None,
            "C2": raw["C2"] if detected_idler else None,
            "total": (raw["C1"] + raw["C2"]) if detected_idler else None,
        },
        "corrected_coincidence_rates_cps": {
            "C1": corrected["C1"] if detected_idler else None,
            "C2": corrected["C2"] if detected_idler else None,
            "total": corrected["total"] if detected_idler else None,
        },
        "accidental_rates_cps": {
            "C1": accidentals["C1_cps"] if detected_idler else None,
            "C2": accidentals["C2_cps"] if detected_idler else None,
            "total": accidentals["total_cps"] if detected_idler else None,
        },
        "accidental_fraction_of_raw_total": (
            accidentals["fraction_of_raw_total"] if detected_idler else None
        ),
    }


def _analyze_run(repo_root: Path, spec: RunSpec) -> Dict[str, Any]:
    run_dir = repo_root / spec.directory
    points_path = run_dir / "points.jsonl"
    plan_path = run_dir / "plan.json"
    run_label = f"run {spec.run} {spec.condition}"

    dark = load_dark(run_dir)
    plan_payload = _read_json(plan_path)
    if not isinstance(plan_payload, dict):
        raise ValueError(f"{run_label}: plan.json must contain an object")
    records = _read_points(points_path)

    plan = _plan_details(
        plan_payload,
        run_label=run_label,
        record_count=len(records),
    )
    dark_report = _dark_summary(dark)
    analysis, cross_check = _analyze_records(
        records,
        dark,
        run_label=run_label,
    )

    return {
        "run": spec.run,
        "run_name": spec.run_name,
        "condition": spec.condition,
        "detected_idler": spec.detected_idler,
        "directory": spec.directory,
        "inputs": {
            "dark": f"{spec.directory}/dark.json",
            "points": f"{spec.directory}/points.jsonl",
            "plan": f"{spec.directory}/plan.json",
        },
        "plan": plan,
        "point_count": analysis["point_count"],
        "total_exposure_s": analysis["total_exposure_s"],
        "dark_run_background_measurements": dark_report,
        "rate_budget": _nullable_rate_budget(
            analysis,
            detected_idler=spec.detected_idler,
        ),
        "diagnostics": {
            "internal_raw_counts": analysis["raw_counts"],
            "unclipped_aggregate_dark_subtracted_singles_cps": analysis[
                "unclipped_aggregate_dark_subtracted_singles_cps"
            ],
            "singles_zero_clipping": analysis["singles_zero_clipping"],
        },
        "cross_check": cross_check,
    }


def _tex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _tex_rate(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return r"\text{N/A}"
    return f"{float(value):.{decimals}f}"


def _tex_value_with_uncertainty(
    value: float,
    uncertainty: float,
    decimals: int = 2,
) -> str:
    value_f = float(value)
    if round(value_f, decimals) == 0.0:
        value_f = 0.0
    return (
        f"{value_f:.{decimals}f}"
        r"\ensuremath{\pm}"
        f"{float(uncertainty):.{decimals}f}"
    )


def _tex_measurement(
    measurement: Dict[str, Any],
    decimals: int = 1,
) -> str:
    return _tex_value_with_uncertainty(
        measurement["rate_cps"],
        measurement["poisson_sigma_cps"],
        decimals,
    )


def _tex_table_header(runs: Sequence[Dict[str, Any]]) -> List[str]:
    groups: List[Tuple[str, int]] = []
    for run in runs:
        group_label = (
            f"{str(run['run_name']).replace('-', '--').capitalize()} run"
        )
        if groups and groups[-1][0] == group_label:
            groups[-1] = (group_label, groups[-1][1] + 1)
        else:
            groups.append((group_label, 1))

    group_cells = [
        rf"\multicolumn{{{span}}}{{c}}{{{_tex_escape(label)}}}"
        for label, span in groups
    ]

    rules: List[str] = []
    start_column = 2
    for _, span in groups:
        end_column = start_column + span - 1
        rules.append(rf"\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column = end_column + 1

    conditions = [
        _tex_escape(run["condition"])
        for run in runs
    ]
    return [
        "Quantity & " + " & ".join(group_cells) + r" \\",
        "".join(rules),
        " & " + " & ".join(conditions) + r" \\",
    ]


def _render_dark_rates_tex(runs: Sequence[Dict[str, Any]]) -> str:
    rows = [
        (
            r"$d_{S_1}$",
            [
                _tex_measurement(run["dark_run_background_measurements"]["S1"])
                for run in runs
            ],
        ),
        (
            r"$d_{S_2}$",
            [
                _tex_measurement(run["dark_run_background_measurements"]["S2"])
                for run in runs
            ],
        ),
        (
            r"$d_I$",
            [
                _tex_measurement(run["dark_run_background_measurements"]["I"])
                for run in runs
            ],
        ),
        (
            r"$d_{C_1}$",
            [
                _tex_measurement(
                    run["dark_run_background_measurements"]["C1"],
                    2,
                )
                for run in runs
            ],
        ),
        (
            r"$d_{C_2}$",
            [
                _tex_measurement(
                    run["dark_run_background_measurements"]["C2"],
                    2,
                )
                for run in runs
            ],
        ),
        (
            r"$b_{C_1}$",
            [
                _tex_value_with_uncertainty(
                    run["dark_run_background_measurements"][
                        "C1_residual_dark_coincidence_background_cps"
                    ],
                    run["dark_run_background_measurements"][
                        "C1_residual_dark_coincidence_background_sigma_cps"
                    ],
                )
                for run in runs
            ],
        ),
        (
            r"$b_{C_2}$",
            [
                _tex_value_with_uncertainty(
                    run["dark_run_background_measurements"][
                        "C2_residual_dark_coincidence_background_cps"
                    ],
                    run["dark_run_background_measurements"][
                        "C2_residual_dark_coincidence_background_sigma_cps"
                    ],
                )
                for run in runs
            ],
        ),
    ]

    lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        *_tex_table_header(runs),
        r"\midrule",
    ]
    for label, values in rows:
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def _render_accidental_corrections_tex(
    runs: Sequence[Dict[str, Any]],
) -> str:
    rows: List[Tuple[str, List[str]]] = []

    for label, key in (
        (r"$R_{S_1}^{\mathrm{raw}}$", "S1"),
        (r"$R_{S_2}^{\mathrm{raw}}$", "S2"),
        (r"$R_I^{\mathrm{raw}}$", "I"),
    ):
        rows.append(
            (
                label,
                [
                    _tex_rate(run["rate_budget"]["raw_singles_cps"][key])
                    for run in runs
                ],
            )
        )

    rows.append(
        (
            r"$R_C^{\mathrm{raw}}$",
            [
                _tex_rate(
                    run["rate_budget"]["raw_coincidence_rates_cps"]["total"],
                    2,
                )
                for run in runs
            ],
        )
    )

    for label, key in (
        (r"$R_{C_1}^{\mathrm{corr}}$", "C1"),
        (r"$R_{C_2}^{\mathrm{corr}}$", "C2"),
        (r"$R_C^{\mathrm{corr}}$", "total"),
    ):
        rows.append(
            (
                label,
                [
                    _tex_rate(
                        run["rate_budget"]["corrected_coincidence_rates_cps"][key],
                        2,
                    )
                    for run in runs
                ],
            )
        )

    rows.append(
        (
            r"$R_{\mathrm{acc}}$",
            [
                _tex_rate(run["rate_budget"]["accidental_rates_cps"]["total"], 2)
                for run in runs
            ],
        )
    )
    rows.append(
        (
            r"$f_{\mathrm{acc}}$ (\%)",
            [
                _tex_rate(
                    (
                        None
                        if run["rate_budget"]["accidental_fraction_of_raw_total"]
                        is None
                        else 100.0
                        * float(
                            run["rate_budget"][
                                "accidental_fraction_of_raw_total"
                            ]
                        )
                    ),
                    2,
                )
                for run in runs
            ],
        )
    )

    lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        *_tex_table_header(runs),
        r"\midrule",
    ]
    for label, values in rows:
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Report dark-run background rates, rate budgets, and "
            "accidental-coincidence corrections for the four MZI runs."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Configured dataset id. Default: the first configured dataset.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(default_repo_root),
        help="Repository root. Default: inferred from this script.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. Default: the run-1 erase directory, alongside the "
            "other MZI reports."
        ),
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON_FILENAME,
        help=f"JSON output filename. Default: {DEFAULT_JSON_FILENAME}.",
    )
    parser.add_argument(
        "--dark-tex",
        default=DEFAULT_DARK_TEX_FILENAME,
        help=(
            "Dark-run-background LaTeX output filename. "
            f"Default: {DEFAULT_DARK_TEX_FILENAME}."
        ),
    )
    parser.add_argument(
        "--accidental-tex",
        default=DEFAULT_ACCIDENTAL_TEX_FILENAME,
        help=(
            "Accidental-correction LaTeX output filename. "
            f"Default: {DEFAULT_ACCIDENTAL_TEX_FILENAME}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        raise ValueError(f"Repository root is not a directory: {repo_root}")

    run_specs = run_specs_from_config(args.dataset_id)
    runs = [_analyze_run(repo_root, spec) for spec in run_specs]
    payload = {
        "schema_version": 5,
        "terminology": {
            "d_X": (
                "measured in-apparatus dark-run background rate for channel X, "
                "not the intrinsic detector dark-count rate"
            ),
            "dark_run_background_components": (
                "intrinsic detector dark counts, optical or ambient background, "
                "and electronic background"
            ),
        },
        "coincidence_window": {
            "width_s": COINCIDENCE_WINDOW_S,
            "width_ns": COINCIDENCE_WINDOW_S * 1e9,
            "convention": "full effective coincidence-window width",
            "input": {
                "file": "dark.json",
                "field": "w_s",
                "location": "top-level",
                "unit": "s",
            },
            "validation": {
                "required_for_every_run": True,
                "matched_canonical_width": True,
            },
        },
        "channel_mapping": CHANNEL_MAPPING,
        "correction_model": {
            "coincidence_window": (
                "w is the full effective coincidence-window width"
            ),
            "accidental_counts": (
                r"N_{\mathrm{acc},j} = (w / \Delta t) N_I N_{S_j}"
            ),
            "residual_dark_coincidence_background": (
                r"b_{C_j} = d_{C_j} - w d_I d_{S_j}"
            ),
            "corrected_coincidence_rate": (
                r"R_{C_j}^{\mathrm{corr}} = "
                r"(N_{C_j} - N_{\mathrm{acc},j}) / \Delta t - b_{C_j}"
            ),
            "corrected_singles_rate": (
                r"R_{S_j}^{\mathrm{corr}} = "
                r"max(N_{S_j} / \Delta t - d_{S_j}, 0)"
            ),
            "aggregation": "pointwise corrected quantities weighted by exposure",
        },
        "runs": runs,
    }

    if args.outdir:
        outdir = Path(args.outdir).expanduser()
        if not outdir.is_absolute():
            outdir = repo_root / outdir
        outdir = outdir.resolve()
    else:
        outdir = (repo_root / run_specs[0].directory).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / args.json
    dark_tex_path = outdir / args.dark_tex
    accidental_tex_path = outdir / args.accidental_tex

    _atomic_write_text(
        json_path,
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write_text(dark_tex_path, _render_dark_rates_tex(runs))
    _atomic_write_text(
        accidental_tex_path,
        _render_accidental_corrections_tex(runs),
    )

    print(str(json_path))
    print(str(dark_tex_path))
    print(str(accidental_tex_path))


if __name__ == "__main__":
    main()
