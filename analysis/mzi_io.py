"""Public, side-effect-free helpers for reading and validating MZI runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dark_analysis import COINCIDENCE_WINDOW_S
from .mzi_analysis import (
    DarkModel,
    _assert_constant_point_duration,
    scan_from_records,
)


COUNT_FIELDS = ("N_s", "N_i", "N_i2", "N_c", "N_c2")


def _required_number(
    value: Any, field: str, context: str, *, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{context}.{field} must be {qualifier}")
    return result


def _required_count(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.{field} must be a non-negative integer")
    return value


def load_dark(run_dir: Path) -> DarkModel:
    path = Path(run_dir) / "dark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    context = str(path)
    required = ("Ns", "Ni", "Ni2", "Nc", "Nc2", "T_s", "w_s")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    window = _required_number(data["w_s"], "w_s", context, positive=True)
    if not np.isclose(window, COINCIDENCE_WINDOW_S, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"{path}.w_s must equal the canonical {COINCIDENCE_WINDOW_S:g} s"
        )
    return DarkModel.from_counts(
        Ns=_required_count(data["Ns"], "Ns", context),
        Ni=_required_count(data["Ni"], "Ni", context),
        Nc=_required_count(data["Nc"], "Nc", context),
        T=_required_number(data["T_s"], "T_s", context, positive=True),
        w=window,
        Ni2=_required_count(data["Ni2"], "Ni2", context),
        Nc2=_required_count(data["Nc2"], "Nc2", context),
    )


def load_plan(run_dir: Path) -> dict[str, Any]:
    value = json.loads((Path(run_dir) / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plan.json must contain an object")
    return value


def load_points(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(run_dir) / "points.jsonl"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        context = f"{path}:{line_number}"
        for field in ("duration_s", "target_delta_V", "actual_delta_V", "counts"):
            if field not in value:
                raise ValueError(f"{context} is missing required field {field}")
        _required_number(value["duration_s"], "duration_s", context, positive=True)
        _required_number(value["target_delta_V"], "target_delta_V", context)
        _required_number(value["actual_delta_V"], "actual_delta_V", context)
        counts = value["counts"]
        if not isinstance(counts, dict):
            raise ValueError(f"{context}.counts must be an object")
        for field in COUNT_FIELDS:
            if field not in counts:
                raise ValueError(f"{context}.counts is missing required field {field}")
            _required_count(counts[field], field, f"{context}.counts")
        records.append(value)
    if not records:
        raise ValueError(f"no point records in {path}")
    return records


def plan_length(plan: dict[str, Any]) -> int:
    values = plan.get("deltas_exec_V")
    if not isinstance(values, list) or not values:
        raise ValueError("plan.deltas_exec_V must be a non-empty list")
    for index, value in enumerate(values):
        _required_number(value, str(index), "plan.deltas_exec_V")
    return len(values)


def load_plan_length(run_dir: Path) -> int:
    return plan_length(load_plan(run_dir))


def assert_constant_point_duration(
    records: list[dict[str, Any]], context: str
) -> None:
    _assert_constant_point_duration(records, context)


def validate_constant_point_duration(records: Iterable[dict[str, Any]]) -> float:
    records = list(records)
    if not records:
        raise ValueError("cannot validate an empty record collection")
    durations = [
        _required_number(item.get("duration_s"), "duration_s", f"point {index}", positive=True)
        for index, item in enumerate(records)
    ]
    if not all(np.isfinite(value) and value > 0 for value in durations):
        raise ValueError("point durations must be finite and positive")
    if not all(np.isclose(value, durations[0], rtol=0.0, atol=1e-12) for value in durations):
        raise ValueError("point duration is not constant")
    return durations[0]


def split_by_plan_length(records: list[dict[str, Any]], length: int) -> list[list[dict[str, Any]]]:
    if length <= 0:
        raise ValueError("plan length must be positive")
    if not records:
        raise ValueError("cannot split an empty point collection")
    if len(records) % length:
        raise ValueError(
            f"point count {len(records)} is not an exact multiple of plan length {length}"
        )
    return [records[index:index + length] for index in range(0, len(records), length)]


def parse_condition_specs(specs: list[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    names: set[str] = set()
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"invalid condition specification: {spec!r}")
        name, raw_path = (part.strip() for part in spec.split(":", 1))
        if not name or name in names or not raw_path:
            raise ValueError(f"invalid or duplicate condition specification: {spec!r}")
        path = Path(raw_path).expanduser().resolve()
        if not all((path / filename).is_file() for filename in ("dark.json", "plan.json", "points.jsonl")):
            raise ValueError(f"condition input is incomplete: {path}")
        names.add(name)
        output.append((name, path))
    if not output:
        raise ValueError("at least one condition is required")
    return output


def validate_mzi_run(run_dir: Path) -> None:
    load_dark(run_dir)
    records = load_points(run_dir)
    validate_constant_point_duration(records)
    split_by_plan_length(records, load_plan_length(run_dir))


__all__ = [
    "assert_constant_point_duration", "load_dark",
    "load_plan", "load_plan_length", "load_points", "parse_condition_specs",
    "plan_length", "scan_from_records",
    "split_by_plan_length", "validate_constant_point_duration", "validate_mzi_run",
]
