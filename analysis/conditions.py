#!/usr/bin/env python3
"""Replay archived observing conditions and calculate transmission records."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import asdict as _asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple
from zoneinfo import ZoneInfo

from .goes import (
    GoesSourceFiles,
    GoesSummary,
    analyze_goes,
    render_report_and_payload,
)
from .run_libradtran_transmission import AtmosInputs as LRInputs
from .run_libradtran_transmission import estimate_transmission as lr_estimate
from .run_libradtran_transmission import (
    DEFAULT_LAMBDA_REF_NM,
    find_libradtran_data,
    find_uvspec,
    format_text_report,
)

LOCAL_TZ = ZoneInfo("America/Los_Angeles")

AERONET_SITE = "UCSB"

AERONET_VARIANTS = (
    {
        "key": "solar",
        "label": "AERONET",
        "site": AERONET_SITE,
        "lunar_merge": False,
        "csv_suffix": f"AERONET_{AERONET_SITE}",
    },
    {
        "key": "lunar",
        "label": "AERONET Lunar AOD",
        "site": AERONET_SITE,
        "lunar_merge": True,
        "csv_suffix": f"AERONET_LUNAR_{AERONET_SITE}",
    },
)

def _aeronet_metrics_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "record_utc": metrics["dt_utc"].isoformat(),
        "data_level": metrics["data_level"],
        "tau675": metrics["tau675"],
        "tau870": metrics["tau870"],
        "angstrom_alpha": metrics["alpha"],
        "angstrom_source": metrics["angstrom_source"],
        "pwv_cm": metrics["pwv_cm"],
        "pwv_mm": metrics["pwv_mm"],
    }

def _parse_target_dirname(dirname: str) -> datetime:
    name = Path(dirname).name.strip()
    dt = datetime.strptime(name, "%Y-%m-%d-%H-%M-%S")
    return dt.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)

def _extract_json_payload(text: str) -> str:
    t = text.lstrip()
    if t.startswith("[") or t.startswith("{"):
        return text.strip()

    m = re.search(r"<pre>\s*(\[\s*.*\s*\])\s*</pre>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("Could not find JSON payload in response.")
    return m.group(1).strip()

def _parse_dt_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_metar_list(payload_json: str) -> List[Dict[str, Any]]:
    data = json.loads(payload_json)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list from METAR endpoint.")
    return data

def _parse_metar_observation_time(value: Any) -> Optional[datetime]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        timestamp = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

def _select_closest_metar(metars: List[Dict[str, Any]], target_utc: datetime) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    best_abs_s: Optional[float] = None

    for m in metars:
        if not isinstance(m, dict):
            continue
        dt = _parse_metar_observation_time(m.get("obsTime"))
        if dt is None:
            continue
        abs_s = abs((dt - target_utc).total_seconds())
        if best_abs_s is None or abs_s < best_abs_s:
            best = m
            best_abs_s = abs_s

    if best is not None:
        return best
    raise ValueError("No METAR reports have a valid observation time.")

def _load_saved_metar_for_target(
    dir_path: Path,
    pull_prefix: str,
    target_utc: datetime,
) -> Tuple[Dict[str, Any], Path]:
    path = dir_path / f"{pull_prefix}-SBA.json"
    txt = path.read_text(encoding="utf-8")
    payload = _extract_json_payload(txt)
    metars = _parse_metar_list(payload)
    if not metars:
        raise ValueError(f"No METAR reports found in saved file: {path}")
    return _select_closest_metar(metars, target_utc), path

def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def _hpa_to_inhg(hpa: float) -> float:
    return hpa / 33.8638866667

def _rh_from_temp_dew_c(temp_c: Any, dew_c: Any) -> Optional[float]:
    """
    Estimate relative humidity (fraction 0..1) from temperature and dew point in °C
    using the Magnus formula. Returns None if inputs are invalid.
    """
    try:
        T = float(temp_c)
        Td = float(dew_c)
    except Exception:
        return None
    try:
        es = 6.112 * math.exp(17.67 * T / (T + 243.5))
        e = 6.112 * math.exp(17.67 * Td / (Td + 243.5))
        if es <= 0:
            return None
        rh = e / es
    except Exception:
        return None
    return max(0.0, min(1.0, float(rh)))


def _required_metar_number(
    observation: Dict[str, Any], field: str, description: str
) -> float:
    value = observation.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"selected METAR is missing finite numeric {description} ({field})")
    result = float(value)
    if field == "altim" and not 800.0 <= result <= 1100.0:
        raise ValueError(f"selected METAR pressure {result:g} hPa is outside 800--1100 hPa")
    if field == "temp" and result <= -273.15:
        raise ValueError("selected METAR temperature must be above absolute zero")
    return result

def _render_cloud_layers(clouds: Any) -> str:
    if not clouds:
        return "none"
    if not isinstance(clouds, list):
        return "none"

    parts: List[str] = []
    for layer in clouds:
        if not isinstance(layer, dict):
            continue
        cov = layer.get("cover")
        base = layer.get("base")

        if cov is None and base is None:
            continue

        if base is not None:
            try:
                base_ft = int(base)
            except (TypeError, ValueError):
                base_ft = base
            if cov:
                parts.append(f"{cov} {base_ft} ft")
            else:
                parts.append(f"{base_ft} ft")
        else:
            parts.append(str(cov))

    return ", ".join(parts) if parts else "none"

def _render_report(m: Dict[str, Any], target_utc: datetime) -> str:
    icao = m.get("icaoId", "UNKNOWN")
    name = m.get("name", "")
    report_time = m.get("reportTime") or m.get("receiptTime") or ""
    raw = m.get("rawOb", "")

    temp_c = m.get("temp")
    dewp_c = m.get("dewp")
    wdir = m.get("wdir")
    wspd = m.get("wspd")
    visib = m.get("visib")
    altim_hpa = m.get("altim")
    cover = m.get("cover")
    clouds = m.get("clouds")
    flt_cat = m.get("fltCat")

    if isinstance(altim_hpa, (int, float)):
        assert (
            800.0 <= float(altim_hpa) <= 1100.0
        ), f"METAR altim expected hPa in [800, 1100], got {altim_hpa!r} for station {icao}"

    lines: List[str] = []
    lines.append(f"Station: {icao}" + (f" ({name})" if name else ""))
    if report_time:
        lines.append(f"Report time (UTC): {report_time}")
        dt_utc = _parse_dt_utc(report_time)
        if dt_utc is not None:
            pt = ZoneInfo("America/Los_Angeles")
            lines.append(
                "Report time (PT): " + dt_utc.astimezone(pt).strftime("%Y-%m-%d %H:%M:%S %Z")
            )

            delta_h = (dt_utc - target_utc).total_seconds() / 3600.0
            lines.append(f"Delta from target: {delta_h:+.2f} hours")

    lines.append(f"Cloud cover: {cover}")
    lines.append(f"Clouds: {_render_cloud_layers(clouds)}")
    lines.append(f"Visibility: {visib} sm")

    if isinstance(temp_c, (int, float)):
        lines.append(f"Temperature: {float(temp_c):.1f} °C ({_c_to_f(float(temp_c)):.1f} °F)")
    elif temp_c is not None:
        lines.append(f"Temperature: {temp_c} °C")

    if isinstance(dewp_c, (int, float)):
        lines.append(f"Dew point: {float(dewp_c):.1f} °C ({_c_to_f(float(dewp_c)):.1f} °F)")
    elif dewp_c is not None:
        lines.append(f"Dew point: {dewp_c} °C")

    if isinstance(temp_c, (int, float)) and isinstance(dewp_c, (int, float)):
        spread_c = float(temp_c) - float(dewp_c)
        lines.append(f"Temp/dew spread: {spread_c:.1f} °C")

    if isinstance(wspd, (int, float)) and float(wspd) == 0.0:
        lines.append("Wind: calm")
    else:
        if wdir is not None and wspd is not None:
            try:
                wdir_fmt = f"{int(wdir):03d}"
            except (TypeError, ValueError):
                wdir_fmt = str(wdir)
            lines.append(f"Wind: {wdir_fmt}° at {wspd} kt")
        elif wspd is not None:
            lines.append(f"Wind: {wspd} kt")

    if isinstance(altim_hpa, (int, float)):
        lines.append(
            f"Altimeter: {float(altim_hpa):.1f} hPa ({_hpa_to_inhg(float(altim_hpa)):.2f} inHg)"
        )

    if flt_cat is not None:
        lines.append(f"Flight category: {flt_cat}")

    if raw:
        lines.append(f"Raw METAR: {raw}")

    return "\n".join(lines) + "\n"

def _irsa_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: result[key]
        for key in (
            "filter_name",
            "matched_wavelength_um",
            "matched_index",
            "extinction_column",
            "extinction_mag",
            "transmission_fraction",
            "transmission_percent",
            "table_file",
        )
    }

    if "csv_file" in result:
        out["csv_file"] = result["csv_file"]

    return out

def _irsa_sample_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "time_utc": sample["time_utc"].isoformat(),
        "ra_deg": sample["ra_deg"],
        "dec_deg": sample["dec_deg"],
        "T": sample["T"],
    }
    out.update(_irsa_result_payload(sample["result"]))

    return out


_IRSA_REL_TOL = 1e-12
_IRSA_ABS_TOL = 1e-12


def _irsa_fail(path: Path, field: str, detail: str) -> NoReturn:
    raise ValueError(f"Invalid saved IRSA payload {path}: {field}: {detail}")


def _irsa_required(payload: Dict[str, Any], key: str, path: Path, field: str) -> Any:
    if key not in payload:
        _irsa_fail(path, f"{field}.{key}", "missing required field")
    return payload[key]


def _irsa_dict(value: Any, path: Path, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        _irsa_fail(path, field, "must be an object")
    return value


def _irsa_bool(value: Any, path: Path, field: str) -> bool:
    if not isinstance(value, bool):
        _irsa_fail(path, field, "must be a boolean")
    return value


def _irsa_int(value: Any, path: Path, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _irsa_fail(path, field, "must be an integer")
    if value < minimum:
        _irsa_fail(path, field, f"must be at least {minimum}")
    return value


def _irsa_number(
    value: Any,
    path: Path,
    field: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _irsa_fail(path, field, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _irsa_fail(path, field, "must be a finite number")
    if minimum is not None and number < minimum:
        _irsa_fail(path, field, f"must be at least {minimum}")
    if maximum is not None and number > maximum:
        _irsa_fail(path, field, f"must be at most {maximum}")
    return number


def _irsa_string(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _irsa_fail(path, field, "must be a non-empty string")
    return value


def _irsa_datetime(value: Any, path: Path, field: str) -> datetime:
    raw = _irsa_string(value, path, field)
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        _irsa_fail(path, field, "must be a valid ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _irsa_fail(path, field, "must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _irsa_close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=_IRSA_REL_TOL,
        abs_tol=_IRSA_ABS_TOL,
    )


def _irsa_verify_table(result: Dict[str, Any], path: Path, field: str) -> None:
    """Verify the nearest 810 nm SandF band against the archived ECSV table."""
    table_file = result["table_file"]
    if Path(table_file).name != table_file or Path(table_file).suffix != ".ecsv":
        _irsa_fail(path, f"{field}.table_file", "must name a local ECSV file")
    table_path = path.parent / table_file
    columns = [
        "Filter_name", "LamEff", "A_over_E_B_V_SandF", "A_SandF",
        "A_over_E_B_V_SFD", "A_SFD",
    ]
    try:
        lines = table_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "# %ECSV 1.0":
            raise ValueError("expected an ECSV 1.0 table")
        rows = list(csv.reader(
            (line for line in lines if line.strip() and not line.startswith("#")),
            delimiter=" ", skipinitialspace=True, strict=True,
        ))
        if not rows or rows[0] != columns or len(rows) < 2:
            raise ValueError("expected the six IRSA columns and at least one data row")
        bands = []
        for index, row in enumerate(rows[1:]):
            if len(row) != len(columns) or not row[0].strip():
                raise ValueError(f"invalid band row {index}")
            values = [float(value) for value in row[1:]]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"nonfinite or negative value in band row {index}")
            if values[0] == 0.0:
                raise ValueError(f"nonpositive wavelength in band row {index}")
            bands.append((row[0], *values))
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        _irsa_fail(path, f"{field}.table_file", f"invalid table {table_path} ({exc})")

    matched_index = min(
        range(len(bands)),
        key=lambda index: abs(bands[index][1] - DEFAULT_LAMBDA_REF_NM / 1000.0),
    )
    band = bands[matched_index]
    for key, expected in (
        ("matched_index", matched_index),
        ("filter_name", band[0]),
        ("extinction_column", "A_SandF"),
    ):
        if result[key] != expected:
            _irsa_fail(path, f"{field}.{key}", f"does not agree with {table_path}")
    for key, expected in (
        ("matched_wavelength_um", band[1]),
        ("extinction_mag", band[3]),
        ("transmission_fraction", 10.0 ** (-0.4 * band[3])),
    ):
        if not _irsa_close(result[key], expected):
            _irsa_fail(path, f"{field}.{key}", f"does not agree with {table_path}")


def _irsa_result_from_payload(
    payload: Dict[str, Any],
    *,
    path: Path,
    field: str,
    require_csv: bool = False,
) -> Dict[str, Any]:
    filter_name = _irsa_string(
        _irsa_required(payload, "filter_name", path, field),
        path,
        f"{field}.filter_name",
    )
    matched_wavelength_um = _irsa_number(
        _irsa_required(payload, "matched_wavelength_um", path, field),
        path,
        f"{field}.matched_wavelength_um",
        minimum=0.0,
    )
    if matched_wavelength_um == 0.0:
        _irsa_fail(path, f"{field}.matched_wavelength_um", "must be positive")
    matched_index = _irsa_int(
        _irsa_required(payload, "matched_index", path, field),
        path,
        f"{field}.matched_index",
    )
    extinction_column = _irsa_string(
        _irsa_required(payload, "extinction_column", path, field),
        path,
        f"{field}.extinction_column",
    )
    extinction_mag = _irsa_number(
        _irsa_required(payload, "extinction_mag", path, field),
        path,
        f"{field}.extinction_mag",
    )
    transmission_fraction = _irsa_number(
        _irsa_required(payload, "transmission_fraction", path, field),
        path,
        f"{field}.transmission_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    transmission_percent = _irsa_number(
        _irsa_required(payload, "transmission_percent", path, field),
        path,
        f"{field}.transmission_percent",
        minimum=0.0,
        maximum=100.0,
    )
    if not _irsa_close(transmission_percent, 100.0 * transmission_fraction):
        _irsa_fail(
            path,
            f"{field}.transmission_percent",
            "does not agree with transmission_fraction",
        )

    table_file = _irsa_string(
        _irsa_required(payload, "table_file", path, field),
        path,
        f"{field}.table_file",
    )
    result: Dict[str, Any] = {
        "filter_name": filter_name,
        "matched_wavelength_um": matched_wavelength_um,
        "matched_index": matched_index,
        "extinction_column": extinction_column,
        "extinction_mag": extinction_mag,
        "transmission_fraction": transmission_fraction,
        "transmission_percent": transmission_percent,
        "table_file": table_file,
    }

    if require_csv:
        result["csv_file"] = _irsa_string(
            _irsa_required(payload, "csv_file", path, field),
            path,
            f"{field}.csv_file",
        )
    elif "csv_file" in payload:
        result["csv_file"] = _irsa_string(
            payload["csv_file"], path, f"{field}.csv_file"
        )

    return result


def _irsa_sample_from_payload(
    payload: Dict[str, Any],
    *,
    path: Path,
    field: str,
) -> Dict[str, Any]:
    t = _irsa_datetime(
        _irsa_required(payload, "time_utc", path, field),
        path,
        f"{field}.time_utc",
    )
    ra_deg = _irsa_number(
        _irsa_required(payload, "ra_deg", path, field),
        path,
        f"{field}.ra_deg",
        minimum=0.0,
        maximum=360.0,
    )
    if ra_deg == 360.0:
        _irsa_fail(path, f"{field}.ra_deg", "must be less than 360")
    dec_deg = _irsa_number(
        _irsa_required(payload, "dec_deg", path, field),
        path,
        f"{field}.dec_deg",
        minimum=-90.0,
        maximum=90.0,
    )
    transmission = _irsa_number(
        _irsa_required(payload, "T", path, field),
        path,
        f"{field}.T",
        minimum=0.0,
        maximum=1.0,
    )
    result = _irsa_result_from_payload(payload, path=path, field=field)
    if not _irsa_close(transmission, float(result["transmission_fraction"])):
        _irsa_fail(
            path,
            f"{field}.T",
            "does not agree with transmission_fraction",
        )

    sample: Dict[str, Any] = {
        "time_utc": t,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "T": transmission,
        "result": result,
        "table_file": result["table_file"],
    }
    if "csv_file" in result:
        sample["csv_file"] = result["csv_file"]
    return sample


def _irsa_expected_sample_times(
    start_utc: datetime,
    end_utc: datetime,
    cadence_minutes: int,
) -> List[datetime]:
    times = [start_utc]
    cadence = timedelta(minutes=cadence_minutes)
    while times[-1] + cadence < end_utc:
        times.append(times[-1] + cadence)
    if times[-1] != end_utc:
        times.append(end_utc)
    return times


def _irsa_same_result(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    compare_files: bool,
) -> bool:
    if any(
        actual[key] != expected[key]
        for key in ("filter_name", "matched_index", "extinction_column")
    ):
        return False
    if any(
        not _irsa_close(float(actual[key]), float(expected[key]))
        for key in (
            "matched_wavelength_um",
            "extinction_mag",
            "transmission_fraction",
            "transmission_percent",
        )
    ):
        return False
    if compare_files and any(
        actual.get(key) != expected.get(key) for key in ("table_file", "csv_file")
    ):
        return False
    return True


def _irsa_same_sample(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if actual["time_utc"] != expected["time_utc"]:
        return False
    if any(
        not _irsa_close(float(actual[key]), float(expected[key]))
        for key in ("ra_deg", "dec_deg", "T")
    ):
        return False
    return _irsa_same_result(
        actual["result"], expected["result"], compare_files=True
    )


def _load_saved_irsa_for_target(
    dir_path: Path,
    pull_prefix: str,
    target_utc: datetime,
    integrate_minutes: int,
    integration_step: int,
) -> Dict[str, Any]:
    path = dir_path / f"{pull_prefix}-IRSA_zenith.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _irsa_fail(path, "$", f"could not read valid JSON ({exc})")
    if not isinstance(payload, dict):
        _irsa_fail(path, "$", "must be an object")

    available = _irsa_bool(
        _irsa_required(payload, "available", path, "$"), path, "available"
    )
    if not available:
        _irsa_fail(path, "available", "must be true for archived replay")

    expected_target = target_utc.astimezone(timezone.utc)
    saved_target = _irsa_datetime(
        _irsa_required(payload, "target_utc", path, "$"), path, "target_utc"
    )
    if saved_target != expected_target:
        _irsa_fail(path, "target_utc", "does not match the requested target")

    ra_deg = _irsa_number(
        _irsa_required(payload, "ra_deg", path, "$"),
        path,
        "ra_deg",
        minimum=0.0,
        maximum=360.0,
    )
    if ra_deg == 360.0:
        _irsa_fail(path, "ra_deg", "must be less than 360")
    dec_deg = _irsa_number(
        _irsa_required(payload, "dec_deg", path, "$"),
        path,
        "dec_deg",
        minimum=-90.0,
        maximum=90.0,
    )
    point_result = _irsa_result_from_payload(
        payload, path=path, field="$", require_csv=True
    )
    if isinstance(integrate_minutes, bool) or not isinstance(integrate_minutes, int):
        _irsa_fail(path, "requested integration minutes", "must be an integer")
    if integrate_minutes <= 0:
        _irsa_fail(path, "requested integration minutes", "must be positive")
    if isinstance(integration_step, bool) or not isinstance(integration_step, int):
        _irsa_fail(path, "requested integration step", "must be an integer")
    if integration_step <= 0:
        _irsa_fail(path, "requested integration step", "must be positive")

    saved_integration = _irsa_dict(
        _irsa_required(payload, "integration", path, "$"), path, "integration"
    )
    requested = _irsa_bool(
        _irsa_required(saved_integration, "requested", path, "integration"),
        path,
        "integration.requested",
    )
    if not requested:
        _irsa_fail(path, "integration.requested", "must be true")
    integration_available = _irsa_bool(
        _irsa_required(saved_integration, "available", path, "integration"),
        path,
        "integration.available",
    )
    if not integration_available:
        _irsa_fail(path, "integration.available", "must be true for archived replay")

    saved_minutes = _irsa_int(
        _irsa_required(saved_integration, "minutes", path, "integration"),
        path,
        "integration.minutes",
        minimum=1,
    )
    saved_step = _irsa_int(
        _irsa_required(saved_integration, "cadence_minutes", path, "integration"),
        path,
        "integration.cadence_minutes",
        minimum=1,
    )
    if saved_minutes != integrate_minutes:
        _irsa_fail(
            path,
            "integration.minutes",
            f"does not match requested value {integrate_minutes}",
        )
    if saved_step != integration_step:
        _irsa_fail(
            path,
            "integration.cadence_minutes",
            f"does not match requested value {integration_step}",
        )

    start_utc = _irsa_datetime(
        _irsa_required(saved_integration, "start_utc", path, "integration"),
        path,
        "integration.start_utc",
    )
    end_utc = _irsa_datetime(
        _irsa_required(saved_integration, "end_utc", path, "integration"),
        path,
        "integration.end_utc",
    )
    expected_end = expected_target + timedelta(minutes=integrate_minutes)
    if start_utc != expected_target:
        _irsa_fail(path, "integration.start_utc", "does not match target_utc")
    if end_utc != expected_end:
        _irsa_fail(
            path,
            "integration.end_utc",
            "does not match target_utc plus integration.minutes",
        )

    n_attempted = _irsa_int(
        _irsa_required(saved_integration, "n_attempted", path, "integration"),
        path,
        "integration.n_attempted",
    )
    n_success = _irsa_int(
        _irsa_required(saved_integration, "n_success", path, "integration"),
        path,
        "integration.n_success",
    )
    n_failed = _irsa_int(
        _irsa_required(saved_integration, "n_failed", path, "integration"),
        path,
        "integration.n_failed",
    )

    errors_value = saved_integration.get("errors", [])
    if not isinstance(errors_value, list):
        _irsa_fail(path, "integration.errors", "must be an array when present")
    errors = errors_value
    if n_attempted != n_success + n_failed:
        _irsa_fail(
            path,
            "integration counts",
            "n_attempted must equal n_success plus n_failed",
        )
    if n_failed != len(errors):
        _irsa_fail(path, "integration.n_failed", "does not match errors length")
    if n_failed != 0:
        _irsa_fail(path, "integration.n_failed", "must be zero for archived replay")

    samples_value = _irsa_required(saved_integration, "samples", path, "integration")
    if not isinstance(samples_value, list):
        _irsa_fail(path, "integration.samples", "must be an array")
    samples: List[Dict[str, Any]] = []
    for index, value in enumerate(samples_value):
        sample_field = f"integration.samples[{index}]"
        sample_payload = _irsa_dict(value, path, sample_field)
        samples.append(
            _irsa_sample_from_payload(sample_payload, path=path, field=sample_field)
        )
    if not samples:
        _irsa_fail(path, "integration.samples", "must contain successful samples")
    if n_success != len(samples):
        _irsa_fail(path, "integration.n_success", "does not match samples length")

    expected_times = _irsa_expected_sample_times(start_utc, end_utc, saved_step)
    if n_attempted != len(expected_times):
        _irsa_fail(
            path,
            "integration.n_attempted",
            "does not match the integration window and cadence",
        )
    for index, (sample, expected_time) in enumerate(zip(samples, expected_times, strict=True)):
        if sample["time_utc"] != expected_time:
            _irsa_fail(
                path,
                f"integration.samples[{index}].time_utc",
                "does not match the integration cadence",
            )

    first_sample = samples[0]
    for key, point_value in (("ra_deg", ra_deg), ("dec_deg", dec_deg)):
        if not _irsa_close(float(first_sample[key]), point_value):
            _irsa_fail(
                path,
                f"integration.samples[0].{key}",
                "does not agree with the target point",
            )
    if not _irsa_close(float(first_sample["T"]), float(point_result["transmission_fraction"])):
        _irsa_fail(
            path,
            "integration.samples[0].T",
            "does not agree with the target point transmission",
        )
    if not _irsa_same_result(
        first_sample["result"], point_result, compare_files=False
    ):
        _irsa_fail(
            path,
            "integration.samples[0]",
            "result metadata does not agree with the target point",
        )

    summary_values: Dict[str, float] = {}
    for key in ("min_T", "avg_T", "max_T"):
        summary_values[key] = _irsa_number(
            _irsa_required(saved_integration, key, path, "integration"),
            path,
            f"integration.{key}",
            minimum=0.0,
            maximum=1.0,
        )

    summary_samples: Dict[str, Dict[str, Any]] = {}
    for key in ("min_sample", "avg_sample", "max_sample"):
        summary_field = f"integration.{key}"
        summary_payload = _irsa_dict(
            _irsa_required(saved_integration, key, path, "integration"),
            path,
            summary_field,
        )
        summary_samples[key] = _irsa_sample_from_payload(
            summary_payload, path=path, field=summary_field
        )

    computed_min_sample = min(samples, key=lambda sample: float(sample["T"]))
    computed_max_sample = max(samples, key=lambda sample: float(sample["T"]))
    computed_avg = sum(float(sample["T"]) for sample in samples) / len(samples)
    computed_avg_sample = min(
        samples, key=lambda sample: abs(float(sample["T"]) - computed_avg)
    )
    computed_values = {
        "min_T": float(computed_min_sample["T"]),
        "avg_T": computed_avg,
        "max_T": float(computed_max_sample["T"]),
    }
    computed_samples = {
        "min_sample": computed_min_sample,
        "avg_sample": computed_avg_sample,
        "max_sample": computed_max_sample,
    }
    for key, computed in computed_values.items():
        if not _irsa_close(summary_values[key], computed):
            _irsa_fail(path, f"integration.{key}", "does not agree with samples")
    for key, computed in computed_samples.items():
        if not _irsa_same_sample(summary_samples[key], computed):
            _irsa_fail(
                path,
                f"integration.{key}",
                "does not identify the sample selected by the summary",
            )

    _irsa_verify_table(point_result, path, "$")
    for index, sample in enumerate(samples):
        _irsa_verify_table(sample["result"], path, f"integration.samples[{index}]")

    integration = {
        "minutes": saved_minutes,
        "cadence_minutes": saved_step,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "samples": samples,
        **summary_values,
        **summary_samples,
    }
    mw_T = summary_values["avg_T"]

    return {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "point_result": point_result,
        "integration": integration,
        "mw_T": mw_T,
        "source_path": path,
    }


def _render_irsa_report(
    *,
    target_utc: datetime,
    ra_deg: float,
    dec_deg: float,
    result: Dict[str, Any],
    integration_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    lines = [
        "IRSA Galactic Dust (foreground, integrated through the Galaxy) at zenith (ICRS)"
    ]

    def _fmt_time(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _fmt_sample(sample: Dict[str, Any]) -> str:
        return (
            f"{_fmt_time(sample['time_utc'])},"
            f" RA={float(sample['ra_deg']):.6f}°, Dec={float(sample['dec_deg']):.6f}°"
        )

    lines.append(f"Sky coord: RA={ra_deg:.6f}°, Dec={dec_deg:.6f}° (ICRS)")
    lines.append(
        f"Matched band: {result['filter_name']},"
        f" LamEff≈{result['matched_wavelength_um']:.4f} µm"
    )
    lines.append(f"Extinction column: {result['extinction_column']}")
    lines.append(f"A_lambda (mag): {result['extinction_mag']:.3f}")
    lines.append(
        "Transmission:"
        f" T={result['transmission_fraction']:.4f} ({result['transmission_percent']:.1f}%)"
    )

    minutes = integration_summary["minutes"]
    cadence_minutes = integration_summary["cadence_minutes"]
    start_dt = integration_summary["start_utc"]
    end_dt = integration_summary["end_utc"]
    samples = integration_summary["samples"]
    min_sample = integration_summary["min_sample"]
    avg_sample = integration_summary["avg_sample"]
    max_sample = integration_summary["max_sample"]
    avg_T = integration_summary["avg_T"]
    cadence_text = f"{cadence_minutes} minute" + (
        "" if cadence_minutes == 1 else "s"
    )
    lines.append(
        f"Integrated window: {minutes} minutes, sampled every {cadence_text}"
    )
    lines.append(f"Window start (UTC): {_fmt_time(start_dt)}")
    lines.append(f"Window end (UTC): {_fmt_time(end_dt)}")
    lines.append(f"Samples: {len(samples)}")
    lines.append("Transmission summary:")
    lines.append(
        f"  Min T: {float(min_sample['T']):.4f} at {_fmt_sample(min_sample)}"
    )
    lines.append(
        f"  Avg T: {float(avg_T):.4f}; closest sample at {_fmt_sample(avg_sample)}"
        f" (sample T={float(avg_sample['T']):.4f})"
    )
    lines.append(
        f"  Max T: {float(max_sample['T']):.4f} at {_fmt_sample(max_sample)}"
    )

    payload = {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "target_utc": target_utc.isoformat(),
        **_irsa_result_payload(result),
        "integration": {
            "minutes": minutes,
            "cadence_minutes": cadence_minutes,
            "start_utc": start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "min_T": integration_summary["min_T"],
            "avg_T": avg_T,
            "max_T": integration_summary["max_T"],
            "min_sample": _irsa_sample_payload(min_sample),
            "avg_sample": _irsa_sample_payload(avg_sample),
            "max_sample": _irsa_sample_payload(max_sample),
            "samples": [_irsa_sample_payload(sample) for sample in samples],
        },
    }

    return "\n".join(lines) + "\n", payload

_AERONET_REQUIRED_COLUMNS = (
    "Date(dd:mm:yyyy)",
    "Time(hh:mm:ss)",
    "AOD_675nm",
    "AOD_870nm",
    "440-870_Angstrom_Exponent",
    "Precipitable_Water(cm)",
    "Data_Quality_Level",
)

def _required_aeronet_value(row: Dict[str, Any], column: str) -> str:
    value = str(row.get(column) or "").strip()
    if not value:
        raise ValueError(f"AERONET CSV has a blank required value: {column}")
    return value

def _required_aeronet_float(
    row: Dict[str, Any],
    column: str,
    *,
    minimum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> float:
    text = _required_aeronet_value(row, column)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"AERONET CSV has a nonnumeric required value for {column}: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"AERONET CSV has a nonfinite required value for {column}: {text!r}"
        )
    if minimum is not None and (
        value < minimum or (not minimum_inclusive and value == minimum)
    ):
        relation = "at least" if minimum_inclusive else "greater than"
        raise ValueError(
            f"AERONET CSV requires {column} to be {relation} {minimum}: {text!r}"
        )
    return value

def _parse_aeronet_csv(payload: str) -> List[Dict[str, Any]]:
    lines = [ln for ln in (payload or "").splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if "," in ln and "AOD_" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("AERONET CSV table header not found")

    rdr = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    fieldnames = rdr.fieldnames or []
    missing_columns = [
        column for column in _AERONET_REQUIRED_COLUMNS if column not in fieldnames
    ]
    if missing_columns:
        raise ValueError(
            "AERONET CSV is missing required columns: " + ", ".join(missing_columns)
        )

    out: List[Dict[str, Any]] = []
    for row in rdr:
        dd = _required_aeronet_value(row, "Date(dd:mm:yyyy)")
        tt = _required_aeronet_value(row, "Time(hh:mm:ss)")
        try:
            d_s, m_s, y_s = dd.split(":")
            h_s, mi_s, s_s = tt.split(":")
            dt_utc = datetime(
                int(y_s), int(m_s), int(d_s), int(h_s), int(mi_s), int(s_s), tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError(
                f"AERONET CSV has an invalid required timestamp: {dd!r} {tt!r}"
            ) from exc

        row["_dt_utc"] = dt_utc
        row["_AOD_675"] = _required_aeronet_float(
            row, "AOD_675nm", minimum=0.0, minimum_inclusive=False
        )
        row["_AOD_870"] = _required_aeronet_float(
            row, "AOD_870nm", minimum=0.0, minimum_inclusive=False
        )
        row["_AE_440_870"] = _required_aeronet_float(
            row, "440-870_Angstrom_Exponent"
        )
        row["_PWV_cm"] = _required_aeronet_float(
            row, "Precipitable_Water(cm)", minimum=0.0
        )
        row["_Data_Level"] = _required_aeronet_value(row, "Data_Quality_Level")
        out.append(row)

    if not out:
        raise ValueError("AERONET CSV contains no data rows")
    out.sort(key=lambda r: r["_dt_utc"])
    return out

def _compute_aeronet_metrics(r: Dict[str, Any]) -> Dict[str, Any]:
    tau675 = float(r["_AOD_675"])
    tau870 = float(r["_AOD_870"])
    ae_440_870 = float(r["_AE_440_870"])
    alpha = -math.log(tau675 / tau870) / math.log(675.0 / 870.0)
    pwv_cm = float(r["_PWV_cm"])

    return {
        "alpha": alpha,
        "angstrom_source": "derived from AOD 675/870 nm",
        "tau675": tau675,
        "tau870": tau870,
        "ae_440_870": ae_440_870,
        "dt_utc": r["_dt_utc"],
        "data_level": r["_Data_Level"],
        "pwv_cm": pwv_cm,
        "pwv_mm": pwv_cm * 10.0,
    }

def _select_closest_aeronet_row(
    rows: List[Dict[str, Any]], target_utc: datetime
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_abs_s: Optional[float] = None

    for r in rows:
        dt = r.get("_dt_utc")
        if not isinstance(dt, datetime):
            continue
        abs_s = abs((dt - target_utc).total_seconds())
        if best_abs_s is None or abs_s < best_abs_s:
            best = r
            best_abs_s = abs_s

    return best

def _load_saved_aeronet_variant(
    dir_path: Path,
    pull_prefix: str,
    csv_suffix: str,
    target_utc: datetime,
) -> Dict[str, Any]:
    path = dir_path / f"{pull_prefix}-{csv_suffix}.csv"
    csv_text = path.read_text(encoding="utf-8")
    rows = _parse_aeronet_csv(csv_text)
    row = _select_closest_aeronet_row(rows, target_utc)
    if row is None:
        raise ValueError("AERONET CSV contains no selectable data rows")

    return {
        "row": row,
        "source_path": path,
    }

def _render_aeronet_report(
    label: str,
    site: str,
    metrics: Dict[str, Any],
    target_utc: datetime,
) -> str:
    lines = [f"{label} site: {site}"]
    data_level = str(metrics["data_level"])
    data_level_descriptions = {
        "lev10": "Level 1.0 (lev10; unscreened; final calibration may not be applied)",
        "lev15": "Level 1.5 (lev15; cloud screened and quality controlled)",
        "lev20": "Level 2.0 (lev20; quality assured)",
    }
    lines.append(
        "Data quality level: "
        + data_level_descriptions.get(data_level.lower(), data_level)
    )

    dt = metrics["dt_utc"]
    lines.append(f"Record time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(
        "Record time (PT): "
        + dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    delta_h = (dt.astimezone(timezone.utc) - target_utc).total_seconds() / 3600.0
    lines.append(f"Delta from target: {delta_h:+.2f} hours")
    lines.append(
        f"AOD: 675nm={float(metrics['tau675']):.3f},"
        f" 870nm={float(metrics['tau870']):.3f}"
    )
    lines.append(
        f"Angstrom exponent 440–870: {float(metrics['ae_440_870']):.3f}"
    )
    lines.append(f"PWV: {float(metrics['pwv_mm']):.2f} mm")

    lines.append(
        f"Angstrom parameterization: {metrics['angstrom_source']}; "
        f"alpha={float(metrics['alpha']):.3f}"
    )
    return "\n".join(lines) + "\n"

def _goes_summary_to_lr_inputs(
    summary: GoesSummary,
    pressure_hpa: float,
    metar_sky: str,
    metar_temp_c: float,
    metar_rh_pct: Optional[float],
) -> LRInputs:
    """Build libRadtran inputs from a complete, usable GOES summary."""
    def usable_number(measurement: Any, description: str) -> float:
        if measurement.usable is not True:
            raise ValueError(f"GOES {description} is unusable")
        result = measurement.value
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(result)
        ):
            raise ValueError(
                f"GOES {description} is marked usable but is not finite numeric"
            )
        return float(result)

    aod = usable_number(summary.aod, "AOD")
    alpha = usable_number(summary.ae1, "Angstrom exponent")
    tpw = usable_number(summary.tpw, "TPW")

    cod_val = summary.cod.value
    cod_usable = summary.cod.usable
    if cod_usable not in (True, False):
        raise ValueError("GOES COD usability flag is missing")

    # GOES cloud optical depth applies to cloudy pixels.  In this
    # workflow, we want an unusable COD retrieval. This is the GOES
    # indication that there are no clouds.  We only adopt clear sky
    # when the independent METAR report is *also* positively
    # confirming clear sky with CLR/SKC and when confirmed by direct
    # visual observation.
    #
    # See decide_clear_sky().
    cod_status = "invalid"
    if cod_usable:
        if (
            isinstance(cod_val, bool)
            or not isinstance(cod_val, (int, float))
            or not math.isfinite(cod_val)
        ):
            raise ValueError("GOES COD is marked usable but is not finite numeric")
        cod_status = "valid"

    return LRInputs(
        angstrom_alpha=alpha,
        pressure_hpa=pressure_hpa,
        aod_ref=aod,
        aod_ref_wavelength_um=0.55,
        tpw_mm=tpw,
        metar_sky=metar_sky,
        cod_status=cod_status,
        metar_temp_c=metar_temp_c,
        metar_rh_pct=metar_rh_pct,
        source_label=f"GOES-{summary.satellite}",
    )

def _aeronet_metrics_to_lr_inputs(
    metrics: Dict[str, Any],
    pressure_hpa: float,
    metar_sky: str,
    metar_temp_c: float,
    metar_rh_pct: Optional[float],
    source_label: str,
) -> LRInputs:
    """Build libRadtran AtmosInputs from AERONET metrics + METAR data.

    Passes the measured 870 nm AOD directly to libRadtran's Ångström
    parameterisation.
    """
    aod_ref = metrics.get("tau870")
    if not isinstance(aod_ref, (int, float)) or not math.isfinite(aod_ref) or aod_ref <= 0:
        raise ValueError("AERONET metrics lack a positive finite 870 nm AOD")

    alpha = metrics.get("alpha")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("AERONET metrics lack a finite Angstrom exponent")

    pwv_mm = metrics.get("pwv_mm")
    if not isinstance(pwv_mm, (int, float)) or not math.isfinite(pwv_mm) or pwv_mm < 0:
        raise ValueError("AERONET metrics lack finite nonnegative precipitable water")

    return LRInputs(
        angstrom_alpha=float(alpha),
        pressure_hpa=pressure_hpa,
        aod_ref=float(aod_ref),
        aod_ref_wavelength_um=0.870,
        tpw_mm=float(pwv_mm),
        metar_sky=metar_sky,
        cod_status="invalid",
        metar_temp_c=metar_temp_c,
        metar_rh_pct=metar_rh_pct,
        source_label=source_label,
    )

def replay_archived(
    *,
    target_dir: str,
    input_dir: Path,
    output_dir: Path,
    integration_minutes: int,
    metar_pull_prefix: str,
    aeronet_variant: str,
    aeronet_pull_prefix: str,
    irsa_pull_prefix: str,
    atmosphere_source: str,
    goes_pull_prefix: Optional[str],
    goes_sources: Optional[GoesSourceFiles],
    integration_step: int = 5,
) -> None:
    """Replay one observing-conditions record from staged archived inputs."""
    if aeronet_variant not in {"solar", "lunar"}:
        raise ValueError("aeronet_variant must be 'solar' or 'lunar'")
    if atmosphere_source not in {"goes", "aeronet"}:
        raise ValueError("atmosphere_source must be 'goes' or 'aeronet'")
    if atmosphere_source == "goes" and (
        not goes_pull_prefix or goes_sources is None
    ):
        raise ValueError("GOES atmosphere requires a pull prefix and source files")
    if atmosphere_source == "aeronet" and (
        goes_pull_prefix is not None or goes_sources is not None
    ):
        raise ValueError("AERONET atmosphere must not configure GOES inputs")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_utc = _parse_target_dirname(target_dir)
    target_header = (
        "Target time (UTC): "
        + target_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        + "\nTarget time (PT): "
        + target_utc.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        + "\n\n"
    )

    closest_metar, metar_source_path = _load_saved_metar_for_target(
        input_dir,
        metar_pull_prefix,
        target_utc,
    )
    pressure_hpa = _required_metar_number(
        closest_metar, "altim", "surface pressure"
    )
    lr_temp_c = _required_metar_number(
        closest_metar, "temp", "surface temperature"
    )
    metar_report = _render_report(closest_metar, target_utc)
    loaded_irsa = _load_saved_irsa_for_target(
        input_dir,
        irsa_pull_prefix,
        target_utc,
        integration_minutes,
        integration_step,
    )
    irsa_ra_deg = float(loaded_irsa["ra_deg"])
    irsa_dec_deg = float(loaded_irsa["dec_deg"])
    irsa_point_result = loaded_irsa["point_result"]
    irsa_integration = loaded_irsa["integration"]
    mw_transmission = loaded_irsa["mw_T"]
    irsa_source_path = loaded_irsa["source_path"]
    irsa_report, irsa_payload = _render_irsa_report(
        target_utc=target_utc,
        ra_deg=irsa_ra_deg,
        dec_deg=irsa_dec_deg,
        result=irsa_point_result,
        integration_summary=irsa_integration,
    )
    irsa_payload["source"] = "saved"
    irsa_payload["source_file"] = str(irsa_source_path)
    irsa_payload["source_pull_prefix"] = irsa_pull_prefix

    variant = next(
        item for item in AERONET_VARIANTS if item["key"] == aeronet_variant
    )
    aeronet_source = _load_saved_aeronet_variant(
        input_dir,
        aeronet_pull_prefix,
        str(variant["csv_suffix"]),
        target_utc,
    )
    aeronet_metrics = _compute_aeronet_metrics(aeronet_source["row"])
    aeronet_source_path = aeronet_source["source_path"]
    aeronet_report = _render_aeronet_report(
        str(variant["label"]),
        str(variant["site"]),
        aeronet_metrics,
        target_utc,
    )

    metar_sky = str(closest_metar.get("cover") or "")
    dewp_c = closest_metar.get("dewp")
    rh_frac = _rh_from_temp_dew_c(lr_temp_c, dewp_c)
    lr_rh_pct = float(rh_frac) * 100.0 if isinstance(rh_frac, float) else None
    lr_runs: List[Tuple[str, LRInputs]] = []
    goes_report: Optional[str] = None
    goes_payload: Optional[Dict[str, Any]] = None
    if atmosphere_source == "goes":
        assert goes_sources is not None
        assert goes_pull_prefix is not None
        goes_summary = analyze_goes(goes_sources)
        goes_report, goes_payload = render_report_and_payload(
            goes_summary,
            target_utc=target_utc,
            pressure_hpa=pressure_hpa,
            source_pull_prefix=goes_pull_prefix,
        )
        lr_runs.append(
            (
                f"GOES-{goes_summary.satellite}",
                _goes_summary_to_lr_inputs(
                    goes_summary,
                    pressure_hpa,
                    metar_sky,
                    lr_temp_c,
                    lr_rh_pct,
                ),
            )
        )
    label = "AERONET solar" if aeronet_variant == "solar" else "AERONET lunar"
    lr_runs.append(
        (
            f"AERONET-{aeronet_variant}",
            _aeronet_metrics_to_lr_inputs(
                aeronet_metrics,
                pressure_hpa,
                metar_sky,
                lr_temp_c,
                lr_rh_pct,
                source_label=label,
            ),
        )
    )

    libradtran_reports: List[str] = []
    libradtran_payloads: List[Dict[str, Any]] = []
    for run_tag, lr_inputs in lr_runs:
        lr_result, _ = lr_estimate(lr_inputs)
        libradtran_payloads.append(
            {
                "tag": run_tag,
                "inputs": _asdict(lr_inputs),
                "result": _asdict(lr_result),
            }
        )
        libradtran_reports.append(
            format_text_report(
                inputs=lr_inputs,
                result=lr_result,
                uvspec_exe=find_uvspec(None),
                data_path=find_libradtran_data(None),
                workdir=None,
            )
        )

    report_parts = [
        target_header.rstrip("\n"),
        metar_report.rstrip("\n"),
        aeronet_report.rstrip("\n"),
        irsa_report.rstrip("\n"),
    ]
    if goes_report is not None:
        report_parts.append(goes_report.rstrip("\n"))
    report_parts.extend(report.rstrip("\n") for report in libradtran_reports)
    full_report = "\n\n".join(report_parts) + "\n"
    conditions_summary: Dict[str, Any] = {
        "target": {
            "utc": target_utc.isoformat(),
            "local": target_utc.astimezone(LOCAL_TZ).isoformat(),
            "dirname": target_dir,
        },
        "metar": {
            "station": "KSBA",
            "source": "saved",
            "source_file": str(metar_source_path),
            "source_pull_prefix": metar_pull_prefix,
            "selected_report": closest_metar,
            "pressure_hpa": pressure_hpa,
        },
        "milky_way": {
            "T": mw_transmission,
            "ra_deg": irsa_ra_deg,
            "dec_deg": irsa_dec_deg,
        },
        "irsa": irsa_payload,
        "aeronet": {
            "variant": aeronet_variant,
            "source": "saved",
            "source_file": str(aeronet_source_path),
            "source_pull_prefix": aeronet_pull_prefix,
            **_aeronet_metrics_payload(aeronet_metrics),
        },
    }
    if goes_payload is not None:
        conditions_summary["goes"] = goes_payload
    conditions_summary["libradtran"] = libradtran_payloads
    (output_dir / "conditions.txt").write_text(
        full_report,
        encoding="utf-8",
    )
    (output_dir / "conditions.json").write_text(
        json.dumps(conditions_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

__all__ = ["replay_archived"]
