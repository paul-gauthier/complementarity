#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .transmission_values import (
    DEFAULT_MILKY_WAY_BASIS,
    MILKY_WAY_BASES,
    MIRROR_TRANSMISSION,
    WINDOW_SURFACE_REFLECTANCE,
    calculate_transmissions,
)

ROOT = Path(__file__).resolve().parent.parent


class ConditionsError(ValueError):
    pass


def value(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConditionsError(f"missing required field: {path}")
        current = current[part]
    return current


def number(data: dict[str, Any], path: str) -> float:
    result = value(data, path)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ConditionsError(f"required field is not numeric: {path}")
    if not math.isfinite(result):
        raise ConditionsError(f"required field is not finite: {path}")
    return float(result)


def text(data: dict[str, Any], path: str) -> str:
    result = value(data, path)
    if not isinstance(result, str):
        raise ConditionsError(f"required field is not text: {path}")
    return result


def require_true(data: dict[str, Any], path: str) -> None:
    if value(data, path) is not True:
        raise ConditionsError(f"{path} must be true")


def require_close(
    actual: float,
    expected: float,
    description: str,
    *,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ConditionsError(
            f"{description} is inconsistent: got {actual!r}, expected {expected!r}"
        )


def parse_time(timestamp: str) -> datetime:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConditionsError(f"invalid timestamp: {timestamp!r}") from exc


def tagged_record(data: dict[str, Any], tag: str) -> dict[str, Any]:
    records = value(data, "libradtran")
    if not isinstance(records, list):
        raise ConditionsError("libradtran must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("tag") == tag
    ]
    if len(matches) != 1:
        raise ConditionsError(
            f"expected exactly one libradtran record tagged {tag!r}; "
            f"got {len(matches)}"
        )
    return matches[0]


def selected_aeronet_record(data: dict[str, Any]) -> dict[str, Any]:
    records = value(data, "libradtran")
    if not isinstance(records, list):
        raise ConditionsError("libradtran must be an array")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("tag") in {"AERONET-solar", "AERONET-lunar"}
    ]
    if len(matches) != 1:
        raise ConditionsError(
            "expected exactly one AERONET libradtran record; "
            f"got {len(matches)}"
        )
    return matches[0]


def require_goes_inputs(data: dict[str, Any]) -> None:
    require_true(data, "goes.aod.usable")
    require_true(data, "goes.aod.alpha_usable")
    require_true(data, "goes.tpw.usable")
    number(data, "goes.aod.value")
    number(data, "goes.aod.alpha")
    number(data, "goes.tpw.value")


def required_libradtran_records(
    data: dict[str, Any], atmosphere_source: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if atmosphere_source not in {"goes", "aeronet"}:
        raise ConditionsError(f"unexpected atmospheric source: {atmosphere_source!r}")

    aeronet_record = selected_aeronet_record(data)
    require_aeronet_variant(data, aeronet_record)
    if atmosphere_source == "goes":
        require_goes_inputs(data)
        goes_record = tagged_record(data, "GOES-18")
        expected_tags = ["GOES-18", str(aeronet_record["tag"])]
    else:
        if "goes" in data:
            raise ConditionsError("AERONET conditions must not contain GOES data")
        goes_record = None
        expected_tags = [str(aeronet_record["tag"])]

    records = value(data, "libradtran")
    if not isinstance(records, list):
        raise ConditionsError("libradtran must be an array")
    actual_tags = [
        record.get("tag") if isinstance(record, dict) else None for record in records
    ]
    if actual_tags != expected_tags:
        raise ConditionsError(
            f"expected libRadtran records {expected_tags!r}; got {actual_tags!r}"
        )
    return goes_record, aeronet_record


def require_aeronet_variant(
    data: dict[str, Any], record: dict[str, Any]
) -> str:
    variant = text(data, "aeronet.variant")
    if variant not in {"solar", "lunar"}:
        raise ConditionsError(f"unexpected AERONET variant: {variant!r}")
    tag = record.get("tag")
    expected_tag = f"AERONET-{variant}"
    if tag != expected_tag:
        raise ConditionsError(
            f"AERONET variant {variant!r} does not match libRadtran tag {tag!r}"
        )
    return variant


def record_number(record: dict[str, Any], path: str, description: str) -> float:
    try:
        result = value(record, path)
    except ConditionsError as exc:
        raise ConditionsError(f"{description}: {exc}") from exc
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ConditionsError(f"required field is not numeric: {description}.{path}")
    if not math.isfinite(result):
        raise ConditionsError(f"required field is not finite: {description}.{path}")
    return float(result)


def record_text(record: dict[str, Any], path: str, description: str) -> str:
    try:
        result = value(record, path)
    except ConditionsError as exc:
        raise ConditionsError(f"{description}: {exc}") from exc
    if not isinstance(result, str):
        raise ConditionsError(f"required field is not text: {description}.{path}")
    return result


def transmissions(
    data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
):
    goes_record, aeronet_record = required_libradtran_records(
        data, atmosphere_source
    )
    atmosphere_record = goes_record if goes_record is not None else aeronet_record
    return calculate_transmissions(
        atmosphere=record_number(
            atmosphere_record,
            "result.t_band",
            f"{atmosphere_record.get('tag', atmosphere_source)} libRadtran",
        ),
        aeronet_atmosphere=record_number(
            aeronet_record, "result.t_band", "AERONET libRadtran"
        ),
        milky_way_point=number(data, "irsa.transmission_fraction"),
        milky_way_integrated=number(data, "irsa.integration.avg_T"),
        milky_way_minimum=number(data, "irsa.integration.min_T"),
        milky_way_basis=milky_way_basis,
    )


def fixed(result: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(result)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def integer(result: float) -> str:
    rounded = Decimal(str(result)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return "0" if rounded == 0 else str(rounded)


def integer_set(data: dict[str, Any], path: str) -> str:
    result = value(data, path)
    if (
        not isinstance(result, list)
        or not result
        or any(isinstance(item, bool) or not isinstance(item, int) for item in result)
        or result != sorted(set(result))
    ):
        raise ConditionsError(
            f"required field is not a sorted set of integers: {path}"
        )
    rendered = ",".join(str(item) for item in result)
    return rendered if len(result) == 1 else rf"\{{{rendered}\}}"


def optional_fixed(data: dict[str, Any], path: str, decimals: int) -> str:
    value = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value or value[part] is None:
            return "unavailable"
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unavailable"
    return fixed(float(value), decimals)


def tex_text(result: str) -> str:
    return (
        result.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def tex_time(timestamp: datetime) -> str:
    return timestamp.strftime("%H{:}%M{:}%S")


def tex_date(timestamp: datetime) -> str:
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    return f"{months[timestamp.month - 1]} {timestamp.day}, {timestamp.year}"


def right_ascension(degrees: float) -> str:
    total_seconds = round(degrees * 240, 1)
    hours = int(total_seconds // 3600) % 24
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}{{:}}{minutes:02d}{{:}}{seconds:04.1f}"


def declination(degrees: float) -> str:
    sign = "+" if degrees >= 0 else "-"
    total_seconds = round(abs(degrees) * 3600, 1)
    whole_degrees = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60
    return f"{sign}{whole_degrees:02d}{{:}}{minutes:02d}{{:}}{seconds:04.1f}"


def validate(data: dict[str, Any], atmosphere_source: str = "goes") -> None:
    target_utc = text(data, "target.utc")
    if not text(data, "target.dirname"):
        raise ConditionsError("target.dirname must be nonempty")
    if text(data, "metar.station") != "KSBA":
        raise ConditionsError("unexpected METAR station")

    goes_record, aeronet_record = required_libradtran_records(
        data, atmosphere_source
    )
    if goes_record is not None:
        if text(data, "goes.satellite") != "GOES-18":
            raise ConditionsError("unexpected GOES satellite")
        if text(data, "goes.target_utc") != target_utc:
            raise ConditionsError("GOES target time does not match target.utc")

    if text(data, "irsa.target_utc") != target_utc:
        raise ConditionsError("IRSA target time does not match target.utc")

    minutes = number(data, "irsa.integration.minutes")
    if minutes != 49:
        raise ConditionsError(f"unexpected launch-window duration: {minutes!r}")
    samples = value(data, "irsa.integration.samples")
    if not isinstance(samples, list) or not samples:
        raise ConditionsError("IRSA integration samples must be a nonempty array")
    integration_start = parse_time(text(data, "irsa.integration.start_utc"))
    integration_end = parse_time(text(data, "irsa.integration.end_utc"))
    target_time = parse_time(text(data, "target.utc"))
    if integration_start != target_time:
        raise ConditionsError("IRSA integration must start at target.utc")
    require_close(
        (integration_end - integration_start).total_seconds(),
        60 * minutes,
        "IRSA integration duration",
    )

    cadence = number(data, "irsa.integration.cadence_minutes")
    if cadence <= 0:
        raise ConditionsError("IRSA integration cadence must be positive")
    expected_offsets = [0.0]
    while expected_offsets[-1] + cadence < minutes:
        expected_offsets.append(expected_offsets[-1] + cadence)
    if expected_offsets[-1] != minutes:
        expected_offsets.append(minutes)
    if len(samples) != len(expected_offsets):
        raise ConditionsError(
            "IRSA integration sample count does not match its window and cadence"
        )

    sample_transmissions = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ConditionsError(f"IRSA integration sample {index} is not an object")
        description = f"IRSA integration sample {index}"
        sample_time = parse_time(record_text(sample, "time_utc", description))
        require_close(
            (sample_time - integration_start).total_seconds(),
            60 * expected_offsets[index],
            f"{description} time",
        )
        sample_transmission = record_number(sample, "T", description)
        sample_extinction = record_number(sample, "extinction_mag", description)
        require_close(
            sample_transmission,
            10 ** (-0.4 * sample_extinction),
            f"{description} extinction transmission",
        )
        sample_transmissions.append(sample_transmission)

    average_transmission = sum(sample_transmissions) / len(sample_transmissions)
    require_close(
        number(data, "irsa.integration.avg_T"),
        average_transmission,
        "IRSA average transmission",
    )
    require_close(
        number(data, "irsa.integration.min_T"),
        min(sample_transmissions),
        "IRSA minimum sample transmission",
    )
    require_close(
        number(data, "irsa.integration.max_T"),
        max(sample_transmissions),
        "IRSA maximum sample transmission",
    )

    extrema = (
        ("min_sample", min(range(len(samples)), key=sample_transmissions.__getitem__)),
        ("max_sample", max(range(len(samples)), key=sample_transmissions.__getitem__)),
        (
            "avg_sample",
            min(
                range(len(samples)),
                key=lambda index: abs(
                    sample_transmissions[index] - average_transmission
                ),
            ),
        ),
    )
    for field, sample_index in extrema:
        selected = value(data, f"irsa.integration.{field}")
        if not isinstance(selected, dict):
            raise ConditionsError(f"irsa.integration.{field} must be an object")
        description = f"IRSA integration {field}"
        require_close(
            record_number(selected, "T", description),
            sample_transmissions[sample_index],
            f"{description}.T",
        )
        if record_text(selected, "time_utc", description) != record_text(
            samples[sample_index],
            "time_utc",
            f"IRSA integration sample {sample_index}",
        ):
            raise ConditionsError(f"{description} identifies the wrong sample")

    require_close(
        record_number(samples[0], "ra_deg", "IRSA integration sample 0"),
        number(data, "irsa.ra_deg"),
        "initial IRSA sample RA",
    )
    require_close(
        record_number(samples[0], "dec_deg", "IRSA integration sample 0"),
        number(data, "irsa.dec_deg"),
        "initial IRSA sample Dec",
    )
    require_close(
        number(data, "milky_way.ra_deg"),
        number(data, "irsa.ra_deg"),
        "Milky Way/IRSA RA",
    )
    require_close(
        number(data, "milky_way.dec_deg"),
        number(data, "irsa.dec_deg"),
        "Milky Way/IRSA Dec",
    )
    require_close(
        number(data, "milky_way.T"),
        number(data, "irsa.integration.avg_T"),
        "Milky Way average transmission",
    )

    extinction = number(data, "irsa.extinction_mag")
    transmission = number(data, "irsa.transmission_fraction")
    require_close(
        transmission,
        10 ** (-0.4 * extinction),
        "IRSA extinction transmission",
    )
    require_aeronet_variant(data, aeronet_record)
    aeronet_path = "aeronet"
    require_close(
        record_number(samples[0], "T", "initial IRSA integration sample"),
        transmission,
        "initial IRSA transmission",
    )
    records_to_validate = [(str(aeronet_record["tag"]), aeronet_record)]
    if goes_record is not None:
        records_to_validate.insert(0, ("GOES-18", goes_record))

    for tag, record in records_to_validate:
        description = f"{tag} libRadtran"
        if record.get("result", {}).get("used_clear_sky") is not True:
            raise ConditionsError(f"{description} must use clear-sky mode")
        if record_text(record, "inputs.metar_sky", description) != text(
            data, "metar.selected_report.cover"
        ):
            raise ConditionsError(f"{description} METAR sky input is stale")
        if record_text(record, "inputs.cod_status", description) != "invalid":
            raise ConditionsError(f"{description} COD status must be invalid")
        require_close(
            record_number(record, "inputs.pressure_hpa", description),
            number(data, "metar.pressure_hpa"),
            f"{description} pressure input",
        )
        require_close(
            record_number(record, "inputs.metar_temp_c", description),
            number(data, "metar.selected_report.temp"),
            f"{description} temperature input",
        )
        for path in ("result.t_band", "result.t_810"):
            result = record_number(record, path, description)
            if not 0 <= result <= 1:
                raise ConditionsError(f"{description}.{path} is outside [0, 1]")

    if goes_record is not None:
        aod = number(data, "goes.aod.value")
        alpha = number(data, "goes.aod.alpha")
        beta = record_number(
            goes_record, "result.beta_angstrom", "GOES-18 libRadtran"
        )
        require_close(beta, aod * 0.55**alpha, "GOES Ångström coefficient")
        require_close(
            record_number(
                goes_record,
                "inputs.aod_ref",
                "GOES-18 libRadtran",
            ),
            aod,
            "GOES libRadtran AOD input",
        )
        require_close(
            record_number(
                goes_record,
                "inputs.aod_ref_wavelength_um",
                "GOES-18 libRadtran",
            ),
            0.55,
            "GOES libRadtran AOD wavelength",
        )
        require_close(
            record_number(
                goes_record,
                "inputs.angstrom_alpha",
                "GOES-18 libRadtran",
            ),
            alpha,
            "GOES libRadtran Ångström exponent",
        )
        require_close(
            record_number(
                goes_record,
                "inputs.tpw_mm",
                "GOES-18 libRadtran",
            ),
            number(data, "goes.tpw.value"),
            "GOES libRadtran TPW input",
        )

    for name, record, source_path in (
        (str(aeronet_record["tag"]), aeronet_record, aeronet_path),
    ):
        description = f"{name} libRadtran"
        tau675 = number(data, f"{source_path}.tau675")
        tau870 = number(data, f"{source_path}.tau870")
        source_alpha = number(data, f"{source_path}.angstrom_alpha")
        require_close(
            source_alpha,
            -math.log(tau675 / tau870) / math.log(0.675 / 0.870),
            f"{name} Ångström exponent",
        )
        require_close(
            record_number(record, "inputs.aod_ref", description),
            tau870,
            f"{name} AOD input",
        )
        require_close(
            record_number(
                record,
                "inputs.aod_ref_wavelength_um",
                description,
            ),
            0.87,
            f"{name} AOD wavelength",
        )
        require_close(
            record_number(record, "inputs.angstrom_alpha", description),
            source_alpha,
            f"{name} Ångström input",
        )
        require_close(
            record_number(record, "inputs.tpw_mm", description),
            number(data, f"{source_path}.pwv_mm"),
            f"{name} TPW input",
        )
        require_close(
            number(data, f"{source_path}.pwv_mm"),
            10 * number(data, f"{source_path}.pwv_cm"),
            f"{name} PWV conversion",
        )
        require_close(
            record_number(record, "result.beta_angstrom", description),
            tau870 * 0.87**source_alpha,
            f"{name} Ångström coefficient",
        )

    atmosphere_record = (
        goes_record if atmosphere_source == "goes" else aeronet_record
    )
    if atmosphere_record is None:
        raise ConditionsError("selected atmospheric libRadtran result is missing")
    atmosphere_description = (
        f"{atmosphere_record.get('tag', atmosphere_source)} libRadtran"
    )
    band_minimum = record_number(
        atmosphere_record,
        "result.lambda_min_nm",
        atmosphere_description,
    )
    band_maximum = record_number(
        atmosphere_record,
        "result.lambda_max_nm",
        atmosphere_description,
    )
    band_reference = record_number(
        atmosphere_record,
        "result.lambda_ref_nm",
        atmosphere_description,
    )
    if band_minimum >= band_maximum:
        raise ConditionsError("libRadtran wavelength band is empty or reversed")
    for tag, record in records_to_validate:
        description = f"{tag} libRadtran"
        require_close(
            record_number(record, "result.lambda_min_nm", description),
            band_minimum,
            f"{description} lower wavelength limit",
        )
        require_close(
            record_number(record, "result.lambda_max_nm", description),
            band_maximum,
            f"{description} upper wavelength limit",
        )
        require_close(
            record_number(record, "result.lambda_ref_nm", description),
            band_reference,
            f"{description} reference wavelength",
        )

    if not 0 <= transmission <= 1:
        raise ConditionsError("Milky Way transmission is outside [0, 1]")


def build_macros(
    data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> dict[str, str]:
    launch_utc = parse_time(text(data, "target.utc"))
    launch_local = parse_time(text(data, "target.local"))
    metar_observation_time = datetime.fromtimestamp(
        number(data, "metar.selected_report.obsTime"),
        tz=timezone.utc,
    )
    aeronet_record = selected_aeronet_record(data)
    require_aeronet_variant(data, aeronet_record)
    aeronet_path = "aeronet"
    aeronet_time = parse_time(text(data, f"{aeronet_path}.record_utc"))

    atmosphere_record = (
        tagged_record(data, "GOES-18")
        if atmosphere_source == "goes"
        else aeronet_record
    )
    atmosphere_description = (
        f"{atmosphere_record.get('tag', atmosphere_source)} libRadtran"
    )
    atmospheric = record_number(
        atmosphere_record,
        "result.t_band",
        atmosphere_description,
    )
    aeronet_atmospheric = record_number(
        aeronet_record, "result.t_band", "AERONET libRadtran"
    )
    milky_way = number(data, "irsa.transmission_fraction")
    extinction = number(data, "irsa.extinction_mag")
    transmission = transmissions(data, atmosphere_source, milky_way_basis)
    band_minimum = record_number(
        atmosphere_record,
        "result.lambda_min_nm",
        atmosphere_description,
    )
    band_maximum = record_number(
        atmosphere_record,
        "result.lambda_max_nm",
        atmosphere_description,
    )

    goes_macros: dict[str, str] = {}
    if atmosphere_source == "goes":
        goes_time = parse_time(text(data, "goes.aod.start_utc"))
        goes_macros = {
            "ConditionGoesSatellite": tex_text(text(data, "goes.satellite")).replace(
                "-", "{-}"
            ),
            "ConditionGoesFileStartUTC": tex_time(goes_time),
            "ConditionGoesLeadSeconds": integer(
                (launch_utc - goes_time).total_seconds()
            ),
            "ConditionGoesTPWDQF": integer(number(data, "goes.tpw.dqf")),
            "ConditionGoesAODDQF": integer(number(data, "goes.aod.dqf")),
            "ConditionGoesCODDQF": integer_set(data, "goes.cod.dqf_values"),
            "ConditionGoesTPW": fixed(number(data, "goes.tpw.value"), 3),
            "ConditionGoesAOD": optional_fixed(data, "goes.aod.value", 5),
            "ConditionGoesAngstromExponent": optional_fixed(
                data, "goes.aod.alpha", 5
            ),
            "ConditionGoesAngstromCoefficient": fixed(
                record_number(
                    atmosphere_record,
                    "result.beta_angstrom",
                    "GOES-18 libRadtran",
                ),
                5,
            ),
        }

    macros = {
        "ConditionLaunchDate": tex_date(launch_local),
        "ConditionLaunchUTC": tex_time(launch_utc),
        "ConditionLaunchWindowMinutes": integer(
            number(data, "irsa.integration.minutes")
        ),
        "ConditionMetarStation": tex_text(text(data, "metar.station")),
        "ConditionMetarObservationUTC": tex_time(metar_observation_time),
        "ConditionMetarObservationLeadMinutes": integer(
            (launch_utc - metar_observation_time).total_seconds() / 60
        ),
        "ConditionMetarPressure": fixed(number(data, "metar.pressure_hpa"), 1),
        "ConditionMetarTemperature": fixed(
            number(data, "metar.selected_report.temp"), 1
        ),
        "ConditionMetarSky": tex_text(text(data, "metar.selected_report.cover")),
        **goes_macros,
        "ConditionIdlerBandMinimum": integer(band_minimum),
        "ConditionIdlerBandMaximum": integer(band_maximum),
        "ConditionIdlerBand": (
            f"{integer(band_minimum)}--{integer(band_maximum)}"
        ),
        "ConditionLaunchMirrorTransmission": fixed(MIRROR_TRANSMISSION, 3),
        "ConditionLaunchMirrorTransmissionPercent": fixed(
            100 * MIRROR_TRANSMISSION, 1
        ),
        "ConditionLaunchWindowSurfaceReflectance": fixed(
            WINDOW_SURFACE_REFLECTANCE, 4
        ),
        "ConditionLaunchWindowTransmission": fixed(transmission.window, 4),
        "ConditionLaunchOpticsTransmission": fixed(
            transmission.launch_optics, 4
        ),
        "ConditionAtmosphericTransmission": fixed(atmospheric, 4),
        "ConditionAeronetRecordUTC": tex_time(aeronet_time),
        "ConditionAeronetLeadMinutes": integer(
            (launch_utc - aeronet_time).total_seconds() / 60
        ),
        "ConditionAeronetAtmosphericTransmission": fixed(
            aeronet_atmospheric, 4
        ),
        "ConditionZenithRA": right_ascension(number(data, "irsa.ra_deg")),
        "ConditionZenithDec": declination(number(data, "irsa.dec_deg")),
        "ConditionIrsaBand": tex_text(text(data, "irsa.filter_name")),
        "ConditionIrsaWavelength": fixed(
            number(data, "irsa.matched_wavelength_um") * 1000, 1
        ),
        "ConditionIrsaExtinctionColumn": tex_text(
            text(data, "irsa.extinction_column")
        ),
        "ConditionMilkyWayExtinction": fixed(extinction, 3),
        "ConditionMilkyWayOpticalDepth": fixed(0.4 * math.log(10) * extinction, 3),
        "ConditionMilkyWayTransmission": fixed(milky_way, 4),
    }
    return macros


def render(
    source_path: Path,
    source_bytes: bytes,
    data: dict[str, Any],
    atmosphere_source: str = "goes",
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> str:
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    macros = build_macros(data, atmosphere_source, milky_way_basis)
    lines = [
        "% Generated file. Do not edit.",
        f"% Source: {source_path.name}",
        f"% Source SHA-256: {source_hash}",
    ]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{macro_value}}}"
        for name, macro_value in macros.items()
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
        description="Generate semantic TeX macros from pinned conditions JSON."
    )
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--source", type=Path, required=True)
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
        raise ConditionsError(
            f"cannot read pinned conditions {args.source}: {exc}"
        ) from exc

    try:
        data = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ConditionsError(f"invalid JSON in {args.source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConditionsError("conditions result must be a JSON object")

    validate(data, args.atmosphere_source)
    expected = render(
        args.source,
        source_bytes,
        data,
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
    except ConditionsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
