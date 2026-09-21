"""GOES-18 archived-record loading, spatial reduction, and reporting."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import xarray as xr
from pyproj import Proj


GOES_SATELLITE = 18

@dataclass(frozen=True)
class GoesSite:
    name: str
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GOES site name must not be empty")
        if not math.isfinite(self.lat) or not -90.0 <= self.lat <= 90.0:
            raise ValueError("GOES site latitude must be finite and within [-90, 90]")
        if not math.isfinite(self.lon) or not -180.0 <= self.lon <= 180.0:
            raise ValueError("GOES site longitude must be finite and within [-180, 180]")


# OpenStreetMap way 945281329 building centroid, rounded to 1e-5 degrees.
HENLEY_HALL_SITE = GoesSite(
    name="Henley Hall",
    lat=34.41711,
    lon=-119.84459,
)


@dataclass(frozen=True)
class MeasurementSpec:
    key: str
    variable: str
    primary_quality_flag: str
    accepted_quality_values: frozenset[int]
    pessimistic_high: bool
    screening_rule: str = "quality-values"


@dataclass(frozen=True)
class AerosolPixelContext:
    aod: float | None
    aod_dqf: int | None
    ae2: float | None


@dataclass(frozen=True)
class ProductSpec:
    key: str
    product: str
    measurements: tuple[MeasurementSpec, ...]


# GOES DQF reference for every field used by this module.  The authoritative
# definitions are in the NOAA GOES-R Series Product Definition and Users'
# Guide (PUG), Volume 5, Version 3.0:
# https://www.ospo.noaa.gov/resources/documents/PUG/GS%20Series%20416-R-PUG-L2%20Plus-0349%20Vol%205%20v3.0%20final.pdf
#
# AOD.DQF and AE_DQF are separate enumerated flags with the same values (PUG
# Table 5.11.6.1):
#   0 = high-quality retrieval
#   1 = medium-quality retrieval
#   2 = low-quality retrieval
#   3 = no retrieval
# AE1 is screened with AE_DQF. AE_DQF is shared by AE1 and AE2.  The
# NOAA ABI AOD Algorithm Theoretical Basis Document, section 3.4.3,
# says AE_DQF is degraded to 2 when AOD.DQF is low, AOD at 550 nm is
# below 0.2, or either AE1 or AE2 is outside [-1, 3]:
# https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Aerosol_Optical_Depth_v4.2_Feb2018.pdf
# This workflow accepts AE_DQF 0 or 1 directly.  It also accepts AE_DQF 2 only
# when co-located AOD.DQF is 0 or 1, 0 <= AOD < 0.2, and both AE1 and AE2 are
# within [-1, 3].  Those conditions isolate low AOD as the documented reason
# for the downgrade while rejecting low-quality AOD and out-of-range AE.
#
# COD.DQF is a CF flag_masks/flag_values bit field, not an ordered quality
# number (PUG Table 5.7.6.1).  For every pair below, the named condition is
# true when ``DQF & mask == value``:
#   mask   1: 0 = day algorithm;       1 = not day algorithm
#   mask   2: 0 = night algorithm;     2 = not night algorithm
#   mask   4: 0 = good quality;        4 = degraded quality
#   mask   8: 0 = no snow/ice issue;   8 = degraded by snow or sea ice
#   mask  16: 0 = no twilight issue;  16 = degraded by twilight
#   mask  32: 0 = no convergence issue; 32 = degraded by nonconvergence
#   mask  64: 0 = no glint issue;      64 = degraded by glint
#   mask 128: 0 = water phase;        128 = degraded for ice phase
#   mask 256: 0 = not thick cloud;    256 = degraded by thick cloud
#   mask 512: 0 = not thin cloud;     512 = degraded by thin cloud
# Missing COD is represented by COD's _FillValue, not by a special
# scalar DQF value.  This workflow uses DQF only as COD diagnostic
# metadata; COD screening is based on the COD value itself, as
# described beside COD_MEASUREMENT below.
#
# TPW has three separate enumerated flags (PUG Tables 5.15.6.1-1 through -3).
# DQF_Overall:
#    0 = good quality
#    1 = invalid: not geolocated or retrieval LZA threshold exceeded
#    2 = degraded: latitude threshold exceeded
#    3 = degraded: quantitative LZA threshold exceeded
#    4 = invalid: insufficient clear pixels in the field of regard
#    5 = invalid: missing NWP data
#    6 = invalid: missing L1b data or fatal processing error
#    7 = invalid: bad NWP surface-pressure index
#    8 = invalid: indeterminate land-surface emissivity
#    9 = invalid: bad TPW sigma-pressure-level index
#   10 = invalid: not-a-number result
# DQF_Retrieval provides additional retrieval diagnostics:
#   0 = good retrieval; 1 = nonconvergent; 2 = brightness-temperature residual
#   too large; 3 = incomplete convergence; 4 = unrealistic retrieved value;
#   5 = invalid radiative-transfer-model brightness temperature.
# DQF_SkinTemp describes the first-guess skin temperature:
#   0 = good; 1 = above the upper threshold; 2 = below the lower threshold.
# For good-only TPW screening, this workflow requires all three flags to be 0.
AOD_MEASUREMENT = MeasurementSpec(
    key="aod",
    variable="AOD",
    primary_quality_flag="DQF",
    accepted_quality_values=frozenset((0, 1)),
    pessimistic_high=True,
)
AE1_MEASUREMENT = MeasurementSpec(
    key="ae1",
    variable="AE1",
    primary_quality_flag="AE_DQF",
    accepted_quality_values=frozenset((0, 1)),
    pessimistic_high=False,
    screening_rule="ae-low-aod-exception",
)

# GOES cloud optical depth applies to pixels classified as cloudy upstream.
# For our clear-sky veto, every COD pixel overlapping the 5 km circle must be
# either exactly 0.0 or missing/fill-valued.  Any positive finite COD is a
# cloud-optical-depth retrieval and vetoes clear sky, regardless of its packed
# DQF bits.  Zero or missing COD means only that GOES supplied no positive COD;
# it is not by itself an affirmative clear-sky classification.  We adopt clear
# sky only when the independent METAR report also says CLR/SKC and
# when confirmed by direct visual observation.
#
# In the shared measurement representation, ``usable`` means that a positive
# COD retrieval exists.  Consequently, satisfying this all-zero-or-missing
# criterion produces an unusable reduced COD, which downstream code interprets
# as no cloud optical depth to apply.
#
# See run_libradtran_transmission.decide_clear_sky().
COD_MEASUREMENT = MeasurementSpec(
    key="cod",
    variable="COD",
    primary_quality_flag="DQF",
    accepted_quality_values=frozenset(),
    pessimistic_high=True,
    screening_rule="cod-positive-veto",
)
TPW_MEASUREMENT = MeasurementSpec(
    key="tpw",
    variable="TPW",
    primary_quality_flag="DQF_Overall",
    accepted_quality_values=frozenset((0,)),
    pessimistic_high=True,
    screening_rule="tpw-all-quality-good",
)

AOD_PRODUCT = ProductSpec(
    key="aod",
    product="ABI-L2-AODC",
    measurements=(AOD_MEASUREMENT, AE1_MEASUREMENT),
)
COD_PRODUCT = ProductSpec(
    key="cod",
    product="ABI-L2-CODC",
    measurements=(COD_MEASUREMENT,),
)
TPW_PRODUCT = ProductSpec(
    key="tpw",
    product="ABI-L2-TPWC",
    measurements=(TPW_MEASUREMENT,),
)
PRODUCTS = (AOD_PRODUCT, COD_PRODUCT, TPW_PRODUCT)


@dataclass(frozen=True)
class GoesSourceFiles:
    aod: Path
    cod: Path
    tpw: Path

    @classmethod
    def from_mapping(
        cls, sources: Mapping[str, Path | str]
    ) -> GoesSourceFiles:
        return cls(
            aod=Path(sources["goes_aod"]),
            cod=Path(sources["goes_cod"]),
            tpw=Path(sources["goes_tpw"]),
        )

    def for_product(self, product: ProductSpec) -> Path:
        return getattr(self, product.key)


@dataclass(frozen=True)
class GoesScan:
    path: Path
    product: ProductSpec
    satellite: int
    start_utc: datetime
    end_utc: datetime
    midpoint_utc: datetime


@dataclass(frozen=True)
class ScreenedMeasurement:
    spec: MeasurementSpec
    raw_value: float | None
    value: float | None
    quality_flags: tuple[tuple[str, int | None], ...]
    dqf: int
    usable: bool
    qf_reason: str
    reason: str
    screening_context: tuple[tuple[str, float | int | None], ...] = ()

    @property
    def qf(self) -> dict[str, int | None]:
        return dict(self.quality_flags)

    @property
    def context(self) -> dict[str, float | int | None]:
        return dict(self.screening_context)


@dataclass(frozen=True)
class GoesPixel:
    iy: int
    ix: int


@dataclass(frozen=True)
class PixelMeasurement:
    pixel: GoesPixel
    screened: ScreenedMeasurement


@dataclass(frozen=True)
class ReducedMeasurement:
    spec: MeasurementSpec
    value: float | None
    usable: bool
    reduction: str
    selected: PixelMeasurement | None
    inputs: tuple[PixelMeasurement, ...]


@dataclass(frozen=True)
class GoesRecord:
    scan: GoesScan
    pixels: tuple[GoesPixel, ...]
    measurements: tuple[ReducedMeasurement, ...]

    def measurement(self, key: str) -> ReducedMeasurement:
        matches = [item for item in self.measurements if item.spec.key == key]
        if len(matches) != 1:
            raise ValueError(
                f"GOES {self.scan.product.product} record lacks exactly one {key} measurement"
            )
        return matches[0]


@dataclass(frozen=True)
class GoesSummary:
    site: GoesSite
    aod_record: GoesRecord
    cod_record: GoesRecord
    tpw_record: GoesRecord

    @property
    def satellite(self) -> int:
        satellites = {
            self.aod_record.scan.satellite,
            self.cod_record.scan.satellite,
            self.tpw_record.scan.satellite,
        }
        if len(satellites) != 1:
            raise ValueError("GOES records identify different satellites")
        return next(iter(satellites))

    @property
    def aod(self) -> ReducedMeasurement:
        return self.aod_record.measurement("aod")

    @property
    def ae1(self) -> ReducedMeasurement:
        return self.aod_record.measurement("ae1")

    @property
    def cod(self) -> ReducedMeasurement:
        return self.cod_record.measurement("cod")

    @property
    def tpw(self) -> ReducedMeasurement:
        return self.tpw_record.measurement("tpw")


class GoesPixelSelector(Protocol):
    name: str

    def select(self, ds: Any, site: GoesSite) -> tuple[GoesPixel, ...]:
        ...


class GoesPixelReducer(Protocol):
    name: str

    def reduce(
        self,
        spec: MeasurementSpec,
        inputs: tuple[PixelMeasurement, ...],
    ) -> ReducedMeasurement:
        ...


@dataclass(frozen=True)
class FootprintCircleSelector:
    radius_m: float
    name: str = "footprint-circle"

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("GOES selection radius must be finite and positive")

    def select(self, ds: Any, site: GoesSite) -> tuple[GoesPixel, ...]:
        return overlapping_circle_pixels(ds, site, self.radius_m)


@dataclass(frozen=True)
class PessimisticPixelReducer:
    name: str = "pessimistic-extreme"

    def reduce(
        self,
        spec: MeasurementSpec,
        inputs: tuple[PixelMeasurement, ...],
    ) -> ReducedMeasurement:
        usable = tuple(
            item
            for item in inputs
            if item.screened.usable and item.screened.value is not None
        )
        if not usable:
            return ReducedMeasurement(
                spec=spec,
                value=None,
                usable=False,
                reduction=self.name,
                selected=None,
                inputs=inputs,
            )

        def value(item: PixelMeasurement) -> float:
            result = item.screened.value
            if result is None:
                raise AssertionError("usable GOES pixel lacks a value")
            return result

        selected = (
            max(usable, key=value)
            if spec.pessimistic_high
            else min(usable, key=value)
        )
        return ReducedMeasurement(
            spec=spec,
            value=value(selected),
            usable=True,
            reduction=self.name,
            selected=selected,
            inputs=inputs,
        )


PIXEL_SELECTOR = FootprintCircleSelector(radius_m=5_000.0)
PIXEL_REDUCER = PessimisticPixelReducer()


_GOES_FILENAME_RE = re.compile(
    r"OR_ABI-L2-(AODC|CODC|TPWC)-M\d+_G(\d{2})_"
    r"s(\d{14})_e(\d{14})_c(\d{14})\.nc$"
)


def _require_aware_utc(value: datetime, description: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{description} must be timezone-aware")
    return value.astimezone(timezone.utc)


def parse_goes_timestamp(value: str) -> datetime:
    """Parse a GOES YYYYJJJHHMMSSd timestamp, including tenths of a second."""
    if len(value) != 14 or not value.isdigit():
        raise ValueError(f"invalid GOES timestamp: {value!r}")
    year = int(value[:4])
    jday = int(value[4:7])
    hour = int(value[7:9])
    minute = int(value[9:11])
    second = int(value[11:13])
    tenth = int(value[13])
    try:
        days_in_year = datetime(year, 12, 31).timetuple().tm_yday
    except ValueError as exc:
        raise ValueError(f"invalid GOES timestamp: {value!r}") from exc
    if not (
        1 <= jday <= days_in_year
        and 0 <= hour <= 23
        and 0 <= minute <= 59
        and 0 <= second <= 59
    ):
        raise ValueError(f"invalid GOES timestamp: {value!r}")
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=jday - 1,
        hours=hour,
        minutes=minute,
        seconds=second,
        microseconds=tenth * 100_000,
    )


def parse_goes_scan(path: Path, expected_product: ProductSpec) -> GoesScan:
    match = _GOES_FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"GOES filename lacks valid scan metadata: {path.name}")
    product_name = f"ABI-L2-{match.group(1)}"
    if product_name != expected_product.product:
        raise ValueError(
            f"GOES source {path.name} identifies {product_name}; "
            f"expected {expected_product.product}"
        )
    satellite = int(match.group(2))
    if satellite != GOES_SATELLITE:
        raise ValueError(
            f"GOES source {path.name} identifies GOES-{satellite}; "
            f"expected GOES-{GOES_SATELLITE}"
        )
    start = parse_goes_timestamp(match.group(3))
    end = parse_goes_timestamp(match.group(4))
    if end <= start:
        raise ValueError(f"GOES scan end must follow its start: {path.name}")
    return GoesScan(
        path=path,
        product=expected_product,
        satellite=satellite,
        start_utc=start,
        end_utc=end,
        midpoint_utc=start + (end - start) / 2,
    )


def open_local_dataset(path: Path):
    return xr.open_dataset(path, engine="h5netcdf")


def require_variable(ds: Any, name: str) -> str:
    if name in ds.data_vars:
        return name
    available = ", ".join(repr(str(item)) for item in ds.data_vars)
    raise ValueError(
        f"Required GOES data variable not found; expected {name!r}; "
        f"available data variables: {available or 'none'}"
    )


def _validated_coordinates(values: Any, axis: str) -> np.ndarray:
    try:
        coordinates = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GOES {axis} coordinates are not numeric") from exc
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(
            f"GOES {axis} coordinates must be a 1D grid of at least two points"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"GOES {axis} coordinates must be finite")

    differences = np.diff(coordinates)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(f"GOES {axis} coordinates must be strictly monotonic")
    return coordinates


def _goes_projection(ds: Any) -> tuple[Proj, float]:
    proj = ds["goes_imager_projection"]
    height = float(proj.perspective_point_height)
    lon0 = float(proj.longitude_of_projection_origin)
    semi_major = float(proj.semi_major_axis)
    semi_minor = float(proj.semi_minor_axis)
    sweep = str(proj.sweep_angle_axis)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("GOES perspective-point height must be finite and positive")
    if not math.isfinite(lon0):
        raise ValueError("GOES projection-origin longitude must be finite")
    if not math.isfinite(semi_major) or semi_major <= 0.0:
        raise ValueError("GOES semi-major axis must be finite and positive")
    if not math.isfinite(semi_minor) or semi_minor <= 0.0:
        raise ValueError("GOES semi-minor axis must be finite and positive")
    if sweep not in ("x", "y"):
        raise ValueError("GOES sweep-angle axis must be 'x' or 'y'")

    return (
        Proj(
            proj="geos",
            h=height,
            lon_0=lon0,
            a=semi_major,
            b=semi_minor,
            sweep=sweep,
        ),
        height,
    )


def _require_angular_coordinates(ds: Any) -> None:
    x_units = str(ds["x"].attrs.get("units", "")).strip().lower()
    y_units = str(ds["y"].attrs.get("units", "")).strip().lower()
    angular_units = {"rad", "radian", "radians"}
    if x_units not in angular_units or y_units not in angular_units:
        raise ValueError("GOES x/y coordinates must use radians")


def _coordinate_edges(values: Any, axis: str) -> np.ndarray:
    coordinates = _validated_coordinates(values, axis)
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = (coordinates[:-1] + coordinates[1:]) / 2.0
    edges[0] = coordinates[0] - (coordinates[1] - coordinates[0]) / 2.0
    edges[-1] = coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2.0
    return edges


def _point_in_polygon(x: np.ndarray, y: np.ndarray) -> bool:
    inside = False
    previous = x.size - 1
    for current in range(x.size):
        crosses_axis = (y[current] > 0.0) != (y[previous] > 0.0)
        if crosses_axis:
            crossing_x = (
                (x[previous] - x[current])
                * (-y[current])
                / (y[previous] - y[current])
                + x[current]
            )
            if crossing_x > 0.0:
                inside = not inside
        previous = current
    return inside


def _origin_to_segment_distance(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(x0, y0)
    fraction = -(x0 * dx + y0 * dy) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(x0 + fraction * dx, y0 + fraction * dy)


def _polygon_overlaps_circle(
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
) -> bool:
    if np.any(np.hypot(x, y) <= radius_m):
        return True
    if _point_in_polygon(x, y):
        return True
    return any(
        _origin_to_segment_distance(
            float(x[index]),
            float(y[index]),
            float(x[(index + 1) % x.size]),
            float(y[(index + 1) % y.size]),
        )
        <= radius_m
        for index in range(x.size)
    )


def overlapping_circle_pixels(
    ds: Any,
    site: GoesSite,
    radius_m: float,
) -> tuple[GoesPixel, ...]:
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("GOES selection radius must be finite and positive")
    projection, height = _goes_projection(ds)
    _require_angular_coordinates(ds)
    x_edges = _coordinate_edges(ds["x"].values, "x") * height
    y_edges = _coordinate_edges(ds["y"].values, "y") * height
    local = Proj(
        proj="aeqd",
        lat_0=site.lat,
        lon_0=site.lon,
        datum="WGS84",
        units="m",
    )

    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    circle_lon, circle_lat = local(
        radius_m * np.cos(angles),
        radius_m * np.sin(angles),
        inverse=True,
        errcheck=True,
    )
    circle_x, circle_y = projection(
        circle_lon,
        circle_lat,
        errcheck=True,
    )
    x_lower = float(np.min(circle_x))
    x_upper = float(np.max(circle_x))
    y_lower = float(np.min(circle_y))
    y_upper = float(np.max(circle_y))
    if not (
        float(np.min(x_edges)) <= x_lower <= x_upper <= float(np.max(x_edges))
        and float(np.min(y_edges)) <= y_lower <= y_upper <= float(np.max(y_edges))
    ):
        raise ValueError("GOES selection circle is not fully covered by the grid")

    x_min = np.minimum(x_edges[:-1], x_edges[1:])
    x_max = np.maximum(x_edges[:-1], x_edges[1:])
    y_min = np.minimum(y_edges[:-1], y_edges[1:])
    y_max = np.maximum(y_edges[:-1], y_edges[1:])
    x_indices = np.flatnonzero((x_max >= x_lower) & (x_min <= x_upper))
    y_indices = np.flatnonzero((y_max >= y_lower) & (y_min <= y_upper))
    if x_indices.size == 0 or y_indices.size == 0:
        raise ValueError("GOES selection circle lies outside the coordinate grid")

    edge_fraction = np.linspace(0.0, 1.0, 8, endpoint=False)
    selected: list[GoesPixel] = []
    for iy in y_indices:
        y0 = y_edges[iy]
        y1 = y_edges[iy + 1]
        for ix in x_indices:
            x0 = x_edges[ix]
            x1 = x_edges[ix + 1]
            footprint_x = np.concatenate(
                (
                    x0 + (x1 - x0) * edge_fraction,
                    np.full(edge_fraction.size, x1),
                    x1 + (x0 - x1) * edge_fraction,
                    np.full(edge_fraction.size, x0),
                )
            )
            footprint_y = np.concatenate(
                (
                    np.full(edge_fraction.size, y0),
                    y0 + (y1 - y0) * edge_fraction,
                    np.full(edge_fraction.size, y1),
                    y1 + (y0 - y1) * edge_fraction,
                )
            )
            longitude, latitude = projection(
                footprint_x,
                footprint_y,
                inverse=True,
                errcheck=True,
            )
            local_x, local_y = local(
                longitude,
                latitude,
                errcheck=True,
            )
            if _polygon_overlaps_circle(
                np.asarray(local_x, dtype=float),
                np.asarray(local_y, dtype=float),
                radius_m,
            ):
                selected.append(GoesPixel(iy=int(iy), ix=int(ix)))
    if not selected:
        raise ValueError("GOES selection circle overlaps no pixels")
    return tuple(selected)


def read_pixel(ds: Any, variable: str, iy: int, ix: int) -> float | None:
    try:
        raw = ds[variable].values[iy, ix]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"GOES {variable} pixel (iy={iy}, ix={ix}) is unreadable"
        ) from exc
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"GOES {variable} pixel is not numeric")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GOES {variable} pixel is not numeric") from exc
    return value if np.isfinite(value) else None


def read_quality_flags(
    ds: Any, data_variable: str, iy: int, ix: int
) -> tuple[tuple[str, int | None], ...]:
    output: dict[str, int | None] = {}
    ancillary = str(
        ds[data_variable].attrs.get("ancillary_variables", "") or ""
    ).split()

    def read(name: str) -> int:
        try:
            return int(ds[name].values[iy, ix])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"GOES {data_variable} quality flag {name} is unreadable"
            ) from exc

    for name in ancillary:
        if not (name.startswith("DQF") or name == "AE_DQF"):
            continue
        if name not in ds.variables:
            continue
        output[name] = read(name)
    fallback_names = (
        ("AE_DQF",)
        if data_variable in {"AE1", "AE2"}
        else ("DQF", "DQF_Overall")
    )
    for name in fallback_names:
        if name not in output and name in ds.variables:
            output[name] = read(name)
    return tuple(output.items())


def screen_measurement(
    spec: MeasurementSpec,
    raw_value: float | None,
    quality_flags: Mapping[str, int | None],
    *,
    iy: int = -1,
    ix: int = -1,
    aerosol_context: AerosolPixelContext | None = None,
) -> ScreenedMeasurement:
    primary = spec.primary_quality_flag
    if primary not in quality_flags:
        raise ValueError(f"GOES {spec.key} is missing required quality flag {primary}")
    dqf = quality_flags[primary]
    if dqf is None:
        raise ValueError(f"GOES {spec.key} quality flag {primary} is unreadable")

    screening_context: tuple[tuple[str, float | int | None], ...] = ()
    if aerosol_context is not None:
        screening_context = (
            ("AOD", aerosol_context.aod),
            ("AOD_DQF", aerosol_context.aod_dqf),
            ("AE2", aerosol_context.ae2),
        )

    if spec.screening_rule == "ae-low-aod-exception":
        if raw_value is None:
            usable = False
            value = None
            qf_reason = "missing/invalid data value"
        elif dqf in spec.accepted_quality_values:
            usable = True
            value = raw_value
            qf_reason = "accepted"
        elif dqf != 2:
            usable = False
            value = None
            qf_reason = f"rejected by {primary}={dqf}"
        elif aerosol_context is None:
            usable = False
            value = None
            qf_reason = (
                "rejected by AE_DQF=2: missing co-located AOD/AE2 context"
            )
        else:
            rejected: list[str] = []
            if aerosol_context.aod_dqf not in (0, 1):
                rejected.append(f"AOD.DQF={aerosol_context.aod_dqf}")
            if (
                aerosol_context.aod is None
                or not 0.0 <= aerosol_context.aod < 0.2
            ):
                rejected.append(f"AOD={aerosol_context.aod}")
            if not -1.0 <= raw_value <= 3.0:
                rejected.append(f"AE1={raw_value}")
            if (
                aerosol_context.ae2 is None
                or not -1.0 <= aerosol_context.ae2 <= 3.0
            ):
                rejected.append(f"AE2={aerosol_context.ae2}")
            usable = not rejected
            value = raw_value if usable else None
            if usable:
                qf_reason = (
                    "accepted AE_DQF=2 low-AOD exception "
                    f"(AOD={aerosol_context.aod}, "
                    f"AOD.DQF={aerosol_context.aod_dqf}, "
                    f"AE2={aerosol_context.ae2})"
                )
            else:
                qf_reason = (
                    "rejected AE_DQF=2 low-AOD exception: "
                    + ", ".join(rejected)
                )
    elif spec.screening_rule == "cod-positive-veto":
        if raw_value is None:
            usable = False
            value = None
            qf_reason = "no positive COD retrieval: missing/fill value"
        elif raw_value < 0.0:
            raise ValueError(
                f"GOES COD must be nonnegative at pixel (iy={iy}, ix={ix})"
            )
        elif raw_value == 0.0:
            usable = False
            value = None
            qf_reason = "no positive COD retrieval: COD=0.0"
        else:
            # COD DQF is a packed diagnostic bit field.  A degraded DQF does
            # not erase a positive cloud-optical-depth retrieval.
            usable = True
            value = raw_value
            qf_reason = f"positive COD retrieval ({primary}={dqf})"
    elif spec.screening_rule == "tpw-all-quality-good":
        required_flags = (
            "DQF_Overall",
            "DQF_Retrieval",
            "DQF_SkinTemp",
        )
        for name in required_flags:
            if name not in quality_flags:
                raise ValueError(
                    f"GOES TPW is missing required quality flag {name}"
                )
            if quality_flags[name] is None:
                raise ValueError(f"GOES TPW quality flag {name} is unreadable")
        rejected = tuple(
            f"{name}={quality_flags[name]}"
            for name in required_flags
            if quality_flags[name] != 0
        )
        if raw_value is None:
            usable = False
            value = None
            qf_reason = "missing/invalid data value"
        elif rejected:
            usable = False
            value = None
            qf_reason = f"rejected by {', '.join(rejected)}"
        else:
            usable = True
            value = raw_value
            qf_reason = "accepted"
    elif spec.screening_rule != "quality-values":
        raise ValueError(
            f"GOES {spec.key} has unknown screening rule {spec.screening_rule!r}"
        )
    elif raw_value is None:
        usable = False
        value = None
        qf_reason = "missing/invalid data value"
    else:
        usable = dqf in spec.accepted_quality_values
        value = raw_value if usable else None
        qf_reason = "accepted" if usable else f"rejected by {primary}={dqf}"

    if spec.key == "ae1":
        if raw_value is None:
            reason = f"AE1 missing/invalid at pixel (iy={iy}, ix={ix})"
        elif not usable:
            reason = qf_reason
        else:
            reason = spec.variable
    else:
        reason = qf_reason

    return ScreenedMeasurement(
        spec=spec,
        raw_value=raw_value,
        value=value,
        quality_flags=tuple(quality_flags.items()),
        dqf=dqf,
        usable=usable,
        qf_reason=qf_reason,
        reason=reason,
        screening_context=screening_context,
    )


def load_record(scan: GoesScan, site: GoesSite) -> GoesRecord:
    with open_local_dataset(scan.path) as dataset:
        pixels = PIXEL_SELECTOR.select(dataset, site)
        measurements: list[ReducedMeasurement] = []
        for spec in scan.product.measurements:
            variable = require_variable(dataset, spec.variable)
            aod_variable = None
            ae2_variable = None
            if spec.screening_rule == "ae-low-aod-exception":
                aod_variable = require_variable(dataset, "AOD")
                ae2_variable = require_variable(dataset, "AE2")
            inputs: list[PixelMeasurement] = []
            for pixel in pixels:
                raw_value = read_pixel(dataset, variable, pixel.iy, pixel.ix)
                quality_flags = dict(
                    read_quality_flags(dataset, variable, pixel.iy, pixel.ix)
                )
                aerosol_context = None
                if aod_variable is not None and ae2_variable is not None:
                    aod_quality = dict(
                        read_quality_flags(
                            dataset,
                            aod_variable,
                            pixel.iy,
                            pixel.ix,
                        )
                    )
                    if "DQF" not in aod_quality:
                        raise ValueError(
                            "GOES AE1 low-AOD screening is missing AOD.DQF"
                        )
                    aerosol_context = AerosolPixelContext(
                        aod=read_pixel(
                            dataset,
                            aod_variable,
                            pixel.iy,
                            pixel.ix,
                        ),
                        aod_dqf=aod_quality["DQF"],
                        ae2=read_pixel(
                            dataset,
                            ae2_variable,
                            pixel.iy,
                            pixel.ix,
                        ),
                    )
                inputs.append(
                    PixelMeasurement(
                        pixel=pixel,
                        screened=screen_measurement(
                            spec,
                            raw_value,
                            quality_flags,
                            iy=pixel.iy,
                            ix=pixel.ix,
                            aerosol_context=aerosol_context,
                        ),
                    )
                )
            measurements.append(PIXEL_REDUCER.reduce(spec, tuple(inputs)))
    return GoesRecord(
        scan=scan,
        pixels=pixels,
        measurements=tuple(measurements),
    )


def analyze_goes(
    sources: GoesSourceFiles,
    *,
    site: GoesSite = HENLEY_HALL_SITE,
) -> GoesSummary:
    records = {
        product.key: load_record(
            parse_goes_scan(sources.for_product(product), product),
            site,
        )
        for product in PRODUCTS
    }
    return GoesSummary(
        site=site,
        aod_record=records["aod"],
        cod_record=records["cod"],
        tpw_record=records["tpw"],
    )


def format_quality_flags(
    quality_flags: Mapping[str, int | None],
    preferred_order: tuple[str, ...] = (),
) -> str:
    if not quality_flags:
        return ""
    names: list[str] = []
    for name in preferred_order:
        if name in quality_flags and name not in names:
            names.append(name)
    for name in quality_flags:
        if name not in names:
            names.append(name)
    return ", ".join(f"{name}={quality_flags[name]}" for name in names)


def render_report_and_payload(
    summary: GoesSummary,
    *,
    target_utc: datetime,
    pressure_hpa: float,
    source_pull_prefix: str,
) -> tuple[str, dict[str, Any]]:
    """Serialize the analyzed GOES records."""
    target_utc = _require_aware_utc(target_utc, "GOES target time")
    if not source_pull_prefix:
        raise ValueError("archived GOES pull prefix is required")
    pressure = float(pressure_hpa)

    aod_record = summary.aod_record
    cod_record = summary.cod_record
    tpw_record = summary.tpw_record
    aod = summary.aod.selected.screened if summary.aod.selected else None
    alpha = summary.ae1.selected.screened if summary.ae1.selected else None
    cod = summary.cod.selected.screened if summary.cod.selected else None
    tpw = summary.tpw.selected.screened if summary.tpw.selected else None
    aod_value = summary.aod.value
    alpha_value = summary.ae1.value
    cod_value = summary.cod.value
    tpw_value = summary.tpw.value
    aod_start = aod_record.scan.start_utc
    cod_start = cod_record.scan.start_utc
    tpw_start = tpw_record.scan.start_utc

    def format_timestamp(value: datetime) -> str:
        result = value.strftime("%Y-%m-%d %H:%M:%S")
        if value.microsecond:
            result += f".{value.microsecond:06d}".rstrip("0")
        return result

    def format_utc(value: datetime) -> str:
        return format_timestamp(value.astimezone(timezone.utc)) + " UTC"

    def format_local(value: datetime) -> str:
        from zoneinfo import ZoneInfo

        local = value.astimezone(ZoneInfo("America/Los_Angeles"))
        return f"{format_timestamp(local)} {local.strftime('%Z')}"

    def delta_hours(value: datetime) -> float:
        return (value.astimezone(timezone.utc) - target_utc).total_seconds() / 3600.0

    def usable_count(measurement: ReducedMeasurement) -> int:
        return sum(item.screened.usable for item in measurement.inputs)

    def rejected_reason(measurement: ReducedMeasurement) -> str:
        if measurement.spec.screening_rule == "ae-low-aod-exception":
            return (
                "no finite AE_DQF=0,1 pixel or qualifying AE_DQF=2 "
                f"low-AOD exception among {len(measurement.inputs)} "
                "overlapping footprints"
            )
        if measurement.spec.screening_rule == "cod-positive-veto":
            return (
                f"no positive COD retrieval among {len(measurement.inputs)} "
                "overlapping footprints; all values are 0.0 or missing/fill"
            )
        if measurement.spec.screening_rule == "tpw-all-quality-good":
            return (
                "no finite DQF_Overall=0, DQF_Retrieval=0, "
                f"DQF_SkinTemp=0 pixel among {len(measurement.inputs)} "
                "overlapping footprints"
            )
        accepted = ",".join(
            str(value) for value in sorted(measurement.spec.accepted_quality_values)
        )
        return (
            f"no finite {measurement.spec.primary_quality_flag}={accepted} pixel "
            f"among {len(measurement.inputs)} overlapping footprints"
        )

    aod_qf_text = format_quality_flags(aod.qf, ("DQF",)) if aod else ""
    cod_qf_text = format_quality_flags(cod.qf, ("DQF",)) if cod else ""
    tpw_qf_text = format_quality_flags(
        tpw.qf if tpw else {},
        ("DQF_Overall", "DQF_Retrieval", "DQF_SkinTemp", "DQF"),
    )
    alpha_qf_text = (
        format_quality_flags(alpha.qf, ("AE_DQF",)) if alpha else ""
    )

    lines: list[str] = []
    lines.append(f"GOES ABI (GOES-{summary.satellite}) over {summary.site.name}")
    lines.append(
        f"AOD product: {aod_record.scan.product.product} "
        f"(var={summary.aod.spec.variable})"
    )
    lines.append(f"AOD file start (UTC): {format_utc(aod_start)}")
    lines.append(f"AOD file start (PT): {format_local(aod_start)}")
    lines.append(f"AOD delta from target: {delta_hours(aod_start):+.2f} hours")
    lines.append(
        f"AOD spatial pixels: {len(aod_record.pixels)} overlapping, "
        f"{usable_count(summary.aod)} usable"
    )
    lines.append(f"AOD_550: {aod_value}" + (f" ({aod_qf_text})" if aod_qf_text else ""))
    if summary.aod.selected:
        pixel = summary.aod.selected.pixel
        lines.append(f"AOD selected pixel: iy={pixel.iy}, ix={pixel.ix}")
    else:
        lines.append(f"AOD screening: {rejected_reason(summary.aod)}")

    if alpha_value is None:
        message = f"Angstrom exponent used: unavailable ({rejected_reason(summary.ae1)})"
        lines.append(message)
    else:
        message = (
            f"Angstrom exponent used: {alpha_value:.3f} "
            f"(source: {summary.ae1.spec.variable})"
        )
        if alpha_qf_text:
            message += f" [{alpha_qf_text}]"
        if alpha and alpha.qf_reason != "accepted":
            message += f" ({alpha.qf_reason})"
        lines.append(message)
        if summary.ae1.selected:
            pixel = summary.ae1.selected.pixel
            lines.append(f"Angstrom selected pixel: iy={pixel.iy}, ix={pixel.ix}")

    lines.append(
        f"COD product: {cod_record.scan.product.product} "
        f"(var={summary.cod.spec.variable})"
    )
    lines.append(f"COD file start (UTC): {format_utc(cod_start)}")
    lines.append(f"COD file start (PT): {format_local(cod_start)}")
    lines.append(f"COD delta from target: {delta_hours(cod_start):+.2f} hours")
    lines.append(
        f"COD spatial pixels: {len(cod_record.pixels)} overlapping, "
        f"{usable_count(summary.cod)} positive"
    )
    lines.append(f"COD: {cod_value}" + (f" ({cod_qf_text})" if cod_qf_text else ""))
    if summary.cod.selected:
        pixel = summary.cod.selected.pixel
        lines.append(f"COD selected pixel: iy={pixel.iy}, ix={pixel.ix}")
        lines.append(
            "COD screening: positive COD retrieval present; clear-sky veto triggered"
        )
    else:
        lines.append(f"COD screening: {rejected_reason(summary.cod)}")

    lines.append(
        f"TPW product: {tpw_record.scan.product.product} "
        f"(var={summary.tpw.spec.variable})"
    )
    lines.append(f"TPW file start (UTC): {format_utc(tpw_start)}")
    lines.append(f"TPW file start (PT): {format_local(tpw_start)}")
    lines.append(f"TPW delta from target: {delta_hours(tpw_start):+.2f} hours")
    lines.append(
        f"TPW spatial pixels: {len(tpw_record.pixels)} overlapping, "
        f"{usable_count(summary.tpw)} usable"
    )
    lines.append(f"TPW (mm): {tpw_value}" + (f" ({tpw_qf_text})" if tpw_qf_text else ""))
    if summary.tpw.selected:
        pixel = summary.tpw.selected.pixel
        lines.append(f"TPW selected pixel: iy={pixel.iy}, ix={pixel.ix}")
    else:
        lines.append(f"TPW screening: {rejected_reason(summary.tpw)}")

    def measurement_payload(measurement: ReducedMeasurement) -> dict[str, Any]:
        selected = measurement.selected
        screened = selected.screened if selected else None
        quality_counts: dict[str, int] = {}
        for item in measurement.inputs:
            dqf = str(item.screened.dqf)
            quality_counts[dqf] = quality_counts.get(dqf, 0) + 1
        return {
            "raw_value": screened.raw_value if screened else None,
            "value": measurement.value,
            "qf": screened.qf if screened else {},
            "dqf": screened.dqf if screened else None,
            "dqf_values": sorted(
                {item.screened.dqf for item in measurement.inputs}
            ),
            "qf_primary": measurement.spec.primary_quality_flag,
            "accepted_quality_values": sorted(
                measurement.spec.accepted_quality_values
            ),
            "usable": measurement.usable,
            "qf_reason": screened.qf_reason if screened else rejected_reason(measurement),
            "screening_rule": measurement.spec.screening_rule,
            "screening_context": screened.context if screened else {},
            "reduction": measurement.reduction,
            "pixel_count": len(measurement.inputs),
            "usable_pixel_count": usable_count(measurement),
            "quality_counts": quality_counts,
            "selected_pixel": (
                {"iy": selected.pixel.iy, "ix": selected.pixel.ix}
                if selected
                else None
            ),
        }

    aod_payload = measurement_payload(summary.aod)
    alpha_payload = measurement_payload(summary.ae1)
    cod_payload = measurement_payload(summary.cod)
    tpw_payload = measurement_payload(summary.tpw)

    payload: dict[str, Any] = {
        "source": "saved",
        "source_pull_prefix": source_pull_prefix,
        "satellite": f"GOES-{summary.satellite}",
        "lat": summary.site.lat,
        "lon": summary.site.lon,
        "spatial_selection": {
            "method": PIXEL_SELECTOR.name,
            "radius_m": PIXEL_SELECTOR.radius_m,
        },
        "target_utc": target_utc.isoformat(),
        "aod": {
            "product": aod_record.scan.product.product,
            "file": str(aod_record.scan.path),
            "start_utc": aod_start.isoformat(),
            "var": summary.aod.spec.variable,
            **aod_payload,
            "alpha": alpha_value,
            "alpha_source": (
                summary.ae1.spec.variable
                if alpha_value is not None
                else alpha_payload["qf_reason"]
            ),
            "alpha_raw_value": alpha_payload["raw_value"],
            "alpha_qf": alpha_payload["qf"],
            "alpha_dqf": alpha_payload["dqf"],
            "alpha_qf_primary": summary.ae1.spec.primary_quality_flag,
            "alpha_accepted_quality_values": alpha_payload[
                "accepted_quality_values"
            ],
            "alpha_screening_rule": alpha_payload["screening_rule"],
            "alpha_screening_context": alpha_payload["screening_context"],
            "alpha_usable": summary.ae1.usable,
            "alpha_reason": alpha_payload["qf_reason"],
            "alpha_reduction": alpha_payload["reduction"],
            "alpha_pixel_count": alpha_payload["pixel_count"],
            "alpha_usable_pixel_count": alpha_payload["usable_pixel_count"],
            "alpha_quality_counts": alpha_payload["quality_counts"],
            "alpha_selected_pixel": alpha_payload["selected_pixel"],
        },
        "cod": {
            "product": cod_record.scan.product.product,
            "file": str(cod_record.scan.path),
            "start_utc": cod_start.isoformat(),
            "var": summary.cod.spec.variable,
            **cod_payload,
        },
        "tpw": {
            "product": tpw_record.scan.product.product,
            "file": str(tpw_record.scan.path),
            "start_utc": tpw_start.isoformat(),
            "var": summary.tpw.spec.variable,
            **tpw_payload,
        },
        "pressure_hpa": pressure,
    }
    return "\n".join(lines) + "\n", payload


__all__ = [
    "AOD_MEASUREMENT",
    "AOD_PRODUCT",
    "AE1_MEASUREMENT",
    "AerosolPixelContext",
    "COD_MEASUREMENT",
    "COD_PRODUCT",
    "GOES_SATELLITE",
    "FootprintCircleSelector",
    "GoesPixel",
    "GoesPixelReducer",
    "GoesPixelSelector",
    "GoesRecord",
    "GoesScan",
    "GoesSite",
    "GoesSourceFiles",
    "GoesSummary",
    "MeasurementSpec",
    "PIXEL_REDUCER",
    "PIXEL_SELECTOR",
    "PRODUCTS",
    "PessimisticPixelReducer",
    "PixelMeasurement",
    "ProductSpec",
    "ReducedMeasurement",
    "ScreenedMeasurement",
    "TPW_MEASUREMENT",
    "TPW_PRODUCT",
    "HENLEY_HALL_SITE",
    "analyze_goes",
    "format_quality_flags",
    "load_record",
    "open_local_dataset",
    "overlapping_circle_pixels",
    "parse_goes_scan",
    "parse_goes_timestamp",
    "read_pixel",
    "read_quality_flags",
    "render_report_and_payload",
    "require_variable",
    "screen_measurement",
]
