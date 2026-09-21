#!/usr/bin/env python3

from dataclasses import dataclass

from .artifact_config import load_config
from .igm_transmission import calculate_igm_transmission

# The launch mirror was a Thorlabs BB1-E03 used at 45 degrees.
# Product page: https://www.thorlabs.com/item/BB1-E03 Its specified
# lower bound is R_avg > 99.2% for both polarizations over 780--870
# nm, so the survival budget conservatively uses 0.992 rather than a
# fitted or measured central value.
MIRROR_TRANSMISSION = 0.992

# The roof window was a Thorlabs WG41050-B UVFS window with a B
# antireflection coating on both surfaces. The manufacturer Product
# page: https://www.thorlabs.com/item/WG41050-B specifies average
# reflectance below 0.34% per surface near normal incidence; bulk
# absorption through 5 mm of UVFS at 810 nm is treated as negligible.
WINDOW_SURFACE_REFLECTANCE = 0.0034
WINDOW_TRANSMISSION = (1 - WINDOW_SURFACE_REFLECTANCE) ** 2
LAUNCH_OPTICS_TRANSMISSION = MIRROR_TRANSMISSION * WINDOW_TRANSMISSION
IGM_MODEL = calculate_igm_transmission(load_config()["igm_transmission"])
IGM_TRANSMISSION = IGM_MODEL.nominal_transmission
GRAY_DUST_IGM_TRANSMISSION = IGM_MODEL.gray_transmission
MILKY_WAY_BASES = ("point", "integrated", "minimum")
DEFAULT_MILKY_WAY_BASIS = "minimum"


@dataclass(frozen=True)
class TransmissionValues:
    window: float
    launch_optics: float
    atmosphere: float
    aeronet_atmosphere: float
    milky_way_point: float
    milky_way_integrated: float
    milky_way_minimum: float
    milky_way_basis: str
    milky_way: float
    finite_path: float
    infinity: float
    aeronet_infinity: float
    average_milky_way_infinity: float
    gray_dust_infinity: float


def calculate_transmissions(
    *,
    atmosphere: float,
    aeronet_atmosphere: float,
    milky_way_point: float,
    milky_way_integrated: float,
    milky_way_minimum: float,
    milky_way_basis: str = DEFAULT_MILKY_WAY_BASIS,
) -> TransmissionValues:
    inputs = {
        "atmosphere": atmosphere,
        "aeronet_atmosphere": aeronet_atmosphere,
        "milky_way_point": milky_way_point,
        "milky_way_integrated": milky_way_integrated,
        "milky_way_minimum": milky_way_minimum,
    }
    for name, value in inputs.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} transmission is outside [0, 1]: {value!r}")
    milky_way_values = {
        "point": milky_way_point,
        "integrated": milky_way_integrated,
        "minimum": milky_way_minimum,
    }
    if milky_way_basis not in milky_way_values:
        raise ValueError(
            f"unknown Milky Way transmission basis {milky_way_basis!r}; "
            f"expected one of {', '.join(MILKY_WAY_BASES)}"
        )
    milky_way = milky_way_values[milky_way_basis]

    window = WINDOW_TRANSMISSION
    launch_optics = LAUNCH_OPTICS_TRANSMISSION
    finite_path = launch_optics * atmosphere * milky_way

    return TransmissionValues(
        window=window,
        launch_optics=launch_optics,
        atmosphere=atmosphere,
        aeronet_atmosphere=aeronet_atmosphere,
        milky_way_point=milky_way_point,
        milky_way_integrated=milky_way_integrated,
        milky_way_minimum=milky_way_minimum,
        milky_way_basis=milky_way_basis,
        milky_way=milky_way,
        finite_path=finite_path,
        infinity=finite_path * IGM_TRANSMISSION,
        aeronet_infinity=(
            launch_optics
            * aeronet_atmosphere
            * milky_way
            * IGM_TRANSMISSION
        ),
        average_milky_way_infinity=(
            launch_optics
            * atmosphere
            * milky_way_integrated
            * IGM_TRANSMISSION
        ),
        gray_dust_infinity=finite_path * GRAY_DUST_IGM_TRANSMISSION,
    )
