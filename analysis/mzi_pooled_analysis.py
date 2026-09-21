#!/usr/bin/env python3
"""Pool arbitrary-phase launch-only quadratures across configured datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binomtest, chi2

from .artifact_config import load_config
from .transmission_values import TransmissionValues, calculate_transmissions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = ROOT / "results"
DEFAULT_JSON_OUTPUT = ROOT / "build" / "analysis" / "mzi-pooled-analysis.json"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ETA_MAX = 1.0
DEFAULT_COVERAGE_SIMULATIONS = 5000
DEFAULT_COVERAGE_ETA_GRID = (
    0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0
)
DEFAULT_RANDOM_PHASE_CONFIGURATIONS = 4
DEFAULT_SEED = 12345
PHASE_GRID_SIZE = 256
ETA_GRID_SIZE = 129
UPPER_GRID_SIZE = 257


class PooledAnalysisError(ValueError):
    """Raised when pooled-analysis inputs or inference are invalid."""


def _validate_eta_max(eta_max: float) -> float:
    if isinstance(eta_max, bool):
        raise PooledAnalysisError("eta_max must equal the physical upper bound 1.0")
    value = float(eta_max)
    if not math.isfinite(value) or value != 1.0:
        raise PooledAnalysisError("eta_max must equal the physical upper bound 1.0")
    return value


@dataclass(frozen=True)
class DatasetQuadrature:
    dataset_id: str
    q_hat: np.ndarray
    covariance: np.ndarray
    normalization: float
    period_V: float | None = None
    period_sigma_V: float | None = None
    source: str = ""
    transmission: TransmissionValues | None = None
    normalization_Rpm_cps: float | None = None
    normalization_Rfinite_path_cps: float | None = None
    _precision: np.ndarray = field(init=False, repr=False, compare=False)
    _precision_eigenvalues: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    _precision_eigenvectors: np.ndarray = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        q_hat = np.asarray(self.q_hat, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if q_hat.shape != (2,) or not np.all(np.isfinite(q_hat)):
            raise PooledAnalysisError(
                f"{self.dataset_id}: q_hat must contain two finite values"
            )
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise PooledAnalysisError(
                f"{self.dataset_id}: covariance must be a finite 2 by 2 matrix"
            )
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise PooledAnalysisError(
                f"{self.dataset_id}: covariance must be symmetric"
            )
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(eigenvalues <= 0.0):
            raise PooledAnalysisError(
                f"{self.dataset_id}: covariance must be positive definite"
            )
        normalization = float(self.normalization)
        if not math.isfinite(normalization) or normalization <= 0.0:
            raise PooledAnalysisError(
                f"{self.dataset_id}: normalization must be finite and positive"
            )
        object.__setattr__(self, "q_hat", q_hat.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "normalization", normalization)
        if (self.period_V is None) != (self.period_sigma_V is None):
            raise PooledAnalysisError(
                f"{self.dataset_id}: period value and uncertainty must be supplied together"
            )
        if self.period_V is not None and self.period_sigma_V is not None:
            period = float(self.period_V)
            period_sigma = float(self.period_sigma_V)
            if not math.isfinite(period) or period <= 0.0:
                raise PooledAnalysisError(
                    f"{self.dataset_id}: period must be finite and positive"
                )
            if not math.isfinite(period_sigma) or period_sigma < 0.0:
                raise PooledAnalysisError(
                    f"{self.dataset_id}: period uncertainty must be finite and nonnegative"
                )
            object.__setattr__(self, "period_V", period)
            object.__setattr__(self, "period_sigma_V", period_sigma)
        if self.transmission is not None:
            if (
                self.normalization_Rpm_cps is None
                or self.normalization_Rfinite_path_cps is None
            ):
                raise PooledAnalysisError(
                    f"{self.dataset_id}: physical normalization metadata is incomplete"
                )
            rpm = float(self.normalization_Rpm_cps)
            rfinite_path = float(self.normalization_Rfinite_path_cps)
            if not math.isfinite(rpm) or rpm <= 0.0:
                raise PooledAnalysisError(
                    f"{self.dataset_id}: Rpm must be finite and positive"
                )
            if not math.isfinite(rfinite_path) or rfinite_path <= 0.0:
                raise PooledAnalysisError(
                    f"{self.dataset_id}: Rfinite_path must be finite and positive"
                )
            if not math.isclose(
                rfinite_path,
                self.transmission.finite_path * rpm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise PooledAnalysisError(
                    f"{self.dataset_id}: Rfinite_path is inconsistent"
                )
            if not math.isclose(
                normalization,
                self.transmission.infinity * rpm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise PooledAnalysisError(
                    f"{self.dataset_id}: Rinfinity is inconsistent"
                )
            object.__setattr__(self, "normalization_Rpm_cps", rpm)
            object.__setattr__(
                self, "normalization_Rfinite_path_cps", rfinite_path
            )
        precision = np.linalg.inv(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(precision)
        object.__setattr__(self, "_precision", precision)
        object.__setattr__(
            self, "_precision_eigenvalues", eigenvalues
        )
        object.__setattr__(
            self, "_precision_eigenvectors", eigenvectors
        )

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    @property
    def precision_eigenvalues(self) -> np.ndarray:
        return self._precision_eigenvalues

    @property
    def precision_eigenvectors(self) -> np.ndarray:
        return self._precision_eigenvectors


@dataclass(frozen=True)
class ProfilePoint:
    eta: float
    deviance: float
    phases: tuple[float, ...]


@dataclass(frozen=True)
class PooledFit:
    eta_hat: float
    deviance_min: float
    phases_hat: tuple[float, ...]
    confidence: float
    nominal_cutoff: float
    eta_upper_nominal: float
    upper_limit_truncated: bool
    full_restoration_deviance: float
    full_restoration_q: float
    full_restoration_nominal_sigma: float
    full_restoration_phases: tuple[float, ...]
    nominal_gof_df: int
    nominal_gof_pvalue: float


@dataclass(frozen=True)
class NormalizationInference:
    """A pooled fit and its endpoint under one physical normalization."""

    name: str
    datasets: tuple[DatasetQuadrature, ...]
    fit: PooledFit
    eta_upper: float
    profile_cutoff: float
    upper_limit_truncated: bool


@dataclass(frozen=True)
class PhaseConfiguration:
    name: str
    phases: tuple[float, ...]
    kind: str


def _number(data: dict[str, Any], field: str, source: Path) -> float:
    current: Any = data
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PooledAnalysisError(f"{source}: missing numeric field {field}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise PooledAnalysisError(f"{source}: field {field} is not numeric")
    value = float(current)
    if not math.isfinite(value):
        raise PooledAnalysisError(f"{source}: field {field} is not finite")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PooledAnalysisError(f"cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PooledAnalysisError(f"{path}: expected a JSON object")
    return data


def _calculate_dataset_transmissions(
    config: dict[str, Any],
    dataset: dict[str, Any],
    conditions: dict[str, Any],
    source: Path,
) -> TransmissionValues:
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
        raise PooledAnalysisError(
            f"{source}: incomplete transmission record for {atmosphere_tag}"
        ) from exc
    return calculate_transmissions(
        atmosphere=atmosphere,
        aeronet_atmosphere=aeronet_atmosphere,
        milky_way_point=_number(
            conditions, "irsa.transmission_fraction", source
        ),
        milky_way_integrated=_number(
            conditions, "irsa.integration.avg_T", source
        ),
        milky_way_minimum=_number(
            conditions, "irsa.integration.min_T", source
        ),
        milky_way_basis=config["transmission_analysis"]["milky_way_basis"],
    )


def load_dataset_quadratures(
    config: dict[str, Any], source_root: Path
) -> list[DatasetQuadrature]:
    datasets: list[DatasetQuadrature] = []
    used_runs: set[str] = set()
    for dataset in config["datasets"]:
        dataset_id = dataset["id"]
        dataset_runs = {dataset["launch_run"], dataset["preserve_run"]}
        overlap = used_runs.intersection(dataset_runs)
        if overlap:
            raise PooledAnalysisError(
                f"{dataset_id}: acquisition reused across datasets: {sorted(overlap)}"
            )
        used_runs.update(dataset_runs)

        source = (
            source_root
            / "datasets"
            / dataset_id
            / "analysis"
            / "mzi-null-bounds.json"
        )
        bounds = _load_json(source)
        conditions_source = (
            source_root
            / "datasets"
            / dataset_id
            / "conditions"
            / "conditions.json"
        )
        conditions = _load_json(conditions_source)
        transmission = _calculate_dataset_transmissions(
            config, dataset, conditions, conditions_source
        )
        rpm = _number(bounds, "summary.Rpm_cps", source)
        rinfinity = _number(bounds, "summary.Rinfty_cps", source)
        summary_tinfinity = _number(bounds, "summary.Tinf", source)
        if not math.isclose(
            summary_tinfinity,
            transmission.infinity,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PooledAnalysisError(
                f"{dataset_id}: dataset-level and conditions transmissions disagree"
            )
        rfinite_path = transmission.finite_path * rpm
        common_igm = transmission.infinity / transmission.finite_path
        if not math.isclose(
            rinfinity,
            common_igm * rfinite_path,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PooledAnalysisError(
                f"{dataset_id}: Rinfinity and Rfinite_path are inconsistent"
            )
        c_sigma = _number(bounds, "summary.CL_sigma_cps", source)
        s_sigma = _number(bounds, "summary.SL_sigma_cps", source)
        covariance = np.array(
            [
                [
                    c_sigma * c_sigma,
                    _number(bounds, "summary.CLSL_cov_cps2", source),
                ],
                [
                    _number(bounds, "summary.CLSL_cov_cps2", source),
                    s_sigma * s_sigma,
                ],
            ],
            dtype=float,
        )
        datasets.append(
            DatasetQuadrature(
                dataset_id=dataset_id,
                q_hat=np.array(
                    [
                        _number(bounds, "summary.CL_hat_cps", source),
                        _number(bounds, "summary.SL_hat_cps", source),
                    ],
                    dtype=float,
                ),
                covariance=covariance,
                normalization=rinfinity,
                period_V=_number(bounds, "summary.period_V", source),
                period_sigma_V=_number(
                    bounds, "summary.period_sigma_V", source
                ),
                source=source.relative_to(source_root).as_posix(),
                transmission=transmission,
                normalization_Rpm_cps=rpm,
                normalization_Rfinite_path_cps=rfinite_path,
            )
        )
    if not datasets:
        raise PooledAnalysisError("no configured dataset inputs were found")
    return datasets


def _phase_deviance(
    phase: float,
    *,
    eta: float,
    dataset: DatasetQuadrature,
    q_hat: np.ndarray,
) -> float:
    mean = eta * dataset.normalization * np.array(
        [math.cos(phase), math.sin(phase)], dtype=float
    )
    residual = q_hat - mean
    return float(residual @ dataset.precision @ residual)


def profile_dataset(
    eta: float,
    dataset: DatasetQuadrature,
    *,
    q_hat: np.ndarray | None = None,
    phase_grid_size: int = PHASE_GRID_SIZE,
) -> tuple[float, float]:
    eta = float(eta)
    if not math.isfinite(eta) or eta < 0.0:
        raise PooledAnalysisError("eta must be finite and nonnegative")
    if phase_grid_size < 16:
        raise PooledAnalysisError("phase_grid_size must be at least 16")
    observed = dataset.q_hat if q_hat is None else np.asarray(q_hat, dtype=float)
    if observed.shape != (2,) or not np.all(np.isfinite(observed)):
        raise PooledAnalysisError("profiled q_hat must contain two finite values")
    if eta == 0.0:
        residual = observed
        return float(residual @ dataset.precision @ residual), 0.0

    step = 2.0 * math.pi / phase_grid_size
    phases = step * np.arange(phase_grid_size, dtype=float)
    unit = np.column_stack((np.cos(phases), np.sin(phases)))
    residual = observed[None, :] - eta * dataset.normalization * unit
    values = np.einsum(
        "ni,ij,nj->n", residual, dataset.precision, residual, optimize=True
    )
    index = int(np.argmin(values))
    center = float(phases[index])
    result = minimize_scalar(
        lambda phase: _phase_deviance(
            phase, eta=eta, dataset=dataset, q_hat=observed
        ),
        bounds=(center - step, center + step),
        method="bounded",
        options={"xatol": 1e-13},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise PooledAnalysisError(
            f"{dataset.dataset_id}: phase profiling failed at eta={eta}"
        )
    return float(result.fun), float(result.x % (2.0 * math.pi))


def _profile_dataset_batch(
    eta: np.ndarray,
    dataset: DatasetQuadrature,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile one configured dataset for many simulated samples."""
    observed = np.asarray(observations, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise PooledAnalysisError("batch observations must have shape (n, 2)")
    if not np.all(np.isfinite(observed)):
        raise PooledAnalysisError("batch observations must be finite")
    eta_values = np.asarray(eta, dtype=float)
    if eta_values.ndim == 0:
        eta_values = np.full(observed.shape[0], float(eta_values))
    if eta_values.shape != (observed.shape[0],):
        raise PooledAnalysisError("batch eta must be scalar or have shape (n,)")
    if np.any(~np.isfinite(eta_values)) or np.any(eta_values < 0.0):
        raise PooledAnalysisError("batch eta must be finite and nonnegative")

    eigenvalues = dataset.precision_eigenvalues
    eigenvectors = dataset.precision_eigenvectors
    rotated = observed @ eigenvectors
    radius = eta_values * dataset.normalization
    n_samples = observed.shape[0]
    closest = np.zeros_like(rotated)
    units = np.zeros_like(rotated)

    zero_radius = radius == 0.0
    active = ~zero_radius
    if np.any(active):
        p = rotated[active]
        a = radius[active]
        if math.isclose(
            float(eigenvalues[0]),
            float(eigenvalues[1]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            norms = np.linalg.norm(p, axis=1)
            nonzero = norms > 0.0
            local_units = np.zeros_like(p)
            local_units[nonzero] = p[nonzero] / norms[nonzero, None]
            local_units[~nonzero, 0] = 1.0
            units[active] = local_units
            closest[active] = a[:, None] * local_units
        else:
            w = eigenvalues
            wp = p * w[None, :]
            lower_value = np.nextafter(-float(w[0]), math.inf)
            lower = np.full(a.shape, lower_value)
            norm_p = np.linalg.norm(p, axis=1)
            upper = (
                float(w[1])
                * np.maximum(norm_p / np.maximum(a, np.finfo(float).tiny), 1.0)
                + float(w[1])
            )

            def secular(lagrange: np.ndarray) -> np.ndarray:
                scaled = wp / (w[None, :] + lagrange[:, None])
                return np.sum(scaled * scaled, axis=1) - a * a

            lower_values = secular(lower)
            regular = np.isfinite(lower_values) & (lower_values > 0.0)
            upper_values = secular(upper)
            for _ in range(64):
                expand = regular & (upper_values > 0.0)
                if not np.any(expand):
                    break
                upper[expand] = 2.0 * upper[expand] + float(w[1])
                upper_values = secular(upper)
            if np.any(regular & (upper_values > 0.0)):
                raise PooledAnalysisError("failed to bracket batch phase profile")

            lo = lower.copy()
            hi = upper.copy()
            for _ in range(64):
                midpoint = 0.5 * (lo + hi)
                values = secular(midpoint)
                above = regular & (values > 0.0)
                below = regular & ~above
                lo[above] = midpoint[above]
                hi[below] = midpoint[below]
            lagrange = 0.5 * (lo + hi)
            local_closest = wp / (
                w[None, :] + lagrange[:, None]
            )

            # The trust-region hard case occurs only when the observation has
            # no component along the least-precision eigenvector. Handle it
            # explicitly for deterministic axis-aligned simulation scenarios.
            hard = ~regular
            if np.any(hard):
                hard_p = p[hard]
                fixed_second = (
                    w[1] * hard_p[:, 1] / (w[1] - w[0])
                )
                remainder = np.maximum(
                    0.0, a[hard] * a[hard] - fixed_second * fixed_second
                )
                local_closest[hard, 0] = np.sqrt(remainder)
                local_closest[hard, 1] = fixed_second

            local_norms = np.linalg.norm(local_closest, axis=1)
            if np.any(local_norms <= 0.0):
                raise PooledAnalysisError("invalid batch phase-profile solution")
            local_units = local_closest / local_norms[:, None]
            units[active] = local_units
            closest[active] = a[:, None] * local_units

    # At eta=0 the phase is unidentified. Select the observed radial direction
    # for a stable coordinate-descent update if such a row is encountered.
    if np.any(zero_radius):
        p = rotated[zero_radius]
        norms = np.linalg.norm(p, axis=1)
        local_units = np.zeros_like(p)
        nonzero = norms > 0.0
        local_units[nonzero] = p[nonzero] / norms[nonzero, None]
        local_units[~nonzero, 0] = 1.0
        units[zero_radius] = local_units

    residual = rotated - closest
    deviance = np.sum(
        eigenvalues[None, :] * residual * residual, axis=1
    )
    units_original = units @ eigenvectors.T
    if deviance.shape != (n_samples,) or units_original.shape != (n_samples, 2):
        raise PooledAnalysisError("internal batch profile shape error")
    return deviance, units_original


def profile_joint_batch(
    eta: float | np.ndarray,
    datasets: Sequence[DatasetQuadrature],
    observations: Sequence[np.ndarray],
) -> np.ndarray:
    if len(observations) != len(datasets) or not datasets:
        raise PooledAnalysisError("batch observations must match nonempty datasets")
    first = np.asarray(observations[0], dtype=float)
    if first.ndim != 2 or first.shape[1] != 2:
        raise PooledAnalysisError("batch observations must have shape (n, 2)")
    eta_values = np.asarray(eta, dtype=float)
    if eta_values.ndim == 0:
        eta_values = np.full(first.shape[0], float(eta_values))
    total = np.zeros(first.shape[0], dtype=float)
    for dataset, observed in zip(datasets, observations):
        current = np.asarray(observed, dtype=float)
        if current.shape != first.shape:
            raise PooledAnalysisError(
                "all batch observation arrays must have the same shape"
            )
        deviance, _ = _profile_dataset_batch(eta_values, dataset, current)
        total += deviance
    return total


def fit_pooled_batch(
    datasets: Sequence[DatasetQuadrature],
    observations: Sequence[np.ndarray],
    *,
    eta_max: float = DEFAULT_ETA_MAX,
    tolerance: float = 1e-11,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit many simulated datasets by alternating exact conditional minima."""
    if len(observations) != len(datasets) or not datasets:
        raise PooledAnalysisError("batch observations must match nonempty datasets")
    arrays = [np.asarray(value, dtype=float) for value in observations]
    shape = arrays[0].shape
    if (
        arrays[0].ndim != 2
        or shape[1:] != (2,)
        or any(value.shape != shape for value in arrays)
    ):
        raise PooledAnalysisError(
            "all batch observation arrays must have shape (n, 2)"
        )
    eta_max = _validate_eta_max(eta_max)

    numerator = np.zeros(shape[0], dtype=float)
    denominator = 0.0
    for dataset, observed in zip(datasets, arrays):
        weight = float(np.trace(dataset.precision) / 2.0)
        numerator += (
            dataset.normalization * weight * np.linalg.norm(observed, axis=1)
        )
        denominator += dataset.normalization**2 * weight
    eta = np.clip(numerator / denominator, 0.0, eta_max)

    for iteration in range(1, max_iterations + 1):
        numerator.fill(0.0)
        denominator_values = np.zeros(shape[0], dtype=float)
        for dataset, observed in zip(datasets, arrays):
            _, unit = _profile_dataset_batch(eta, dataset, observed)
            precision_unit = unit @ dataset.precision
            numerator += dataset.normalization * np.sum(
                precision_unit * observed, axis=1
            )
            denominator_values += dataset.normalization**2 * np.sum(
                precision_unit * unit, axis=1
            )
        updated = np.clip(
            numerator / denominator_values, 0.0, eta_max
        )
        if float(np.max(np.abs(updated - eta))) <= tolerance:
            eta = updated
            break
        eta = updated
    else:
        raise PooledAnalysisError(
            f"batch pooled fit did not converge in {max_iterations} iterations"
        )
    deviance = profile_joint_batch(eta, datasets, arrays)
    return eta, deviance, iteration


def profile_joint(
    eta: float,
    datasets: Sequence[DatasetQuadrature],
    *,
    observations: Sequence[np.ndarray] | None = None,
) -> ProfilePoint:
    if not datasets:
        raise PooledAnalysisError("at least one dataset is required")
    if observations is None:
        observed = [dataset.q_hat for dataset in datasets]
    else:
        observed = [np.asarray(value, dtype=float) for value in observations]
        if len(observed) != len(datasets):
            raise PooledAnalysisError("observations must match the number of datasets")
    deviance = 0.0
    phases: list[float] = []
    for dataset, q_hat in zip(datasets, observed):
        value, phase = profile_dataset(eta, dataset, q_hat=q_hat)
        deviance += value
        phases.append(phase)
    return ProfilePoint(float(eta), float(deviance), tuple(phases))


def _global_minimum(
    function: Callable[[float], float],
    *,
    lower: float,
    upper: float,
    grid_size: int = ETA_GRID_SIZE,
) -> tuple[float, float]:
    if not 0.0 <= lower < upper:
        raise PooledAnalysisError("invalid minimization interval")
    grid = np.linspace(lower, upper, grid_size)
    values = np.array([function(float(value)) for value in grid], dtype=float)
    index = int(np.argmin(values))
    if index == 0:
        return float(grid[0]), float(values[0])
    if index == grid_size - 1:
        return float(grid[-1]), float(values[-1])
    result = minimize_scalar(
        function,
        bounds=(float(grid[index - 1]), float(grid[index + 1])),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise PooledAnalysisError("eta profiling failed")
    candidates = [
        (float(result.x), float(result.fun)),
        (float(grid[index]), float(values[index])),
    ]
    return min(candidates, key=lambda item: item[1])


def _upper_endpoint(
    function: Callable[[float], float],
    *,
    eta_hat: float,
    target: float,
    eta_max: float,
    grid_size: int = UPPER_GRID_SIZE,
) -> tuple[float, bool]:
    if eta_hat >= eta_max:
        return float(eta_max), True
    grid = np.linspace(eta_hat, eta_max, grid_size)
    values = np.array([function(float(value)) - target for value in grid])
    inside = np.flatnonzero(values <= 0.0)
    if not inside.size:
        raise PooledAnalysisError("profile minimum is outside the confidence set")
    last_inside = int(inside[-1])
    if last_inside == grid_size - 1:
        return float(eta_max), True
    next_outside = last_inside + 1
    while next_outside < grid_size and values[next_outside] <= 0.0:
        next_outside += 1
    if next_outside == grid_size:
        return float(eta_max), True
    root = brentq(
        lambda value: function(float(value)) - target,
        float(grid[last_inside]),
        float(grid[next_outside]),
        xtol=1e-12,
        rtol=1e-14,
    )
    return float(root), False


def fit_pooled(
    datasets: Sequence[DatasetQuadrature],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    eta_max: float = DEFAULT_ETA_MAX,
    observations: Sequence[np.ndarray] | None = None,
) -> PooledFit:
    if not datasets:
        raise PooledAnalysisError("at least one dataset is required")
    confidence = float(confidence)
    eta_max = _validate_eta_max(eta_max)
    if not 0.5 < confidence < 1.0:
        raise PooledAnalysisError("confidence must be between 0.5 and 1")
    cache: dict[float, ProfilePoint] = {}

    def point(eta: float) -> ProfilePoint:
        key = float(eta)
        if key not in cache:
            cache[key] = profile_joint(key, datasets, observations=observations)
        return cache[key]

    eta_hat, deviance_min = _global_minimum(
        lambda eta: point(eta).deviance,
        lower=0.0,
        upper=eta_max,
    )
    best = point(eta_hat)
    # For a regular scalar profile, a one-sided confidence level C uses the
    # two-sided chi-square probability 2C-1.
    nominal_cutoff = float(chi2.ppf(2.0 * confidence - 1.0, df=1))
    eta_upper, truncated = _upper_endpoint(
        lambda eta: point(eta).deviance,
        eta_hat=eta_hat,
        target=deviance_min + nominal_cutoff,
        eta_max=eta_max,
    )
    full = point(1.0)
    full_q = max(0.0, float(full.deviance - deviance_min))
    nominal_gof_df = len(datasets) - 1
    nominal_gof_pvalue = (
        float(chi2.sf(deviance_min, nominal_gof_df))
        if nominal_gof_df > 0
        else float("nan")
    )
    return PooledFit(
        eta_hat=float(eta_hat),
        deviance_min=float(deviance_min),
        phases_hat=best.phases,
        confidence=confidence,
        nominal_cutoff=nominal_cutoff,
        eta_upper_nominal=eta_upper,
        upper_limit_truncated=truncated,
        full_restoration_deviance=float(full.deviance),
        full_restoration_q=full_q,
        full_restoration_nominal_sigma=math.sqrt(full_q),
        full_restoration_phases=full.phases,
        nominal_gof_df=nominal_gof_df,
        nominal_gof_pvalue=nominal_gof_pvalue,
    )


def upper_limit_for_cutoff(
    datasets: Sequence[DatasetQuadrature],
    *,
    eta_hat: float,
    deviance_min: float,
    cutoff: float,
    eta_max: float = DEFAULT_ETA_MAX,
    observations: Sequence[np.ndarray] | None = None,
) -> tuple[float, bool]:
    eta_max = _validate_eta_max(eta_max)
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise PooledAnalysisError("profile cutoff must be finite and positive")
    cache: dict[float, float] = {}

    def deviance(eta: float) -> float:
        key = float(eta)
        if key not in cache:
            cache[key] = profile_joint(
                key, datasets, observations=observations
            ).deviance
        return cache[key]

    return _upper_endpoint(
        deviance,
        eta_hat=float(eta_hat),
        target=float(deviance_min) + cutoff,
        eta_max=eta_max,
    )


def normalization_view(
    datasets: Sequence[DatasetQuadrature],
    *,
    name: str,
    finite_path: bool = False,
    multiplier: float = 1.0,
) -> tuple[DatasetQuadrature, ...]:
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise PooledAnalysisError("normalization multiplier must be positive")
    result: list[DatasetQuadrature] = []
    for dataset in datasets:
        if finite_path:
            if dataset.normalization_Rfinite_path_cps is None:
                raise PooledAnalysisError(
                    f"{dataset.dataset_id}: finite-path normalization is unavailable"
                )
            normalization = dataset.normalization_Rfinite_path_cps
        else:
            normalization = dataset.normalization
        result.append(
            DatasetQuadrature(
                dataset_id=dataset.dataset_id,
                q_hat=dataset.q_hat,
                covariance=dataset.covariance,
                normalization=multiplier * normalization,
                period_V=dataset.period_V,
                period_sigma_V=dataset.period_sigma_V,
                source=f"{dataset.source}#{name}",
            )
        )
    return tuple(result)


def infer_normalization(
    datasets: Sequence[DatasetQuadrature],
    *,
    name: str,
    profile_cutoff: float,
    confidence: float = DEFAULT_CONFIDENCE,
    eta_max: float = DEFAULT_ETA_MAX,
) -> NormalizationInference:
    fit = fit_pooled(datasets, confidence=confidence, eta_max=eta_max)
    eta_upper, truncated = upper_limit_for_cutoff(
        datasets,
        eta_hat=fit.eta_hat,
        deviance_min=fit.deviance_min,
        cutoff=profile_cutoff,
        eta_max=eta_max,
    )
    return NormalizationInference(
        name=name,
        datasets=tuple(datasets),
        fit=fit,
        eta_upper=eta_upper,
        profile_cutoff=float(profile_cutoff),
        upper_limit_truncated=truncated,
    )


def _common_igm_transmission(datasets: Sequence[DatasetQuadrature]) -> float:
    values: list[float] = []
    for dataset in datasets:
        if dataset.transmission is None:
            raise PooledAnalysisError(
                f"{dataset.dataset_id}: transmission metadata is unavailable"
            )
        values.append(
            dataset.transmission.infinity
            / dataset.transmission.finite_path
        )
    reference = values[0]
    if not all(
        math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
        for value in values[1:]
    ):
        raise PooledAnalysisError(
            "configured datasets do not share one IGM transmission"
        )
    return reference


def validate_common_reparameterization(
    infinity: NormalizationInference,
    finite_path: NormalizationInference,
    common_igm_transmission: float,
    *,
    tolerance: float = 2e-9,
) -> None:
    expected = common_igm_transmission
    comparisons = {
        "eta_hat": (
            finite_path.fit.eta_hat,
            expected * infinity.fit.eta_hat,
        ),
        "eta_upper": (
            finite_path.eta_upper,
            expected * infinity.eta_upper,
        ),
        "deviance_min": (
            finite_path.fit.deviance_min,
            infinity.fit.deviance_min,
        ),
    }
    for name, (actual, target) in comparisons.items():
        if not math.isclose(
            actual, target, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise PooledAnalysisError(
                f"IGM reparameterization failed for {name}: "
                f"{actual} != {target}"
            )
    infinity_null = profile_joint(0.0, infinity.datasets).deviance
    finite_path_null = profile_joint(0.0, finite_path.datasets).deviance
    if not math.isclose(
        infinity_null,
        finite_path_null,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise PooledAnalysisError(
            "IGM reparameterization failed for the null deviance"
        )
    for infinity_dataset, finite_path_dataset in zip(
        infinity.datasets, finite_path.datasets
    ):
        infinity_prediction = (
            infinity.fit.eta_hat * infinity_dataset.normalization
        )
        finite_path_prediction = (
            finite_path.fit.eta_hat * finite_path_dataset.normalization
        )
        if not math.isclose(
            infinity_prediction,
            finite_path_prediction,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise PooledAnalysisError(
                f"{infinity_dataset.dataset_id}: fitted predictions do not map "
                "under the common IGM reparameterization"
            )


def leave_one_out(
    datasets: Sequence[DatasetQuadrature],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    eta_max: float = DEFAULT_ETA_MAX,
    cutoff: float | None = None,
) -> list[dict[str, Any]]:
    if len(datasets) < 2:
        raise PooledAnalysisError("leave-one-out analysis requires at least two datasets")
    results: list[dict[str, Any]] = []
    for omitted_index, omitted in enumerate(datasets):
        retained = [
            dataset for index, dataset in enumerate(datasets) if index != omitted_index
        ]
        fit = fit_pooled(retained, confidence=confidence, eta_max=eta_max)
        selected_cutoff = fit.nominal_cutoff if cutoff is None else float(cutoff)
        eta_upper, truncated = upper_limit_for_cutoff(
            retained,
            eta_hat=fit.eta_hat,
            deviance_min=fit.deviance_min,
            cutoff=selected_cutoff,
            eta_max=eta_max,
        )
        results.append(
            {
                "omitted_dataset_id": omitted.dataset_id,
                "eta_hat": fit.eta_hat,
                "eta_upper_nominal": fit.eta_upper_nominal,
                "eta_upper": eta_upper,
                "profile_cutoff": selected_cutoff,
                "upper_limit_truncated": truncated,
                "deviance_min": fit.deviance_min,
            }
        )
    return results


def build_phase_configurations(
    datasets: Sequence[DatasetQuadrature],
    fit: PooledFit,
    *,
    random_count: int,
    seed: int,
) -> list[PhaseConfiguration]:
    if len(fit.phases_hat) != len(datasets):
        raise PooledAnalysisError("fitted phases must match dataset inputs")
    if random_count < 0:
        raise PooledAnalysisError("random phase count must be nonnegative")

    covariance_axes: list[tuple[float, float]] = []
    for dataset in datasets:
        _, eigenvectors = np.linalg.eigh(dataset.covariance)
        minor_vector = eigenvectors[:, 0]
        major_vector = eigenvectors[:, 1]
        covariance_axes.append(
            (
                float(math.atan2(major_vector[1], major_vector[0]) % (2 * math.pi)),
                float(math.atan2(minor_vector[1], minor_vector[0]) % (2 * math.pi)),
            )
        )
    configurations = [
        PhaseConfiguration(
            "observed_fit",
            tuple(float(value) for value in fit.phases_hat),
            "observed",
        ),
        PhaseConfiguration(
            "all_covariance_major_axes",
            tuple(major for major, _ in covariance_axes),
            "axis_extreme",
        ),
        PhaseConfiguration(
            "all_covariance_minor_axes",
            tuple(minor for _, minor in covariance_axes),
            "axis_extreme",
        ),
        PhaseConfiguration(
            "alternating_covariance_axes",
            tuple(
                major if index % 2 == 0 else minor
                for index, (major, minor) in enumerate(covariance_axes)
            ),
            "axis_extreme",
        ),
    ]
    rng = np.random.default_rng(seed)
    for index in range(random_count):
        configurations.append(
            PhaseConfiguration(
                f"deterministic_random_{index + 1}",
                tuple(
                    float(value)
                    for value in rng.uniform(0.0, 2.0 * math.pi, len(datasets))
                ),
                "deterministic_random",
            )
        )
    return configurations


def simulate_upper_limit_coverage(
    datasets: Sequence[DatasetQuadrature],
    *,
    confidence: float,
    nominal_cutoff: float,
    eta_grid: Sequence[float],
    phase_configurations: Sequence[PhaseConfiguration],
    simulations: int,
    seed: int,
    eta_max: float = DEFAULT_ETA_MAX,
) -> dict[str, Any]:
    eta_max = _validate_eta_max(eta_max)
    if simulations < 100:
        raise PooledAnalysisError("coverage simulations must be at least 100")
    if not phase_configurations:
        raise PooledAnalysisError("at least one phase configuration is required")
    eta_values = sorted({float(value) for value in eta_grid})
    if (
        not eta_values
        or eta_values[0] < 0.0
        or eta_values[-1] > eta_max
        or any(not math.isfinite(value) for value in eta_values)
    ):
        raise PooledAnalysisError(
            "coverage eta grid must be finite and within the fitted domain"
        )
    for configuration in phase_configurations:
        if len(configuration.phases) != len(datasets):
            raise PooledAnalysisError(
                f"{configuration.name}: phases must match dataset inputs"
            )

    seed_sequence = np.random.SeedSequence(int(seed))
    scenario_seeds = seed_sequence.spawn(
        len(eta_values) * len(phase_configurations)
    )
    scenario_records: list[dict[str, Any]] = []
    quantiles: list[float] = []
    seed_index = 0
    for eta in eta_values:
        for configuration in phase_configurations:
            rng = np.random.default_rng(scenario_seeds[seed_index])
            seed_index += 1
            observations = []
            for dataset, phase in zip(datasets, configuration.phases):
                mean = eta * dataset.normalization * np.array(
                    [math.cos(phase), math.sin(phase)]
                )
                observations.append(
                    rng.multivariate_normal(
                        mean, dataset.covariance, size=simulations
                    )
                )
            eta_hat, deviance_min, iterations = fit_pooled_batch(
                datasets, observations, eta_max=eta_max
            )
            deviance_true = profile_joint_batch(eta, datasets, observations)
            delta = np.maximum(0.0, deviance_true - deviance_min)
            upper_statistic = np.where(eta_hat <= eta, delta, 0.0)
            covered = upper_statistic <= nominal_cutoff
            n_covered = int(np.count_nonzero(covered))
            coverage = n_covered / simulations
            interval = binomtest(
                n_covered, simulations
            ).proportion_ci(confidence_level=0.95, method="exact")
            empirical_cutoff = float(
                np.quantile(upper_statistic, confidence, method="higher")
            )
            quantiles.append(empirical_cutoff)
            scenario_records.append(
                {
                    "eta": eta,
                    "phase_configuration": configuration.name,
                    "phase_configuration_kind": configuration.kind,
                    "simulations": simulations,
                    "covered_nominal": n_covered,
                    "coverage_nominal": coverage,
                    "coverage_exact_95ci": [
                        float(interval.low), float(interval.high)
                    ],
                    "empirical_cutoff": empirical_cutoff,
                    "mean_eta_hat": float(np.mean(eta_hat)),
                    "fit_iterations_max": int(iterations),
                }
            )

    minimum_record = min(
        scenario_records, key=lambda record: record["coverage_nominal"]
    )
    maximum_cutoff_record = max(
        scenario_records, key=lambda record: record["empirical_cutoff"]
    )
    empirical_worst_case_cutoff = max(quantiles)
    recommended_cutoff = max(nominal_cutoff, empirical_worst_case_cutoff)
    return {
        "method": "parametric_gaussian_fixed_covariance_grid",
        "confidence": confidence,
        "simulations_per_scenario": simulations,
        "seed": int(seed),
        "eta_grid": eta_values,
        "phase_configurations": [
            {
                "name": configuration.name,
                "kind": configuration.kind,
                "phases_rad": list(configuration.phases),
            }
            for configuration in phase_configurations
        ],
        "n_scenarios": len(scenario_records),
        "nominal_cutoff": nominal_cutoff,
        "minimum_nominal_coverage": minimum_record["coverage_nominal"],
        "minimum_nominal_coverage_scenario": {
            "eta": minimum_record["eta"],
            "phase_configuration": minimum_record["phase_configuration"],
            "coverage_exact_95ci": minimum_record["coverage_exact_95ci"],
        },
        "empirical_worst_case_cutoff": empirical_worst_case_cutoff,
        "empirical_worst_case_cutoff_scenario": {
            "eta": maximum_cutoff_record["eta"],
            "phase_configuration": maximum_cutoff_record[
                "phase_configuration"
            ],
        },
        "recommended_cutoff": recommended_cutoff,
        "calibration_increases_cutoff": bool(
            recommended_cutoff > nominal_cutoff
        ),
        "scenarios": scenario_records,
    }


def simulate_common_effect_gof(
    datasets: Sequence[DatasetQuadrature],
    fit: PooledFit,
    *,
    simulations: int,
    seed: int,
    eta_max: float = DEFAULT_ETA_MAX,
) -> dict[str, Any]:
    eta_max = _validate_eta_max(eta_max)
    if simulations < 100:
        raise PooledAnalysisError("goodness-of-fit simulations must be at least 100")
    if len(fit.phases_hat) != len(datasets):
        raise PooledAnalysisError("fitted phases must match dataset inputs")
    rng = np.random.default_rng(int(seed))
    observations = []
    for dataset, phase in zip(datasets, fit.phases_hat):
        mean = fit.eta_hat * dataset.normalization * np.array(
            [math.cos(phase), math.sin(phase)]
        )
        observations.append(
            rng.multivariate_normal(mean, dataset.covariance, size=simulations)
        )
    eta_hat, deviance_min, iterations = fit_pooled_batch(
        datasets, observations, eta_max=eta_max
    )
    exceedances = int(np.count_nonzero(deviance_min >= fit.deviance_min))
    pvalue = (exceedances + 1.0) / (simulations + 1.0)
    return {
        "method": "parametric_gaussian_at_fitted_common_effect",
        "interpretation": (
            "likelihood_ratio_common_eta_vs_dataset_specific_eta"
        ),
        "simulations": simulations,
        "seed": int(seed),
        "observed_deviance_min": fit.deviance_min,
        "exceedances": exceedances,
        "pvalue": pvalue,
        "simulated_deviance_quantiles": {
            "q05": float(np.quantile(deviance_min, 0.05)),
            "q50": float(np.quantile(deviance_min, 0.50)),
            "q95": float(np.quantile(deviance_min, 0.95)),
        },
        "mean_eta_hat": float(np.mean(eta_hat)),
        "fit_iterations_max": int(iterations),
    }


def simulate_null_test(
    datasets: Sequence[DatasetQuadrature],
    fit: PooledFit,
    *,
    simulations: int,
    seed: int,
    eta_max: float = DEFAULT_ETA_MAX,
) -> dict[str, Any]:
    """Calibrate the common-restoration likelihood ratio under eta=0."""
    eta_max = float(eta_max)
    if not math.isfinite(eta_max) or eta_max <= 0.0:
        raise PooledAnalysisError("eta_max must be finite and positive")
    if simulations < 100:
        raise PooledAnalysisError("null-test simulations must be at least 100")
    observed_null_deviance = profile_joint(0.0, datasets).deviance
    observed_q = max(0.0, observed_null_deviance - fit.deviance_min)

    rng = np.random.default_rng(int(seed))
    observations = [
        rng.multivariate_normal(
            np.zeros(2, dtype=float), dataset.covariance, size=simulations
        )
        for dataset in datasets
    ]
    eta_hat, deviance_min, iterations = fit_pooled_batch(
        datasets, observations, eta_max=eta_max
    )
    null_deviance = profile_joint_batch(0.0, datasets, observations)
    simulated_q = np.maximum(0.0, null_deviance - deviance_min)
    exceedances = int(np.count_nonzero(simulated_q >= observed_q))
    pvalue = (exceedances + 1.0) / (simulations + 1.0)
    return {
        "method": "parametric_gaussian_likelihood_ratio_at_eta_zero",
        "null_hypothesis": "eta_equals_zero",
        "simulations": simulations,
        "seed": int(seed),
        "observed_null_deviance": observed_null_deviance,
        "observed_q": observed_q,
        "exceedances": exceedances,
        "pvalue": pvalue,
        "simulated_q_quantiles": {
            "q05": float(np.quantile(simulated_q, 0.05)),
            "q50": float(np.quantile(simulated_q, 0.50)),
            "q95": float(np.quantile(simulated_q, 0.95)),
        },
        "mean_eta_hat": float(np.mean(eta_hat)),
        "fit_iterations_max": int(iterations),
    }


def _inference_record(inference: NormalizationInference) -> dict[str, Any]:
    fit = inference.fit
    return {
        "eta_hat": fit.eta_hat,
        "eta_upper": inference.eta_upper,
        "profile_cutoff": inference.profile_cutoff,
        "upper_limit_truncated_at_eta_one": (
            inference.upper_limit_truncated
        ),
        "deviance_min": fit.deviance_min,
        "full_restoration_deviance": fit.full_restoration_deviance,
        "full_restoration_q": fit.full_restoration_q,
        "full_restoration_nominal_sigma": (
            fit.full_restoration_nominal_sigma
        ),
        "phases_hat_rad": list(fit.phases_hat),
        "full_restoration_phases_rad": list(
            fit.full_restoration_phases
        ),
    }


def _dataset_record(
    dataset: DatasetQuadrature,
    *,
    phase_hat: float,
    phase_full: float,
    phase_full_finite_path: float,
    phase_full_launch_infinity: float,
    phase_full_launch_finite_path: float,
) -> dict[str, Any]:
    if (
        dataset.transmission is None
        or dataset.normalization_Rpm_cps is None
        or dataset.normalization_Rfinite_path_cps is None
        or dataset.period_V is None
        or dataset.period_sigma_V is None
    ):
        raise PooledAnalysisError(
            f"{dataset.dataset_id}: pooled input metadata is incomplete"
        )
    transmission = dataset.transmission
    return {
        "dataset_id": dataset.dataset_id,
        "source": dataset.source,
        "period_V": dataset.period_V,
        "period_sigma_V": dataset.period_sigma_V,
        "q_hat_cps": dataset.q_hat.tolist(),
        "covariance_cps2": dataset.covariance.tolist(),
        "transmission_launch_optics": transmission.launch_optics,
        "transmission_atmosphere": transmission.atmosphere,
        "transmission_milky_way": transmission.milky_way,
        "transmission_Tfinite_path": (
            transmission.finite_path
        ),
        "transmission_Tinfinity": transmission.infinity,
        "normalization_Rpm_cps": dataset.normalization_Rpm_cps,
        "normalization_Rinfty_cps": dataset.normalization,
        "normalization_Rfinite_path_cps": (
            dataset.normalization_Rfinite_path_cps
        ),
        "normalization_launch_floor_Rinfinity_cps": (
            2.0 * dataset.normalization
        ),
        "normalization_launch_floor_Rfinite_path_cps": (
            2.0 * dataset.normalization_Rfinite_path_cps
        ),
        "phase_hat_rad": phase_hat,
        "phase_full_restoration_rad": phase_full,
        "phase_full_restoration_finite_path_rad": (
            phase_full_finite_path
        ),
        "phase_full_restoration_launch_infinity_rad": (
            phase_full_launch_infinity
        ),
        "phase_full_restoration_launch_finite_path_rad": (
            phase_full_launch_finite_path
        ),
    }


def build_result(
    datasets: Sequence[DatasetQuadrature],
    infinity: NormalizationInference,
    finite_path: NormalizationInference,
    launch_infinity: NormalizationInference,
    launch_finite_path: NormalizationInference,
    loo: Sequence[dict[str, Any]],
    *,
    coverage: dict[str, Any],
    goodness_of_fit: dict[str, Any],
    null_test: dict[str, Any],
    profile_cutoff: float,
) -> dict[str, Any]:
    fit = infinity.fit
    loo_limits = [float(record["eta_upper"]) for record in loo]
    transmissions = [dataset.transmission for dataset in datasets]
    if any(value is None for value in transmissions):
        raise PooledAnalysisError("result inputs are missing transmissions")
    physical_transmissions = [
        value for value in transmissions if value is not None
    ]
    if any(
        dataset.period_V is None or dataset.period_sigma_V is None
        for dataset in datasets
    ):
        raise PooledAnalysisError("result inputs are missing period metadata")
    periods = [
        float(dataset.period_V)
        for dataset in datasets
        if dataset.period_V is not None
    ]
    period_sigmas = [
        float(dataset.period_sigma_V)
        for dataset in datasets
        if dataset.period_sigma_V is not None
    ]
    common_igm = _common_igm_transmission(datasets)
    f_critical = infinity.eta_upper
    inverse_f_critical = 1.0 / f_critical
    additional_optical_depth = -math.log(f_critical)
    critical_igm = common_igm * f_critical
    critical_igm_optical_depth = -math.log(critical_igm)
    nominal_igm_optical_depth = -math.log(common_igm)
    igm_optical_depth_ratio = (
        critical_igm_optical_depth / nominal_igm_optical_depth
    )
    return {
        "schema_version": 2,
        "method": {
            "model": "common_restoration_fraction_independent_dataset_phases",
            "likelihood": "joint_gaussian_dataset_quadratures",
            "normalization_treatment": "conditional_on_stated_dataset_normalizations",
            "eta_domain": [0.0, 1.0],
            "phase_domain_rad": [0.0, 2.0 * math.pi],
            "upper_limit_method": (
                "profile_likelihood_with_parametric_coverage_check"
            ),
            "full_restoration_test": "one_sided_profile_likelihood_ratio",
            "null_test": "parametric_monte_carlo_profile_likelihood_ratio",
            "likelihood_ratio_statistic": (
                "q=-2*log(lambda)=deviance_constrained-deviance_minimum"
            ),
            "launch_normalization_treatment": (
                "conservative_floor_twice_the_stated_detected_rate"
            ),
        },
        "inputs": [
            _dataset_record(
                dataset,
                phase_hat=phase_hat,
                phase_full=phase_full,
                phase_full_finite_path=phase_full_finite_path,
                phase_full_launch_infinity=phase_full_launch_inf,
                phase_full_launch_finite_path=phase_full_launch_finite_path,
            )
            for (
                dataset,
                phase_hat,
                phase_full,
                phase_full_finite_path,
                phase_full_launch_inf,
                phase_full_launch_finite_path,
            ) in zip(
                datasets,
                fit.phases_hat,
                fit.full_restoration_phases,
                finite_path.fit.full_restoration_phases,
                launch_infinity.fit.full_restoration_phases,
                launch_finite_path.fit.full_restoration_phases,
            )
        ],
        "normalizations": {
            "infinity": _inference_record(infinity),
            "finite_path": _inference_record(finite_path),
            "launch_floor_infinity": _inference_record(launch_infinity),
            "launch_floor_finite_path": _inference_record(
                launch_finite_path
            ),
        },
        "robustness": {
            "critical_common_transmission_scale": f_critical,
            "nominal_to_critical_scale_ratio": inverse_f_critical,
            "additional_common_optical_depth": additional_optical_depth,
            "critical_igm_transmission": critical_igm,
            "critical_igm_optical_depth": critical_igm_optical_depth,
            "nominal_igm_optical_depth": nominal_igm_optical_depth,
            "critical_to_nominal_igm_optical_depth_ratio": (
                igm_optical_depth_ratio
            ),
        },
        "summary": {
            "n_datasets": len(datasets),
            "period_V_min": min(periods),
            "period_V_max": max(periods),
            "period_sigma_V_min": min(period_sigmas),
            "period_sigma_V_max": max(period_sigmas),
            "common_igm_transmission": common_igm,
            "T_infinity_min": min(
                value.infinity for value in physical_transmissions
            ),
            "T_infinity_max": max(
                value.infinity for value in physical_transmissions
            ),
            "T_finite_path_min": min(
                value.finite_path
                for value in physical_transmissions
            ),
            "T_finite_path_max": max(
                value.finite_path
                for value in physical_transmissions
            ),
            "launch_optics_transmission": physical_transmissions[
                0
            ].launch_optics,
            "atmosphere_transmission_min": min(
                value.atmosphere for value in physical_transmissions
            ),
            "atmosphere_transmission_max": max(
                value.atmosphere for value in physical_transmissions
            ),
            "milky_way_transmission_min": min(
                value.milky_way for value in physical_transmissions
            ),
            "milky_way_transmission_max": max(
                value.milky_way for value in physical_transmissions
            ),
            "eta_hat": fit.eta_hat,
            "confidence": fit.confidence,
            "nominal_profile_cutoff": fit.nominal_cutoff,
            "eta_upper_nominal": fit.eta_upper_nominal,
            "profile_cutoff": profile_cutoff,
            "profile_cutoff_calibrated": bool(
                profile_cutoff > fit.nominal_cutoff
            ),
            "eta_upper": infinity.eta_upper,
            "upper_limit_truncated_at_eta_one": (
                infinity.upper_limit_truncated
            ),
            "deviance_min": fit.deviance_min,
            "nominal_gof_df": fit.nominal_gof_df,
            "nominal_gof_pvalue": fit.nominal_gof_pvalue,
            "monte_carlo_gof_pvalue": goodness_of_fit["pvalue"],
            "heterogeneity_monte_carlo_pvalue": goodness_of_fit["pvalue"],
            "null_deviance": null_test["observed_null_deviance"],
            "null_q": null_test["observed_q"],
            "null_monte_carlo_pvalue": null_test["pvalue"],
            "full_restoration_deviance": fit.full_restoration_deviance,
            "full_restoration_q": fit.full_restoration_q,
            "full_restoration_nominal_sigma": (
                fit.full_restoration_nominal_sigma
            ),
            "eta_hat_finite_path": (
                finite_path.fit.eta_hat
            ),
            "eta_upper_finite_path": (
                finite_path.eta_upper
            ),
            "finite_path_full_restoration_deviance": (
                finite_path.fit.full_restoration_deviance
            ),
            "finite_path_full_restoration_q": (
                finite_path.fit.full_restoration_q
            ),
            "finite_path_full_restoration_nominal_sigma": (
                finite_path.fit.full_restoration_nominal_sigma
            ),
            "eta_upper_launch_infinity": launch_infinity.eta_upper,
            "launch_infinity_full_restoration_deviance": (
                launch_infinity.fit.full_restoration_deviance
            ),
            "launch_infinity_full_restoration_q": (
                launch_infinity.fit.full_restoration_q
            ),
            "launch_infinity_full_restoration_nominal_sigma": (
                launch_infinity.fit.full_restoration_nominal_sigma
            ),
            "eta_upper_launch_finite_path": (
                launch_finite_path.eta_upper
            ),
            "launch_finite_path_full_restoration_deviance": (
                launch_finite_path.fit.full_restoration_deviance
            ),
            "launch_finite_path_full_restoration_q": (
                launch_finite_path.fit.full_restoration_q
            ),
            "launch_finite_path_full_restoration_nominal_sigma": (
                launch_finite_path.fit.full_restoration_nominal_sigma
            ),
            "critical_common_transmission_scale": f_critical,
            "nominal_to_critical_scale_ratio": inverse_f_critical,
            "additional_common_optical_depth": additional_optical_depth,
            "critical_igm_transmission": critical_igm,
            "critical_igm_optical_depth": critical_igm_optical_depth,
            "nominal_igm_optical_depth": nominal_igm_optical_depth,
            "critical_to_nominal_igm_optical_depth_ratio": (
                igm_optical_depth_ratio
            ),
            "leave_one_out_eta_upper_min": min(loo_limits),
            "leave_one_out_eta_upper_max": max(loo_limits),
        },
        "coverage": coverage,
        "goodness_of_fit": goodness_of_fit,
        "null_test": null_test,
        "leave_one_out": list(loo),
    }


def _write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_eta_grid(value: str) -> list[float]:
    try:
        result = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "coverage eta grid must be comma-separated numbers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("coverage eta grid must not be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pool launch-only quadratures across all configured datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX)
    parser.add_argument(
        "--coverage-simulations",
        type=int,
        default=DEFAULT_COVERAGE_SIMULATIONS,
    )
    parser.add_argument(
        "--coverage-eta-grid",
        type=_parse_eta_grid,
        default=list(DEFAULT_COVERAGE_ETA_GRID),
    )
    parser.add_argument(
        "--random-phase-configurations",
        type=int,
        default=DEFAULT_RANDOM_PHASE_CONFIGURATIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    datasets = load_dataset_quadratures(load_config(), args.source_root)
    fit = fit_pooled(
        datasets, confidence=args.confidence, eta_max=args.eta_max
    )
    phase_configurations = build_phase_configurations(
        datasets,
        fit,
        random_count=args.random_phase_configurations,
        seed=args.seed + 1,
    )
    coverage = simulate_upper_limit_coverage(
        datasets,
        confidence=args.confidence,
        nominal_cutoff=fit.nominal_cutoff,
        eta_grid=args.coverage_eta_grid,
        phase_configurations=phase_configurations,
        simulations=args.coverage_simulations,
        seed=args.seed + 2,
        eta_max=args.eta_max,
    )
    goodness_of_fit = simulate_common_effect_gof(
        datasets,
        fit,
        simulations=args.coverage_simulations,
        seed=args.seed + 3,
        eta_max=args.eta_max,
    )
    null_test = simulate_null_test(
        datasets,
        fit,
        simulations=args.coverage_simulations,
        seed=args.seed + 4,
        eta_max=args.eta_max,
    )
    selected_cutoff = float(coverage["recommended_cutoff"])
    eta_upper, upper_truncated = upper_limit_for_cutoff(
        datasets,
        eta_hat=fit.eta_hat,
        deviance_min=fit.deviance_min,
        cutoff=selected_cutoff,
        eta_max=args.eta_max,
    )
    infinity = NormalizationInference(
        name="infinity",
        datasets=tuple(datasets),
        fit=fit,
        eta_upper=eta_upper,
        profile_cutoff=selected_cutoff,
        upper_limit_truncated=upper_truncated,
    )
    finite_path = infer_normalization(
        normalization_view(
            datasets, name="finite_path", finite_path=True
        ),
        name="finite_path",
        profile_cutoff=selected_cutoff,
        confidence=args.confidence,
        eta_max=args.eta_max,
    )
    launch_infinity = infer_normalization(
        normalization_view(
            datasets, name="launch_floor_infinity", multiplier=2.0
        ),
        name="launch_floor_infinity",
        profile_cutoff=selected_cutoff,
        confidence=args.confidence,
        eta_max=args.eta_max,
    )
    launch_finite_path = infer_normalization(
        normalization_view(
            datasets,
            name="launch_floor_finite_path",
            finite_path=True,
            multiplier=2.0,
        ),
        name="launch_floor_finite_path",
        profile_cutoff=selected_cutoff,
        confidence=args.confidence,
        eta_max=args.eta_max,
    )
    validate_common_reparameterization(
        infinity,
        finite_path,
        _common_igm_transmission(datasets),
    )
    for name, actual, expected in (
        (
            "infinity launched endpoint",
            launch_infinity.eta_upper,
            0.5 * infinity.eta_upper,
        ),
        (
            "finite-path launched endpoint",
            launch_finite_path.eta_upper,
            0.5 * finite_path.eta_upper,
        ),
    ):
        if not math.isclose(
            actual, expected, rel_tol=2e-9, abs_tol=2e-9
        ):
            raise PooledAnalysisError(
                f"common normalization scaling failed for {name}"
            )
    loo = leave_one_out(
        datasets,
        confidence=args.confidence,
        eta_max=args.eta_max,
        cutoff=selected_cutoff,
    )
    result = build_result(
        datasets,
        infinity,
        finite_path,
        launch_infinity,
        launch_finite_path,
        loo,
        coverage=coverage,
        goodness_of_fit=goodness_of_fit,
        null_test=null_test,
        profile_cutoff=selected_cutoff,
    )
    _write_json_atomically(args.json_output, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PooledAnalysisError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
