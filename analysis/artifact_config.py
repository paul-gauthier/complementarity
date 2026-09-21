"""Load and validate the repository's canonical dataset configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "artifact.json"
RAW_FILENAMES = ("plan.json", "points.jsonl", "dark.json", "meta.json")
GOES_SOURCE_PRODUCTS = {
    "goes_aod": "ABI-L2-AODC",
    "goes_cod": "ABI-L2-CODC",
    "goes_tpw": "ABI-L2-TPWC",
}

GLOBAL_OUTPUTS = (
    "README.md",
    "analysis/igm-transmission.json",
    "analysis/mzi-pooled-analysis.json",
    "figures/mzi-normalized-quadratures.pdf",
    "figures/mzi-normalized-quadratures.png",
    "manuscript-values/pooled-analysis-values.tex",
    "manuscript-values/igm-transmission-values.tex",
    "tables/dataset-transmission.tex",
    "tables/dataset-transmission.png",
    "tables/dataset-analysis.tex",
    "tables/dataset-analysis.png",
)
DATASET_OUTPUTS = (
    "README.md",
    "analysis/mzi-null-bounds.json",
    "analysis/mzi-alt-joint.json",
    "analysis/darks-accidentals.json",
    "figures/mzi-alt-by-pass.pdf",
    "figures/mzi-alt-by-pass.png",
    "figures/mzi-null-plot.pdf",
    "figures/mzi-null-plot.png",
    "tables/mzi-alt-by-pass-run-1.tex",
    "tables/mzi-alt-by-pass-run-1.png",
    "tables/mzi-alt-by-pass-run-2.tex",
    "tables/mzi-alt-by-pass-run-2.png",
    "tables/mzi-alt-joint.tex",
    "tables/mzi-alt-joint.png",
    "tables/dark-rates.tex",
    "tables/dark-rates.png",
    "tables/accidental-corrections.tex",
    "tables/accidental-corrections.png",
)
CONDITIONS_OUTPUTS = (
    "conditions/conditions.json",
    "conditions/conditions.txt",
    "manuscript-values/analysis-values.tex",
    "manuscript-values/conditions-values.tex",
)


class ConfigError(ValueError):
    """Raised when artifact.json is missing, unsafe, or inconsistent."""


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ConfigError(f"{field} is not a safe repository-relative path: {value!r}")
    return value


def dataset_runs(config: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    acquisitions = config["acquisitions"]
    launch = acquisitions[dataset["launch_run"]]
    preserve = acquisitions[dataset["preserve_run"]]
    return [
        {
            "id": "run-1-erase", "run": "1", "run_name": "launch-erase",
            "condition": "Erase", "path": f"{launch}/lab", "detected_idler": True,
        },
        {
            "id": "run-1-launch", "run": "1", "run_name": "launch-erase",
            "condition": "Launch", "path": f"{launch}/launch", "detected_idler": False,
        },
        {
            "id": "run-2-erase", "run": "2", "run_name": "preserve-erase",
            "condition": "Erase", "path": f"{preserve}/lab", "detected_idler": True,
        },
        {
            "id": "run-2-preserve", "run": "2", "run_name": "preserve-erase",
            "condition": "Preserve", "path": f"{preserve}/launch", "detected_idler": True,
        },
    ]


def canonical_outputs(config: dict[str, Any]) -> list[str]:
    outputs: list[str] = list(GLOBAL_OUTPUTS)
    for dataset in config["datasets"]:
        prefix = f"datasets/{dataset['id']}"
        outputs.extend(f"{prefix}/{relative}" for relative in DATASET_OUTPUTS)
        outputs.extend(f"{prefix}/{relative}" for relative in CONDITIONS_OUTPUTS)
    return outputs


def get_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [dataset for dataset in config["datasets"] if dataset["id"] == dataset_id]
    if len(matches) != 1:
        raise ConfigError(f"unknown or duplicate dataset id: {dataset_id!r}")
    return matches[0]


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {source}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 10:
        raise ConfigError("artifact configuration schema_version must be 10")
    for key in (
        "acquisitions", "datasets", "conditions_profiles", "null_bounds",
        "transmission_analysis", "igm_transmission", "pooled_analysis",
    ):
        if key not in config:
            raise ConfigError(f"artifact configuration is missing {key!r}")

    acquisitions = config["acquisitions"]
    if not isinstance(acquisitions, dict) or not acquisitions:
        raise ConfigError("acquisitions must be a non-empty object")
    for timestamp, path in acquisitions.items():
        if not isinstance(timestamp, str) or not timestamp:
            raise ConfigError("acquisition timestamps must be non-empty strings")
        _relative_path(path, f"acquisitions.{timestamp}")

    transmission_analysis = config["transmission_analysis"]
    if not isinstance(transmission_analysis, dict):
        raise ConfigError("transmission_analysis must be an object")
    if set(transmission_analysis) != {"milky_way_basis"}:
        raise ConfigError(
            "transmission_analysis must contain exactly 'milky_way_basis'"
        )
    if transmission_analysis["milky_way_basis"] not in {
        "point", "integrated", "minimum"
    }:
        raise ConfigError(
            "transmission_analysis.milky_way_basis must be 'point', "
            "'integrated', or 'minimum'"
        )

    igm_transmission = config["igm_transmission"]
    expected_igm_input_keys = {
        "hubble_constant_km_s_mpc",
        "omega_m",
        "omega_lambda",
        "idler_wavelength_nm",
        "dust_visual_extinction_mag",
        "dust_normalization_distance_gpc",
        "dust_r_v",
        "electron_density_cm3",
        "thomson_cross_section_cm2",
        "galaxy_number_density_h3_mpc3",
        "galaxy_radius_hinv_pc",
    }
    expected_igm_reference_keys = {
        "cosmology",
        "dust_normalization",
        "dust_extinction_curve",
        "galaxy_interception",
    }
    if (
        not isinstance(igm_transmission, dict)
        or set(igm_transmission) != expected_igm_input_keys | {"references"}
    ):
        raise ConfigError(
            "igm_transmission must contain exactly the configured cosmology, "
            "dust, electron, and galaxy inputs plus references"
        )
    for key in expected_igm_input_keys:
        value = igm_transmission[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConfigError(f"igm_transmission.{key} must be finite and positive")
    references = igm_transmission["references"]
    if (
        not isinstance(references, dict)
        or set(references) != expected_igm_reference_keys
    ):
        raise ConfigError(
            "igm_transmission.references must contain exactly cosmology, "
            "dust_normalization, dust_extinction_curve, and galaxy_interception"
        )
    for key in expected_igm_reference_keys:
        value = references[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"igm_transmission.references.{key} must be a non-empty string"
            )
    if not math.isclose(
        float(igm_transmission["omega_m"])
        + float(igm_transmission["omega_lambda"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ConfigError(
            "igm_transmission density parameters must describe a flat model"
        )
    if float(igm_transmission["idler_wavelength_nm"]) < 1000.0 / 3.3:
        raise ConfigError(
            "igm_transmission.idler_wavelength_nm is outside the implemented "
            "Cardelli optical/near-IR range"
        )

    pooled_analysis = config["pooled_analysis"]
    if not isinstance(pooled_analysis, dict) or set(pooled_analysis) != {
        "confidence",
        "eta_max",
        "coverage_simulations",
        "coverage_eta_grid",
        "random_phase_configurations",
    }:
        raise ConfigError(
            "pooled_analysis must contain exactly confidence, eta_max, "
            "coverage_simulations, coverage_eta_grid, and "
            "random_phase_configurations"
        )
    confidence = pooled_analysis["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.5 < float(confidence) < 1.0
    ):
        raise ConfigError("pooled_analysis.confidence must be between 0.5 and 1")
    eta_max = pooled_analysis["eta_max"]
    if (
        isinstance(eta_max, bool)
        or not isinstance(eta_max, (int, float))
        or not math.isfinite(float(eta_max))
        or float(eta_max) != 1.0
    ):
        raise ConfigError("pooled_analysis.eta_max must equal 1.0")
    simulations = pooled_analysis["coverage_simulations"]
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations < 100
    ):
        raise ConfigError(
            "pooled_analysis.coverage_simulations must be an integer of at least 100"
        )
    random_phases = pooled_analysis["random_phase_configurations"]
    if (
        isinstance(random_phases, bool)
        or not isinstance(random_phases, int)
        or random_phases < 0
    ):
        raise ConfigError(
            "pooled_analysis.random_phase_configurations must be a nonnegative integer"
        )
    eta_grid = pooled_analysis["coverage_eta_grid"]
    if not isinstance(eta_grid, list) or not eta_grid:
        raise ConfigError("pooled_analysis.coverage_eta_grid must be a nonempty array")
    normalized_grid: list[float] = []
    for value in eta_grid:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= float(eta_max)
        ):
            raise ConfigError(
                "pooled_analysis.coverage_eta_grid values must be finite and "
                "within [0, eta_max]"
            )
        normalized_grid.append(float(value))
    if normalized_grid != sorted(set(normalized_grid)):
        raise ConfigError(
            "pooled_analysis.coverage_eta_grid must be strictly increasing"
        )
    if normalized_grid[0] != 0.0 or normalized_grid[-1] != float(eta_max):
        raise ConfigError(
            "pooled_analysis.coverage_eta_grid must span zero through eta_max"
        )

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("datasets must be a non-empty array")
    dataset_ids: set[str] = set()
    display_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ConfigError(f"datasets[{index}] must be an object")
        for key in (
            "id", "display_id", "launch_run", "preserve_run", "visual_conditions"
        ):
            if key not in dataset:
                raise ConfigError(f"datasets[{index}] is missing {key!r}")
        if dataset["id"] in dataset_ids:
            raise ConfigError(f"duplicate dataset id: {dataset['id']!r}")
        dataset_ids.add(dataset["id"])
        display_id = dataset["display_id"]
        if not isinstance(display_id, str) or not display_id:
            raise ConfigError(
                f"datasets[{index}].display_id must be a non-empty string"
            )
        if display_id in display_ids:
            raise ConfigError(f"duplicate dataset display_id: {display_id!r}")
        display_ids.add(display_id)
        for role in ("launch_run", "preserve_run"):
            if dataset[role] not in acquisitions:
                raise ConfigError(
                    f"datasets[{index}].{role} references unknown acquisition "
                    f"{dataset[role]!r}"
                )
        profile = dataset.get("conditions_profile")
        if not isinstance(profile, str) or not profile:
            raise ConfigError(f"datasets[{index}] is missing 'conditions_profile'")
        if profile not in config["conditions_profiles"]:
            raise ConfigError(f"unknown conditions profile: {profile!r}")
        if config["conditions_profiles"][profile].get("target_timestamp") != dataset["launch_run"]:
            raise ConfigError(
                f"datasets[{index}] conditions target must equal its launch_run"
            )

    referenced_profiles = [dataset["conditions_profile"] for dataset in datasets]
    if len(referenced_profiles) != len(set(referenced_profiles)):
        raise ConfigError("each dataset must reference a unique conditions profile")
    if set(referenced_profiles) != set(config["conditions_profiles"]):
        raise ConfigError("every conditions profile must be referenced by exactly one dataset")
    expected_profile_keys = {
        "target_timestamp",
        "integration_minutes",
        "aeronet_variant",
        "atmosphere_source",
        "pull_prefixes",
        "sources",
    }
    for name, profile in config["conditions_profiles"].items():
        if not isinstance(profile, dict) or set(profile) != expected_profile_keys:
            raise ConfigError(
                f"conditions_profiles.{name} must contain exactly "
                + ", ".join(sorted(expected_profile_keys))
            )
        if profile["aeronet_variant"] not in {"solar", "lunar"}:
            raise ConfigError(
                f"conditions_profiles.{name}.aeronet_variant must be 'solar' or 'lunar'"
            )
        if profile["atmosphere_source"] not in {"goes", "aeronet"}:
            raise ConfigError(
                f"conditions_profiles.{name}.atmosphere_source must be 'goes' or 'aeronet'"
            )
        if (
            isinstance(profile["integration_minutes"], bool)
            or not isinstance(profile["integration_minutes"], int)
            or profile["integration_minutes"] <= 0
        ):
            raise ConfigError(
                f"conditions_profiles.{name}.integration_minutes must be a positive integer"
            )
        uses_goes = profile["atmosphere_source"] == "goes"
        expected_prefixes = {"metar", "aeronet", "irsa"}
        if uses_goes:
            expected_prefixes.add("goes")
        prefixes = profile["pull_prefixes"]
        if not isinstance(prefixes, dict) or set(prefixes) != expected_prefixes:
            raise ConfigError(
                f"conditions_profiles.{name}.pull_prefixes must contain exactly "
                + ", ".join(sorted(expected_prefixes))
            )
        for source_name, prefix in prefixes.items():
            if (
                not isinstance(prefix, str)
                or not prefix
                or PurePosixPath(prefix).name != prefix
            ):
                raise ConfigError(
                    f"conditions_profiles.{name}.pull_prefixes.{source_name} "
                    "must be a filename prefix"
                )
        expected_sources = {
            "metar", "aeronet", "irsa_json", "irsa_csv", "irsa_ecsv"
        }
        if uses_goes:
            expected_sources.update({"goes_aod", "goes_cod", "goes_tpw"})
        sources = profile["sources"]
        if not isinstance(sources, dict) or set(sources) != expected_sources:
            raise ConfigError(
                f"conditions_profiles.{name}.sources must contain exactly "
                + ", ".join(sorted(expected_sources))
            )
        for source_name, value in profile["sources"].items():
            field = f"conditions_profiles.{name}.sources.{source_name}"
            relative = _relative_path(value, field)
            if source_name in GOES_SOURCE_PRODUCTS:
                product = GOES_SOURCE_PRODUCTS[source_name]
                if product not in PurePosixPath(relative).name:
                    raise ConfigError(f"{field} must identify a {product} file")

    outputs = canonical_outputs(config)
    if len(outputs) != len(set(outputs)):
        raise ConfigError("derived canonical output paths are not unique")
    return config


def resolve_repository_path(value: str, root: Path = REPOSITORY_ROOT) -> Path:
    relative = _relative_path(value, "path")
    resolved = (root.resolve() / relative).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise ConfigError(f"path escapes repository: {value!r}")
    return resolved


def validate_raw_inputs(config: dict[str, Any], root: Path = REPOSITORY_ROOT) -> None:
    from .mzi_io import validate_mzi_run

    problems: list[str] = []
    for base in config["acquisitions"].values():
        for destination in ("lab", "launch"):
            directory = resolve_repository_path(f"{base}/{destination}", root)
            absent = [
                str(directory / name)
                for name in RAW_FILENAMES
                if not (directory / name).is_file()
            ]
            problems.extend(absent)
            if not absent:
                try:
                    validate_mzi_run(directory)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    problems.append(f"{directory} (invalid MZI input: {exc})")
    for profile in config["conditions_profiles"].values():
        for value in profile["sources"].values():
            path = resolve_repository_path(value, root)
            if not path.is_file():
                problems.append(str(path))
        irsa_json = resolve_repository_path(profile["sources"]["irsa_json"], root)
        if irsa_json.is_file():
            try:
                payload = json.loads(irsa_json.read_text(encoding="utf-8"))
                integration = payload["integration"]
                samples = integration["samples"]
                if (
                    integration["minutes"] != profile["integration_minutes"]
                    or integration["n_attempted"] != len(samples)
                    or integration["n_success"] != len(samples)
                    or integration["n_failed"] != 0
                    or not integration["available"]
                ):
                    raise ValueError("incomplete integration metadata")
                for sample in samples:
                    table = irsa_json.parent / sample["table_file"]
                    if not table.is_file():
                        problems.append(str(table))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"{irsa_json} (invalid IRSA integration: {exc})")
    if problems:
        raise ConfigError("invalid configured raw inputs:\n  " + "\n  ".join(problems))
