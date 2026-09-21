"""Read-only validation of a local build against reviewed results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

from analysis.artifact_config import canonical_outputs, load_config, validate_raw_inputs
from analysis.generate_conditions_values import validate as validate_conditions
from analysis.igm_transmission import build_record, calculate_igm_transmission
from analysis.transmission_values import (
    GRAY_DUST_IGM_TRANSMISSION,
    IGM_TRANSMISSION,
    calculate_transmissions,
)

ROOT = Path(__file__).resolve().parent.parent


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def json_equivalent(expected: Any, actual: Any) -> bool:
    """Compare JSON structures while allowing insignificant float roundoff."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            json_equivalent(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            json_equivalent(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    return type(expected) is type(actual) and expected == actual


def pdf_page_size(path: Path) -> tuple[float, float]:
    match = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        path.read_bytes(),
    )
    if match is None:
        raise ValueError("PDF has no readable MediaBox")
    return float(match.group(1)), float(match.group(2))


def verify_checksums() -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate((ROOT / "data/checksums.sha256").read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed checksum line {line_number}")
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing raw input: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"raw checksum mismatch: {relative}")
    return errors


def verify_results_checksums(results: Path, outputs: list[str]) -> list[str]:
    """Validate the reviewed-results manifest and canonical file inventory."""
    errors: list[str] = []
    expected_outputs = set(outputs)
    manifest = results / "checksums.sha256"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read reviewed-results checksum manifest: {exc}"]

    recorded: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(
                f"malformed reviewed-results checksum line {line_number}"
            )
            continue
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(
                f"invalid reviewed-results checksum on line {line_number}"
            )
            continue
        if relative in recorded:
            errors.append(
                f"duplicate reviewed-results checksum entry: {relative}"
            )
            continue
        recorded[relative] = digest

    recorded_outputs = set(recorded)
    for relative in sorted(expected_outputs - recorded_outputs):
        errors.append(f"missing reviewed-results checksum entry: {relative}")
    for relative in sorted(recorded_outputs - expected_outputs):
        errors.append(f"unexpected reviewed-results checksum entry: {relative}")

    actual_outputs = {
        path.relative_to(results).as_posix()
        for path in results.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(results).parts
        )
        and path.relative_to(results).as_posix()
        not in {"index.md", "checksums.sha256"}
    }
    for relative in sorted(expected_outputs - actual_outputs):
        errors.append(f"missing reviewed result: {relative}")
    for relative in sorted(actual_outputs - expected_outputs):
        errors.append(f"unexpected reviewed result: {relative}")

    for relative in sorted(expected_outputs & recorded_outputs & actual_outputs):
        actual = hashlib.sha256((results / relative).read_bytes()).hexdigest()
        if actual != recorded[relative]:
            errors.append(f"reviewed-results checksum mismatch: {relative}")
    return errors


def compare(build: Path, results: Path, outputs: list[str]) -> list[str]:
    errors: list[str] = []
    expected = set(outputs)
    for relative in outputs:
        left, right = build / relative, results / relative
        if not left.is_file():
            errors.append(f"missing build product: {relative}")
            continue
        if not right.is_file():
            errors.append(f"missing reference product: {relative}")
            continue
        if not left.stat().st_size or not right.stat().st_size:
            errors.append(f"empty product: {relative}")
            continue
        if left.suffix == ".json":
            try:
                if load_json(left) != load_json(right):
                    errors.append(f"different JSON product: {relative}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON product {relative}: {exc}")
        elif left.suffix in {".csv", ".tex", ".md"} and left.read_bytes() != right.read_bytes():
            errors.append(f"different text product: {relative}")
        elif left.suffix in {".png", ".pdf"}:
            signature = left.read_bytes()[:8]
            if left.suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
                errors.append(f"invalid PNG product: {relative}")
            elif left.suffix == ".png":
                left_size = struct.unpack(">II", left.read_bytes()[16:24])
                right_size = struct.unpack(">II", right.read_bytes()[16:24])
                if left_size != right_size:
                    errors.append(
                        f"different PNG dimensions: {relative} "
                        f"({left_size} != {right_size})"
                    )
            if left.suffix == ".pdf" and not signature.startswith(b"%PDF-"):
                errors.append(f"invalid PDF product: {relative}")
    for tree, label in ((build, "build"), (results, "reference")):
        actual = {
            path.relative_to(tree).as_posix()
            for path in tree.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(tree).parts)
            and path.relative_to(tree).as_posix()
            not in {"index.md", "checksums.sha256"}
        }
        for relative in sorted(actual - expected):
            errors.append(f"unexpected {label} product: {relative}")
    return errors


def verify_dark_analysis(darks: Any) -> list[str]:
    """Validate the published dark-analysis record and coincidence-window contract."""
    errors: list[str] = []
    if not isinstance(darks, dict):
        return ["dark analysis must be an object"]
    if set(darks) != {
        "channel_mapping", "coincidence_window", "correction_model", "runs",
        "schema_version", "terminology",
    }:
        errors.append("dark analysis fields do not match schema version 5")
    if darks.get("schema_version") != 5:
        errors.append("dark analysis schema_version must be 5")

    window = darks.get("coincidence_window")
    if not isinstance(window, dict):
        errors.append("dark analysis coincidence_window must be an object")
    else:
        if set(window) != {"convention", "input", "validation", "width_ns", "width_s"}:
            errors.append("dark analysis coincidence_window fields do not match schema version 5")
        if window.get("input") != {
            "file": "dark.json",
            "field": "w_s",
            "location": "top-level",
            "unit": "s",
        }:
            errors.append("dark analysis has a noncanonical coincidence-window input")
        if window.get("validation") != {
            "required_for_every_run": True,
            "matched_canonical_width": True,
        }:
            errors.append("dark analysis coincidence-window validation is incomplete")

    runs = darks.get("runs")
    if not isinstance(runs, list):
        errors.append("dark analysis runs must be a list")
        return errors
    if len(runs) != 4:
        errors.append("dark analysis must contain four runs")
    for index, run in enumerate(runs):
        context = f"dark analysis runs[{index}]"
        if not isinstance(run, dict):
            errors.append(f"{context} must be an object")
            continue
        if set(run) != {
            "condition", "cross_check", "dark_run_background_measurements",
            "detected_idler", "diagnostics", "directory", "inputs", "plan",
            "point_count", "rate_budget", "run", "run_name", "total_exposure_s",
        }:
            errors.append(f"{context} fields do not match schema version 5")
        plan = run.get("plan")
        if not isinstance(plan, dict):
            errors.append(f"{context}.plan must be an object")
        elif set(plan) != {
            "planned_voltage_max_V", "planned_voltage_min_V", "points_per_scan",
            "scan_count",
        }:
            errors.append(f"{context}.plan fields do not match schema version 5")
    return errors


def verify_invariants(build: Path, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        igm_result = calculate_igm_transmission(config["igm_transmission"])
        igm_record = load_json(build / "analysis/igm-transmission.json")
        if not json_equivalent(
            build_record(config["igm_transmission"], igm_result),
            igm_record,
        ):
            errors.append("IGM transmission JSON does not match configured inputs")
        if not math.isclose(
            IGM_TRANSMISSION,
            igm_result.nominal_transmission,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            errors.append("nominal IGM transmission alias is inconsistent")
        if not math.isclose(
            GRAY_DUST_IGM_TRANSMISSION,
            igm_result.gray_transmission,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            errors.append("gray IGM transmission alias is inconsistent")
        igm_macros = (
            build / "manuscript-values/igm-transmission-values.tex"
        ).read_text(encoding="utf-8")
        for macro in (
            r"\IgmDensityKernelIntegral",
            r"\IgmDustKernelIntegral",
            r"\IgmDustOpticalDepth",
            r"\IgmElectronOpticalDepth",
            r"\IgmGalaxyOpticalDepth",
            r"\IgmNominalOpticalDepth",
            r"\IgmNominalTransmission",
            r"\IgmGrayTransmission",
        ):
            if rf"\newcommand{{{macro}}}" not in igm_macros:
                errors.append(f"IGM manuscript values are missing {macro}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"IGM transmission invariant failure: {exc}")
    exact_transmissions: dict[str, Any] = {}
    for dataset in config["datasets"]:
        prefix = build / "datasets" / dataset["id"]
        try:
            bounds = load_json(prefix / "analysis/mzi-null-bounds.json")
            joint = load_json(prefix / "analysis/mzi-alt-joint.json")
            darks = load_json(prefix / "analysis/darks-accidentals.json")
            conditions = load_json(prefix / "conditions/conditions.json")
            period = float(bounds["summary"]["period_V"])
            if not math.isfinite(period) or period <= 0:
                errors.append(f"{dataset['id']}: period_V must be finite and positive")
            if not joint.get("joint_fits"):
                errors.append(f"{dataset['id']}: joint fit collection is empty")
            errors.extend(
                f"{dataset['id']}: {error}" for error in verify_dark_analysis(darks)
            )
            profile = config["conditions_profiles"][dataset["conditions_profile"]]
            if conditions["target"]["dirname"] != dataset["launch_run"]:
                errors.append(f"{dataset['id']}: conditions target is not the launch run")
            validate_conditions(conditions, profile["atmosphere_source"])
            records = conditions.get("libradtran", [])
            tags = [
                record.get("tag") if isinstance(record, dict) else None
                for record in records
            ]
            expected_aeronet = f"AERONET-{profile['aeronet_variant']}"
            goes = conditions.get("goes", {})
            goes_aod = goes.get("aod", {}) if isinstance(goes, dict) else {}
            goes_tpw = goes.get("tpw", {}) if isinstance(goes, dict) else {}

            def finite_number(value: object) -> bool:
                return (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                )

            goes_inputs_usable = (
                isinstance(goes, dict)
                and isinstance(goes_aod, dict)
                and goes_aod.get("usable") is True
                and finite_number(goes_aod.get("value"))
                and goes_aod.get("alpha_usable") is True
                and finite_number(goes_aod.get("alpha"))
                and isinstance(goes_tpw, dict)
                and goes_tpw.get("usable") is True
                and finite_number(goes_tpw.get("value"))
            )
            if profile["atmosphere_source"] == "goes":
                if not goes_inputs_usable:
                    errors.append(
                        f"{dataset['id']}: selected GOES atmosphere lacks usable "
                        "AOD, Angstrom exponent, or TPW"
                    )
                expected_tags = ["GOES-18", expected_aeronet]
            else:
                expected_tags = [expected_aeronet]
                if "goes" in conditions:
                    errors.append(
                        f"{dataset['id']}: AERONET atmosphere unexpectedly contains GOES data"
                    )
            if tags != expected_tags:
                errors.append(
                    f"{dataset['id']}: expected libRadtran records {expected_tags}; "
                    f"got {tags}"
                )
            atmosphere_tag = "GOES-18" if profile["atmosphere_source"] == "goes" else expected_aeronet
            records_by_tag = {
                record["tag"]: record
                for record in conditions.get("libradtran", [])
                if isinstance(record, dict) and isinstance(record.get("tag"), str)
            }
            exact = calculate_transmissions(
                atmosphere=float(records_by_tag[atmosphere_tag]["result"]["t_band"]),
                aeronet_atmosphere=float(
                    records_by_tag[expected_aeronet]["result"]["t_band"]
                ),
                milky_way_point=float(conditions["irsa"]["transmission_fraction"]),
                milky_way_integrated=float(conditions["irsa"]["integration"]["avg_T"]),
                milky_way_minimum=float(conditions["irsa"]["integration"]["min_T"]),
                milky_way_basis=config["transmission_analysis"]["milky_way_basis"],
            )
            exact_transmissions[dataset["id"]] = exact
            if not math.isclose(float(bounds["summary"]["Tinf"]), exact.infinity, rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{dataset['id']}: MZI Tinf does not match selected atmospheric source")
            conditions_tex = (
                prefix / "manuscript-values/conditions-values.tex"
            ).read_text(encoding="utf-8")
            defines_goes = r"\newcommand{\ConditionGoes" in conditions_tex
            if profile["atmosphere_source"] == "aeronet" and defines_goes:
                errors.append(
                    f"{dataset['id']}: AERONET atmosphere unexpectedly defines GOES TeX macros"
                )
            if profile["atmosphere_source"] == "goes" and not defines_goes:
                errors.append(
                    f"{dataset['id']}: GOES atmosphere is missing GOES TeX macros"
                )
            selected_atmosphere = float(
                records_by_tag[atmosphere_tag]["result"]["t_band"]
            )
            expected_atmosphere_macro = (
                rf"\newcommand{{\ConditionAtmosphericTransmission}}"
                rf"{{{selected_atmosphere:.4f}}}"
            )
            if expected_atmosphere_macro not in conditions_tex:
                errors.append(
                    f"{dataset['id']}: generic atmospheric TeX macro does not use "
                    f"{atmosphere_tag}"
                )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{dataset['id']}: cross-file invariant failure: {exc}")
    try:
        pooled_path = build / "analysis/mzi-pooled-analysis.json"
        pooled = load_json(pooled_path)
        if pooled.get("schema_version") != 2:
            errors.append("pooled analysis schema_version must be 2")
        if pooled.get("method", {}).get("likelihood_ratio_statistic") != (
            "q=-2*log(lambda)="
            "deviance_constrained-deviance_minimum"
        ):
            errors.append(
                "pooled likelihood-ratio statistic is not explicitly defined"
            )
        configured_ids = {dataset["id"] for dataset in config["datasets"]}
        input_records = {
            record["dataset_id"]: record for record in pooled["inputs"]
        }
        if set(input_records) != configured_ids:
            errors.append("pooled analysis inputs do not match configured datasets")
        for dataset in config["datasets"]:
            dataset_id = dataset["id"]
            bounds = load_json(
                build
                / "datasets"
                / dataset_id
                / "analysis/mzi-null-bounds.json"
            )
            summary = bounds["summary"]
            record = input_records[dataset_id]
            expected_values = {
                "period_V": float(summary["period_V"]),
                "period_sigma_V": float(summary["period_sigma_V"]),
                "C_hat": float(summary["CL_hat_cps"]),
                "S_hat": float(summary["SL_hat_cps"]),
                "C_variance": float(summary["CL_sigma_cps"]) ** 2,
                "CS_covariance": float(summary["CLSL_cov_cps2"]),
                "S_variance": float(summary["SL_sigma_cps"]) ** 2,
                "normalization": float(summary["Rinfty_cps"]),
                "T_infinity": exact_transmissions[dataset_id].infinity,
                "T_finite_path": exact_transmissions[
                    dataset_id
                ].finite_path,
                "R_finite_path": (
                    exact_transmissions[dataset_id].finite_path
                    * float(summary["Rpm_cps"])
                ),
            }
            actual_values = {
                "period_V": float(record["period_V"]),
                "period_sigma_V": float(record["period_sigma_V"]),
                "C_hat": float(record["q_hat_cps"][0]),
                "S_hat": float(record["q_hat_cps"][1]),
                "C_variance": float(record["covariance_cps2"][0][0]),
                "CS_covariance": float(record["covariance_cps2"][0][1]),
                "S_variance": float(record["covariance_cps2"][1][1]),
                "normalization": float(record["normalization_Rinfty_cps"]),
                "T_infinity": float(record["transmission_Tinfinity"]),
                "T_finite_path": float(
                    record["transmission_Tfinite_path"]
                ),
                "R_finite_path": float(
                    record["normalization_Rfinite_path_cps"]
                ),
            }
            for name, expected in expected_values.items():
                if not math.isclose(
                    actual_values[name], expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    errors.append(
                        f"{dataset_id}: pooled {name} does not match dataset-level fit"
                    )

        pooled_options = config["pooled_analysis"]
        coverage = pooled["coverage"]
        if coverage["simulations_per_scenario"] != pooled_options[
            "coverage_simulations"
        ]:
            errors.append("pooled coverage simulation count is not configured")
        if coverage["eta_grid"] != pooled_options["coverage_eta_grid"]:
            errors.append("pooled coverage eta grid is not configured")
        expected_scenarios = len(pooled_options["coverage_eta_grid"]) * (
            4 + pooled_options["random_phase_configurations"]
        )
        if coverage["n_scenarios"] != expected_scenarios:
            errors.append("pooled coverage scenario count is inconsistent")
        if coverage["minimum_nominal_coverage"] < pooled_options["confidence"]:
            errors.append("pooled nominal upper limit undercovers the configured grid")

        summary = pooled["summary"]
        if summary["n_datasets"] != len(config["datasets"]):
            errors.append("pooled dataset count is inconsistent")
        figure_pdf = build / "figures/mzi-normalized-quadratures.pdf"
        figure_png = build / "figures/mzi-normalized-quadratures.png"
        pdf_width, pdf_height = pdf_page_size(figure_pdf)
        if not math.isclose(pdf_width, 244.8, rel_tol=0.0, abs_tol=1e-6):
            errors.append("normalized quadrature PDF is not 3.4 inches wide")
        if pdf_height <= 0.0:
            errors.append("normalized quadrature PDF height is invalid")
        with figure_png.open("rb") as handle:
            png_header = handle.read(24)
        if (
            png_header[:8] != b"\x89PNG\r\n\x1a\n"
            or len(png_header) != 24
        ):
            errors.append("normalized quadrature PNG is invalid")
        else:
            png_width, png_height = struct.unpack(">II", png_header[16:24])
            if png_width != 680:
                errors.append("normalized quadrature PNG is not 680 pixels wide")
            if png_height <= 0:
                errors.append("normalized quadrature PNG height is invalid")
        periods = [
            float(record["period_V"]) for record in input_records.values()
        ]
        period_sigmas = [
            float(record["period_sigma_V"])
            for record in input_records.values()
        ]
        expected_period_extrema = {
            "period_V_min": min(periods),
            "period_V_max": max(periods),
            "period_sigma_V_min": min(period_sigmas),
            "period_sigma_V_max": max(period_sigmas),
        }
        for key, expected in expected_period_extrema.items():
            if not math.isclose(
                float(summary[key]),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(f"pooled {key} is inconsistent")
        if summary["profile_cutoff"] < summary["nominal_profile_cutoff"]:
            errors.append("pooled selected cutoff is below the nominal cutoff")
        for name, inference in pooled["normalizations"].items():
            if inference["profile_cutoff"] != summary["profile_cutoff"]:
                errors.append(
                    f"pooled {name} does not reuse the selected cutoff"
                )
        common_igm = float(summary["common_igm_transmission"])
        if not math.isclose(
            summary["eta_hat_finite_path"],
            common_igm * summary["eta_hat"],
            rel_tol=2e-9,
            abs_tol=2e-9,
        ):
            errors.append(
                "pooled finite-path estimate is not the IGM reparameterization"
            )
        if not math.isclose(
            summary["eta_upper_finite_path"],
            common_igm * summary["eta_upper"],
            rel_tol=2e-9,
            abs_tol=2e-9,
        ):
            errors.append(
                "pooled finite-path endpoint is not the IGM reparameterization"
            )
        if not math.isclose(
            summary["eta_upper_launch_infinity"],
            summary["eta_upper"] / 2.0,
            rel_tol=2e-9,
            abs_tol=2e-9,
        ):
            errors.append("pooled launched infinity endpoint is inconsistent")
        if not math.isclose(
            summary["eta_upper_launch_finite_path"],
            summary["eta_upper_finite_path"] / 2.0,
            rel_tol=2e-9,
            abs_tol=2e-9,
        ):
            errors.append("pooled launched finite-path endpoint is inconsistent")
        if not math.isclose(
            summary["critical_common_transmission_scale"],
            summary["eta_upper"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled critical common scale is inconsistent")
        if not math.isclose(
            summary["nominal_to_critical_scale_ratio"],
            1.0 / summary["critical_common_transmission_scale"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled inverse critical scale is inconsistent")
        if not math.isclose(
            summary["additional_common_optical_depth"],
            -math.log(summary["critical_common_transmission_scale"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled additional optical depth is inconsistent")
        if not math.isclose(
            summary["critical_igm_transmission"],
            common_igm * summary["critical_common_transmission_scale"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled critical IGM transmission is inconsistent")
        if not math.isclose(
            summary["critical_igm_optical_depth"],
            -math.log(summary["critical_igm_transmission"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled critical IGM optical depth is inconsistent")
        if not math.isclose(
            summary["nominal_igm_optical_depth"],
            -math.log(common_igm),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled nominal IGM optical depth is inconsistent")
        if not math.isclose(
            summary["critical_to_nominal_igm_optical_depth_ratio"],
            summary["critical_igm_optical_depth"]
            / summary["nominal_igm_optical_depth"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled IGM optical-depth ratio is inconsistent")
        if not math.isclose(
            summary["full_restoration_q"],
            summary["full_restoration_deviance"] - summary["deviance_min"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled full-restoration likelihood ratio is inconsistent")
        if not math.isclose(
            summary["full_restoration_nominal_sigma"] ** 2,
            summary["full_restoration_q"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled nominal full-restoration sigma is inconsistent")
        if not math.isclose(
            summary["null_q"],
            summary["null_deviance"] - summary["deviance_min"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("pooled null likelihood ratio is inconsistent")
        if not 0.0 <= pooled["null_test"]["pvalue"] <= 1.0:
            errors.append("pooled Monte Carlo null-test p-value is invalid")
        if not 0.0 <= pooled["goodness_of_fit"]["pvalue"] <= 1.0:
            errors.append("pooled Monte Carlo goodness-of-fit p-value is invalid")
        omitted = {
            record["omitted_dataset_id"] for record in pooled["leave_one_out"]
        }
        if omitted != configured_ids:
            errors.append("pooled leave-one-out records are incomplete")
        pooled_macros = (
            build / "manuscript-values/pooled-analysis-values.tex"
        ).read_text(encoding="utf-8")
        required_macros = {
            r"\ResultPooledEtaEstimateLetter",
            r"\ResultPooledEtaInfinityBoundLetter",
            r"\ResultPooledNullPValueLetter",
            r"\ResultPooledGoodnessOfFitPValueLetter",
            r"\ResultPooledLeaveOneOutMinimum",
            r"\ResultPooledLeaveOneOutMaximum",
            r"\ResultPooledLaunchOpticsTransmissionLetter",
            r"\ResultPooledAtmosphericTransmissionMinimumLetter",
            r"\ResultPooledAtmosphericTransmissionMaximumLetter",
            r"\ResultPooledMilkyWayTransmissionMinimumLetter",
            r"\ResultPooledMilkyWayTransmissionMaximumLetter",
            r"\ResultPooledCommonIgmTransmission",
            r"\ResultPooledCommonIgmTransmissionLetter",
            r"\ResultPooledTInfinityMinimumLetter",
            r"\ResultPooledTInfinityMaximumLetter",
            r"\ResultPooledTInfinityMinimumPercentLetter",
            r"\ResultPooledTInfinityMaximumPercentLetter",
            r"\ResultPooledTFinitePathMinimumLetter",
            r"\ResultPooledTFinitePathMaximumLetter",
            r"\ResultPooledEtaFinitePathBoundLetter",
            r"\ResultPooledEtaLaunchInfinityBoundLetter",
            r"\ResultPooledEtaLaunchFinitePathBoundLetter",
            r"\ResultPooledLaunchOpticsLossPercentLetter",
            r"\ResultPooledAtmosphericLossMinimumPercentLetter",
            r"\ResultPooledAtmosphericLossMaximumPercentLetter",
            r"\ResultPooledMilkyWayLossMinimumPercentLetter",
            r"\ResultPooledMilkyWayLossMaximumPercentLetter",
            r"\ResultPooledIgmLossPercentLetter",
            r"\ResultPooledTInfinityLossMinimumPercentLetter",
            r"\ResultPooledTInfinityLossMaximumPercentLetter",
            r"\ResultPooledTFinitePathLossMinimumPercentLetter",
            r"\ResultPooledTFinitePathLossMaximumPercentLetter",
        }
        for macro in required_macros:
            if rf"\newcommand{{{macro}}}" not in pooled_macros:
                errors.append(f"pooled manuscript values are missing {macro}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"pooled analysis invariant failure: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    try:
        validate_raw_inputs(config)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    outputs = canonical_outputs(config)
    errors = verify_checksums() + verify_invariants(args.build, config)
    if not args.internal_only:
        errors += verify_results_checksums(args.results, outputs)
        errors += compare(args.build, args.results, outputs)
    if errors:
        print("\n".join(f"ERROR: {item}" for item in errors), file=sys.stderr)
        return 1
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
