#!/usr/bin/env python3
"""Recreate every configured dataset artifact in build/."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from analysis.artifact_config import (
    canonical_outputs,
    dataset_runs,
    load_config,
    validate_raw_inputs,
)
from analysis.transmission_values import calculate_transmissions

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def run_module(module: str, *arguments: str) -> None:
    config = load_config()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / "build/.matplotlib"))
    env.setdefault("SOURCE_DATE_EPOCH", str(config["source_date_epoch"]))
    subprocess.run(
        [str(PYTHON), "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def verify_raw_checksums() -> None:
    for line in (ROOT / "data/checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing checksummed raw input: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"raw checksum mismatch: {relative}")


def render_tex(source: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="complementarity-tex-") as name:
        work = Path(name)
        shutil.copyfile(source, work / "fragment.tex")
        (work / "render.tex").write_text(
            "\\documentclass[border=6pt]{standalone}\n"
            "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\\usepackage{adjustbox}\\usepackage{xcolor}\n"
            "\\pagecolor{white}\\begin{document}\\begin{minipage}{7.2in}\n"
            "\\centering\\small\\begin{adjustbox}{max width=\\linewidth}\n"
            "\\input{fragment.tex}\\end{adjustbox}\\end{minipage}\\end{document}\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "render.tex"],
            cwd=work, stdout=subprocess.DEVNULL, check=True,
        )
        subprocess.run(
            [
                "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                "-r200", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                f"-sOutputFile={output}", str(work / "render.pdf"),
            ],
            check=True,
        )


def reproduce_dataset(config: dict, dataset: dict, dataset_root: Path) -> None:
    for directory in ("analysis", "conditions", "figures", "tables", "manuscript-values", ".work"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    work = dataset_root / ".work"
    runs = {item["id"]: ROOT / item["path"] for item in dataset_runs(config, dataset)}
    options = config["null_bounds"]

    profile_id = dataset["conditions_profile"]
    atmosphere_source = config["conditions_profiles"][profile_id]["atmosphere_source"]
    milky_way_basis = config["transmission_analysis"]["milky_way_basis"]
    run_module(
        "analysis.conditions_sources",
        "--profile-id", profile_id,
        "--output-dir", str(dataset_root / "conditions"),
    )
    run_module(
        "analysis.generate_conditions_values",
        "--source", str(dataset_root / "conditions/conditions.json"),
        "--output", str(dataset_root / "manuscript-values/conditions-values.tex"),
        "--atmosphere-source", atmosphere_source,
        "--milky-way-basis", milky_way_basis,
    )
    conditions = json.loads(
        (dataset_root / "conditions/conditions.json").read_text(encoding="utf-8")
    )
    records = {
        record["tag"]: record for record in conditions["libradtran"]
        if isinstance(record, dict) and isinstance(record.get("tag"), str)
    }
    aeronet_tag = f"AERONET-{config['conditions_profiles'][profile_id]['aeronet_variant']}"
    atmosphere_tag = "GOES-18" if atmosphere_source == "goes" else aeronet_tag
    try:
        transmissions = calculate_transmissions(
            atmosphere=float(records[atmosphere_tag]["result"]["t_band"]),
            aeronet_atmosphere=float(records[aeronet_tag]["result"]["t_band"]),
            milky_way_point=float(conditions["irsa"]["transmission_fraction"]),
            milky_way_integrated=float(conditions["irsa"]["integration"]["avg_T"]),
            milky_way_minimum=float(conditions["irsa"]["integration"]["min_T"]),
            milky_way_basis=milky_way_basis,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{dataset['id']}: incomplete conditions transmission results"
        ) from exc

    run_module(
        "analysis.mzi_null_bounds",
        "--condition", f"Erase:{runs['run-1-erase']}",
        "--condition", f"Launch:{runs['run-1-launch']}",
        "--condition", f"Preserve:{runs['run-2-preserve']}",
        "--launch", options["launch"], "--control", options["control"],
        "--phase-reference", options["phase_reference"], "--phase-channel", options["phase_channel"],
        "--rpm-erased", options["rpm_erased"], "--rpm-unmodified", options["rpm_unmodified"],
        "--Tinf", repr(transmissions.infinity),
        "--normalization-T", ",".join(
            map(repr, (transmissions.infinity, transmissions.finite_path))
        ),
        "--x-match-tol", repr(options["voltage_tolerance"]),
        "--bootstrap", str(options["bootstrap"]), "--permutations", str(options["permutations"]),
        "--seed", str(config["random_seed"]),
        "--output", str(dataset_root / "analysis/mzi-null-bounds.json"),
    )
    bounds = json.loads((dataset_root / "analysis/mzi-null-bounds.json").read_text())
    period = bounds["summary"]["period_V"]
    run_module(
        "analysis.mzi_plot_alternating_by_pass",
        "--period", repr(period), "--output-dir", str(work),
        "--condition", f"Erase:{runs['run-1-erase']}",
        "--condition", f"Launch:{runs['run-1-launch']}",
        "--condition", f"Preserve:{runs['run-2-preserve']}",
        "--condition", f"Erase2:{runs['run-2-erase']}",
        "--order", "Preserve:C", "--order", "Erase:C", "--order", "Launch:I",
        "--order", "Erase:I", "--order", "Preserve:I",
        "--table-order", "1:Launch:I", "--table-order", "1:Erase:C",
        "--table-order", "1:Erase:I", "--table-order", "2:Preserve:C",
        "--table-order", "2:Preserve:I", "--table-order", "2:Erase2:C",
        "--table-order", "2:Erase2:I",
    )
    shutil.move(work / "mzi-alt-joint.json", dataset_root / "analysis/mzi-alt-joint.json")
    for suffix in ("pdf", "png"):
        shutil.move(work / f"mzi-alt-by-pass.{suffix}", dataset_root / f"figures/mzi-alt-by-pass.{suffix}")
    for name in ("mzi-alt-joint.tex", "mzi-alt-by-pass-run-1.tex", "mzi-alt-by-pass-run-2.tex"):
        shutil.move(work / name, dataset_root / "tables" / name)

    run_module(
        "analysis.darks_accidentals",
        "--dataset-id", dataset["id"], "--outdir", str(work),
    )
    shutil.move(
        work / "darks-accidentals.json",
        dataset_root / "analysis/darks-accidentals.json",
    )
    for name in ("dark-rates.tex", "accidental-corrections.tex"):
        shutil.move(work / name, dataset_root / "tables" / name)
    run_module(
        "analysis.mzi_null_plot",
        "--json", str(dataset_root / "analysis/mzi-null-bounds.json"),
        "--outfile", str(dataset_root / "figures/mzi-null-plot"),
    )

    run_module(
        "analysis.generate_analysis_values",
        "--source", str(dataset_root / "analysis/mzi-null-bounds.json"),
        "--joint-source", str(dataset_root / "analysis/mzi-alt-joint.json"),
        "--conditions-source", str(dataset_root / "conditions/conditions.json"),
        "--atmosphere-source", atmosphere_source,
        "--milky-way-basis", milky_way_basis,
        "--output", str(dataset_root / "manuscript-values/analysis-values.tex"),
    )

    for stem in (
        "mzi-alt-by-pass-run-1", "mzi-alt-by-pass-run-2", "mzi-alt-joint",
        "dark-rates", "accidental-corrections",
    ):
        render_tex(dataset_root / f"tables/{stem}.tex", dataset_root / f"tables/{stem}.png")
    run_module(
        "analysis.generate_results_readmes",
        "--output-root", str(dataset_root.parent.parent),
        "--dataset-id", dataset["id"],
    )
    shutil.rmtree(work)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        action="append",
        help="Reproduce only this configured dataset (repeatable); preserve other build datasets.",
    )
    args = parser.parse_args()
    config = load_config()
    validate_raw_inputs(config)
    verify_raw_checksums()
    build = ROOT / "build"
    if build.exists() and not args.dataset_id:
        shutil.rmtree(build)
    run_module(
        "analysis.igm_transmission",
        "--output", str(build / "analysis/igm-transmission.json"),
    )
    run_module(
        "analysis.generate_igm_values",
        "--source", str(build / "analysis/igm-transmission.json"),
        "--output", str(build / "manuscript-values/igm-transmission-values.tex"),
    )
    datasets_root = build / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    selected = [
        dataset for dataset in config["datasets"]
        if not args.dataset_id or dataset["id"] in set(args.dataset_id)
    ]
    if args.dataset_id and len(selected) != len(set(args.dataset_id)):
        raise ValueError("one or more --dataset-id values are not configured")
    for index, dataset in enumerate(selected, 1):
        dataset_root = datasets_root / dataset["id"]
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        print(f"[{index}/{len(selected)}] {dataset['id']}", flush=True)
        reproduce_dataset(config, dataset, dataset_root)
    summary_inputs = [
        datasets_root / dataset["id"] / relative
        for dataset in config["datasets"]
        for relative in (
            "analysis/mzi-null-bounds.json",
            "conditions/conditions.json",
        )
    ]
    if all(path.is_file() for path in summary_inputs):
        pooled = config["pooled_analysis"]
        run_module(
            "analysis.mzi_pooled_analysis",
            "--source-root", str(build),
            "--json-output", str(build / "analysis/mzi-pooled-analysis.json"),
            "--confidence", repr(pooled["confidence"]),
            "--eta-max", repr(pooled["eta_max"]),
            "--coverage-simulations", str(pooled["coverage_simulations"]),
            "--coverage-eta-grid", ",".join(
                map(repr, pooled["coverage_eta_grid"])
            ),
            "--random-phase-configurations",
            str(pooled["random_phase_configurations"]),
            "--seed", str(config["random_seed"]),
        )
        run_module(
            "analysis.generate_pooled_values",
            "--source", str(build / "analysis/mzi-pooled-analysis.json"),
            "--output",
            str(build / "manuscript-values/pooled-analysis-values.tex"),
        )
        run_module(
            "analysis.mzi_normalized_quadrature_plot",
            "--json", str(build / "analysis/mzi-pooled-analysis.json"),
            "--outfile",
            str(build / "figures/mzi-normalized-quadratures"),
        )
        run_module(
            "analysis.generate_dataset_summary",
            "--source-root", str(build),
            "--transmission-output", str(build / "tables/dataset-transmission.tex"),
            "--analysis-output", str(build / "tables/dataset-analysis.tex"),
        )
        for stem in ("dataset-transmission", "dataset-analysis"):
            render_tex(
                build / f"tables/{stem}.tex",
                build / f"tables/{stem}.png",
            )
        run_module(
            "analysis.generate_results_readmes",
            "--output-root", str(build),
            "--global-readme",
        )
    expected = [
        build / relative
        for relative in canonical_outputs(config)
        if not args.dataset_id
        or (
            relative.startswith("datasets/")
            and relative.split("/", 2)[1] in set(args.dataset_id)
        )
    ]
    incomplete = [str(path) for path in expected if not path.is_file() or not path.stat().st_size]
    if incomplete:
        raise RuntimeError(
            "reproduction did not produce every required nonempty artifact:\n  "
            + "\n  ".join(incomplete)
        )
    print(f"Reproduced {len(canonical_outputs(config))} canonical products in {build}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
