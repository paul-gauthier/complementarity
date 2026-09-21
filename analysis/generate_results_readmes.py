#!/usr/bin/env python3
"""Generate the global and per-dataset results README files."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from .artifact_config import get_dataset, load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "build"


def render_global_readme(config: dict) -> str:
    lines = [
        "# Reviewed dataset products",
        "",
        "This directory contains the reviewed reference reproduction. Analysis commands",
        "write to ignored `build/`; only `scripts/promote_results.py` updates this tree.",
        "",
        "Products are grouped under",
        "`datasets/<timestamp>--<timestamp>/`.",
        "Each complete dataset contains machine-readable analysis, figures, tables,",
        "environmental reports, and manuscript macros.",
        "",
        "## Cross-dataset summary",
        "",
        "![Normalized quadrature confidence regions](figures/mzi-normalized-quadratures.png)",
        "",
        "![Dataset transmissions](tables/dataset-transmission.png)",
        "",
        "![Dataset analysis results](tables/dataset-analysis.png)",
        "",
        "The pooled analysis is available as",
        "[`analysis/mzi-pooled-analysis.json`](analysis/mzi-pooled-analysis.json).",
        "The intergalactic-transmission budget is available as",
        "[`analysis/igm-transmission.json`](analysis/igm-transmission.json).",
        "",
        "The captionless TeX fragments are",
        "[`tables/dataset-transmission.tex`](tables/dataset-transmission.tex)",
        "and [`tables/dataset-analysis.tex`](tables/dataset-analysis.tex).",
        "",
        "## Datasets",
        "",
    ]
    for dataset in config["datasets"]:
        lines.append(
            f"- **{dataset['display_id']}**: [`{dataset['id']}`](datasets/{dataset['id']}/) "
            f"— launch `{dataset['launch_run']}`, preserve `{dataset['preserve_run']}`"
        )
    lines.extend(
        [
            "",
            "The semantic launch and preserve roles are declared `config/artifact.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_dataset_readme(dataset: dict) -> str:
    return "\n".join(
        [
            f"# Dataset {dataset['display_id']} reviewed reference products",
            "",
            f"Dataset: `{dataset['id']}`",
            "",
            f"- Launch run: `{dataset['launch_run']}`",
            f"- Preserve run: `{dataset['preserve_run']}`",
            f"- Visual conditions: {dataset['visual_conditions']}",
            "",
            "## Figures",
            "",
            "![MZI fringe results](figures/mzi-alt-by-pass.png)",
            "",
            "![Launch-null exclusion plot](figures/mzi-null-plot.png)",
            "",
            "## Tables",
            "",
            "![MZI joint fit](tables/mzi-alt-joint.png)",
            "",
            "![Run 1 per-pass fits](tables/mzi-alt-by-pass-run-1.png)",
            "",
            "![Run 2 per-pass fits](tables/mzi-alt-by-pass-run-2.png)",
            "",
            "![In-apparatus dark-run background rates](tables/dark-rates.png)",
            "",
            "![Accidental corrections](tables/accidental-corrections.png)",
            "",
            "Machine-readable analysis is under `analysis/`, environmental reports under",
            "`conditions/`, and generated manuscript macros under `manuscript-values/`.",
            "",
        ]
    )


def write_atomically(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate browsable README files for reviewed results."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-id", action="append")
    parser.add_argument("--global-readme", action="store_true")
    args = parser.parse_args()
    config = load_config()

    dataset_ids = args.dataset_id or (
        [] if args.global_readme else [dataset["id"] for dataset in config["datasets"]]
    )
    for dataset_id in dataset_ids:
        dataset = get_dataset(config, dataset_id)
        write_atomically(
            args.output_root / "datasets" / dataset_id / "README.md",
            render_dataset_readme(dataset),
        )
    if args.global_readme or not args.dataset_id:
        write_atomically(
            args.output_root / "README.md",
            render_global_readme(config),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
