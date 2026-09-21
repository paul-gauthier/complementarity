"""Promote configured build products to the reviewed results tree."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from analysis.artifact_config import canonical_outputs, load_config

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    subprocess.run(
        [sys.executable, "-m", "scripts.verify_results", "--internal-only"],
        cwd=ROOT,
        check=True,
    )
    config = load_config()
    build, results = ROOT / "build", ROOT / "results"
    outputs = canonical_outputs(config)
    configured = set(outputs)
    for path in results.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(results).as_posix()
        if (
            relative not in configured
            and relative != "checksums.sha256"
        ):
            path.unlink()
    for relative in outputs:
        source, destination = build / relative, results / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    lines = []
    for relative in outputs:
        digest = hashlib.sha256((results / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    manifest = results / "checksums.sha256"
    temporary = manifest.with_name(".checksums.sha256.tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(manifest)
    for path in sorted(results.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f"Promoted {len(outputs)} canonical products.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
