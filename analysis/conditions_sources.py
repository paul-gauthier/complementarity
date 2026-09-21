"""Archived-source-only observing-condition replay."""

from __future__ import annotations

import argparse
import os
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import conditions
from .artifact_config import (
    REPOSITORY_ROOT,
    load_config,
    resolve_repository_path,
)
from .conditions_report import canonicalize_json
from .goes import GoesSourceFiles


@contextmanager
def network_disabled() -> Iterator[None]:
    original_connection = socket.create_connection

    def reject(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during archived replay")

    socket.create_connection = reject
    try:
        yield
    finally:
        socket.create_connection = original_connection


def _stage_sources(
    profile: dict[str, Any], directory: Path
) -> tuple[dict[str, str], dict[str, Path]]:
    source_by_name: dict[str, str] = {}
    staged_by_role: dict[str, Path] = {}
    configured = profile["sources"]
    for source_name, relative in configured.items():
        source = resolve_repository_path(relative)
        source_by_name[source.name] = relative
        destination = directory / source.name
        os.symlink(source, destination)
        staged_by_role[source_name] = destination

    irsa = resolve_repository_path(configured["irsa_json"]).parent
    irsa_prefix = profile["pull_prefixes"]["irsa"]
    for source in sorted(irsa.glob(f"{irsa_prefix}-IRSA_zenith-sample-*.ecsv")):
        relative = source.relative_to(REPOSITORY_ROOT).as_posix()
        source_by_name[source.name] = relative
        os.symlink(source, directory / source.name)
    return source_by_name, staged_by_role


def replay_archived_conditions(output_dir: Path, profile_id: str) -> None:
    config = load_config()
    profile = config["conditions_profiles"][profile_id]
    prefixes = profile["pull_prefixes"]
    aeronet_variant = profile["aeronet_variant"]
    atmosphere_source = profile["atmosphere_source"]
    goes_pull_prefix = prefixes["goes"] if atmosphere_source == "goes" else None
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conditions-input-") as name:
        staged = Path(name)
        source_by_name, staged_by_role = _stage_sources(profile, staged)
        goes_sources = (
            GoesSourceFiles.from_mapping(staged_by_role)
            if atmosphere_source == "goes"
            else None
        )
        with network_disabled():
            conditions.replay_archived(
                target_dir=profile["target_timestamp"],
                input_dir=staged,
                output_dir=output_dir,
                integration_minutes=profile["integration_minutes"],
                metar_pull_prefix=prefixes["metar"],
                aeronet_variant=aeronet_variant,
                aeronet_pull_prefix=prefixes["aeronet"],
                irsa_pull_prefix=prefixes["irsa"],
                atmosphere_source=atmosphere_source,
                goes_pull_prefix=goes_pull_prefix,
                goes_sources=goes_sources,
            )

    canonicalize_json(
        output_dir / "conditions.json",
        output_dir / "conditions.json",
        source_by_name,
    )


__all__ = ["network_disabled", "replay_archived_conditions"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay configured archived conditions inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    replay_archived_conditions(args.output_dir, args.profile_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
