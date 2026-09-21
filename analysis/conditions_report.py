"""Canonical, atomic conditions report normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def normalize_provenance(value: Any, source_by_name: dict[str, str]) -> Any:
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key in {"source_file", "file"} and isinstance(item, str):
                item = source_by_name.get(Path(item).name, item)
            normalized[key] = normalize_provenance(item, source_by_name)
        return normalized
    if isinstance(value, list):
        return [normalize_provenance(item, source_by_name) for item in value]
    return value


def canonicalize_json(
    source: Path, destination: Path, source_by_name: dict[str, str]
) -> None:
    value = json.loads(source.read_text(encoding="utf-8"))
    normalized = normalize_provenance(value, source_by_name)
    atomic_write_text(
        destination,
        json.dumps(normalized, indent=2, allow_nan=False) + "\n",
    )


__all__ = ["atomic_write_text", "canonicalize_json", "normalize_provenance"]
