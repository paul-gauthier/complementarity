"""Compatibility fixes for byte-reproducible Matplotlib PDF output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from matplotlib.backends.backend_pdf import PdfFile


def _canonical_subset_prefix(
    original: Callable[[Any], str],
    charset: Iterable[int],
) -> str:
    """Call Matplotlib's subset-prefix helper with a value-hashed collection."""
    return original(frozenset(charset))


def make_pdf_font_subsets_deterministic() -> None:
    """Prevent TeX font subset names from depending on a temporary object's id.

    Matplotlib 3.11 passes ``dict_values`` directly to ``_get_subset_prefix``
    while embedding TeX Type-1 fonts. Its hash is identity-based, so the font
    subset name and embedded font bytes otherwise vary between processes.
    """
    original = PdfFile._get_subset_prefix
    if getattr(original, "_complementarity_deterministic", False):
        return

    def deterministic_subset_prefix(charset: Iterable[int]) -> str:
        return _canonical_subset_prefix(original, charset)

    deterministic_subset_prefix._complementarity_deterministic = True  # type: ignore[attr-defined]
    PdfFile._get_subset_prefix = staticmethod(deterministic_subset_prefix)
