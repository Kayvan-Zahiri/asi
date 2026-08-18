"""Trust-boundary validation for plasticity_comparators sanitized errors."""

from __future__ import annotations

import pathlib


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/plasticity_comparators.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    assert "matched_axes must equal '{required_axes}'" in text
