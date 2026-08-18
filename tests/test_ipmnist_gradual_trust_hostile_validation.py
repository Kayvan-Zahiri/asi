"""Trust-boundary validation for ipmnist_gradual sanitized errors."""

from __future__ import annotations

import pathlib


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/ipmnist_gradual.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    assert "schema must be '{_PAIR_RESULT_SCHEMA}'" in text
