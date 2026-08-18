"""Trust-boundary validation for reference_life sanitized errors."""

from __future__ import annotations

import pathlib


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/reference_life.py")
    text = p.read_text(encoding="utf-8")
    assert "life configuration field {key!r} is not bound" not in text
    assert "life configuration field '{key}' is not bound" in text
