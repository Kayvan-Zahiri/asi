"""Trust-boundary validation for prototype_agent sanitized errors."""

from __future__ import annotations

import pathlib


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/core/prototype_agent.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    assert "prototype state uses unsupported PRNG implementation '{implementation}'" in text
