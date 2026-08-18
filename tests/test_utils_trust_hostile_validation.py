"""Trust-boundary validation for utils sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.utils.experiments import _require_exact_str
from alberta_framework.utils.statistics import SignificanceResult


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("value", _StringSubclass("x"))
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("val")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        _require_exact_str("value", evil2)
    assert "EvilStr" not in str(exc2.value)


def test_significance_result_replace_unexpected_sanitized() -> None:
    res = SignificanceResult(
        test_name="paired_t",
        statistic=1.5,
        p_value=0.01,
        significant=True,
        alpha=0.05,
        effect_size=0.8,
        method_a="A",
        method_b="B",
    )
    with pytest.raises(ValueError, match="Got unexpected field names") as exc:
        res._replace(unknown_field=1)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "unknown_field" in msg


def test_source_contains_no_repr_leak() -> None:
    p1 = pathlib.Path("alberta_framework/utils/experiments.py")
    text1 = p1.read_text(encoding="utf-8")
    assert "Got unexpected field names: {sorted(unexpected)!r}" not in text1
    assert "Got unexpected field names: {sorted(unexpected)}" in text1

    p2 = pathlib.Path("alberta_framework/utils/statistics.py")
    text2 = p2.read_text(encoding="utf-8")
    assert "Got unexpected field names: {sorted(unexpected)!r}" not in text2
    assert "Got unexpected field names: {sorted(unexpected)}" in text2


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
