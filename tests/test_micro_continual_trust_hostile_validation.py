"""Trust-boundary validation for micro_continual sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.micro_continual import (
    _require_exact_str,
    micro_arm_spec,
)


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


def test_micro_arm_spec_unknown_sanitized() -> None:
    with pytest.raises(KeyError, match="unknown micro arm") as exc:
        micro_arm_spec("evil_arm")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_arm'" in msg


def test_micro_arm_spec_hostile() -> None:
    evil = _EvilStr("evil_arm")
    with pytest.raises(TypeError, match="must be an exact string"):
        micro_arm_spec(evil)


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/micro_continual.py")
    text = p.read_text(encoding="utf-8")
    assert "unknown micro arm {name!r}" not in text
    assert "arm_name {result.arm_name!r}" not in text
    assert "unknown arm {arm_name!r}" not in text
    assert "unknown micro arm '{host_name}'" in text
    assert "arm_name '{host_arm_name}'" in text
    assert "unknown arm '{host_arm_name}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
