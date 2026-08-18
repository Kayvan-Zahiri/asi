"""Trust-boundary validation for reference_life_scorecard sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    _control_adapter,
    _require_exact_str,
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


def test_control_adapter_unsupported_env_sanitized() -> None:
    with pytest.raises(ValueError, match="unsupported environment") as exc:
        _control_adapter(
            arm="sarsa",
            arm_definition={},
            environment_kind="evil_env",
            switching_config=None,
            river_config=None,
            horizon=1000,
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_env'" in msg


def test_control_adapter_unsupported_env_hostile() -> None:
    evil = _EvilStr("evil_env")
    with pytest.raises(ValueError, match="must be an exact string"):
        _control_adapter(
            arm="sarsa",
            arm_definition={},
            environment_kind=evil,
            switching_config=None,
            river_config=None,
            horizon=1000,
        )


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/reference_life_scorecard.py")
    text = p.read_text(encoding="utf-8")
    assert "duplicate run identity {identity!r}" not in text
    assert "missing scheduled identity {identity!r}" not in text
    assert "duplicate JSON key {key!r}" not in text
    assert "unsupported environment {environment_kind!r}" not in text
    assert "arm {arm!r} is not a control adapter" not in text
    assert "duplicate JSON key '{host_key}'" in text
    assert "unsupported environment '{host_env}'" in text
    assert "arm '{host_arm}' is not a control adapter" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
