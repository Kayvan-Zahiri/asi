"""Trust-boundary validation for ipmnist_screening sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    _require_exact_str,
    screening_spec,
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


def test_screening_spec_unknown_sanitized() -> None:
    with pytest.raises(ValueError, match="unknown screening config") as exc:
        screening_spec("evil_config")
    msg = str(exc.value)
    assert "!r" not in msg


def test_screening_spec_hostile() -> None:
    evil = _EvilStr("evil_config")
    with pytest.raises(ValueError, match="must be an exact string"):
        screening_spec(evil)


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/ipmnist_screening.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    assert "control '{host_control_name}' is not among the merged shards" in text
    assert "config '{host_config_name}' has inconsistent base_learner across seeds" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert screening_spec("adamw_control").name == "adamw_control"
