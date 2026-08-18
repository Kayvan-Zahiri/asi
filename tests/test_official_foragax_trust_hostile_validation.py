"""Trust-boundary validation for official_foragax sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.official_foragax import (
    OfficialForagaxValidationError,
    _require_exact_str,
    _strict_json_loads,
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


def test_duplicate_keys_sanitized() -> None:
    json_bytes = b'{"a": 1, "a": 2}'
    with pytest.raises(
        OfficialForagaxValidationError, match="contains duplicate object key"
    ) as exc:
        _strict_json_loads(json_bytes, label="test payload")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'a'" in msg


def test_source_contains_no_error_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/official_foragax.py")
    text = p.read_text(encoding="utf-8")
    assert "duplicate object key '{host_key}'" in text
    assert "official repository has no tracked files under '{pathspec}'" in text
    assert "unsupported official manifest schema '{schema_version}'" in text


def test_valid_strict_json() -> None:
    assert _strict_json_loads('{"a": 1, "b": "ok"}', label="valid") == {"a": 1, "b": "ok"}
