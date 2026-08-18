"""Trust-boundary validation for forager_results sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_results import (
    _json_without_duplicate_keys,
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


def test_json_without_duplicate_keys_sanitized() -> None:
    payload = b'{"a": 1, "a": 2}'
    with pytest.raises(ValueError, match="contains duplicate JSON key") as exc:
        _json_without_duplicate_keys(payload, path=pathlib.Path("test.json"))
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'a'" in msg


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_results.py")
    text = p.read_text(encoding="utf-8")
    assert "contains duplicate JSON key {key!r}" not in text
    assert "contains non-standard JSON constant {value!r}" not in text
    assert "contains non-finite JSON number {value!r}" not in text
    assert "contains duplicate JSON key '{host_key}'" in text
    assert "contains non-standard JSON constant '{host_value}'" in text
    assert "contains non-finite JSON number '{host_value}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert _json_without_duplicate_keys(b'{"a": 1, "b": 2}', path=pathlib.Path("ok.json")) == {
        "a": 1,
        "b": 2,
    }
