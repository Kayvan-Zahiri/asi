"""Trust-boundary validation for forager_matched_protocol sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocolError,
    _duplicate_free_object,
    _parse_json_float,
    _reject_nonfinite,
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


def test_duplicate_keys_sanitized() -> None:
    pairs = [("a", 1), ("a", 2)]
    with pytest.raises(ForagerMatchedProtocolError, match="duplicate JSON object key") as exc:
        _duplicate_free_object(pairs)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'a'" in msg


def test_reject_nonfinite_sanitized() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="non-finite JSON number") as exc:
        _reject_nonfinite("NaN")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'NaN'" in msg


def test_parse_json_float_sanitized() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="non-finite JSON number") as exc:
        _parse_json_float("inf")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'inf'" in msg


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_matched_protocol.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    assert "duplicate JSON object key '{host_key}'" in text
    assert "non-finite JSON number '{host_token}' is forbidden" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert _duplicate_free_object([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    assert _parse_json_float("1.25") == 1.25
