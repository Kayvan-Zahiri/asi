"""Trust-boundary validation for forager_matched_final_analysis sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_final_analysis import (
    ForagerMatchedFinalAnalysisError,
    _expected_entrypoint_binding,
    _require_identifier,
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


def test_require_identifier_rejects_subclass() -> None:
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="must be a non-empty string"):
        _require_identifier(_StringSubclass("x"), "candidate_id")


def test_require_identifier_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="must be a non-empty string") as exc:
        _require_identifier(evil, "candidate_id")
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_expected_entrypoint_binding_unknown_sanitized() -> None:
    with pytest.raises(
        ForagerMatchedFinalAnalysisError, match="has no frozen entrypoint binding"
    ) as exc:
        _expected_entrypoint_binding("evil_cand")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_cand'" in msg


def test_expected_entrypoint_binding_hostile() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="must be a non-empty string"):
        _expected_entrypoint_binding(evil)


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_matched_final_analysis.py")
    text = p.read_text(encoding="utf-8")
    assert "candidate {candidate_id!r}" not in text
    assert "unknown candidate {candidate_id!r}" not in text
    assert "candidate '{host_candidate_id}' has no frozen entrypoint binding" in text
    assert "evaluation source manifest names unknown candidate '{host_candidate_id}'" in text
    assert "evaluation command template names unknown candidate '{host_candidate_id}'" in text


def test_valid_still_passes() -> None:
    assert _require_identifier("valid_cand", "candidate_id") == "valid_cand"
