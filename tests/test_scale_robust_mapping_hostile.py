"""Hostile dict/list identity gate for scale-robust artifact mapping helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _mapping,
    _parsed_pairs,
    threshold_calibration_ready,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def keys(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping keys hook executed")


class _HostileList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile sequence iteration hook executed")


def test_mapping_rejects_hostile_dict_before_set_compare() -> None:
    errors: list[str] = []
    assert _mapping(_HostileDict({"phase_index": 0}), "phase", errors) is None
    assert errors == ["phase must be an object"]


def test_parsed_pairs_rejects_hostile_list_before_iteration() -> None:
    assert _parsed_pairs(_HostileList([[0, 1]]), None) is None


def test_threshold_calibration_rejects_hostile_calibration_before_get() -> None:
    thresholds: dict[str, object] = {"calibration": _HostileDict({"status": "x"})}
    # threshold_calibration_ready reads module thresholds; patch via monkeypatch
    import alberta_framework.evaluation.scale_robust_feature_artifact as module

    original = module._threshold_payload
    module._threshold_payload = lambda: thresholds  # type: ignore[method-assign, assignment]
    try:
        assert threshold_calibration_ready() is False
    finally:
        module._threshold_payload = original  # type: ignore[method-assign]
