"""Hostile dict/list identity gate for recurring feature artifact mapping helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import (
    _extract_seed_metrics,
    _mapping,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def keys(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping keys hook executed")


class _HostileList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile sequence iteration hook executed")


def test_mapping_rejects_hostile_dict_before_set_compare() -> None:
    parent: dict[str, object] = {"nested": _HostileDict({"seed": 0})}
    errors: list[str] = []
    assert _mapping(parent, "nested", "parent", errors) is None
    assert errors == ["parent.nested must be an object"]


def test_extract_seed_metrics_rejects_hostile_list_before_iteration() -> None:
    errors: list[str] = []
    seeds, variants = _extract_seed_metrics(_HostileList([{"seed": 0}]), errors)
    assert seeds == []
    assert variants == {"retained": [], "no_retention": []}
    assert errors == ["scientific_payload.seed_summaries must be a non-empty array"]


def test_extract_seed_metrics_rejects_hostile_summary_before_set_compare() -> None:
    errors: list[str] = []
    seeds, variants = _extract_seed_metrics([_HostileDict({"seed": 0})], errors)
    assert seeds == []
    assert variants == {"retained": [], "no_retention": []}
    assert any("must be an object" in error for error in errors)
