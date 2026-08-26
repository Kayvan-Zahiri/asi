"""Hostile exact JSON gates for reference-life scorecard array walks."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import _iter_array_leaves

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.values must not run")


class HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileList.__iter__ must not run")


def test_iter_array_leaves_rejects_dict_subclass_without_iter_hooks() -> None:
    HostileDict.calls = 0
    with pytest.raises(StopIteration):
        next(_iter_array_leaves(HostileDict({"a": 1})))
    assert HostileDict.calls == 0


def test_iter_array_leaves_rejects_list_subclass_without_iter_hooks() -> None:
    HostileList.calls = 0
    with pytest.raises(StopIteration):
        next(_iter_array_leaves(HostileList([1])))
    assert HostileList.calls == 0
