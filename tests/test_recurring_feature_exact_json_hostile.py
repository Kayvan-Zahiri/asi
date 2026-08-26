"""Hostile exact JSON gates for recurring-feature artifact validation."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import _mapping, _pair_set

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


class HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileList.__iter__ must not run")


def test_mapping_rejects_dict_subclass_without_iter_hooks() -> None:
    parent: dict[str, object] = {"child": HostileDict({"a": 1})}
    errors: list[str] = []
    HostileDict.calls = 0
    assert _mapping(parent, "child", "root", errors) is None
    assert errors == ["root.child must be an object"]
    assert HostileDict.calls == 0


def test_pair_set_rejects_list_subclass_without_iter_hooks() -> None:
    HostileList.calls = 0
    assert _pair_set(HostileList([[0, 1]])) is None
    assert HostileList.calls == 0
