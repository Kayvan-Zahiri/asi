"""Hostile exact JSON gates for scale-robust artifact validation."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _mapping,
    _parsed_pairs,
)

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
    errors: list[str] = []
    HostileDict.calls = 0
    assert _mapping(HostileDict({"a": 1}), "root", errors) is None
    assert errors == ["root must be an object"]
    assert HostileDict.calls == 0


def test_parsed_pairs_rejects_list_subclass_without_iter_hooks() -> None:
    HostileList.calls = 0
    assert _parsed_pairs(HostileList([[0, 1]])) is None
    assert HostileList.calls == 0


def test_parsed_pairs_rejects_pair_list_subclass_without_iter_hooks() -> None:
    class HostilePairList(list):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostilePairList.__iter__ must not run")

    HostilePairList.calls = 0
    assert _parsed_pairs([HostilePairList([0, 1])]) is None
    assert HostilePairList.calls == 0
