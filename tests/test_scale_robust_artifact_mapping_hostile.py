"""Hostile dict gate for scale-robust artifact _mapping."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import _mapping

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __getitem__(self, key: object):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__getitem__ must not run")


def test_mapping_rejects_subclass_before_walk() -> None:
    HostileDict.calls = 0
    errors: list[str] = []
    result = _mapping(HostileDict({"a": 1}), "probe", errors)
    assert result is None
    assert any("exact object" in err for err in errors)
    assert HostileDict.calls == 0
