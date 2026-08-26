"""Hostile exact JSON gates for scale-robust phase/metrics paths."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _metrics_from_condition,
)

pytestmark = pytest.mark.unit


class HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileList.__iter__ must not run")


def test_metrics_from_condition_rejects_list_subclass_without_iter_hooks() -> None:
    condition: dict[str, object] = {"phase_windows": HostileList([])}
    HostileList.calls = 0
    result = _metrics_from_condition(condition)
    assert all(value is None for value in result.values())
    assert HostileList.calls == 0
