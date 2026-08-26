"""Hostile mapping gate for FTL online-agent development validation."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.ftl_online_agent_development import validate_result

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_validate_result_rejects_mapping_subclass_without_iter_hooks() -> None:
    HostileDict.calls = 0
    with pytest.raises(ValueError, match="result must be an exact DevelopmentResult"):
        validate_result(HostileDict({"schema": "x"}))
    assert HostileDict.calls == 0
