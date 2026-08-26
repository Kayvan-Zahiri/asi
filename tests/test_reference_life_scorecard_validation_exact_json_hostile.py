"""Hostile exact JSON gates for reference-life scorecard manifest validation."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    _validate_agent_manifest_descriptor,
)

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_validate_agent_manifest_rejects_dict_subclass_without_iter_hooks() -> None:
    HostileDict.calls = 0
    with pytest.raises(ValueError, match="must be a complete agent manifest"):
        _validate_agent_manifest_descriptor(HostileDict({"api_version": "x"}), path="agent")
    assert HostileDict.calls == 0
