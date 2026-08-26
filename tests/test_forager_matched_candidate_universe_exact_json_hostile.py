"""Hostile exact JSON gates for candidate-universe validation."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_candidate_universe import (
    ForagerMatchedCandidateUniverseError,
    _require_mapping,
)

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_require_mapping_rejects_dict_subclass_without_iter_hooks() -> None:
    HostileDict.calls = 0
    with pytest.raises(
        ForagerMatchedCandidateUniverseError,
        match="root must be an object",
    ):
        _require_mapping(HostileDict({"a": 1}), "root")
    assert HostileDict.calls == 0
