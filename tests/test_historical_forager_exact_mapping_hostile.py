"""Hostile mapping-subclass guards for historical forager JSON copies."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.historical_forager import HistoricalForagerContractError, _json_mapping_copy


def test_json_mapping_copy_rejects_dict_subclass_before_canonicalization() -> None:
    class HostileDict(dict):
        calls = 0

        def items(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileDict.items must not run")

    HostileDict.calls = 0
    with pytest.raises(HistoricalForagerContractError, match="must be a mapping"):
        _json_mapping_copy(HostileDict({"a": 1}), name="probe")
    assert HostileDict.calls == 0
