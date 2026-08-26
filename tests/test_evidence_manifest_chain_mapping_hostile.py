"""Hostile dict gate for evidence manifest _chain_mapping."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import _chain_mapping

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.keys must not run")


def test_chain_mapping_rejects_subclass_before_key_compare() -> None:
    HostileDict.calls = 0
    errors: list[str] = []
    result = _chain_mapping(
        HostileDict({"a": 1}),
        location="chain",
        expected_keys={"a"},
        errors=errors,
    )
    assert result is None
    assert any("exact object" in err for err in errors)
    assert HostileDict.calls == 0
