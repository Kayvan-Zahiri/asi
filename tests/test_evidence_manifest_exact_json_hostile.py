"""Hostile exact JSON gates for evidence manifest mapping helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import _required_mapping

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_required_mapping_rejects_dict_subclass_without_iter_hooks() -> None:
    owner: dict[str, object] = {"nested": HostileDict({"a": 1})}
    HostileDict.calls = 0
    with pytest.raises(RuntimeError, match="owner.nested must be a non-empty mapping"):
        _required_mapping(owner, "nested", owner="owner")
    assert HostileDict.calls == 0
