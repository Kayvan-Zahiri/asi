"""Hostile dict gate for forager CLI _json_safe before item walk."""

from __future__ import annotations

import pytest

from alberta_framework.forager_cli import _json_safe

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.items must not run")


def test_json_safe_rejects_mapping_subclass_before_items() -> None:
    HostileDict.calls = 0
    hostile = HostileDict({"a": 1})
    result = _json_safe(hostile)
    assert HostileDict.calls == 0
    assert result is hostile


def test_json_safe_accepts_plain_dict() -> None:
    assert _json_safe({"a": 1}) == {"a": 1}
