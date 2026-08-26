"""Hostile container validation for forager CLI JSON serialization."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook executed")


class _HostileList(list[object]):
    calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile list hook executed")


def test_json_safe_rejects_hostile_dict_before_items() -> None:
    from alberta_framework.forager_cli import _json_safe

    _HostileDict.calls = 0
    with pytest.raises(TypeError, match="exact dict"):
        _json_safe(_HostileDict({"a": 1}))
    assert _HostileDict.calls == 0


def test_json_safe_rejects_hostile_list_before_iteration() -> None:
    from alberta_framework.forager_cli import _json_safe

    _HostileList.calls = 0
    with pytest.raises(TypeError, match="exact list"):
        _json_safe(_HostileList([1]))
    assert _HostileList.calls == 0
