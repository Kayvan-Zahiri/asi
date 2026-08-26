"""Hostile dict/list identity gate for evidence manifest required helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    _required_mapping,
    _required_string_tuple,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping iteration hook executed")


class _HostileList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile sequence iteration hook executed")


def test_required_mapping_rejects_hostile_dict_before_truthiness() -> None:
    payload: dict[str, object] = {"seed_roles": _HostileDict({"development": [0]})}
    with pytest.raises(RuntimeError, match="must be a non-empty mapping"):
        _required_mapping(payload, "seed_roles", owner="test.protocol")


def test_required_string_tuple_rejects_hostile_list_before_iteration() -> None:
    payload: dict[str, object] = {"excluded_claims": _HostileList(["a", "b"])}
    with pytest.raises(RuntimeError, match="must be a non-empty string array"):
        _required_string_tuple(payload, ("excluded_claims",), owner="test.protocol")
