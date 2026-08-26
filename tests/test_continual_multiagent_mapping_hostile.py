"""Hostile dict identity gate for continual multiagent required mapping helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    _required_mapping,
    _validate_source_provenance,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def keys(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping keys hook executed")

    def __bool__(self) -> bool:
        raise AssertionError("hostile mapping truth hook executed")


def test_required_mapping_rejects_hostile_dict_before_set_compare() -> None:
    parent: dict[str, object] = {"content": _HostileDict({"thresholds": {}})}
    errors: list[str] = []
    assert _required_mapping(parent, "content", errors) is None
    assert errors == ["content must be an object"]


def test_validate_source_provenance_rejects_hostile_dict_before_set_compare() -> None:
    errors: list[str] = []
    _validate_source_provenance(_HostileDict({"repository_subtree": "x"}), errors)
    assert errors == ["content.source_provenance must be an object"]
