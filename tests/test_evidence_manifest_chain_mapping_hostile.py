"""Hostile dict identity gate for evidence manifest FTL chain helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    _chain_mapping,
    _ftl_acceptance_passed,
    _ftl_scientific_digest,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def keys(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping keys hook executed")

    def __bool__(self) -> bool:
        raise AssertionError("hostile mapping truth hook executed")


def test_chain_mapping_rejects_hostile_dict_before_set_compare() -> None:
    errors: list[str] = []
    assert (
        _chain_mapping(
            _HostileDict({"a": 1}),
            location="artifact",
            expected_keys={"a"},
            errors=errors,
        )
        is None
    )
    assert errors == ["artifact must be an object"]


def test_ftl_acceptance_rejects_hostile_scientific_before_set_compare() -> None:
    errors: list[str] = []
    artifact: dict[str, object] = {"scientific_payload": _HostileDict({"acceptance": {}})}
    assert _ftl_acceptance_passed(artifact, location="artifact", errors=errors) is None
    assert errors == ["artifact.scientific_payload must be an object"]


def test_ftl_scientific_digest_rejects_hostile_digest_before_set_compare() -> None:
    errors: list[str] = []
    artifact: dict[str, object] = {
        "scientific_payload": {},
        "scientific_digest": _HostileDict({"sha256": "a" * 64}),
    }
    assert _ftl_scientific_digest(artifact, location="artifact", errors=errors) is None
    assert errors == ["artifact.scientific_digest must be an object"]
