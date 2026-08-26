"""Hostile exact JSON gates for evidence-manifest acceptance/digest helpers."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import _ftl_acceptance_passed

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_ftl_acceptance_passed_rejects_scientific_payload_subclass() -> None:
    artifact: dict[str, object] = {"scientific_payload": HostileDict({"acceptance": {}})}
    errors: list[str] = []
    HostileDict.calls = 0
    assert _ftl_acceptance_passed(artifact, location="artifact", errors=errors) is None
    assert errors == ["artifact.scientific_payload must be an object"]
    assert HostileDict.calls == 0
