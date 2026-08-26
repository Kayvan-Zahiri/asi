"""Hostile dict identity gate for evidence manifest artifact digest lookup."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import _artifact_digest

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        if key == "sha256":
            return "a" * 64
        return super().get(key, default)

    def __bool__(self) -> bool:
        raise AssertionError("hostile mapping truth hook executed")


def test_artifact_digest_rejects_hostile_digest_record_before_truthiness() -> None:
    artifact: dict[str, object] = {"content_digest": _HostileDict()}
    assert _artifact_digest(artifact) is None
