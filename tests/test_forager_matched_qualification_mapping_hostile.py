"""Hostile mapping validation for matched qualification bundles."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_qualification import (
    ForagerMatchedQualificationError,
    MatchedCurrentQualificationBundle,
)

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook executed")


def test_bundle_rejects_hostile_candidate_qualifications_before_hooks() -> None:
    _HostileDict.calls = 0
    with pytest.raises(ForagerMatchedQualificationError, match="candidate mappings"):
        MatchedCurrentQualificationBundle(
            output_root=Path("/tmp/out"),
            cpu_qualification_root=Path("/tmp/cpu"),
            rng_parity_qualification_root=Path("/tmp/rng"),
            runtime_qualification=object(),
            candidate_qualifications=_HostileDict(),
            candidate_assets={},
            manifest={"schema": "test"},
            manifest_bytes=b'{"schema":"test"}',
            manifest_sha256="a" * 64,
        )
    assert _HostileDict.calls == 0
