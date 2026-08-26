"""Hostile exact JSON gates for final-analysis bundle validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_final_analysis import (
    ContentVerifiedFinalAnalysisBundle,
    ForagerMatchedFinalAnalysisError,
)

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_content_verified_bundle_rejects_manifest_dict_subclass() -> None:
    bundle = object.__new__(ContentVerifiedFinalAnalysisBundle)
    object.__setattr__(bundle, "output_root", Path("."))
    object.__setattr__(bundle, "manifest", HostileDict({"a": 1}))
    object.__setattr__(bundle, "seal_content", object())
    object.__setattr__(bundle, "evaluation_score_evidence", object())
    object.__setattr__(bundle, "evaluation_verification_request", object())
    object.__setattr__(bundle, "open_bindings_cache", object())
    object.__setattr__(bundle, "evaluation_bindings_cache", object())
    object.__setattr__(bundle, "analysis_runtime_source", {"runtime": "x"})
    object.__setattr__(bundle, "contract", object())
    object.__setattr__(bundle, "result", object())
    HostileDict.calls = 0
    with pytest.raises(
        ForagerMatchedFinalAnalysisError,
        match="manifest must be a Mapping",
    ):
        ContentVerifiedFinalAnalysisBundle.__post_init__(bundle)
    assert HostileDict.calls == 0
