"""Hostile input and boundary validation for Forager matched qualification records."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_qualification import (
    ForagerMatchedQualificationError,
    ProbeInvocation,
)


def _make_probe_invocation() -> ProbeInvocation:
    return ProbeInvocation(
        candidate_id="cand_1",
        source_key="alberta",
        source_root=Path("/source"),
        probe_path=Path("/probe.py"),
        probe_sha256="a" * 64,
        configuration=Path("/config.json"),
        configuration_sha256="b" * 64,
        entrypoint_path="main.py",
        entrypoint_sha256="c" * 64,
        entrypoint_family="ppo",
        implementation_kind="jax",
        invocation_style="module_entrypoint",
        result_root="results",
        seed_transport="flag",
        expected_agent="ppo_agent",
        horizon=1000,
    )


def test_probe_invocation_valid_construction() -> None:
    inv = _make_probe_invocation()
    assert inv.candidate_id == "cand_1"
    assert inv.horizon == 1000


def test_probe_invocation_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedQualificationError, match="candidate_id must be a non-empty string"
    ):
        inv = _make_probe_invocation()
        ProbeInvocation(
            candidate_id="",
            source_key=inv.source_key,
            source_root=inv.source_root,
            probe_path=inv.probe_path,
            probe_sha256=inv.probe_sha256,
            configuration=inv.configuration,
            configuration_sha256=inv.configuration_sha256,
            entrypoint_path=inv.entrypoint_path,
            entrypoint_sha256=inv.entrypoint_sha256,
            entrypoint_family=inv.entrypoint_family,
            implementation_kind=inv.implementation_kind,
            invocation_style=inv.invocation_style,
            result_root=inv.result_root,
            seed_transport=inv.seed_transport,
            expected_agent=inv.expected_agent,
            horizon=inv.horizon,
        )

    with pytest.raises(ForagerMatchedQualificationError, match="horizon must be a positive int"):
        inv = _make_probe_invocation()
        ProbeInvocation(
            candidate_id=inv.candidate_id,
            source_key=inv.source_key,
            source_root=inv.source_root,
            probe_path=inv.probe_path,
            probe_sha256=inv.probe_sha256,
            configuration=inv.configuration,
            configuration_sha256=inv.configuration_sha256,
            entrypoint_path=inv.entrypoint_path,
            entrypoint_sha256=inv.entrypoint_sha256,
            entrypoint_family=inv.entrypoint_family,
            implementation_kind=inv.implementation_kind,
            invocation_style=inv.invocation_style,
            result_root=inv.result_root,
            seed_transport=inv.seed_transport,
            expected_agent=inv.expected_agent,
            horizon=0,
        )


@pytest.mark.parametrize("field", ["probe_sha256", "configuration_sha256", "entrypoint_sha256"])
def test_probe_invocation_rejects_nonhex_sha256(field: str) -> None:
    inv = _make_probe_invocation()
    with pytest.raises(ForagerMatchedQualificationError, match=f"{field} must be"):
        replace(inv, **{field: "z" * 64})


def test_qualification_bundle_rejects_mapping_subclass_without_iter_hooks() -> None:
    from alberta_framework.benchmarks.forager_matched_qualification import (
        MatchedCurrentQualificationBundle,
    )

    class HostileDict(dict):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileDict.__iter__ must not run")

    HostileDict.calls = 0
    with pytest.raises(
        ForagerMatchedQualificationError,
        match="manifest must be an exact dict or MappingProxyType",
    ):
        MatchedCurrentQualificationBundle(
            output_root=Path("out"),
            cpu_qualification_root=Path("cpu"),
            rng_parity_qualification_root=Path("rng"),
            runtime_qualification=None,
            candidate_qualifications={},
            candidate_assets={},
            manifest=HostileDict(),
            manifest_bytes=b"{}",
            manifest_sha256="a" * 64,
        )
    assert HostileDict.calls == 0
