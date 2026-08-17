"""Hostile input and boundary validation for matched candidate dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    CandidateRuntimeBinding,
    EnvironmentRNGContract,
    ForagerMatchedProtocolError,
    ObservationAccess,
)


def test_observation_access_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="invalid access_mode"):
        ObservationAccess(
            access_mode="invalid_mode",  # type: ignore[arg-type]
            observation_type="full",
            aperture_size=3,
            privileged_fields=(),
        )

    with pytest.raises(
        ForagerMatchedProtocolError,
        match="aperture_size must be -1 or a positive integer",
    ):
        ObservationAccess(
            access_mode="partial_observation",
            observation_type="full",
            aperture_size=0,
            privileged_fields=(),
        )


def test_environment_rng_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="identity must be a non-empty string"):
        EnvironmentRNGContract(
            identity="",
            schedule_sha256="0" * 64,
        )


def test_candidate_runtime_binding_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="qualification_trust_anchor_identity must be a non-empty string",
    ):
        CandidateRuntimeBinding(
            image_sha256=valid_sha,
            runtime_profile_sha256=valid_sha,
            task_identity_sha256=valid_sha,
            qualified_capability_descriptor_sha256=valid_sha,
            capability_qualification_receipt_sha256=valid_sha,
            qualification_trust_anchor_identity="",
        )
