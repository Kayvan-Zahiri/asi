"""Hostile input and boundary validation for EvidenceSpec dataclass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    EvidenceSpec,
    ValidationResult,
)


class DummyValidationResult(ValidationResult):
    @property
    def valid(self) -> bool:
        return True

    @property
    def accepted(self) -> bool:
        return True

    @property
    def errors(self) -> tuple[str, ...]:
        return ()


def _make_spec(**kwargs: Any) -> EvidenceSpec:
    base: dict[str, Any] = {
        "name": "spec_1",
        "claim_scope": "scope_1",
        "evidence_class": "scientific_evidence",
        "evidence_level": "gold",
        "promotes_scientific_claim": True,
        "relative_path": Path("outputs/test.json"),
        "expected_schema": "schema_v1",
        "command_argv": ("run.py",),
        "protocol": {},
        "configuration": {},
        "seeds": {},
        "thresholds": {},
        "limitations": (),
        "source_paths": (Path("src/test.py"),),
        "required_environment_fields": (),
        "loader": lambda p: {},
        "validator": lambda p: DummyValidationResult(),
    }
    base.update(kwargs)
    return EvidenceSpec(**base)


def test_evidence_spec_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _make_spec(name="")

    with pytest.raises(TypeError, match="promotes_scientific_claim must be a bool"):
        _make_spec(promotes_scientific_claim=1)

    with pytest.raises(TypeError, match="relative_path must be a pathlib.Path"):
        _make_spec(relative_path="outputs/test.json")

    with pytest.raises(TypeError, match="source_paths must contain only pathlib.Path objects"):
        _make_spec(source_paths=("src/test.py",))

    with pytest.raises(TypeError, match="loader must be callable"):
        _make_spec(loader=None)
