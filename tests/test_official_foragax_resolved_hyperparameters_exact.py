"""Exact dict/list gates for official resolved-hyperparameter validation."""

from __future__ import annotations

import hashlib
import json

import pytest

from alberta_framework.benchmarks import official_foragax as mod


def _probe(payload: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "resolved_hyperparameters": payload,
        "resolved_hyperparameters_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def test_resolved_hyperparameters_accept_exact_dict_tree() -> None:
    probe = _probe({"alpha": 0.1, "nested": {"path": "relative/ok"}})
    assert mod._validated_resolved_hyperparameters(probe)["alpha"] == 0.1


def test_resolved_hyperparameters_reject_mapping_subclass_without_hooks() -> None:
    class HostileDict(dict):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileDict.__iter__ must not run")

    HostileDict.calls = 0
    with pytest.raises(
        mod.OfficialForagaxValidationError, match="resolved hyperparameters"
    ):
        mod._validated_resolved_hyperparameters(
            {
                "resolved_hyperparameters": HostileDict({"alpha": 0.1}),
                "resolved_hyperparameters_sha256": "0" * 64,
            }
        )
    assert HostileDict.calls == 0
