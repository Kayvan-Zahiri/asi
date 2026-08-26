"""Hostile exact JSON gates for recurring-feature seed-summary validation."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import _mapping

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_mapping_rejects_configuration_dict_subclass_without_iter_hooks() -> None:
    parent: dict[str, object] = {"configuration": HostileDict({"pairs": []})}
    errors: list[str] = []
    HostileDict.calls = 0
    assert _mapping(parent, "configuration", "scientific_payload", errors) is None
    assert errors == ["scientific_payload.configuration must be an object"]
    assert HostileDict.calls == 0
