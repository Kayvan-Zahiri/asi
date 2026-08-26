"""Hostile dict gate for reference-life canonical JSON digest."""

from __future__ import annotations

import pytest

from alberta_framework.reference_life import _canonical_json

pytestmark = pytest.mark.unit


class HostileDict(dict):
    pass


def test_canonical_json_rejects_non_exact_dict_after_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"a": 1}

    def hostile_loads(_text: str) -> object:
        return HostileDict(payload)

    monkeypatch.setattr("alberta_framework.reference_life.json.loads", hostile_loads)
    with pytest.raises(ValueError, match="exact JSON object"):
        _canonical_json(payload)
