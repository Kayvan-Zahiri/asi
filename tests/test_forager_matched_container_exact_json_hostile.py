"""Hostile exact JSON gates for matched-container scorer JSON."""

from __future__ import annotations

import json

import pytest

from alberta_framework.benchmarks._forager_matched_container import (
    ContainerError,
    _strict_json,
)

pytestmark = pytest.mark.unit


class HostileDict(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not run")


def test_strict_json_rejects_dict_subclass_without_iter_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HostileDict.calls = 0
    monkeypatch.setattr(json, "loads", lambda *_args, **_kwargs: HostileDict({"ok": True}))
    with pytest.raises(ContainerError, match="scorer result must be a JSON object"):
        _strict_json(b"{}")
    assert HostileDict.calls == 0
