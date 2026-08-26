"""Hostile mapping validation for lifetime gauntlet legacy migration."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook executed")


def test_migrate_legacy_rejects_hostile_dict_before_field_manifest() -> None:
    from alberta_framework.streams.gauntlet import migrate_legacy_lifetime_gauntlet_state

    payload = _HostileDict(
        {
            "key": None,
            "step_count": None,
            "w_fresh": None,
            "w_c": None,
            "w_d": None,
        }
    )
    _HostileDict.calls = 0
    with pytest.raises(TypeError, match="exact dict"):
        migrate_legacy_lifetime_gauntlet_state(payload, stream=object())  # type: ignore[arg-type]
    assert _HostileDict.calls == 0
