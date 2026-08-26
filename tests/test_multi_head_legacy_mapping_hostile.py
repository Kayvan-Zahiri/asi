"""Hostile mapping validation for legacy multi-head migration."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook executed")


def test_migrate_legacy_rejects_hostile_dict_before_field_manifest() -> None:
    from alberta_framework.core.multi_head_learner import migrate_legacy_multi_head_mlp_state

    payload = _HostileDict({"head_params": ()})
    _HostileDict.calls = 0
    with pytest.raises(TypeError, match="exact dict"):
        migrate_legacy_multi_head_mlp_state(payload)
    assert _HostileDict.calls == 0
