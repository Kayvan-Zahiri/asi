"""Hostile mapping validation for legacy normalizer migration."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook executed")


def test_migrate_legacy_rejects_hostile_dict_before_field_manifest() -> None:
    from alberta_framework.core.normalizers import migrate_legacy_normalizer_state

    payload = _HostileDict(
        {
            "decay": 0.1,
            "mean": [0.0],
            "var": [1.0],
            "sample_count": 1.0,
        }
    )
    _HostileDict.calls = 0
    with pytest.raises(TypeError, match="exact dict"):
        migrate_legacy_normalizer_state(payload, normalizer_type="EMANormalizer")
    assert _HostileDict.calls == 0
