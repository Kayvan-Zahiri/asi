"""Hostile dict identity gate for scorecard reward aggregation helpers."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks import reference_life_scorecard as scorecard

pytestmark = pytest.mark.unit


class _HostileDict(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        if key == "reward_sum":
            return 1.0
        return super().get(key, default)

    def __bool__(self) -> bool:
        raise AssertionError("hostile mapping truth hook executed")


def test_reward_sum_rejects_hostile_outcome_before_get() -> None:
    record: dict[str, object] = {"outcome": _HostileDict({"reward_sum": 1.0})}
    with pytest.raises(ValueError, match="lacks an outcome"):
        scorecard._reward_sum(record)


def test_summarize_rejects_hostile_record_before_identity() -> None:
    plan = scorecard.build_development_plan()
    hostile_record = _HostileDict(
        {
            "environment_kind": "switching_two_state",
            "arm": "prototype",
            "seed": 0,
            "status": "completed",
            "outcome": {"reward_sum": 0.0},
        }
    )
    with pytest.raises(ValueError, match="every run record must be an object"):
        scorecard._summarize_validated_run_records(plan, [hostile_record])
