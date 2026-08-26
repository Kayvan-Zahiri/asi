import json
from pathlib import Path

import pytest

from alberta_framework.benchmarks.rule_discovery_summary import (
    CHAMPION,
    SCREEN_ARMS,
    build_rule_discovery_summary,
)


def _write_shard(path: Path, arm: str, seed: int, n_tasks: int) -> None:
    payload = {
        "config_name": arm,
        "seed": seed,
        "per_task_accuracy": [0.85] * n_tasks,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rule_discovery_summary_validates_arm_name_and_task_count(tmp_path: Path) -> None:
    screen_dir = tmp_path / "screen"
    confirm_dir = tmp_path / "confirm"
    screen_dir.mkdir()
    confirm_dir.mkdir()

    # Populate valid screen directory (60 tasks)
    for name in SCREEN_ARMS:
        _write_shard(screen_dir / f"{name}_seed0.json", name, 0, 60)

    # Populate valid confirm directory (200 tasks)
    for name in ("disc_r1_pscale_norms", CHAMPION):
        _write_shard(confirm_dir / f"{name}_seed0.json", name, 0, 200)

    res = build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))
    assert res["schema"] == "asi.rule_discovery.real_screen.v2"
    assert "disc_r1" in res["screen_60_task"]

    # Reject mismatched arm name
    bad_shard = screen_dir / "disc_r1_seed0.json"
    _write_shard(bad_shard, "sigma0_shiftnorm_d099", 0, 60)
    with pytest.raises(ValueError, match="does not match expected arm"):
        build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))

    # Fix arm name but use wrong task count (e.g. 200 tasks in screen)
    _write_shard(bad_shard, "disc_r1", 0, 200)
    with pytest.raises(ValueError, match="tasks, expected 60"):
        build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))

    # Fix screen shard, then test wrong task count in confirm (e.g. 60 tasks in confirm)
    _write_shard(bad_shard, "disc_r1", 0, 60)
    bad_confirm = confirm_dir / "disc_r1_pscale_norms_seed0.json"
    _write_shard(bad_confirm, "disc_r1_pscale_norms", 0, 60)
    with pytest.raises(ValueError, match="tasks, expected 200"):
        build_rule_discovery_summary(screen_dir, confirm_dir, seeds=(0,))
