"""Trust-boundary validation for forager sanitized errors."""

from __future__ import annotations

import pathlib
from typing import Any, cast

import pytest

from alberta_framework.benchmarks.forager import (
    ForagerEnvConfig,
    _require_exact_str,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("value", _StringSubclass("x"))
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("val")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        _require_exact_str("value", evil2)
    assert "EvilStr" not in str(exc2.value)


def test_forager_env_config_unknown_preset_sanitized() -> None:
    with pytest.raises(ValueError, match="unknown Forager preset") as exc:
        ForagerEnvConfig(preset=cast(Any, "evil_preset"))
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_preset'" in msg


def test_forager_env_config_hostile_preset() -> None:
    evil = _EvilStr("evil_preset")
    with pytest.raises(ValueError, match="must be an exact string"):
        ForagerEnvConfig(preset=cast(Any, evil))


def test_forager_env_config_unknown_observation_type_sanitized() -> None:
    with pytest.raises(ValueError, match="unknown observation_type") as exc:
        ForagerEnvConfig(preset="relearning", observation_type=cast(Any, "evil_obs"))
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_obs'" in msg


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager.py")
    text = p.read_text(encoding="utf-8")
    assert "unknown Forager preset {self.preset!r}" not in text
    assert "unknown observation_type {self.observation_type!r}" not in text
    assert "unknown preset {preset!r}" not in text
    assert "unknown Forager preset '{host_preset}'" in text
    assert "unknown observation_type '{host_obs}'" in text
    assert "unknown preset '{host_preset}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    cfg = ForagerEnvConfig(preset="relearning")
    assert cfg.preset == "relearning"
