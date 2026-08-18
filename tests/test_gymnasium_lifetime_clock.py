"""Saturating lifetime clock keeps GymnasiumStream step_count non-negative."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

# Mock gymnasium if absent so pure stream unit tests run deterministically
if "gymnasium" not in sys.modules:
    dummy_gym = types.ModuleType("gymnasium")
    dummy_spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, shape: tuple[int, ...] = (2,), dtype: Any = np.float32) -> None:
            self.shape = shape
            self.dtype = dtype

    dummy_spaces.Box = Box  # type: ignore[attr-defined]
    dummy_spaces.Discrete = type("Discrete", (), {})  # type: ignore[attr-defined]
    dummy_spaces.MultiDiscrete = type("MultiDiscrete", (), {})  # type: ignore[attr-defined]
    dummy_spaces.Space = object  # type: ignore[attr-defined]
    dummy_gym.spaces = dummy_spaces  # type: ignore[attr-defined]
    sys.modules["gymnasium"] = dummy_gym
    sys.modules["gymnasium.spaces"] = dummy_spaces

from alberta_framework.streams.gymnasium import (  # noqa: E402
    _INT32_MAX,
    GymnasiumStream,
    PredictionMode,
    TDStream,
)


class DummyEnv:
    def __init__(self) -> None:
        import gymnasium.spaces

        self.observation_space = gymnasium.spaces.Box(shape=(2,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(shape=(2,), dtype=np.float32)

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return np.zeros(2, dtype=np.float32), 1.0, False, False, {}

    def reset(self, *, seed: object = None) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros(2, dtype=np.float32), {}


def test_gymnasium_stream_step_count_saturates_at_int32_max() -> None:
    env: Any = DummyEnv()
    stream = GymnasiumStream(
        env,
        mode=PredictionMode.REWARD,
        policy=lambda obs: np.zeros(2, dtype=np.float32),
        seed=0,
    )
    stream._step_count = _INT32_MAX
    next(stream)
    assert stream.step_count == _INT32_MAX
    assert stream.step_count >= 0


def test_td_stream_step_count_saturates_at_int32_max() -> None:
    env: Any = DummyEnv()
    stream = TDStream(
        env,
        policy=lambda obs: np.zeros(2, dtype=np.float32),
        seed=0,
    )
    stream._step_count = _INT32_MAX
    next(stream)
    assert stream.step_count == _INT32_MAX
    assert stream.step_count >= 0
