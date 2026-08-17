"""Hostile input and boundary validation for PavlovianPhase dataclass."""

from __future__ import annotations

from typing import Any, cast

import pytest

from alberta_framework.streams.pavlovian import PavlovianPhase


def _make_phase(**kwargs: Any) -> PavlovianPhase:
    base: dict[str, Any] = {
        "name": "acq",
        "n_steps": 100,
        "cs_us_contingency": 1.0,
        "cs_active": (0,),
        "compound_index": -1,
    }
    base.update(kwargs)
    ctor: Any = PavlovianPhase
    return cast(PavlovianPhase, ctor(**base))


def test_pavlovian_phase_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _make_phase(name="")

    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        _make_phase(n_steps=0)

    with pytest.raises(ValueError, match=r"cs_us_contingency must be a float in \[0\.0, 1\.0\]"):
        _make_phase(cs_us_contingency=1.5)

    with pytest.raises(TypeError, match="cs_active must be a tuple"):
        _make_phase(cs_active=[0])

    with pytest.raises(ValueError, match="compound_index must be an integer >= -1"):
        _make_phase(compound_index=-2)
