"""Fail-closed interval bound checks for shared measurement validation."""

from __future__ import annotations

import math

import pytest

from alberta_framework.evaluation._measurement_validation import validate_interval_bounds


def test_validate_interval_bounds_accepts_finite_ordered_interval() -> None:
    validate_interval_bounds(lower=0.1, upper=0.9, confidence_level=0.95)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (math.nan, 1.0),
        (0.0, math.nan),
        (math.inf, math.inf),
        (-math.inf, 0.0),
        (math.nan, math.nan),
    ],
)
def test_validate_interval_bounds_rejects_nonfinite_bounds(
    lower: float, upper: float
) -> None:
    """IEEE ``lower > upper`` is always false for NaN, so finiteness must be gated."""
    with pytest.raises(ValueError, match="must be finite"):
        validate_interval_bounds(lower=lower, upper=upper, confidence_level=0.95)


def test_validate_interval_bounds_rejects_nonfinite_confidence() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_interval_bounds(lower=0.0, upper=1.0, confidence_level=math.nan)


def test_validate_interval_bounds_still_rejects_inverted_finite_bounds() -> None:
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        validate_interval_bounds(lower=0.9, upper=0.1, confidence_level=0.95)
