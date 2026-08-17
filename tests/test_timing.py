"""Tests for timing utilities."""

import time

import pytest

from alberta_framework.utils.timing import Timer, format_duration


def test_format_duration_formatting() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(90.5) == "1m 30.50s"
    assert format_duration(3665) == "1h 1m 5.00s"
    assert format_duration(59.999) == "1m 0.00s"
    assert format_duration(3599.999) == "1h 0m 0.00s"


def test_timer_successful_run_message() -> None:
    messages: list[str] = []
    with Timer("Training", print_fn=messages.append) as timer:
        time.sleep(0.001)

    assert len(messages) == 1
    assert messages[0].startswith("Training completed in ")
    assert timer.duration > 0.0
    assert timer.end_time >= timer.start_time


def test_timer_failed_run_message() -> None:
    messages: list[str] = []
    with pytest.raises(RuntimeError, match="simulated failure"):
        with Timer("Training", print_fn=messages.append) as timer:
            time.sleep(0.001)
            raise RuntimeError("simulated failure")

    assert len(messages) == 1
    assert messages[0].startswith("Training failed after ")
    assert timer.duration > 0.0
    assert timer.end_time >= timer.start_time


def test_timer_silent_mode() -> None:
    messages: list[str] = []
    with Timer("Silent", verbose=False, print_fn=messages.append) as timer:
        time.sleep(0.001)

    assert len(messages) == 0
    assert timer.duration > 0.0
