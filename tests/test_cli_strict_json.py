"""Tests for CLI strict JSON output."""

import io
import json
from unittest.mock import patch

import pytest

from alberta_framework.cli import _print_json, step1_smoke_main, step2_smoke_main


def test_print_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        _print_json({"final_window_mse": float("nan"), "finite": False})

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        _print_json({"final_window_mse": float("inf"), "finite": False})


def test_print_json_emits_valid_json() -> None:
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        _print_json({"final_window_mse": 0.05, "finite": True})
    output = buf.getvalue()
    parsed = json.loads(output)
    assert parsed == {"final_window_mse": 0.05, "finite": True}


def test_step1_and_step2_smoke_main_cli_execution() -> None:
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code1 = step1_smoke_main(["--steps", "16", "--seed", "0", "--final-window", "8"])
    assert code1 == 0
    parsed1 = json.loads(buf.getvalue())
    assert parsed1["finite"] is True

    buf2 = io.StringIO()
    with patch("sys.stdout", buf2):
        code2 = step2_smoke_main(["--steps", "16", "--seed", "0", "--final-window", "8"])
    assert code2 == 0
    parsed2 = json.loads(buf2.getvalue())
    assert parsed2["finite"] is True
