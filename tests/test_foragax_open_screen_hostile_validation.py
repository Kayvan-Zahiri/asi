"""Hostile input and boundary validation for foragax open screen dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.foragax_open_screen import (
    FrozenConfiguration,
    ProcessCapture,
    ProtocolSnapshot,
    ScreenError,
    load_frozen_protocol,
)

_ROOT = Path(__file__).resolve().parent.parent


def test_frozen_configuration_rejects_invalid_inputs() -> None:
    with pytest.raises(ScreenError, match="path must be a non-empty string"):
        FrozenConfiguration(
            path="",
            sha256="a" * 64,
            agent="dqn",
            entrypoint="main.py",
        )

    with pytest.raises(ScreenError, match="sha256 must be a lowercase SHA-256 hex digest"):
        FrozenConfiguration(
            path="agent.py",
            sha256="invalid",
            agent="dqn",
            entrypoint="main.py",
        )


def test_protocol_snapshot_rejects_invalid_inputs() -> None:
    protocol = load_frozen_protocol(_ROOT / "outputs/forager/fov_baseline_screening_cpu_v3")
    with pytest.raises(ScreenError, match="protocol must be a FrozenProtocol"):
        ProtocolSnapshot(
            protocol=None,  # type: ignore[arg-type]
            inventory=(),
            inventory_sha256="a" * 64,
        )

    with pytest.raises(ScreenError, match="inventory must be a tuple"):
        ProtocolSnapshot(
            protocol=protocol,
            inventory=[],  # type: ignore[arg-type]
            inventory_sha256="a" * 64,
        )


def test_process_capture_rejects_invalid_inputs() -> None:
    with pytest.raises(ScreenError, match="returncode must be an integer"):
        ProcessCapture(
            returncode=True,
            stdout=b"",
            stderr=b"",
        )

    with pytest.raises(ScreenError, match="stdout must be bytes"):
        ProcessCapture(returncode=0, stdout=bytearray(), stderr=b"")


def test_frozen_configuration_rejects_string_subclass_before_len_hook() -> None:
    calls = 0

    class HostileString(str):
        def __len__(self) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("length hook reached")

    with pytest.raises(ScreenError, match="path must be a non-empty string"):
        FrozenConfiguration(
            path=HostileString("config.json"),
            sha256="a" * 64,
            agent="dqn",
            entrypoint="main.py",
        )
    assert calls == 0

    with pytest.raises(ScreenError, match="stdout must be bytes"):
        ProcessCapture(
            returncode=0,
            stdout="not bytes",  # type: ignore[arg-type]
            stderr=b"",
        )


def test_open_screen_probe_rejects_mapping_subclass_task_without_iter_hooks() -> None:
    from alberta_framework.benchmarks import _foragax_open_screen_probe as probe

    class HostileDict(dict):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileDict.__iter__ must not run")

    HostileDict.calls = 0
    with pytest.raises(RuntimeError, match="protocol"):
        probe._validate_task_intake({"task": HostileDict({"seeds": [0], "steps": 1})})
    assert HostileDict.calls == 0


def test_open_screen_probe_rejects_list_subclass_seeds_without_iter_hooks() -> None:
    from alberta_framework.benchmarks import _foragax_open_screen_probe as probe

    class HostileList(list):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileList.__iter__ must not run")

    HostileList.calls = 0
    with pytest.raises(RuntimeError, match="protocol"):
        probe._validate_task_intake({"task": {"seeds": HostileList([0]), "steps": 1}})
    assert HostileList.calls == 0
