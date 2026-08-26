"""Hostile validation for UPGD IPMNIST trust boundary."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.upgd_ipmnist import (
    _require_exact_str,
    _validated_hyperparameter,
    resolve_hyperparameters,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_validated_hyperparameter_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("step_size")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _validated_hyperparameter(evil, 1.0)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_validated_hyperparameter_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _validated_hyperparameter(_StringSubclass("step_size"), 1.0)


def test_validated_hyperparameter_sanitized() -> None:
    with pytest.raises(ValueError, match="hyperparameter") as exc:
        _validated_hyperparameter("bad_param", "not_a_number")
    assert "!r" not in str(exc.value)


def test_resolve_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("bad_learner")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        resolve_hyperparameters(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_resolve_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        resolve_hyperparameters(_StringSubclass("bad_learner"))


def test_resolve_sanitized() -> None:
    with pytest.raises(ValueError, match="unknown learner") as exc:
        resolve_hyperparameters("bad_learner")
    assert "!r" not in str(exc.value)
    assert "bad_learner" in str(exc.value)


def test_duplicate_key_sanitized() -> None:
    import tempfile
    from pathlib import Path

    from alberta_framework.benchmarks.upgd_ipmnist import _strict_json_object

    dup_json = '{"a": 1, "a": 2}'
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dup.json"
        p.write_text(dup_json)
        with pytest.raises(ValueError, match="duplicate JSON key") as exc:
            _strict_json_object(p)
        assert "!r" not in str(exc.value)
        assert "a" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/upgd_ipmnist.py").read_text()
    assert "!r" not in text


def test_numeric_matrix_rejects_list_subclass() -> None:
    from alberta_framework.evaluation import upgd_ipmnist_nonpromoting as mod

    class HostileList(list):
        calls = 0

        def __iter__(self):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileList.__iter__ must not run")

    errors: list[str] = []
    HostileList.calls = 0
    assert (
        mod._numeric_matrix(
            HostileList([[0.0]]),
            field="logits",
            row_count=1,
            column_count=1,
            errors=errors,
            lower=0.0,
            upper=None,
            lower_inclusive=True,
        )
        is None
    )
    assert any("logits must have shape" in error for error in errors)
    assert HostileList.calls == 0


def test_decode_strict_json_object_rejects_dict_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    from alberta_framework.evaluation import upgd_ipmnist_nonpromoting as mod

    class HostileDict(dict):
        calls = 0

        def __contains__(self, key):  # type: ignore[override]
            type(self).calls += 1
            raise AssertionError("HostileDict.__contains__ must not run")

    def fake_loads(raw: object, *args: object, **kwargs: object) -> HostileDict:
        return HostileDict({"schema": "x"})

    monkeypatch.setattr(mod.json, "loads", fake_loads)
    HostileDict.calls = 0
    with pytest.raises(ValueError, match="JSON object"):
        mod._decode_strict_json_object(b'{"schema":"x"}')
    assert HostileDict.calls == 0
