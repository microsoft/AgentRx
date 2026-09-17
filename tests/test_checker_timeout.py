"""python_check timeout fences infinite loops (commit a6a8471).

The SIGALRM path was Unix-only. The current implementation uses a daemon
thread with .join(timeout). This test verifies that:

  - An infinite-loop check raises (telemetry caught) within the configured
    budget instead of hanging the process.
  - A normal passing/failing check still works.
  - The module no longer imports ``signal``.
"""
from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setenv("AGENTRX_PYCHECK_TIMEOUT_SEC", "2")
    import agentrx.invariants.checker as ck
    importlib.reload(ck)
    assert ck.PYCHECK_TIMEOUT_SEC == 2
    return ck


def _make_verifier(ck):
    """Instantiate AllVerifier without running its full __init__."""
    inst = ck.AllVerifier.__new__(ck.AllVerifier)
    inst.telemetry = []
    inst.policy_text = ""
    inst.total_python_checks = 0
    return inst


def _make_invariant(name: str, body_lines: list[str]) -> dict:
    return {
        "assertion_name": name,
        "check_type": "python_check",
        "python_check": {
            "function_name": "check",
            "code_lines": ["def check(traj, step_pos):", *("    " + ln for ln in body_lines)],
        },
        "invariant_type": "test",
        "severity": "medium",
    }


def test_module_uses_threading_not_signal(checker):
    assert "threading" in dir(checker)
    assert "signal" not in dir(checker), (
        "checker.py must not re-introduce SIGALRM: it is Unix-only and breaks "
        "Windows-WSL reproduction collaborators"
    )


def test_inf_loop_times_out_within_budget(checker):
    inst = _make_verifier(checker)
    inv = _make_invariant("inf_loop", ["while True:", "    pass", "return True"])
    traj = {"task_id": "T0", "steps": [{"index": 0, "substeps": []}]}

    t0 = time.perf_counter()
    v = inst._check_python_invariant("T0", traj, 0, inv, [])
    elapsed = time.perf_counter() - t0

    # Budget is 2s; allow 3x grace for slow CI.
    assert elapsed < checker.PYCHECK_TIMEOUT_SEC * 3, (
        f"timeout did not fire: took {elapsed:.2f}s with budget {checker.PYCHECK_TIMEOUT_SEC}s"
    )
    assert v is None, "infinite-loop check should become a swallowed exception, not a violation"
    last = inst.telemetry[-1]
    assert last.success is False
    err = last.error or ""
    assert "PyCheckTimeout" in err or "exceeded" in err, f"expected timeout marker, got: {err!r}"


def test_passing_check_returns_no_violation(checker):
    inst = _make_verifier(checker)
    inv = _make_invariant("ok", ["return True"])
    traj = {"task_id": "T1", "steps": [{"index": 0, "substeps": []}]}
    v = inst._check_python_invariant("T1", traj, 0, inv, [])
    assert v is None
    assert inst.telemetry[-1].success is True


def test_failing_check_produces_violation(checker):
    inst = _make_verifier(checker)
    inv = _make_invariant("fail", ["return False"])
    traj = {"task_id": "T2", "steps": [{"index": 0, "substeps": []}]}
    v = inst._check_python_invariant("T2", traj, 0, inv, [])
    assert v is not None
    assert v.assertion_name == "fail"
    assert inst.telemetry[-1].success is True
