"""Judge prompt dispatch is driven by RunConfig fields, not env vars.

These tests fence the (PROMPT_STYLE, INCLUDE_NL_VIO) module globals and the
release/paper branch of ``get_system_prompt`` so that future edits can't
re-introduce an AGENTRX_* env-var fallback or swap the dispatched template.
"""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def judge(monkeypatch):
    """Import judge.py with NO AGENTRX_* env vars present.

    The module no longer reads them at import time, but we still strip them
    so we'd catch a regression that re-introduced env-var defaults.
    """
    for var in ("AGENTRX_JUDGE_PROMPT_STYLE", "AGENTRX_NO_NL_VIO"):
        monkeypatch.delenv(var, raising=False)
    import agentrx.judge.judge as j
    importlib.reload(j)
    # Required globals normally set by run.py::run_judge.
    j.PROMPT_MODE = "combined"
    j.EXECUTION_MODE = "violations-after"
    return j


def test_module_defaults_match_paper_release(judge):
    assert judge.PROMPT_STYLE == "release"
    assert judge.INCLUDE_NL_VIO is True


def test_module_has_no_env_var_globals(judge):
    # Old shim symbols must not exist.
    assert not hasattr(judge, "JUDGE_PROMPT_STYLE")
    assert not hasattr(judge, "NO_NL_VIO")


def test_release_prompt_section_order(judge):
    judge.PROMPT_STYLE = "release"
    prompt = judge.get_system_prompt(invariants_violation_context="x")
    # Section markers expected in the originally-released template.
    assert "GIVEN INPUT" in prompt
    assert "FAILURE TAXONOMY" in prompt
    assert prompt.index("GIVEN INPUT") < prompt.index("FAILURE TAXONOMY")


def test_paper_mirror_prompt_distinct_signature(judge):
    judge.PROMPT_STYLE = "paper"
    prompt = judge.get_system_prompt(invariants_violation_context="x")
    # Paper-mirror builder uses the "Expert Failure-Categorization Judge" lead.
    assert "Expert Failure-Categorization Judge" in prompt


def test_release_vs_paper_prompts_differ(judge):
    judge.PROMPT_STYLE = "release"
    rp = judge.get_system_prompt(invariants_violation_context="x")
    judge.PROMPT_STYLE = "paper"
    pp = judge.get_system_prompt(invariants_violation_context="x")
    assert rp != pp, "release and paper prompt builders must produce distinct strings"


def test_invariant_context_filter_respects_module_global(judge, tmp_path, monkeypatch):
    """When INCLUDE_NL_VIO=False, nl_check entries are dropped before judge sees them."""
    # Loader layout: <VIOLATION_CONTEXT_DIR>/<task_id>/violations_<domain>.json
    ctx_dir = tmp_path / "judge_context"
    task_subdir = ctx_dir / "T0"
    task_subdir.mkdir(parents=True)
    domain_tag = "unittest"
    (task_subdir / f"violations_{domain_tag}.json").write_text(
        '[{"check_type": "nl_check", "msg": "drop me"},'
        ' {"check_type": "python_check", "msg": "keep me"}]'
    )
    monkeypatch.setattr(judge, "VIOLATION_CONTEXT_DIR", str(ctx_dir))
    monkeypatch.setattr(judge, "DOMAIN", domain_tag)

    monkeypatch.setattr(judge, "INCLUDE_NL_VIO", True)
    full = judge.load_invariant_violation_context("T0")
    assert full is not None and len(full) == 2

    monkeypatch.setattr(judge, "INCLUDE_NL_VIO", False)
    filtered = judge.load_invariant_violation_context("T0")
    assert filtered is not None and len(filtered) == 1
    assert filtered[0]["check_type"] == "python_check"
