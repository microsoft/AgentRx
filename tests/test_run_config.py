"""RunConfig is the centralized contract for judge-stage knobs.

These tests fence its public surface so that future refactors can't
silently change paper-faithful defaults or weaken the validation that
catches bogus config at construction time.
"""
from __future__ import annotations

import dataclasses

import pytest

from agentrx.pipeline.profiles import (
    PAPER_DEFAULT,
    PAPER_MIRROR_DEFAULT,
    RunConfig,
)


def test_paper_default_field_values():
    assert PAPER_DEFAULT.prompt_mode == "combined"
    assert PAPER_DEFAULT.exec_mode == "violations-after"
    assert PAPER_DEFAULT.with_context is True
    assert PAPER_DEFAULT.num_runs == 3
    assert PAPER_DEFAULT.prompt_style == "release"
    assert PAPER_DEFAULT.include_nl_check_violations is True


def test_paper_mirror_default_differs_only_in_prompt_style():
    assert PAPER_MIRROR_DEFAULT.prompt_style == "paper"
    # Every other field matches the release default.
    for f in dataclasses.fields(RunConfig):
        if f.name == "prompt_style":
            continue
        assert getattr(PAPER_MIRROR_DEFAULT, f.name) == getattr(PAPER_DEFAULT, f.name), (
            f"PAPER_MIRROR_DEFAULT.{f.name} drifted from PAPER_DEFAULT"
        )


def test_runconfig_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PAPER_DEFAULT.prompt_mode = "baseline"  # type: ignore[misc]


@pytest.mark.parametrize("field,bad_value,err_substr", [
    ("prompt_mode", "bogus", "prompt_mode must be one of"),
    ("exec_mode", "bogus", "exec_mode must be one of"),
    ("prompt_style", "bogus", "prompt_style must be one of"),
    ("num_runs", 0, "num_runs must be >= 1"),
    ("num_runs", -1, "num_runs must be >= 1"),
])
def test_runconfig_validation(field, bad_value, err_substr):
    kwargs = dict(
        prompt_mode="combined",
        exec_mode="violations-after",
        with_context=True,
    )
    kwargs[field] = bad_value
    with pytest.raises(ValueError) as exc:
        RunConfig(**kwargs)
    assert err_substr in str(exc.value)


def test_runconfig_replace_yields_paper_ablation():
    """Ablation expression style sanity check."""
    paper = dataclasses.replace(PAPER_DEFAULT, prompt_style="paper")
    assert paper.prompt_style == "paper"
    assert paper.prompt_mode == PAPER_DEFAULT.prompt_mode
    assert paper.include_nl_check_violations is True
