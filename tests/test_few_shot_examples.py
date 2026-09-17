"""Few-shot examples ship in-tree and fail fast when missing.

The judge's PROMPT_MODE='combined' (paper-faithful default) requires per-
category examples. Earlier versions silently substituted "No example
available." when a file was missing, which changed what the judge saw on a
per-run basis and made Tab-1/Tab-2 numbers irreproducible. The current
loader raises FileNotFoundError listing every missing/malformed file.

This test fences both invariants: every example file exists and parses,
and the loader is fail-fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

EXPECTED_EXAMPLE_FILES = {
    1: "instruction_adherence_failure.json",
    2: "invention_of_new_information.json",
    3: "invalid_invocation.json",
    4: "misinterpretation_of_tool_output.json",
    5: "intent_plan_misalignment.json",
    6: "underspecified_user_intent.json",
    7: "intent_not_supported.json",
    8: "guardrails_triggered.json",
    9: "system_failure.json",
}


@pytest.fixture
def judge():
    import agentrx.judge.judge as j
    return j


@pytest.fixture
def examples_dir(judge):
    return Path(judge.__file__).parent / "few_shot_examples"


@pytest.mark.parametrize("category,filename", sorted(EXPECTED_EXAMPLE_FILES.items()))
def test_example_file_exists_and_parses(examples_dir, category, filename):
    path = examples_dir / filename
    assert path.exists(), f"missing few-shot example for category {category}: {path}"
    payload = json.loads(path.read_text())
    assert payload, f"empty few-shot payload for category {category}: {path}"


def test_loader_returns_all_nine_categories(judge):
    judge.FEW_SHOT_EXAMPLES = None  # force reload
    examples = judge.ensure_few_shot_examples_loaded()
    assert set(examples.keys()) == set(EXPECTED_EXAMPLE_FILES.keys())


def test_loader_fails_fast_when_directory_missing(judge, tmp_path):
    judge.FEW_SHOT_EXAMPLES = None
    judge.EXAMPLES_DIR = str(tmp_path / "does_not_exist")
    try:
        with pytest.raises(FileNotFoundError, match="Few-shot examples directory"):
            judge.ensure_few_shot_examples_loaded()
    finally:
        judge.FEW_SHOT_EXAMPLES = None
        judge.EXAMPLES_DIR = None


def test_loader_fails_fast_when_file_missing(judge, tmp_path):
    """Drop ONE example file and verify the loader names it in the error."""
    # Copy all 9 files into tmp_path, then remove one.
    import shutil
    real_dir = Path(judge.__file__).parent / "few_shot_examples"
    for fn in EXPECTED_EXAMPLE_FILES.values():
        shutil.copy(real_dir / fn, tmp_path / fn)
    dropped = "invalid_invocation.json"
    (tmp_path / dropped).unlink()

    judge.FEW_SHOT_EXAMPLES = None
    judge.EXAMPLES_DIR = str(tmp_path)
    try:
        with pytest.raises(FileNotFoundError) as exc:
            judge.ensure_few_shot_examples_loaded()
        assert dropped in str(exc.value)
    finally:
        judge.FEW_SHOT_EXAMPLES = None
        judge.EXAMPLES_DIR = None
