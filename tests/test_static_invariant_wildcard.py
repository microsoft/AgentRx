"""Static (global) invariants must trigger at every step (commit 4d6f0d6).

Background
----------
``StaticInvariantGenerator`` asks an LLM to emit, per invariant, an
``event_trigger.step_index``. Static/global invariants describe policies that
must hold across the *whole* trajectory, so their trigger must be the wildcard
string ``"*"``. The released prompt only documented the wildcard inside a
decorative JSON comment (``"step_index": "*|int|range"  // use "*" ...``), which
the LLM treated as filler and ignored, emitting the concrete integer ``0``
instead.

That is fatal because trajectory steps are **1-indexed** at match time: the
runtime matcher (``AllVerifier._step_matches_trigger``) compares the trigger
against ``step.index`` and an integer ``0`` therefore matches no real step. The
net effect on the tau-bench split was 271/271 generated static invariants
silently inert.

The fix is prompt-only: replace the decorative declaration with the literal
``"step_index": "*"`` plus a COMPLETE INVARIANT EXAMPLE block that inlines the
wildcard, because LLMs reliably copy worked JSON examples.

This module fences both halves of the contract:

* ``TestStaticPromptEmitsWildcard`` is the regression guard for the fix — it
  fails if either prompt template reverts to the decorative form or loses the
  worked example.
* ``TestMatcherDropsIntegerStepIndex`` documents *why* the wildcard is required
  by pinning the runtime matcher contract: integer ``0`` never matches a
  1-indexed trajectory, while ``"*"`` always does.
"""
from __future__ import annotations

import pytest

from agentrx.invariants.static_invariant_generator import (
    STATIC_INVARIANT_PROMPT,
    STATIC_INVARIANT_PROMPT_PYTHON_ONLY,
)
from agentrx.invariants.checker import AllVerifier


PROMPTS = {
    "STATIC_INVARIANT_PROMPT_PYTHON_ONLY": STATIC_INVARIANT_PROMPT_PYTHON_ONLY,
    "STATIC_INVARIANT_PROMPT": STATIC_INVARIANT_PROMPT,
}


class TestStaticPromptEmitsWildcard:
    """The prompt must steer the LLM to the literal ``"step_index": "*"``."""

    @pytest.mark.parametrize("name", list(PROMPTS))
    def test_prompt_specifies_literal_wildcard(self, name: str) -> None:
        prompt = PROMPTS[name]
        assert '"step_index": "*"' in prompt, (
            f"{name} no longer specifies the literal wildcard step_index"
        )

    @pytest.mark.parametrize("name", list(PROMPTS))
    def test_prompt_has_worked_example(self, name: str) -> None:
        # LLMs copy worked JSON examples but ignore decorative comments; the
        # fix relies on a COMPLETE INVARIANT EXAMPLE block to carry the wildcard.
        prompt = PROMPTS[name]
        assert "COMPLETE INVARIANT EXAMPLE" in prompt, (
            f"{name} lost its worked invariant example block"
        )

    @pytest.mark.parametrize("name", list(PROMPTS))
    def test_prompt_drops_decorative_declaration(self, name: str) -> None:
        # The two pre-fix declarations that the LLM ignored.
        prompt = PROMPTS[name]
        assert '"*|int|range"' not in prompt, (
            f"{name} reintroduced the decorative '*|int|range' declaration"
        )
        assert '"step_index": "int"' not in prompt, (
            f"{name} reintroduced the bare '\"step_index\": \"int\"' declaration"
        )

    @pytest.mark.parametrize("name", list(PROMPTS))
    def test_prompt_warns_against_integer_stand_in(self, name: str) -> None:
        # Explicit guidance that 0/1/2 must not be used as "every step".
        prompt = PROMPTS[name]
        assert "Never use 0, 1, 2" in prompt, (
            f"{name} dropped the explicit 'never use 0,1,2' warning"
        )


class TestMatcherDropsIntegerStepIndex:
    """Runtime contract the prompt fix depends on (matcher is 1-indexed).

    ``_step_matches_trigger`` uses no instance state, so we exercise it on a
    bare instance built with ``__new__`` to avoid the file/LLM-client setup in
    ``AllVerifier.__init__``.
    """

    @pytest.fixture
    def verifier(self) -> AllVerifier:
        return AllVerifier.__new__(AllVerifier)

    def test_wildcard_matches_every_step(self, verifier: AllVerifier) -> None:
        for pos, idx in enumerate([1, 2, 3]):
            step = {"index": idx, "substeps": []}
            assert verifier._step_matches_trigger("*", pos, step) is True

    def test_integer_zero_never_matches_one_indexed_trajectory(
        self, verifier: AllVerifier
    ) -> None:
        # This is the exact defect: an LLM emitting 0 for a "global" invariant
        # produces a trigger that fires on no real (1-indexed) step.
        for pos, idx in enumerate([1, 2, 3]):
            step = {"index": idx, "substeps": []}
            assert verifier._step_matches_trigger(0, pos, step) is False

    def test_specific_integer_matches_only_its_own_step(
        self, verifier: AllVerifier
    ) -> None:
        step1 = {"index": 1, "substeps": []}
        step2 = {"index": 2, "substeps": []}
        assert verifier._step_matches_trigger(1, 0, step1) is True
        assert verifier._step_matches_trigger(1, 1, step2) is False
        assert verifier._step_matches_trigger(2, 1, step2) is True
