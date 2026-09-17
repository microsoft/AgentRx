"""Centralized judge-stage configuration.

The judge stage exposes a small set of orthogonal axes that the AgentRx paper
varies per table cell. They are bundled into a single frozen dataclass so that
every call site (run.py, downstream sweep scripts, tests) refers to one source
of truth instead of repeating string literals or reading process-global env
vars.

There is no profile *registry* by design. The dataclass IS the centralization
primitive: paper-default recipe lives in PAPER_DEFAULT, and ablations are
expressed as ``dataclasses.replace(PAPER_DEFAULT, prompt_style="paper")``.
Orthogonal pipeline-stage knobs (dynamic-mode, skip-static, ...) deliberately
remain on their own CLI flags rather than being bundled into a named recipe.
"""
from dataclasses import dataclass
from typing import Literal, get_args

PromptMode = Literal["baseline", "checklist", "examples", "combined"]
ExecMode = Literal["violations-after", "stepbystep", "violations-before"]
PromptStyle = Literal["paper", "release"]


@dataclass(frozen=True)
class RunConfig:
    """Judge-stage knobs needed to reproduce a paper table cell.

    prompt_mode
        Which taxonomy block the judge sees (see judge.py::build_taxonomy_text).
    exec_mode
        When violations are surfaced relative to category labelling.
    with_context
        Inject deduplicated violation context into the judge prompt. Runtime
        value also requires the upstream check stage to have produced a context
        directory; see run.py::run_judge.
    num_runs
        Number of independent judge iterations to perform. When > 1, the
        pipeline writes per-run outputs under ``runs/run{N}.json`` and emits a
        mean/std aggregate summary. The paper uses 3.
    prompt_style
        ``"paper"`` reproduces the paper-mirror concat prompt builder verbatim.
        ``"release"`` uses the originally-released f-string templates (section
        order GIVEN-INPUT / TAXONOMY / ALGORITHM / VIOLATIONS / OUTPUT), which
        dominate the paper-mirror style on the tau-29 ablation (cat 0.425 vs
        0.414; step 0.494 vs 0.379). Default is ``"release"`` — see
        ``PAPER_DEFAULT`` vs ``PAPER_MIRROR_DEFAULT`` below.
    include_nl_check_violations
        When ``False``, ``nl_check`` violations are filtered out before the
        judge sees the context (reproduces the paper's "Without NL Check Viol."
        appendix table). Default ``True``.
    """
    prompt_mode: PromptMode
    exec_mode: ExecMode
    with_context: bool
    num_runs: int = 1
    prompt_style: PromptStyle = "release"
    include_nl_check_violations: bool = True

    def __post_init__(self) -> None:
        if self.num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {self.num_runs}")
        if self.prompt_mode not in get_args(PromptMode):
            raise ValueError(
                f"prompt_mode must be one of {get_args(PromptMode)}, got {self.prompt_mode!r}"
            )
        if self.exec_mode not in get_args(ExecMode):
            raise ValueError(
                f"exec_mode must be one of {get_args(ExecMode)}, got {self.exec_mode!r}"
            )
        if self.prompt_style not in get_args(PromptStyle):
            raise ValueError(
                f"prompt_style must be one of {get_args(PromptStyle)}, got {self.prompt_style!r}"
            )


PAPER_DEFAULT = RunConfig(
    prompt_mode="combined",
    exec_mode="violations-after",
    with_context=True,
    num_runs=3,
    prompt_style="release",
    include_nl_check_violations=True,
)

PAPER_MIRROR_DEFAULT = RunConfig(
    prompt_mode="combined",
    exec_mode="violations-after",
    with_context=True,
    num_runs=3,
    prompt_style="paper",
    include_nl_check_violations=True,
)
