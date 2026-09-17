"""
Step-accuracy primitive.

A judge prediction is *exact* when its rounded failure-step index lands on
ANY ground-truth failure step for the trajectory, and *within ±r* when its
distance to the NEAREST ground-truth failure step is ≤ r. Scoring against a
single positional GT failure (for instance the root cause, or ``failures[0]``)
penalises correct hits on a non-first failure in multi-failure trajectories
and is not the metric reported in the paper.

Every consumer that turns a predicted step plus a ground-truth set of failure
steps into a distance must call ``step_distance_to_nearest_gt`` so the metric
definition lives in exactly one place.
"""

from __future__ import annotations

from typing import Iterable


def step_distance_to_nearest_gt(
    predicted_step: float, gt_step_numbers: Iterable[int]
) -> int:
    """Return |rounded(predicted_step) − g*| where g* is the nearest GT step.

    ``gt_step_numbers`` must be non-empty; the function deliberately raises
    rather than silently defaulting to 0 so a missing-GT bug surfaces at the
    call site instead of inflating step-accuracy.
    """
    steps = [int(s) for s in gt_step_numbers]
    if not steps:
        raise ValueError("gt_step_numbers must be non-empty")
    pred = int(round(float(predicted_step)))
    return min(abs(pred - g) for g in steps)
