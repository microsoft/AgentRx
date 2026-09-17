#!/usr/bin/env python3
"""Score the tau-29 judge-prompt-style ablation against tau ground truth.

Reads ``runs/ablation/<cell>/<tid>/judge_output/runs/run{1..N}.json`` and the
``runs/azure_tau29_<tid>/`` baseline, producing a comparison table of
(cat_acc, step_acc, step±1_acc, step_any, step±1_any) per cell along with
per-run breakdowns and knob-effect deltas.

Cells (see scripts.run_tau_ablation):
    paper_nl_on / paper_nl_off / release_nl_on / release_nl_off

Usage (from repo root):
    python -m scripts.score_tau_ablation --num-runs 3
    python -m scripts.score_tau_ablation --num-runs 3 --json-out runs/ablation/_scores.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_PATH = REPO_ROOT / "data" / "ground_truth" / "tau_ground_truth.json"

# Human-readable failure category name -> FailureCase enum int (per judge.py).
NAME_TO_INT = {
    "Underspecified User Intent": 6,
    "Intent Not Supported": 7,
    "Intent Plan Misalignment": 5,
    "Intent / Plan Misalignment": 5,
    "Instruction Adherence Failure": 1,
    "Instruction / Plan Adherence Failure": 1,
    "Invention of New Information": 2,
    "Invalid Invocation": 3,
    "Invocation Failure": 3,
    "Misinterpretation of Tool Output": 4,
    "Guardrails Triggered": 8,
    "System Failure": 9,
    "Inconclusive": 10,
    "Other / Out of Scope": 10,
}

ABLATION_CELLS = ("paper_nl_on", "paper_nl_off", "release_nl_on", "release_nl_off")


def load_ground_truth() -> dict[str, tuple[int | None, int | None, tuple[int, ...]]]:
    gt = json.loads(GT_PATH.read_text())
    by_task: dict[str, tuple[int | None, int | None, tuple[int, ...]]] = {}
    for e in gt:
        tid = str(e["trajectory_id"])
        rc_fid = e["root_cause"]["failure_id"]
        rc_step, rc_cat_name = None, None
        for f in e["failures"]:
            if f["failure_id"] == rc_fid:
                rc_step = f["step_number"]
                rc_cat_name = f["failure_category"]
                break
        all_steps = tuple(sorted(int(f["step_number"]) for f in e["failures"]))
        by_task[tid] = (NAME_TO_INT.get(rc_cat_name), rc_step, all_steps)
    return by_task


def score_run_file(run_path: Path, gt_by_task: dict) -> dict | None:
    """Extract prediction from one ``runK.json``."""
    try:
        rj = json.loads(run_path.read_text())
    except Exception:
        return None
    det = rj.get("detailed_results") or []
    if not det:
        return None
    d = det[0]
    if not d.get("failures"):
        return None
    fail = d["failures"][0]
    tid = str(d.get("task_id"))
    if tid not in gt_by_task:
        return None
    gt_cat, gt_step, gt_steps = gt_by_task[tid]
    pc = fail.get("failure_case")
    ps = fail.get("step_number")
    cat_correct = pc is not None and gt_cat is not None and int(pc) == int(gt_cat)
    step_correct = ps is not None and gt_step is not None and int(ps) == int(gt_step)
    step_pm1_correct = (
        ps is not None and gt_step is not None and abs(int(ps) - int(gt_step)) <= 1
    )
    step_any_correct = False
    step_pm1_any_correct = False
    if ps is not None and gt_steps:
        diffs = [abs(int(ps) - int(s)) for s in gt_steps]
        step_any_correct = min(diffs) == 0
        step_pm1_any_correct = min(diffs) <= 1
    return {
        "tid": tid,
        "pred_cat": pc,
        "pred_step": ps,
        "gt_cat": gt_cat,
        "gt_step": gt_step,
        "gt_steps": list(gt_steps),
        "cat_correct": cat_correct,
        "step_correct": step_correct,
        "step_pm1_correct": step_pm1_correct,
        "step_any_correct": step_any_correct,
        "step_pm1_any_correct": step_pm1_any_correct,
    }


def _aggregate_per_run(per_run: dict[int, dict]) -> dict:
    n = sum(1 for v in per_run.values() if v["total"] > 0)
    if n == 0:
        return {"per_run": {}, "mean": {}, "n_tasks": 0}
    means = {
        "cat":          sum(per_run[k]["cat"]      / per_run[k]["total"] for k in per_run) / n,
        "step":         sum(per_run[k]["step"]     / per_run[k]["total"] for k in per_run) / n,
        "step_pm1":     sum(per_run[k]["step1"]    / per_run[k]["total"] for k in per_run) / n,
        "step_any":     sum(per_run[k]["step_any"] / per_run[k]["total"] for k in per_run) / n,
        "step_pm1_any": sum(per_run[k]["step1_any"] / per_run[k]["total"] for k in per_run) / n,
    }
    cat_per_run  = [per_run[k]["cat"]  / per_run[k]["total"] for k in per_run if per_run[k]["total"] > 0]
    step_per_run = [per_run[k]["step"] / per_run[k]["total"] for k in per_run if per_run[k]["total"] > 0]
    stds = {
        "cat":  statistics.stdev(cat_per_run)  if len(cat_per_run)  > 1 else 0.0,
        "step": statistics.stdev(step_per_run) if len(step_per_run) > 1 else 0.0,
    }
    return {
        "per_run": dict(per_run),
        "mean": means,
        "std": stds,
        "n_tasks": max(per_run[k]["total"] for k in per_run),
    }


def _empty_run_counters() -> dict[int, dict]:
    return defaultdict(lambda: {"cat": 0, "step": 0, "step1": 0, "step_any": 0, "step1_any": 0, "total": 0})


def score_cell(cell_dir: Path, gt_by_task: dict, num_runs: int) -> dict:
    per_run = _empty_run_counters()
    for tid in sorted(gt_by_task.keys(), key=lambda x: int(x) if x.isdigit() else x):
        task_dir = cell_dir / tid
        if not task_dir.exists():
            continue
        for k in range(1, num_runs + 1):
            rp = task_dir / "judge_output" / "runs" / f"run{k}.json"
            if not rp.exists():
                continue
            r = score_run_file(rp, gt_by_task)
            if r is None:
                continue
            per_run[k]["total"] += 1
            if r["cat_correct"]: per_run[k]["cat"] += 1
            if r["step_correct"]: per_run[k]["step"] += 1
            if r["step_pm1_correct"]: per_run[k]["step1"] += 1
            if r["step_any_correct"]: per_run[k]["step_any"] += 1
            if r["step_pm1_any_correct"]: per_run[k]["step1_any"] += 1
    return _aggregate_per_run(per_run)


def score_baseline_tau29(gt_by_task: dict, num_runs: int) -> dict:
    """Score the original ``runs/azure_tau29_<tid>/`` sweep as the current default."""
    per_run = _empty_run_counters()
    for tid in sorted(gt_by_task.keys(), key=lambda x: int(x) if x.isdigit() else x):
        for k in range(1, num_runs + 1):
            rp = REPO_ROOT / "runs" / f"azure_tau29_{tid}" / "judge_output" / "runs" / f"run{k}.json"
            if not rp.exists():
                continue
            r = score_run_file(rp, gt_by_task)
            if r is None:
                continue
            per_run[k]["total"] += 1
            if r["cat_correct"]: per_run[k]["cat"] += 1
            if r["step_correct"]: per_run[k]["step"] += 1
            if r["step_pm1_correct"]: per_run[k]["step1"] += 1
            if r["step_any_correct"]: per_run[k]["step_any"] += 1
            if r["step_pm1_any_correct"]: per_run[k]["step1_any"] += 1
    return _aggregate_per_run(per_run)


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--ablation-root", default=str(REPO_ROOT / "runs" / "ablation"))
    ap.add_argument("--cells", nargs="+", default=list(ABLATION_CELLS))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gt = load_ground_truth()
    print(f"Ground truth tasks: {len(gt)}\n")

    rows: list[tuple[str, dict]] = []
    rows.append(("[baseline] azure_tau29 (release default, NL on)", score_baseline_tau29(gt, args.num_runs)))
    for cell in args.cells:
        rows.append((f"[ablation] {cell}", score_cell(Path(args.ablation_root) / cell, gt, args.num_runs)))

    print(f"{'CELL':<50} {'n_tasks':>7} {'cat_mean':>9} {'cat_std':>8}  {'step_mean':>9} {'step_std':>8}  "
          f"{'step±1':>7}  {'step_any':>9}  {'step±1_any':>11}")
    print("-" * 130)
    for label, s in rows:
        if not s.get("mean"):
            print(f"{label:<50} {'<no data>':>7}")
            continue
        m = s["mean"]
        st = s.get("std", {})
        print(f"{label:<50} {s['n_tasks']:>7} "
              f"{_fmt(m['cat']):>9} {_fmt(st.get('cat', 0)):>8}  "
              f"{_fmt(m['step']):>9} {_fmt(st.get('step', 0)):>8}  "
              f"{_fmt(m['step_pm1']):>7}  {_fmt(m['step_any']):>9}  {_fmt(m['step_pm1_any']):>11}")

    print("\n--- Per-run breakdown ---")
    for label, s in rows:
        if not s.get("per_run"):
            continue
        print(f"\n{label}")
        for k in sorted(s["per_run"].keys()):
            v = s["per_run"][k]
            tot = v["total"]
            if tot == 0:
                continue
            print(f"  run{k}: cat={v['cat']:>2}/{tot} ({v['cat']/tot:.3f})  "
                  f"step={v['step']:>2}/{tot} ({v['step']/tot:.3f})  "
                  f"step±1={v['step1']:>2}/{tot} ({v['step1']/tot:.3f})  "
                  f"step_any={v['step_any']:>2}/{tot} ({v['step_any']/tot:.3f})")

    print("\n--- Knob effect deltas (vs paper_nl_on as reference) ---")
    ref = next((s for label, s in rows if "paper_nl_on" in label), None)
    if ref and ref.get("mean"):
        for label, s in rows:
            if not s.get("mean") or s is ref:
                continue
            for metric in ("cat", "step", "step_pm1", "step_any", "step_pm1_any"):
                d = s["mean"].get(metric, 0) - ref["mean"].get(metric, 0)
                sign = "+" if d >= 0 else ""
                print(f"  {label:<50} Δ{metric:<10} = {sign}{d:.4f}")
            print()

    if args.json_out:
        out = {label: s for label, s in rows}
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
