#!/usr/bin/env python3
"""Score the flash-42 sweep against flash ground truth.

Reads ``runs/azure_flash42_<tid>/judge_output/runs/run{1..N}.json``. Flash
judge output stores ``gt_failure_case`` / ``gt_step_number(s)`` already, so
this driver reads them directly and computes
cat / step / step±1 / step_any / step±1_any accuracy.

Usage (from repo root):
    python -m scripts.score_flash --num-runs 3
    python -m scripts.score_flash --num-runs 3 --json-out runs/azure_flash42_scored.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def score_one(run_path: Path) -> dict | None:
    try:
        rj = json.loads(run_path.read_text())
    except Exception:
        return None
    det = rj.get("detailed_results") or []
    if not det or not det[0].get("failures"):
        return None
    d = det[0]
    fail = d["failures"][0]
    pc = fail.get("failure_case")
    ps = fail.get("step_number")
    try:
        gt_cat = int(d["gt_failure_case"])
    except Exception:
        gt_cat = None
    gt_step = d.get("gt_step_number")
    gt_steps = d.get("gt_step_numbers") or ([gt_step] if gt_step is not None else [])
    if pc is None or ps is None or gt_cat is None or gt_step is None:
        return None
    cat_ok = int(pc) == int(gt_cat)
    step_ok = int(ps) == int(gt_step)
    step_pm1 = abs(int(ps) - int(gt_step)) <= 1
    step_any = int(ps) in [int(s) for s in gt_steps]
    step_pm1_any = min(abs(int(ps) - int(s)) for s in gt_steps) <= 1 if gt_steps else step_pm1
    return {
        "tid": d.get("task_id"),
        "pred_cat": pc, "pred_step": ps,
        "gt_cat": gt_cat, "gt_step": gt_step, "gt_steps": list(gt_steps),
        "cat_ok": cat_ok, "step_ok": step_ok,
        "step_pm1": step_pm1, "step_any": step_any, "step_pm1_any": step_pm1_any,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default=str(REPO_ROOT / "runs" / "azure_flash42_*"),
                    help="Glob pattern for flash run directories (default: runs/azure_flash42_*)")
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(args.pattern))
    print(f"Found {len(run_dirs)} run dirs matching {args.pattern}\n")

    per_run_stats: dict[int, dict] = {
        k: {"cat": 0, "step": 0, "step1": 0, "step_any": 0, "step1_any": 0, "total": 0, "missing": []}
        for k in range(1, args.num_runs + 1)
    }
    per_task = []

    for rd in run_dirs:
        for k in range(1, args.num_runs + 1):
            rp = Path(rd) / "judge_output" / "runs" / f"run{k}.json"
            if not rp.exists():
                per_run_stats[k]["missing"].append(Path(rd).name)
                continue
            r = score_one(rp)
            if r is None:
                per_run_stats[k]["missing"].append(Path(rd).name)
                continue
            per_run_stats[k]["total"] += 1
            if r["cat_ok"]:        per_run_stats[k]["cat"]       += 1
            if r["step_ok"]:       per_run_stats[k]["step"]      += 1
            if r["step_pm1"]:      per_run_stats[k]["step1"]     += 1
            if r["step_any"]:      per_run_stats[k]["step_any"]  += 1
            if r["step_pm1_any"]:  per_run_stats[k]["step1_any"] += 1
            per_task.append({"run": k, **r})

    print(f"{'METRIC':<14} " + " ".join(f"{'run'+str(k):>8}" for k in per_run_stats) + f"  {'mean':>8}  {'std':>7}")
    print("-" * 90)
    for metric, key in [
        ("cat_acc",       "cat"),
        ("step_acc",      "step"),
        ("step±1_acc",    "step1"),
        ("step_any_acc",  "step_any"),
        ("step±1_any",    "step1_any"),
    ]:
        accs = []
        cells = []
        for k in per_run_stats:
            tot = per_run_stats[k]["total"]
            if tot == 0:
                cells.append("    n/a")
                continue
            acc = per_run_stats[k][key] / tot
            accs.append(acc)
            cells.append(f"{acc:>8.4f}")
        mean = statistics.mean(accs) if accs else 0
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{metric:<14} " + " ".join(cells) + f"  {mean:>8.4f}  {std:>7.4f}")

    n_total = max(per_run_stats[k]["total"] for k in per_run_stats)
    print(f"\nTotal scored: {n_total} trajectories")
    for k in per_run_stats:
        miss = per_run_stats[k]["missing"]
        if miss:
            head = miss[:3]
            tail = "..." if len(miss) > 3 else ""
            print(f"  run{k} missing/empty: {len(miss)}: {head}{tail}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "per_run": per_run_stats,
            "per_task": per_task,
        }, indent=2, default=str))
        print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
