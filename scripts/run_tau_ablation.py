#!/usr/bin/env python3
"""Tau-29 judge-only ablation matrix (prompt-style × NL-violation inclusion).

Reuses existing IR/static/dynamic/check artifacts in
``runs/azure_tau29_<tid>/`` and re-runs ONLY the judge stage with different
``--prompt-style`` and ``--include-nl-violations`` / ``--exclude-nl-violations``
combinations. Outputs land in ``runs/ablation/<cell>/<tid>/judge_output/`` so
the original judge_output is preserved.

Cells (4 total, n=3 each => 348 LLM calls):
    paper_nl_on      --prompt-style paper   --include-nl-violations
    paper_nl_off     --prompt-style paper   --exclude-nl-violations
    release_nl_on    --prompt-style release --include-nl-violations
    release_nl_off   --prompt-style release --exclude-nl-violations

Cells run in parallel, each pinned to its own Azure endpoint so they do NOT
contend with an in-flight flash sweep (which holds aiops-llm-eus2). Within a
cell, tasks run sequentially (single thread) so each cell is one endpoint
-> one in-flight call at a time. Total wall time ~ 30 min - 1 hr.

Usage (from repo root):
    python -m scripts.run_tau_ablation --num-runs 3
    python -m scripts.run_tau_ablation --num-runs 3 --cells paper_nl_on release_nl_off
    python -m scripts.run_tau_ablation --task-ids 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

SRC_PATTERN = "runs/azure_tau29_{tid}"
OUT_ROOT = REPO_ROOT / "runs" / "ablation"
GT_PATH = REPO_ROOT / "data" / "ground_truth" / "tau_ground_truth.json"

# Endpoints (excluding aiops-llm-eus2 which the live flash sweep holds).
ENDPOINTS = {
    "eus": "https://aiops-llm-eus.openai.azure.com/",
    "ch":  "https://aiops-llm-ch.openai.azure.com/",
    "sw":  "https://aipos-llm-sw.openai.azure.com/",
}

# Cell catalogue: (name, extra CLI flags for run.py, endpoint_key).
# These flags map onto RunConfig.prompt_style and
# RunConfig.include_nl_check_violations via run.py's argparser.
CELLS: list[tuple[str, list[str], str]] = [
    ("paper_nl_on",    ["--prompt-style", "paper",   "--include-nl-violations"], "eus"),
    ("paper_nl_off",   ["--prompt-style", "paper",   "--exclude-nl-violations"], "ch"),
    ("release_nl_on",  ["--prompt-style", "release", "--include-nl-violations"], "sw"),
    ("release_nl_off", ["--prompt-style", "release", "--exclude-nl-violations"], "eus"),
]

# Symlinked artifacts (read-only inputs to the judge stage).
SYMLINK_ARTIFACTS = [
    "trajectory_ir.json",
    "static_invariants.json",
    "dynamic_invariants",
    "checker_results",
]


def load_tau_task_ids() -> list[str]:
    gt = json.loads(GT_PATH.read_text())
    return [str(e["trajectory_id"]) for e in gt]


def setup_task_dir(src_dir: Path, dst_dir: Path) -> bool:
    """Prepare ``dst_dir`` with symlinks to read-only artifacts + fresh state.

    Returns True iff all required source artifacts exist.
    """
    if not src_dir.exists():
        return False
    for name in SYMLINK_ARTIFACTS:
        src_path = (src_dir / name).resolve()
        if not src_path.exists():
            return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in SYMLINK_ARTIFACTS:
        link = dst_dir / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to((src_dir / name).resolve())
    state_src = src_dir / "state.json"
    state_dst = dst_dir / "state.json"
    if state_src.exists() and not state_dst.exists():
        shutil.copy2(state_src, state_dst)
    jo = dst_dir / "judge_output"
    if jo.exists():
        shutil.rmtree(jo)
    return True


def rejudge_one(
    tid: str,
    cell_name: str,
    extra_flags: list[str],
    endpoint_url: str,
    num_runs: int,
    log_path: Path,
) -> tuple[str, int, float]:
    """Run ``--stage judge`` on the per-cell dir for one task."""
    src_dir = REPO_ROOT / SRC_PATTERN.format(tid=tid)
    dst_dir = OUT_ROOT / cell_name / str(tid)
    if not setup_task_dir(src_dir, dst_dir):
        return tid, 99, 0.0

    env = os.environ.copy()
    env["AGENT_VERIFY_ENDPOINT"] = endpoint_url

    input_file = REPO_ROOT / "data" / "tau_dataset" / f"{tid}.json"
    if not input_file.exists():
        # Fall back to trajectory_ir.json (the judge's actual input anyway).
        input_file = dst_dir / "trajectory_ir.json"

    cmd = [
        PYTHON, "run.py", str(input_file),
        "--domain", "tau",
        "--endpoint", "azure",
        "--ground-truth", str(GT_PATH),
        "--stage", "judge",
        "--run-dir", str(dst_dir),
        "--num-runs", str(num_runs),
        *extra_flags,
    ]
    t0 = time.perf_counter()
    with log_path.open("a") as lf:
        lf.write(f"\n=== [{datetime.now().isoformat()}] cell={cell_name} tid={tid} "
                 f"endpoint={endpoint_url} ===\n")
        lf.write(f"  cmd: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT),
                              stdout=lf, stderr=subprocess.STDOUT)
    return tid, proc.returncode, round(time.perf_counter() - t0, 1)


def run_cell(
    cell_name: str,
    extra_flags: list[str],
    endpoint_url: str,
    task_ids: list[str],
    num_runs: int,
) -> dict:
    """Run all tasks for one cell sequentially. Returns summary dict."""
    log_dir = OUT_ROOT / cell_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "_cell.log"
    log_path.write_text(
        f"# Cell {cell_name} started {datetime.now().isoformat()}\n"
        f"# extra_flags={extra_flags}\n"
        f"# endpoint={endpoint_url}\n"
        f"# num_runs={num_runs}\n"
        f"# n_tasks={len(task_ids)}\n\n"
    )
    print(f"[{cell_name}] START n_tasks={len(task_ids)} endpoint={endpoint_url}", flush=True)
    results = []
    t_cell = time.perf_counter()
    for i, tid in enumerate(task_ids, 1):
        tid_str, rc, elapsed = rejudge_one(tid, cell_name, extra_flags, endpoint_url, num_runs, log_path)
        results.append({"task_id": tid_str, "rc": rc, "elapsed_sec": elapsed})
        cum = round(time.perf_counter() - t_cell, 1)
        print(f"[{cell_name}] {i}/{len(task_ids)} tid={tid_str} rc={rc} {elapsed}s (cum {cum}s)",
              flush=True)
        (log_dir / "_summary.json").write_text(json.dumps({
            "cell": cell_name,
            "extra_flags": extra_flags,
            "endpoint": endpoint_url,
            "num_runs": num_runs,
            "n_tasks": len(task_ids),
            "completed": i,
            "results": results,
        }, indent=2))
    return {
        "cell": cell_name,
        "extra_flags": extra_flags,
        "endpoint": endpoint_url,
        "num_runs": num_runs,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--cells", nargs="+", default=[c[0] for c in CELLS],
                    help="Which cells to run (default: all 4).")
    ap.add_argument("--task-ids", nargs="+", default=None,
                    help="Restrict to these task IDs (default: all 29).")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    task_ids = args.task_ids or load_tau_task_ids()
    cells_to_run = [c for c in CELLS if c[0] in args.cells]
    if not cells_to_run:
        valid = [c[0] for c in CELLS]
        print(f"No cells matched {args.cells}; valid: {valid}", file=sys.stderr)
        sys.exit(2)

    print(f"=== Ablation matrix: {len(cells_to_run)} cells × {len(task_ids)} tasks × n={args.num_runs} ===")
    for n, e, ep in cells_to_run:
        print(f"  cell={n:<16} flags={e} endpoint={ENDPOINTS[ep]}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cells_to_run)) as ex:
        futures = {
            ex.submit(run_cell, name, flags, ENDPOINTS[ep], task_ids, args.num_runs): name
            for (name, flags, ep) in cells_to_run
        }
        all_results = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                all_results[name] = fut.result()
            except Exception as e:
                print(f"[{name}] FAILED: {e}", file=sys.stderr)
                all_results[name] = {"cell": name, "error": str(e)}

    total = round(time.perf_counter() - t0, 1)
    print(f"\n=== ALL CELLS DONE in {total}s ===")
    out_path = OUT_ROOT / "_matrix_summary.json"
    out_path.write_text(json.dumps({
        "started_at": datetime.now().isoformat(),
        "total_sec": total,
        "task_ids": task_ids,
        "cells": all_results,
    }, indent=2))
    print(f"  Summary: {out_path}")


if __name__ == "__main__":
    main()
