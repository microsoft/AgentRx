#!/usr/bin/env python3
"""
AgentVerify push-button runner.

Run the full pipeline on a trajectory file with a single command:

    python run.py trajectory.json                     # auto-detect domain, run everything
    python run.py trajectory.json --domain tau        # specify domain
    python run.py trajectory.json --stage ir          # run only IR normalization
    python run.py trajectory.json --stage check       # run only invariant checking (requires prior stages)
    python run.py trajectory.json --skip-dynamic      # skip dynamic invariant generation (faster)
    python run.py trajectory.json --skip-judge        # skip judge stage

Stages (executed in order):
  ir        → Normalize raw trajectory into canonical IR
  static    → Generate static invariants from policy
  dynamic   → Generate per-step dynamic invariants
  check     → Check invariants against trajectory
  judge     → Run LLM-as-a-Judge for root-cause classification
  report    → Generate failure frequency plots

All outputs go to: runs/<run_name>/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so emoji/unicode prints don't crash
# the pipeline with UnicodeEncodeError (cp1252 default)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
REPO_ROOT = Path(__file__).resolve().parent

import agentrx.pipeline.globals as g
from agentrx.pipeline.profiles import RunConfig, PAPER_DEFAULT

# ---------- Stage definitions ----------

STAGES = ["ir", "static", "dynamic", "check", "judge", "report"]

def stage_index(name: str) -> int:
    return STAGES.index(name)


# ---------- Helpers ----------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_degenerate_ir(data: list, input_path: str) -> bool:
    """Check if IR conversion produced a degenerate result (empty/single-step with no content)
    when the raw input file clearly has more data."""
    if not data:
        return True
    # Check if all trajectories are trivially empty
    total_steps = sum(len(t.get("steps") or []) for t in data)
    total_content = sum(
        len(sub.get("content") or "")
        for t in data
        for s in (t.get("steps") or [])
        for sub in (s.get("substeps") or [])
    )
    # If the output is tiny but the input file is substantial, the converter didn't understand it
    input_size = os.path.getsize(input_path)
    if total_steps <= 1 and total_content == 0 and input_size > 500:
        return True
    if total_content < 100 and input_size > 5000:
        return True
    return False


def load_state(run_dir: str) -> dict:
    state_path = os.path.join(run_dir, "state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"completed_stages": [], "config": {}}


def save_state(run_dir: str, state: dict):
    with open(os.path.join(run_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def banner(msg: str):
    width = max(len(msg) + 4, 60)
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}\n")


def validate_endpoint_config(endpoint: str):
    """Validate that required environment variables are set for the chosen endpoint."""
    missing = []
    if endpoint == "copilot":
        # Copilot CLI needs no env vars — just the binary on PATH
        pass
    elif endpoint == "trapi":
        if not g.TRAPI_INSTANCE:
            missing.append("AGENT_VERIFY_TRAPI_INSTANCE")
        if not g.TRAPI_DEPLOYMENT_NAME:
            missing.append("AGENT_VERIFY_TRAPI_DEPLOYMENT_NAME")
        if not os.environ.get("SCOPE", ""):
            missing.append("SCOPE")
    elif endpoint == "azure":
        if not g.ENDPOINT:
            missing.append("AGENT_VERIFY_ENDPOINT")
    else:
        print(f"\nError: Unknown endpoint '{endpoint}'. Must be 'copilot', 'azure', or 'trapi'.")
        sys.exit(1)

    if missing:
        print(f"\nError: Missing required environment variables for --endpoint {endpoint}:")
        for v in missing:
            print(f"  - {v}")
        print(f"\nSee .env.example for a template: cp .env.example .env")
        sys.exit(1)


# ---------- Stage: IR ----------

def run_ir(input_path: str, run_dir: str, domain: str, endpoint: str, state: dict) -> str:
    """Normalize trajectory to IR format. Returns path to IR output."""
    from agentrx.ir.trajectory_ir import load_trajectories, validate_ir, markdown_ir
    from agentrx.invariants.domain_registry import get_domain_config

    banner("Stage 1/6: IR Normalization")

    ir_out_path = os.path.join(run_dir, "trajectory_ir.json")

    raw = load_trajectories(input_path)

    # If the loader detected markdown format, use the dedicated markdown IR
    # converter instead of the domain converter (which doesn't understand it)
    is_markdown = any(t.get("_markdown_ir") for t in raw)
    if is_markdown:
        print("  [INFO] Markdown conversation detected — using markdown IR converter")
        data = markdown_ir(raw)
    else:
        cfg = get_domain_config(domain)
        ir_fn = cfg.ir_converter
        data = ir_fn(raw)

    if not isinstance(data, list):
        data = [data]

    # Detect degenerate IR (converter didn't understand the format) and fall back to llm_ir
    used_llm_fallback = False
    if _is_degenerate_ir(data, input_path):
        print("  [INFO] Domain converter produced degenerate IR — falling back to LLM-based converter")
        from agentrx.ir.trajectory_ir import llm_ir
        data = llm_ir(raw, endpoint=endpoint)
        if not isinstance(data, list):
            data = [data]
        used_llm_fallback = True

    state["ir_used_llm_fallback"] = used_llm_fallback
    state["ir_from_markdown"] = is_markdown
    if used_llm_fallback:
        print("  [INFO] Unknown format detected — domain-specific tools will NOT be used")

    # Validate each trajectory
    valid_count = 0
    for traj in data:
        try:
            validate_ir(traj)
            valid_count += 1
        except Exception as e:
            tid = traj.get("trajectory_id", "?")
            print(f"  [WARN] Trajectory {tid} failed IR validation: {e}")

    with open(ir_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Trajectories: {len(data)} loaded, {valid_count} valid")
    print(f"  Output: {ir_out_path}")
    return ir_out_path


# ---------- Stage: Static Invariants ----------

def run_static(input_path: str, run_dir: str, domain: str, endpoint: str,
               state: dict) -> str:
    """Generate static invariants. Returns path to output JSON."""
    from agentrx.invariants.static_invariant_generator import StaticInvariantGenerator
    from agentrx.invariants.domain_registry import get_domain_config

    banner("Stage 2/6: Static Invariant Generation")

    out_path = os.path.join(run_dir, "static_invariants.json")
    cfg = get_domain_config(domain)
    used_llm_fallback = state.get("ir_used_llm_fallback", False)

    # Use the already-converted IR as sample (don't re-run the domain converter)
    ir_path = os.path.join(run_dir, "trajectory_ir.json")
    with open(ir_path, "r", encoding="utf-8") as f:
        ir_data = json.load(f)
    sample_traj = ir_data[0] if isinstance(ir_data, list) else ir_data

    # If the domain converter failed (LLM fallback was used) or the input is
    # markdown, don't inject domain-specific tools — they're irrelevant
    if used_llm_fallback or state.get("ir_from_markdown", False):
        tools_list = []
        tools_structure = None
        policy_path = None
        print("  [INFO] Using empty tools (domain tools don't apply to this trajectory format)")
    else:
        tools_list = cfg.tools_list
        tools_structure = cfg.tools_structure
        policy_path = None
        if cfg.default_policy_path:
            candidate = os.path.join(str(REPO_ROOT), cfg.default_policy_path)
            if os.path.exists(candidate):
                policy_path = candidate

    gen = StaticInvariantGenerator(
        traj_for_enums=sample_traj,
        tools_list=tools_list,
        tools_structure=tools_structure,
        domain=domain,
        policy_document_path=policy_path or "",
        out_path=out_path,
        include_nl_check=True,
        endpoint=endpoint,
    )
    gen.run()

    print(f"  Output: {out_path}")
    return out_path


# ---------- Stage: Dynamic Invariants ----------

def run_dynamic(input_path: str, run_dir: str, domain: str, endpoint: str,
                static_invariants_path: str, state: dict,
                dynamic_mode: str = "stepbystep") -> str:
    """Generate dynamic invariants. Returns path to output directory."""
    from agentrx.invariants.dynamic_invariant_generator import DynamicInvariantGenerator, OneShotDynamicInvariantGenerator
    from agentrx.invariants.domain_registry import get_domain_config

    mode_label = "one-shot" if dynamic_mode == "oneshot" else "step-by-step"
    banner(f"Stage 3/6: Dynamic Invariant Generation ({mode_label})")

    out_dir = os.path.join(run_dir, "dynamic_invariants")
    ensure_dir(out_dir)

    cfg = get_domain_config(domain)
    used_llm_fallback = state.get("ir_used_llm_fallback", False)

    if used_llm_fallback or state.get("ir_from_markdown", False):
        tools_list = []
        tools_structure = None
        print("  [INFO] Using empty tools (domain tools don't apply to this trajectory format)")
    else:
        tools_list = cfg.tools_list
        tools_structure = cfg.tools_structure

    GeneratorClass = OneShotDynamicInvariantGenerator if dynamic_mode == "oneshot" else DynamicInvariantGenerator
    gen = GeneratorClass(
        out_dir=out_dir,
        static_invariants_path=static_invariants_path,
        domain=domain,
        tools_list=tools_list,
        tools_structure=tools_structure,
        include_nl_check=True,
        endpoint=endpoint,
    )

    if used_llm_fallback or state.get("ir_from_markdown", False):
        # Use the already-converted IR file instead of re-reading the raw file
        # (which would go through the wrong domain converter)
        ir_path = os.path.join(run_dir, "trajectory_ir.json")
        with open(ir_path, "r", encoding="utf-8") as f:
            ir_data = json.load(f)
        if not isinstance(ir_data, list):
            ir_data = [ir_data]
        gen.run_from_ir_data(ir_data, source_label=ir_path)
    else:
        gen.run_file(input_path)

    print(f"  Output: {out_dir}")
    return out_dir


# ---------- Stage: Check ----------

def run_check(ir_path: str, run_dir: str, domain: str, endpoint: str,
              static_invariants_path: str, dynamic_invariants_dir: str) -> str:
    """Check invariants against trajectory. Returns path to results directory."""
    from agentrx.invariants.checker import AllVerifier
    from agentrx.ir.trajectory_ir import load_trajectories
    from agentrx.invariants.domain_registry import get_domain_config

    banner("Stage 4/6: Invariant Checking")

    results_dir = os.path.join(run_dir, "checker_results")
    ensure_dir(results_dir)

    cfg = get_domain_config(domain)

    # Resolve policy path
    policy_path = ""
    if cfg.default_policy_path:
        candidate = os.path.join(str(REPO_ROOT), cfg.default_policy_path)
        if os.path.exists(candidate):
            policy_path = candidate

    # Load IR trajectories
    with open(ir_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    # Initialize static verifier
    static_verifier = AllVerifier(
        invariants_path=static_invariants_path,
        policy_document_path=policy_path,
        client=endpoint,
    )

    for i, traj in enumerate(trajectories):
        task_id = str(traj.get("trajectory_id") or traj.get("task_id") or i)
        out_dir = os.path.join(results_dir, task_id)
        ensure_dir(out_dir)

        all_violations = []
        all_telemetry = []

        # Static check
        static_violations = static_verifier.verify_trajectory(task_id=task_id, traj=traj)
        all_violations.extend([v.to_dict() for v in static_violations])
        all_telemetry.extend([t.to_dict() for t in static_verifier.telemetry])

        # Dynamic check (if dynamic invariants exist for this trajectory)
        if dynamic_invariants_dir:
            # Try both step-by-step (out_{id}.json) and one-shot (out_{id}_oneshot.json) names
            dyn_file = None
            for cand in (
                os.path.join(dynamic_invariants_dir, f"out_{task_id}.json"),
                os.path.join(dynamic_invariants_dir, f"out_{task_id}_oneshot.json"),
            ):
                if os.path.exists(cand):
                    dyn_file = cand
                    break
            if dyn_file:
                dyn_verifier = AllVerifier(
                    invariants_path=dyn_file,
                    policy_document_path=policy_path,
                    client=endpoint,
                )
                dyn_violations = dyn_verifier.verify_trajectory(task_id=task_id, traj=traj)
                all_violations.extend([v.to_dict() for v in dyn_violations])
                all_telemetry.extend([t.to_dict() for t in dyn_verifier.telemetry])

        all_violations.sort(key=lambda v: v.get("step_index", 0))

        with open(os.path.join(out_dir, f"violations_{domain}.json"), "w") as f:
            json.dump(all_violations, f, indent=2)
        with open(os.path.join(out_dir, f"telemetry_{domain}.json"), "w") as f:
            json.dump(all_telemetry, f, indent=2)

        print(f"  Task {task_id}: {len(all_violations)} violations found")

    print(f"  Output: {results_dir}")
    return results_dir


# ---------- Stage: Judge ----------

def run_judge(input_path: str, run_dir: str, domain: str, endpoint: str,
              violation_context_dir: str = None, ground_truth_file: str = None,
              config: RunConfig = None) -> str:
    """Run LLM-as-a-Judge. Returns path to judge output directory.

    config selects the paper-table-cell recipe (prompt_mode / exec_mode /
    with_context). Defaults to the paper-faithful recipe.
    """
    import agentrx.judge.judge as judge_module

    if config is None:
        config = PAPER_DEFAULT

    banner("Stage 5/6: LLM-as-a-Judge")

    judge_out_dir = os.path.join(run_dir, "judge_output")
    ensure_dir(judge_out_dir)

    # Set globals that judge.py expects
    judge_module.DOMAIN = domain
    judge_module.ENDPOINT_USED = endpoint
    judge_module.PROMPT_MODE = config.prompt_mode
    judge_module.EXECUTION_MODE = config.exec_mode
    judge_module.PROMPT_STYLE = config.prompt_style
    judge_module.INCLUDE_NL_VIO = config.include_nl_check_violations
    # with_context can only be honoured when the check stage produced a context
    # directory; if it didn't, the profile's request collapses to False.
    judge_module.RUN_WITH_CONTEXT = config.with_context and violation_context_dir is not None
    judge_module.USE_GROUND_TRUTH = ground_truth_file is not None

    if violation_context_dir:
        judge_module.VIOLATION_CONTEXT_DIR = violation_context_dir

    # --- DEBUG: Log all inputs to the judge stage ---
    print(f"\n  [DEBUG][run_judge] input_path:            {input_path}")
    print(f"  [DEBUG][run_judge] run_dir:               {run_dir}")
    print(f"  [DEBUG][run_judge] domain:                {domain}")
    print(f"  [DEBUG][run_judge] endpoint:              {endpoint}")
    print(f"  [DEBUG][run_judge] violation_context_dir: {violation_context_dir}")
    print(f"  [DEBUG][run_judge] ground_truth_file:     {ground_truth_file}")
    print(f"  [DEBUG][run_judge] PROMPT_MODE:           {judge_module.PROMPT_MODE}")
    print(f"  [DEBUG][run_judge] EXECUTION_MODE:        {judge_module.EXECUTION_MODE}")
    print(f"  [DEBUG][run_judge] PROMPT_STYLE:          {judge_module.PROMPT_STYLE}")
    print(f"  [DEBUG][run_judge] INCLUDE_NL_VIO:        {judge_module.INCLUDE_NL_VIO}")
    print(f"  [DEBUG][run_judge] RUN_WITH_CONTEXT:      {judge_module.RUN_WITH_CONTEXT}")
    print(f"  [DEBUG][run_judge] USE_GROUND_TRUTH:      {judge_module.USE_GROUND_TRUTH}")
    print(f"  [DEBUG][run_judge] NUM_RUNS:              {config.num_runs}")

    # Load ground truth if provided
    gt_failures = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        gt_failures = judge_module.load_failures_from_json(ground_truth_file)
        print(f"  [DEBUG][run_judge] Loaded {len(gt_failures)} ground truth failures")

    import agentrx.pipeline.globals as g
    api_version = g.API_VERSION
    model_name = g.DEPLOYMENT if endpoint == "azure" else g.TRAPI_DEPLOYMENT_NAME
    print(f"  [DEBUG][run_judge] api_version:           {api_version}")
    print(f"  [DEBUG][run_judge] model_name:            {model_name}")

    # Use the IR file (already normalized) so the judge doesn't re-normalize
    # and lose trajectories.
    ir_file = os.path.join(run_dir, "trajectory_ir.json")
    log_file = ir_file if os.path.exists(ir_file) else input_path

    # Perform `config.num_runs` independent judge iterations. The judge writes
    # each iteration's per-task results to runs/run{N}.json under judge_out_dir.
    for run_number in range(1, config.num_runs + 1):
        if config.num_runs > 1:
            print(f"\n  [run_judge] === Iteration {run_number}/{config.num_runs} ===")
        judge_module.run_single_iteration(
            run_number=run_number,
            base_output_dir=judge_out_dir,
            ground_truth_failures=gt_failures,
            api_version=api_version,
            model_name=model_name,
            log_file=log_file,
        )

    # When > 1 iteration was run, emit the mean/std aggregate the paper reports.
    if config.num_runs > 1:
        judge_module.create_aggregate_summary(judge_out_dir, config.num_runs)

    print(f"  Output: {judge_out_dir}")
    return judge_out_dir


# ---------- Stage: Report ----------

def run_report(judge_out_dir: str, run_dir: str):
    """Generate failure frequency plots."""
    from agentrx.reports.analyze_failure_frequencies import (
        load_and_analyze_json, plot_predicted_frequency,
        plot_ground_truth_frequency, plot_comparison,
    )

    banner("Stage 6/6: Report Generation")

    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)

    # Find the run results JSON
    runs_dir = os.path.join(judge_out_dir, "runs")
    if not os.path.isdir(runs_dir):
        print("  [SKIP] No judge runs found, skipping report generation")
        return

    json_files = [f for f in os.listdir(runs_dir) if f.endswith(".json")]
    if not json_files:
        print("  [SKIP] No judge result files found")
        return

    for jf in json_files:
        json_path = os.path.join(runs_dir, jf)
        try:
            pred_freq, gt_freq = load_and_analyze_json(json_path)
            plot_predicted_frequency(pred_freq, output_file=os.path.join(plots_dir, "predicted.png"))
            if gt_freq:
                plot_ground_truth_frequency(gt_freq, output_file=os.path.join(plots_dir, "gt.png"))
                plot_comparison(pred_freq, gt_freq, output_file=os.path.join(plots_dir, "comparison.png"))
            print(f"  Plots saved to: {plots_dir}")
        except Exception as e:
            print(f"  [WARN] Plotting failed for {jf}: {e}")

    print(f"  Output: {plots_dir}")


# ---------- Domain auto-detection ----------

def guess_domain(input_path: str) -> str:
    """Try to guess the domain from the file path or content."""
    path_lower = input_path.lower()
    if "tau" in path_lower or "retail" in path_lower:
        return "tau"
    if "magentic" in path_lower:
        return "magentic"
    if "flash" in path_lower or "incident" in path_lower:
        return "flash"

    # Peek at the file content
    try:
        with open(input_path, "r", encoding="utf-8-sig") as f:
            head = f.read(5000)
        if "tau" in head.lower() or "retail" in head.lower():
            return "tau"
        if "magentic" in head.lower():
            return "magentic"
    except Exception:
        pass

    # Default to flash (uses LLM-based IR fallback for unknown formats)
    return "flash"


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="AgentVerify: Push-button pipeline for trajectory analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py data/my_trajectory.json
  python run.py data/my_trajectory.json --domain tau
  python run.py data/my_trajectory.json --stage ir          # IR only
  python run.py data/my_trajectory.json --from-stage check   # resume from checking
  python run.py data/my_trajectory.json --skip-dynamic       # faster, no per-step invariants
  python run.py data/my_trajectory.json --skip-judge         # skip LLM judge
  python run.py data/my_trajectory.json --endpoint trapi     # use TRAPI (Microsoft Research internal)
  python run.py data/my_trajectory.json --run-name my_run    # custom run name
        """,
    )
    parser.add_argument("input", help="Path to trajectory file (JSON/JSONL)")
    parser.add_argument("--domain", default=None,
                        choices=["tau", "flash", "magentic"],
                        help="Domain (auto-detected if not specified)")
    parser.add_argument("--endpoint", default=g.DEFAULT_ENDPOINT, choices=["copilot", "azure", "trapi"],
                        help="LLM endpoint (default: copilot). Options: copilot (GitHub CLI), azure, trapi.")
    parser.add_argument("--stage", default=None, choices=STAGES,
                        help="Run ONLY this stage")
    parser.add_argument("--from-stage", default=None, choices=STAGES,
                        help="Resume from this stage (skips earlier stages)")
    parser.add_argument("--skip-static", action="store_true",
                        help="Skip static invariant generation (use empty static set; only dynamic invariants will drive checking)")
    parser.add_argument("--skip-dynamic", action="store_true",
                        help="Skip dynamic invariant generation (faster)")
    parser.add_argument("--dynamic-mode", default="stepbystep", choices=["stepbystep", "oneshot"],
                        help="Dynamic invariant mode: stepbystep (per-step, slower) or oneshot (single LLM call per trajectory, faster)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge and report stages")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground truth JSON for judge accuracy comparison")
    parser.add_argument("--run-name", default=None,
                        help="Custom name for this run (default: auto-generated)")
    parser.add_argument("--run-dir", default=None,
                        help="Resume into an existing run directory")

    # --- Judge-stage knobs (override individual axes of agentrx.pipeline.profiles.PAPER_DEFAULT) ---
    parser.add_argument("--prompt-mode", default=PAPER_DEFAULT.prompt_mode,
                        choices=["baseline", "checklist", "examples", "combined"],
                        help=f"Judge prompt taxonomy mode (default: {PAPER_DEFAULT.prompt_mode}, paper-faithful)")
    parser.add_argument("--exec-mode", default=PAPER_DEFAULT.exec_mode,
                        choices=["violations-after", "stepbystep", "violations-before"],
                        help=f"Judge execution mode (default: {PAPER_DEFAULT.exec_mode}, paper-faithful)")
    parser.add_argument("--prompt-style", default=PAPER_DEFAULT.prompt_style,
                        choices=["paper", "release"],
                        help=f"Judge system-prompt builder (default: {PAPER_DEFAULT.prompt_style!r}). "
                             "'release' is the originally-released f-string templates and dominates the "
                             "paper-mirror style on the tau-29 ablation. 'paper' reproduces the paper-mirror "
                             "concat builder verbatim.")
    parser.add_argument("--no-context", action="store_true",
                        help="Do not inject deduplicated violation context into the judge prompt "
                             "(paper-faithful default injects context when the check stage produced it)")
    nl_group = parser.add_mutually_exclusive_group()
    nl_group.add_argument("--include-nl-violations", dest="include_nl_violations",
                          action="store_true", default=None,
                          help=f"Include nl_check violations in the judge context (default: "
                               f"{PAPER_DEFAULT.include_nl_check_violations}, paper-faithful)")
    nl_group.add_argument("--exclude-nl-violations", dest="include_nl_violations",
                          action="store_false",
                          help="Drop nl_check violations from the judge context "
                               "(reproduces the paper's 'Without NL Check Viol.' appendix table)")
    parser.add_argument("--num-runs", type=int, default=PAPER_DEFAULT.num_runs,
                        help=f"Number of independent judge iterations (default: {PAPER_DEFAULT.num_runs}, "
                             "paper-faithful). When >1, writes runs/run{N}.json per iteration and "
                             "emits a mean/std aggregate summary.")

    args = parser.parse_args()

    # Resolve judge-stage axes into a single immutable RunConfig that flows
    # through run_pipeline -> run_judge. Anything not overridden inherits from
    # PAPER_DEFAULT, so an unflagged invocation produces the paper recipe.
    args.judge_config = RunConfig(
        prompt_mode=args.prompt_mode,
        exec_mode=args.exec_mode,
        with_context=PAPER_DEFAULT.with_context and not args.no_context,
        num_runs=args.num_runs,
        prompt_style=args.prompt_style,
        include_nl_check_violations=(
            PAPER_DEFAULT.include_nl_check_violations
            if args.include_nl_violations is None
            else args.include_nl_violations
        ),
    )

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    # If input is a directory, collect all trajectory files and run each one
    if os.path.isdir(input_path):
        trajectory_files = sorted(
            p for p in Path(input_path).rglob("*")
            if p.is_file() and p.suffix.lower() in (".json", ".jsonl")
        )
        if not trajectory_files:
            print(f"Error: No .json or .jsonl files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(trajectory_files)} trajectory file(s) in {input_path}:\n")
        for i, f in enumerate(trajectory_files, 1):
            print(f"  {i}. {f.relative_to(input_path)}")
        print()

        results = {}
        for i, traj_file in enumerate(trajectory_files, 1):
            banner(f"File {i}/{len(trajectory_files)}: {traj_file.name}")
            try:
                run_pipeline(str(traj_file), args)
                results[str(traj_file)] = "OK"
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Stopping batch run.")
                results[str(traj_file)] = "INTERRUPTED"
                break
            except Exception as e:
                print(f"\n[ERROR] Failed on {traj_file.name}: {e}")
                results[str(traj_file)] = f"FAILED: {e}"

        banner("Batch Run Summary")
        for filepath, status in results.items():
            name = Path(filepath).name
            print(f"  {name:50s} {status}")
        print()
        failed = sum(1 for s in results.values() if s.startswith("FAILED"))
        print(f"  {len(results)} processed, {len(results) - failed} succeeded, {failed} failed")
        print()
        sys.exit(1 if failed else 0)
    else:
        run_pipeline(input_path, args)


def run_pipeline(input_path: str, args):
    """Run the full pipeline on a single trajectory file."""
    # Detect domain
    domain = args.domain or guess_domain(input_path)
    print(f"Domain: {domain}")

    # Validate endpoint config before doing any work
    validate_endpoint_config(args.endpoint)

    # Set up run directory
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
    else:
        stem = Path(input_path).stem
        run_name = args.run_name or f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(str(REPO_ROOT), "runs", run_name)
    ensure_dir(run_dir)

    print(f"Run directory: {run_dir}")

    # Load or init state
    state = load_state(run_dir)
    state["config"] = {
        "input": input_path,
        "domain": domain,
        "endpoint": args.endpoint,
        "started": state.get("config", {}).get("started", datetime.now().isoformat()),
    }
    save_state(run_dir, state)

    # Determine which stages to run
    completed = set(state.get("completed_stages", []))

    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        start = stage_index(args.from_stage)
        stages_to_run = STAGES[start:]
    else:
        stages_to_run = list(STAGES)

    if args.skip_dynamic:
        stages_to_run = [s for s in stages_to_run if s != "dynamic"]
    if args.skip_static:
        stages_to_run = [s for s in stages_to_run if s != "static"]
    if args.skip_judge:
        stages_to_run = [s for s in stages_to_run if s not in ("judge", "report")]
    # Print plan
    print(f"\nStages to run: {' -> '.join(stages_to_run)}")
    if completed:
        print(f"Previously completed: {', '.join(completed)}")
    print()

    pipeline_start = time.perf_counter()

    # Paths that get filled in as stages complete (or loaded from prior runs)
    ir_path = os.path.join(run_dir, "trajectory_ir.json")
    static_inv_path = os.path.join(run_dir, "static_invariants.json")
    dynamic_inv_dir = os.path.join(run_dir, "dynamic_invariants")
    checker_dir = os.path.join(run_dir, "checker_results")
    judge_dir = os.path.join(run_dir, "judge_output")

    try:
        # --- IR ---
        if "ir" in stages_to_run:
            ir_path = run_ir(input_path, run_dir, domain, args.endpoint, state)
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"ir"})
            save_state(run_dir, state)
        elif not os.path.exists(ir_path):
            print("[INFO] Running IR stage (required by later stages)")
            ir_path = run_ir(input_path, run_dir, domain, args.endpoint, state)
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"ir"})
            save_state(run_dir, state)

        # --- Static Invariants ---
        if "static" in stages_to_run:
            static_inv_path = run_static(input_path, run_dir, domain, args.endpoint, state)
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"static"})
            save_state(run_dir, state)
        elif not os.path.exists(static_inv_path) and any(s in stages_to_run for s in ["check", "dynamic"]):
            if args.skip_static:
                print("[INFO] --skip-static set: writing empty static_invariants.json")
                with open(static_inv_path, "w", encoding="utf-8") as f:
                    json.dump({"invariants": []}, f, indent=2)
                state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"static"})
                save_state(run_dir, state)
            else:
                print("[INFO] Running static invariant stage (required by later stages)")
                static_inv_path = run_static(input_path, run_dir, domain, args.endpoint, state)
                state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"static"})
                save_state(run_dir, state)

        # --- Dynamic Invariants ---
        if "dynamic" in stages_to_run:
            dynamic_inv_dir = run_dynamic(input_path, run_dir, domain, args.endpoint, static_inv_path, state,
                                          dynamic_mode=args.dynamic_mode)
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"dynamic"})
            save_state(run_dir, state)

        # --- Check ---
        if "check" in stages_to_run:
            dyn_dir = dynamic_inv_dir if os.path.isdir(dynamic_inv_dir) else None
            checker_dir = run_check(ir_path, run_dir, domain, args.endpoint, static_inv_path, dyn_dir)
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"check"})
            save_state(run_dir, state)

        # --- Judge ---
        if "judge" in stages_to_run:
            violation_ctx = checker_dir if os.path.isdir(checker_dir) else None
            judge_dir = run_judge(
                ir_path, run_dir, domain, args.endpoint,
                violation_context_dir=violation_ctx,
                ground_truth_file=args.ground_truth,
                config=args.judge_config,
            )
            state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"judge"})
            save_state(run_dir, state)

        # --- Report ---
        if "report" in stages_to_run:
            if os.path.isdir(judge_dir):
                run_report(judge_dir, run_dir)
                state["completed_stages"] = list(set(state.get("completed_stages", [])) | {"report"})
                save_state(run_dir, state)
            else:
                print("  [SKIP] No judge output to report on")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress saved. Resume with:")
        not_done = [s for s in stages_to_run if s not in state.get("completed_stages", [])]
        if not_done:
            print(f"  python run.py {input_path} --run-dir {run_dir} --from-stage {not_done[0]}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Stage failed: {e}")
        not_done = [s for s in stages_to_run if s not in state.get("completed_stages", [])]
        if not_done:
            print(f"\nResume with:")
            print(f"  python run.py {input_path} --run-dir {run_dir} --from-stage {not_done[0]}")
        raise

    elapsed = time.perf_counter() - pipeline_start

    banner("Pipeline Complete")
    print(f"  Run directory: {run_dir}")
    print(f"  Completed stages: {', '.join(state.get('completed_stages', []))}")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print("  Outputs:")
    for label, path in [
        ("IR",         ir_path),
        ("Static Inv", static_inv_path),
        ("Dynamic Inv", dynamic_inv_dir),
        ("Violations",  checker_dir),
        ("Judge",       judge_dir),
        ("Plots",       os.path.join(run_dir, "plots")),
    ]:
        if os.path.exists(path):
            print(f"    {label:12s} {path}")
    print()


if __name__ == "__main__":
    main()
