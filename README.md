# AgentRx 🩺

**Diagnosing AI Agent Failures from Execution Trajectories**

[[Paper]](https://arxiv.org/abs/2602.02475) [[Dataset]](https://huggingface.co/datasets/microsoft/AgentRx)

AI agents often fail in ways that are difficult to localize — executions are probabilistic, long-horizon, multi-agent, and mediated by noisy tool outputs. **AgentRx** is an automated, domain-agnostic diagnostic framework that pinpoints the *critical failure step* in a failed agent trajectory. It synthesizes constraints (invariants), evaluates them step-by-step, and produces an auditable validation log of constraint violations with associated evidence. An LLM-based judge uses this log to localize the critical step and classify the failure into a grounded 10-category taxonomy.

AgentRx improves step localization and failure attribution over existing baselines across three domains: structured API workflows (Tau-bench), incident management (Flash), and open-ended web/file tasks (Magentic-One).

```
Raw logs ──▶ Trajectory IR ──▶ Invariants ──▶ Checker ──▶ Judge ──▶ Reports
```

---

## Quick Start

```bash
# Setup
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -e .                # installs agentrx + all dependencies
cp .env.example .env            # Fill in your Azure or TRAPI endpoint details

# Local dev: skip ManagedIdentity IMDS probe
export AZURE_TOKEN_CREDENTIALS=dev  # or add to your .env file

# Run the full pipeline end-to-end
python run.py trajectory.json

# Specify domain explicitly
python run.py trajectory.json --domain tau
```

You can also install directly from GitHub without cloning:
```bash
pip install git+https://github.com/microsoft/AgentRx.git
```

All outputs are saved to `runs/<run_name>/`.

---

## Step-by-Step Usage

You can run each stage individually and inspect the results between stages:

```bash
# 1. Normalize raw logs into Trajectory IR
python run.py trajectory.json --stage ir --run-name my_run

# 2. Generate static invariants
python run.py trajectory.json --stage static --run-dir runs/my_run

# 3. Generate dynamic (per-step) invariants
python run.py trajectory.json --stage dynamic --run-dir runs/my_run

# 4. Check all invariants against the trajectory
python run.py trajectory.json --stage check --run-dir runs/my_run

# 5. Run LLM judge for root-cause classification
python run.py trajectory.json --stage judge --run-dir runs/my_run

# 6. Generate report plots
python run.py trajectory.json --stage report --run-dir runs/my_run
```

---

## Pipeline Stages

| # | Stage | Output |
|---|-------|--------|
| 1 | **IR** — Normalize raw logs into canonical Trajectory IR | `trajectory_ir.json` |
| 2 | **Static** — Generate policy/tool/structure invariants | `static_invariants.json` |
| 3 | **Dynamic** — Generate per-step context-aware invariants | `dynamic_invariants/` |
| 4 | **Check** — Evaluate invariants, record violations | `checker_results/` |
| 5 | **Judge** — LLM classifies root-cause failure (10-category taxonomy) | `judge_output/` |
| 6 | **Report** — Failure frequency plots | `plots/` |

---

## Directory Structure

```
AgentRx/
├── run.py                       # CLI entry point (backward-compatible)
├── pyproject.toml               # Package configuration (pip install -e .)
├── requirements.txt             # Python dependencies
├── agentrx/                     # Main package
│   ├── cli.py                   # Console script entry point
│   ├── ir/                      # Trajectory IR normalization
│   ├── invariants/              # Invariant generation & checking
│   ├── judge/                   # LLM-as-a-Judge evaluation
│   ├── llm_clients/             # Azure OpenAI & TRAPI clients
│   ├── pipeline/                # Config (globals.py), utilities
│   └── reports/                 # Analysis & visualization
├── data/                        # Domain policies, tool schemas, ground truth
├── trajectories/                # Sample trajectories (tau, magentic, test)
└── runs/                        # Pipeline outputs (one folder per run)
```

---

## Supported Domains

| Domain | Flag | Description |
|--------|------|-------------|
| **tau** | `--domain tau` | Tau-bench retail customer service |
| **magentic** | `--domain magentic` | Magentic-One multi-agent |
| **flash** | `--domain flash` | Flash/orchestrator incident traces |
| *(auto)* | *(default)* | Auto-detected; unknown formats use LLM-based IR fallback |

---

## Reproducing the Paper

The paper-default recipe is encoded in `agentrx.pipeline.profiles.PAPER_DEFAULT`
and is what `run.py` uses when no judge-stage flags are passed:

```
prompt_mode  = combined            # taxonomy block fed to the judge
exec_mode    = violations-after    # when category labelling sees violations
with_context = True                # inject deduplicated violation context
num_runs     = 3                   # paper reports mean ± std over n=3 runs
```

```bash
# Paper-default: produces runs/<name>/judge_output/runs/run{1..3}.json plus
# an aggregate summary with mean/std for every table cell.
python run.py trajectory.json
```

Each axis is overridable on the CLI to reproduce specific ablation cells. The
flags compose orthogonally; omitted flags inherit from `PAPER_DEFAULT`.

| Flag | Values | Paper cell |
|------|--------|------------|
| `--prompt-mode` | `baseline` \| `checklist` \| `examples` \| `combined` | Tables 2, 4 (prompt-mode ablations) |
| `--exec-mode` | `violations-after` \| `stepbystep` \| `violations-before` | Tables 3, 5 (execution-mode ablations) |
| `--no-context` | (flag) | "Without violation context" rows |
| `--num-runs N` | integer ≥ 1 | Set to `1` for fast smoke tests (skips aggregate) |
| `--dynamic-mode` | `stepbystep` \| `oneshot` | Table 3 (one-shot dynamic invariants) |
| `--skip-static` | (flag) | Dynamic-only ablation |
| `--skip-dynamic` | (flag) | Global/static-only ablation |
| `--prompt-style` | `release` \| `paper` | Selects which judge system-prompt builder is used. `release` is the default shipped here; `paper` mirrors the verbatim taxonomy + section ordering of the camera-ready prompt and is what the paper's table rows were produced with. |
| `--include-nl-violations` / `--exclude-nl-violations` | (mutually exclusive flags) | Controls whether `nl_check` violations are forwarded into the judge context. `include` is the release default; `exclude` produces the "without NL-check violations" ablation row. |

When `--num-runs > 1`, the wrapper invokes
`agentrx.judge.judge.create_aggregate_summary` after the loop, which writes
the mean/std/CV table the paper reports under
`judge_output/aggregate_summary.json`.

### Reproducibility matrix (paper vs. release)

Two prompt-style profiles are encoded in `agentrx.pipeline.profiles`:

| Profile | `prompt_style` | `include_nl_check_violations` | Use this when... |
|---------|----------------|-------------------------------|------------------|
| `PAPER_DEFAULT` | `release` | `True` | You want the shipped release defaults (production behaviour). |
| `PAPER_MIRROR_DEFAULT` | `paper` | `True` | You want to reproduce the verbatim paper judge prompt for table comparisons. |

Recipe table for the headline rows:

| Goal | Command |
|------|---------|
| Release default (this repo) | `python run.py trajectory.json` |
| Paper Tables 4 (best cell) — paper prompt | `python run.py trajectory.json --prompt-style paper` |
| Paper "without NL-check violations" ablation | `python run.py trajectory.json --prompt-style paper --exclude-nl-violations` |
| Release sanity — no violation context at all | `python run.py trajectory.json --no-context` |

`--prompt-style` and `--include/--exclude-nl-violations` compose orthogonally
with the other ablation flags above. Both knobs are first-class fields on
`RunConfig` (`agentrx/pipeline/profiles.py`) and are validated at construction
time via `Literal[...]` membership; passing an invalid value raises before any
LLM call is issued. There are **no environment variables** that silently
toggle judge behaviour — every paper axis is a CLI flag, recorded into
`run_config.json` for every run.

### Reproduction driver scripts

The end-to-end paper sweeps and aggregators live under the top-level
[scripts/](scripts/) package so they stay versioned and importable:

| Script | Purpose |
|--------|---------|
| `python -m scripts.run_tau_ablation` | Runs the τ-bench paper ablation cells (every `prompt_mode` × `exec_mode` × `prompt_style`) and stores per-cell `runs/` outputs. |
| `python -m scripts.score_tau_ablation` | Aggregates per-run JSON across the τ ablation cells into a single side-by-side table. |
| `python -m scripts.score_flash` | Aggregates Flash-domain runs (requires the internal Flash dataset). |
| `bash scripts/sweeps/run_tau29.sh` | One-shot driver for the 29-trajectory τ subset shipped under `data/tau_dataset/`. |
| `bash scripts/sweeps/run_flash42.sh` | One-shot driver for the 42-trajectory Flash subset (internal only). |
| `bash scripts/sweeps/run_magentic27.sh` | One-shot driver for the 27-trajectory `magentic*` subset. |

The shell drivers resolve the repo root from their own location and honour
two overrides for non-default Python environments and log destinations:

```bash
AGENTRX_PYTHON=/path/to/venv/bin/python \
AGENTRX_LOG=/tmp/tau29.log \
bash scripts/sweeps/run_tau29.sh
```

### Operational knobs (not paper axes)

These tune runtime behaviour and are deliberately kept as environment
variables — they do not affect any reported number:

| Env var | Default | Purpose |
|---------|---------|---------|
| `AGENTRX_PYCHECK_TIMEOUT_SEC` | `2.0` | Wall-clock budget for each `python_check` invariant. The checker uses a `threading.Thread`+`join(timeout=...)` pattern (cross-platform — works on Windows, where `signal.SIGALRM` is unavailable). |

### What this repo reproduces directly

End-to-end from a fresh clone you can reproduce, on the trajectories shipped under [data/](data/):

- **τ-bench retail rows** of Tables 4, 5, 6 — all `prompt_mode` × `exec_mode` × `with_context` cells.
- **Magentic-One rows** of Tables 4, 5, 6 — same axes, run over the 44-trajectory `magentic_dataset/`.
- **Magentic\*** (27-trajectory subset) rows — once you constrain inputs to the ids in [data/ground_truth/magentic_star_ids.json](data/ground_truth/magentic_star_ids.json). The id list is shipped; a `--subset` CLI filter is not yet wired into `run.py` (planned), so today you must filter input files manually.
- **Table 3 (token statistics)** — derivable from per-run `judge_output/*.json` (a paper-format renderer is not yet ported into this repo; see below).

### What this repo does not reproduce, and why

- **Flash domain (Tables 4, 5, 6, and the Flash policy used by static invariant generation).** The Flash dataset is Microsoft-internal production incident data and cannot be redistributed. No GT or trajectories are shipped, and there is no public Flash policy. Flash rows of the paper cannot be reproduced from this repo alone.
- **Table 2 (comparison against Who&When).** Requires running an external baseline; see the next section.
- **Sampling determinism for `gpt-5` / `o3`.** The paper does not specify `temperature`, `top_p`, `seed`, or `max_tokens` for the judge LLM, and the underlying models are not bit-exact reproducible. Expect cell-level deviations within the n=3 standard deviations the paper reports.
- **Headline 23.6% / 22.9% improvement figures from the abstract.** Those numbers are not traceable to a specific table cell in the camera-ready and cannot be regenerated from a single sweep.
- **Constraint-generator prompts in the paper appendix** (sections marked `\TODO{}` in the LaTeX source). The actual prompts live in [agentrx/invariants/static_invariant_generator.py](agentrx/invariants/static_invariant_generator.py) / [dynamic_invariant_generator.py](agentrx/invariants/dynamic_invariant_generator.py); the paper text is incomplete, not the code.

### Reproducing Table 2 (Who&When comparison)

Table 2 compares AgentRx's judge against the **Who&When** (W&W) failure-attribution baseline, plus a prompt-modified variant the paper calls **W&W\***. W&W is third-party code with its own license and is not vendored here. To reproduce the table:

1. Clone the upstream W&W repository:

   ```bash
   git clone https://github.com/mingyin1/Agents_Failure_Attribution
   ```

2. Run W&W's `Lib/utils.py:all_at_once` (and/or `step_by_step` / `binary_search`) judges against the 16-trajectory subset already shipped under [data/magentic_dataset_whowhen/](data/magentic_dataset_whowhen/). This subset is the intersection of the 44 Magentic GT trajectories with W&W's input-staging format, which is what the paper evaluates on.

3. To reproduce the **W&W\*** row, apply the single prompt modification the paper specifies (`eval.tex` §4.2, "first unrecoverable critical step"): in W&W's prompt templates, replace `"the first mistake"` / `"first error"` / `"first made mistake"` with `"first UNRECOVERABLE critical mistake"`. Leave everything else (system prompt, ground-truth-in-prompt convention, output format, regex parsing) untouched so W&W's own `evaluate.py` works unmodified.

4. Run AgentRx's judge against the same 16-trajectory subset under the paper-default recipe (see above) for the AgentRx row.

5. Compare step-accuracy on those 16 trajectories. The paper reports W&W's `all_at_once` variant at ~12.5% on Magentic, W&W\* materially higher, and AgentRx materially higher again.

A turnkey W&W runner has intentionally not been ported here pending a license/attribution review for the upstream `mingyin1` code; the procedure above is the supported path.

---

## Configuration

LLM settings are loaded from environment variables (via `.env` or shell):

Copy the template and fill in your values:
```bash
cp .env.example .env
```

```bash
# Azure OpenAI (default endpoint)
AGENT_VERIFY_ENDPOINT=                # e.g., "https://my-resource.openai.azure.com/"
AGENT_VERIFY_DEPLOYMENT=              # e.g., "gpt-5"
AGENT_VERIFY_MODEL_NAME=              # e.g., "gpt-5" (must match deployment; used by judge)

# TRAPI (Microsoft Research internal, use --endpoint trapi)
AGENT_VERIFY_TRAPI_INSTANCE=          # e.g., "my-instance/my-pool"
AGENT_VERIFY_TRAPI_DEPLOYMENT_NAME=   # e.g., "my-deployment-name"
SCOPE=                                # Azure AD scope for TRAPI
```

Both endpoints use **Azure AD token-based auth** (`az login` or Managed Identity).

> **Note:** TRAPI is a Microsoft Research internal endpoint. External teams should use `--endpoint azure` (default).

---

## Failure Taxonomy

| # | Category | Description |
|---|----------|-------------|
| 1 | **Instruction/Plan Adherence Failure** | Skips steps or adds unnecessary actions |
| 2 | **Invention of New Information** | Fabricates or omits ungrounded facts |
| 3 | **Invalid Invocation** | Malformed tool call (wrong args/types/schema) |
| 4 | **Misinterpretation of Tool Output** | Incorrect reasoning about tool results |
| 5 | **Intent-Plan Misalignment** | Pursues wrong objective |
| 6 | **Underspecified User Intent** | Missing information to proceed |
| 7 | **Intent Not Supported** | Action can't be performed with available tools |
| 8 | **Guardrails Triggered** | Blocked by safety/RAI/access policies |
| 9 | **System Failure** | Infra errors (timeouts, unreachable endpoints) |
| 10 | **Inconclusive** | Insufficient evidence to classify |

---

## Running Individual Modules

Each module can also be run standalone:

**Static Invariant Generator** — generate policy/tool invariants:
```bash
python agentrx/invariants/static_invariant_generator.py --input-path trajectory.json --domain tau
```

**Dynamic Invariant Generator** — generate per-step context-aware invariants:
```bash
python agentrx/invariants/dynamic_invariant_generator.py --input-path trajectory.json --domain tau --mode stepbystep
```

**Checker** — evaluate invariants against a trajectory:
```bash
python agentrx/invariants/checker.py --input-path trajectory.json --static-invariants static_inv.json --dynamic-invariants-dir dyn_inv/
```

**Judge** — run LLM-as-a-Judge classification:
```bash
python agentrx/judge/judge.py --domain tau --log_file trajectory.json --mode combined
```

---

## Third-Party Code

This project uses the following third-party open source packages (installed via `requirements.txt`):

- **openai** — OpenAI Python client (MIT License)
- **azure-identity** / **azure-core** — Azure SDK authentication (MIT License)
- **matplotlib** — Plotting and visualization (PSF-based License)
- **tiktoken** — Token counting (MIT License)
- **httpx** — HTTP client (BSD License)

See [requirements.txt](requirements.txt) for the full list of dependencies.

---

## Troubleshooting

### `DefaultAzureCredential` timeout on local machines

The Azure SDK's `DefaultAzureCredential` tries `ManagedIdentityCredential` before `AzureCliCredential`. On a local dev machine this probes the IMDS endpoint which doesn't exist locally, causing a ~5-10s timeout before falling back. This is expected behavior — the probe is how `DefaultAzureCredential` detects the hosting environment.

**Fix:** Set the `AZURE_TOKEN_CREDENTIALS` environment variable to `dev` to exclude deployed-service credentials (e.g. `ManagedIdentityCredential`, `WorkloadIdentityCredential`) from the chain, so `DefaultAzureCredential` skips straight to developer-tool credentials like `AzureCliCredential`:

```bash
# PowerShell
$env:AZURE_TOKEN_CREDENTIALS = "dev"

# Bash / Linux / macOS
export AZURE_TOKEN_CREDENTIALS=dev
```

Or add `AZURE_TOKEN_CREDENTIALS=dev` to your `.env` file.

> Requires `azure-identity >= 1.23.0`. See [Exclude a credential type category](https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#exclude-a-credential-type-category) for details.

---

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

---

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.

---

## Citation

If you use AgentRx, please cite:

```bibtex
@article{barke2026agentrx,
  title={AgentRx: Diagnosing AI Agent Failures from Execution Trajectories},
  author={Barke, Shraddha and Goyal, Arnav and Khare, Alind and Singh, Avaljot and Nath, Suman and Bansal, Chetan},
  journal={arXiv preprint arXiv:2602.02475},
  year={2026}
}
```

