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

