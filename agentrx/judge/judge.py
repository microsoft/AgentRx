#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation System (Merged)

This module consolidates the best features of the refactored flexible-prompt system
and the updated robust execution system with 'Synth' log normalization.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
import argparse
import re
import traceback

from enum import Enum
from collections import Counter
from statistics import mean, median, stdev, variance, multimode
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import List, Optional, Dict, Any

# --- Imports from agent_verify framework ---
try:
    import agentrx.pipeline.globals as g
    from agentrx.llm_clients.azure import LLMAgent as LLMAgentAzure
    from agentrx.llm_clients.trapi import LLMAgent as LLMAgentTrapi
    # Add metrics if available
    import agentrx.reports.metrics as metrics
    
    # Analysis & Pipeline tools
    from agentrx.reports.analyze_failure_frequencies import load_and_analyze_json, plot_predicted_frequency, plot_ground_truth_frequency, plot_comparison
    from agentrx.ir.trajectory_ir import tau_bench_ir, load_trajectories, flash_ir, magentic_ir, validate_ir, llm_ir, ensure_ir
    from agentrx.invariants.domain_registry import DOMAIN_REGISTRY, get_domain_config, register_domain
except ImportError:
    pass

# --- Configurations ---

# Default output directory
RESULTS_DIR = "output_results"

# Global Configs (will be updated by main)
RUN_WITH_CONTEXT = False
ENDPOINT_USED = None  # will be set in main — "copilot", "azure", or "trapi"
USE_GROUND_TRUTH = True
PROMPT_MODE = "combined"     # "baseline", "checklist", "examples", "combined"
EXECUTION_MODE = "violations-after" # "violations-after", "stepbystep", "violations-before"
DOMAIN = None

# Directory paths (can be overridden via command line)
EXAMPLES_DIR = None  # Will default to os.path.join(os.path.dirname(__file__), "few_shot_examples")
VIOLATION_CONTEXT_DIR = None  # Will default to os.path.join(os.path.dirname(__file__), "pipeline", "violation_results_20251222_045346", "judge_context")

# Debug Flags
DEBUG = os.getenv("DEBUG", "0") == "1"
DEBUG_PROMPTS = os.getenv("DEBUG_PROMPTS", "0") == "1"
DEBUG_SYNTH = os.getenv("DEBUG_SYNTH", "0") == "1"

def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}")

# --- Taxonomy Definitions (from Refactored) ---

TAXONOMY_DATA = {
    1: {
        "name": "Instruction/Plan Adherence Failure",
        "desc_standard": (
            "The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. "
            "\nThis covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, "
            "e.g., extra tool calls) that deviate from the static plan, \ndomain policy or orchestrator plan."
        ),
        "desc_checklist": (
            "Goal is correct, but the agent deviates from the required plan by ignoring directives and skipping steps "
            "despite having enough information. \nThis covers both under-execution (missed steps) and over-execution "
            "(unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, "
            "domain policy or orchestrator plan."
        ),
        "checklist": [
            "Can you state the user's goal, and do the agent's intent and end goal match that goal (i.e., the agent is not solving the wrong problem)?",
            "Was all the required information already available at this step (user intent, required context, prior tool outputs)?",
            "Is there a step where the ground-truth/policy requires an action (tool call, question, confirmation, ordering) and the agent did something different (skipped it / reordered it / added extra unneeded action)?"
        ]
    },
    2: {
        "name": "Invention of New Information",
        "desc_standard": (
            "The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. "
            "This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification."
        ),
        "desc_checklist": (
            "The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. "
            "This includes fabricating unsupported facts, hallucinating details, or omitting relevant information."
        ),
        "checklist": [
            "Can you pinpoint the exact invented/altered/omitted claim, value, or assumption the agent used?",
            "Is that claim absent from all evidence available up to that step (user text, provided context, tool outputs)?",
            "Did the agent rely on that claim to decide an action or produce the failing conclusion (not just harmless wording)?",
        ]
    },
    3: {
        "name": "Invalid Invocation",
        "desc_standard": (
            "The agent encounters errors triggered by inputs that can't be parsed or validated e.g., Kusto syntax errors "
            "or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs."
        ),
        "desc_checklist": (
            "Tool call fails because the request is ill-formed (missing args, wrong fields/types, malformed query, schema mismatch)."
        ),
        "checklist": [
            "At the failure step, did the agent attempt a tool call with a concrete invocation payload/arguments?",
            "Does the tool/runtime explicitly report a parse/validation/schema/syntax error for that call (e.g., missing field, invalid type, cannot parse, malformed query)?",
            "Is the error NOT primarily a network/timeout/service-unavailable/endpoint-unreachable issue (infra/connectivity)?",
            "Is the error NOT primarily a CAPTCHA/login/paywall refusal (access/guardrail block)?",
        ]
    },
    4: {
        "name": "Misinterpretation of Tool Output / Handoff Failure",
        "desc_standard": (
            "The agent incorrectly reasons about its own or another agent's tool output (like computation errors), "
            "leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output."
        ),
        "desc_checklist": (
            "The agent incorrectly reasons about its own or another agent's tool output, leading to incorrect assumptions or actions. "
            "This also includes cases where the agent considered only partial tool output."
        ),
        "checklist": [
            "Before (or at) the failure step, did the agent receive tool output or handoff output that is relevant to the failing decision?",
            "Did the agent state or imply a specific reasoning derived from that tool output?",
            "Does that reasoning contradict the tool output, omit a crucial part, or reflect a clear computation/logic error relative to the output?",
        ]
    },
    5: {
        "name": "Intent-Plan Misalignment",
        "desc_standard": (
            "The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. "
            "Covers both bad ordering/structure and plans aimed at the wrong objective."
        ),
        "desc_checklist": (
            "Agent misunderstands the user's intent/constraints and pursues the wrong objective or violates key constraints due to misunderstanding."
        ),
        "checklist": [
            "Do the agent's actions/plan optimize for a different goal OR violate a key constraint (not a minor wording/format issue)?",
            "Is the misalignment due to misunderstanding of intent/constraints (rather than missing required info from the user/context/tool outputs)?",
            "Is the misalignment not primarily caused by a tool error (invalid invocation, infra failure, or access/guardrail block)?",
        ]
    },
    6: {
        "name": "Underspecified User Intent",
        "desc_standard": (
            "The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution."
        ),
        "desc_checklist": (
            "The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution."
        ),
        "checklist": [
            "Can you identify a specific missing piece of information that is required to proceed correctly (e.g., date, address, account id, item variant)?",
            "Is that information absent from all evidence available up to that step (user text, provided context, and tool outputs)?",
            "Did the agent fail because it proceeded without obtaining this information OR because it did not ask for it when needed?",
        ]
    },
    7: {
        "name": "Intent Not Supported",
        "desc_standard": (
            "The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file."
        ),
        "desc_checklist": (
            "Requested action cannot be performed with available tools/capabilities."
        ),
        "checklist": [
            "Is the user requesting an action that requires an external capability/tool (e.g., listen to audio, access a private system, perform a human action)?",
            "Given the tool set available in this environment, is there no tool that can perform the requested action?",
            "Is the failure not primarily caused by infrastructure/connectivity issues?",
        ]
    },
    8: {
        "name": "Guardrails Triggered",
        "desc_standard": (
            "The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. "
            "Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, "
            "login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; "
            "it is an external/guardrail block."
        ),
        "desc_checklist": (
            "The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. "
        ),
        "checklist": [
            "Is there an explicit refusal/block signal (policy refusal, CAPTCHA, login required, 403, paywall, robots.txt, automation forbidden)?",
            "Would the plan be feasible and correct if this block were removed (i.e., the agent is not pursuing the wrong goal/constraints)?",
            "Is the failure not primarily due to malformed tool invocation (schema/syntax/args validation error)?",
            "Is the failure not primarily due to infrastructure/connectivity issues (timeouts, endpoint unreachable)?",
        ],
    },
    9: {
        "name": "System Failure",
        "desc_standard": (
            "The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable"
        ),
        "desc_checklist": (
            "The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
        ),
        "checklist": [
            "At the failure step, did the agent attempt a tool call or rely on a tool that should have been callable?",
            "Is there an explicit infra/connectivity error signal (timeout, connection refused, DNS failure, endpoint unreachable, service unavailable, premature termination)?",
            "Is the failure not primarily a parse/validation/schema/syntax error caused by malformed arguments?",
        ],
    },
    10: {
        "name": "Inconclusive (USE SPARINGLY)",
        "desc_standard": (
            "If you are not able to classify the failure into any of the above categories, label it as inconclusive and create your own category."
        ),
        "desc_checklist": (
            "None of 1-10 clearly apply; must provide a custom category label."
        ),
        "checklist": [
            "If labeling as 10, did you provide a non-empty custom_category describing the failure type?",
        ],
    }
}


STEP_INDEX_PROMPT_TEMPLATE = """

GIVEN INPUT:
- a full trajectory of an agent's interaction with a user (step-indexed)
- the ground-truth tool-call/action sequence the agent should have made
- (optional) Further, sometimes you may also be provided with a set of responses that the agent should have made in its response to the user at some step in the trajectory. Along with them, we will supply a boolean value indicating whether the agent's response included that output or not.

YOUR TASK is to determine ONLY the root-cause failure step index using the algorithm below.

ROOT-CAUSE DETECTION ALGORITHM:

Step 1 — Locate the first failure: Scan the trajectory step-by-step from the start and record the first failure.
Step 2 — Check if that failure was resolved: Look ahead in the trajectory for evidence that the error was resolved. If yes → Resolved; if no such evidence → Not resolved.
Step 3 — Decide and continue:
If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
If Not resolved: treat this step as the root-cause failure for the run and output ONLY that step index.

Output a JSON object in the following format:
{{
    "reason_for_index": <string>,
    "index": <int>
}}

""".strip()

# --- Classes (from Updated) ---

class FailureCase(Enum):
    NO_ERROR_PREDICTED = 0  # Judge claims no error - treat as wrong prediction
    INSTRUCTION_OR_PLAN_ADHERENCE_FAILURE = 1
    INVENTION_OF_NEW_INFORMATION = 2
    INVALID_INVOCATION = 3
    MISINTERPRETATION_OF_TOOL_OUTPUT = 4
    INTENT_PLAN_MISALIGNMENT = 5
    UNDERSPECIFIED_USER_INTENT = 6
    INTENT_NOT_SUPPORTED = 7
    GUARDRAILS_TRIGGERED = 8
    SYSTEM_FAILURE = 9
    INCONCLUSIVE = 10

class Failure:
    def __init__(self, task_id, failure_case, description, step_number, checklist_reasoning=None):
        self.task_id = task_id
        if isinstance(failure_case, int):
            try:
                failure_case = FailureCase(failure_case)
            except ValueError:
                # Handle invalid failure_case values (e.g., negative numbers or > 10)
                print(f"[WARN] Invalid failure_case value {failure_case}, treating as INCONCLUSIVE")
                failure_case = FailureCase.INCONCLUSIVE
        self.failure_case = failure_case
        self.description = description
        self.step_number = step_number
        self.checklist_reasoning = checklist_reasoning

class Report:
    def __init__(self, task_id, trajectory_length=1):
        self.task_id = task_id
        self.failures = []
        self.num_judges = 0
        self.trajectory_length = trajectory_length
        self.llm_call_telemetry: metrics.LLMCallTelemetry = None

    def add_failure(self, failure):
        self.failures.append(failure)
        self.num_judges += 1

    def to_dict(self):
        from dataclasses import asdict
        
        result = {}
        for key, value in self.__dict__.items():
            if key == "failures":
                result[key] = [
                    {
                        "task_id": f.task_id,
                        "failure_case": f.failure_case.value if hasattr(f.failure_case, 'value') else str(f.failure_case),
                        "description": f.description,
                        "step_number": f.step_number,
                        "checklist_reasoning": f.checklist_reasoning
                    } for f in value
                ]
            elif key == "llm_call_telemetry":
                # Convert LLMCallTelemetry dataclass to JSON-serializable dict
                if value is not None:
                    try:
                        telemetry_dict = asdict(value)
                        # Convert datetime objects to ISO format strings
                        if 'time' in telemetry_dict and telemetry_dict['time']:
                            time_info = telemetry_dict['time']
                            if 'start_time' in time_info and hasattr(time_info['start_time'], 'isoformat'):
                                time_info['start_time'] = time_info['start_time'].isoformat()
                            if 'end_time' in time_info and hasattr(time_info['end_time'], 'isoformat'):
                                time_info['end_time'] = time_info['end_time'].isoformat()
                        result[key] = telemetry_dict
                    except Exception:
                        result[key] = None
                else:
                    result[key] = None
            else:
                result[key] = value
            
        return result

    def compute_stats(self, gt_failure):
        failure_cases = [f.failure_case for f in self.failures]
        step_numbers = [f.step_number for f in self.failures]
        gt_failure_case = gt_failure.failure_case
        gt_step_number = gt_failure.step_number
        total = len(self.failures)

        if total == 0:
            return

        failure_count = Counter(failure_cases)
        self.frequency = {str(k.value): v for k, v in failure_count.items()}
        self.most_common_failure = str(failure_count.most_common(1)[0][0].value)
        self.modes = [str(mode.value) for mode in multimode(failure_cases)]
        
        failure_values = [fc.value for fc in failure_cases]
        
        # Failure case stats
        self.mean = mean(failure_values)
        self.median = median(failure_values)
        self.std_dev = stdev(failure_values) if total > 1 else 0.0
        self.variance = variance(failure_values) if total > 1 else 0.0
        self.min = min(failure_values)
        self.max = max(failure_values)
        self.proportions = {str(k.value): v / total for k, v in failure_count.items()}
        
        # Step number stats
        self.step_mean = mean(step_numbers) if step_numbers else 0
        self.step_median = median(step_numbers) if step_numbers else 0
        self.step_std_dev = stdev(step_numbers) if len(step_numbers) > 1 else 0.0
        self.step_variance = variance(step_numbers) if len(step_numbers) > 1 else 0.0
        self.step_min = min(step_numbers) if step_numbers else 0
        self.step_max = max(step_numbers) if step_numbers else 0
        
        # Comparison to Ground Truth
        failure_matches = [fc == gt_failure_case for fc in failure_cases]
        step_abs_errors = [abs(s - gt_step_number) for s in step_numbers]

        self.failure_case_accuracy = sum(failure_matches) / total if total > 0 else 0
        self.step_mae = mean(step_abs_errors) if step_abs_errors else 0
        self.step_error_distribution = dict(Counter(step_abs_errors))

        self.gt_failure_case = str(gt_failure.failure_case.value)
        self.gt_step_number = gt_failure.step_number
        self.gt_failure_description = gt_failure.description

# --- Few-Shot Examples (from Refactored) ---

def load_few_shot_examples():
    """Load few-shot examples from the few_shot_examples directory."""
    global EXAMPLES_DIR
    
    # Use global EXAMPLES_DIR if set, otherwise use default
    if EXAMPLES_DIR is None:
        examples_dir = os.path.join(os.path.dirname(__file__), "few_shot_examples")
    else:
        examples_dir = EXAMPLES_DIR
    
    # Mapping of category numbers to their example file names
    example_files = {
        1: "instruction_adherence_failure.json",
        2: "invention_of_new_information.json",
        3: "invalid_invocation.json",
        4: "misinterpretation_of_tool_output.json",
        5: "intent_plan_misalignment.json",
        6: "underspecified_user_intent.json",
        7: "intent_not_supported.json",
        8: "guardrails_triggered.json",
        9: "system_failure.json",
        10: None 
    }
    
    examples = {}
    loaded_count = 0
    missing_count = 0
    skipped_count = 0
    
    print(f"[FEW-SHOT] Looking for examples in: {examples_dir}")
    if not os.path.exists(examples_dir):
        print(f"[FEW-SHOT] WARNING: Examples directory does not exist: {examples_dir}")
    
    for category_num, filename in example_files.items():
        if filename:
            path = os.path.join(examples_dir, filename)
            try:
                with open(path, 'r') as f:
                    examples[category_num] = json.load(f)
                    loaded_count += 1
                    print(f"[FEW-SHOT] [OK] Loaded example for category {category_num}: {filename}")
            except FileNotFoundError:
                examples[category_num] = None
                missing_count += 1
                print(f"[FEW-SHOT] [MISS] Missing example for category {category_num}: {filename}")
            except json.JSONDecodeError as e:
                examples[category_num] = None
                missing_count += 1
                print(f"[FEW-SHOT] [ERR] Invalid JSON in example for category {category_num}: {filename} - {e}")
        else:
            examples[category_num] = None
            skipped_count += 1
    
    print(f"[FEW-SHOT] Summary: {loaded_count} loaded, {missing_count} missing, {skipped_count} skipped (no file defined)")
    
    return examples

# Initialize as None - will be loaded in main() after parsing args
FEW_SHOT_EXAMPLES = None

def ensure_few_shot_examples_loaded():
    """Ensure FEW_SHOT_EXAMPLES is loaded. Called after command line args are parsed."""
    global FEW_SHOT_EXAMPLES
    if FEW_SHOT_EXAMPLES is None:
        FEW_SHOT_EXAMPLES = load_few_shot_examples()
    return FEW_SHOT_EXAMPLES

def format_example_for_prompt(example_data):
    if example_data is None or (isinstance(example_data, str) and not example_data.strip()):
        return "No example available."
    example_json = json.dumps(example_data, separators=(',', ': '))
    return f"```json\n{example_json}\n```"

# --- Prompt Construction (The "Refactored" Logic) ---

def build_taxonomy_text(mode):
    use_checklist = mode in ["checklist", "combined"]
    use_examples = mode in ["examples", "combined"]
    
    # Ensure examples are loaded if needed
    if use_examples:
        ensure_few_shot_examples_loaded()
    
    # If using checklist, we prefer the Checklist Definitions.
    # If not using checklist (baseline/examples), we use Standard Definitions.
    use_checklist_desc = use_checklist
    
    taxonomy_text = "The failure taxonomy has the following categories:\n\n"
    
    for cat_id in sorted(TAXONOMY_DATA.keys()):
        data = TAXONOMY_DATA[cat_id]
        name = data["name"]
        
        # Select description
        if use_checklist_desc:
            desc = data["desc_checklist"]
        else:
            desc = data["desc_standard"]
            
        taxonomy_text += f"{cat_id}. {name}: {desc}\n"
        
        # Add checklist if enabled
        if use_checklist:
            checklist_items = data.get("checklist", [])
            if checklist_items:
                taxonomy_text += "   Checklist:\n"
                for item in checklist_items:
                    taxonomy_text += f"   - {item}\n"
        
        # Add examples if enabled
        if use_examples:
            example_data = FEW_SHOT_EXAMPLES.get(cat_id)
            if example_data:
                taxonomy_text += f"   Example:\n   {format_example_for_prompt(example_data)}\n"
        
        taxonomy_text += "\n"
        
    return taxonomy_text

# ---------------------------------------------------------------------------
# Prompt templates — each is self-contained with {taxonomy_block} and
# {invariants_violation_context} placeholders.  Use double-braces {{ }} to
# escape literal braces in the JSON examples.
#
# Matches the 5 active templates from agentrx.judge.py:
#   _TMPL_VIOLATIONS_BEFORE  ↔  BASE_SYSTEM_PROMPT_VIOLATIONS_BEFORE
#   _TMPL_NO_CONTEXT         ↔  BASE_SYSTEM_PROMPT
#   _TMPL_WITH_CONTEXT       ↔  BASE_SYSTEM_PROMPT_WITH_CONTEXT
#   _TMPL_FAILURE            ↔  FAILURE_PROMPT
#   STEP_INDEX_PROMPT_TEMPLATE (already defined above)  ↔  STEP_INDEX_PROMPT
# ---------------------------------------------------------------------------

_TMPL_VIOLATIONS_BEFORE = """
GIVEN INPUT:
- a full trajectory of an agent's interaction with a user (step-indexed)
- the ground-truth tool-call/action sequence the agent should have made
- optional: expected responses/outputs for some steps

YOUR TASK is to determine why the agent failed, which failure category applies from the taxonomy below, and exactly which step index the failure occurred at.

You are also provided a list of violations that have been generated through the trajectory through various invariants. Use these to help you identify the root cause category, failure step and agent.
Static invariants have been generated through the domain policy and system prompt. Each static invariant is associated with a tool call to ensure it adheres to the domain policy.
Dynamic invariants have been generated to cover computation checks, data accuracy, argument validity, and tool output consistency.
Each invariant returns a boolean, and if it returns false, it indicates a violation. Note that some violations may be false positives and not all violations may be relevant to the root cause failure.

Here are the list of violations noted by static and dynamic invariants:

{invariants_violation_context}

FAILURE TAXONOMY CATEGORIES:
{taxonomy_block}

ROOT-CAUSE DETECTION ALGORITHM:

Step 1 — Locate the first failure: Scan the trajectory step-by-step from the start and record the first failure.
Step 2 — Check if that failure was resolved: Look ahead in the trajectory for evidence that the error was resolved. If yes → Resolved; if no such evidence → Not resolved.
Step 3 — Decide and continue:
If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
If Not resolved: treat this step as the root-cause failure for the run and assign the taxonomy at this step.

Output a JSON object in the following format:
{{
    "reason_for_failure": <string>,
    "failure_case": <int 1-10>,
    "reason_for_index": <string>,
    "index": <int>
}}
""".strip()

_TMPL_NO_CONTEXT = """
GIVEN INPUT:
- a full trajectory of an agent's interaction with a user (step-indexed)
- the ground-truth tool-call/action sequence the agent should have made
- optional: expected responses/outputs for some steps

YOUR TASK is to determine why the agent failed, which failure category applies from the taxonomy below, and exactly which step index the failure occurred at.

FAILURE TAXONOMY CATEGORIES:
{taxonomy_block}

ROOT-CAUSE DETECTION ALGORITHM:

Step 1 — Locate the first failure: Scan the trajectory step-by-step from the start and record the first failure.
Step 2 — Check if that failure was resolved: Look ahead in the trajectory for evidence that the error was resolved. If yes → Resolved; if no such evidence → Not resolved.
Step 3 — Decide and continue:
If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
If Not resolved: treat this step as the root-cause failure for the run and assign the taxonomy at this step.

Output a JSON object in the following format:
{{
    "taxonomy_checklist_reasoning": <string>,
    "reason_for_failure": <string>,
    "failure_case": <int 1-10>,
    "reason_for_index": <string>,
    "index": <int>
}}
""".strip()

_TMPL_WITH_CONTEXT = """
GIVEN INPUT:
- a full trajectory of an agent's interaction with a user (step-indexed)
- the ground-truth tool-call/action sequence the agent should have made
- optional: expected responses/outputs for some steps

YOUR TASK is to determine why the agent failed, which failure category applies from the taxonomy below, and exactly which step index the failure occurred at.

FAILURE TAXONOMY CATEGORIES:
{taxonomy_block}

ROOT-CAUSE DETECTION ALGORITHM:

Step 1 — Locate the first failure: Scan the trajectory step-by-step from the start and record the first failure.
Step 2 — Check if that failure was resolved: Look ahead in the trajectory for evidence that the error was resolved. If yes → Resolved; if no such evidence → Not resolved.
Step 3 — Decide and continue:
If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
If Not resolved: treat this step as the root-cause failure for the run and assign the taxonomy at this step.

You are also provided a list of violations that have been generated through the trajectory through various invariants. Use these to help you identify the root cause category, failure step and agent.
Static invariants have been generated through the domain policy and system prompt. Each static invariant is associated with a tool call to ensure it adheres to the domain policy.
Dynamic invariants have been generated to cover computation checks, data accuracy, argument validity, and tool output consistency.
Each invariant returns a boolean, and if it returns false, it indicates a violation. Note that some violations may be false positives and not all violations may be relevant to the root cause failure.

Here are the list of violations noted by static and dynamic invariants:

{invariants_violation_context}

Output a JSON object in the following format:
{{
    "taxonomy_checklist_reasoning": <string>,
    "reason_for_failure": <string>,
    "failure_case": <int 1-10>,
    "reason_for_index": <string>,
    "index": <int>
}}
""".strip()

_TMPL_FAILURE = """
GIVEN INPUT:
- a full trajectory of an agent's interaction with a user (step-indexed)
- the ground-truth tool-call/action sequence the agent should have made
- the exact step index at which the failure occurs

YOUR TASK is to determine why the agent failed and which failure category applies from the taxonomy below.

FAILURE TAXONOMY CATEGORIES:
{taxonomy_block}

You are also provided a list of violations that have been generated through the trajectory through various invariants. Use these to help you identify the root cause category, failure step and agent.
Static invariants have been generated through the domain policy and system prompt. Each static invariant is associated with a tool call to ensure it adheres to the domain policy.
Dynamic invariants have been generated to cover computation checks, data accuracy, argument validity, and tool output consistency.
Each invariant returns a boolean, and if it returns false, it indicates a violation. Note that some violations may be false positives and not all violations may be relevant to the root cause failure.

Here are the list of violations noted by static and dynamic invariants:

{invariants_violation_context}

Output a JSON object in the following format:
{{
    "reason_for_failure": <string>,
    "failure_case": <int 1-10>
}}
""".strip()


def get_system_prompt(invariants_violation_context=None, is_failure_prompt=False):
    """Build the system prompt by selecting the right template and formatting it."""
    taxonomy_block = build_taxonomy_text(PROMPT_MODE)
    inv = invariants_violation_context or ""

    if is_failure_prompt:
        template = _TMPL_FAILURE
    elif invariants_violation_context and EXECUTION_MODE == "violations-before":
        template = _TMPL_VIOLATIONS_BEFORE
    elif invariants_violation_context:
        template = _TMPL_WITH_CONTEXT
    else:
        template = _TMPL_NO_CONTEXT

    return template.format(taxonomy_block=taxonomy_block, invariants_violation_context=inv)


# --- Synth Normalizer (from Updated) ---

_SYNTH_NORMALIZER_FN = None
_SYNTH_NORMALIZER_CODE = None
FN_NAME = "normalize_synth_events"

def _extract_python(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else raw).strip()

def _safe_exec_env():
    # Keep exec reasonably safe: no import, no file I/O, no print
    safe_builtins = {
        "str": str, "int": int, "float": float, "bool": bool,
        "dict": dict, "list": list, "tuple": tuple, "set": set,
        "len": len, "range": range, "enumerate": enumerate,
        "isinstance": isinstance,
        "min": min, "max": max, "sum": sum, "sorted": sorted,
        "ValueError": ValueError, "Exception": Exception,
    }
    return {"__builtins__": safe_builtins, "validate_ir": validate_ir}

def _build_synth_prompt(sample, last_error: str | None) -> str:
    err = f"\n\nPREVIOUS ERROR (fix this):\n{last_error}\n" if last_error else ""
    return f"""
Write EXACTLY ONE Python function and nothing else (no markdown, no explanation).

def {FN_NAME}(trajectories):
    ...
    return ir_list

Input:
Sample input (first 2 trajectories):
{repr(sample)}

Output: a list of IR dicts, one per trajectory, each MUST satisfy validate_ir(ir):
  {{
    "trajectory_id": str,
    "instruction": str,
    "steps": [
      {{
        "index": int,                 # MUST be 0-based consecutive: 0..n-1
        "substeps": [
          {{"sub_index": 1, "role": str, "content": str}}
        ]
      }},
      ...
    ]
  }}

Rules (MUST FOLLOW):
- Return a LIST of IR dicts (one per trajectory).
- Each *event* becomes one *step* with exactly ONE substep (sub_index = 1).
- Ignore events where event.get("type") == "LLMCallEvent".
- role = event.get("role") or event.get("source") or event.get("type") or "unknown"
- content:
    c = event.get("content")
    if c is None: c = event.get("message")
    if c is None: c = ""
    else: c = str(c)
- Always include "instruction" in the IR dict (use empty string if missing).
- For each trajectory IR dict, call validate_ir(ir) BEFORE appending to ir_list.
- Do NOT import. Do NOT print. Use only basic Python.


{err}
""".strip()

def get_or_build_synth_normalizer(judge, raw_trajectories, max_attempts: int = 3):
    print("### DEBUG_SYNTH: get_or_build_synth_normalizer called")
    global _SYNTH_NORMALIZER_FN, _SYNTH_NORMALIZER_CODE
    if _SYNTH_NORMALIZER_FN is not None:
        return _SYNTH_NORMALIZER_FN
    
    print("### DEBUG_SYNTH: synthesizing new normalizer function via LLM")
    sample = raw_trajectories[:2] if isinstance(raw_trajectories, list) else []
    last_error = None

    for attempt in range(1, max_attempts + 1):
        prompt = _build_synth_prompt(sample, last_error)
        
        resp = judge.get_llm_response(
            messages=[
                {"role": "system", "content": "Output ONLY raw python code. No markdown. No explanation."},
                {"role": "user", "content": prompt},
            ])
        raw_code = (resp.choices[0].message.content or "").strip()
        code = _extract_python(raw_code)

        try:
            g_env = _safe_exec_env()
            l = {}
            exec(code, g_env, l)
            fn = l.get(FN_NAME) or g_env.get(FN_NAME)
            if not callable(fn):
                raise ValueError(f"LLM did not define callable {FN_NAME}()")

            ir_list = fn(sample)
            if not isinstance(ir_list, list):
                raise ValueError(f"{FN_NAME} must return a list")
            for ir in ir_list:
                validate_ir(ir)

            _SYNTH_NORMALIZER_FN = fn
            _SYNTH_NORMALIZER_CODE = code
            print("### DEBUG_SYNTH: ✅ normalizer installed in-memory and validated on sample")
            return fn

        except Exception:
            last_error = traceback.format_exc()[-2000:]
            print(f"### DEBUG_SYNTH: ❌ attempt {attempt} failed")

    raise RuntimeError(
        f"Failed to synthesize valid {FN_NAME} after {max_attempts} attempts."
    )

# --- Trajectory Loading (from Updated) ---

def iter_load_trajectories_from_dir(input_dir: str, pattern: str = "*.jsonl"):
    from pathlib import Path
    input_dir = os.path.abspath(input_dir)
    paths = sorted(Path(input_dir).rglob(pattern))
    print(f"### DEBUG: iter_load_trajectories_from_dir dir={input_dir} patterns={pattern} files={len(paths)}")
    for fp in paths:
        try:
            yield str(fp), load_trajectories(str(fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")

def _trajectory_len(traj: Any) -> int:
    if not isinstance(traj, list):
        if isinstance(traj, dict) and "steps" in traj:
            traj = traj["steps"]
        else:
            return 0
    total = 0
    for step in traj:
        if isinstance(step, dict) and "substeps" in step:
            total += len(step["substeps"])
        else:
            total += 1
    return total


def _normalize_by_domain(domain: str, raw):
    domain = (domain or "").strip().lower()
    # If the domain is already registered, use its converter
    if domain in DOMAIN_REGISTRY:
        converter = get_domain_config(domain).ir_converter
        return converter(raw)
    # Unknown domain: use LLM-based IR converter and register it
    print(f"[IR] Domain '{domain}' not in registry — using llm_ir() and registering")
    result = llm_ir(raw)
    register_domain(domain, ir_converter_name="llm_ir")
    print(f"[IR] Registered domain '{domain}' in domain registry with llm_ir converter")
    return result


# --- Judge Class (Merged Logic) ---

def get_llm_judge_class():
    if ENDPOINT_USED == "copilot":
        from agentrx.llm_clients.copilot_cli import LLMAgent as LLMAgentCopilot
        base_class = LLMAgentCopilot
    elif ENDPOINT_USED == "azure":
        base_class = LLMAgentAzure
    else:
        base_class = LLMAgentTrapi
    
    class LLMJudge(base_class):
        def _parse_json_response(self, response, context_name="LLM", max_retries=2):
            """
            Parse JSON from LLM response with retry logic for empty/invalid responses.

            Args:
                response: The LLM response object
                context_name: Name for error messages (e.g., "Step 1 Index", "Step 2 Category")
                max_retries: Number of retries for failed parsing
            Returns:
                Parsed JSON dict

            Raises:
                RuntimeError if parsing fails after retries
            """
            content = (response.choices[0].message.content or "").strip()

            if not content:
                raise ValueError(f"{context_name}: Empty response from LLM")

            # Try to extract JSON from markdown code blocks if present
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1).strip()
            # Try parsing the JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # Log the problematic content for debugging
                print(f"[WARN] {context_name} - Failed to parse JSON. Content preview: {content[:200]}...")
                raise ValueError(f"{context_name}: Invalid JSON - {e}")

        def judge_response(self, task_id, trajectory, ground_truth=None, outputs=None, invariants_violation_context=None):

            user_message = f'Conversation: {trajectory}'
            max_retries = 3

            if EXECUTION_MODE == "stepbystep":
                # Phase 1: Get Index with retry logic
                prompt_index = STEP_INDEX_PROMPT_TEMPLATE
                
                failure_index = None
                last_error = None
                for attempt in range(1, max_retries + 1):
                    try:
                        resp_index = self.get_llm_response(
                            messages=[
                                {"role": "system", "content": prompt_index},
                                {"role": "user", "content": user_message},
                            ]
                        )
                        completion_index = self._parse_json_response(resp_index, "Step 1 Index")
                        failure_index = int(completion_index["index"])
                        break  # Success
                    except Exception as e:
                        last_error = e
                        print(f"[WARN] Step 1 Index Parse Error (attempt {attempt}/{max_retries}): {e}")
                        if attempt < max_retries:
                            import time
                            time.sleep(1)  # Brief pause before retry

                if failure_index is None:
                    raise RuntimeError(f"Step 1 Index Parse Error after {max_retries} attempts: {last_error}")

                # Phase 2: Get Failure Category with retry logic
                failure_system_prompt = get_system_prompt(
                    invariants_violation_context=invariants_violation_context, 
                    is_failure_prompt=True
                )
                
                user_message_b = f"{user_message}\n\nFAILURE STEP INDEX: {failure_index}\n\n"
                
                completion = None
                last_error = None
                for attempt in range(1, max_retries + 1):
                    try:
                        resp_cat = self.get_llm_response(
                            messages=[
                                {"role": "system", "content": failure_system_prompt},
                                {"role": "user", "content": user_message_b}
                            ]
                        )
                        completion = self._parse_json_response(resp_cat, "Step 2 Category")
                        break  # Success
                    except Exception as e:
                        last_error = e
                        print(f"[WARN] Step 2 Category Parse Error (attempt {attempt}/{max_retries}): {e}")
                        if attempt < max_retries:
                            import time
                            time.sleep(1)

                if completion is None:
                    raise RuntimeError(f"Step 2 Category Parse Error after {max_retries} attempts: {last_error}")
                
                return Failure(
                    task_id=task_id,
                    failure_case=int(completion["failure_case"]),
                    description=completion.get("reason_for_failure", ""),
                    step_number=failure_index
                )

            else: 
                # Single pass (violations-after or violations-before strategy usually implies single pass with context)
                # We use the unified refactored prompt builder
                
                system_prompt = get_system_prompt(
                    invariants_violation_context=invariants_violation_context,
                    is_failure_prompt=False
                )
                
                completion = None
                last_error = None
                for attempt in range(1, max_retries + 1):
                    try:
                        response = self.get_llm_response(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message},
                            ]
                        )
                        if DEBUG_PROMPTS:
                            try:
                                print("System Prompt:", system_prompt.encode('ascii', 'replace').decode())
                                print("User Message:", user_message[:500].encode('ascii', 'replace').decode())
                            except Exception:
                                pass
                        completion = self._parse_json_response(response, "Single Pass")
                        break  # Success
                    except Exception as e:
                        last_error = e
                        print(f"[WARN] Single Pass Parse Error (attempt {attempt}/{max_retries}): {e}")
                        if attempt < max_retries:
                            import time
                            time.sleep(1)
                
                if completion is None:
                    raise RuntimeError(f"Single Pass Parse Error after {max_retries} attempts: {last_error}")
                
                return Failure(
                    task_id=task_id,
                    failure_case=completion["failure_case"],
                    description=completion.get("reason_for_failure", ""),
                    step_number=completion.get("index", 0),
                    checklist_reasoning=completion.get("taxonomy_checklist_reasoning")
                )

    return LLMJudge

# --- Main Execution Flow (from Updated) ---

def load_invariant_violation_context(task_id):
    """
    Load invariant violation context for a specific task.

    Resolution order for the base directory:
        1. --violation_context_dir  (CLI override → VIOLATION_CONTEXT_DIR global)
        2. Default: <script_dir>/pipeline/out/<task_id>/

    The filename is ``violations_{DOMAIN}.json`` so it adapts to whatever
    domain is being evaluated (tau, magentic, flash, …).
    """
    global VIOLATION_CONTEXT_DIR
    
    domain_tag = (DOMAIN or "unknown").strip().lower()
    
    # Use global VIOLATION_CONTEXT_DIR if set, otherwise use default
    if VIOLATION_CONTEXT_DIR is None:
        context_dir = os.path.join(os.path.dirname(__file__), "pipeline", "out")
    else:
        context_dir = VIOLATION_CONTEXT_DIR
    
    file_path = os.path.join(context_dir, f"{task_id}", f"violations_{domain_tag}.json")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            print(f"[CONTEXT] [OK] Loaded violation context for task {task_id}: {file_path}")
            return context_data
    except FileNotFoundError:
        print(f"[CONTEXT] [MISS] No violation context file found for task {task_id}: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[CONTEXT] [ERR] Invalid JSON in context file for task {task_id}: {file_path} - {e}")
        return None
    except Exception as e:
        print(f"[CONTEXT] [ERR] Error reading context file for task {task_id}: {file_path} - {e}")
        return None

def convert_to_failure_case(case_input):
    """
    Robustly convert input (int or string) to FailureCase Enum.
    Supports varied string formats found in different GT files.
    Handles 0 (no error predicted) and invalid values gracefully.
    """
    if isinstance(case_input, int):
        try:
            return FailureCase(case_input)
        except ValueError:
            print(f"[WARN] Invalid failure_case int value {case_input}, treating as INCONCLUSIVE")
            return FailureCase.INCONCLUSIVE
    
    if isinstance(case_input, FailureCase):
        return case_input

    case_str = str(case_input).strip().lower()
    
    # Direct integer string check
    if case_str.isdigit() or (case_str.startswith('-') and case_str[1:].isdigit()):
        try:
            return FailureCase(int(case_str))
        except ValueError:
            print(f"[WARN] Invalid failure_case string value '{case_str}', treating as INCONCLUSIVE")
            return FailureCase.INCONCLUSIVE
        
    # Heuristic matching
    if "instruction" in case_str and "adherence" in case_str:
        return FailureCase.INSTRUCTION_OR_PLAN_ADHERENCE_FAILURE
    elif "invention" in case_str and "information" in case_str:
        return FailureCase.INVENTION_OF_NEW_INFORMATION
    elif "invalid" in case_str and "invocation" in case_str:
        return FailureCase.INVALID_INVOCATION
    elif "misinterpretation" in case_str or "handoff" in case_str:
        return FailureCase.MISINTERPRETATION_OF_TOOL_OUTPUT
    elif "intent" in case_str and "misalignment" in case_str:
        return FailureCase.INTENT_PLAN_MISALIGNMENT
    elif "underspecified" in case_str:
        return FailureCase.UNDERSPECIFIED_USER_INTENT
    elif "unsupported" in case_str or ("intent" in case_str and "not supported" in case_str):
        return FailureCase.INTENT_NOT_SUPPORTED
    elif "guardrail" in case_str:
        return FailureCase.GUARDRAILS_TRIGGERED
    elif "system" in case_str and "failure" in case_str:
        return FailureCase.SYSTEM_FAILURE
    else:
        # Fallback
        return FailureCase.INCONCLUSIVE

def judge_trajectories(log_file, num_runs=1, ground_truth_task_ids=None):
    """
    Judge trajectories from an IR-format file (trajectory_ir.json).
    
    Args:
        log_file: Path to IR trajectory file (produced by Stage 1)
        num_runs: Number of evaluation runs
        ground_truth_task_ids: Optional set of task IDs to filter trajectories. 
                              If provided, only trajectories with matching IDs are processed.
    """
    # Setup Judge
    LLMJudge = get_llm_judge_class()
    # Assuming params from global g or env
    # Note: Azure globals only has API_VERSION, MODEL_NAME, DEPLOYMENT (no MODEL_VERSION)
    # TRAPI has TRAPI_API_VERSION, TRAPI_MODEL_NAME, TRAPI_MODEL_VERSION, TRAPI_DEPLOYMENT_NAME
    if ENDPOINT_USED == "copilot":
        api_version, model_name, model_version, deployment_name = "", "copilot-cli", "", "copilot-cli"
    elif ENDPOINT_USED == "azure":
        api_version, model_name = g.API_VERSION, g.MODEL_NAME
        model_version, deployment_name = g.MODEL_NAME, g.DEPLOYMENT
    else:
        api_version, model_name = g.TRAPI_API_VERSION, g.TRAPI_MODEL_NAME
        model_version, deployment_name = g.TRAPI_MODEL_VERSION, g.TRAPI_DEPLOYMENT_NAME

    judge = LLMJudge(api_version=api_version, model_name=model_name, model_version=model_version, deployment_name=deployment_name)
    
    try:
        # Auto-detect: if the file is already IR, use it directly;
        # otherwise convert from raw using the domain converter.
        data = ensure_ir(log_file, DOMAIN)

        print(f"### DEBUG: loaded IR trajectories: count={len(data)}")
        #if domain_norm == "magentic":
        #    magentic_task_ids_set = set(g.FILTERED_TASKS_MAGENTIC)
        #    original_count = len(data)
        #    data = [traj for traj in data if str(traj.get('trajectory_id', '')) in magentic_task_ids_set]
        #    print(f"### DEBUG: filtered magentic trajectories to FILTERED_TASKS_MAGENTIC: {len(data)} of {original_count}")
        # Filter to only process trajectories that have ground truth
        if ground_truth_task_ids is not None:
            original_count = len(data)
            data = [traj for traj in data if str(traj.get('trajectory_id', '')) in ground_truth_task_ids]
            print(f"### DEBUG: filtered trajectories based on ground truth: {len(data)} of {original_count} (matched {len(ground_truth_task_ids)} ground truth IDs)")
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading {log_file}: {e}")
    
    responses = []
    trajectory_lengths = {}  # Store trajectory lengths for normalization
    for i in range(len(data)):
        # if data[i]['reward'] > 0:
        #    continue # only looking at failed trajectories
        task_id = data[i]['trajectory_id']
        trajectory = data[i]["steps"]
        # Calculate trajectory length (number of conversation turns)
        trajectory_length = _trajectory_len(trajectory)
        trajectory_lengths[task_id] = trajectory_length
        responses.append(Report(task_id, trajectory_length))

    # Runs
    for j in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {j+1}/{num_runs}")
        print(f"{'='*60}")
        for i, item in enumerate(data):
            # Simplification: we assume data aligned with responses by index as built above
            task_id = item['trajectory_id']
            # Find matching report
            report = responses[i] 
            
            print(f"Processing run {j + 1}, entry {i + 1}/{len(data)}, task_id: {task_id}")
            
            try:
                invariants = None
                if RUN_WITH_CONTEXT:
                    invariants = load_invariant_violation_context(task_id)
                    print(f"  Loaded invariant context for task {task_id}")

                res = judge.judge_response(task_id, item["steps"], invariants_violation_context=invariants)
                report.add_failure(res)
                
                # Capture telemetry from the judge's last LLM call
                if hasattr(judge, 'last_call_telemetry') and judge.last_call_telemetry is not None:
                    report.llm_call_telemetry = judge.last_call_telemetry
                
                print(f"  Result: {res.failure_case.name} @ Step {res.step_number}")
                print(f"  Description: {res.description[:100]}..." if len(res.description) > 100 else f"  Description: {res.description}")
                
            except Exception as e:
                print(f"  ERROR judging task {task_id}: {e}")
                import traceback
                traceback.print_exc()

    return responses

def filter_task_ids(responses, ground_truth_failures):
    task_ids = [str(f.task_id) for f in ground_truth_failures]
    print(f"Task IDs which have ground truth: {task_ids}")
    filtered = [r for r in responses if str(r.task_id) in task_ids]
    print(f"Filtered {len(filtered)} responses from {len(responses)} total")
    return filtered

def sort_responses_by_task_id(responses):
    """Sort evaluation responses by task ID for consistent ordering."""
    return sorted(responses, key=lambda x: x.task_id)

def validate_responses(responses, ground_truth_failures):
    """Validate that responses and ground truth have matching task IDs in same order."""
    print(f"Validating {len(responses)} responses against {len(ground_truth_failures)} ground truth entries...")
    for i in range(len(responses)):
        print(f"  Response {i}: {responses[i].task_id} vs GT: {ground_truth_failures[i].task_id}")
        if str(responses[i].task_id) != str(ground_truth_failures[i].task_id):
            print(f"    WARNING: Mismatch: {responses[i].task_id} vs {ground_truth_failures[i].task_id}")

def load_failures_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    failures = []
    for item in items:
        # Support both old and new gt formats if possible
        
        # Check for new format (root_cause object)
        if 'root_cause' in item and isinstance(item['root_cause'], dict):
            try:
                task_id = item.get('trajectory_id') or item.get('task_id')
                rc_id = item['root_cause'].get('failure_id')
                
                # Try to find specific failure in list if available, else rely on root_cause
                if "failures" in item and isinstance(item["failures"], list) and rc_id is not None:
                     rc_failure = next((f for f in item["failures"] if f.get("failure_id") == rc_id), None)
                     if rc_failure:
                         failures.append(Failure(
                             task_id=task_id,
                             failure_case=convert_to_failure_case(rc_failure.get('failure_category') or rc_failure.get('failure_case')),
                             description=item['root_cause'].get('reason_for_root_cause', ""),
                             step_number=int(rc_failure.get('step_number', 0))
                         ))
                         continue

                # If no failures list or match, try basic root_cause fields
                failures.append(Failure(
                    task_id=task_id,
                    failure_case=convert_to_failure_case(item['root_cause'].get('failure_category') or item['root_cause'].get('failure_case') or "inconclusive"),
                    description=item['root_cause'].get('reason_for_root_cause', ""),
                    step_number=int(item['root_cause'].get('index', 0))
                ))
            except Exception as e:
                print(f"Error parsing item {item.get('trajectory_id')}: {e}")
            continue

        # Fallback to Refactored / Flat format
        fc_val = item.get("failure_case") or item.get("failure_class") or item.get("failure_category")
        
        if fc_val:
            failures.append(Failure(
                task_id=item.get("task_id", item.get("trajectory_id")), 
                failure_case=convert_to_failure_case(fc_val), 
                description=item.get("reason_for_failure", ""), 
                step_number=int(item.get("index", item.get("failure_step", 0)))
            ))
    
    # Debug: print loaded failures
    print(f"Loaded {len(failures)} failures from {file_path}")
    for f in failures:
        print(f"  Task {f.task_id}: {f.failure_case.name} @ Step {f.step_number}")
    
    return failures

def analysis(data, output_file_path=None, model_name=None, api_version=None):
    """
    Perform high-level analysis of evaluation results and print summary statistics.
    Computes accuracy, step distance metrics, and writes summary to the output file.
    """
    correct_cases = 0
    incorrect_cases = 0
    correct_distance = 0
    incorrect_distance = 0
    correct_step_predictions = 0
    incorrect_step_predictions = 0
    correct_normalized_distance = 0
    incorrect_normalized_distance = 0
    
    # Tolerance-based step prediction accuracy counters
    step_within_tolerance = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for task in data:
        # Check safely for GT data existence in the report
        if 'gt_failure_case' not in task:
            continue
            
        # Parse failure case enum string if needed (e.g. "FailureCase.INVALID_INVOCATION" -> just name)
        # But Report.to_dict stores .value (int) or .name? 
        # Refactored Report.to_dict stored .value (int)
        # Updated Report.to_dict stored str(failure.failure_case) which is "FailureCase.NAME" usually
        
        # Let's rely on what compute_stats did.
        # most_common is a str(int_value) in compute_stats
        
        # Calculate raw distance
        # step_mean might be float
        step_mean = task.get('step_mean', 0)
        gt_step = task.get('gt_step_number', 0)
        
        distance = abs(step_mean - gt_step)
        trajectory_length = task.get('trajectory_length', 1)
        normalized_distance = distance / trajectory_length if trajectory_length > 0 else distance
        
        # Accuracy check
        # most_common_failure and gt_failure_case are strings of the int value in compute_stats
        if str(task.get('most_common_failure')) == str(task.get('gt_failure_case')):
            correct_cases += 1
            correct_distance += distance
            correct_normalized_distance += normalized_distance
        else:
            incorrect_cases += 1
            incorrect_distance += distance
            incorrect_normalized_distance += normalized_distance
        
        # Step accuracy (rounded)
        if round(step_mean) == gt_step:
            correct_step_predictions += 1
        else:
            incorrect_step_predictions += 1
        
        # Tolerance-based step accuracy
        step_error = abs(round(step_mean) - gt_step)
        for tolerance in [1, 2, 3, 4, 5]:
            if step_error <= tolerance:
                step_within_tolerance[tolerance] += 1
    
    total_cases = correct_cases + incorrect_cases
    if total_cases == 0:
        print("No cases with ground truth comparison found.")
        # Still save results to disk even without GT comparisons
        if output_file_path:
            output_data = {
                "summary": {
                    "model_name": model_name,
                    "api_version": api_version,
                    "note": "No ground truth comparison available",
                },
                "detailed_results": data,
            }
            with open(output_file_path, 'w') as f:
                json.dump(output_data, f, indent=4)
        return

    avg_correct_distance = correct_distance / correct_cases if correct_cases > 0 else 0
    avg_incorrect_distance = incorrect_distance / incorrect_cases if incorrect_cases > 0 else 0
    overall_avg_distance = (correct_distance + incorrect_distance) / total_cases if total_cases > 0 else 0
    step_number_accuracy = correct_step_predictions / total_cases if total_cases > 0 else 0
    
    normalized_avg_correct_distance = correct_normalized_distance / correct_cases if correct_cases > 0 else 0
    normalized_avg_incorrect_distance = incorrect_normalized_distance / incorrect_cases if incorrect_cases > 0 else 0
    normalized_overall_avg_distance = (correct_normalized_distance + incorrect_normalized_distance) / total_cases if total_cases > 0 else 0
    
    print(f"Correct cases: {correct_cases}, Incorrect cases: {incorrect_cases}")
    print(f"Average distance for correct cases: {avg_correct_distance:.4f}, Average distance for incorrect cases: {avg_incorrect_distance:.4f}, Overall average distance: {overall_avg_distance:.4f}")
    print(f"Normalized average distance for correct cases: {normalized_avg_correct_distance:.4f}, Normalized average distance for incorrect cases: {normalized_avg_incorrect_distance:.4f}, Normalized overall average distance: {normalized_overall_avg_distance:.4f}")
    print(f"Correct step predictions: {correct_step_predictions}, Incorrect step predictions: {incorrect_step_predictions}, Step number accuracy: {step_number_accuracy:.2%}")
    
    # Calculate and print tolerance-based step accuracies
    step_accuracy_within_tolerance = {}
    for tolerance in [1, 2, 3, 4, 5]:
        accuracy = step_within_tolerance[tolerance] / total_cases if total_cases > 0 else 0
        step_accuracy_within_tolerance[tolerance] = accuracy
        print(f"Step accuracy within ±{tolerance}: {accuracy:.2%} ({step_within_tolerance[tolerance]}/{total_cases})")
    
    if output_file_path:
        try:
            # We assume output_file_path already has the list of reports written to it 
            # OR we rewrite it completely.
            # In run_single_iteration we write the list first.
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as f:
                    existing_data = json.load(f)
                # Handle either format:
                #   - list of per-task dicts (legacy)
                #   - {"summary": ..., "detailed_results": [...]} (current)
                if isinstance(existing_data, dict) and "detailed_results" in existing_data:
                    existing_data = existing_data["detailed_results"]
                # Fallback: if format is unexpected, prefer the freshly-computed `data`
                if not isinstance(existing_data, list):
                    existing_data = data
            else:
                existing_data = data # Should match

            # Calculate aggregate token and timing metrics
            total_prompt_tokens = 0
            total_output_tokens = 0
            total_execution_time_sec = 0.0
            for result in existing_data:
                telemetry = result.get("llm_call_telemetry")
                if telemetry:
                    tokens = telemetry.get("tokens", {})
                    total_prompt_tokens += tokens.get("prompt_tokens", 0)
                    total_output_tokens += tokens.get("output_tokens", 0)
                    time_info = telemetry.get("time", {})
                    total_execution_time_sec += time_info.get("execution_time_sec", 0.0)

            summary = {
                "model_name": model_name,
                "api_version": api_version,
                "Correct cases": correct_cases,
                "Incorrect cases": incorrect_cases,
                "Average distance for correct cases": avg_correct_distance,
                "Average distance for incorrect cases": avg_incorrect_distance,
                "Overall average distance": overall_avg_distance,
                "Normalized average distance for correct cases": normalized_avg_correct_distance,
                "Normalized average distance for incorrect cases": normalized_avg_incorrect_distance,
                "Normalized overall average distance": normalized_overall_avg_distance,
                "Correct step number predictions": correct_step_predictions,
                "Incorrect step number predictions": incorrect_step_predictions,
                "Step number accuracy": step_number_accuracy,
                "Step accuracy within +-1": step_accuracy_within_tolerance.get(1, 0),
                "Step accuracy within +-2": step_accuracy_within_tolerance.get(2, 0),
                "Step accuracy within +-3": step_accuracy_within_tolerance.get(3, 0),
                "Step accuracy within +-4": step_accuracy_within_tolerance.get(4, 0),
                "Step accuracy within +-5": step_accuracy_within_tolerance.get(5, 0),
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_prompt_tokens + total_output_tokens,
                "total_execution_time_sec": round(total_execution_time_sec, 4)
            }
            
            print(f"\nToken Usage: prompt={total_prompt_tokens}, output={total_output_tokens}, total={total_prompt_tokens + total_output_tokens}")
            print(f"Total Execution Time: {total_execution_time_sec:.2f} seconds")
            
            output_data = {
                "summary": summary,
                "detailed_results": existing_data
            }
            
            with open(output_file_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
        except Exception as e:
            print(f"Error writing analysis summary: {e}")

def run_single_iteration(run_number, base_output_dir, ground_truth_failures, api_version, model_name, log_file):
    print(f"\n{'='*80}")
    print(f"STARTING RUN {run_number}")
    print(f"{'='*80}\n")
    
    # Extract ground truth task IDs for filtering trajectories
    ground_truth_task_ids = None
    if ground_truth_failures:
        ground_truth_task_ids = set(str(f.task_id) for f in ground_truth_failures)
        print(f"Filtering trajectories to {len(ground_truth_task_ids)} ground truth task IDs")
    
    responses = judge_trajectories(log_file, num_runs=1, ground_truth_task_ids=ground_truth_task_ids)
    print(f"Generated {len(responses)} evaluation responses")
    
    # Always filter and compute stats if GT is available (USE_GROUND_TRUTH only controls prompt content)
    if ground_truth_failures:
        filtered = filter_task_ids(responses, ground_truth_failures)
        # Compute stats against GT
        gt_map = {str(f.task_id): f for f in ground_truth_failures}
        for r in filtered:
            if str(r.task_id) in gt_map:
                r.compute_stats(gt_map[str(r.task_id)])
        valid_responses = filtered
    else:
        print("No ground truth failures loaded - skipping accuracy comparison")
        valid_responses = responses

    # Save
    runs_dir = os.path.join(base_output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    json_path = os.path.join(runs_dir, f"run{run_number}.json")
    
    output_data = [r.to_dict() for r in valid_responses]
    
    # Perform analysis and save results (including summary)
    analysis(output_data, output_file_path=json_path, model_name=model_name, api_version=api_version)
    
    print(f"Results saved to {json_path}")
        
    # Analysis Plots
    print("\n" + "="*80)
    print(f"GENERATING FAILURE FREQUENCY VISUALIZATIONS FOR RUN {run_number}")
    print("="*80)

    try:
        analysis_dir = os.path.join(base_output_dir, "analysis", f"run{run_number}")
        os.makedirs(analysis_dir, exist_ok=True)
        print(f"Plots will be saved to: {analysis_dir}")
        
        pred_freq, gt_freq = load_and_analyze_json(json_path)
        
        print("Generating predicted failure frequency plot...")
        plot_predicted_frequency(pred_freq, output_file=os.path.join(analysis_dir, 'predicted.png'))
        
        if gt_freq:
            print("Generating ground truth failure frequency plot...")
            plot_ground_truth_frequency(gt_freq, output_file=os.path.join(analysis_dir, 'gt.png'))
            print("Generating comparison plot...")
            plot_comparison(pred_freq, gt_freq, output_file=os.path.join(analysis_dir, 'comparison.png'))
        
        print(f"\nAll plots for run {run_number} saved to {analysis_dir}/")
        print("="*80)
    except Exception as e:
        print(f"Plotting failed: {e}")

def load_and_analyze_run_for_metrics(json_path):
    """
    Helper to reconstruct metrics from a saved run JSON if they weren't explicitly saved as a summary block.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # If the file has a "summary" key at the root, use it (Updated file style)
        if isinstance(data, dict) and 'summary' in data:
            return data['summary']
        
        # If it's a list of reports (Refactored style), we need to re-compute the summary 
        # But wait - analysis() prints them but doesn't return them easily in the merged code yet.
        # Let's see if we can extract from the list
        reports = data if isinstance(data, list) else data.get('reports', [])
        if not reports: return None
        
        # Basic reconstruction
        correct = 0
        incorrect = 0
        dist_correct = 0
        dist_incorrect = 0
        norm_dist_correct = 0
        norm_dist_incorrect = 0
        correct_step = 0
        incorrect_step = 0
        step_within_tolerance = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for r in reports:
            # Check if this report has ground truth comparison data
            if 'gt_failure_case' in r:
                # Comparison logic
                is_correct = str(r.get('failure_case')) == str(r.get('gt_failure_case'))
                step_err = abs(int(r.get('step_number', 0)) - int(r.get('gt_step_number', 0)))
                # For normalization, we need trajectory length. 
                # Ideally report has 'trajectory_length'
                traj_len = r.get('trajectory_length', 1)
                norm_err = step_err / traj_len if traj_len > 0 else step_err
                
                if is_correct:
                    correct += 1
                    dist_correct += step_err
                    norm_dist_correct += norm_err
                else:
                    incorrect += 1
                    dist_incorrect += step_err
                    norm_dist_incorrect += norm_err
                    
                if step_err == 0:
                    correct_step += 1
                else:
                    incorrect_step += 1
                
                # Tolerance-based step accuracy
                for tolerance in [1, 2, 3, 4, 5]:
                    if step_err <= tolerance:
                        step_within_tolerance[tolerance] += 1

        total = correct + incorrect
        if total == 0: return None
        
        # Calculate token and time metrics from detailed results
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_execution_time_sec = 0.0
        for r in reports:
            telemetry = r.get("llm_call_telemetry")
            if telemetry:
                tokens = telemetry.get("tokens", {})
                total_prompt_tokens += tokens.get("prompt_tokens", 0)
                total_output_tokens += tokens.get("output_tokens", 0)
                time_info = telemetry.get("time", {})
                total_execution_time_sec += time_info.get("execution_time_sec", 0.0)
        
        return {
            'Correct cases': correct,
            'Incorrect cases': incorrect,
            'Average distance for correct cases': dist_correct / correct if correct else 0,
            'Average distance for incorrect cases': dist_incorrect / incorrect if incorrect else 0,
            'Overall average distance': (dist_correct + dist_incorrect) / total,
            'Normalized average distance for correct cases': norm_dist_correct / correct if correct else 0,
            'Normalized average distance for incorrect cases': norm_dist_incorrect / incorrect if incorrect else 0,
            'Normalized overall average distance': (norm_dist_correct + norm_dist_incorrect) / total,
            'Correct step number predictions': correct_step,
            'Incorrect step number predictions': incorrect_step,
            'Step number accuracy': correct_step / total,
            'Step accuracy within +-1': step_within_tolerance[1] / total,
            'Step accuracy within +-2': step_within_tolerance[2] / total,
            'Step accuracy within +-3': step_within_tolerance[3] / total,
            'Step accuracy within +-4': step_within_tolerance[4] / total,
            'Step accuracy within +-5': step_within_tolerance[5] / total,
            'total_prompt_tokens': total_prompt_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_prompt_tokens + total_output_tokens,
            'total_execution_time_sec': round(total_execution_time_sec, 4)
        }
            
    except Exception as e:
        print(f"Error extracting metrics from {json_path}: {e}")
        return None

def create_aggregate_summary(base_output_dir, num_iterations):
    """
    Create an aggregate summary JSON file from all run results.
    Aggregates statistics across all runs and computes stability metrics.
    """
    print("\n" + "="*80)
    print("GENERATING AGGREGATE SUMMARY")
    print("="*80)
    
    runs_dir = os.path.join(base_output_dir, "runs")
    analysis_base_dir = os.path.join(base_output_dir, "analysis")
    os.makedirs(analysis_base_dir, exist_ok=True)
    
    # Collect summaries from all runs
    run_summaries = []
    
    for i in range(1, num_iterations + 1):
        run_file = os.path.join(runs_dir, f"run{i}.json")
        summary = load_and_analyze_run_for_metrics(run_file)
        if summary:
            run_summaries.append(summary)
        else:
            print(f"Warning: No valid summary found/computed for {run_file}")
    
    if not run_summaries:
        print("No summaries found. Skipping aggregate summary generation.")
        return
    
    # Extract metrics from each run
    accuracies = []
    correct_cases_list = []
    incorrect_cases_list = []
    avg_distance_correct_list = []
    avg_distance_incorrect_list = []
    overall_avg_distance_list = []
    normalized_avg_distance_correct_list = []
    normalized_avg_distance_incorrect_list = []
    normalized_overall_avg_distance_list = []
    correct_step_predictions_list = []
    incorrect_step_predictions_list = []
    step_number_accuracies = []
    
    # Tolerance-based step accuracy lists
    step_accuracy_tolerance_1_list = []
    step_accuracy_tolerance_2_list = []
    step_accuracy_tolerance_3_list = []
    step_accuracy_tolerance_4_list = []
    step_accuracy_tolerance_5_list = []
    
    # Cost metrics lists
    total_prompt_tokens_list = []
    total_output_tokens_list = []
    total_tokens_list = []
    total_execution_time_list = []
    
    for summary in run_summaries:
        # Calculate accuracy for each run
        total_cases = summary['Correct cases'] + summary['Incorrect cases']
        accuracy = summary['Correct cases'] / total_cases if total_cases > 0 else 0
        accuracies.append(accuracy)
        
        correct_cases_list.append(summary['Correct cases'])
        incorrect_cases_list.append(summary['Incorrect cases'])
        avg_distance_correct_list.append(summary['Average distance for correct cases'])
        avg_distance_incorrect_list.append(summary['Average distance for incorrect cases'])
        overall_avg_distance_list.append(summary['Overall average distance'])
        
        # Extract normalized distance metrics
        normalized_avg_distance_correct_list.append(summary.get('Normalized average distance for correct cases', 0))
        normalized_avg_distance_incorrect_list.append(summary.get('Normalized average distance for incorrect cases', 0))
        normalized_overall_avg_distance_list.append(summary.get('Normalized overall average distance', 0))
        
        # Extract step number prediction metrics
        correct_step_predictions_list.append(summary.get('Correct step number predictions', 0))
        incorrect_step_predictions_list.append(summary.get('Incorrect step number predictions', 0))
        step_number_accuracies.append(summary.get('Step number accuracy', 0))
        
        # Extract tolerance-based step accuracy metrics
        step_accuracy_tolerance_1_list.append(summary.get('Step accuracy within +-1', 0))
        step_accuracy_tolerance_2_list.append(summary.get('Step accuracy within +-2', 0))
        step_accuracy_tolerance_3_list.append(summary.get('Step accuracy within +-3', 0))
        step_accuracy_tolerance_4_list.append(summary.get('Step accuracy within +-4', 0))
        step_accuracy_tolerance_5_list.append(summary.get('Step accuracy within +-5', 0))
        
        # Extract cost metrics
        total_prompt_tokens_list.append(summary.get('total_prompt_tokens', 0))
        total_output_tokens_list.append(summary.get('total_output_tokens', 0))
        total_tokens_list.append(summary.get('total_tokens', 0))
        total_execution_time_list.append(summary.get('total_execution_time_sec', 0.0))
    
    # Compute aggregate statistics
    total_correct = sum(correct_cases_list)
    total_incorrect = sum(incorrect_cases_list)
    total_cases = total_correct + total_incorrect
    overall_accuracy = total_correct / total_cases if total_cases > 0 else 0
    
    # Compute overall averages (weighted by number of cases per run)
    if total_correct > 0:
        overall_avg_distance_correct = sum(c * d for c, d in zip(correct_cases_list, avg_distance_correct_list)) / total_correct
        overall_normalized_avg_distance_correct = sum(c * d for c, d in zip(correct_cases_list, normalized_avg_distance_correct_list)) / total_correct
    else:
        overall_avg_distance_correct = 0
        overall_normalized_avg_distance_correct = 0

    if total_incorrect > 0:
        overall_avg_distance_incorrect = sum(i * d for i, d in zip(incorrect_cases_list, avg_distance_incorrect_list)) / total_incorrect
        overall_normalized_avg_distance_incorrect = sum(i * d for i, d in zip(incorrect_cases_list, normalized_avg_distance_incorrect_list)) / total_incorrect
    else:
        overall_avg_distance_incorrect = 0
        overall_normalized_avg_distance_incorrect = 0

    if total_cases > 0:
        overall_avg_distance = ((total_correct * overall_avg_distance_correct) + (total_incorrect * overall_avg_distance_incorrect)) / total_cases
        overall_normalized_avg_distance = ((total_correct * overall_normalized_avg_distance_correct) + (total_incorrect * overall_normalized_avg_distance_incorrect)) / total_cases
    else:
        overall_avg_distance = 0
        overall_normalized_avg_distance = 0
    
    # Compute step number prediction aggregates
    total_correct_step_predictions = sum(correct_step_predictions_list)
    total_incorrect_step_predictions = sum(incorrect_step_predictions_list)
    overall_step_number_accuracy = total_correct_step_predictions / total_cases if total_cases > 0 else 0
    
    # Compute tolerance-based step accuracy aggregates (mean across runs)
    overall_step_accuracy_tolerance_1 = mean(step_accuracy_tolerance_1_list) if step_accuracy_tolerance_1_list else 0
    overall_step_accuracy_tolerance_2 = mean(step_accuracy_tolerance_2_list) if step_accuracy_tolerance_2_list else 0
    overall_step_accuracy_tolerance_3 = mean(step_accuracy_tolerance_3_list) if step_accuracy_tolerance_3_list else 0
    overall_step_accuracy_tolerance_4 = mean(step_accuracy_tolerance_4_list) if step_accuracy_tolerance_4_list else 0
    overall_step_accuracy_tolerance_5 = mean(step_accuracy_tolerance_5_list) if step_accuracy_tolerance_5_list else 0
    
    # Compute cost metrics aggregates
    grand_total_prompt_tokens = sum(total_prompt_tokens_list)
    grand_total_output_tokens = sum(total_output_tokens_list)
    grand_total_tokens = sum(total_tokens_list)
    grand_total_execution_time = sum(total_execution_time_list)
    
    # Compute stability metrics
    def compute_stats(values):
        """Helper to compute mean, std dev, variance, min, max"""
        if not values:
            return None
        n = len(values)
        return {
            "mean": mean(values),
            "std_dev": stdev(values) if n > 1 else 0.0,
            "variance": variance(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values)
        }
    
    # Build the aggregate summary structure
    aggregate_summary = {
        "num_runs": num_iterations,
        "individual_run_summaries": run_summaries,
        "aggregate_statistics": {
            "overall_correct_cases": total_correct,
            "overall_incorrect_cases": total_incorrect,
            "overall_total_cases": total_cases,
            "overall_accuracy": overall_accuracy,
            "overall_avg_distance_for_correct_cases": overall_avg_distance_correct,
            "overall_avg_distance_for_incorrect_cases": overall_avg_distance_incorrect,
            "overall_avg_distance": overall_avg_distance,
            "overall_normalized_avg_distance_for_correct_cases": overall_normalized_avg_distance_correct,
            "overall_normalized_avg_distance_for_incorrect_cases": overall_normalized_avg_distance_incorrect,
            "overall_normalized_avg_distance": overall_normalized_avg_distance,
            "overall_correct_step_number_predictions": total_correct_step_predictions,
            "overall_incorrect_step_number_predictions": total_incorrect_step_predictions,
            "overall_step_number_accuracy": overall_step_number_accuracy,
            "overall_step_accuracy_within_+-1": overall_step_accuracy_tolerance_1,
            "overall_step_accuracy_within_+-2": overall_step_accuracy_tolerance_2,
            "overall_step_accuracy_within_+-3": overall_step_accuracy_tolerance_3,
            "overall_step_accuracy_within_+-4": overall_step_accuracy_tolerance_4,
            "overall_step_accuracy_within_+-5": overall_step_accuracy_tolerance_5,
            "grand_total_prompt_tokens": grand_total_prompt_tokens,
            "grand_total_output_tokens": grand_total_output_tokens,
            "grand_total_tokens": grand_total_tokens,
            "grand_total_execution_time_sec": round(grand_total_execution_time, 4),
            "avg_prompt_tokens_per_run": grand_total_prompt_tokens / num_iterations if num_iterations > 0 else 0,
            "avg_output_tokens_per_run": grand_total_output_tokens / num_iterations if num_iterations > 0 else 0,
            "avg_tokens_per_run": grand_total_tokens / num_iterations if num_iterations > 0 else 0,
            "avg_execution_time_per_run_sec": round(grand_total_execution_time / num_iterations, 4) if num_iterations > 0 else 0
        },
        "stability_metrics": {
            "accuracy": compute_stats(accuracies),
            "correct_cases": compute_stats(correct_cases_list),
            "incorrect_cases": compute_stats(incorrect_cases_list),
            "avg_distance_for_correct_cases": compute_stats(avg_distance_correct_list),
            "avg_distance_for_incorrect_cases": compute_stats(avg_distance_incorrect_list),
            "overall_avg_distance": compute_stats(overall_avg_distance_list),
            "normalized_avg_distance_for_correct_cases": compute_stats(normalized_avg_distance_correct_list),
            "normalized_avg_distance_for_incorrect_cases": compute_stats(normalized_avg_distance_incorrect_list),
            "normalized_overall_avg_distance": compute_stats(normalized_overall_avg_distance_list),
            "correct_step_number_predictions": compute_stats(correct_step_predictions_list),
            "incorrect_step_number_predictions": compute_stats(incorrect_step_predictions_list),
            "step_number_accuracy": compute_stats(step_number_accuracies),
            "step_accuracy_within_+-1": compute_stats(step_accuracy_tolerance_1_list),
            "step_accuracy_within_+-2": compute_stats(step_accuracy_tolerance_2_list),
            "step_accuracy_within_+-3": compute_stats(step_accuracy_tolerance_3_list),
            "step_accuracy_within_+-4": compute_stats(step_accuracy_tolerance_4_list),
            "step_accuracy_within_+-5": compute_stats(step_accuracy_tolerance_5_list),
            "total_prompt_tokens": compute_stats(total_prompt_tokens_list),
            "total_output_tokens": compute_stats(total_output_tokens_list),
            "total_tokens": compute_stats(total_tokens_list),
            "total_execution_time_sec": compute_stats(total_execution_time_list)
        }
    }
    
    # Add coefficient of variation (CV)
    if aggregate_summary["stability_metrics"]["accuracy"] and aggregate_summary["stability_metrics"]["accuracy"]["mean"] > 0:
        cv_accuracy = (aggregate_summary["stability_metrics"]["accuracy"]["std_dev"] / 
                      aggregate_summary["stability_metrics"]["accuracy"]["mean"])
        aggregate_summary["stability_metrics"]["accuracy"]["coefficient_of_variation"] = cv_accuracy
    
    # Save to analysis folder
    summary_file = os.path.join(analysis_base_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(aggregate_summary, f, indent=4)
    
    print(f"Aggregate summary saved to: {summary_file}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    if aggregate_summary['stability_metrics']['accuracy']:
        print(f"Accuracy Std Dev: {aggregate_summary['stability_metrics']['accuracy']['std_dev']:.4f}")
    print(f"\nCost Metrics (across all {num_iterations} runs):")
    print(f"  Total Prompt Tokens: {grand_total_prompt_tokens}")
    print(f"  Total Output Tokens: {grand_total_output_tokens}")
    print(f"  Total Tokens: {grand_total_tokens}")
    print(f"  Total Execution Time: {grand_total_execution_time:.2f} seconds")
    print(f"  Avg Tokens per Run: {grand_total_tokens / num_iterations if num_iterations > 0 else 0:.0f}")
    print(f"  Avg Execution Time per Run: {grand_total_execution_time / num_iterations if num_iterations > 0 else 0:.2f} seconds")
    print("="*80)

def main():
    global RUN_WITH_CONTEXT, ENDPOINT_USED, PROMPT_MODE, EXECUTION_MODE, DOMAIN, USE_GROUND_TRUTH
    global EXAMPLES_DIR, VIOLATION_CONTEXT_DIR, FEW_SHOT_EXAMPLES
    
    parser = argparse.ArgumentParser(
        description='LLM Judge Merged - Evaluation System for AI Agent Failures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of evaluation runs per trajectory')
    
    # Flags from Refactored
    parser.add_argument('--mode', dest='prompt_mode', choices=['baseline', 'checklist', 'examples', 'combined'], default='combined',
                        help='Prompt Content Mode: baseline (standard descriptions), checklist (with decision checklists), examples (with few-shot examples), combined (checklist + examples)')
    parser.add_argument('--with_ground_truth', action='store_true',
                        help='Include ground truth tool sequence in LLM prompt (helps judge better). Note: GT labels for accuracy comparison are always loaded.')
    
    # Flags from Updated
    parser.add_argument('--exec_mode', choices=['violations-after', 'stepbystep', 'violations-before'], default='violations-after',
                        help='Execution Strategy: violations-after (single pass with context after taxonomy), stepbystep (two-phase: index then category), violations-before (context before taxonomy)')
    parser.add_argument('--domain', default='tau',
                        help='Trajectory domain for IR normalization. Built-in: tau, flash, magentic, synth. Any other value uses the LLM-based IR converter automatically.')
    parser.add_argument('--log_file', default='pipeline/agent_trajectory_tau_retail.json',
                        help='Path to trajectory log file or directory')
    parser.add_argument('--ground_truth_file', default='ground_truth_tau_retail.json',
                        help='Path to ground truth JSON file for accuracy evaluation')
    parser.add_argument('--endpoint', default=g.DEFAULT_ENDPOINT, choices=['copilot', 'azure', 'trapi'],
                        help='LLM API endpoint to use (default: copilot)')
    parser.add_argument('--with-context', action='store_true',
                        help='Include invariant violation context in prompts')
    
    # Directory path arguments
    parser.add_argument('--examples_dir', default=None,
                        help='Path to directory containing few-shot example JSON files (default: ./few_shot_examples)')
    parser.add_argument('--violation_context_dir', default=None,
                        help='Path to directory containing violation context files (default: ./pipeline/violation_results_20251222_045346/judge_context)')

    args = parser.parse_args()
    
    # Set directory paths from arguments (before loading examples)
    if args.examples_dir:
        EXAMPLES_DIR = args.examples_dir
    if args.violation_context_dir:
        VIOLATION_CONTEXT_DIR = args.violation_context_dir
    
    # Load few-shot examples now that EXAMPLES_DIR is set
    FEW_SHOT_EXAMPLES = load_few_shot_examples()
    
    PROMPT_MODE = args.prompt_mode
    EXECUTION_MODE = args.exec_mode
    RUN_WITH_CONTEXT = args.with_context
    DOMAIN = args.domain
    USE_GROUND_TRUTH = args.with_ground_truth
    
    # Set endpoint globally
    ENDPOINT_USED = args.endpoint
    if ENDPOINT_USED == "copilot":
        api_version = ""
        model_name = "copilot-cli"
    elif ENDPOINT_USED == "azure":
        api_version = g.API_VERSION
        model_name = g.MODEL_NAME
    else:
        api_version = g.TRAPI_API_VERSION
        model_name = g.TRAPI_MODEL_NAME
    
    print(f"Config: Prompt={PROMPT_MODE}, Exec={EXECUTION_MODE}, Domain={DOMAIN}, Context={RUN_WITH_CONTEXT}, GT_in_prompt={USE_GROUND_TRUTH}")
    print(f"Endpoint: {args.endpoint}, Model: {model_name}")
    
    # Always load GT dataset for comparison (this is different from USE_GROUND_TRUTH which controls prompt content)
    gt_failures = []
    try:
        gt_failures = load_failures_from_json(args.ground_truth_file)
        gt_failures = sort_responses_by_task_id(gt_failures)
        print(f"Loaded {len(gt_failures)} ground truth failures for accuracy comparison")
    except Exception as e:
        print(f"Warning: GT file '{args.ground_truth_file}' not found or failed to load: {e}")
        print("Accuracy metrics will not be available.")

    # Output Dir - 'prompt_gt' indicates whether GT is in prompt, GT labels always used for comparison
    current_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_name = f"output_{PROMPT_MODE}_prompt-gt-{USE_GROUND_TRUTH}_domain-{DOMAIN}_{current_time_stamp}"
    print(f"Results will be saved to directory: {base_dir_name}")
    out_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR, base_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(1, args.iterations + 1):
        run_single_iteration(i, out_dir, gt_failures, api_version, model_name, args.log_file)
        
    create_aggregate_summary(out_dir, args.iterations)

if __name__ == "__main__":
    main()