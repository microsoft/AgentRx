#!/usr/bin/env python3
"""
Domain-agnostic STATIC invariant generator.

Generates static invariants (policy-derived, tool-schema-derived) that apply
across all trajectories for a given domain.

Supports domains: flash, tau, magentic (via --domain flag)
Supports python_check only or python_check + nl_check (via --include-nl-check)

Usage (run from src/ directory):
  # Tau-retail (python-only invariants, default):
  python -m invariants.static_invariant_generator --domain tau --input-path ../trajectories/tau-retail/instruction_adherence_failure.json --out-path out/static_tau.json --endpoint trapi

  # Magentic-one:
  python -m invariants.static_invariant_generator --domain magentic --input-path ../trajectories/magentic-one/trajectories/invent_new_info.json --out-path out/static_magentic.json --endpoint trapi

  # With NL check invariants enabled:
  python -m invariants.static_invariant_generator --domain tau --input-path ../trajectories/tau-retail/instruction_adherence_failure.json --out-path out/static_tau_nl.json --endpoint trapi --include-nl-check

  # Custom policy document:
  python -m invariants.static_invariant_generator --domain tau --input-path ../trajectories/tau-retail/instruction_adherence_failure.json --out-path out/static.json --policy-path /path/to/policy.txt

  # Using Azure endpoint instead of TRAPI:
  python -m invariants.static_invariant_generator --domain magentic --input-path ../trajectories/magentic-one/trajectories/invent_new_info.json --out-path out/static.json --endpoint azure

Also importable for the larger pipeline:
  from agentrx.invariants.static_invariant_generator import StaticInvariantGenerator
"""
import argparse
import os
import json
import time
import datetime
import traceback
from typing import Any, Dict, List, Optional, Union, Set
from agentrx.llm_clients.trapi import LLMAgent as LLMAgentTrapi
from agentrx.llm_clients.azure import LLMAgent as LLMAgentAzure
import agentrx.pipeline.globals as g
from agentrx.invariants.domain_registry import get_domain_config, list_domains

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "1") == "1"

# NOTE: Tool lists, tool structures, and example blocks are now centralised in
#       invariants.domain_registry.  They were removed from this file to
#       eliminate duplication.  Use get_domain_config(domain) to access them.

RUBRIC_EVALUATION_ALGORITHM = """
Rubric Evaluation Algorithm:
Step 1: For each criterion in the rubric, evaluate whether it can be CLEARLY judged as PASS or FAIL based solely on the provided events.
- Mark as CLEAR_PASS if the criterion is demonstrably satisfied by the evidence
- Mark as CLEAR_FAIL if the criterion is demonstrably violated by the evidence  
- Mark as UNCLEAR if: insufficient events to judge, criterion is ambiguous in this context, or pass/fail cannot be decisively determined

Step 2: Discard ALL criteria marked as UNCLEAR from consideration.

Step 3: Determine final verdict:
- If ANY remaining criterion is CLEAR_FAIL → return verdict 'fail'
- If ALL remaining criteria are CLEAR_PASS (or no criteria remain after discarding UNCLEAR) → return verdict 'pass'

Important: Only fail when you have CLEAR evidence of failure. When in doubt, mark as UNCLEAR and discard.
"""

OUTPUT_FORMAT = """
Output Format:
You MUST return a JSON response with this EXACT structure:
{
  "verdict": "pass" or "fail",
  "rubric_results": [
    {
      "criterion_index": 0,
      "criterion": "full text of the criterion from the rubric",
      "evaluation": "CLEAR_PASS" or "CLEAR_FAIL" or "UNCLEAR",
      "reasoning": "detailed explanation of why this evaluation was chosen, citing specific evidence from events"
    }
    // ... one entry per rubric criterion
  ],
  "final_reasoning": "explanation of how the verdict was determined from rubric_results, stating which criteria were discarded as UNCLEAR and which criteria drove the final decision"
}

This detailed output is critical for debugging and understanding the evaluation.
"""

NL_CHECK_JUDGE_SYSTEM_PROMPT = """
You are a strict compliance judge. Evaluate only with evidence in the provided events and the policy text.
Do not infer intent beyond explicit statements.
CRITICAL: If required evidence is missing or ambiguous, mark the criterion as UNCLEAR.
When criteria are marked as UNCLEAR, they are discarded and do NOT contribute to a fail verdict.
If all criteria are UNCLEAR or CLEAR_PASS after discarding UNCLEAR ones, the final verdict is PASS.
Only mark as CLEAR_FAIL when there is definitive, unambiguous evidence of violation.
Ambiguity or missing evidence leads to PASS, not FAIL.
"""

# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------
def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)

def abspath_rel(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_text(path: str) -> str:
    p = abspath_rel(path)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def read_json(path: str) -> dict:
    p = abspath_rel(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------------------------------------------------------
# PRETTY PRINT HELPERS
# ------------------------------------------------------------------------------------
def _json_dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)

def pretty_json_text(text: str, *, default: str = "[]") -> str:
    s = (text or "").strip()
    if not s:
        return default
    try:
        return _json_dumps_pretty(json.loads(s))
    except Exception:
        # If it's not JSON, return raw (still useful for prompts/logs)
        return s

def pretty_json_obj(obj: Any, *, default: str = "[]") -> str:
    if obj is None:
        return default
    try:
        return _json_dumps_pretty(obj)
    except Exception:
        return str(obj)

# ------------------------------------------------------------------------------------
# TRAJECTORY ENUM EXTRACTION
# ------------------------------------------------------------------------------------
def extract_prompt_enums(
    traj: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Extract what can be derived from the Trajectory IR (no domain knowledge).

    Accepts either:
      - a single trajectory dict
      - OR a list of trajectories (we will use the first one by default)
    """
    # ---- coerce list -> single trajectory
    if isinstance(traj, list):
        if not traj:
            raise ValueError("trajectory list is empty")
        traj = traj[0]

    # ---- derive step indices
    step_indices: Set[int] = set()
    for step in traj.get("steps", []) or []:
        idx = step.get("index")
        if isinstance(idx, int):
            step_indices.add(idx)
    step_indices_sorted = sorted(step_indices)

    # ---- derive agent names from role (base part before '(')
    agents: Set[str] = set()
    for step in traj.get("steps", []) or []:
        for ss in step.get("substeps", []) or []:
            role = (ss.get("role") or "").strip()
            base = role.split("(", 1)[0].strip()
            if base:
                agents.add(base)
    agent_names = sorted(agents)

    # ---- build agent enum strings
    agent_union_parts = list(agent_names)
    agent_union_parts.append("*")

    agent_union = " | ".join(agent_union_parts) if agent_union_parts else "*"

    return {
        "step_indices": step_indices_sorted,
        "agent_union": agent_union,
    }



# ------------------------------------------------------------------------------------
# PROMPT TEMPLATES (flash-style with <<PLACEHOLDER>> substitution)
# ------------------------------------------------------------------------------------
STATIC_INVARIANT_PROMPT_PYTHON_ONLY = """You are an expert at analyzing domain policies to generate meaningful static assertions for agent trajectory verification. Your task is to identify important rules that prevent policy violations, errors, and ensure proper operation.
You must extract checkable rules that can be programmatically verified when specific tool calls are made during an agent's execution.

## Your Task:
Analyze the provided policy document and generate a focused set of static assertions that:
1. Are simple enough to implement programmatically and are objectively verifiable from trajectory data
2. Can be checked at specific triggers
3. Focus on critical policy violations that must be prevented
4. Include both critical business logic AND important error checking
5. Avoid trivial input checks, like assertions that check field completeness
6. Provide working Python code for each assertion that can be executed

## Focus Areas - Generate assertions for:
### **Precondition Checks**
- Required state/status before agent execution
- Examples:
  - Incident Management: "An incident_id must be identifiedbefore any incident-scoped investigation or communication."
  - Incident Management: "Kusto queries must be non-empty and must not contain placeholders like TODO, TBD, or <CLUSTER> before execution."
  - Retail: "Order status must be 'pending' before cancellation"
  - Healthcare: "Patient must be 'checked-in' before treatment actions"
  - Financial: "Account must be 'active' before transaction processing"
  - System Admin: "Service must be 'stopped' before configuration changes"

### **Sequential Dependencies** 
Required ordering between investigation steps and communication steps
- Examples:
  - Incident Management:: "IncidentAgent (fetch incident description) must precede KustoAgent (run diagnostics) for the same incident."
  - General: "User authentication must precede any data access operations"
  - Retail: "User must give explicit confirmation (yes) before any consequential database write actions"

### **Business Rule Constraints**
- Domain-specific operational limits and rules. Incident-management operational limits, safety/compliance constraints, and evidence-grounding rules
- Examples:
  - Incident Management: "Incident updates (or equivalent external communication steps) must be grounded in prior tool outputs (e.g., Kusto results, incident descriptions) within the same trajectory."
  - Incident Management: "IncidentAgent must only return incident descriptions (no extra troubleshooting); KustoAgent must only execute/provide query results (no invented interpretation)."
  - Retail: "Items can only be exchanged within same product category"
  - Healthcare: "Prescriptions require valid medical license verification"
  - Financial: "Transfer amounts cannot exceed daily limits"
  - Legal: "Document modifications require authorized personnel only"

### **Single-Use Restrictions**
- Actions that should occur at most once per incident (or once per approval) to avoid duplicate/unsafe operations
- Examples:
  - Incident Management: "Do not re-run the same Kusto query repeatedly without a stated reason (e.g., changed time window, different cluster, new hypothesis)."
  - Healthcare: "Critical procedure authorization can only be granted once"
  - Financial: "Account closure can only be initiated once per request"
  - System: "Database migration can only be executed once per maintenance window"
  
CRITICAL FORMATTING REQUIREMENTS:
1. Output ONLY valid JSON - NO markdown code blocks (no ```json or ```)
2. Use double quotes for all strings (not single quotes)
3. Ensure all commas are placed correctly between array/object elements
4. Ensure all brackets and braces are properly closed
5. Follow the EXACT schema below - do not add or remove fields
6. Validate your JSON before outputting
7. TAXONOMY TARGETS: Use ONLY values from the allowed list.
8. ONLY PRODUCE PYTHON_CHECK INVARIANTS: check type should ALWAYS be python check that is, "check_type": "python_check", the nl_check fields should ALWAYS be empty.

================================================================================================
TRAJECTORY FORMAT (IMPORTANT)
================================================================================================

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

  "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., Kusto syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.

    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*",  // REQUIRED: use the literal JSON string "*" (wildcard, matches every step). Static/global invariants describe policies that must hold throughout the trajectory and MUST use "*". Only use a specific int (e.g., 3) for the rare invariant that applies to one fixed step. Never use 0, 1, 2 as a stand-in for "every step" — the runtime checker treats those as concrete step indices and will silently never fire.
    "substep_index": "*",  // use "*" unless the invariant applies to one specific substep
    "role_name": "<<AGENT_UNION>>",  // agent names derived from trajectory; "*" matches all
    "content_regex": "*",  // regex applied to substep.content; "*" matches all
    "tool_name": "*"      // "*" matches all; pin to a specific tool name if the invariant targets one tool
  },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \\"\\\"Return True iff invariant holds.\\"\\\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

  "nl_check": {
    ALWAYS EMPTY
  }
}

================================================================================================
COMPLETE INVARIANT EXAMPLE (copy the event_trigger shape verbatim)
================================================================================================
The event_trigger.step_index value below is the literal JSON string "*". Reproduce that
literal for every static/global invariant you emit. Do NOT replace "*" with 0, 1, or any
other integer unless the invariant is tied to one specific step index.

{
  "assertion_name": "authenticate_before_privileged_operations",
  "taxonomy_targets": ["Instruction/PlanAdherenceFailure"],
  "invariant_type": "TEMPORAL",
  "event_trigger": {
    "step_index": "*",
    "substep_index": "*",
    "role_name": "assistant",
    "content_regex": "*",
    "tool_name": "*"
  },
  "check_hint": "Before any privileged tool call (e.g., get_user_details, cancel_pending_order, modify_*), verify that an earlier assistant tool call invoked one of the authentication tools (find_user_id_by_email or find_user_id_by_name_zip). If no prior auth call exists, this is a violation.",
  "check_type": "python_check",
  "python_check": {
    "function_name": "authenticate_before_privileged_operations",
    "args": ["trajectory", "current_step_index"],
    "code_lines": [
      "def authenticate_before_privileged_operations(trajectory, current_step_index):",
      "    priv = {'get_user_details','get_order_details','cancel_pending_order','modify_pending_order_address','modify_pending_order_items','return_delivered_order_items','exchange_delivered_order_items','modify_user_address'}",
      "    auth = {'find_user_id_by_email','find_user_id_by_name_zip'}",
      "    step = trajectory['steps'][current_step_index]",
      "    is_priv = any(ss.get('role')=='tool_call' and ss.get('tool_name') in priv for ss in step.get('substeps', []))",
      "    if not is_priv: return True",
      "    for prev in trajectory['steps'][:current_step_index]:",
      "        for ss in prev.get('substeps', []):",
      "            if ss.get('role')=='tool_call' and ss.get('tool_name') in auth:",
      "                return True",
      "    return False"
    ]
  },
  "nl_check": {}
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants

## Instructions:
1. Read the policy document thoroughly
2. Identify important constraints, rules, and requirements
3. Include critical business logic, essential error checking, and policy compliance
4. Focus on invariants that ensure proper operation and prevent violations
5. Generate a comprehensive but focused set of static invariants
6. Generate an invariant that specifically checks whether user is following the task instruction, preferably NL. 
7. AVOID very trivial invariants checks.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Kusto result:\\s*Query successful"
  because real traces often contain:
    "**Kusto result:**\\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

IMPORTANT: ALWAYS PRODUCE PYTHON CHECK INVARIANTS. DO NOT PRODUCE NL CHECK INVARIANTS.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Handle edge cases and missing data gracefully - verify field existence before access
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, etc.) instead of returning False. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables and extracted JSON fields to aid in debugging and verification

================================================================================================
EXAMPLE TRAJECTORY: To understand the format of trajectory and events - DO NOT OVERFIT ON THIS
================================================================================================

<<SAMPLE_TRAJECTORY_JSON>>

================================================================================================
The agents available to the LLM are:
================================================================================================

<<TOOLS_LIST>>

================================================================================================
**IMPORTANT: Review this JSON structure carefully before writing any code. Use ONLY the field names that appear in this structure:**
================================================================================================

<<TOOLS_STRUCTURE>>

================================================================================================
Policy Document:
================================================================================================

<<POLICY_TEXT>>"""


STATIC_INVARIANT_PROMPT = """You are an expert at analyzing domain policies to generate meaningful static assertions for agent trajectory verification. Your task is to identify important rules from policy documents that prevent policy violations, errors, and ensure proper operation.
You must extract checkable rules from policy documents that can be  verified either using python check or natural language check when specific tool calls are made during an agent's execution.

## Your Task:
Analyze the provided policy document and generate a focused set of static assertions that:
1. Are simple enough to implement and are objectively verifiable from trajectory data
2. Can be checked at specific triggers
3. Focus on critical policy violations that must be prevented
4. Include both critical business logic AND important error checking
5. Avoid trivial input checks, like assertions that check field completeness
6. Provide working Python code for each assertion that can be executed

## Focus Areas - Generate assertions for:
### **Precondition Checks**
- Required state/status before agent execution
- Examples:
  - Incident Management: "An incident_id must be identifiedbefore any incident-scoped investigation or communication."
  - Incident Management: "Kusto queries must be non-empty and must not contain placeholders like TODO, TBD, or <CLUSTER> before execution."
  - Retail: "Order status must be 'pending' before cancellation"
  - Healthcare: "Patient must be 'checked-in' before treatment actions"
  - Financial: "Account must be 'active' before transaction processing"
  - System Admin: "Service must be 'stopped' before configuration changes"

### **Sequential Dependencies** 
Required ordering between investigation steps and communication steps
- Examples:
  - Incident Management:: "IncidentAgent (fetch incident description) must precede KustoAgent (run diagnostics) for the same incident."
  - General: "User authentication must precede any data access operations"
  - Retail: "User must give explicit confirmation (yes) before any consequential database write actions"

### **Business Rule Constraints**
- Domain-specific operational limits and rules. Incident-management operational limits, safety/compliance constraints, and evidence-grounding rules
- Examples:
  - Incident Management: "Incident updates (or equivalent external communication steps) must be grounded in prior tool outputs (e.g., Kusto results, incident descriptions) within the same trajectory."
  - Incident Management: "IncidentAgent must only return incident descriptions (no extra troubleshooting); KustoAgent must only execute/provide query results (no invented interpretation)."
  - Retail: "Items can only be exchanged within same product category"
  - Healthcare: "Prescriptions require valid medical license verification"
  - Financial: "Transfer amounts cannot exceed daily limits"
  - Legal: "Document modifications require authorized personnel only"

### **Single-Use Restrictions**
- Actions that should occur at most once per incident (or once per approval) to avoid duplicate/unsafe operations
- Examples:
  - Incident Management: "Do not re-run the same Kusto query repeatedly without a stated reason (e.g., changed time window, different cluster, new hypothesis)."
  - Healthcare: "Critical procedure authorization can only be granted once"
  - Financial: "Account closure can only be initiated once per request"
  - System: "Database migration can only be executed once per maintenance window"

================================================================================================
TRAJECTORY FORMAT (IMPORTANT)
================================================================================================

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

  "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., Kusto syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.

    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*",  // REQUIRED: use the literal JSON string "*" (wildcard, matches every step). Static/global invariants describe policies that must hold throughout the trajectory and MUST use "*". Only use a specific int (e.g., 3) for the rare invariant that applies to one fixed step. Never use 0, 1, 2 as a stand-in for "every step" — the runtime checker treats those as concrete step indices and will silently never fire.
    "substep_index": "*",  // use "*" unless the invariant applies to one specific substep
    "role_name": "<<AGENT_UNION>>",  // agent names derived from trajectory; "*" matches all
    "content_regex": "*",  // regex applied to substep.content; "*" matches all
    "tool_name": "*"      // "*" matches all; pin to a specific tool name if the invariant targets one tool
  },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \\"\\\"Return True iff invariant holds.\\"\\\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

  "nl_check": {
    "judge_system_prompt_template": <<NL_CHECK_JUDGE_SYSTEM_PROMPT>>,
    "judge_user_prompt_template": "template using {TASK_INSTRUCTION} {CURRENT_EVENT_JSON} {WINDOW_EVENTS_JSON}",
    "judge_scope_notes": "what events are in scope and what counts as evidence",
    "focus_steps_instruction": "REQUIRED: Clear, actionable instruction identifying which 2-4 specific events to examine. Must specify events by relative position (e.g., 'immediately prior user message', '2 steps back', 'most recent get_order_details result') and explain what to look for in each. See FOCUS STEPS INSTRUCTION section for detailed examples.",
    "judge_rubric": ["objective criterion 1", "objective criterion 2", "..."],
    "rubric_evaluation_algorithm_template": <<RUBRIC_EVALUATION_ALGORITHM>>,
    "output_format_template": <<OUTPUT_FORMAT>>
  }
}

================================================================================================
COMPLETE INVARIANT EXAMPLE (copy the event_trigger shape verbatim)
================================================================================================
The event_trigger.step_index value below is the literal JSON string "*". Reproduce that
literal for every static/global invariant you emit. Do NOT replace "*" with 0, 1, or any
other integer unless the invariant is tied to one specific step index.

{
  "assertion_name": "user_agent_follows_task_instruction",
  "taxonomy_targets": ["UnderspecifiedUserIntent"],
  "invariant_type": "TEMPORAL",
  "event_trigger": {
    "step_index": "*",
    "substep_index": "*",
    "role_name": "user",
    "content_regex": "*",
    "tool_name": "*"
  },
  "check_hint": "Verify that the simulated user agent's message is semantically consistent with the original task instruction. Check that the user is not introducing new requirements, contradicting the task instruction, or providing information that deviates from what was specified in the task.",
  "check_type": "nl_check",
  "python_check": {},
  "nl_check": {
    "judge_system_prompt_template": "You are a strict compliance judge. Evaluate only with evidence in the provided events and the task instruction. Do not infer intent beyond explicit statements. CRITICAL: If required evidence is missing or ambiguous, mark the criterion as UNCLEAR.",
    "judge_user_prompt_template": "TASK INSTRUCTION: {TASK_INSTRUCTION}\\n\\nCURRENT USER MESSAGE: {CURRENT_EVENT_JSON}\\n\\nEvaluate whether the user agent's message is consistent with the task instruction.",
    "judge_scope_notes": "Compare the user agent's current message against the original task instruction to detect semantic drift, contradictions, or invention of new requirements.",
    "focus_steps_instruction": "Focus on: (1) The current user message being evaluated. (2) The original task instruction provided at the start. (3) Check for semantic alignment.",
    "judge_rubric": [
      "The user agent's request type matches what is specified in the task instruction",
      "If the user mentions specific items, quantities, or attributes, they are consistent with the task instruction",
      "The user agent does not introduce new constraints not present in the task instruction"
    ],
    "rubric_evaluation_algorithm_template": "{RUBRIC_EVALUATION_ALGORITHM}",
    "output_format_template": "{OUTPUT_FORMAT}"
  }
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants

## Instructions:
1. Read the policy document thoroughly
2. Identify important constraints, rules, and requirements
3. Include critical business logic, essential error checking, and policy compliance
4. Focus on invariants that ensure proper operation and prevent violations
5. Generate a comprehensive but focused set of static invariants
6. Generate an invariant that specifically checks whether user is following the task instruction, preferably NL. 
7. AVOID very trivial invariants checks.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Kusto result:\\s*Query successful"
  because real traces often contain:
    "**Kusto result:**\\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

================================================================================================
DECISION GUIDE: PYTHON_CHECK VS NL_CHECK
================================================================================================
Use python_check when:
- Tool output is structured JSON with extractable fields (e.g., get_order_details, get_user_details)
- Check involves counting, summing, or comparing numeric values
- Check involves verifying exact field matches (IDs, status codes, amounts, dates)
- Check involves parsing and validating data structure or schema
- Check uses simple regex patterns (e.g., matching an order_id format like r"#W\\d+" or extracting keys or specific count in the content message)
- Check involves deterministic string containment or exact substring matching

Use nl_check when:
- Check requires understanding paraphrasing or semantic equivalence (e.g., "bigger size" = "larger size")
- Check requires judging whether two different phrasings express the same intent
- Check requires evaluating whether user agent messages align with task instruction semantically
- The content is unstructured natural language without clear fields to extract
- Check would require a long list of synonyms or variations to cover programmatically

A naive python_check approach would try to match exact phrases like:
  - "I want to return"
  - "I'd like to exchange"
  - "cancel my order"
  - "modify my address"
  - "get a refund"
  - "change the items"
But users express the same intent in many ways:
  - "Could you help me send back..." (return)
  - "I need to swap this for..." (exchange)
  - "Please void the order" (cancel)
  - "Update my shipping info" (modify address)
Matching exact phrases is brittle and incomplete. Use nl_check for semantic intent.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Handle edge cases and missing data gracefully - verify field existence before access
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, etc.) instead of returning False. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables and extracted JSON fields to aid in debugging and verification

================================================================================================
## NL CHECK GUIDELINES
================================================================================================

For nl_check invariants, MUST include a "focus_steps_instruction" field that tells the judge which specific events to pay attention to.

The instruction should:
1. Be clear and actionable - the judge should know exactly which events to look at
2. Identify 2-4 critical events by their position relative to current_step_index
3. Explain WHY each event is important for the evaluation
4. Use relative positioning (e.g., "immediately prior", "2 steps back", "most recent X tool result")

EXAMPLES OF GOOD FOCUS STEPS INSTRUCTIONS:

For confirmation checks:
"Focus your evaluation on these key events: (1) The immediately prior message - check for explicit confirmation keywords like 'yes', 'confirm', 'proceed', 'ok'. (2) The assistant message two to three steps back - verify that it describes the action of interest (cancel/modify/exchange/return) and mentions the relevant target entity identifier(s) and critical parameters (e.g., record/entity ID, amount, destination, items, scope). (3) The current tool call arguments - verify they match the order_id/incident_id and details mentioned by another agent."

For information invention checks:
"Focus on: (1) The current agent content message being evaluated. (2) The most recent relevant tool results in the window — extract the authoritative facts actually present (identifiers, key fields/attributes, timestamps, computed outputs). (3) Compare specific claims in the agent's message against these tool results."

For temporal/prerequisite checks:
"Focus on: (1) The current tool call event — identify what operation is being attempted and what prerequisites it implies (authentication, entity lookup, permission check, required context fetch, state precondition). (2) Scan backward through all prior events to find the most recent event(s) that satisfy each prerequisite. (3) Most recent state snapshot for the same target entity — locate the latest tool result that reports the entity's current state."

KEY PRINCIPLES:
- Be specific about WHICH events (by relative position or role+name)
- Explain WHAT to look for in each event
- Make it easy for the judge to locate and examine the right evidence
- Reduce cognitive load by narrowing focus to what matters most

================================================================================================
COMPLETE NL_CHECK INVARIANT EXAMPLES
================================================================================================

## EXAMPLE 1 : Explicit User Confirmation Before Write Actions

{
    "assertion_name": "explicit_user_confirmation_before_write_actions",
    "taxonomy_targets": [
        "Instruction/PlanAdherenceFailure",
        "IntentPlanMisalignment"
    ],
    "invariant_type": "<INVARIANT_TYPE>",
    "event_trigger": {
        "step_index": <STEP_INDEX>,
        "role_name": "<AGENT_NAME>"
    },
    "check_hint": "Before any write-action tool invocation (cancel, exchange, return, modify order/address), verify that the assistant previously described the intended action and target ID, and that the user explicitly confirmed (e.g., 'yes', 'please proceed'). The current tool call's identifier (order_id or user_id) must match the one mentioned earlier in the assistant's explanation.",
    "check_type": "nl_check",
    "python_check": {},
    "nl_check": {
        "judge_system_prompt_template": <<NL_CHECK_JUDGE_SYSTEM_PROMPT>>,
        "judge_user_prompt_template": "TASK INSTRUCTION:\\n{TASK_INSTRUCTION}\\n\\nCURRENT EVENT:\\n{CURRENT_EVENT_JSON}\\n\\nCONTEXT WINDOW EVENTS:\\n{WINDOW_EVENTS_JSON}\\n\\nEvaluate whether, before the current write-action tool call (cancel, modify, exchange, return, address update), the assistant clearly described the action and entity ID, and the user explicitly confirmed proceeding with that action, and that the IDs match between description and tool call.",
        "judge_scope_notes": "Judge only within the window of events provided. You must determine whether this specific write-action tool call is properly preceded by a clear assistant explanation and explicit user confirmation for the same action and same order_id/user_id.",
        "focus_steps_instruction": "Focus on: (1) The current assistant tool-call step to identify which write-action tool is being invoked and what identifier (order_id or user_id) is present in its arguments. (2) The immediately prior user message to check for explicit affirmation language indicating consent to proceed with the described action. (3) The assistant messages in the 2-3 steps before the current tool call to see whether the assistant described the intended action (cancel/modify/exchange/return/address update) and mentioned the same identifier as in the tool call.",
        "judge_rubric": [
            "There exists an assistant message earlier in the context that explicitly describes the specific write action type (cancel, modify, exchange, return, or address update) and includes the same identifier (order_id or user_id) that appears in the current tool call arguments.",
            "There exists a user message after the assistant's action description and before the current tool call that contains explicit confirmation language agreeing to proceed with that specific action (for example, 'yes', 'confirm', 'please proceed', 'go ahead').",
            "The action parameters described by the assistant (such as which items to exchange/return, or what address/payment will be changed) are consistent with the scope implied by the current tool call; there is no clear indication that the tool call is broader or different than what the user confirmed."
        ],
        "rubric_evaluation_algorithm_template": <<RUBRIC_EVALUATION_ALGORITHM>>,
        "output_format_template": <<OUTPUT_FORMAT>>
    }
}

================================================================================================
EXAMPLE TRAJECTORY: To understand the format of trajectory and events - DO NOT OVERFIT ON THIS
================================================================================================

<<SAMPLE_TRAJECTORY_JSON>>

================================================================================================
The agents available to the LLM are:
================================================================================================

<<TOOLS_LIST>>

================================================================================================
**IMPORTANT: Review this JSON structure carefully before writing any code. Use ONLY the field names that appear in this structure:**
================================================================================================

<<TOOLS_STRUCTURE>>

================================================================================================
Policy Document:
================================================================================================

<<POLICY_TEXT>>"""


# ------------------------------------------------------------------------------------
# GENERATOR
# ------------------------------------------------------------------------------------
class StaticInvariantGenerator:
    def __init__(
        self,
        *,
        traj_for_enums: Union[Dict[str, Any], List[Dict[str, Any]]],
        tools_list: Optional[List[str]],
        tools_structure: Optional[Union[dict, str]],
        domain: str = "flash",
        policy_document_path: Optional[str] = None,
        out_path: str,
        model_name: Optional[str] = None,
        include_nl_check: bool = True,
        endpoint: str = "trapi",
    ) -> None:
        self.traj_for_enums = traj_for_enums
        self.tools_list = [t.strip() for t in (tools_list or []) if (t or "").strip()]
        self.tools_structure = tools_structure
        self.domain = domain
        self.include_nl_check = include_nl_check
        self.out_path = abspath_rel(out_path)

        if policy_document_path is None:
            # Look up default from domain registry
            try:
                cfg = get_domain_config(domain)
                if cfg.default_policy_path:
                    policy_document_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "..", cfg.default_policy_path
                    )
            except ValueError:
                pass  # unknown domain, caller must supply path
        
        # Load from file
        if policy_document_path and os.path.exists(policy_document_path):
            with open(policy_document_path, 'r', encoding='utf-8') as f:
                self.policy_document = f.read()
            print(f"Loaded policy document from: {policy_document_path}")
        else:
                self.policy_document = ""
            
        ensure_dir(os.path.dirname(self.out_path))

        if endpoint == "copilot":
            from agentrx.llm_clients.copilot_cli import copilot_mk_client
            self.client = copilot_mk_client()
            self.model_name = model_name or "copilot-cli"
            self._endpoint_url = "copilot-cli"
        elif endpoint == "azure":
            self.client = LLMAgentAzure.azure_mk_client()
            self.model_name = model_name or g.DEPLOYMENT
            self._endpoint_url = g.ENDPOINT
        else:
            self.client = LLMAgentTrapi.trapi_mk_client()
            self.model_name = model_name or g.TRAPI_DEPLOYMENT_NAME
            self._endpoint_url = f"{g.TRAPI_ENDPOINT_PREFIX}{g.TRAPI_INSTANCE}"


    def build_prompt(self) -> str:
        """Build the LLM prompt using flash-style placeholder substitution."""
        # Derive agent union from trajectory
        enums = extract_prompt_enums(self.traj_for_enums)
        agent_union = enums["agent_union"]

        # Serialize tools data
        tools_list_str = json.dumps(self.tools_list or [], indent=2, ensure_ascii=False)
        tools_structure_str = json.dumps(self.tools_structure or {}, indent=2, ensure_ascii=False)

        # Use input trajectory as sample example
        sample_trajectory_json = json.dumps(self.traj_for_enums, indent=2, ensure_ascii=False)

        # Choose template based on nl_check flag
        template = STATIC_INVARIANT_PROMPT_PYTHON_ONLY if not self.include_nl_check else STATIC_INVARIANT_PROMPT

        # Build prompt via simple placeholder substitution (no double-injection)
        prompt = (template
            .replace("<<SAMPLE_TRAJECTORY_JSON>>", sample_trajectory_json)
            .replace("<<TOOLS_LIST>>", tools_list_str)
            .replace("<<TOOLS_STRUCTURE>>", tools_structure_str)
            .replace("<<AGENT_UNION>>", agent_union)
            .replace("<<POLICY_TEXT>>", self.policy_document)
        )

        # Substitute nl_check template constants (json.dumps ensures proper
        # quoting + escaping so the schema example is valid JSON)
        if self.include_nl_check:
            prompt = (prompt
                .replace("<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>", json.dumps(NL_CHECK_JUDGE_SYSTEM_PROMPT.strip()))
                .replace("<<RUBRIC_EVALUATION_ALGORITHM>>", json.dumps(RUBRIC_EVALUATION_ALGORITHM.strip()))
                .replace("<<OUTPUT_FORMAT>>", json.dumps(OUTPUT_FORMAT.strip()))
            )

        return prompt

    def run(self, debug_prompt_path: Optional[str] = None) -> Dict[str, Any]:
        prompt = self.build_prompt()

        # --- Debug: dump the full prompt so you can inspect it ---------------
        if debug_prompt_path is None:
            debug_prompt_path = os.path.join(
                os.path.dirname(self.out_path),
                f"DEBUG_static_prompt_{self.domain}.txt",
            )
        ensure_dir(os.path.dirname(debug_prompt_path))
        with open(debug_prompt_path, "w", encoding="utf-8") as dbg_f:
            dbg_f.write(prompt)
        print(f"[DEBUG] Full prompt written to: {debug_prompt_path}")
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        # --------------------------------------------------------------------

        start_ts = datetime.datetime.now()
        start = time.perf_counter()

        from openai import RateLimitError as _RLE
        _static_retries = 5
        for _attempt in range(_static_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                break
            except _RLE:
                if _attempt < _static_retries - 1:
                    _wait = 4 * (2 ** _attempt)
                    print(f"[RATE LIMIT] Static gen retry in {_wait}s ({_attempt+1}/{_static_retries})", flush=True)
                    time.sleep(_wait)
                else:
                    raise

        end = time.perf_counter()
        end_ts = datetime.datetime.now()

        raw = (resp.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
            # Copilot CLI may return a bare array instead of {"invariants": [...]}.
            # Wrap it so downstream code always sees a dict.
            if isinstance(obj, list):
                obj = {"invariants": obj}
            if not isinstance(obj, dict):
                raise ValueError(f"model returned {type(obj).__name__}, expected dict or list")
        except Exception as e:
            raise RuntimeError(
                f"Static invariants JSON parse failed: {e}\n")

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

        usage = getattr(resp, "usage", None)
        tel = {
            "model": self.model_name,
            "endpoint": self._endpoint_url,
            "time": {
                "start_time": start_ts.isoformat(timespec="seconds"),
                "end_time": end_ts.isoformat(timespec="seconds"),
                "execution_time_sec": round(end - start, 4),
            },
            "tokens": None,
        }
        if usage is not None:
            tel["tokens"] = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }

        return obj

if __name__ == "__main__":
    # Usage: see module docstring at top of file
    from agentrx.ir.trajectory_ir import load_trajectories

    available_domains = list_domains()

    parser = argparse.ArgumentParser(description="Static invariant generator")
    parser.add_argument("--domain", type=str, default="flash",
                        choices=available_domains,
                        help=f"Domain to run (registered: {available_domains})")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to input trajectory file")
    parser.add_argument("--out-path", type=str, default=None,
                        help="Output path for static invariants JSON")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to policy document (default: from domain registry)")
    parser.add_argument("--endpoint", type=str, default="trapi",
                        choices=["azure", "trapi"],
                        help="LLM endpoint to use (default: trapi)")
    grp = parser.add_mutually_exclusive_group()
    parser.set_defaults(include_nl_check=False)
    grp.add_argument("--include-nl-check", dest="include_nl_check", action="store_true")
    args = parser.parse_args()

    domain = args.domain
    cfg = get_domain_config(domain)
    out_path = args.out_path or os.getenv("OUT_PATH", f"out/static_invariants_{domain}.json")

    # Domain-specific tools / policy — all from registry
    tools_list = cfg.tools_list
    tools_structure = cfg.tools_structure
    policy_document_path = args.policy_path
    if policy_document_path is None and cfg.default_policy_path:
        policy_document_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", cfg.default_policy_path
        )

    # Load trajectory via registry IR converter
    input_path_abs = os.path.abspath(args.input_path)  # resolve relative to cwd, not script dir
    if not os.path.exists(input_path_abs):
        print(f"[ERROR] input file not found: {input_path_abs}", flush=True)
        exit(1)

    raw = load_trajectories(input_path_abs)
    ir_fn = cfg.ir_converter
    data = ir_fn(raw)

    if not isinstance(data, list) or not data:
        print(f"[ERROR] Expected a non-empty list of trajectories, got {type(data)}", flush=True)
        exit(1)
    trajectory = data[0]

    print(f"[CONFIG] domain={domain} input={args.input_path} nl_check={args.include_nl_check}")

    gen = StaticInvariantGenerator(
        traj_for_enums=trajectory,
        tools_list=tools_list,
        tools_structure=tools_structure,
        domain=domain,
        policy_document_path=policy_document_path or "",
        out_path=out_path,
        include_nl_check=args.include_nl_check,
        endpoint=args.endpoint,
    )
    gen.run()