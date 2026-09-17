#!/usr/bin/env python3
"""
Invariant Checker — verifies agent trajectories against static and dynamic invariants.

Usage (run from agentverify/src/):

  # Tau domain (with policy document + dynamic invariants):
  python -m invariants.checker --domain tau \
    --input-path "../trajectories/tau-retail/hallucination_doubt.json" \
    --static-invariants "invariants/out/static_tau.json" \
    --dynamic-invariants-dir "invariants/dynamic_invariant_outputs" \
    --policy-path "../data/policies/retail_policy.txt" \
    --out-dir "invariants/out/checker_results" \
    --trapi

  # Magentic domain:
  python -m invariants.checker --domain magentic \
    --input-path "../data/magentic_dataset" \
    --static-invariants "invariants/out/static.json" \
    --dynamic-invariants-dir "invariants/dynamic_invariant_outputs" \
    --out-dir "invariants/out/checker_results" \
    --azure

  # Use --azure (default) or --trapi to select the LLM endpoint.
  # Set SKIP_NL=1 to skip nl_check invariants (no LLM calls, faster).
  # Set DEBUG=0 to suppress debug output.

Outputs per trajectory (written to --out-dir/<task_id>/):
  violations_<domain>.json   — detected violations
  telemetry_<domain>.json    — per-check timing, tokens, inputs/outputs
  debug_skips_<domain>.json  — skipped steps/invariants + metrics
"""

import argparse
import sys
import os, json, re, time
import threading
import traceback 
import io 
from dataclasses import dataclass, asdict
import types
from typing import Any, Dict, List, Optional, Tuple, Union
from xml import dom
from agentrx.invariants.static_invariant_generator import (
    RUBRIC_EVALUATION_ALGORITHM,
    OUTPUT_FORMAT,
    NL_CHECK_JUDGE_SYSTEM_PROMPT
)
from agentrx.llm_clients.trapi import LLMAgent as LLMAgentTrapi
from agentrx.llm_clients.azure import LLMAgent as LLMAgentAzure

from agentrx.ir.trajectory_ir import tau_bench_ir, load_trajectories, flash_ir, magentic_ir
import agentrx.pipeline.globals as g

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "1") == "1"
LOG_VERBOSE = os.getenv("LOG_VERBOSE", "0") == "1"
SKIP_NL = os.getenv("SKIP_NL", "0") == "1"
def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None

DEBUG_MATCH = os.getenv("DEBUG_MATCH", "1") == "1"       
DEBUG_APPLY = os.getenv("DEBUG_APPLY", "1") == "1" 
DEBUG_SUBSTEPS = os.getenv("DEBUG_SUBSTEPS", "1") == "1" 
DEBUG_TOOL_PARSE = os.getenv("DEBUG_TOOL_PARSE", "1") == "1"
DEBUG_PY_EXEC = os.getenv("DEBUG_PY_EXEC", "1") == "1" 
DEBUG_PY_CAPTURE_STDOUT = os.getenv("DEBUG_PY_CAPTURE_STDOUT", "1") == "1"
DEBUG_NL_PROMPTS = os.getenv("DEBUG_NL_PROMPTS", "1") == "1"
DEBUG_INV_DUMP = os.getenv("DEBUG_INV_DUMP", "1") == "1" 
DEBUG_ONLY_ASSERTION = os.getenv("DEBUG_ONLY_ASSERTION", "").strip() 
DEBUG_MAX_JSON_CHARS = safe_int(os.getenv("DEBUG_MAX_JSON_CHARS", "4000")) or 4000 
DEBUG_MAX_TEXT_CHARS = safe_int(os.getenv("DEBUG_MAX_TEXT_CHARS", "600")) or 600

# Hard wall-clock cap for a single LLM-generated python_check (exec + fn call).
# Prevents infinite loops in generated code from hanging the whole sweep.
PYCHECK_TIMEOUT_SEC = safe_int(os.getenv("AGENTRX_PYCHECK_TIMEOUT_SEC", "30")) or 30


class PyCheckTimeout(Exception):
    """Raised when a python_check exceeds PYCHECK_TIMEOUT_SEC wall-clock seconds."""


def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)

def die(msg: str) -> None:
    raise RuntimeError(msg)

def short(x: Any, n: Optional[int] = None) -> str:
    n = DEBUG_MAX_TEXT_CHARS if n is None else n
    s = "" if x is None else str(x)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…(trunc,len={len(s)})"

def jdump(x: Any, maxlen: Optional[int] = None) -> str:
    maxlen = DEBUG_MAX_JSON_CHARS if maxlen is None else maxlen
    try:
        s = json.dumps(x, indent=2, ensure_ascii=False)
    except Exception:
        s = repr(x)
    if len(s) <= maxlen:
        return s
    return s[:maxlen] + f"\n…(trunc,len={len(s)})"

def inv_name(inv: Dict[str, Any]) -> str:
    return str(inv.get("assertion_name") or "<missing-assertion_name>")

def focus_inv(inv: Dict[str, Any]) -> bool:
    if not DEBUG_ONLY_ASSERTION:
        return True
    return (inv.get("assertion_name") or "") == DEBUG_ONLY_ASSERTION

def _line() -> None:
    if DEBUG:
        print("[DEBUG] " + "-" * 90, flush=True)

# ------------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------------
@dataclass
class Violation:
    task_id: Union[int, str]
    step_index: int
    assertion_name: str
    invariant_type: str
    check_type: str  # python_check | nl_check
    severity: str
    check_hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    taxonomy_targets: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CheckTelemetry:
    task_id: Union[int, str]
    step_index: int
    assertion_name: str
    check_type: str
    check_time_sec: float
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None
    check_input: Optional[Dict[str, Any]] = None
    check_output: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ------------------------------------------------------------------------------------
# VERIFIER
# ------------------------------------------------------------------------------------
class AllVerifier:
    def __init__(self, invariants_path: str, policy_document_path: str, log_verbose: bool = False, client: str = "azure"):
        dbg(f"Flags: DEBUG={DEBUG} LOG_VERBOSE={LOG_VERBOSE} SKIP_NL={SKIP_NL}")
        self.client = client

        
        dbg(
            f"DebugFlags: DEBUG_MATCH={DEBUG_MATCH} DEBUG_APPLY={DEBUG_APPLY} DEBUG_SUBSTEPS={DEBUG_SUBSTEPS} "
            f"DEBUG_TOOL_PARSE={DEBUG_TOOL_PARSE} DEBUG_PY_EXEC={DEBUG_PY_EXEC} "
            f"DEBUG_PY_CAPTURE_STDOUT={DEBUG_PY_CAPTURE_STDOUT} DEBUG_NL_PROMPTS={DEBUG_NL_PROMPTS} "
            f"DEBUG_INV_DUMP={DEBUG_INV_DUMP} DEBUG_ONLY_ASSERTION={DEBUG_ONLY_ASSERTION!r} "
            f"DEBUG_MAX_JSON_CHARS={DEBUG_MAX_JSON_CHARS} DEBUG_MAX_TEXT_CHARS={DEBUG_MAX_TEXT_CHARS}"
        )

        self.invariants_path = invariants_path
        self.policy_document_path = policy_document_path
        self.log_verbose = log_verbose

        # Load policy document for nl_check
        self.policy_text = ""
        if policy_document_path:
            with open(policy_document_path, "r", encoding="utf-8") as f:
                self.policy_text = f.read()

        # Load invariants
        with open(invariants_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.invariants: List[Dict[str, Any]] = self.extract_invariants(data)
        dbg(f"Loaded flattened invariants: {len(self.invariants)}")
        
        if DEBUG and self.invariants:
            sample = self.invariants[: min(3, len(self.invariants))]
            dbg("Sample invariants (first up to 3):")
            for i, inv in enumerate(sample):
                dbg(f"  [{i}] assertion_name={inv.get('assertion_name')!r} keys={sorted(list(inv.keys()))}")
                trig = inv.get("event_trigger")
                if trig is not None:
                    dbg(f"      event_trigger={jdump(trig, maxlen=800)}")
                dbg(f"      check_type_field={inv.get('check_type')!r} check_field={inv.get('check')!r} inferred={self._infer_check_type(inv)!r}")

        if self.client == "copilot":
            from agentrx.llm_clients.copilot_cli import copilot_mk_client
            self.llm_client = copilot_mk_client()
            self.model_name = "copilot-cli"
        elif self.client == "azure":
            self.llm_client = LLMAgentAzure.azure_mk_client()
            self.model_name = g.DEPLOYMENT
        else:
            self.llm_client = LLMAgentTrapi.trapi_mk_client()
            self.model_name = g.TRAPI_DEPLOYMENT_NAME

        # Metrics / logs
        self.total_python_checks = 0
        self.total_nl_checks = 0
        self.total_tokens_used = 0
        self.total_python_check_errors = 0

        self.telemetry: List[CheckTelemetry] = []
        self.skipped_steps: List[Dict[str, Any]] = []
        self.skipped_invariants: List[Dict[str, Any]] = []

        dbg("Verifier initialized.")

    def _infer_check_type(self, inv: Dict[str, Any]) -> Optional[str]:
        # your model output uses either "check" or "check_type"
        ct = inv.get("check_type") or inv.get("check")
        if ct in ("python_check", "nl_check"):
            return ct
        if isinstance(inv.get("python_check"), dict) and inv["python_check"]:
            return "python_check"
        if isinstance(inv.get("nl_check"), dict) and inv["nl_check"]:
            return "nl_check"
        return None

    # ---------------------------
    def extract_invariants(self, data) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []

        out: List[Dict[str, Any]] = []

        def _looks_like_invariant(d: Dict[str, Any]) -> bool:
            # Real invariants have these fields; per-step wrappers do not.
            return isinstance(d, dict) and (
                "assertion_name" in d or "event_trigger" in d or "check_type" in d
            )

        def add_payload(x: Any) -> None:
            if x is None:
                return
            if isinstance(x, dict):
                # If this is a per-step wrapper {step_num, decision, invariant: [...]},
                # recurse into the nested 'invariant'/'invariants' list. Otherwise
                # treat the dict as a leaf invariant object.
                if not _looks_like_invariant(x) and (
                    "invariant" in x or "invariants" in x
                ):
                    add_payload(x.get("invariant"))
                    add_payload(x.get("invariants"))
                    return
                out.append(x)
                return
            if isinstance(x, list):
                for y in x:
                    add_payload(y)

        # Load static invariants from static_invariants_used (stored as JSON string)
        static_str = data.get("static_invariants_used")
        if isinstance(static_str, str) and static_str.strip():
            try:
                static_data = json.loads(static_str)
                if isinstance(static_data, dict) and "invariant" in static_data:
                    add_payload(static_data.get("invariant"))
                    dbg(f"Loaded {len(out)} static invariants from static_invariants_used")
            except json.JSONDecodeError as e:
                dbg(f"Failed to parse static_invariants_used: {e}")

        # Load dynamic invariants from per_step_outputs
        pso = data.get("per_step_outputs")
        if isinstance(pso, list):
            dynamic_count = 0
            for item in pso:
                if not isinstance(item, dict):
                    continue
                parsed = item.get("parsed")
                if not isinstance(parsed, dict):
                    continue
                before = len(out)
                add_payload(parsed.get("invariant"))
                dynamic_count += len(out) - before
            if dynamic_count > 0:
                dbg(f"Loaded {dynamic_count} dynamic invariants from per_step_outputs")
            return out

        # Also check one-shot format: dynamic invariants stored directly
        if "dynamic_invariants" in data:
            add_payload(data.get("dynamic_invariants"))

        if "invariants" in data:
            add_payload(data.get("invariants"))
            return out

        if "invariant" in data:
            add_payload(data.get("invariant"))
            return out

        return out

    def _base_agent(self, role: str) -> str:
        return (role or "").split("(", 1)[0].strip()
    
    def _debug_parse_tool_wrapper(self, content: str) -> Dict[str, Any]:
        """
        Best-effort parser for tool wrapper strings for DEBUGGING only.
        Does NOT affect matching logic.
        """
        info = {"mode": "unknown", "tool_name": None, "response_json_ok": False, "response_json": None}
        if not content or not isinstance(content, str):
            info["mode"] = "empty"
            return info
        s = content.strip()
        # raw json?
        if s.startswith("{") and s.endswith("}"):
            info["mode"] = "raw_json"
            try:
                info["response_json"] = json.loads(s)
                info["response_json_ok"] = True
            except Exception:
                info["response_json_ok"] = False
            return info
        if "[function]" in s and "[response]" in s:
            info["mode"] = "wrapper"
            m = re.search(r"\[function\]\s*([^\n\r]+)", s)
            if m:
                info["tool_name"] = m.group(1).strip()
            m2 = re.search(r"\[response\]\s*(.*)\s*$", s, re.DOTALL)
            if m2:
                payload = m2.group(1).strip()
                if payload.startswith("{") and payload.endswith("}"):
                    try:
                        info["response_json"] = json.loads(payload)
                        info["response_json_ok"] = True
                    except Exception:
                        info["response_json_ok"] = False
            return info
        info["mode"] = "text"
        return info

    def _debug_trigger_summary(self, trig: Any) -> str:
        try:
            return jdump(trig, maxlen=1200)
        except Exception:
            return repr(trig)

    def _debug_step_summary(self, step_obj: Dict[str, Any], step_pos: int) -> str:
        idx = step_obj.get("index")
        subs = step_obj.get("substeps") or []
        return f"step_pos={step_pos} step.index={idx!r} substeps={len(subs) if isinstance(subs, list) else 'NONLIST'}"

    def _step_matches_trigger(self, trig_step: Any, step_pos: int, step_obj: Dict[str, Any]) -> bool:
        """
        Match a trigger.step_index against the current step.

        The dynamic-invariant generator is shown the IR with each step's
        ``index`` field (1-indexed) and emits ``trigger.step_index`` using
        that same numbering.  So when the step has an ``index`` field, that
        is the authoritative match — we MUST NOT also accept ``step_pos``
        (0-indexed) or ``step_pos + 1`` because that would make every
        trigger fire at two adjacent steps (off-by-one duplicates).

        Match semantics (in priority order):
          1. trigger in (None, "", "*")           -> always match
          2. step_obj has int ``index``           -> match iff trigger == index
          3. step_obj has no usable ``index``     -> fall back to step_pos
                                                     OR step_pos + 1 (legacy
                                                     IR without index field)
          4. trigger is a range "lo-hi"           -> apply same precedence
        """
        if trig_step in (None, "", "*"):
            return True

        step_index_field = safe_int(step_obj.get("index"))

        # Try simple integer match
        t = safe_int(trig_step)
        if t is not None:
            if step_index_field is not None:
                # Authoritative path: IR has 1-indexed `index`, only match
                # against it. No off-by-one fallback.
                return t == step_index_field
            # Legacy IR (no `index`): permissive 0-/1-indexed fallback
            return t == step_pos or t == step_pos + 1

        # Handle range triggers like "20-21", "15-20"
        trig_str = str(trig_step).strip()
        if "-" in trig_str:
            parts = trig_str.split("-", 1)
            lo = safe_int(parts[0].strip())
            hi = safe_int(parts[1].strip())
            if lo is not None and hi is not None:
                if step_index_field is not None:
                    return lo <= step_index_field <= hi
                # Legacy IR fallback
                return (lo <= step_pos <= hi) or (lo <= step_pos + 1 <= hi)

        return False

    def _should_check_invariant_with_debug(
        self,
        invariant: Dict[str, Any],
        traj: Dict[str, Any],
        step_pos: int,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:

        steps = traj.get("steps") or []
        if step_pos < 0 or step_pos >= len(steps):
            return False, f"step_pos out of range: {step_pos}", []

        trig = invariant.get("event_trigger") or {}
        event_types = trig.get("event_types", ["*"])

        if not isinstance(event_types, list) or not event_types:
            event_types = ["*"]
        if not isinstance(trig, dict):
            trig = {}

        step_obj = steps[step_pos]
        trig_step = trig.get("step_index", "*")
        if not self._step_matches_trigger(trig_step, step_pos, step_obj):
            if DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH] FAIL step_index mismatch for {inv_name(invariant)}")
                dbg(f"[MATCH]   trigger.step_index={trig_step!r}")
                dbg(f"[MATCH]   {self._debug_step_summary(step_obj, step_pos)}")
            return False, f"step_index mismatch (trigger={trig_step!r}, step_pos={step_pos}, step.index={step_obj.get('index')!r})", []

        subs = step_obj.get("substeps") or []
        if not isinstance(subs, list):
            if DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH] FAIL step.substeps not a list for {inv_name(invariant)} -> type={type(subs)}")
            return False, "step.substeps not a list", []

        sub_sel = trig.get("substep_index", "*")
        want_sub = None if sub_sel in (None, "", "*") else safe_int(sub_sel)

        tool_name = trig.get("tool_name", "*")
        role_name = trig.get("role_name", "*")
        content_pat = trig.get("content_regex", "*")

        role_re = None
        cont_re = None
        try:
            role_re = None if role_name in (None, "", "*") else re.compile(str(role_name))
        except Exception as e:
            if DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH] FAIL invalid role_name for {inv_name(invariant)} role_name={role_name!r} err={e}")
            return False, f"invalid role_name={role_name!r} ({e})", []
        try:
            cont_re = None if content_pat in (None, "", "*") else re.compile(str(content_pat), re.DOTALL)
        except Exception as e:
            if DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH] FAIL invalid content_regex for {inv_name(invariant)} content_regex={content_pat!r} err={e}")
            return False, f"invalid content_regex={content_pat!r} ({e})", []

        if DEBUG_MATCH and focus_inv(invariant):
            dbg(f"[MATCH] TRY {inv_name(invariant)} on {self._debug_step_summary(step_obj, step_pos)}")
            dbg(f"[MATCH]   trigger={self._debug_trigger_summary(trig)}")
            if tool_name not in (None, "", "*"):
                dbg("[MATCH]   NOTE: tool_name matching will be checked by presence in ss['content'] (NOT ss['tool_name']).")

        matched: List[Dict[str, Any]] = []
        for sub_pos, ss in enumerate(subs):
            if not isinstance(ss, dict):
                continue

            if want_sub is not None and sub_pos != want_sub:
                if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                    dbg(f"[MATCH]   sub_pos={sub_pos} skip: want_sub={want_sub}")
                continue

            role = ss.get("role") or ""
            content = ss.get("content") or ""

            if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH]   sub_pos={sub_pos} role={role!r} content={short(content)}")
                if DEBUG_TOOL_PARSE and role == "tool":
                    info = self._debug_parse_tool_wrapper(content)
                    dbg(f"[MATCH]     tool_wrapper_parse={jdump(info, maxlen=800)}")

            if "*" not in event_types:
                if role_name not in event_types:
                    if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                        dbg(f"[MATCH]     reject(event_types): event_types={event_types!r} role_name={role_name!r}")
                    continue
            
            if role_name not in (None, "", "*"):
                base = self._base_agent(role)
                if base != role_name and role != role_name:
                    if DEBUG_MATCH and focus_inv(invariant):
                        dbg(f"[CMP] role_name/compare :: trigger.role_name={role_name!r} == base(role)={base!r} => FAIL")
                        dbg(f"[CMP] role_name/compare :: trigger.role_name={role_name!r} == role={role!r} => FAIL")
                        dbg(f"[CMP] role_name/decision :: PASS if (base(role)==role_name) OR (role==role_name) => FAIL")
                    if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                        dbg(f"[MATCH]     reject(role_name): need={role_name!r} base(role)={base!r} role={role!r}")
                    continue
            if tool_name not in (None, "", "*"):
                need = str(tool_name)
                hay = "" if content is None else str(content)

                try:
                    tool_re = re.compile(need, re.IGNORECASE)
                    ok_present = bool(tool_re.search(hay))
                except re.error:
                    ok_present = (need in hay)

                if DEBUG_MATCH and focus_inv(invariant):
                    dbg(f"[CMP] tool_name_in_content/compare :: need={need!r} in content? => {ok_present} | content_excerpt={short(hay)}")

                if not ok_present:
                    if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                        dbg(f"[MATCH]     reject(tool_name_in_content): need={need!r} not found in content")
                    continue

            if role_re and not role_re.search(role):
                if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                    dbg(f"[MATCH]     reject(role_name): pattern={role_name!r} role={role!r}")
                continue

            if cont_re and not cont_re.search(content):
                if DEBUG_SUBSTEPS and DEBUG_MATCH and focus_inv(invariant):
                    dbg(f"[MATCH]     reject(content_regex): pattern={content_pat!r} content={short(content)}")
                continue

            matched.append(ss)

        if not matched:
            if DEBUG_MATCH and focus_inv(invariant):
                dbg(f"[MATCH] FAIL no substeps matched for {inv_name(invariant)} at step_pos={step_pos}")
                dbg(f"[MATCH]   trigger={self._debug_trigger_summary(trig)}")
                dbg(f"[MATCH]   step has {len(subs)} substeps")
            return False, "no substeps matched filters", []

        if DEBUG_MATCH and focus_inv(invariant):
            dbg(f"[MATCH] OK {inv_name(invariant)} matched {len(matched)} substeps at step_pos={step_pos}")
            if DEBUG_SUBSTEPS:
                for i, ss in enumerate(matched[:5]):
                    dbg(f"[MATCH]   matched[{i}] role={ss.get('role')!r} content={short(ss.get('content'))}")
                if len(matched) > 5:
                    dbg(f"[MATCH]   ...(matched truncated, total={len(matched)})")

        return True, "matched", matched
    # ---------------------------
    def _check_python_invariant(
        self,
        task_id: Union[int, str],
        traj: Dict[str, Any],
        step_pos: int,
        invariant: Dict[str, Any],
        matched_substeps: List[Dict[str, Any]],
    ) -> Optional[Violation]:
        self.total_python_checks += 1
        start = time.perf_counter()

        assertion_name = invariant.get("assertion_name") or "<missing>"
        python_check = invariant.get("python_check", {}) or {}

        code_lines = python_check.get("code_lines") or []
        function_name = python_check.get("function_name")

        if DEBUG_PY_EXEC and focus_inv(invariant):
            _line()
            dbg(f"[PY] START python_check assertion_name={assertion_name!r} step_pos={step_pos} step.index={(traj.get('steps') or [{}])[step_pos].get('index') if (traj.get('steps') or []) else None!r}")
            dbg(f"[PY] function_name={function_name!r} code_lines={len(code_lines)} matched_substeps={len(matched_substeps)}")
            if DEBUG_INV_DUMP:
                dbg(f"[PY] invariant dump:\n{jdump(invariant)}")

        if not code_lines or not function_name:
            end = time.perf_counter()
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="python_check",
                check_time_sec=round(end - start, 4),
                success=False,
                error="python_check missing code_lines or function_name",
                check_input={"step_pos": step_pos},
                check_output=None,
            ))
            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] SKIP missing code_lines or function_name (function_name={function_name!r}, code_lines={len(code_lines)})")
            return None

        try:
            code = "\n".join(code_lines)

            # If invariant expects policy text but we don't have it (flash/magentic), SKIP.
            needs_policy = ("policy_text" in code) or ("POLICY_TEXT" in code)
            if needs_policy and not (self.policy_text or "").strip():
                end = time.perf_counter()
                self.telemetry.append(CheckTelemetry(
                    task_id=task_id, step_index=step_pos, assertion_name=assertion_name,
                    check_type="python_check", check_time_sec=round(end - start, 4),
                    success=True, error=None,
                    check_input={"step_pos": step_pos, "skipped": True, "reason": "missing POLICY_TEXT"},
                    check_output={"skipped": True},
                ))
                self.skipped_invariants.append({"task_id": task_id, "step_pos": step_pos,
                                               "assertion_name": assertion_name,
                                               "reason": "missing POLICY_TEXT"})
                return None

            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] code preview:\n{code[:min(len(code), 1200)]}{'…(trunc)' if len(code) > 1200 else ''}")

            # Hacky: Provide an in-memory module so "from policy_text import POLICY_TEXT" works.
            m = types.ModuleType("policy_text")
            m.POLICY_TEXT = self.policy_text
            sys.modules["policy_text"] = m

            # Allow __import__ so generated checks can do 'import json', 'import re', etc.
            # We restrict to a safe allowlist of modules.
            _allowed_modules = {"json", "re", "math", "datetime", "collections", "itertools", "functools", "string"}
            _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
            def _safe_import(name, *args, **kwargs):
                if name.split(".")[0] not in _allowed_modules:
                    raise ImportError(f"Import of '{name}' is not allowed in invariant checks")
                return _real_import(name, *args, **kwargs)

            import collections as _collections
            import math as _math
            _safe_builtins = {
                "__import__": _safe_import,
                "str": str, "int": int, "float": float, "bool": bool,
                "dict": dict, "list": list, "tuple": tuple, "set": set,
                "len": len, "range": range, "enumerate": enumerate,
                "isinstance": isinstance, "type": type,
                "min": min, "max": max, "sum": sum, "sorted": sorted,
                "any": any, "all": all, "zip": zip, "map": map, "filter": filter,
                "abs": abs, "round": round, "reversed": reversed,
                "ValueError": ValueError, "TypeError": TypeError,
                "KeyError": KeyError, "IndexError": IndexError,
                "Exception": Exception, "True": True, "False": False, "None": None,
                "print": print,
            }
            glb = {
                "__builtins__": _safe_builtins,
                "POLICY_TEXT": self.policy_text,
                "json": json,
                "re": re,
                "math": _math,
                "collections": _collections,
            }
            loc: Dict[str, Any] = {}
   
            cap_out = io.StringIO()
            cap_err = io.StringIO()

            # Cross-platform wall-clock timeout around exec + fn invocation.
            #
            # We run the LLM-generated check on a daemon thread and join it
            # with a timeout. SIGALRM was previously used here but is Unix-only
            # and the AgentRx supported targets include Windows (paper Table
            # 1's WnW dataset is reproduced on Windows-WSL by collaborators).
            #
            # Trade-off (documented because Python lacks cooperative thread
            # cancellation): if the runner is still alive after the timeout,
            # it is abandoned and we proceed. daemon=True guarantees it does
            # not block process exit. The abandoned thread leaks until exit,
            # but cannot corrupt other checks because it writes only into the
            # captured `cap_out` buffer (the safe-builtins `print` is wrapped
            # below) — never into process-global sys.stdout/sys.stderr.
            _runner_box: Dict[str, Any] = {}
            _capture_streams = bool(DEBUG_PY_CAPTURE_STDOUT and DEBUG_PY_EXEC and focus_inv(invariant))
            if _capture_streams:
                _orig_print = _safe_builtins["print"]
                def _captured_print(*args, **kwargs):
                    kwargs.setdefault("file", cap_out)
                    return _orig_print(*args, **kwargs)
                _safe_builtins["print"] = _captured_print

            def _pycheck_runner() -> None:
                try:
                    exec(code, glb, loc)
                    if function_name not in loc:
                        _runner_box["status"] = "no_fn"
                        return
                    fn = loc[function_name]
                    if DEBUG_PY_EXEC and focus_inv(invariant):
                        # Direct stdout — runs only when capture is disabled.
                        dbg(f"[PY] calling {function_name}(traj, step_pos)")
                    _runner_box["result"] = fn(traj, step_pos)
                    _runner_box["status"] = "ok"
                except BaseException as exc:
                    _runner_box["status"] = "error"
                    _runner_box["error"] = exc
                    _runner_box["traceback"] = traceback.format_exc()

            _t = threading.Thread(target=_pycheck_runner, name=f"pycheck:{assertion_name}", daemon=True)
            _t.start()
            _t.join(timeout=PYCHECK_TIMEOUT_SEC)

            if _t.is_alive():
                raise PyCheckTimeout(
                    f"python_check '{assertion_name}' exceeded {PYCHECK_TIMEOUT_SEC}s"
                )

            status = _runner_box.get("status")
            if status == "error":
                # Re-raise inside the outer `try` so the existing telemetry
                # path records the exception identically to in-thread raises.
                raise _runner_box["error"]
            if status == "no_fn":
                end_time = time.perf_counter()

                if DEBUG_PY_EXEC and focus_inv(invariant):
                    dbg(f"[PY] ERROR function not found: {function_name!r}")
                    dbg(f"[PY] available locals: {sorted(list(loc.keys()))[:50]}")
                    if _capture_streams:
                        o = cap_out.getvalue()
                        e = cap_err.getvalue()
                        if o.strip():
                            dbg(f"[PY] captured stdout:\n{short(o, n=2000)}")
                        if e.strip():
                            dbg(f"[PY] captured stderr:\n{short(e, n=2000)}")

                self.telemetry.append(CheckTelemetry(
                    task_id=task_id,
                    step_index=step_pos,
                    assertion_name=assertion_name,
                    check_type="python_check",
                    check_time_sec=round(end_time - start, 4),
                    success=False,
                    error=f"Function '{function_name}' not found",
                    check_input={"function_name": function_name, "code_length": len(code)},
                    check_output=None
                ))
                return None

            result = _runner_box["result"]

            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] raw result type={type(result)} value={result!r}")
                if _capture_streams:
                    o = cap_out.getvalue()
                    e = cap_err.getvalue()
                    if o.strip():
                        dbg(f"[PY] captured stdout:\n{short(o, n=2000)}")
                    if e.strip():
                        dbg(f"[PY] captured stderr:\n{short(e, n=2000)}")

            end = time.perf_counter()
            current_event = matched_substeps[0] if matched_substeps else None

            check_input = {
                "step_pos": step_pos,
                "step_index": (traj.get("steps") or [{}])[step_pos].get("index"),
                "function_name": function_name,
                "matched_substeps_count": len(matched_substeps),
                "trajectory_length": len(traj.get("steps") or []),
            }
            if LOG_VERBOSE:
                check_input["code_lines"] = code_lines
                check_input["matched_substeps"] = matched_substeps

            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] telemetry.check_input={jdump(check_input, maxlen=1200)}")

            check_output = {"result": bool(result), "violated": (not bool(result))}
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="python_check",
                check_time_sec=round(end - start, 4),
                success=True,
                check_input=check_input,
                check_output=check_output,
            ))

            if not bool(result):
                if DEBUG_PY_EXEC and focus_inv(invariant):
                    dbg(f"[PY] VIOLATION (result evaluated False)")
                return Violation(
                    task_id=task_id,
                    step_index=step_pos,
                    assertion_name=assertion_name,
                    invariant_type=invariant.get("invariant_type") or "",
                    check_type="python_check",
                    severity=invariant.get("severity", "medium"),
                    check_hint=invariant.get("check_hint"),
                    evidence={"matched_substeps": matched_substeps, "current_event": current_event},
                    taxonomy_targets=invariant.get("taxonomy_targets"),
                )

            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] PASS")
            return None

        except Exception as e:
            end = time.perf_counter()

            tb = traceback.format_exc()
            if DEBUG_PY_EXEC and focus_inv(invariant):
                dbg(f"[PY] EXCEPTION: {e}")
                dbg(f"[PY] traceback:\n{tb}")
    
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="python_check",
                check_time_sec=round(end - start, 4),
                success=False,
                error=f"Exception: {str(e)} Traceback: {tb}",
                check_input={"step_pos": step_pos, "step_index": step_pos, "trajectory_length": len(traj.get("steps") or []), },
                check_output=None,
            ))
            return None

    def _format_trajectory_window(self, traj: Dict[str, Any], step_pos: int) -> str:
        steps = traj.get("steps") or []
        upto = min(step_pos, len(steps) - 1)
        chunks: List[str] = []
        for i in range(upto + 1):
            step = steps[i] or {}
            label = step.get("index")
            if not isinstance(label, int):
                label = i
            chunks.append(f"[STEP {label}]")
            subs = step.get("substeps") or []
            if isinstance(subs, list):
                for ss in subs:
                    if not isinstance(ss, dict):
                        continue
                    role = ss.get("role", "")
                    content = ss.get("content", "")
                    chunks.append(f"{role}: {content}")
            chunks.append("")
        return "\n".join(chunks).strip()

    def _check_nl_invariant(
        self,
        task_id: Union[int, str],
        traj: Dict[str, Any],
        step_pos: int,
        invariant: Dict[str, Any],
        matched_substeps: List[Dict[str, Any]],
    ) -> Optional[Violation]:
        self.total_nl_checks += 1
        start = time.perf_counter()

        assertion_name = invariant.get("assertion_name") or "<missing>"
        nl_check = invariant.get("nl_check", {}) or {}
        focus_steps_instruction = nl_check.get("focus_steps_instruction", "") or ""
        judge_scope_notes = nl_check.get("judge_scope_notes", "") or ""
        judge_rubric = nl_check.get("judge_rubric", []) or []
        judge_user_prompt_template = nl_check.get("judge_user_prompt_template") or ""

        rubric_algo_template = nl_check.get("rubric_evaluation_algorithm_template", "") or ""
        output_format_template = nl_check.get("output_format_template", "") or ""
        judge_system_prompt_template = nl_check.get("judge_system_prompt_template", "") or ""

        if rubric_algo_template:
            rubric_algo_template = rubric_algo_template.replace("{RUBRIC_EVALUATION_ALGORITHM}", RUBRIC_EVALUATION_ALGORITHM)
        if output_format_template:
            output_format_template = output_format_template.replace("{OUTPUT_FORMAT}", OUTPUT_FORMAT)
        judge_system_prompt = judge_system_prompt_template.replace("{NL_CHECK_JUDGE_SYSTEM_PROMPT}", NL_CHECK_JUDGE_SYSTEM_PROMPT)
        
        if self.llm_client is None:
            end = time.perf_counter()
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="nl_check",
                check_time_sec=round(end - start, 4),
                success=True,
                error=None,
                check_input={"skipped": True, "reason": "SKIP_NL=1"},
                check_output={"verdict": "pass", "note": "skipped nl_check"},
            ))
            return None

        window_text = self._format_trajectory_window(traj, step_pos)
        rule_text = invariant.get("check_hint") or ""
        rubric = nl_check.get("judge_rubric") or []
        rubric_text = "\n".join([f"- {x}" for x in rubric]) if rubric else ""
        rubric_text_user = "\n".join([f"- {x}" for x in judge_rubric]) if judge_rubric else ""
        enhanced_system_prompt = judge_system_prompt
        if judge_scope_notes:
            enhanced_system_prompt += f"\n\nScope: {judge_scope_notes}"
        if judge_rubric:
            rubric_text_sys = "\n".join([f"{i+1}. {c}" for i, c in enumerate(judge_rubric)])
            enhanced_system_prompt += f"\n\nEvaluation Rubric:\n{rubric_text_sys}"
        enhanced_system_prompt += "\n\n" + (rubric_algo_template or RUBRIC_EVALUATION_ALGORITHM)
        enhanced_system_prompt += "\n\n" + (output_format_template or OUTPUT_FORMAT)
        if focus_steps_instruction:
            enhanced_system_prompt += f"\n\nFocus Steps:\n{focus_steps_instruction}"

        replacements = {
            "{POLICY_TEXT}": self.policy_text,
            "{TASK_INSTRUCTION}": traj.get("instruction") or "",
            "{RULE_NATURAL_LANGUAGE}": rule_text,
            "{RUBRIC}": rubric_text_user,
            "{CURRENT_EVENT_JSON}": json.dumps({"step_pos": step_pos, "matched_substeps": matched_substeps}, indent=2),
            "{WINDOW_EVENTS_JSON}": json.dumps({"trajectory_id": traj.get("trajectory_id"), "steps": (traj.get("steps") or [])[: step_pos + 1]}, indent=2),
            "{trajectory_till_current_step}": window_text,
        }

        user_prompt = judge_user_prompt_template
        for k, v in replacements.items():
            user_prompt = user_prompt.replace(k, v)

        
        if focus_inv(invariant):
            _line()
            dbg(f"[NL] START nl_check assertion_name={assertion_name!r} step_pos={step_pos} step.index={(traj.get('steps') or [{}])[step_pos].get('index') if (traj.get('steps') or []) else None!r}")
            dbg(f"[NL] has_scope_notes={bool(judge_scope_notes)} has_rubric={bool(judge_rubric)} has_focus_steps_instruction={bool(focus_steps_instruction)}")
            dbg(f"[NL] system_prompt:\n{enhanced_system_prompt}")
            dbg(f"[NL] user_prompt:\n{user_prompt}")
            if DEBUG_INV_DUMP:
                dbg(f"[NL] invariant dump:\n{jdump(invariant)}")

        raw = ""  # allow exception handler to report raw excerpt if available

        try:
            from openai import RateLimitError as _RLE
            _check_retries = 5
            for _attempt in range(_check_retries):
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        #temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    break
                except _RLE:
                    if _attempt < _check_retries - 1:
                        _wait = 4 * (2 ** _attempt)
                        print(f"[RATE LIMIT] Checker retry in {_wait}s ({_attempt+1}/{_check_retries})", flush=True)
                        import time as _time; _time.sleep(_wait)
                    else:
                        raise
            end = time.perf_counter()

            tokens_used = 0
            if hasattr(response, "usage") and response.usage is not None:
                tokens_used = int(getattr(response.usage, "total_tokens", 0) or 0)
                self.total_tokens_used += tokens_used

            raw = response.choices[0].message.content or ""

            
            if DEBUG_NL_PROMPTS and focus_inv(invariant):
                dbg(f"[NL] raw response:\n{raw[:min(len(raw), 2500)]}{'…(trunc)' if len(raw) > 2500 else ''}")
                dbg("[NL] NOTE: this function parses judge_result twice in the original code. If raw is not JSON, the second json.loads(raw) will throw and land in the outer except.")
    
            try:
                judge_result = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                # Gold-style fallback: treat as non-JSON response
                low = raw.lower()
                violated_fallback = ("fail" in low) or ("violat" in low)
                judge_result = {
                    "verdict": "fail" if violated_fallback else "pass",
                    "parse_error": "JSONDecodeError",
                    "raw_excerpt": raw[:500],
                }
            judge_result = json.loads(raw)

            verdict = str(judge_result.get("verdict", "")).strip().lower()
            violated = (verdict == "fail")

            check_input = {
                "step_pos": step_pos,
                "step.index": (traj.get("steps") or [{}])[step_pos].get("index"),
                "model": self.model_name,
                "temperature": 0.0,
                "window_steps": step_pos + 1,
                "matched_substeps_count": len(matched_substeps),
                "has_scope_notes": bool(judge_scope_notes),
                "has_rubric": bool(judge_rubric),
                "has_focus_steps_instruction": bool(focus_steps_instruction),
                "uses_standard_templates": True,
            }
            if LOG_VERBOSE:
                check_input["system_prompt"] = enhanced_system_prompt
                check_input["user_prompt"] = user_prompt

            check_output = {
                "verdict": verdict,
                "violated": violated,
                "judge_result": judge_result,
            }

            
            if DEBUG_NL_PROMPTS and focus_inv(invariant):
                dbg(f"[NL] verdict={verdict!r} violated={violated} tokens_used={tokens_used}")
                dbg(f"[NL] telemetry.check_input={jdump(check_input, maxlen=1200)}")
                dbg(f"[NL] telemetry.check_output={jdump(check_output, maxlen=1200)}")
    
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="nl_check",
                check_time_sec=round(end - start, 4),
                tokens_used=tokens_used,
                success=True,
                check_input=check_input,
                check_output=check_output,
            ))

            if violated:
                step_obj = (traj.get("steps") or [{}])[step_pos] or {}
                return Violation(
                    task_id=task_id,
                    step_index=step_pos,
                    assertion_name=assertion_name,
                    invariant_type=invariant.get("invariant_type") or "",
                    check_type="nl_check",
                    severity=invariant.get("severity", "medium"),
                    check_hint=invariant.get("check_hint"),
                    evidence={"step_pos": step_pos,
                        "step_index": step_obj.get("index"), 
                        "matched_substeps": matched_substeps, 
                        "judge_response": judge_result},
                    taxonomy_targets=invariant.get("taxonomy_targets"),
                )
            return None

        except Exception as e:
            end = time.perf_counter()
            if "context_length_exceeded" in str(e):
                raise RuntimeError(
                    f"NL check prompt too large for the model's context window. "
                    f"Use a model with a larger context limit or set SKIP_NL=1.\n{e}"
                )

            tb = traceback.format_exc()
            if DEBUG_NL_PROMPTS and focus_inv(invariant):
                dbg(f"[NL] EXCEPTION: {e}")
                if raw:
                    dbg(f"[NL] raw excerpt:\n{raw[:min(len(raw), 2000)]}{'…(trunc)' if len(raw) > 2000 else ''}")
                dbg(f"[NL] traceback:\n{tb}")
    
            self.telemetry.append(CheckTelemetry(
                task_id=task_id,
                step_index=step_pos,
                assertion_name=assertion_name,
                check_type="nl_check",
                check_time_sec=round(end - start, 4),
                tokens_used=0,
                success=False,
                error=f"Exception: {str(e)} Traceback: {tb}",
                check_input={"step_pos": step_pos, "step_index": step_pos},
                check_output=None,
            ))
            return None

    # ---------------------------
    # PUBLIC API
    # ---------------------------
    def verify_trajectory(self, task_id: Union[int, str], traj: Dict[str, Any]) -> List[Violation]:
        violations: List[Violation] = []
        steps = traj.get("steps") or []
        dbg(f"verify_trajectory: task_id={task_id} steps={len(steps)}")

        for step_pos in range(len(steps)):
            violations.extend(self.verify_trajectory_step(task_id, traj, step_pos))
        return violations

    def verify_trajectory_step(self, task_id: Union[int, str], traj: Dict[str, Any], step_pos: int) -> List[Violation]:
        violations: List[Violation] = []
        skipped_invariants = []  # Track which invariants were skipped
        steps = traj.get("steps") or []
        step_obj = steps[step_pos] if 0 <= step_pos < len(steps) else {}
        dbg(f"--- step_pos={step_pos} step.index={step_obj.get('index')!r} ---")

        applied = 0
        for inv in self.invariants:
            if not focus_inv(inv):
                continue
            ok, reason, matched_subs = self._should_check_invariant_with_debug(inv, traj, step_pos)
            if not ok:
                if DEBUG and ("step_index mismatch" in reason or "no substeps" in reason):
                    # keep this relatively focused to avoid insane spam
                    dbg(f"skip {inv.get('assertion_name')} -> {reason}")
                
                if DEBUG_MATCH:
                    ct_field = inv.get("check_type")
                    inferred = self._infer_check_type(inv)
                    trig = inv.get("event_trigger")
                    dbg(f"[SKIP] {inv_name(inv)} reason={reason} check_type_field={ct_field!r} inferred={inferred!r}")
                    if trig is not None:
                        dbg(f"[SKIP] trigger={self._debug_trigger_summary(trig)}")
        
                skipped_invariants.append({
                    "assertion_name": inv.get("assertion_name"),
                    "reason": reason,
                })
                continue

            applied += 1
            ct = inv.get("check_type")
            inferred_ct = self._infer_check_type(inv)
            if DEBUG_APPLY:
                _line()
                dbg(f"[APPLY] {inv_name(inv)} step_pos={step_pos} step.index={step_obj.get('index')!r}")
                dbg(f"[APPLY] check_type field={ct!r} inferred={inferred_ct!r} (NOTE: code uses field-only unless your invariant set check_type)")
                dbg(f"[APPLY] matched_substeps={len(matched_subs)} first_role={matched_subs[0].get('role') if matched_subs else None!r}")
                if DEBUG_INV_DUMP:
                    dbg(f"[APPLY] invariant dump:\n{jdump(inv)}")
                else:
                    dbg(f"[APPLY] invariant keys={sorted(list(inv.keys()))}")
                    if inv.get("event_trigger") is not None:
                        dbg(f"[APPLY] event_trigger={jdump(inv.get('event_trigger'), maxlen=1200)}")
    
            dbg(f"apply {inv.get('assertion_name')} check_type={ct} matched_substeps={len(matched_subs)}")

            vio: Optional[Violation] = None
            if ct == "python_check":
                vio = self._check_python_invariant(task_id, traj, step_pos, inv, matched_subs)
            elif ct == "nl_check":
                vio = self._check_nl_invariant(task_id, traj, step_pos, inv, matched_subs)
            else:
                
                if DEBUG_APPLY:
                    dbg(f"[APPLY] skip {inv_name(inv)} -> unknown check_type={ct!r}. inferred={inferred_ct!r}. This is a common source of 'nothing runs'.")
                self.skipped_invariants.append({
                    "task_id": task_id,
                    "step_pos": step_pos,
                    "assertion_name": inv.get("assertion_name"),
                    "reason": f"unknown check_type={ct!r}",
                })
                dbg(f"skip {inv.get('assertion_name')} -> unknown check_type={ct!r}")

            if vio:
                dbg(f"VIOLATION: {vio.assertion_name} at step_pos={step_pos}")
                violations.append(vio)

        if applied == 0:
            self.skipped_steps.append({
                "task_id": task_id,
                "step_pos": step_pos,
                "step.index": step_obj.get("index"),
                "reason": "no applicable invariants",
                "skipped_invariants": skipped_invariants,
            })
            
            if DEBUG_MATCH:
                dbg(f"[STEP] no applicable invariants for step_pos={step_pos} step.index={step_obj.get('index')!r}")
                if skipped_invariants:
                    dbg(f"[STEP] skipped_invariants count={len(skipped_invariants)} sample={jdump(skipped_invariants[:5], maxlen=1200)}")
    
            dbg("no applicable invariants for this step")

        return violations

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "total_python_checks": self.total_python_checks,
            "total_nl_checks": self.total_nl_checks,
            "total_tokens_used": self.total_tokens_used,
            "telemetry_count": len(self.telemetry),
            "skipped_steps": len(self.skipped_steps),
            "skipped_invariants": len(self.skipped_invariants),
        }

def main():
    parser = argparse.ArgumentParser(description="Invariant Checker for trajectory verification")
    endpoint_grp = parser.add_mutually_exclusive_group()
    endpoint_grp.add_argument(
        "--azure",
        action="store_const",
        const="azure",
        dest="client",
        help="Use Azure OpenAI client (default)"
    )
    endpoint_grp.add_argument(
        "--trapi",
        action="store_const",
        const="trapi",
        dest="client",
        help="Use TRAPI client"
    )
    parser.set_defaults(client=g.DEFAULT_ENDPOINT)
    parser.add_argument("--domain", type=str, default="flash",
                        choices=["flash", "tau", "magentic"],
                        help="Domain to run (default: flash)")
    parser.add_argument("--input-path", type=str, default=None,
                        help="Path to input trajectory file or directory (default: auto per domain)")
    parser.add_argument("--static-invariants", type=str, default=None,
                        help="Path to static invariants JSON file")
    parser.add_argument("--dynamic-invariants-dir", type=str, default=None,
                        help="Directory containing per-trajectory dynamic invariant files (out_<id>.json)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Base output directory for results (default: out/checker_results)")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to policy document (only needed for tau domain)")
    args = parser.parse_args()

    domain = args.domain
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve static invariants path
    static_invariants_path = args.static_invariants or os.getenv(
        "STATIC_INVARIANTS_PATH", f"out/static_invariants_{domain}.json")
    print(f"Using static_invariants_path={static_invariants_path}")

    # Resolve policy document path
    if args.policy_path:
        policy_document_path = args.policy_path
    elif domain == "tau":
        policy_document_path = os.path.join(script_dir, "policy_documents", "retail_policy.txt")
    else:
        policy_document_path = ""

    # Resolve input path
    input_path = args.input_path
    if input_path is None:
        if domain == "tau":
            input_path = "tau_dataset_test.json"
        elif domain == "magentic":
            input_path = "magentic_dataset"
        else:
            input_path = "flash_dataset"

    # Resolve base output directory
    base_out_dir = args.out_dir or os.path.join("out", "checker_results")

    # Resolve dynamic invariants directory
    dynamic_invariants_dir = args.dynamic_invariants_dir or "dynamic_invariant_outputs"

    # Load trajectories based on domain
    if domain == "tau":
        data = tau_bench_ir(load_trajectories(input_path))
        trajectories = data if isinstance(data, list) else [data]
        golden_task_ids = [2, 3, 20, 34, 47, 72, 74]
    elif domain in ("magentic", "flash"):
        trajectories = []
        ir_fn = magentic_ir if domain == "magentic" else flash_ir
        inp = os.path.abspath(input_path) if not os.path.isabs(input_path) else input_path
        if os.path.isdir(inp):
            for root, _, files in os.walk(inp):
                for fn in files:
                    if not fn.endswith(".json") and not fn.endswith(".jsonl"):
                        continue
                    fp = os.path.join(root, fn)
                    data = ir_fn(load_trajectories(fp))
                    if isinstance(data, list):
                        trajectories.extend(data)
                    else:
                        trajectories.append(data)
        else:
            data = ir_fn(load_trajectories(inp))
            trajectories = data if isinstance(data, list) else [data]
        golden_task_ids = []
    else:
        trajectories = []
        golden_task_ids = []

    # Initialize the static verifier (same static verifier can be used for all the trajectories)
    static_verifier = AllVerifier(invariants_path=static_invariants_path, policy_document_path=policy_document_path, client=args.client)
    golden_set = {str(x) for x in golden_task_ids}

    for i, traj in enumerate(trajectories):

        raw_id = traj.get("trajectory_id") or traj.get("task_id")
        if domain == "tau":
            task_id: Union[int, str] = safe_int(raw_id)
            if task_id is None:
                task_id = i
        else:
            task_id = str(raw_id) if raw_id is not None else str(i)
        print(f"task id is {task_id}")
        # Initialize all the paths for output files
        out_dir = os.path.join(base_out_dir, f"{task_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"violations_{domain}.json")
        telemetry_path = os.path.join(out_dir, f"telemetry_{domain}.json")
        debug_path = os.path.join(out_dir, f"debug_skips_{domain}.json")

        # Initialize the metrics
        all_violations: List[Dict[str, Any]] = []
        all_telemetry: List[Dict[str, Any]] = []
        all_skipped_steps: List[Dict[str, Any]] = []
        all_skipped_invariants: List[Dict[str, Any]] = []
        total_metrics = {
            "total_python_checks": 0,
            "total_nl_checks": 0,
            "total_tokens_used": 0,
            "telemetry_count": 0,
            "skipped_steps": 0,
            "skipped_invariants": 0,
        }

        # Initialize the dynamic verifier if dynamic invariants for that trajectory exist, else skip dynamic invariant verification
        # Try both step-by-step (out_{id}.json) and one-shot (out_{id}_oneshot.json) filenames.
        dynamic_invariants_file = None
        for candidate in (
            os.path.join(dynamic_invariants_dir, f"out_{task_id}.json"),
            os.path.join(dynamic_invariants_dir, f"out_{task_id}_oneshot.json"),
        ):
            if os.path.exists(candidate):
                dynamic_invariants_file = candidate
                break
        dynamic_verifier = None
        if dynamic_invariants_file:
            print(f"Using dynamic invariants file: {dynamic_invariants_file} for task_id={task_id}")
            dynamic_verifier = AllVerifier(invariants_path=dynamic_invariants_file, policy_document_path=policy_document_path, client=args.client)
        else:
            dbg(f"Dynamic invariants file not found for task_id={task_id} in {dynamic_invariants_dir}, skipping dynamic verification")

        dbg(f"=== TRAJ {i} task_id={task_id} trajectory_id={traj.get('trajectory_id')!r} ===")

        if DEBUG:
            steps = traj.get("steps") or []
            dbg(f"[TRAJ] steps={len(steps)} top_keys={sorted(list(traj.keys()))}")
            if steps:
                dbg(f"[TRAJ] step[0] keys={sorted(list((steps[0] or {}).keys()))} substeps={(steps[0] or {}).get('substeps') and len((steps[0] or {}).get('substeps'))}")

        static_violations = static_verifier.verify_trajectory(task_id=task_id, traj=traj)
        all_violations.extend([v.to_dict() for v in static_violations])
        all_telemetry.extend([t.to_dict() for t in static_verifier.telemetry])
        all_skipped_steps.extend(static_verifier.skipped_steps)
        all_skipped_invariants.extend(static_verifier.skipped_invariants)
        
        # Aggregate static verifier metrics
        static_metrics = static_verifier.get_metrics_summary()
        for key in total_metrics:
            total_metrics[key] += static_metrics.get(key, 0)

        if dynamic_verifier:
            dynamic_violations = dynamic_verifier.verify_trajectory(task_id=task_id, traj=traj)
            all_violations.extend([v.to_dict() for v in dynamic_violations])
            all_telemetry.extend([t.to_dict() for t in dynamic_verifier.telemetry])
            all_skipped_steps.extend(dynamic_verifier.skipped_steps)
            all_skipped_invariants.extend(dynamic_verifier.skipped_invariants)
            
            # Aggregate dynamic verifier metrics
            dynamic_metrics = dynamic_verifier.get_metrics_summary()
            for key in total_metrics:
                total_metrics[key] += dynamic_metrics.get(key, 0)

        # Sort violations by step_index before writing
        all_violations.sort(key=lambda v: v.get("step_index", 0))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_violations, f, indent=2)

        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(all_telemetry, f, indent=2)

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "skipped_steps": all_skipped_steps,
                "skipped_invariants": all_skipped_invariants,
                "metrics": total_metrics,
            }, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
