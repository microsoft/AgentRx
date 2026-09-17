"""analysis() must never reuse a stale on-disk runK.json (commit 1a26e6e).

Earlier versions of analysis() read back the pre-existing output file and
silently merged its ``detailed_results`` into the on-disk payload. That made
the JSON disagree with the printed summary whenever an AAD-interrupted write,
a stale skeleton, or a previous iteration's file was already on disk. The fix
declared the freshly computed ``data`` argument the single source of truth.

This test fences that contract.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def judge():
    import agentrx.judge.judge as j
    return j


def _make_task_record(*, tid: str, gt_cat: str, gt_step: int,
                      pred_cat: str, pred_step: float,
                      traj_len: int = 10) -> dict:
    """Minimal payload matching what compute_stats produces."""
    return {
        "task_id": tid,
        "gt_failure_case": gt_cat,
        "gt_step_number": gt_step,
        "gt_step_numbers": [gt_step],
        "most_common_failure": pred_cat,
        "step_mean": pred_step,
        "trajectory_length": traj_len,
        "llm_call_telemetry": {
            "tokens": {"prompt_tokens": 100, "output_tokens": 50},
            "time": {"execution_time_sec": 1.25},
        },
    }


def test_analysis_overwrites_stale_detailed_results(judge, tmp_path):
    """A bogus pre-existing run.json must NOT bleed into the fresh write."""
    out_path = tmp_path / "run1.json"

    # Stale on-disk artifact with completely bogus content.
    bogus = {
        "summary": {"this": "is", "fake": True},
        "detailed_results": [
            {"task_id": "BOGUS", "gt_failure_case": "999", "most_common_failure": "999",
             "step_mean": -1, "gt_step_number": -1, "gt_step_numbers": [-1],
             "trajectory_length": 1, "llm_call_telemetry": None},
        ],
    }
    out_path.write_text(json.dumps(bogus))

    fresh = [
        _make_task_record(tid="T1", gt_cat="3", gt_step=4, pred_cat="3", pred_step=4.0),
        _make_task_record(tid="T2", gt_cat="5", gt_step=2, pred_cat="6", pred_step=3.0),
    ]
    judge.analysis(fresh, output_file_path=str(out_path),
                   model_name="test-model", api_version="test-version")

    written = json.loads(out_path.read_text())
    written_tids = sorted(r["task_id"] for r in written["detailed_results"])
    assert written_tids == ["T1", "T2"], "stale detailed_results bled into fresh write"
    assert "BOGUS" not in written_tids
    # Summary must reflect the fresh data: 1 correct out of 2.
    assert written["summary"]["Correct cases"] == 1
    assert written["summary"]["Incorrect cases"] == 1
    # Token totals roll up from fresh telemetry (100*2 = 200 prompt, 50*2 = 100 out).
    assert written["summary"]["total_prompt_tokens"] == 200
    assert written["summary"]["total_output_tokens"] == 100


def test_analysis_writes_when_file_absent(judge, tmp_path):
    out_path = tmp_path / "subdir" / "run1.json"
    out_path.parent.mkdir(parents=True)
    fresh = [_make_task_record(tid="T1", gt_cat="3", gt_step=4, pred_cat="3", pred_step=4.0)]
    judge.analysis(fresh, output_file_path=str(out_path),
                   model_name="m", api_version="v")
    written = json.loads(out_path.read_text())
    assert written["detailed_results"][0]["task_id"] == "T1"
    assert written["summary"]["Correct cases"] == 1
