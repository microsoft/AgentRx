#!/bin/bash
# Tau-29 sweep — full pipeline (ir/static/dynamic/check/judge) with n=3 judge
# samples per trajectory at the release default (paper-faithful PROMPT_STYLE,
# NL violations included).
#
# Inputs : data/tau_dataset/<task_id>.json (29 task IDs)
# Outputs: runs/azure_tau29_<task_id>/judge_output/runs/run{1,2,3}.json
# Idempotent: skips trajectories whose run3.json already exists.
#
# Usage (from repo root):
#   scripts/sweeps/run_tau29.sh
#   nohup scripts/sweeps/run_tau29.sh > runs/azure_tau29.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${AGENTRX_PYTHON:-python}"
LOG="${AGENTRX_LOG:-runs/azure_tau29.log}"
GT="data/ground_truth/tau_ground_truth.json"

mkdir -p "$(dirname "$LOG")"

IDS=$(ls data/tau_dataset/*.json | xargs -n1 basename | sed 's/.json$//' | sort -n)
TOTAL=$(echo "$IDS" | wc -l | tr -d ' ')
i=0
START_EPOCH=$(date +%s)

echo "==== Tau-29 sweep started @ $(date -Iseconds) (TOTAL=$TOTAL) ====" | tee -a "$LOG"

for tid in $IDS; do
  i=$((i+1))
  RUN_NAME="azure_tau29_${tid}"
  RUN_DIR="runs/$RUN_NAME"
  if [ -f "$RUN_DIR/judge_output/runs/run3.json" ]; then
    echo "[$i/$TOTAL] SKIP (already done) $tid" | tee -a "$LOG"
    continue
  fi
  T0=$(date +%s)
  echo "[$i/$TOTAL] START $tid @ $(date +%H:%M:%S)" | tee -a "$LOG"
  $PY run.py \
    "data/tau_dataset/$tid.json" \
    --domain tau \
    --endpoint azure \
    --ground-truth "$GT" \
    --run-name "$RUN_NAME" \
    >> "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  TOTAL_ELAPSED=$((T1 - START_EPOCH))
  echo "[$i/$TOTAL] DONE  $tid rc=$RC ${ELAPSED}s (cum ${TOTAL_ELAPSED}s)" | tee -a "$LOG"
done

echo "==== Tau-29 sweep finished @ $(date -Iseconds) ====" | tee -a "$LOG"
