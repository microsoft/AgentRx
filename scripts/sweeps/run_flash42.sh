#!/bin/bash
# Flash-42 sweep — full pipeline with n=3 judge samples per trajectory at the
# release default (paper-faithful PROMPT_STYLE, NL violations included).
#
# Inputs : data/flash_dataset/<trajectory_id>.jsonl (42 files, internal data)
# Outputs: runs/azure_flash42_<trajectory_id>/judge_output/runs/run{1,2,3}.json
# Idempotent: skips trajectories whose run3.json already exists.
#
# Usage (from repo root):
#   scripts/sweeps/run_flash42.sh
#   nohup scripts/sweeps/run_flash42.sh > runs/azure_flash42.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${AGENTRX_PYTHON:-python}"
LOG="${AGENTRX_LOG:-runs/azure_flash42.log}"
GT="data/ground_truth/flash_ground_truth.json"

mkdir -p "$(dirname "$LOG")"

IDS=$(ls data/flash_dataset/*.jsonl | xargs -n1 basename | sed 's/.jsonl$//' | sort)
TOTAL=$(echo "$IDS" | wc -l | tr -d ' ')
i=0
START_EPOCH=$(date +%s)

echo "==== Flash-42 sweep started @ $(date -Iseconds) (TOTAL=$TOTAL) ====" | tee -a "$LOG"

for tid in $IDS; do
  i=$((i+1))
  RUN_NAME="azure_flash42_${tid}"
  RUN_DIR="runs/$RUN_NAME"
  if [ -f "$RUN_DIR/judge_output/runs/run3.json" ]; then
    echo "[$i/$TOTAL] SKIP (already done) $tid" | tee -a "$LOG"
    continue
  fi
  T0=$(date +%s)
  echo "[$i/$TOTAL] START $tid @ $(date +%H:%M:%S)" | tee -a "$LOG"
  $PY run.py \
    "data/flash_dataset/${tid}.jsonl" \
    --domain flash \
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

echo "==== Flash-42 sweep finished @ $(date -Iseconds) ====" | tee -a "$LOG"
