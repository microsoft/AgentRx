#!/bin/bash
# Magentic*-27 sweep — full pipeline with n=1 judge sample per trajectory at
# the release default (paper-faithful PROMPT_STYLE, NL violations included).
#
# Inputs : data/magentic_dataset/<trajectory_id>.json (27 IDs listed in
#          data/ground_truth/magentic_star_ids.json)
# Outputs: runs/azure_m27_<trajectory_id>/judge_output/runs/run1.json
# Idempotent: skips trajectories whose run1.json already exists.
#
# Usage (from repo root):
#   scripts/sweeps/run_magentic27.sh
#   nohup scripts/sweeps/run_magentic27.sh > runs/azure_magentic27.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${AGENTRX_PYTHON:-python}"
LOG="${AGENTRX_LOG:-runs/azure_magentic27.log}"
GT="data/ground_truth/magentic_one_ground_truth.json"

mkdir -p "$(dirname "$LOG")"

IDS=$($PY -c "import json; [print(e['trajectory_id']) for e in json.load(open('data/ground_truth/magentic_star_ids.json'))['ids']]")
TOTAL=$(echo "$IDS" | wc -l | tr -d ' ')
i=0
START_EPOCH=$(date +%s)

echo "==== Magentic*-27 sweep started @ $(date -Iseconds) (TOTAL=$TOTAL) ====" | tee -a "$LOG"

for tid in $IDS; do
  i=$((i+1))
  RUN_NAME="azure_m27_${tid}"
  RUN_DIR="runs/$RUN_NAME"
  if [ -f "$RUN_DIR/judge_output/runs/run1.json" ]; then
    echo "[$i/$TOTAL] SKIP (already done) $tid" | tee -a "$LOG"
    continue
  fi
  T0=$(date +%s)
  echo "[$i/$TOTAL] START $tid @ $(date +%H:%M:%S)" | tee -a "$LOG"
  $PY run.py \
    "data/magentic_dataset/${tid}.json" \
    --domain magentic \
    --endpoint azure \
    --num-runs 1 \
    --ground-truth "$GT" \
    --run-name "$RUN_NAME" \
    >> "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  TOTAL_ELAPSED=$((T1 - START_EPOCH))
  echo "[$i/$TOTAL] DONE  $tid rc=$RC ${ELAPSED}s (cum ${TOTAL_ELAPSED}s)" | tee -a "$LOG"
done

echo "==== Magentic*-27 sweep finished @ $(date -Iseconds) ====" | tee -a "$LOG"
