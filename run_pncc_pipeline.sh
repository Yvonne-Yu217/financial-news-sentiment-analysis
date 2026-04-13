#!/bin/bash
# Auto-pipeline: waits for script 9 pncc to finish, then runs 10, 13, regression
set -e
cd /Volumes/Data_Drive/research0322226/financial-news-sentiment-analysis

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

SCRIPT9_PID=18414
log "Pipeline started. Waiting for script 9 (PID $SCRIPT9_PID) to complete..."
while kill -0 $SCRIPT9_PID 2>/dev/null; do
    sleep 30
done
log "Script 9 done. Checking checkpoint..."
python3 -c "
import json
ckpt = json.load(open('run_artifacts/9sentiment_analyzer_pncc/sentiment_checkpoint_pncc.json'))
years = ckpt.get('years', {})
done = [y for y,v in years.items() if v.get('completed')]
pending = [y for y,v in years.items() if not v.get('completed')]
print(f'Completed years: {sorted(done)}')
print(f'Pending: {sorted(pending)}')
"

log "Running script 10 (PCNN -> results/weighted_photopes.csv)..."
python3 10calculate_daily_photopes.py \
    --collection-suffix _pncc \
    --output results/weighted_photopes.csv \
    --plot \
    --plot-dir plots \
    > run_artifacts/10photopes_pncc.log 2>&1
log "Script 10 done."

log "Running script 13 (merge + returns)..."
python3 13merge_data_and_calculate_returns.py \
    > run_artifacts/13merge_pncc.log 2>&1
log "Script 13 done."

log "Running regression_tables.py..."
python3 regression_tables.py \
    > run_artifacts/regression_pncc.log 2>&1
log "Regression done. All results in results/regression_output/"

log "=== Pipeline complete ==="
