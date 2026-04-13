#!/bin/bash

# Configuration
START_YEAR=2021
END_YEAR=2026
DELAY_HOURS=3
ARTIFACT_DIR="run_artifacts/3image_full_2021_2026"
LOG_FILE="$ARTIFACT_DIR/delayed_launch.log"

# Create artifact directory
mkdir -p "$ARTIFACT_DIR"

echo "Current time: $(date)" > "$LOG_FILE"
echo "Waiting $DELAY_HOURS hours before starting image downloading for $START_YEAR-$END_YEAR..." >> "$LOG_FILE"

# Wait for 3 hours (3 * 3600 seconds)
sleep $((DELAY_HOURS * 3600))

echo "Wait complete. Starting 3image_downlowder.py at $(date)" >> "$LOG_FILE"

# Activate venv and run the script
source /Volumes/Data_Drive/research0322226/.venv/bin/activate

nohup python 3image_downlowder.py \
    --start-year $START_YEAR \
    --end-year $END_YEAR \
    --batch-size 10 \
    --semaphore 24 \
    --artifact-dir "$ARTIFACT_DIR" \
    >> "$ARTIFACT_DIR/nohup.out" 2>&1 &

echo "Process started with PID $! at $(date)" >> "$LOG_FILE"
