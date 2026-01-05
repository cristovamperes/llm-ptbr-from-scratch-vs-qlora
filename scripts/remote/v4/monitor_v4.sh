#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
cd /root/TCC_v4 2>/dev/null || true
LOG="versions/v4-subword-lstm/logs/monitor_v4.log"
LOG_DIR="versions/v4-subword-lstm/logs"
PATTERN="train_char_model_brwac.py --arch v4"

latest_batch_log() {
  ls -1t "${LOG_DIR}"/batches_*.log 2>/dev/null | head -n1
}

: > "$LOG"
printf "%s monitor started\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"

while pgrep -f "$PATTERN" > /dev/null; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"
  LATEST_LOG="$(latest_batch_log)"
  if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    LAST="$(tail -n1 "$LATEST_LOG" 2>/dev/null || echo 'batch log empty')"
  else
    LAST="no batch log yet"
  fi
  printf "%s training running: %s\n" "$TS" "$LAST" >> "$LOG"
  sleep 1800
done

TS="$(date '+%Y-%m-%d %H:%M:%S')"
printf "%s training finished\n" "$TS" >> "$LOG"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
{
  printf "%s running subword evaluation\n" "$(date '+%Y-%m-%d %H:%M:%S')"
  bash scripts/run_evaluate_subword_models.sh
  printf "%s running subword samples\n" "$(date '+%Y-%m-%d %H:%M:%S')"
  MODEL_KEYS=v4_brwac_subword OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4.json python analysis/generate_samples.py
  printf "%s post steps done\n" "$(date '+%Y-%m-%d %H:%M:%S')"
} >> "$LOG" 2>&1
