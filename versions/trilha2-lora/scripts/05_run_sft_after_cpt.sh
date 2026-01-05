#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOGS_DIR="versions/trilha2-lora/logs"
mkdir -p "$LOGS_DIR"

CPT_PID_FILE="$LOGS_DIR/train_cpt_qlora.pid"
CPT_OUT_FILE="$LOGS_DIR/train_cpt_qlora.out"

SFT_PID_FILE="$LOGS_DIR/train_sft_qlora.pid"
SFT_OUT_FILE="$LOGS_DIR/train_sft_qlora.out"

echo "[INFO] Repo root: $REPO_ROOT"

if [[ ! -f "$CPT_PID_FILE" ]]; then
  echo "[ERROR] CPT PID file not found: $CPT_PID_FILE"
  echo "[ERROR] Start CPT first (02_train_cpt_qlora.py) and create the pid file."
  exit 1
fi

CPT_PID="$(cat "$CPT_PID_FILE" | tr -d '[:space:]' || true)"
if [[ -z "$CPT_PID" ]]; then
  echo "[ERROR] CPT PID file is empty: $CPT_PID_FILE"
  exit 1
fi

echo "[INFO] Waiting for CPT to finish (pid=$CPT_PID)..."
while kill -0 "$CPT_PID" 2>/dev/null; do
  echo "[$(date -Is)] CPT still running (pid=$CPT_PID). Last log line:"
  tail -n 1 "$CPT_OUT_FILE" 2>/dev/null || true
  sleep 300
done
echo "[INFO] CPT finished."

echo "[INFO] Starting SFT..."
nohup python3 versions/trilha2-lora/scripts/04_train_sft_qlora.py \
  --config versions/trilha2-lora/configs/sft_qlora_llama31_8b_instruct_canarim10k.json \
  > "$SFT_OUT_FILE" 2>&1 &
echo $! > "$SFT_PID_FILE"
echo "[INFO] SFT PID: $(cat "$SFT_PID_FILE")"

SFT_PID="$(cat "$SFT_PID_FILE" | tr -d '[:space:]' || true)"
while kill -0 "$SFT_PID" 2>/dev/null; do
  echo "[$(date -Is)] SFT still running (pid=$SFT_PID). Last log line:"
  tail -n 1 "$SFT_OUT_FILE" 2>/dev/null || true
  sleep 300
done
echo "[INFO] SFT finished."

echo "[INFO] Packing artifacts..."
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_TAR="$LOGS_DIR/trilha2_artifacts_${STAMP}.tar.gz"

tar --ignore-failed-read -czf "$OUT_TAR" \
  versions/trilha2-lora/configs \
  versions/trilha2-lora/datasets \
  versions/trilha2-lora/logs \
  versions/trilha2-lora/outputs/cpt_qlora_llama31_8b_brwac10k/adapter \
  versions/trilha2-lora/outputs/cpt_qlora_llama31_8b_brwac10k/tokenizer \
  versions/trilha2-lora/outputs/sft_qlora_llama31_8b_instruct_canarim10k/adapter \
  versions/trilha2-lora/outputs/sft_qlora_llama31_8b_instruct_canarim10k/tokenizer

echo "[OK] Artifact bundle: $OUT_TAR"
