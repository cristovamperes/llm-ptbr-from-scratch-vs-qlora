#!/usr/bin/env bash
set -euo pipefail

# Unified post-training pipeline for Transformer v5.
#  1. Poll training log until completion.
#  2. Run qualitative sample generation + evaluation.
#  3. Sync artifacts (models, logs, analyses) to local machine.
#
# Usage:
#   ./monitor_and_finalize.sh [stage]
#     stage ∈ {baseline, main, scale, legacy}

SSH_KEY="N:/Coding/TCC/secrets/ssh_keys_vast-ai/vastai"
SSH_PORT=9036
SSH_HOST="45.14.246.9"
REMOTE_DIR="/workspace/TCC"
LOCAL_DIR="N:/Coding/TCC"

STAGE="${1:-main}"
LOG_FILE="$REMOTE_DIR/versions/v5-transformer/logs/train_v5_${STAGE}.log"
MODEL_FILE="$REMOTE_DIR/versions/v5-transformer/models/modelo_v5_${STAGE}.keras"
MAP_FILE="$REMOTE_DIR/versions/v5-transformer/mappings/mapeamentos_v5_${STAGE}.pkl"
SAMPLES_FILE="$REMOTE_DIR/analysis/artifacts/samples/samples_v5_${STAGE}.json"
RESULTS_FILE="$REMOTE_DIR/analysis/artifacts/results/results_v5_${STAGE}.json"

# Backward compatibility with legacy workflow.
if [[ "${STAGE}" == "legacy" ]]; then
  LOG_FILE="$REMOTE_DIR/train_v5_output.log"
  MODEL_FILE="$REMOTE_DIR/versions/v5-transformer/models/modelo_v5_fixed.keras"
  MAP_FILE="$REMOTE_DIR/versions/v5-transformer/mappings/mapeamentos_v5_fixed.pkl"
  SAMPLES_FILE="$REMOTE_DIR/analysis/artifacts/samples/samples_v5_fixed.json"
  RESULTS_FILE="$REMOTE_DIR/analysis/artifacts/results/results_v5_fixed.json"
fi

echo "==================================================================="
echo "Monitorando treinamento (stage=${STAGE}) no host ${SSH_HOST}"
echo "==================================================================="

check_training_done() {
    ssh -i "${SSH_KEY}" -p ${SSH_PORT} root@${SSH_HOST} \
        "grep -q 'Treino concluído' '${LOG_FILE}' 2>/dev/null && echo DONE || echo RUNNING"
}

while true; do
    STATUS="$(check_training_done)"
    if [[ "${STATUS}" == "DONE" ]]; then
        echo ""
        echo "✓ Treinamento concluído!"
        break
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Treinamento em andamento..."
    ssh -i "${SSH_KEY}" -p ${SSH_PORT} root@${SSH_HOST} \
        "tail -n 5 '${LOG_FILE}' 2>/dev/null | grep -E 'Epoch|loss|accuracy|ppl|tokens'" || true
    sleep 120
done

echo ""
echo "==================================================================="
echo "Etapa 2: Gerando amostras qualitativas"
echo "==================================================================="

ssh -i "${SSH_KEY}" -p ${SSH_PORT} root@${SSH_HOST} <<'ENDSSH'
set -euo pipefail
cd /workspace/TCC
MODEL_KEYS="v5_brwac_transformer_fixed" OUTPUT_PATH="analysis/artifacts/samples/samples_v5_.json" python3 analysis/generate_samples.py
echo "✓ Amostras geradas em analysis/artifacts/samples/samples_v5_.json"
ENDSSH

SAMPLES_FILE="$REMOTE_DIR/analysis/artifacts/samples/samples_v5_.json"

echo ""
echo "==================================================================="
echo "Etapa 3: Avaliação quantitativa"
echo "==================================================================="

ssh -i "${SSH_KEY}" -p ${SSH_PORT} root@${SSH_HOST} <<ENDSSH
set -euo pipefail
cd /workspace/TCC
python3 analysis/evaluate_transformer_v5.py \
    --model "${MODEL_FILE}" \
    --mapping "${MAP_FILE}" \
    --output "analysis/artifacts/results/results_v5_${STAGE}.json"
echo "✓ Avaliação salva em analysis/artifacts/results/results_v5_${STAGE}.json"
ENDSSH

RESULTS_FILE="$REMOTE_DIR/analysis/artifacts/results/results_v5_${STAGE}.json"

echo ""
echo "==================================================================="
echo "Etapa 4: Sincronizando artefatos para a máquina local"
echo "==================================================================="

mkdir -p \
  "${LOCAL_DIR}/versions/v5-transformer/models" \
  "${LOCAL_DIR}/versions/v5-transformer/mappings" \
  "${LOCAL_DIR}/versions/v5-transformer/logs" \
  "${LOCAL_DIR}/analysis/artifacts/samples" \
  "${LOCAL_DIR}/analysis/artifacts/results"

echo "Baixando modelos..."
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${MODEL_FILE}" "${LOCAL_DIR}/versions/v5-transformer/models/"
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${REMOTE_DIR}/versions/v5-transformer/models/"modelo_v5_*_checkpoint.keras \
    "${LOCAL_DIR}/versions/v5-transformer/models/" 2>/dev/null || echo "Checkpoint adicional não encontrado."

echo "Baixando mapeamentos..."
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${MAP_FILE%.*}".* "${LOCAL_DIR}/versions/v5-transformer/mappings/"

echo "Baixando logs..."
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${LOG_FILE}" "${LOCAL_DIR}/versions/v5-transformer/logs/"
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${REMOTE_DIR}/versions/v5-transformer/logs/"*.json \
    "${LOCAL_DIR}/versions/v5-transformer/logs/" 2>/dev/null || true
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${REMOTE_DIR}/versions/v5-transformer/logs/"*.csv \
    "${LOCAL_DIR}/versions/v5-transformer/logs/" 2>/dev/null || true

echo "Baixando análises..."
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${SAMPLES_FILE}" "${LOCAL_DIR}/analysis/artifacts/samples/"
scp -i "${SSH_KEY}" -P ${SSH_PORT} "${RESULTS_FILE}" "${LOCAL_DIR}/analysis/artifacts/results/"

echo ""
echo "==================================================================="
echo "✓ Sincronização concluída. Artefatos em:"
echo "  - Modelos:   ${LOCAL_DIR}/versions/v5-transformer/models/"
echo "  - Logs:      ${LOCAL_DIR}/versions/v5-transformer/logs/"
echo "  - Análises:  ${LOCAL_DIR}/analysis/"
echo "==================================================================="
echo ""
echo "Lembre-se de encerrar a instância no painel Vast.ai (${SSH_HOST}:${SSH_PORT})."
