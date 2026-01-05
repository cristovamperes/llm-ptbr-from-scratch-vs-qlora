#!/usr/bin/env bash
# Script corrigido para treinar o Transformer v5 com configuracoes otimizadas
#
# Correcoes aplicadas:
# 1. seq_len=256 (alinhado com o modelo, nao 320)
# 2. batch_size=32 sequencias (nao 192 tokens)
# 3. stride=128 (reduz overlap excessivo)
# 4. learning_rate=1e-4 (mais conservador)
# 5. warmup_steps=500 (5% do total, nao 20%)
# 6. weight_decay=0.01 (menos agressivo)
# 7. Arquitetura menor: d_model=256, num_layers=4
# 8. max_docs=50000 (mais dados para generalizacao)
# 9. steps_per_epoch e total_steps calculados automaticamente

set -euo pipefail

# Configurar CUDA paths (se necessario)
if [[ -z "${CUDA_PIP_LIB:-}" ]]; then
  CUDA_PIP_LIB="$(python3 - <<'PY'
import site
import sys
from pathlib import Path

def candidates():
    seen = set()
    for base in list(site.getsitepackages()) + [site.getusersitepackages()] + sys.path:
        if not base:
            continue
        path = Path(base).resolve()
        if path in seen:
            continue
        seen.add(path)
        candidate = path / "nvidia"
        if candidate.is_dir():
            yield str(candidate)

for option in candidates():
    print(option)
    break
PY
)"
  CUDA_PIP_LIB="${CUDA_PIP_LIB:-/usr/local/lib/python3.10/dist-packages/nvidia}"
fi
if [[ -d "${CUDA_PIP_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_PIP_LIB}/cudnn/lib:${CUDA_PIP_LIB}/cublas/lib:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_DIR="versions/v5-transformer/checkpoints/fixed"
METRICS_DIR="versions/v5-transformer/logs/metrics_fixed"

# Tokenizer do v4 (4k + byte fallback)
if [[ -z "${TOKENIZER_MODEL:-}" ]]; then
  TOKENIZER_MODEL="versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model"
fi

echo "==================================================================="
echo "Treinando Transformer v5 (CONFIGURACAO CORRIGIDA)"
echo "==================================================================="
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "Arquitetura: 4 layers, d_model=256, 8 heads, d_ff=1024"
echo "Dados: 50k docs, seq_len=256, batch=32, stride=128"
echo "Treino: 4 epocas, lr=1e-4, warmup=500, weight_decay=0.01"
echo "Regularizacao: dropout=0.1, label_smoothing=0.05, fallback_penalty=0.05"
echo "==================================================================="
echo ""

python3 scripts/llm_transformer_v5.py \
  --tokenizer_path "${TOKENIZER_MODEL}" \
  --max_docs 50000 \
  --min_len 200 \
  --valid_split 0.1 \
  --seq_len 256 \
  --stride 128 \
  --batch_size 32 \
  --shuffle_buffer 10000 \
  --add_bos \
  --add_eos \
  --seed 42 \
  --train \
  --epochs 4 \
  --steps_per_epoch 0 \
  --validation_steps 0 \
  --total_steps 0 \
  --learning_rate 1e-4 \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 1024 \
  --label_smoothing 0.05 \
  --fallback_penalty 0.05 \
  --model_output "versions/v5-transformer/models/modelo_v5_fixed.keras" \
  --mappings_output "versions/v5-transformer/mappings/mapeamentos_v5_fixed.pkl" \
  --log_json_output "versions/v5-transformer/logs/train_v5_fixed.json" \
  --csv_log_output "versions/v5-transformer/logs/history_v5_fixed.csv" \
  --tensorboard_logdir "versions/v5-transformer/logs/tensorboard_fixed" \
  --metrics_dir "${METRICS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --keep_n_checkpoints 8 \
  "$@"

echo ""
echo "==================================================================="
echo "Treino concluido! Modelo salvo em:"
echo "  - versions/v5-transformer/models/modelo_v5_fixed.keras"
echo "  - versions/v5-transformer/mappings/mapeamentos_v5_fixed.pkl"
echo "==================================================================="
