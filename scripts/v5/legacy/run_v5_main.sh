#!/usr/bin/env bash
# Script OTIMIZADO para treinar o Transformer v5 com parâmetros balanceados
#
# DIAGNÓSTICO REALIZADO:
# - V4 viu 7.2B tokens em 6 epochs (stride=2)
# - V5 fixed viu apenas 240M tokens em 4 epochs (stride=128)
# - V5 viu 30x MENOS tokens → por isso val_loss pior (4.91 vs 4.31)
#
# CORREÇÃO APLICADA:
# - stride=64 (reduzir de 128, dobrar windows)
# - batch_size=48 (aumentar de 32, melhor GPU usage)
# - epochs=50 (aumentar de 4, ver ~6B tokens)
# → Tempo estimado: ~14 horas
# → Total tokens: ~6B (83% do V4)

set -euo pipefail

# Configurar CUDA paths (se necessário)
CUDA_PIP_LIB="/usr/local/lib/python3.10/dist-packages/nvidia"
export LD_LIBRARY_PATH="${CUDA_PIP_LIB}/cudnn/lib:${CUDA_PIP_LIB}/cublas/lib:${LD_LIBRARY_PATH:-}"

# Tokenizer do v4 (4k + byte fallback)
if [[ -z "${TOKENIZER_MODEL:-}" ]]; then
  TOKENIZER_MODEL="versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model"
fi

echo "==================================================================="
echo "Treinando Transformer v5 (CONFIGURAÇÃO OTIMIZADA - MAIN)"
echo "==================================================================="
echo "Tokenizer: ${TOKENIZER_MODEL}"
echo "Arquitetura: 4 layers, d_model=256, 8 heads, d_ff=1024"
echo "Dados: 50k docs, seq_len=256, batch=48, stride=64"
echo "Treino: 50 épocas, lr=3e-4, warmup=2000, weight_decay=0.01"
echo "Estimativas:"
echo "  - Steps/epoch: ~9,765"
echo "  - Tokens/epoch: ~119.9M"
echo "  - Total tokens (50 epochs): ~6B tokens"
echo "  - Tempo estimado: ~14 horas"
echo "  - Comparação V4: 83% dos tokens vistos (6B vs 7.2B)"
echo "==================================================================="
echo ""

python3 scripts/llm_transformer_v5.py \
  --tokenizer_path "${TOKENIZER_MODEL}" \
  --max_docs 50000 \
  --min_len 200 \
  --valid_split 0.1 \
  --seq_len 256 \
  --stride 64 \
  --batch_size 48 \
  --shuffle_buffer 10000 \
  --add_bos \
  --add_eos \
  --seed 42 \
  --train \
  --epochs 50 \
  --steps_per_epoch 0 \
  --validation_steps 0 \
  --total_steps 0 \
  --learning_rate 3e-4 \
  --warmup_steps 2000 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --label_smoothing 0.05 \
  --gradient_clip_norm 1.0 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 1024 \
  --model_output "versions/v5-transformer/models/modelo_v5_main.keras" \
  --mappings_output "versions/v5-transformer/mappings/mapeamentos_v5_main.pkl" \
  --log_json_output "versions/v5-transformer/logs/train_v5_main.json" \
  --csv_log_output "versions/v5-transformer/logs/history_v5_main.csv" \
  --tensorboard_logdir "versions/v5-transformer/logs/tensorboard_main" \
  --checkpoint_dir "versions/v5-transformer/checkpoints/main" \
   \
  --keep_n_checkpoints 5 \
  "$@"

echo ""
echo "==================================================================="
echo "Treino concluído! Modelo salvo em:"
echo "  - versions/v5-transformer/models/modelo_v5_main.keras"
echo "  - versions/v5-transformer/mappings/mapeamentos_v5_main.pkl"
echo "  - Checkpoints em: versions/v5-transformer/checkpoints/main/"
echo "==================================================================="
