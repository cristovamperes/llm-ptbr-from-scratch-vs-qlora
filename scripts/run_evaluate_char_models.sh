#!/usr/bin/env bash
set -euo pipefail

# Executa a avaliação padronizada (v1/v2/v3) no subset BrWaC 20k.
# Requer ambiente virtual com dependências já ativado.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python analysis/evaluate_char_models.py \
  --model v1_brwac_20k:versions/v1-char-rnn/models/modelo_brwac_v1_20k.keras:versions/v1-char-rnn/mappings/mapeamentos_brwac_v1_20k.pkl \
  --model v2_brwac_20k:versions/v2-char-lm/models/modelo_brwac_v2_20k.keras:versions/v2-char-lm/mappings/mapeamentos_brwac_v2_20k.pkl \
  --model v3_brwac_20k:versions/v3-stacked-lstm/models/modelo_brwac_v3_20k.keras:versions/v3-stacked-lstm/mappings/mapeamentos_brwac_v3_20k.pkl \
  --max_docs 2000 \
  --min_len 200 \
  --stride_eval 4 \
  --batch_size 1024 \
  --max_windows 300000 \
  --output analysis/artifacts/results/results_char_models_20k.json
