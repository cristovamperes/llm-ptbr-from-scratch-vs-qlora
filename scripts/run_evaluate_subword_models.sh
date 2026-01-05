#!/usr/bin/env bash
set -euo pipefail

# Avalia modelos subword (v4) no subset padrao do BrWaC.
# Requer tokenizer SentencePiece e modelo treinado em versions/v4-subword-lstm/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_SPECS="${MODEL_SPECS:-v4_brwac_subword:versions/v4-subword-lstm/models/modelo_brwac_v4_subword.keras:versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4.pkl}"
IFS=' ' read -r -a MODEL_ARRAY <<< "${MODEL_SPECS}"
MODEL_ARGS=()
for spec in "${MODEL_ARRAY[@]}"; do
  MODEL_ARGS+=("--model" "${spec}")
done
if [ "${#MODEL_ARGS[@]}" -eq 0 ]; then
  echo "[ERROR] Nenhum modelo especificado via MODEL_SPECS." >&2
  exit 1
fi

python analysis/evaluate_char_models.py \
  "${MODEL_ARGS[@]}" \
  --max_docs "${MAX_DOCS:-2000}" \
  --min_len "${MIN_LEN:-200}" \
  --stride_eval "${STRIDE_EVAL:-2}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --max_windows "${MAX_WINDOWS:-250000}" \
  --output "${OUTPUT_JSON:-analysis/artifacts/results/results_subword_models_20k.json}"
