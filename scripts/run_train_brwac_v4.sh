#!/usr/bin/env bash
set -euo pipefail

# Treina o modelo v4 (subword + GRU/LSTM) no BrWaC.
# Requer tokenizer SentencePiece ja treinado e salvo em versions/v4-subword-lstm/tokenizer/.

TOKENIZER_DIR="${TOKENIZER_DIR:-versions/v4-subword-lstm/tokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${TOKENIZER_DIR}/spm_v4.model}"
VOCAB_SIZE="${VOCAB_SIZE:-3200}"
MAX_TEXTOS="${MAX_TEXTOS:-20000}"
MIN_LEN="${MIN_LEN:-200}"
EPOCAS="${EPOCAS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEQ_LEN="${SEQ_LEN:-192}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
STRIDE="${STRIDE:-2}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-200000}"
DROPOUT="${DROPOUT:-0.1}"
RECURRENT_DROPOUT="${RECURRENT_DROPOUT:-0.0}"
FINAL_DROPOUT="${FINAL_DROPOUT:-0.1}"
RNN_TYPE="${RNN_TYPE:-gru}"

if [ ! -f "${TOKENIZER_MODEL}" ]; then
  echo "[ERROR] Tokenizer SentencePiece nao encontrado em ${TOKENIZER_MODEL}" >&2
  exit 1
fi

python scripts/train_char_model_brwac.py \
  --arch v4 \
  --tokenization subword \
  --tokenizer_path "${TOKENIZER_MODEL}" \
  --vocab_size "${VOCAB_SIZE}" \
  --max_textos "${MAX_TEXTOS}" \
  --min_len "${MIN_LEN}" \
  --epocas "${EPOCAS}" \
  --batch_size "${BATCH_SIZE}" \
  --tamanho_sequencia "${SEQ_LEN}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --stride "${STRIDE}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --dropout "${DROPOUT}" \
  --recurrent_dropout "${RECURRENT_DROPOUT}" \
  --final_dropout "${FINAL_DROPOUT}" \
  --rnn_type "${RNN_TYPE}" \
  --subword_add_bos \
  --subword_add_eos \
  --modelo_saida versions/v4-subword-lstm/models/modelo_brwac_v4_subword.keras \
  --mapeamentos_saida versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4.pkl \
  --log_json_saida versions/v4-subword-lstm/logs/train_brwac_v4_subword.json
