#!/usr/bin/env bash
set -euo pipefail
cd /root/TCC_v4
export TF_FORCE_GPU_ALLOW_GROWTH=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
source .venv/bin/activate
python scripts/train_char_model_brwac.py \
  --arch v4 \
  --tokenization subword \
  --tokenizer_path versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model \
  --vocab_size 4000 \
  --max_textos 20000 \
  --min_len 200 \
  --epocas 2 \
  --batch_size 192 \
  --tamanho_sequencia 192 \
  --embedding_dim 512 \
  --stride 2 \
  --shuffle_buffer 200000 \
  --dropout 0.1 \
  --recurrent_dropout 0.0 \
  --final_dropout 0.1 \
  --rnn_type gru \
  --subword_add_bos \
  --subword_add_eos \
  --modelo_saida versions/v4-subword-lstm/models/modelo_brwac_v4_subword_v4k_bf_ep2.keras \
  --mapeamentos_saida versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4k_bf_ep2.pkl \
  --log_json_saida versions/v4-subword-lstm/logs/train_brwac_v4_subword_v4k_bf_ep2.json \
  --batch_log_freq 500 \
  "$@"
