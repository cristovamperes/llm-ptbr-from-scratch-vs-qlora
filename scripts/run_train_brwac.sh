#!/usr/bin/env bash
set -euo pipefail

# Sequencia padrao de treinos char-level no BrWaC.
# Execute em um ambiente com GPU (ex.: instancia remota) apos ativar o venv.

python scripts/train_char_model_brwac.py \
  --arch v1 \
  --max_textos 20000 \
  --epocas 2 \
  --batch_size 256 \
  --tamanho_sequencia 160 \
  --tamanho_lstm 256 \
  --modelo_saida versions/v1-char-rnn/models/modelo_brwac_v1_20k.keras \
  --mapeamentos_saida versions/v1-char-rnn/mappings/mapeamentos_brwac_v1_20k.pkl \
  --log_json_saida versions/v1-char-rnn/logs/train_brwac_v1_20k.json

python scripts/train_char_model_brwac.py \
  --arch v2 \
  --max_textos 20000 \
  --epocas 2 \
  --batch_size 256 \
  --tamanho_sequencia 160 \
  --tamanho_lstm 512 \
  --embedding_dim 256 \
  --stride 4 \
  --shuffle_buffer 200000 \
  --dropout 0.1 \
  --clipnorm 1.0 \
  --modelo_saida versions/v2-char-lm/models/modelo_brwac_v2_20k.keras \
  --mapeamentos_saida versions/v2-char-lm/mappings/mapeamentos_brwac_v2_20k.pkl \
  --log_json_saida versions/v2-char-lm/logs/train_brwac_v2_20k.json

python scripts/train_char_model_brwac.py \
  --arch v3 \
  --max_textos 20000 \
  --epocas 2 \
  --batch_size 256 \
  --tamanho_sequencia 160 \
  --tamanho_lstm 512 \
  --stack_units 512,512 \
  --embedding_dim 256 \
  --stride 4 \
  --shuffle_buffer 200000 \
  --dropout 0.1 \
  --clipnorm 1.0 \
  --final_dropout 0.1 \
  --layer_norm \
  --modelo_saida versions/v3-stacked-lstm/models/modelo_brwac_v3_20k.keras \
  --mapeamentos_saida versions/v3-stacked-lstm/mappings/mapeamentos_brwac_v3_20k.pkl \
  --log_json_saida versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json
