# v4 — subword (SentencePiece) + GRU/LSTM

Quarta versão da linha experimental: introduz tokenização **subword** (SentencePiece) e modelos GRU/LSTM para melhorar legibilidade e preparar a migração para Transformers.

> Snapshot público: pesos/checkpoints (`.keras`) **não** são versionados aqui. Evidências vêm de tokenizers, logs e artefatos de análise.

## Principais mudanças vs. v1–v3

- Tokenização SentencePiece (subword), reduzindo sequência efetiva vs. char-level.
- Experimentos com GRU e LSTM, e variantes de `seq_len`/`stride`.
- Introdução do **byte fallback** (baseline `4k`) para reduzir `<unk>` e melhorar robustez.

## Onde estão os artefatos

- Tokenizer baseline (4k + byte fallback):
  - `tokenizer_v4k_bf/spm_v4k_bf.model`
  - `tokenizer_v4k_bf/spm_v4k_bf.vocab`
  - `tokenizer_v4k_bf/stats.json`
- Logs do treino (JSON + stdout):
  - `logs/train_brwac_v4_subword.json`
  - `logs/train_brwac_v4_subword_ep6.json`
  - `logs/train_brwac_v4_subword_v4k_bf_ep2.json`
  - `logs/train_brwac_v4_lstm_v4k_bf_ep2.json`
- Métricas e amostras consolidadas (comparáveis):
  - `analysis/artifacts/results/results_subword_models_20k.json` (v4 “legacy”)
  - `analysis/artifacts/results/results_subword_models_v4k_bf_ep2.json`
  - `analysis/artifacts/results/results_subword_models_v4k_bf_lstm_ep2.json`
  - `analysis/artifacts/samples/samples_brwac_v4*.json`

## Encerramento da versão

O v4 encerra a fase de migração char-level → subword e estabelece um baseline subword estável para a primeira iteração Transformer (v5), que depois evolui para v5b e v6.
