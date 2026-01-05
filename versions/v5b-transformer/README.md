# v5b — Transformer + tokenizer unigram (melhoria de tokenização)

O v5b reinicia a linha Transformer após os problemas observados no v5, com foco explícito em **tokenização** e guardrails qualitativos (fragmentação/byte fallback).

> Snapshot público: pesos/checkpoints do modelo não são versionados aqui. Evidências e reprodutibilidade vêm de logs, métricas por época, tokenizers e artefatos de análise.

## O que mudou vs. v5

- Dados mais limpos (linhas mínimas e filtro por razão alfabética).
- Tokenizer SentencePiece **unigram** (com byte fallback) para reduzir fragmentação.
- Observabilidade: métricas por época + amostras padronizadas + guardrails.

## Resultado principal (v5b_long)

Extraído de `logs/metrics_v5b_long/metrics_epoch_*.json`:

- Épocas executadas: `4`
- Melhor `val_loss`: `3.3843` (época 4)
- Melhor `val_ppl`: `29.4983` (época 4)
- Tempo total: ~`2.91` h (soma de `epoch_time_sec`)

## Onde estão os artefatos

- Tokenizers:
  - `tokenizers/unigram_12k/spm_v5b.model` + `tokenizers/unigram_12k/fragmentation_report.json`
- Logs e métricas:
  - Métricas por época: `logs/metrics_v5b_long/metrics_epoch_*.json`
  - Histórico CSV: `logs/history_v5b_long.csv`
  - Stdout do treino: `logs/long_stdout_20251105_relaunch.log`
- Guardrails e amostras (comparáveis):
  - `analysis/artifacts/guardrails/guardrails_v5b_long.json`
  - `analysis/artifacts/samples/samples_v5b_long_final_tp075.json`
