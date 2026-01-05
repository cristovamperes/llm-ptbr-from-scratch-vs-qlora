# v5 — Transformer (primeira iteração)

Primeira versão Transformer (decoder-only) do projeto, construída a partir do baseline subword do v4 (SentencePiece 4k + byte fallback).

> Snapshot público: pesos/checkpoints do modelo não são versionados aqui. A rastreabilidade do experimento vem de logs, métricas por época e artefatos de análise.

## Objetivo da versão

- Migrar de RNN (v1–v4) para Transformer em GPU única.
- Medir custo/throughput e entender gargalos de qualidade (tokenização, limpeza e hiperparâmetros).
- Produzir evidências para orientar a evolução (v5b/v6).

## Resultado principal (v5_main)

Extraído de `logs/train_v5_main.json`:

- Épocas: `50`
- Melhor `val_loss`: `3.8918` (época 50)
- Melhor `val_ppl`: `48.9991` (época 50)
- Tempo total: ~`3.16` h (`total_time_sec` no log)

## Onde estão os artefatos

- Log consolidado: `logs/train_v5_main.json` (histórico completo por época)
- Históricos CSV: `logs/history_v5_main.csv` (e variantes)
- Diagnóstico/ajustes: `DIAGNOSTICO_E_CORRECOES.md`
- Execuções legadas/experimentos que falharam: `legacy/failed_training/`

## Observação sobre comparabilidade

As métricas intrínsecas (`loss`/`ppl`) não são diretamente comparáveis entre tokenizers distintos. O v5 usa tokenização diferente de v5b/v6; compare principalmente *tendências* e resultados qualitativos (amostras/guardrails), ou versões com o mesmo tokenizer.
