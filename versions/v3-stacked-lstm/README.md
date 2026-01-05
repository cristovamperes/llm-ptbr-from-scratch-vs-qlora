# v3 — char-level (LSTM empilhada)

Terceira versão da linha char-level, com LSTM empilhada (mais capacidade e regularização) para avaliar limites qualitativos antes da migração para subword/Transformer.

> Snapshot público: pesos/checkpoints (`.keras`) **não** são versionados aqui. Evidências vêm de logs e artefatos de análise.

## Artefatos neste diretório

- Log do treino: `logs/train_brwac_v3_20k.json`
- Históricos CSV: `logs/history_*.csv`
- Avaliação local (subset): `logs/results_eval_20k.json`
- Log completo (stdout): `logs/train_brwac_v3_20k_full.log`
- Mapeamentos (vocab/índices): `mappings/`

## Resultados (métricas intrínsecas)

Os números consolidados desta versão estão em:

- `analysis/artifacts/results/results_char_models_20k.json`

## Reprodução (opcional)

Treino char-level:

- `scripts/train_char_model_brwac.py` (na raiz do repositório)
