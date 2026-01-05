# v2 — char-level (LSTM) com pipeline `tf.data`

Evolução do v1 (char-level), consolidando o pipeline com `tf.data` e maior capacidade do modelo. Mantém o mesmo objetivo da linha inicial: medir custo/limitações em escala reduzida e produzir logs/auditoria reprodutível.

> Snapshot público: pesos/checkpoints (`.keras`) **não** são versionados aqui. Evidências vêm de logs, mapeamentos e artefatos de análise.

## Artefatos neste diretório

- Logs do treino: `logs/train_brwac_v2_20k.json`
- Histórico CSV: `logs/history_*.csv`
- Log de batches: `logs/batches_*.log`
- Mapeamentos (vocab/índices): `mappings/`
- Notebook (didático): `notebooks/`

## Resultados (métricas intrínsecas)

Os números consolidados desta versão estão em:

- `analysis/artifacts/results/results_char_models_20k.json`

## Reprodução (opcional)

Treino char-level:

- `scripts/train_char_model_brwac.py` (na raiz do repositório)
