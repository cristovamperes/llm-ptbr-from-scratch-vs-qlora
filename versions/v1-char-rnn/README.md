# v1 — char-level (LSTM) em subconjunto do BrWaC

Primeiro protótipo do projeto (linha experimental): modelo gerador de texto **por caracteres**, treinado em um subconjunto do **BrWaC** (20k documentos). Esta versão estabelece o pipeline mínimo de treino/validação e gera evidências comparáveis (logs + amostras).

> Snapshot público: pesos/checkpoints (`.keras`) **não** são versionados aqui. Os resultados reportados no TCC podem ser auditados via logs e artefatos de análise.

## Artefatos neste diretório

- Logs do treino: `logs/train_brwac_v1_20k.json`
- Histórico CSV: `logs/history_*.csv`
- Log de batches: `logs/batches_*.log`
- Mapeamentos (vocab/índices): `mappings/`
- Notebook (didático): `notebooks/`

## Resultados (métricas intrínsecas)

Os números consolidados desta versão estão em:

- `analysis/artifacts/results/results_char_models_20k.json`

## Reprodução (opcional)

O treino char-level é executado pelo script comum às versões iniciais:

- `scripts/train_char_model_brwac.py` (na raiz do repositório)
