# v6 (release candidate) - Transformer + tokenizer unigram

Este diretório contém os artefatos do **v6** (pipeline Transformer em escala reduzida), incluindo tokenizers, logs de treino, métricas por época, amostras geradas e guardrails.

> Nota: este repositório público não inclui **pesos/checkpoints** do modelo. A auditoria dos resultados é feita via logs, métricas, amostras e artefatos versionados.

## O que foi executado

Foram executadas duas variações do treino “release candidate”, ambas com:

- Tokenizer: SentencePiece **unigram 16k** (`tokenizers/unigram_16k/spm_v6.model`)
- Dados: BrWaC, amostra de `100000` documentos (split 90/10 para validação)
- Sequência: `seq_len=256`, `stride=96`, `batch_size=64`, `epochs=24`
- Hardware: 1x **NVIDIA GeForce RTX 4090** (24 GB) (evidência nos logs “head”)

## Por que existem dois runs (aligned vs. não-aligned)

- `v6_rc16k_v5b`: treino com uma limpeza/normalização de texto diferente da usada na preparação do corpus do tokenizer (ex.: lowercasing e normalização de números).
- `v6_rc16k_v5b_aligned`: treino com a limpeza **alinhada** à usada no tokenizer (mesmas regras de filtragem/normalização), reduzindo discrepância entre “texto que o tokenizer viu” vs. “texto que o modelo viu”.

Na prática, o run **aligned** resultou em ganho **marginal** nas métricas intrínsecas, com diferenças pequenas mas consistentes em `val_loss`/`val_ppl`.

## Resultados (validação intrínseca)

Os números abaixo foram extraídos diretamente dos logs JSON:

| Execução | Melhor época (val_loss / val_ppl) | Final (val_loss / val_ppl) | Tokens vistos (total) | Tempo (h) |
|---|---:|---:|---:|---:|
| `v6_rc16k_v5b` | ep 23: 3.3001 / 27.1159 | ep 24: 3.3002 / 27.1173 | 3,760,717,824 | 13.32 |
| `v6_rc16k_v5b_aligned` *(v6 final do TCC)* | ep 24: 3.2935 / 26.9362 | ep 24: 3.2935 / 26.9362 | 3,632,922,624 | 16.56 |

Fontes:

- `logs/train_v6_rc16k_v5b.json`
- `logs/train_v6_rc16k_v5b_aligned.json`

## Onde estão os artefatos

- Tokenizers
  - `tokenizers/unigram_16k/` (v6) e `tokenizers/unigram_12k/` (baseline/backup)
  - Cada pasta contém: `spm_v6.model`, `spm_v6.vocab`, `meta.json`, `fragmentation_report.json`
- Logs e métricas
  - Históricos completos: `logs/train_*.json` e `logs/history_*.csv`
  - Métricas por época: `logs/metrics_v6_rc16k_v5b*/metrics_epoch_*.json`
  - Evidência de configuração/hardware: `logs/run_v6_rc16k_v5b_head.txt` e `logs/train_v6_rc16k_v5b_aligned_head.txt`
- Amostras e guardrails
  - Amostras (comparáveis): `analysis/samples_v6_rc16k_v5b.json` e `analysis/samples_v6_rc16k_v5b_aligned.json`
  - Guardrails: `analysis/guardrails_v6_rc16k_v5b.json` e `analysis/guardrails_v6_rc16k_v5b_aligned.json`
