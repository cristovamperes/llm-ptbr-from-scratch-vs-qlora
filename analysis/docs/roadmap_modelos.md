# Mapa de versões (v1–v6 + Trilha 2)

Este documento substitui um roadmap antigo (pré-v5/v6). A versão anterior foi arquivada em:

- nalysis/docs/legacy/roadmap_modelos_legacy_pre_v6.md

## Onde está a evidência

- Trilha 1 (v1–v6): ersions/v*-*/ (logs/artefatos por versão)
- Trilha 2 (QLoRA): ersions/trilha2-lora/ (treinos + avaliação extrínseca)
- Artefatos consolidados (samples/resultados/guardrails): nalysis/artifacts/

## Trilha 1 — treino do zero (escala reduzida)

- ersions/v1-char-rnn/: baseline char-level (primeiro pipeline completo).
- ersions/v2-char-lm/: char-level com melhorias de throughput/treino.
- ersions/v3-stacked-lstm/: char-level empilhado (melhor baseline char).
- ersions/v4-subword-lstm/: transição para subword (SentencePiece) e modelos recorrentes.
- ersions/v5-transformer/: primeiro Transformer (baseline Transformer).
- ersions/v5b-transformer/: melhorias de tokenização e qualidade textual (base para consolidar v6).
- ersions/v6-release-candidate/: versão final da Trilha 1 (inclui run alinhado de limpeza).

## Trilha 2 — pós-treinamento eficiente (QLoRA)

- ersions/trilha2-lora/:
  - CPT QLoRA em BrWaC (10k amostras)
  - SFT QLoRA em Canarim (10k exemplos)
  - avaliação extrínseca (QA/sumarização/reescrita), com repetição em múltiplas seeds
