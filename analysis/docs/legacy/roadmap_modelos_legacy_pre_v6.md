# Roadmap de Evolucao dos Modelos

> Documento legado (pré-v5/v6): mantido apenas para registro. Para a visão atual, veja `analysis/docs/roadmap_modelos.md`.

## Objetivo geral
Conduzir o projeto ate um `v_final` baseado em Transformer (estilo GPT), passando por etapas intermediarias que isolam ganhos em dados, arquitetura e pipeline. Cada versao registra objetivos, mudancas tecnicas e metricas esperadas.

## Linha do tempo proposta

### v1 (baseline atual)
- **Tokenizacao:** caracteres puros, vocabulario completo do texto.
- **Modelo:** `Embedding(64) -> LSTM(256) -> Dense(vocab)`.
- **Pipeline:** janelas geradas em NumPy com stride 1; treino sequencial classico.
- **Objetivo:** estabelecer referencia minima de desempenho e custo.
- **Status atual:** concluído — baseline registrado em `analysis/artifacts/results/results_char_models_20k.json` e amostras em `analysis/artifacts/samples/samples_brwac_20k.json`.

### v2 (baseline com tf.data)
- **Tokenizacao:** caracteres.
- **Modelo:** `Embedding(256) -> LSTM(512)` com dropout/recurrent_dropout e clipping opcional.
- **Pipeline:** janelas com `tf.data`, stride configuravel, token UNK, logs detalhados.
- **Objetivo:** corrigir gargalos de memoria/throughput do v1 e melhorar generalizacao.
- **Status atual:** concluído — ganhos consolidados vs. v1, artefatos em `versions/v2-char-lm/` e resultados no mesmo JSON comparativo.

### v3 (LSTM empilhada + regularizacao)
- **Tokenizacao:** caracteres (mesmo do v2).
- **Modelo:** 2-3 camadas LSTM empilhadas (ex.: 512 -> 512), dropout adicional, opcional layer norm/residual.
- **Pipeline:** `tf.data` com stride ajustavel, possivel aumento de batch e coleta de stats.
- **Objetivo:** explorar limite do paradigma char-level antes de trocar a representacao.
- **Status atual:** concluído — perplexidade 3.96 registrada, encerrando a fase char-level com logs em `versions/v3-stacked-lstm/logs`.

### v4 (Subword + LSTM/GRU)
- **Tokenizacao:** SentencePiece/BPE (vocab 2k-4k) com encode/decode integrados. Script dedicado em `scripts/sentencepiece_pipeline.py`.
- **Modelo:** 1-2 camadas GRU/LSTM empilhadas (embedding 256-512), dropout opcional, janela menor (seq 160-200).
- **Pipeline:** `train_char_model_brwac.py` suporta `--tokenization subword`; avaliacao/geracao lidam automaticamente com SentencePiece.
- **Objetivo:** diminuir sequencia efetiva, trazer semantica de subword e preparar o pipeline para Transformers.
- **Status atual:** concluído — tokenizer 4k + byte fallback validado, amostras e avaliação externa registradas (`analysis/artifacts/results/evaluation_llm_review.json`).
- **Metas concluídas:** eliminar `<unk>`, comprovar ganho qualitativo sobre char-level, estabilizar scripts para reutilização no v5.

### v5 (Transformer Encoder pequeno) [opcional, recomendado]
- **Tokenizacao:** subword do v4.
- **Objetivo geral:** exercitar componentes de atenção (masking, embeddings posicionais, AdamW + warmup) e preparar o pipeline para o GPT pequeno.
- **Arquitetura candidata (autoregressiva, decoder-only):**
  - `d_model = 384`, `n_heads = 6`, `d_ff = 1536`, `num_layers = 6`, embedding compartilhado com logits.
  - Seq. máxima: 256 tokens (subword 4k + fallback); posicional sinusoidal.
  - Dropout 0.1 em atenção e FFN; layer norm pré-atencão (Pre-LN).
- **Treino proposto:**
  - Dataset BrWaC 20k (mesma limpeza da v4), janela 256 tokens, stride 2.
  - Batch global 64 (8 x 8) → 16k tokens/step; 50k steps (~3 épocas sobre 20k docs).
  - Otimizador AdamW (`lr=3e-4`, `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`), warmup linear 2k steps + decaimento coseno.
  - Mixed precision (fp16/bfloat16) se disponível; gradiente acumulado opcional para caber em 12‑16 GB.
- **Recursos estimados:**
  - Tokens por época ≈ 1.3 M (com stride 2); 3 épocas ≈ 3.9 M tokens; 50k steps × 16k tokens = 800 M tokens efetivos (contando repetição).
  - VRAM alvo: 12 GB (RTX 3060 Ti/3080) com mixed precision e `batch_size=8` por GPU; tempo estimado 12‑18 h.
  - TFLOPs aproximados (por step): `6 * (2*d_model^2 + 2*d_model*d_ff)` ≈ 1.2e9; × 16k tokens ≈ 1.9e13 FLOPs/step → 19 TFLOPs totais por mil steps.
- **Avaliação:**
  - Adaptar `analysis/evaluate_char_models.py` para suportar modelos TensorFlow autoregressivos com embeddings subword e máscara causal.
  - Métricas: loss/ppl em 250k janelas (stride 2), tokens/s, custo total.
  - Geração: reutilizar `analysis/generate_samples.py` (modo embedding) com prompts longos.
- **Artefatos previstos:**
  - Script `llm_transformer_v5.py` (modelo + treino).
  - Shell `run_v5_transformer.sh` (setup + parâmetros padrão).
  - Diretório `versions/v5-transformer/` com `models/`, `mappings/`, `logs/`, `README.md`.
- **Marcos:**
  1. Implementar módulo Transformer + máscara causal (validar com batch sintético).
  2. Integrar pipeline de dados (subword 4k fallback) e testar um dry-run curto (1k steps).
  3. Executar treino completo (50k steps) e registrar logs JSON/CSV.
  4. Avaliar perplexidade x v4 e gerar amostras.
  5. Rodar avaliador externo (`evaluate_samples_with_openai.py`) e documentar conclusões.

### v_final (GPT pequeno)
- **Tokenizacao:** subword herdado do v4.
- **Modelo:** Transformer autoregressivo mascarado (ex.: 6 camadas, `d_model=512`, 8 cabecas, FFN 2048, posicional learned).
- **Pipeline:** batches 128-256 tokens, gradient clipping, opcional mixed precision; avaliacao/geracao convertendo subwords.
- **Objetivo:** entregar baseline GPT nacional e documentar ganhos vs. v3/v4.
- **Metas:** registrar custo total (tokens/s, TFLOPs, memoria) e produzir material final para o TCC.

## Tarefas proximas
1. Implementar `versions/v5-transformer/` com script de treino `llm_transformer_v5.py` e shell `run_v5_transformer.sh`.
2. Ajustar pipeline de dados para janelas subword 256 tokens com máscara causal e registrar dry-run (≤1k steps).
3. Adaptar `analysis/evaluate_char_models.py` ou criar avaliador específico para o Transformer (perplexidade + logs).
4. Preparar monitoramento de treino: TensorBoard ou logs JSON com loss, lr, tokens/s, TFLOPs.
5. Planejar uso de mixed precision e checar disponibilidade de VRAM (≥12 GB); definir estratégia de gradient accumulation se necessário.
6. Após validação do v5, atualizar documentação (README, evaluation_char_models.md, roadmap) e rodar avaliador externo com prompts padrão.

## Referencias cruzadas
- Metricas atuais: `analysis/artifacts/results/results_char_models_20k.json`, `analysis/artifacts/results/results_subword_models_20k.json`, `analysis/artifacts/results/results_subword_models_v4k_bf_ep2.json`, `analysis/artifacts/results/results_subword_models_v4k_bf_lstm_ep2.json`.
- Amostras atuais: `analysis/artifacts/samples/samples_brwac_20k.json`, `analysis/artifacts/samples/samples_brwac_v4.json`, `analysis/artifacts/samples/samples_brwac_v4k_bf_ep2.json`, `analysis/artifacts/samples/samples_brwac_v4k_bf_lstm_ep2.json`.
- Scripts principais: `analysis/evaluate_char_models.py`, `analysis/generate_samples.py`, `scripts/train_char_model_brwac.py`, `scripts/sentencepiece_pipeline.py`.
- Logs recentes: `versions/v1-char-rnn/logs/train_brwac_v1_20k.json`, `versions/v2-char-lm/logs/train_brwac_v2_20k.json`, `versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json`, `versions/v4-subword-lstm/logs/`.
