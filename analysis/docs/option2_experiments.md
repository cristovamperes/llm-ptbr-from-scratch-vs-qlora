# Option 2 – Experimentos

## Ambiente baseline
- GPU: NVIDIA GeForce RTX 3090 (24 GB) – ver `analysis/nvidia_smi_baseline.txt`
- Virtualenv: `python3.10` + `tensorflow==2.16.1`
- Tokenizer base: `versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model`

## Amostras de referência
- Decoding padrão (T=0.85): `analysis/samples_v5_main_baseline.json`
- Greedy (T=0.3/top_p=0.90): `analysis/samples_v5_main_greedy_baseline.json`
- Observação: outputs contêm fragmentação severa (`clín`, `encamatos`, `suasóveisõe`).

## Diagnósticos tokenização
- Script: `scripts/diagnostics/diagnose_tokenizer.py`
- Saída completa: `analysis/tokenizer_diagnostics_baseline.txt`
- Destaques:
  - Tokenizer reconhece subwords como `clín`, `clínica`, `clínico` (não há bug no vocabulário).
  - Byte fallback presente (256 peças) → possível excesso de fragmentação.

### Estatísticas corpus/tokenizer (SentencePiece pipeline)
- Comando: `python scripts/sentencepiece_pipeline.py stats --tokenizer versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model --limit 8000 --sample-docs 1500 --min-len 200 --output analysis/tokenizer_stats_v4k_bf.json`
- Resultados principais (`analysis/tokenizer_stats_v4k_bf.json`):
  - Documentos processados: 1500
  - Tokens médios/doc: ~690 (P95 ≈ 2322 | P99 ≈ 5034)
  - Tokens por caractere: ~0.28 → forte segmentação subword.

### Tokenizers expandidos (SentencePiece)
- 8k (`versions/v5-option2/tokenizers/8k/spm_v5_opt2_8k.model`)
  - Tokens médios/doc: ~517 | P95 ≈ 1873
  - Tokens por caractere: ~0.24
- 16k (`versions/v5-option2/tokenizers/16k/spm_v5_opt2_16k.model`)
  - Tokens médios/doc: ~458 | P95 ≈ 1540
  - Tokens por caractere: ~0.22

### Ensaios iniciais (2 epochs, config base Transformer v5)
- 8k trial (`versions/v5-option2/models/modelo_v5_opt2_8k_trial.keras`)
  - Hiperparâmetros: batch 96, `label_smoothing=0.0`, `dropout=0.05`, demais iguais ao V5 main.
  - Val loss 4.38 | val ppl ≈ 79.8 após 2 épocas (210M tokens vistos).
  - Saídas: `analysis/samples_v5_opt2_8k_trial.json` (T=0.85) e `analysis/samples_v5_opt2_8k_trial_greedy.json` (greedy).
  - Observação: perplexidade caiu pouco e texto ainda apresenta fragmentação — possivelmente treino curto.
- 8k long (warm-start dos 2 epochs → 10 epochs totais): `versions/v5-option2/models/modelo_v5_opt2_8k_long.keras`
  - Mesma arquitetura da trial; continuou de `epoch_02`.
  - Val loss 3.95 | val ppl ≈ 51.9 (≈1B tokens vistos; ~2.9h).
  - Amostras: `analysis/samples_v5_opt2_8k_long.json` / `_greedy.json`.
  - Observação: perplexidade aproxima V5 main, porém textos seguem altamente fragmentados (mesmo padrão de subwords “clín …”).
- 16k trial (`versions/v5-option2/models/modelo_v5_opt2_16k_trial.keras`)
  - Mesma configuração de treino; vocabulário 16k.
  - Val loss 4.69 | val ppl ≈ 108.4 após 2 épocas (191M tokens vistos).
  - Saídas: `analysis/samples_v5_opt2_16k_trial.json` e `analysis/samples_v5_opt2_16k_trial_greedy.json`.
  - Observação: perplexidade pior que 8k/v4k; precisa de mais épocas ou ajustes adicionais.
- 8k archA (`versions/v5-option2/models/modelo_v5_opt2_8k_archA.keras`)
  - Ajustes: `d_model=320`, `num_layers=6`, `d_ff=1280`, warmup 3000, batch 96.
  - Treino 4 épocas: val loss 3.95 | val ppl ≈ 51.7 (tempo ~0.4h); sem ganhos perceptíveis nos samples (`analysis/samples_v5_opt2_8k_archA*.json`).
  - Observação: maior capacidade melhora perda rapidamente, mas a geração continua degradada → evidência de problema no alvo/dados e não apenas capacidade.

## Próximos passos planejados
1. Treinar tokenizers 8k e 16k (SentencePiece) com corpus limpo atualizado. ✅
2. Executar dry-runs (2 epochs) comparando perplexidade x fragmentação. ✅
3. Ajustar treino (batch 96+, `label_smoothing=0`, `dropout=0.05`) e medir impacto. ✅ (modelo 8k long / archA)
4. Instrumentar métricas qualitativas (wordpiece/word ratio, repeat penalties) após novos treinos. ⬜


## v5b long run (tokenizer 40k)
- Treino completo (12 epochs) com `d_model=320`, 6 camadas, batch 32, dropout 0.03, lr 1.5e-4 e tokenizer `versions/v5b-transformer/tokenizers/unigram_40k/spm_v5b.model`.
- Metricas finais (`versions/v5b-transformer/logs/history_v5b_long.csv`): loss 3.921 (ppl 50.46) / acc 29.96% e val_loss 4.236 (val_ppl 69.13) / val_acc 27.41%.
- Checkpoint: `versions/v5b-transformer/models/modelo_v5b_long_checkpoint.keras` (pesos sincronizados com `logs/checkpoints_v5b_long/epoch_12.weights.h5`).

### Samples + guardrails
- Nucleus T=0.8 / top_p=0.9 (`analysis/artifacts/samples/samples_v5b_long_tp08.json`) -> guardrails aprovados (`analysis/artifacts/guardrails/samples_v5b_long_tp08_guardrails_test.json`, fallback 0%, short-piece 8.1%).
- Nucleus T=0.7 / top_p=0.85 (`analysis/artifacts/samples/samples_v5b_long_tp07.json`) -> guardrails aprovados (`analysis/artifacts/guardrails/samples_v5b_long_tp07_guardrails.json`, fallback 0%, short-piece 6.0%).
- Beam search (beam=5, alpha=0.6) (`analysis/artifacts/samples/samples_v5b_long_beam.json`) -> reprovado em short-piece (14.6% em 1 prompt) (`analysis/artifacts/guardrails/samples_v5b_long_beam_guardrails.json`). Mantido para referencia qualitativa apenas.
- Final set (T=0.75 / top_p=0.88, 220 tokens) em `analysis/artifacts/samples/samples_v5b_long_final_tp075.json` -> guardrails aprovados com folga (`analysis/artifacts/guardrails/samples_v5b_long_final_tp075_guardrails.json`, fallback 0%, short-piece 5.0%).
