# Avaliacao comparativa - modelos char/subword (BrWaC 20k)

## Ambiente
- Treinos realizados em instancia VAST.ai com GPU NVIDIA GeForce RTX 3080 (CUDA 12.9, cuDNN 9.14). SO: Ubuntu 22.04, Python 3.10.12, TensorFlow 2.20.0.
- Avaliacoes executadas na mesma instancia (venv `tcc_venv`, TensorFlow 2.20.0 GPU) via `analysis/evaluate_char_models.py`.
- Subconjunto de avaliacao: `nlpufg/brwac`, primeiros 2.000 documentos apos limpeza (`min_len >= 200`, normalizacao NFKC, remocao de metadados em caixa alta). Resultado: 1.787 textos validos (~4,69 M caracteres). Janelas de 160 caracteres, stride 4, limite de 300.000 janelas.

## Metricas automaticas (char-level)
Arquivo: `analysis/artifacts/results/results_char_models_20k.json`

| Modelo         | Loss medio | Perplexidade | Acuracia | Seq. len | Janelas avaliadas |
|----------------|-----------:|-------------:|---------:|---------:|-------------------:|
| `v3_brwac_20k` | **1.3753** | **3.96**     | **0.5818** | 160     | 300000            |
| `v2_brwac_20k` | 1.4366     | 4.21         | 0.5669   | 160     | 300000            |
| `v1_brwac_20k` | 1.5218     | 4.58         | 0.5500   | 160     | 300000            |

Observacoes:
- O empilhamento LSTM do `v3` (512  512 com layer norm + dropout final) entrega novo ganho sobre o `v2`: -0.061 no loss medio, perplexidade ~6% menor e +1,5 p.p. de acuracia.
- Comparado ao baseline `v1`, o salto acumulado agora passa de -0.146 no loss e +3,2 p.p. de acuracia, mantendo o mesmo dataset/stride.
- Os logs completos de inferencia estao no JSON; todos os modelos avaliam 300k janelas para comparacao direta.

## Custo aproximado de treinamento (char-level)
Fonte: `versions/v1-char-rnn/logs/train_brwac_v1_20k.json`, `versions/v2-char-lm/logs/train_brwac_v2_20k.json` e `versions/v3-stacked-lstm/logs/train_brwac_v3_20k.json`.

| Modelo         | Tempo (s) | Tokens totais (aprox.) | Tokens/s | TFLOPs estimados |
|----------------|----------:|-----------------------:|---------:|-----------------:|
| `v2_brwac_20k` | **3156**  | **3,47 B**             | **1,10 M** | **35.743**       |
| `v3_brwac_20k` | 7143      | 3,47 B                 | 0,486 M   | 79.437           |
| `v1_brwac_20k` | 3880      | 13,88 B                | 3,58 M    | 27.389           |

Notas rapidas:
- O `v1` gera janelas com stride 1 (cerca de 4x mais tokens). Apesar do volume maior, termina pouco depois do `v2` porque nao usa tf.data; o custo em TFLOPs fica menor por ter menos parametros.
- O `v3` reutiliza o pipeline do `v2`, mas dobra as camadas LSTM + layer norm e dropout final: treino leva ~2,3x mais tempo e quase dobra o custo em TFLOPs, mantendo o mesmo volume de tokens.
- Esses numeros servem como referencia pratica de throughput na RTX 3080 ao escalar para arquiteturas maiores.

## Amostras de geracao (char-level)
- Arquivo: `analysis/artifacts/samples/samples_brwac_20k.json`
- Configuracao fixa: temperatura 0.7, `top_k=40`, `top_p=0.95`, comprimento 280, seeds 42-46.
- Resumo qualitativo: o `v3` sustenta coerencia similar ao `v2`, mas com menor repeticao e transicoes mais suaves; ainda ha deriva topica ocasional apos ~200 tokens. O `v2` continua superior ao `v1`, que alterna trechos coerentes com fragmentos ruidosos. As amostras completas estao no JSON.

## Extensao subword (v4)

### Metricas automaticas (SentencePiece)

| Modelo                                   | Tokenizer                   | Épocas | Loss  | PPL   | Acc   | Observações |
|------------------------------------------|-----------------------------|-------:|------:|------:|------:|-------------|
| `v4_brwac_subword_ep6`                   | BPE 3.2k (sem fallback)     | 6      | 4.2867 | 72.73 | 0.2362 | Melhor ppl, mas gera `<unk>` (`⁇`) nas amostras. |
| `v4_brwac_subword_v4k_bf_ep2`            | BPE 4.0k + byte fallback    | 2      | 4.4345 | 84.31 | 0.2264 | Textos legíveis, sem `<unk>`; ainda repetitivo. |
| `v4_brwac_subword_v4k_bf_lstm_ep2`       | BPE 4.0k + byte fallback    | 2      | 4.4470 | 85.37 | 0.2257 | Variante LSTM; métricas próximas ao GRU. |

- Todas as avaliações usam o mesmo subset (stride 2, 250k janelas ≈ 1,32 M tokens).
- Byte fallback permite reconstruir caracteres arbitrários, eliminando símbolos estranhos nas amostras.
- Artefatos/resultados: `analysis/artifacts/results/results_subword_models_20k.json`, `analysis/artifacts/results/results_subword_models_v4_ep6.json`, `analysis/artifacts/results/results_subword_models_v4k_bf_ep2.json`, `analysis/artifacts/results/results_subword_models_v4k_bf_lstm_ep2.json` e arquivos `analysis/artifacts/samples/samples_brwac_v4*.json` correspondentes.

## Reprodutibilidade

```bash
# Avaliacao automatica (gera analysis/artifacts/results/results_char_models_20k.json com v1/v2/v3)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe analysis/evaluate_char_models.py ^
  --model v1_brwac_20k:versions/v1-char-rnn/models/modelo_brwac_v1_20k.keras:versions/v1-char-rnn/mappings/mapeamentos_brwac_v1_20k.pkl ^
  --model v2_brwac_20k:versions/v2-char-lm/models/modelo_brwac_v2_20k.keras:versions/v2-char-lm/mappings/mapeamentos_brwac_v2_20k.pkl ^
  --model v3_brwac_20k:versions/v3-stacked-lstm/models/modelo_brwac_v3_20k.keras:versions/v3-stacked-lstm/mappings/mapeamentos_brwac_v3_20k.pkl ^
  --max_docs 2000 --min_len 200 --stride_eval 4 --batch_size 1024 --max_windows 300000 ^
  --output analysis/artifacts/results/results_char_models_20k.json

# Avaliacao subword (gera analysis/artifacts/results/results_subword_models_20k.json com v4)
.venv\Scripts\python.exe analysis/evaluate_char_models.py ^
  --model v4_brwac_subword:versions/v4-subword-lstm/models/modelo_brwac_v4_subword.keras:versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4.pkl ^
  --max_docs 2000 --min_len 200 --stride_eval 2 --batch_size 1024 --max_windows 250000 ^
  --output analysis/artifacts/results/results_subword_models_20k.json

# Avaliacao subword (Tokenizador 4k + byte_fallback, GRU)
.venv\Scripts\python.exe analysis/evaluate_char_models.py ^
  --model v4_brwac_subword_v4k_bf_ep2:versions/v4-subword-lstm/models/modelo_brwac_v4_subword_v4k_bf_ep2.keras:versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4k_bf_ep2.pkl ^
  --max_docs 2000 --min_len 200 --stride_eval 2 --batch_size 1024 --max_windows 250000 ^
  --output analysis/artifacts/results/results_subword_models_v4k_bf_ep2.json

# Avaliacao subword (Tokenizador 4k + byte_fallback, LSTM)
.venv\Scripts\python.exe analysis/evaluate_char_models.py ^
  --model v4_brwac_subword_v4k_bf_lstm_ep2:versions/v4-subword-lstm/models/modelo_brwac_v4_lstm_v4k_bf_ep2.keras:versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4_lstm_v4k_bf_ep2.pkl ^
  --max_docs 2000 --min_len 200 --stride_eval 2 --batch_size 1024 --max_windows 250000 ^
  --output analysis/artifacts/results/results_subword_models_v4k_bf_lstm_ep2.json

# Amostras de geracao
.venv\Scripts\python.exe analysis/generate_samples.py
# Apenas v4 (exporta analysis/artifacts/samples/samples_brwac_v4.json)
MODEL_KEYS=v4_brwac_subword OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4.json ^
  .venv\Scripts\python.exe analysis/generate_samples.py
# Byte fallback (GRU e LSTM)
MODEL_KEYS=v4_brwac_subword_v4k_bf_ep2 OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4k_bf_ep2.json ^
  .venv\Scripts\python.exe analysis/generate_samples.py
MODEL_KEYS=v4_brwac_subword_v4k_bf_lstm_ep2 OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4k_bf_lstm_ep2.json ^
  .venv\Scripts\python.exe analysis/generate_samples.py
```
# Treino subword (GRU, tokenizer 4k + byte fallback, 2 épocas)
bash scripts/v4/run_train_brwac_v4k_bf_ep2.sh

# Treino subword (LSTM, tokenizer 4k + byte fallback, 2 épocas)
python scripts/train_char_model_brwac.py --arch v4 --tokenization subword --rnn_type lstm --tokenizer_path versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model ...
```

## Resumo das versoes anteriores (v1-v3)

- **v1_char_rnn** — baseline didático; perplexidade ~4.58 em `analysis/artifacts/results/results_char_models_20k.json`, forte repetição e uso de `<unk>`; serviu para validar pipeline de geração/salvamento (artefatos em `versions/v1-char-rnn/`).
- **v2_char_lm** — migrou para `tf.data`, aumentou embedding e unidades LSTM, reduzindo a perplexidade para ~4.21 e melhorando acurácia; tornou-se o novo baseline char-level com logs em `versions/v2-char-lm/logs/`.
- **v3_stacked_lstm** — empilhou duas LSTMs 512+512 com regularização, atingindo perplexidade ~3.96 e melhor coerência qualitativa (`analysis/artifacts/samples/samples_brwac_20k.json`), encerrando a fase char-level.

## Conclusão da versão v4

- O objetivo principal — substituir `<unk>` e elevar a legibilidade — foi atingido com o tokenizer SentencePiece 4k + byte fallback. Os modelos GRU e LSTM entregam textos consistentes com prompts longos (`analysis/artifacts/samples/samples_brwac_v4k_bf_ep2.json`, `analysis/artifacts/samples/samples_brwac_v4k_bf_lstm_ep2.json`, `analysis/artifacts/samples/samples_brwac_custom_prompts.json`).
- A avaliação imparcial com `gpt-4o-mini` (`analysis/artifacts/results/evaluation_llm_review.json`) reforça a superioridade das variantes subword sobre os modelos char-level: na média, as notas de relevância subiram para ~0.2 e a coerência se manteve entre 0.8 e 1.0, enquanto os modelos char ficaram em zero de relevância com coerência ~0.8. A versão LSTM subword sofreu mais com números normalizados para “0”.
- Ajustes adicionais (mais épocas, schedulers, reintrodução de dígitos reais) são incrementais e não alteram a conclusão do capítulo. O pipeline subword está pronto para servir de base ao roadmap Transformer (v5/v_final).

## Proximos passos
1. Priorizar a transição para arquiteturas Transformer (v5/v_final), reutilizando o tokenizer 4k + fallback e o pipeline consolidado.
2. Opcional: caso deseje refinar a v4, testar mais épocas ou scheduler adaptativo e reavaliar com o script automático (`analysis/evaluate_samples_with_openai.py`).
3. Atualizar o relatório do TCC com a síntese char × subword e o resumo das notas do avaliador externo.
