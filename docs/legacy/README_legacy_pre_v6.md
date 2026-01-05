# LLM Brasileiro  Projeto de TCC em Ciencia de Dados

> README legado (pré-v5/v6): mantido apenas para registro. O README atual está em `README.md`. Este snapshot público não inclui pesos/checkpoints.

Este repositorio consolida os experimentos de modelos de linguagem baseados no
corpus BrWaC. Cada versao registra um passo da evolucao arquitetural planejada
para o TCC, alem de scripts e analises associados.

## Estrutura principal

- `analysis/`
  - `evaluation_char_models.md`: resultados e notas das avaliacoes char-level (v1/v3).
  - `evaluate_char_models.py`: agora suporta modelos char e SentencePiece (v4).
  - `generate_samples.py`: gera amostras textuais (char/subword) e permite filtrar modelos via `MODEL_KEYS`.
  - `evaluate_samples_with_openai.py`: envia amostras para avaliacao automatizada via OpenAI (gpt-4o-mini por padrao).
  - `evaluation_llm_review.json`: relatorio consolidado da avaliacao imparcial das amostras (relevancia, coerencia, estilo).
  - `results_char_models_20k.json`: metricas consolidadas dos modelos char.
  - `results_subword_models_20k.json`: avaliacao do v4 (tokenizer 3.2k, GRU).
  - `results_subword_models_v4k_bf_ep2.json`, `results_subword_models_v4k_bf_lstm_ep2.json`: variantes com tokenizer 4k + byte fallback (GRU/LSTM).
  - `samples_brwac_v4.json`, `samples_brwac_v4k_bf_ep2.json`, `samples_brwac_v4k_bf_lstm_ep2.json`: amostras subword correspondentes.
- `scripts/`
  - `train_char_model_brwac.py`: script de treino generico (v1/v2/v3/v4) com flag `--tokenization`.
  - `sentencepiece_pipeline.py`: utilitario para treinar/inspecionar tokenizers SentencePiece (train/encode/decode/stats).
  - `run_train_brwac.sh`: executa os treinos padronizados (v1  v3) em sequencia.
  - `run_train_brwac_v4.sh`: treina o modelo subword (tokenizer 3.2k).
  - `v4/run_train_brwac_v4k_bf_ep2.sh`: treina o modelo subword com tokenizer 4k + byte fallback (GRU, 2 epocas).
  - `run_evaluate_char_models.sh` e `run_evaluate_subword_models.sh`: avaliacoes char e subword, respectivamente.
  - `run_generate_char_samples.sh` e `run_generate_subword_samples.sh`: geracao de amostras filtrada por tipo.
  - Variantes adicionais (ex. LSTM) podem ser executadas chamando `train_char_model_brwac.py` diretamente com `--rnn_type lstm`.
  - `infra/`: utilitarios para provisionar/diagnosticar VMs (instalacao, checagens de GPU, etc.).
- `tcc_llm/`
  - Módulos compartilhados (dataset, tokenização e modelos menores) usados pelos scripts.
- `versions/`
  - `v1-char-rnn/`: baseline (Embedding(64) + LSTM(256)) com notebooks ilustrativos.
  - `v2-char-lm/`: evolucao em char-level usando `tf.data`, Embedding(256) + LSTM(512).
  - `v3-stacked-lstm/`: versao empilhada (512512, layer norm opcional) com logs de treino e avaliacao.
  - `v4-subword-lstm/`: pipeline SentencePiece + GRU/LSTM (embedding configuravel, seq. menores) com README atualizado.

Artefatos treinados (`.keras`, `.pkl`) ficam dentro de cada versao nos subdiretorios
`models/` e `mappings/`. Os logs de execucao (`logs/`) registram metadata e metricas.

## Fluxo de uso

> Ative previamente o ambiente virtual (ex.: `source .venv/bin/activate` ou `.venv\Scripts\Activate.ps1`)
> e instale as dependencias com `pip install -r requirements.txt`.

### 1. (Opcional) Preparar tokenizer SentencePiece (v4)

```bash
# Tokenizer legado (3.2k, sem byte fallback)
python scripts/sentencepiece_pipeline.py train ^
  --output-dir versions/v4-subword-lstm/tokenizer ^
  --model-prefix spm_v4 ^
  --vocab-size 3200 ^
  --limit 20000 --min-len 200 --input-sentence-size 2000000

# Tokenizer recomendado (4k + byte fallback)
python scripts/sentencepiece_pipeline.py train ^
  --output-dir versions/v4-subword-lstm/tokenizer_v4k_bf ^
  --model-prefix spm_v4k_bf ^
  --vocab-size 4000 ^
  --byte-fallback ^
  --limit 20000 --min-len 200 --input-sentence-size 2000000
```

O comando acima limpa o BrWaC, exporta um corpus temporario, treina o SentencePiece e grava
artefatos em `versions/v4-subword-lstm/tokenizer/`. O mesmo script possui subcomandos
`encode`, `decode` e `stats` para inspecionar o tokenizer.

### 2. Treinar os modelos char-level (v1/v2/v3)

```bash
bash scripts/run_train_brwac.sh
```

Os artefatos e logs serao salvos em `versions/v*-*/{models,mappings,logs}` com sufixo `_20k`.
A configuracao padrao treina 2 epocas sobre 20k documentos do BrWaC.

### 3. Treinar o modelo subword (v4)

```bash
bash scripts/run_train_brwac_v4.sh
# ou (tokenizer 4k + byte fallback, GRU 2 epocas)
bash scripts/v4/run_train_brwac_v4k_bf_ep2.sh

# Variante LSTM (exemplo)
python scripts/train_char_model_brwac.py \
  --arch v4 --tokenization subword \
  --tokenizer_path versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model \
  --vocab_size 4000 --rnn_type lstm \
  --max_textos 20000 --min_len 200 --epocas 2 --batch_size 192 --tamanho_sequencia 192 --embedding_dim 512 --stride 2 \
  --modelo_saida versions/v4-subword-lstm/models/modelo_brwac_v4_lstm_v4k_bf_ep2.keras \
  --mapeamentos_saida versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4_lstm_v4k_bf_ep2.pkl
```

Certifique-se de apontar `TOKENIZER_MODEL` para o arquivo `.model` gerado pelo passo anterior.
Os artefatos serao salvos em `versions/v4-subword-lstm/{models,mappings,logs}`.

### 4. Avaliar os modelos

```bash
bash scripts/run_evaluate_char_models.sh
# ou, para apenas o v4:
bash scripts/run_evaluate_subword_models.sh
```

A avaliacao padrao gera `analysis/artifacts/results/results_char_models_20k.json` (char) e,
quando solicitado, `analysis/artifacts/results/results_subword_models_20k.json` para os modelos subword.
Ambas utilizam o mesmo subset (2.000 docs) com limpeza padrao.

### 5. Gerar amostras comparativas

```bash
bash scripts/run_generate_char_samples.sh
# ou exportando apenas o v4 (legacy 3.2k):
bash scripts/run_generate_subword_samples.sh
# demais variantes:
MODEL_KEYS=v4_brwac_subword_v4k_bf_ep2 OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4k_bf_ep2.json ./run_generate_subword_samples.sh
MODEL_KEYS=v4_brwac_subword_v4k_bf_lstm_ep2 OUTPUT_PATH=analysis/artifacts/samples/samples_brwac_v4k_bf_lstm_ep2.json ./run_generate_subword_samples.sh
```

As amostras (temperatura 0.7, `top_k=40`, `top_p=0.95`, seeds 42-46) sao gravadas em
`analysis/artifacts/samples/samples_brwac_20k.json` (char) e nos arquivos `analysis/artifacts/samples/samples_brwac_v4*.json`
(subword). Use `MODEL_KEYS` e `OUTPUT_PATH` para personalizar a geracao.

### 6. Avaliar as amostras com um LLM de referencia

```bash
# Requer OPENAI_API_KEY exportada no ambiente
.venv\Scripts\python.exe analysis\evaluate_samples_with_openai.py ^
  --input analysis/artifacts/samples/samples_brwac_custom_prompts.json ^
  --output analysis/artifacts/results/evaluation_llm_review.json ^
  --model gpt-4o-mini
```

O JSON de saida traz notas de relevancia, coerencia e estilo para cada amostra,
alem de medias por modelo. Esse passo foi usado para encerrar a versao v4.

## Conclusoes por versao

- **v1 (char, LSTM 256)** — baseline pedagógico; resultados limitados por repetição e perplexidade alta, mas estabeleceu pipeline básico de geração e salvamento de artefatos (`analysis/artifacts/results/results_char_models_20k.json`).
- **v2 (char, LSTM 512 + tf.data)** — consolidou ganhos de throughput e qualidade sobre v1 (perplexidade ~4.2), servindo como baseline char-level para comparações futuras.
- **v3 (char, LSTM empilhada)** — atingiu melhor perplexidade char-level (~3.96) com estabilidade de treino; amostras mais coerentes e logs completos em `versions/v3-stacked-lstm/logs`.
- **v4 (subword, tokenizer 4k + fallback)** — removeu `<unk>`, entregou textos legíveis em GRU/LSTM, validado por avaliação externa (`analysis/artifacts/results/evaluation_llm_review.json`); pronto para transição a Transformers (v5 → v5b → v6).

## Conclusoes da versao v4

- O tokenizer SentencePiece 4k + byte fallback eliminou `<unk>` e ampliou o
  vocabulario efetivo sem penalizar o pipeline de treino (`train_char_model_brwac.py`).
- Os modelos GRU/LSTM com `tamanho_sequencia=192` geram textos legiveis; ainda ha
  repeticao com numeros zerados devido a normalizacao de digitos no corpus.
- A avaliacao automatica (`analysis/artifacts/results/evaluation_llm_review.json`, gpt-4o-mini)
  confirma ganhos qualitativos das variantes subword frente aos char-level,
  mesmo que as notas de relevancia permaneçam limitadas pelo conjunto de 2 epocas.
- O objetivo do capitulo v4 foi cumprido: temos baseline subword consistente para
  evoluir rumo a arquiteturas Transformer (roadmap v5 → v5b → v6).

## Resultados atuais (BrWaC 20k)

| Versao | Tokenizacao | Epocas | Loss (eval) | PPL | Acc | Observacoes |
|--------|-------------|-------:|------------:|----:|----:|-------------|
| v1     | Char (160)  | 2 | 1.5218 | 4.58 | 0.550 | Baseline emb64 + LSTM256. |
| v2     | Char (160)  | 2 | 1.4366 | 4.21 | 0.567 | `tf.data`, LSTM512. |
| v3     | Char (160)  | 2 | 1.3753 | 3.96 | 0.582 | LSTM empilhada 512→512. |
| v4     | Subword BPE 3.2k | 6 | 4.2867 | 72.73 | 0.236 | GRU; gera `<unk>` nas amostras. |
| v4 (4k BF) | Subword BPE 4.0k + byte fallback | 2 | 4.4345 | 84.31 | 0.226 | GRU; textos legíveis. |
| v4 (4k BF LSTM) | Subword BPE 4.0k + byte fallback | 2 | 4.4470 | 85.37 | 0.226 | LSTM; comportamento semelhante ao GRU. |

Logs e detalhes completos encontram-se em `analysis/docs/evaluation_char_models.md`.

## Ambiente e dependencias

- Utilize Python 3.10+ com TensorFlow 2.20 (GPU opcional, mas recomendado para treinos).
- Dependencias extras: `sentencepiece>=0.1.99` para o pipeline subword.
- `.venv/`, `__pycache__/` e derivados ja estao no `.gitignore`, entao voce pode manter
  ambientes locais na raiz sem afetar o repositorio.
- Scripts em `scripts/infra/` auxiliam na configuracao de instancias VAST.ai (instalacao de
  pacotes, validacoes de GPU, etc.).

## Proximos passos

Consulte `analysis/docs/roadmap_modelos.md` para a visao geral da evolucao planejada:

1. v4  tokenizacao SentencePiece + LSTM/GRU.
2. v5  Transformer encoder pequeno (opcional).
3. v6  Transformer (tokenizer unigram) e consolidação.

Cada etapa registra objetivos, metricas esperadas e artefatos necessarios para o TCC.
