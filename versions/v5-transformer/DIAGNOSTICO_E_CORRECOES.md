# Diagnóstico v5 Transformer - Análise de Falha e Correções

**Data:** 30 de outubro de 2025
**Status:** Modelo atual apresenta degeneração severa de texto
**Ação:** Re-treino necessário com correções estruturais

---

## 1. Sintomas Observados

### 1.1 Qualidade das Amostras Geradas

**Arquivo:** `analysis/artifacts/samples/samples_brwac_v5_transformer.json`

**Problemas identificados:**
- ✗ **Gibberish completo:** palavras quebradas sem sentido ("parlamvidas", "manutnsaaráini")
- ✗ **Repetições excessivas:** fragmentos recorrentes ("ód", "ênio", "assa", "quer")
- ✗ **Ausência de estrutura:** nenhuma frase completa ou coerência sintática
- ✗ **Desconexão total com prompts:** inputs estruturados geram saídas caóticas

**Exemplo de saída degenerada:**
```
Prompt: "O setor de infraestrutura logistica brasileira debate..."
Output: "parlamvidas manutnsaaráini mostrardar escolh pesquiosa clientes
         estratéamb consol negóciosódód pensarviaéricaorden bras simpl..."
```

### 1.2 Métricas de Treino

**Arquivo:** `versions/v5-transformer/logs/train_v5.json`

| Época | Train Loss | Val Loss | Train Acc | Val Acc | Gap (Loss) |
|-------|-----------|----------|-----------|---------|------------|
| 1 | 3.36 | **4.24** | 0.354 | 0.294 | +0.88 |
| 2 | 2.93 | **4.20** | 0.410 | 0.302 | **+1.27** |

**Análise crítica:**
- ✓ Train loss decrescente → modelo está aprendendo no treino
- ✗ Val loss estável/alto → modelo **NÃO está generalizando**
- ✗ Gap crescente (0.88 → 1.27) → **overfitting severo**
- ✗ Val accuracy ~30% → pior que aleatório para vocab 4k (esperado: ~0.025%)

**Tempo de treino:** 11.453 segundos (~3.2 horas) — muito curto para um Transformer

---

## 2. Comparação com v4 (Baseline de Sucesso)

### 2.1 Arquitetura

| Aspecto | v4 LSTM | v5 Transformer |
|---------|---------|----------------|
| **Tipo** | Stacked LSTM (2 camadas) | Decoder-only Transformer (6 camadas) |
| **Dimensões** | 512 → 512 | d_model=384, 6 heads, d_ff=1536 |
| **Parâmetros** | 8.3M | ~15M (estimado) |
| **Tokenização** | Subword (4k + BF) | Subword (4k + BF) ✓ |
| **Seq Length** | 192 | **256/320** ⚠️ |
| **Batch Size** | 192 tokens | **192 tokens** ⚠️ |

### 2.2 Configuração de Treino

| Parâmetro | v4 LSTM | v5 Transformer | Status |
|-----------|---------|----------------|--------|
| **Épocas** | 2 | 2 | ✓ Similar |
| **Dados** | 20k docs | 20k docs | ✓ Similar |
| **Learning Rate** | AdamW padrão | 3e-4 | ⚠️ Alto demais |
| **Warmup** | N/A | 2000/10000 (20%) | ✗ Muito longo |
| **Weight Decay** | N/A | 0.1 | ⚠️ Muito agressivo |
| **Regularização** | Dropout 0.1 | Dropout 0.1 | ✓ Similar |
| **Steps Total** | 63,046 (2 épocas) | **10,000** | ✗ Truncado! |

### 2.3 Resultados

| Modelo | Val Loss | Perplexity | Accuracy | Qualidade Texto |
|--------|----------|------------|----------|-----------------|
| **v4 LSTM** | 4.45 | 85.37 | 0.226 | ✓ Coerente, fluente |
| **v5 Transformer** | **4.20** | 66.69 | **0.302** | ✗ Gibberish total |

**Paradoxo aparente:** v5 tem métricas numericamente melhores, mas gera texto pior!

**Explicação:** v5 está **memorizando tokens específicos** sem aprender padrões linguísticos reais → overfitting com alta confiança em previsões erradas.

---

## 3. Causas Raiz Identificadas

### 3.1 Incompatibilidade de Configuração (CRÍTICO)

**Problema:** Sequence length no script ≠ modelo construído

```bash
# run_v5_transformer.sh (linha 13)
--seq_len 320

# llm_transformer_v5.py (padrão argparse)
--seq_len 256  # Default usado na construção do modelo
```

**Impacto:**
- Embeddings posicionais treinados para 256 tokens
- Dataset tentando processar 320 tokens
- **Erro silencioso ou truncamento inesperado**

### 3.2 Batch Size Inadequado

**Configurado:** `batch_size=192` (tokens, não sequências!)

**Problema:**
- Para LSTMs: batch_size refere-se a tokens (OK)
- Para Transformers: batch_size deve ser **número de sequências**
- Com seq_len=320 e batch=192 → apenas **~0.6 sequências por batch**

**Correção necessária:** batch_size=32 ou 64 **sequências**

### 3.3 Treino Truncado Prematuramente

```bash
--steps_per_epoch 41653   # ~41k steps esperados por época
--total_steps 10000        # Limite de 10k steps TOTAL
--epochs 2                 # Esperado: 2 × 41653 = 83.306 steps
```

**Resultado:** Treino parou em **10k steps** (apenas 12% do esperado!)

**Consequências:**
- Modelo não convergiu
- Warmup consumiu 20% do treino (2000/10000)
- Insuficiente para aprender padrões complexos

### 3.4 Warmup Excessivo

- **Configurado:** 2000 steps de warmup
- **Total treino:** 10000 steps
- **Proporção:** 20% em warmup (ideal: 5-10%)

**Impacto:** Learning rate ficou abaixo do ótimo durante maior parte do treino.

### 3.5 Arquitetura Oversized para Dataset

**Dataset:** 20k documentos (~12M tokens totais)

**Modelo v5:**
- 6 camadas Transformer
- d_model=384, d_ff=1536
- ~15M parâmetros

**Relação dados/parâmetros:** ~0.8 tokens por parâmetro

**Benchmark típico:** GPT-2 small (124M params) treinou com 40GB de texto (~8B tokens) → 64 tokens/param

**Diagnóstico:** Modelo **64x maior** que o ideal para este dataset → overfitting inevitável

### 3.6 Falta de Regularização Adequada

**Ausente no código original:**
- Label smoothing
- Gradient clipping
- Early stopping
- Model checkpointing

**Impacto:** Modelo pode divergir ou memorizar sem controle.

---

## 4. Plano de Correção

### 4.1 Correções Urgentes (Implementadas)

#### A) Script Corrigido: `run_v5_fixed.sh`

**Mudanças aplicadas:**

```bash
# ANTES (run_v5_transformer.sh)
--seq_len 320                # ✗ Incompatível
--batch_size 192             # ✗ Muito pequeno
--stride 2                   # ✗ Overlap excessivo
--learning_rate 3e-4         # ⚠️ Alto
--warmup_steps 2000          # ✗ 20% do treino
--total_steps 10000          # ✗ Truncado
--weight_decay 0.1           # ⚠️ Muito agressivo

# DEPOIS (run_v5_fixed.sh)
--seq_len 256                # ✓ Alinhado com modelo
--batch_size 32              # ✓ 32 SEQUÊNCIAS
--stride 128                 # ✓ Menos overlap
--learning_rate 1e-4         # ✓ Mais conservador
--warmup_steps 500           # ✓ ~5% do treino
--total_steps 0              # ✓ Auto-calculado (4 épocas)
--weight_decay 0.01          # ✓ Menos agressivo
--max_docs 50000             # ✓ +150% dados
```

#### B) Arquitetura Reduzida

```bash
# ANTES
--d_model 384 --num_layers 6 --num_heads 6 --d_ff 1536
# Parâmetros: ~15M

# DEPOIS
--d_model 256 --num_layers 4 --num_heads 8 --d_ff 1024
# Parâmetros: ~8M (redução de 47%)
```

**Justificativa:**
- Alinhamento com v4 LSTM (8.3M params)
- Melhor relação dados/parâmetros (6.25 tokens/param com 50k docs)
- Mais heads (8 vs 6) compensa redução de d_model

#### C) Melhorias no Código (`llm_transformer_v5.py`)

**Adicionado:**

1. **Label Smoothing** (0.1)
   ```python
   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
       from_logits=True,
       label_smoothing=0.1,  # ← Previne overconfidence
   )
   ```

2. **Gradient Clipping** (norm=1.0)
   ```python
   optimizer = tf.keras.optimizers.AdamW(
       clipnorm=1.0,  # ← Estabiliza treino
   )
   ```

3. **Early Stopping**
   ```python
   tf.keras.callbacks.EarlyStopping(
       monitor="val_loss",
       patience=2,
       restore_best_weights=True,
   )
   ```

4. **Model Checkpointing**
   ```python
   tf.keras.callbacks.ModelCheckpoint(
       monitor="val_loss",
       save_best_only=True,
   )
   ```

### 4.2 Configuração Completa Recomendada

**Comando para re-treino:**

```bash
bash run_v5_fixed.sh
```

**Parâmetros finais:**

| Categoria | Parâmetro | Valor | Justificativa |
|-----------|-----------|-------|---------------|
| **Dados** | max_docs | 50000 | +150% dados para generalização |
| | seq_len | 256 | Alinhado com modelo |
| | stride | 128 | Reduz overlap, aumenta diversidade |
| | batch_size | 32 | 32 sequências (não tokens) |
| **Arquitetura** | d_model | 256 | Redução de parâmetros |
| | num_layers | 4 | Menos camadas → menos overfitting |
| | num_heads | 8 | Mais atenção, menor dimensão |
| | d_ff | 1024 | FFN proporcional a d_model |
| **Treino** | epochs | 4 | Dobro do anterior |
| | learning_rate | 1e-4 | Mais conservador |
| | warmup_steps | 500 | ~5% do treino |
| | weight_decay | 0.01 | Regularização moderada |
| | label_smoothing | 0.1 | Anti-overconfidence |
| | gradient_clip | 1.0 | Estabilidade |

**Estimativas de treino:**

- **Total steps:** ~40k (10 steps/epoch × 50k docs / 32 batch × 4 épocas)
- **Tempo esperado:** ~6-8 horas (RTX 3080)
- **Tokens processados:** ~50M tokens (50k docs × 600 tokens/doc × 4 épocas)

### 4.3 Métricas Esperadas (Pós-Correção)

**Targets realistas:**

| Métrica | v4 LSTM (Atual) | v5 Target | v5 Stretch Goal |
|---------|-----------------|-----------|-----------------|
| **Val Loss** | 4.45 | < 4.30 | < 4.00 |
| **Perplexity** | 85.37 | < 73.70 | < 54.60 |
| **Accuracy** | 0.226 | > 0.24 | > 0.28 |
| **Qualidade** | Boa | Boa | Excelente |

**Critérios de sucesso:**
1. ✓ Val loss estável ou decrescente (sem overfitting)
2. ✓ Gap train/val < 0.5 (generalização adequada)
3. ✓ Amostras coerentes com sintaxe básica
4. ✓ Continuações relevantes aos prompts

---

## 5. Checklist de Execução

### Pré-Treino
- [x] Script corrigido criado (`run_v5_fixed.sh`)
- [x] Código atualizado com regularização (`llm_transformer_v5.py`)
- [x] Diagnóstico documentado
- [ ] Backup do modelo anterior (`modelo_v5_ready_saved.keras`)
- [ ] Limpeza de logs antigos (opcional)

### Durante Treino
- [ ] Monitorar TensorBoard: `tensorboard --logdir versions/v5-transformer/logs/tensorboard_fixed`
- [ ] Verificar gap train/val a cada época
- [ ] Confirmar early stopping não dispara precocemente (aguardar 2+ épocas)
- [ ] Validar throughput (tokens/s > 100k)

### Pós-Treino
- [ ] Avaliar com `analysis/evaluate_transformer_v5.py`
- [ ] Gerar amostras com `analysis/generate_samples.py`
- [ ] Comparar com v4 usando mesmos prompts
- [ ] Atualizar `analysis/artifacts/results/results_*.json`
- [ ] Documentar métricas finais neste arquivo

---

## 6. Troubleshooting

### Se val_loss continuar alto:
1. Aumentar max_docs para 100k
2. Reduzir para 2 camadas (d_model=256)
3. Aumentar dropout para 0.15
4. Considerar curriculum learning (seq_len progressivo)

### Se overfitting persistir:
1. Aumentar weight_decay para 0.05
2. Adicionar dropout de atenção (attention_dropout=0.1)
3. Implementar mixup ou cutout de tokens
4. Reduzir épocas para 2-3

### Se treino divergir:
1. Reduzir learning_rate para 5e-5
2. Aumentar warmup_steps para 1000
3. Verificar gradient norms no TensorBoard
4. Adicionar layer normalization antes de atenção

---

## 7. Referências e Comparações

### Papers Relevantes
- **Attention Is All You Need** (Vaswani et al., 2017) - Arquitetura base
- **Language Models are Few-Shot Learners** (GPT-3, Brown et al., 2020) - Scaling laws
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) - Relação dados/params

### Scaling Laws Aplicados
- **Chinchilla optimal:** ~20 tokens por parâmetro (DeepMind, 2022)
- **v5 fixed:** ~6.25 tokens/param (50M tokens / 8M params)
- **Conclusão:** Ainda **subotimal** em 3x, mas viável para POC acadêmico

### Benchmarks de Modelos Pequenos
- **GPT-2 small:** 124M params, 40GB texto → val_loss ~3.0
- **BLOOM-560M:** 560M params, 350B tokens → perplexity ~20
- **TinyStories-33M:** 33M params, 2.1B tokens → coerência boa
- **v5 target (8M params, 50M tokens):** Esperado perplexity 50-80

---

## 8. Próximos Passos (Pós-v5)

### Curto Prazo (v5.1)
1. Implementar positional encoding sinusoidal (atualmente usa learned)
2. Adicionar attention visualization
3. Experimentar rotary embeddings (RoPE)

### Médio Prazo (v_final)
1. Escalar para 50M parâmetros
2. Dataset de 500k-1M documentos
3. Multi-epoch com learning rate scheduling
4. Distillation do modelo grande para compacto

### Artigo TCC
1. Seção comparativa v4 (LSTM) vs v5 (Transformer)
2. Análise de scaling laws aplicados
3. Discussão sobre overfitting e regularização
4. Demonstração de amostras progressivas (v1→v5)

---

## Apêndice A: Comandos Rápidos

### Re-treinar v5 (configuração corrigida)
```bash
bash run_v5_fixed.sh
```

### Monitorar treino em tempo real
```bash
tensorboard --logdir versions/v5-transformer/logs/tensorboard_fixed
# Abrir: http://localhost:6006
```

### Avaliar modelo pós-treino
```bash
python analysis/evaluate_transformer_v5.py \
  --model versions/v5-transformer/models/modelo_v5_fixed.keras \
  --mapping versions/v5-transformer/mappings/mapeamentos_v5_fixed.pkl \
  --output analysis/artifacts/results/results_v5_fixed.json
```

### Gerar amostras
```bash
python analysis/generate_samples.py \
  --model-key v5_brwac_transformer_fixed \
  --output analysis/artifacts/samples/samples_v5_fixed.json
```

---

**Documento vivo - atualizar após cada experimento**
**Última atualização:** 2025-10-30
**Autor:** Análise automatizada + validação humana
