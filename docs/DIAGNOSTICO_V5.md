# Diagnóstico Completo: V5 Transformer Training

**Data:** 2025-11-02
**Status:** ✅ Problema identificado e correção proposta

---

## 🔍 Problema Relatado

Usuário reportou que os samples da v5 Transformer estavam **piores que v4**, quebrando a progressão esperada v1 > v2 > v3 > v4 > v5.

---

## 📊 Comparação V4 vs V5 (Antes da Correção)

| Métrica | V4 LSTM (ep6) | V5 Transformer (ep4) | Análise |
|---------|---------------|----------------------|---------|
| **Documentos** | 20k | 50k | V5 tem 2.5x mais |
| **Tokens** | 12.5M | ~30M | V5 tem 2.4x mais |
| **Seq Length** | 192 | 256 | V5 maior contexto |
| **Stride** | **2** | **128** | ⚠️ V5 tem 64x menos overlap |
| **Batch Size** | 192 | 32 | V5 tem 6x menor |
| **Epochs** | 6 | 4 | V5 treinou menos |
| | | | |
| **Steps/epoch** | 32,478 | 7,324 | V5 tem 4.4x menos |
| **Total Steps** | 194,868 | 29,296 | V5 tem 6.7x menos |
| **Tokens Vistos** | **7.2B** | **240M** | ⚠️ V5 viu 30x MENOS! |
| | | | |
| **Tempo Total** | 3.7h | 8 min | V5 27x mais rápido |
| **Val Loss** | **4.31** | **4.91** | ❌ V5 pior |
| **Val Accuracy** | 23.83% | 23.86% | Similar |
| **Val Perplexity** | 74.4 | 135.4 | ❌ V5 muito pior |

---

## 🐛 Investigações Realizadas

### 1. Hipótese Inicial: BOS Token Missing
**Investigação:** Verificado que `encode_prompt()` não usava `add_bos=True`
**Resultado:** ❌ FALSO - Correção aplicada mas samples idênticos
**Conclusão:** BOS token já estava correto no treinamento

### 2. Hipótese: Dataset Exhaustion Bug
**Investigação:** Verificado cálculo de `steps_per_epoch` usando `seq_len` vs `seq_plus_one`
**Resultado:** ❌ FALSO - Diferença desprezível (1 window, 0 batches)
**Conclusão:** Warning "ran out of data" era red herring

### 3. Hipótese: Training Speed Anomaly
**Investigação:** V5 treinou em 8min vs V4 em 3.7h - dataset bugado?
**Resultado:** ✅ VERDADEIRO MAS LEGÍTIMO
**Conclusão:** Treino está correto, mas configuração insuficiente

---

## ✅ Causa Raiz Identificada

### **STRIDE MUITO ALTO (128) CAUSA UNDERFITTING**

V5 usou `stride=128` (vs V4 `stride=2`), causando:

1. **64x menos overlap entre sequences**
2. **4.4x menos steps por epoch**
3. **6.7x menos steps totais** (mesmo com mais documentos!)
4. **30x menos tokens vistos** durante todo o treino

**Cálculo:**
```
V4: 194,868 steps × 36,864 tokens/step = 7.2B tokens
V5: 29,296 steps × 8,192 tokens/step = 240M tokens
Razão: 7.2B / 240M = 30x MENOS
```

**Consequência:**
Modelo V5 **subtreinado (underfitted)**, por isso:
- Val loss pior (4.91 vs 4.31)
- Val perplexity muito pior (135 vs 74)
- Samples com fragmentação de subwords

---

## 🎯 Solução Proposta

### Configuração Otimizada (`run_v5_main.sh`):

```bash
--max_docs 50000              # Manter (2.5x mais que V4)
--seq_len 256                 # Manter (contexto maior)
--stride 64                   # ⬇️ Reduzir de 128 → dobrar windows
--batch_size 48               # ⬆️ Aumentar de 32 → melhor GPU usage
--epochs 50                   # ⬆️ Aumentar de 4 → ver mais dados
--learning_rate 3e-4          # ⬆️ Aumentar de 1e-4 → convergir mais rápido
--warmup_steps 2000           # ⬆️ Aumentar de 500
--label_smoothing 0.05        # ⬇️ Reduzir de 0.1 → menos regularização
```

### Estimativas:

| Métrica | V5 Fixed (atual) | V5 Main (proposto) | Ganho |
|---------|------------------|--------------------|-------|
| **Steps/epoch** | 7,324 | ~9,765 | 1.3x |
| **Total steps** | 29,296 | ~488,250 | 16.7x |
| **Tokens/epoch** | 59.9M | 119.9M | 2x |
| **Tokens totais** | 240M | ~6B | 25x |
| **Tempo total** | 8 min | ~14h | - |
| **% tokens V4** | 3.3% | 83% | - |

**Resultado esperado:**
- Val loss: ~4.0-4.3 (similar ou melhor que V4)
- Val perplexity: ~55-75 (melhor que V4)
- Samples: palavras completas, menos fragmentação

---

## 📝 Lições Aprendidas

1. **Stride alto (>seq_len/4) é perigoso** para LMs
   - Reduz drasticamente exposição aos dados
   - Causa underfitting mesmo com mais documentos

2. **"Dataset exhaustion" pode ser red herring**
   - Treino pode estar "correto" mas insuficiente
   - Verificar SEMPRE tokens vistos, não só loss

3. **Transformers precisam MUITO mais dados que LSTMs**
   - V5 com 240M tokens: underfit
   - V4 com 7.2B tokens: converge bem

4. **Comparar throughput não é suficiente**
   - V5 e V4 têm ~500k tokens/s similar
   - Mas V5 termina 27x mais rápido por ter 30x menos dados

---

## ⏭️ Próximos Passos

1. ✅ Script `scripts/v5/legacy/run_v5_main.sh` criado
2. ⏳ Executar treino otimizado no servidor remoto (~14h)
3. ⏳ Gerar samples após cada 10 epochs
4. ⏳ Monitorar val_loss convergência
5. ⏳ Comparar samples v5_main com v4 e v5_fixed

---

## 🔗 Arquivos Relacionados

- Script otimizado: [scripts/v5/legacy/run_v5_main.sh](../scripts/v5/legacy/run_v5_main.sh)
- Diagnóstico dataset: [scripts/diagnostics/diagnose_dataset_v5.py](../scripts/diagnostics/diagnose_dataset_v5.py)
- Análise velocidade: [scripts/diagnostics/analyze_training_speed_v5.py](../scripts/diagnostics/analyze_training_speed_v5.py)
- Código treinamento: [scripts/llm_transformer_v5.py](../scripts/llm_transformer_v5.py)
- Geração samples: [analysis/generate_samples.py](../analysis/generate_samples.py)

---

**Autor:** Claude (Anthropic)
**Revisão:** Cristovam Peres
