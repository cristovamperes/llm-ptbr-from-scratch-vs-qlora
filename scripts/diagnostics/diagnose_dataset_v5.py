#!/usr/bin/env python3
"""
Script para diagnosticar o problema de dataset exhaustion no v5.
Calcula windows e batches esperados vs reais.
"""

import sys
from pathlib import Path

# Parâmetros do treino v5_fixed
seq_len = 256
stride = 128
batch_size = 32
max_docs = 50000

# Estimativa conservadora de tokens (baseada em v4: ~600 tokens/doc)
avg_tokens_per_doc = 600
total_tokens_estimate = max_docs * avg_tokens_per_doc

print("=" * 70)
print("DIAGNÓSTICO: Dataset Exhaustion V5 Transformer")
print("=" * 70)
print(f"\nParâmetros de Treino:")
print(f"  - max_docs: {max_docs}")
print(f"  - seq_len: {seq_len}")
print(f"  - stride: {stride}")
print(f"  - batch_size: {batch_size}")
print(f"  - avg_tokens_per_doc (estimado): {avg_tokens_per_doc}")
print(f"  - total_tokens (estimado): {total_tokens_estimate:,}")

# Cálculo ERRADO (como está no código linha 389-390)
windows_wrong = max(0, (total_tokens_estimate - seq_len) // stride)
batches_wrong = windows_wrong // batch_size

# Cálculo CORRETO (seq_plus_one)
seq_plus_one = seq_len + 1  # 257
windows_correct = max(0, (total_tokens_estimate - seq_plus_one) // stride)
batches_correct = windows_correct // batch_size

print(f"\n{'─' * 70}")
print("CÁLCULO COM BUG (linha 389-390: usa seq_len):")
print(f"  windows = ({total_tokens_estimate:,} - {seq_len}) // {stride} = {windows_wrong:,}")
print(f"  batches = {windows_wrong:,} // {batch_size} = {batches_wrong:,}")
print(f"  → steps_per_epoch usado: {batches_wrong:,}")

print(f"\n{'─' * 70}")
print("CÁLCULO CORRETO (dataset real usa seq_plus_one=257):")
print(f"  windows = ({total_tokens_estimate:,} - {seq_plus_one}) // {stride} = {windows_correct:,}")
print(f"  batches = {windows_correct:,} // {batch_size} = {batches_correct:,}")
print(f"  → steps_per_epoch real: {batches_correct:,}")

print(f"\n{'─' * 70}")
print("DIFERENÇA:")
diff_windows = windows_wrong - windows_correct
diff_batches = batches_wrong - batches_correct
diff_percent = (diff_batches / batches_correct * 100) if batches_correct > 0 else 0
print(f"  Δ windows: {diff_windows:,} ({diff_windows / windows_correct * 100:.2f}% a mais)")
print(f"  Δ batches: {diff_batches:,} ({diff_percent:.2f}% a mais)")
print(f"  → Keras espera {batches_wrong:,} steps mas dataset só tem {batches_correct:,}")

# Análise do treino observado (485s para 4 epochs, 7324 steps/epoch)
observed_steps_per_epoch = 7324
observed_total_time_sec = 485
observed_time_per_step_ms = (observed_total_time_sec / 4 / observed_steps_per_epoch) * 1000

print(f"\n{'─' * 70}")
print("TREINO OBSERVADO (train_v5_fixed.json):")
print(f"  steps_per_epoch observado: {observed_steps_per_epoch:,}")
print(f"  tempo total: {observed_total_time_sec}s (~{observed_total_time_sec/60:.1f} min)")
print(f"  tempo/step: ~{observed_time_per_step_ms:.1f}ms")

print(f"\n{'─' * 70}")
print("ANÁLISE:")
if observed_steps_per_epoch == batches_wrong:
    print(f"  ✅ steps_per_epoch={observed_steps_per_epoch:,} CORRESPONDE ao cálculo BUGADO")
    print(f"  ❌ Mas dataset real só tem ~{batches_correct:,} batches")
    print(f"  → Dataset esgota após ~{batches_correct:,} steps, Keras espera {batches_wrong:,}")
    print(f"  → Warning 'Your input ran out of data' é ESPERADO")
elif observed_steps_per_epoch == batches_correct:
    print(f"  ✅ steps_per_epoch={observed_steps_per_epoch:,} CORRESPONDE ao cálculo CORRETO")
    print(f"  → Dataset não deveria esgotar")
else:
    print(f"  ⚠️  steps_per_epoch={observed_steps_per_epoch:,} não corresponde a nenhum cálculo")
    print(f"     Esperado (bug): {batches_wrong:,}")
    print(f"     Esperado (correto): {batches_correct:,}")

# Tempo esperado
tokens_per_step = batch_size * seq_len
tokens_per_epoch_expected = tokens_per_step * observed_steps_per_epoch
v4_tokens_per_sec = 545455  # Da análise v4_ep6
expected_time_per_epoch_sec = tokens_per_epoch_expected / v4_tokens_per_sec
expected_total_time_sec = expected_time_per_epoch_sec * 4

print(f"\n{'─' * 70}")
print("TEMPO ESPERADO (baseado em throughput do V4):")
print(f"  tokens/step: {tokens_per_step:,}")
print(f"  tokens/epoch: {tokens_per_epoch_expected:,}")
print(f"  V4 throughput: ~{v4_tokens_per_sec:,} tokens/s")
print(f"  Tempo esperado/epoch: ~{expected_time_per_epoch_sec:.0f}s (~{expected_time_per_epoch_sec/60:.1f} min)")
print(f"  Tempo total esperado (4 epochs): ~{expected_total_time_sec:.0f}s (~{expected_total_time_sec/3600:.1f}h)")
print(f"  Tempo REAL: {observed_total_time_sec}s (~{observed_total_time_sec/60:.1f} min)")
print(f"  → Treino foi {expected_total_time_sec / observed_total_time_sec:.1f}x mais RÁPIDO que esperado!")

print(f"\n{'═' * 70}")
print("CONCLUSÃO:")
print("  1. BUG CONFIRMADO: Cálculo de batches usa seq_len, mas dataset usa seq_plus_one")
print(f"  2. Keras configura steps_per_epoch={batches_wrong:,} (cálculo bugado)")
print(f"  3. Dataset real só tem ~{batches_correct:,} batches disponíveis")
print(f"  4. Diferença: ~{diff_batches:,} steps ({diff_percent:.1f}%) FALTANDO")
print("  5. Dataset esgota prematuramente, gerando warning")
print("  6. Treino continua mas sem dados novos (possivelmente repete últimos)")
print(f"\n  → FIX: Corrigir linha 389 para usar seq_plus_one em vez de seq_len")
print("=" * 70)
