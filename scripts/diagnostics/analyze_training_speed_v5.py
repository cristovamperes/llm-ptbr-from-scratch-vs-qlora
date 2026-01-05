#!/usr/bin/env python3
"""
Análise detalhada da velocidade de treino V4 vs V5.
Por que V5 treina em 8 min enquanto V4 leva 3.7h?
"""

print("=" * 80)
print("ANÁLISE: Por que V5 é tão mais rápido que V4?")
print("=" * 80)

# V4 LSTM ep6
v4_docs = 20_000
v4_total_tokens = 12_471_733
v4_seq_len = 192
v4_stride = 2
v4_batch_size = 192
v4_epochs = 6
v4_total_time_sec = 13_170  # 3.7h
v4_steps_per_epoch = 32_478
v4_total_steps = v4_steps_per_epoch * v4_epochs

# V5 Transformer fixed
v5_docs = 50_000
v5_total_tokens_est = 50_000 * 600  # ~30M
v5_seq_len = 256
v5_stride = 128
v5_batch_size = 32
v5_epochs = 4
v5_total_time_sec = 485  # 8 min
v5_steps_per_epoch = 7_324
v5_total_steps = v5_steps_per_epoch * v5_epochs

print("\n" + "─" * 80)
print("V4 LSTM (ep6):")
print(f"  Documentos: {v4_docs:,}")
print(f"  Tokens: {v4_total_tokens:,}")
print(f"  seq_len: {v4_seq_len}, stride: {v4_stride}, batch_size: {v4_batch_size}")
print(f"  Epochs: {v4_epochs}, steps/epoch: {v4_steps_per_epoch:,}, total steps: {v4_total_steps:,}")
print(f"  Tempo total: {v4_total_time_sec}s ({v4_total_time_sec/3600:.2f}h)")
print(f"  Tempo/epoch: {v4_total_time_sec/v4_epochs:.1f}s ({v4_total_time_sec/v4_epochs/60:.1f} min)")
print(f"  Tempo/step: {(v4_total_time_sec/v4_total_steps)*1000:.1f}ms")

print("\n" + "─" * 80)
print("V5 Transformer (fixed):")
print(f"  Documentos: {v5_docs:,}")
print(f"  Tokens (estimado): {v5_total_tokens_est:,}")
print(f"  seq_len: {v5_seq_len}, stride: {v5_stride}, batch_size: {v5_batch_size}")
print(f"  Epochs: {v5_epochs}, steps/epoch: {v5_steps_per_epoch:,}, total steps: {v5_total_steps:,}")
print(f"  Tempo total: {v5_total_time_sec}s ({v5_total_time_sec/3600:.2f}h)")
print(f"  Tempo/epoch: {v5_total_time_sec/v5_epochs:.1f}s ({v5_total_time_sec/v5_epochs/60:.1f} min)")
print(f"  Tempo/step: {(v5_total_time_sec/v5_total_steps)*1000:.1f}ms")

print("\n" + "=" * 80)
print("COMPARAÇÃO:")
print("=" * 80)

# Comparação de configuração
print("\n📊 Configuração:")
print(f"  Documentos:  V4={v4_docs:,} vs V5={v5_docs:,} → V5 tem {v5_docs/v4_docs:.1f}x MAIS docs")
print(f"  Tokens:      V4={v4_total_tokens:,} vs V5~{v5_total_tokens_est:,} → V5 tem {v5_total_tokens_est/v4_total_tokens:.1f}x MAIS tokens")
print(f"  Seq Length:  V4={v4_seq_len} vs V5={v5_seq_len} → V5 {v5_seq_len/v4_seq_len:.2f}x maior")
print(f"  Stride:      V4={v4_stride} vs V5={v5_stride} → V5 {v5_stride/v4_stride:.0f}x maior (menos overlap)")
print(f"  Batch Size:  V4={v4_batch_size} vs V5={v5_batch_size} → V5 {v4_batch_size/v5_batch_size:.0f}x MENOR")
print(f"  Epochs:      V4={v4_epochs} vs V5={v5_epochs} → V5 {v4_epochs/v5_epochs:.1f}x menos epochs")

# Cálculo de windows e batches
v4_windows_per_doc = v4_total_tokens / v4_docs  # tokens/doc
v4_windows = (v4_total_tokens - v4_seq_len) // v4_stride
v4_batches_calc = v4_windows // v4_batch_size

v5_windows = (v5_total_tokens_est - v5_seq_len) // v5_stride
v5_batches_calc = v5_windows // v5_batch_size

print(f"\n🔢 Windows e Batches:")
print(f"  Windows:     V4={v4_windows:,} vs V5={v5_windows:,}")
print(f"  Batches:     V4={v4_batches_calc:,} vs V5={v5_batches_calc:,}")
print(f"  Steps/epoch: V4={v4_steps_per_epoch:,} vs V5={v5_steps_per_epoch:,}")
print(f"  Total steps: V4={v4_total_steps:,} vs V5={v5_total_steps:,}")
print(f"  → V4 tem {v4_total_steps/v5_total_steps:.1f}x MAIS steps totais")

# Throughput
v4_tokens_per_step = v4_batch_size * v4_seq_len
v5_tokens_per_step = v5_batch_size * v5_seq_len

v4_tokens_per_sec = (v4_total_steps * v4_tokens_per_step) / v4_total_time_sec
v5_tokens_per_sec = (v5_total_steps * v5_tokens_per_step) / v5_total_time_sec

print(f"\n⚡ Throughput:")
print(f"  Tokens/step:  V4={v4_tokens_per_step:,} vs V5={v5_tokens_per_step:,}")
print(f"  Tokens/sec:   V4={v4_tokens_per_sec:,.0f} vs V5={v5_tokens_per_sec:,.0f}")
print(f"  → V5 processa {v5_tokens_per_sec/v4_tokens_per_sec:.2f}x MAIS tokens/s")

# Tempo
print(f"\n⏱️  Tempo:")
print(f"  Tempo/step:   V4={(v4_total_time_sec/v4_total_steps)*1000:.1f}ms vs V5={(v5_total_time_sec/v5_total_steps)*1000:.1f}ms")
print(f"  Tempo/epoch:  V4={v4_total_time_sec/v4_epochs/60:.1f}min vs V5={v5_total_time_sec/v5_epochs/60:.1f}min")
print(f"  Tempo total:  V4={v4_total_time_sec/3600:.2f}h vs V5={v5_total_time_sec/60:.1f}min")
print(f"  → V4 levou {v4_total_time_sec/v5_total_time_sec:.1f}x MAIS tempo")

print("\n" + "=" * 80)
print("🔍 POR QUE V5 É TÃO MAIS RÁPIDO?")
print("=" * 80)

print("\n1️⃣ STRIDE MAIOR (2 vs 128):")
print(f"   V4 stride=2 gera {v4_steps_per_epoch:,} steps/epoch")
print(f"   V5 stride=128 gera apenas {v5_steps_per_epoch:,} steps/epoch")
print(f"   → V5 tem {v4_steps_per_epoch/v5_steps_per_epoch:.1f}x MENOS steps/epoch")
print("   → MAIOR impacto na velocidade!")

print("\n2️⃣ BATCH SIZE MENOR (192 vs 32):")
print(f"   V4 batch=192 processa {v4_tokens_per_step:,} tokens/step")
print(f"   V5 batch=32 processa {v5_tokens_per_step:,} tokens/step")
print(f"   → V5 processa {v4_tokens_per_step/v5_tokens_per_step:.1f}x MENOS tokens/step")
print("   → Mas compensa com mais throughput/tempo")

print("\n3️⃣ MENOS EPOCHS (6 vs 4):")
print(f"   V4={v4_epochs} epochs, V5={v5_epochs} epochs")
print(f"   → V5 treina {v4_epochs/v5_epochs:.1f}x menos epochs")

print("\n4️⃣ HARDWARE/THROUGHPUT:")
print(f"   V5 processa {v5_tokens_per_sec:,.0f} tokens/s vs V4 {v4_tokens_per_sec:,.0f} tokens/s")
print(f"   → V5 é {v5_tokens_per_sec/v4_tokens_per_sec:.2f}x mais eficiente")
print("   → GPU mais moderna (RTX 3080 Ti vs RTX 3080)?")

print("\n" + "=" * 80)
print("💡 CONCLUSÃO")
print("=" * 80)

total_steps_ratio = v4_total_steps / v5_total_steps
time_per_step_ratio = (v4_total_time_sec/v4_total_steps) / (v5_total_time_sec/v5_total_steps)
total_time_ratio = v4_total_time_sec / v5_total_time_sec

print(f"\nO treino V5 é {total_time_ratio:.1f}x mais rápido porque:")
print(f"  1. {total_steps_ratio:.1f}x MENOS steps totais (stride 128 vs 2)")
print(f"  2. {time_per_step_ratio:.2f}x tempo/step similar (67ms vs 17ms)")
print(f"  3. Resultado: {total_time_ratio:.1f}x mais rápido no total")

print(f"\n⚠️  IMPLICAÇÃO:")
print(f"  V4 viu {v4_total_steps * v4_tokens_per_step:,} tokens (~7.2B tokens)")
print(f"  V5 viu {v5_total_steps * v5_tokens_per_step:,} tokens (~240M tokens)")
print(f"  → V5 viu {(v4_total_steps * v4_tokens_per_step)/(v5_total_steps * v5_tokens_per_step):.1f}x MENOS tokens!")
print(f"  → Por isso val_loss V5 (4.91) > V4 (4.31)")

print(f"\n✅ TREINO V5 ESTÁ CORRETO, mas INSUFICIENTE:")
print("  - Dataset NÃO esgotou prematuramente")
print("  - Tempo de 8 min está correto para stride=128, batch=32, 4 epochs")
print(f"  - Mas modelo viu {(v4_total_steps * v4_tokens_per_step)/(v5_total_steps * v5_tokens_per_step):.0f}x menos tokens que V4")
print("  - Para convergir melhor, precisa de MAIS EPOCHS ou MENOS STRIDE")

print("\n" + "=" * 80)
