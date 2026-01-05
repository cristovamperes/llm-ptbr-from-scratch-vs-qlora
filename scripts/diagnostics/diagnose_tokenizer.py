#!/usr/bin/env python3
"""
Diagnóstico do tokenizer SentencePiece v4k_bf.
Investiga problema de byte-fallback na decodificação.
"""

import sentencepiece as spm
from pathlib import Path

# Carregar tokenizer
tokenizer_path = Path("versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model")
sp = spm.SentencePieceProcessor()
sp.Load(str(tokenizer_path))

print("=" * 80)
print("DIAGNÓSTICO: Tokenizer SentencePiece v4k_bf")
print("=" * 80)

# Informações básicas
print(f"\nVocab size: {sp.vocab_size()}")
print(f"BOS ID: {sp.bos_id()}")
print(f"EOS ID: {sp.eos_id()}")
print(f"UNK ID: {sp.unk_id()}")
print(f"PAD ID: {sp.pad_id()}")

# Teste com texto limpo
test_text = "O setor de infraestrutura logística brasileira debate novas concessões"
print(f"\n{'─' * 80}")
print("TESTE 1: Texto limpo")
print(f"Input: {test_text}")

# Encode
ids = sp.EncodeAsIds(test_text)
pieces = sp.EncodeAsPieces(test_text)

print(f"\nIDs ({len(ids)}): {ids[:20]}...")
print(f"Pieces ({len(pieces)}): {pieces[:20]}...")

# Decode
decoded = sp.DecodeIds(ids)
print(f"\nDecoded: {decoded}")
print(f"Match original: {decoded == test_text.lower()}")

# Verificar tokens específicos problemáticos dos samples
print(f"\n{'─' * 80}")
print("TESTE 2: Tokens problemáticos dos samples")

# Pegar alguns tokens que aparecem nos samples ruins
problematic_tokens = [
    "clín",
    "encamatos",
    "suasóveisõe",
    "clubeson",
    "plapenho",
    "pen�",
]

print("\nTokens problemáticos encontrados nos samples:")
for token in problematic_tokens:
    ids = sp.EncodeAsIds(token)
    pieces = sp.EncodeAsPieces(token)
    decoded = sp.DecodeIds(ids)
    print(f"  '{token}' -> IDs:{ids} -> Pieces:{pieces} -> Decoded:'{decoded}'")

# Verificar byte-fallback tokens
print(f"\n{'─' * 80}")
print("TESTE 3: Byte-fallback tokens no vocabulário")

byte_fallback_count = 0
byte_fallback_examples = []

for i in range(min(sp.vocab_size(), 4000)):
    piece = sp.IdToPiece(i)
    if piece.startswith('<0x'):  # Byte-fallback token
        byte_fallback_count += 1
        if len(byte_fallback_examples) < 10:
            byte_fallback_examples.append((i, piece))

print(f"\nTotal byte-fallback tokens: {byte_fallback_count}")
print(f"Exemplos: {byte_fallback_examples}")

# Testar decodificação de sequência com byte-fallback
print(f"\n{'─' * 80}")
print("TESTE 4: Decodificar sample real do modelo")

# Pegar primeiros tokens do sample problemático
sample_start = "clín informações viagem claro"
ids_sample = sp.EncodeAsIds(sample_start)
pieces_sample = sp.EncodeAsPieces(sample_start)

print(f"\nSample start: {sample_start}")
print(f"IDs: {ids_sample}")
print(f"Pieces: {pieces_sample}")
print(f"Decoded: {sp.DecodeIds(ids_sample)}")

# Verificar se "clín" é uma palavra válida ou fragmento
print(f"\n{'─' * 80}")
print("TESTE 5: Análise de 'clín' (primeiro token problemático)")

clin_ids = sp.EncodeAsIds("clín")
clin_pieces = sp.EncodeAsPieces("clín")
print(f"'clín' -> IDs: {clin_ids}, Pieces: {clin_pieces}")

# Tentar variações
for variant in ["clínica", "clínico", "clín", "clin"]:
    ids = sp.EncodeAsIds(variant)
    pieces = sp.EncodeAsPieces(variant)
    decoded = sp.DecodeIds(ids)
    print(f"  '{variant}' -> {len(ids)} tokens -> Pieces:{pieces} -> '{decoded}'")

# Testar texto português válido
print(f"\n{'─' * 80}")
print("TESTE 6: Texto português válido longo")

valid_text = """
A infraestrutura logística brasileira enfrenta desafios importantes.
O setor debate novas concessões ferroviárias e metas de produtividade.
As empresas buscam integração com portos para melhorar os corredores de exportação.
"""

ids_valid = sp.EncodeAsIds(valid_text.strip())
decoded_valid = sp.DecodeIds(ids_valid)

print(f"Texto original ({len(valid_text.strip())} chars):")
print(valid_text.strip())
print(f"\nTokens: {len(ids_valid)}")
print(f"\nDecoded ({len(decoded_valid)} chars):")
print(decoded_valid)
print(f"\nMatch: {decoded_valid == valid_text.strip().lower()}")

# Verificar caracteres especiais no decoded
special_chars = [c for c in decoded_valid if ord(c) > 127 or c == '�']
if special_chars:
    print(f"\nCaracteres especiais encontrados: {set(special_chars)}")
    print(f"Contagem de '�': {decoded_valid.count('�')}")

print("\n" + "=" * 80)
print("CONCLUSÃO:")
print("=" * 80)

# Análise final
if decoded_valid == valid_text.strip().lower():
    print("✅ Tokenizer decodifica corretamente texto português válido")
else:
    print("❌ Tokenizer tem problemas com decodificação de texto português")

if byte_fallback_count > 0:
    print(f"⚠️  Vocabulário contém {byte_fallback_count} tokens de byte-fallback")
    print("   → Isso pode causar fragmentação em palavras não vistas no treino")

print("\n💡 HIPÓTESE:")
print("   Os samples ruins podem ser causados por:")
print("   1. Modelo gerando sequências de tokens inválidas")
print("   2. Temperatura/sampling gerando tokens raros de byte-fallback")
print("   3. Modelo não aprendeu padrões corretos de composição de subwords")
print("=" * 80)
