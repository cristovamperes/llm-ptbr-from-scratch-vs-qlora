#!/usr/bin/env python3
"""
Mede métricas de fragmentação para um tokenizer SentencePiece.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcular métricas de fragmentação de tokens.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Arquivo .model do SentencePiece.")
    parser.add_argument("--report", type=Path, required=True, help="Arquivo JSON de saída.")
    parser.add_argument("--sample-docs", type=int, default=2000, help="Quantidade de documentos a analisar.")
    parser.add_argument("--max-docs", type=int, default=50000, help="Limite superior para iterar o dataset.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_byte_fallback(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def main() -> None:
    args = parse_args()
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    dataset = load_dataset("nlpufg/brwac", split=f"train[:{args.max_docs}]").shuffle(seed=args.seed)

    total_tokens = 0
    total_chars = 0
    docs = 0
    fallback_tokens = 0
    long_fragments = 0

    for example in dataset:
        if docs >= args.sample_docs:
            break
        text = example["text"].strip()
        if not text:
            continue
        ids = sp.EncodeAsIds(text)
        pieces = sp.IdToPiece(ids)
        doc_tokens = len(ids)
        doc_chars = len(text)
        total_tokens += doc_tokens
        total_chars += doc_chars
        docs += 1
        fallback_tokens += sum(1 for piece in pieces if is_byte_fallback(piece))
        long_fragments += sum(1 for piece in pieces if len(piece) <= 2 and piece.isalpha())

    if docs == 0 or total_chars == 0:
        raise RuntimeError("Nenhum documento válido processado.")

    report = {
        "tokenizer": str(args.tokenizer),
        "docs": docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": total_tokens / docs,
        "tokens_per_char_ratio": total_tokens / total_chars,
        "byte_fallback_tokens": fallback_tokens,
        "byte_fallback_ratio": fallback_tokens / total_tokens,
        "short_alpha_pieces": long_fragments,
        "short_alpha_ratio": long_fragments / total_tokens,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[V5B] Fragmentation report salvo em {args.report}")


if __name__ == "__main__":
    main()
