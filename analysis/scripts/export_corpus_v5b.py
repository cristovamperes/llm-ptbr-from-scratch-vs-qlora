#!/usr/bin/env python3
"""
Exporta um corpus limpo específico para a linha v5b.

Diferenças principais em relação aos exports anteriores:
  * Linhas com ≥80 caracteres úteis e relação alfa > 0.6.
  * Mantém caixa original (sem lower) e números.
  * Opcionalmente remove documentos com porcentagem alta de byte fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportar corpus limpo para a V5b.")
    parser.add_argument("--output", type=Path, required=True, help="Arquivo destino (UTF-8).")
    parser.add_argument("--max-docs", type=int, default=80000, help="Quantidade máxima de documentos.")
    parser.add_argument("--min-len", type=int, default=200, help="Comprimento mínimo por documento após limpeza.")
    parser.add_argument("--min-line-chars", type=int, default=80, help="Comprimento mínimo por linha.")
    parser.add_argument("--min-alpha-ratio", type=float, default=0.6, help="Proporção mínima de letras por linha.")
    parser.add_argument("--end-inline-sep", choices=["space", "newline"], default="newline")
    parser.add_argument("--keep-numbers", action="store_true", default=True)
    parser.add_argument("--keep-upper-metadata", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("nlpufg/brwac", split="train")
    dataset = dataset.shuffle(seed=args.seed)

    end_sep = resolve_end_inline_sep(args.end_inline_sep)
    clean_kwargs = dict(
        lowercase=False,
        end_inline_sep=end_sep,
        min_line_chars=args.min_line_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
    )

    written = 0
    skipped_short = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for example in dataset:
            if written >= args.max_docs:
                break
            cleaned = preparar_texto(example["text"], **clean_kwargs)
            if len(cleaned) < args.min_len:
                skipped_short += 1
                continue
            fh.write(cleaned.strip() + "\n\n")
            written += 1

    print(f"[V5B] Corpus salvo em {args.output} | docs: {written} | pulados (curtos): {skipped_short}")


if __name__ == "__main__":
    main()
