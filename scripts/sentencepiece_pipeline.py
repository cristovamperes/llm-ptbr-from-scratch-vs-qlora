#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities to train and inspect SentencePiece tokenizers tailored for BrWaC.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from tcc_llm.brwac_dataset import CorpusExportResult, export_clean_corpus, iter_clean_documents
from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep
from tcc_llm.sentencepiece_utils import decode_ids, encode_text, load_processor, train_sentencepiece


def _default_clean_kwargs(args: argparse.Namespace) -> dict:
    end_inline_sep = resolve_end_inline_sep(args.end_inline_sep)
    return dict(
        lowercase=not args.no_lowercase,
        end_inline_sep=end_inline_sep,
        min_line_chars=args.min_line_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
    )


def _ensure_corpus_file(args: argparse.Namespace) -> CorpusExportResult:
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            raise SystemExit(f"Arquivo de entrada nao encontrado: {path}")
        text = path.read_text(encoding="utf-8")
        count = text.count("\n\n") + 1
        return CorpusExportResult(count=count, path=path, total_chars=len(text))

    dataset = load_dataset("nlpufg/brwac", split="train")
    dataset = dataset.shuffle(seed=args.seed)
    max_docs = min(args.limit, len(dataset))
    clean_kwargs = _default_clean_kwargs(args)
    print(f"[INFO] Exportando {max_docs} documentos limpos para treinar o tokenizer...")
    result = export_clean_corpus(
        dataset,
        max_docs,
        min_len=args.min_len,
        clean_kwargs=clean_kwargs,
    )
    print(f"[INFO] Corpus salvo em {result.path} | docs={result.count} | chars={result.total_chars}")
    return result


def cmd_train(args: argparse.Namespace) -> None:
    corpus = _ensure_corpus_file(args)
    extra_options = []
    if args.hard_vocab_limit is not None:
        extra_options.append(f"--hard_vocab_limit={'true' if args.hard_vocab_limit else 'false'}")
    if args.byte_fallback:
        extra_options.append("--byte_fallback=true")
    if args.max_sentence_length:
        extra_options.append(f"--max_sentence_length={args.max_sentence_length}")

    user_symbols = args.user_symbol or []
    extra_user_options = [f"--user_defined_symbols={symbol}" for symbol in user_symbols]

    paths = train_sentencepiece(
        corpus.path,
        Path(args.output_dir),
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=not args.no_shuffle_sentences,
        seed=args.seed,
        extra_options=extra_options + extra_user_options,
    )
    print(f"[INFO] Modelo SentencePiece salvo em {paths.model}")
    print(f"[INFO] Vocabulario salvo em {paths.vocab}")

    if args.meta_output:
        metadata = {
            "vocab_size": args.vocab_size,
            "model_type": args.model_type,
            "character_coverage": args.character_coverage,
            "input_sentence_size": args.input_sentence_size,
            "byte_fallback": args.byte_fallback,
            "limit_docs": args.limit,
            "min_len": args.min_len,
            "corpus_chars": corpus.total_chars,
            "corpus_docs": corpus.count,
            "cleaning": _default_clean_kwargs(args),
            "model_path": str(paths.model),
            "vocab_path": str(paths.vocab),
        }
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.meta_output, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        print(f"[INFO] Metadata registrada em {args.meta_output}")

    if args.dry_run_docs > 0:
        processor = load_processor(paths.model)
        dry_run_docs = min(args.dry_run_docs, corpus.count)
        print(f"[INFO] Calculando estatisticas de tokens com {dry_run_docs} documentos...")
        with open(corpus.path, "r", encoding="utf-8") as fh:
            texts = fh.read().strip().split("\n\n")[:dry_run_docs]
        stats = compute_stats(texts, processor)
        print_stats(stats)
        if args.stats_output:
            Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_output, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, ensure_ascii=False, indent=2)
            print(f"[INFO] Estatisticas salvas em {args.stats_output}")


def compute_stats(texts: Sequence[str], processor, *, include_char_stats: bool = True) -> dict:
    char_lengths = [len(text) for text in texts]
    token_lengths = [len(encode_text(text, processor, add_bos=False, add_eos=False)) for text in texts]

    def _percentiles(values: Sequence[int]) -> dict:
        arr = np.array(values, dtype=np.int64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }

    stats = {
        "docs": len(texts),
        "total_tokens": int(sum(token_lengths)),
        "avg_tokens_per_doc": float(statistics.fmean(token_lengths)) if token_lengths else 0.0,
        "median_tokens_per_doc": float(statistics.median(token_lengths)) if token_lengths else 0.0,
        "percentiles_tokens": _percentiles(token_lengths) if token_lengths else {},
    }
    if include_char_stats:
        total_chars = int(sum(char_lengths))
        avg_chars = float(statistics.fmean(char_lengths)) if char_lengths else 0.0
        tokens_per_char_ratio = float(stats["total_tokens"] / total_chars) if total_chars else 0.0
        stats.update(
            {
                "total_chars": total_chars,
                "avg_chars_per_doc": avg_chars,
                "tokens_per_char_ratio": tokens_per_char_ratio,
            }
        )
    return stats


def print_stats(stats: dict) -> None:
    print(
        "[INFO] Docs: {docs} | tokens totais: {total_tokens} | "
        "media tokens/doc: {avg_tokens_per_doc:.2f} | mediana: {median_tokens_per_doc:.2f}".format(**stats)
    )
    if "total_chars" in stats:
        print(
            "[INFO] Caracteres totais: {total_chars} | media chars/doc: {avg_chars_per_doc:.2f} | "
            "tokens/char: {tokens_per_char_ratio:.4f}".format(**stats)
        )
    percentiles = stats.get("percentiles_tokens")
    if percentiles:
        print(
            "[INFO] Percentis tokens/doc -> p50: {p50:.0f}, p75: {p75:.0f}, p90: {p90:.0f}, "
            "p95: {p95:.0f}, p99: {p99:.0f}, max: {max:.0f}".format(**percentiles)
        )


def cmd_encode(args: argparse.Namespace) -> None:
    processor = load_processor(args.tokenizer)
    clean_kwargs = _default_clean_kwargs(args)

    if args.text is None and args.file is None:
        raise SystemExit("Forneca --text ou --file")

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        content = args.text

    if args.clean:
        content = preparar_texto(content, **clean_kwargs)

    ids = encode_text(content, processor, add_bos=args.add_bos, add_eos=args.add_eos)
    serialized = " ".join(str(idx) for idx in ids)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


def _parse_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for chunk in raw.replace(",", " ").split():
        chunk = chunk.strip()
        if not chunk:
            continue
        ids.append(int(chunk))
    return ids


def cmd_decode(args: argparse.Namespace) -> None:
    processor = load_processor(args.tokenizer)
    if args.ids is None and args.file is None:
        raise SystemExit("Forneca --ids ou --file com IDs")

    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
    else:
        raw = args.ids

    ids = _parse_ids(raw)
    text = decode_ids(ids, processor, skip_special=args.skip_special)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


def cmd_stats(args: argparse.Namespace) -> None:
    processor = load_processor(args.tokenizer)
    clean_kwargs = _default_clean_kwargs(args)
    dataset = load_dataset("nlpufg/brwac", split=f"train[:{args.limit}]")
    texts = []
    for texto in iter_clean_documents(dataset, args.limit, min_len=args.min_len, clean_kwargs=clean_kwargs):
        texts.append(texto)
        if len(texts) >= args.sample_docs:
            break
    stats = compute_stats(texts, processor)
    print_stats(stats)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)
        print(f"[INFO] Estatisticas salvas em {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ferramentas SentencePiece adaptadas ao BrWaC.")
    sub = parser.add_subparsers(dest="command")

    def add_cleaning_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--no-lowercase", action="store_true", help="Nao converter para minusculas.")
        subparser.add_argument("--keep-numbers", action="store_true", help="Nao normalizar numeros.")
        subparser.add_argument(
            "--keep-upper-metadata",
            action="store_true",
            help="Preservar linhas em caixa alta (util para manchetes).",
        )
        subparser.add_argument(
            "--end-inline-sep", choices=["space", "newline"], default="newline", help="Separador para marcadores <END>."
        )
        subparser.add_argument("--min-line-chars", type=int, default=40, help="Comprimento minimo por linha valida.")
        subparser.add_argument("--min-alpha-ratio", type=float, default=0.4, help="Proporcao minima de letras por linha.")

    train_parser = sub.add_parser("train", help="Treinar um tokenizador SentencePiece a partir do BrWaC.")
    train_parser.add_argument("--output-dir", type=str, default="versions/v4-subword-lstm/tokenizer")
    train_parser.add_argument("--model-prefix", type=str, default="spm_v4")
    train_parser.add_argument("--vocab-size", type=int, default=3200)
    train_parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram", "word", "char"])
    train_parser.add_argument("--character-coverage", type=float, default=0.9995)
    train_parser.add_argument("--input-sentence-size", type=int, default=2_000_000)
    train_parser.add_argument("--no-shuffle-sentences", action="store_true", help="Nao embaralhar sentencas ao samplear.")
    train_parser.add_argument("--limit", type=int, default=20000, help="Numero maximo de documentos do BrWaC.")
    train_parser.add_argument("--min-len", type=int, default=200, help="Comprimento minimo por documento.")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--input-file", type=str, default="", help="Arquivo ja limpo para treinar (skip dataset).")
    train_parser.add_argument("--byte-fallback", action="store_true", help="Ativar byte fallback no SentencePiece.")
    train_parser.add_argument(
        "--user-symbol",
        action="append",
        default=[],
        help="Adicionar símbolo personalizado ao vocabulário (pode repetir).",
    )
    train_parser.add_argument("--hard-vocab-limit", type=bool, default=None, help="Forcar limite fixo de vocabulario.")
    train_parser.add_argument("--max-sentence-length", type=int, default=0, help="Comprimento maximo de sentenca.")
    train_parser.add_argument("--meta-output", type=str, default="", help="Opcional: salvar metadata em JSON.")
    train_parser.add_argument("--dry-run-docs", type=int, default=0, help="Quantos documentos usar para dry-run.")
    train_parser.add_argument("--stats-output", type=str, default="", help="Caminho para salvar estatisticas (JSON).")
    add_cleaning_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    encode_parser = sub.add_parser("encode", help="Codificar um texto usando SentencePiece.")
    encode_parser.add_argument("--tokenizer", required=True, help="Caminho para o .model SentencePiece.")
    encode_parser.add_argument("--text", type=str, help="Texto bruto a ser codificado.")
    encode_parser.add_argument("--file", type=str, help="Arquivo com o texto a ser codificado.")
    encode_parser.add_argument("--output", type=str, help="Salvar IDs em arquivo em vez de imprimir.")
    encode_parser.add_argument("--clean", action="store_true", help="Aplicar limpeza padrao antes da tokenizacao.")
    encode_parser.add_argument("--add-bos", action="store_true", help="Adicionar token BOS.")
    encode_parser.add_argument("--add-eos", action="store_true", help="Adicionar token EOS.")
    add_cleaning_args(encode_parser)
    encode_parser.set_defaults(func=cmd_encode)

    decode_parser = sub.add_parser("decode", help="Decodificar IDs para texto.")
    decode_parser.add_argument("--tokenizer", required=True, help="Caminho para o .model SentencePiece.")
    decode_parser.add_argument("--ids", type=str, help="IDs separados por espaco ou virgula.")
    decode_parser.add_argument("--file", type=str, help="Arquivo contendo IDs.")
    decode_parser.add_argument("--output", type=str, help="Salvar saida em arquivo.")
    decode_parser.add_argument("--skip-special", action="store_true", help="Remover tokens especiais (PAD).")
    decode_parser.set_defaults(func=cmd_decode)

    stats_parser = sub.add_parser("stats", help="Dry-run: medir razao tokens/caracteres.")
    stats_parser.add_argument("--tokenizer", required=True, help="Caminho para o .model SentencePiece.")
    stats_parser.add_argument("--limit", type=int, default=5000, help="Documentos maximos a analisar.")
    stats_parser.add_argument("--sample-docs", type=int, default=2000, help="Documentos usados na estatistica.")
    stats_parser.add_argument("--min-len", type=int, default=200, help="Comprimento minimo apos limpeza.")
    stats_parser.add_argument("--output", type=str, default="", help="Arquivo JSON de saida opcional.")
    add_cleaning_args(stats_parser)
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
