from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta amostra (10k) do BrWaC para CPT (JSONL).")
    p.add_argument("--dataset", default="nlpufg/brwac", help="Dataset HF (default: nlpufg/brwac).")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_docs", type=int, default=10_000)
    p.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa streaming do datasets (recomendado para corpora grandes).",
    )
    p.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=200_000,
        help="Buffer do shuffle em streaming (maior = amostragem mais 'aleatória', mas usa mais RAM).",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats_out", type=Path, required=True)

    p.add_argument("--min_doc_chars", type=int, default=200)
    p.add_argument("--lowercase", action="store_true", default=False)
    p.add_argument("--end_inline_sep", default="newline", choices=["newline", "space"])
    p.add_argument("--min_line_chars", type=int, default=80)
    p.add_argument("--min_alpha_ratio", type=float, default=0.6)
    p.add_argument("--normalize_numbers", action="store_true", default=False)
    p.add_argument("--drop_uppercase_metadata", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=int(args.shuffle_buffer_size))
    else:
        ds = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)

    end_inline_sep = resolve_end_inline_sep(args.end_inline_sep)
    out_lines: List[str] = []
    raw_docs = 0
    kept_docs = 0
    kept_chars: List[int] = []

    for ex in ds:
        raw_docs += 1
        raw_text = ex.get("text", "") or ""
        cleaned = preparar_texto(
            raw_text,
            lowercase=args.lowercase,
            end_inline_sep=end_inline_sep,
            min_line_chars=args.min_line_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            normalize_numbers=args.normalize_numbers,
            drop_uppercase_metadata=args.drop_uppercase_metadata,
        )
        cleaned = cleaned.strip()
        if len(cleaned) < args.min_doc_chars:
            continue
        out_lines.append(json.dumps({"text": cleaned}, ensure_ascii=False))
        kept_docs += 1
        kept_chars.append(len(cleaned))
        if kept_docs >= args.n_docs:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    stats: Dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "requested_docs": args.n_docs,
        "streaming": args.streaming,
        "shuffle_buffer_size": int(args.shuffle_buffer_size) if args.streaming else None,
        "raw_docs_scanned": raw_docs,
        "kept_docs": kept_docs,
        "min_doc_chars": args.min_doc_chars,
        "cleaning": {
            "lowercase": args.lowercase,
            "end_inline_sep": args.end_inline_sep,
            "min_line_chars": args.min_line_chars,
            "min_alpha_ratio": args.min_alpha_ratio,
            "normalize_numbers": args.normalize_numbers,
            "drop_uppercase_metadata": args.drop_uppercase_metadata,
        },
        "chars": {
            "avg": (sum(kept_chars) / len(kept_chars)) if kept_chars else 0,
            "min": min(kept_chars) if kept_chars else 0,
            "max": max(kept_chars) if kept_chars else 0,
        },
        "artifacts": {
            "jsonl": str(args.out),
        },
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {kept_docs} docs to {args.out}")
    print(f"[OK] Stats: {args.stats_out}")


if __name__ == "__main__":
    main()
