#!/usr/bin/env python3
"""
Compute heuristic guardrails for generated samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristics on generated samples.")
    parser.add_argument("--samples", type=Path, required=True, help="JSON file produced by generate_samples.py.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model used for decoding.")
    parser.add_argument("--report", type=Path, required=True, help="Path to write the guardrail report (JSON).")
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Model key inside the samples JSON. Required when the file contains multiple models.",
    )
    parser.add_argument(
        "--max-fallback-ratio",
        type=float,
        default=0.03,
        dest="max_fallback_ratio",
        help="Upper bound for the fraction of byte-fallback tokens.",
    )
    parser.add_argument(
        "--max-short-piece-ratio",
        type=float,
        default=0.12,
        dest="max_short_piece_ratio",
        help="Upper bound for the fraction of short alphabetic pieces (len<=2).",
    )
    return parser.parse_args()


def is_byte_fallback(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def is_short_alpha(piece: str) -> bool:
    return len(piece) <= 2 and piece.isalpha()


@dataclass
class SampleMetrics:
    seed: int
    prompt: str
    output: str
    token_count: int
    char_count: int
    fallback_tokens: int
    short_alpha_tokens: int
    fallback_ratio: float
    short_alpha_ratio: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "fallback_tokens": self.fallback_tokens,
            "fallback_ratio": self.fallback_ratio,
            "short_alpha_tokens": self.short_alpha_tokens,
            "short_alpha_ratio": self.short_alpha_ratio,
        }


def select_model(samples: dict, requested: str | None) -> Tuple[str, List[dict]]:
    if requested:
        if requested not in samples:
            raise SystemExit(f"Model key '{requested}' not found in samples file.")
        return requested, samples[requested]
    if len(samples) == 1:
        key = next(iter(samples))
        return key, samples[key]
    keys = ", ".join(sorted(samples))
    raise SystemExit(f"Multiple model keys present ({keys}); please use --model-key.")


def load_samples(samples_path: Path, model_key: str | None) -> Tuple[str, List[dict]]:
    try:
        data = json.loads(samples_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse samples JSON: {exc}") from exc
    if isinstance(data, list):
        key = model_key or "default"
        return key, data
    if not isinstance(data, dict):
        raise SystemExit("Samples JSON must be a mapping from model key to sample list.")
    key, sample_list = select_model(data, model_key)
    if not isinstance(sample_list, list):
        raise SystemExit(f"Samples for key '{key}' are not a list.")
    return key, sample_list


def analyze_sample(sample: dict, processor: spm.SentencePieceProcessor) -> SampleMetrics:
    seed = int(sample.get("seed", -1))
    prompt = sample.get("prompt", "")
    output = sample.get("output", "")

    ids = processor.EncodeAsIds(output)
    pieces = processor.IdToPiece(ids)
    token_count = len(ids)
    char_count = len(output)
    fallback_tokens = sum(1 for piece in pieces if is_byte_fallback(piece))
    short_alpha_tokens = sum(1 for piece in pieces if is_short_alpha(piece))
    denom = token_count if token_count > 0 else 1
    fallback_ratio = fallback_tokens / denom
    short_alpha_ratio = short_alpha_tokens / denom

    return SampleMetrics(
        seed=seed,
        prompt=prompt,
        output=output,
        token_count=token_count,
        char_count=char_count,
        fallback_tokens=fallback_tokens,
        short_alpha_tokens=short_alpha_tokens,
        fallback_ratio=fallback_ratio,
        short_alpha_ratio=short_alpha_ratio,
    )


def main() -> None:
    args = parse_args()
    model_key, samples = load_samples(args.samples, args.model_key)

    processor = spm.SentencePieceProcessor()
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer not found: {args.tokenizer}")
    processor.Load(str(args.tokenizer))

    metrics: List[SampleMetrics] = [analyze_sample(sample, processor) for sample in samples]
    if not metrics:
        raise SystemExit("No samples available to evaluate.")

    total_tokens = sum(m.token_count for m in metrics) or 1
    total_chars = sum(m.char_count for m in metrics)
    total_fallback = sum(m.fallback_tokens for m in metrics)
    total_short_alpha = sum(m.short_alpha_tokens for m in metrics)

    aggregate = {
        "model_key": model_key,
        "samples_count": len(metrics),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_sample": total_tokens / len(metrics),
        "avg_chars_per_sample": total_chars / len(metrics),
        "fallback_tokens": total_fallback,
        "fallback_ratio": total_fallback / total_tokens,
        "short_alpha_tokens": total_short_alpha,
        "short_alpha_ratio": total_short_alpha / total_tokens,
    }

    violations = []
    if aggregate["fallback_ratio"] > args.max_fallback_ratio:
        violations.append(
            {
                "type": "aggregate_fallback_ratio",
                "value": aggregate["fallback_ratio"],
                "threshold": args.max_fallback_ratio,
            }
        )
    if aggregate["short_alpha_ratio"] > args.max_short_piece_ratio:
        violations.append(
            {
                "type": "aggregate_short_piece_ratio",
                "value": aggregate["short_alpha_ratio"],
                "threshold": args.max_short_piece_ratio,
            }
        )

    for sample_metrics in metrics:
        if sample_metrics.fallback_ratio > args.max_fallback_ratio:
            violations.append(
                {
                    "type": "sample_fallback_ratio",
                    "seed": sample_metrics.seed,
                    "value": sample_metrics.fallback_ratio,
                    "threshold": args.max_fallback_ratio,
                }
            )
        if sample_metrics.short_alpha_ratio > args.max_short_piece_ratio:
            violations.append(
                {
                    "type": "sample_short_piece_ratio",
                    "seed": sample_metrics.seed,
                    "value": sample_metrics.short_alpha_ratio,
                    "threshold": args.max_short_piece_ratio,
                }
            )

    report = {
        "model_key": model_key,
        "tokenizer": str(args.tokenizer),
        "thresholds": {
            "max_fallback_ratio": args.max_fallback_ratio,
            "max_short_piece_ratio": args.max_short_piece_ratio,
        },
        "aggregate": aggregate,
        "per_sample": [metric.to_dict() for metric in metrics],
        "passes": len(violations) == 0,
        "violations": violations,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[V5B] Guardrail report written to {args.report}")


if __name__ == "__main__":
    main()
