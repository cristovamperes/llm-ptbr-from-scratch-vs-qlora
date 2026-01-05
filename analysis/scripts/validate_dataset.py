#!/usr/bin/env python
"""
Dataset preflight checks for Transformer v5.

This utility reuses the tokenizer + cleaning pipeline to:
  * encode a capped subset of BrWaC (train/validation);
  * compute token/window/batch statistics for the current stride;
  * sample decoded windows for spot checks;
  * emit a JSON report consumable by CI or experiment tracking.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.llm_transformer_v5 import CorpusConfig, DataConfig, TransformerDataBuilder, set_seeds  # noqa: E402
from scripts.brwac_preprocess import resolve_end_inline_sep  # noqa: E402


def decode_sample(window: np.ndarray, builder: TransformerDataBuilder) -> str:
    """Decode a single token window into text."""
    ids: List[int] = window.astype(int).tolist()
    return builder.processor.DecodeIds(ids)  # type: ignore[arg-type]


def sample_windows(tokens: np.ndarray, builder: TransformerDataBuilder, *, count: int, seed: int) -> List[Dict[str, object]]:
    """Pick random windows and decode them for manual inspection."""
    rng = np.random.default_rng(seed)
    seq_len = builder.cfg.seq_len
    if tokens.size <= seq_len:
        return [{
            "start": 0,
            "end": int(tokens.size),
            "decoded": decode_sample(tokens, builder),
        }]

    samples: List[Dict[str, object]] = []
    for _ in range(count):
        start = int(rng.integers(0, tokens.size - seq_len))
        end = start + seq_len
        window = tokens[start:end]
        decoded = decode_sample(window, builder)
        samples.append({"start": start, "end": end, "decoded": decoded})
    return samples


def build_datasets(
    *,
    data_cfg: DataConfig,
    corpus_cfg: CorpusConfig,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    set_seeds(seed)
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=seed)
    split = ds_train_full.train_test_split(test_size=corpus_cfg.valid_split, seed=seed)
    ds_train = split["train"]
    ds_val = split["test"]

    builder = TransformerDataBuilder(data_cfg)

    clean_kwargs = dict(
        lowercase=corpus_cfg.lowercase,
        end_inline_sep=resolve_end_inline_sep(corpus_cfg.end_inline_sep),
        min_line_chars=40,
        min_alpha_ratio=0.4,
        normalize_numbers=corpus_cfg.normalize_numbers,
        drop_uppercase_metadata=corpus_cfg.drop_uppercase_metadata,
    )

    max_train = min(corpus_cfg.max_docs, len(ds_train))
    max_val = min(int(corpus_cfg.max_docs * corpus_cfg.valid_split), len(ds_val))

    train_tokens, train_stats = builder.encode_split(
        ds_train,
        max_train,
        min_len=corpus_cfg.min_len,
        clean_kwargs=clean_kwargs,
    )
    val_tokens, val_stats = builder.encode_split(
        ds_val,
        max_val,
        min_len=corpus_cfg.min_len,
        clean_kwargs=clean_kwargs,
    )

    train_steps = train_stats.windows // max(1, data_cfg.batch_size)
    val_steps = val_stats.windows // max(1, data_cfg.batch_size)

    summary = {
        "data": {
            "tokenizer_path": str(data_cfg.tokenizer_path),
            "seq_len": data_cfg.seq_len,
            "stride": data_cfg.stride,
            "batch_size": data_cfg.batch_size,
            "add_bos": data_cfg.add_bos,
            "add_eos": data_cfg.add_eos,
        },
        "stats": {
            "train": train_stats.__dict__,
            "validation": val_stats.__dict__,
            "train_steps_per_epoch": int(train_steps),
            "validation_steps": int(val_steps),
            "expected_epoch_minutes": round((train_steps * 0.015) / 60, 2),
        },
        "samples": {
            "train": sample_windows(train_tokens, builder, count=3, seed=seed + 7),
            "validation": sample_windows(val_tokens, builder, count=2, seed=seed + 13),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight dataset checks for Transformer v5.")
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument("--max_docs", type=int, default=50_000)
    parser.add_argument("--min_len", type=int, default=200)
    parser.add_argument("--valid_split", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--add_eos", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_json",
        type=Path,
        default=Path("analysis/artifacts/misc/dataset_preflight.json"),
    )
    parser.add_argument(
        "--samples_out",
        type=Path,
        default=Path("analysis/artifacts/misc/dataset_preflight_samples.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        shuffle_buffer=20_000,
    )
    corpus_cfg = CorpusConfig(
        max_docs=args.max_docs,
        min_len=args.min_len,
        valid_split=args.valid_split,
        lowercase=True,
        normalize_numbers=True,
    )
    report = build_datasets(data_cfg=data_cfg, corpus_cfg=corpus_cfg, seed=args.seed)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Report saved to {args.report_json}")

    # Write decoded samples into a human-readable text file
    lines: List[str] = []
    for split_name, samples in report["samples"].items():
        lines.append(f"# {split_name.upper()} samples")
        for idx, sample in enumerate(samples, 1):
            lines.append(f"--- {split_name}:{idx} ({sample['start']}:{sample['end']}) ---")
            lines.append(sample["decoded"])
            lines.append("")
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    args.samples_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Samples saved to {args.samples_out}")


if __name__ == "__main__":
    main()
