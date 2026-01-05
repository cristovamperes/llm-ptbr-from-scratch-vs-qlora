#!/usr/bin/env python3
"""Evaluate a Transformer v5 model on the BrWaC validation split."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

from scripts.llm_transformer_v5 import (  # noqa: E402
    CorpusConfig,
    DataConfig,
    OutputProjection,
    build_transformer_model,
    TransformerDataBuilder,
    WarmupCosineSchedule,
    resolve_end_inline_sep,
    set_seeds,
)


def load_mapping(path: Path) -> Dict[str, Any]:
    if path.suffix == ".pkl":
        with path.open("rb") as fh:
            return pickle.load(fh)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported mapping format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer v5 on BrWaC validation split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained .keras model.")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to the mapping file (.pkl or .json).")
    parser.add_argument("--max_docs", type=int, default=50_000, help="Maximum documents for training split (for reproducible split).")
    parser.add_argument("--min_len", type=int, default=200, help="Minimum characters per document after cleaning.")
    parser.add_argument("--valid_split", type=float, default=0.1, help="Validation ratio used during training.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for evaluation (defaults to mapping value).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/artifacts/results/results_v5_scaleplus.json"),
        help="Where to store metrics JSON.",
    )
    return parser.parse_args()


def build_validation_dataset(
    builder: TransformerDataBuilder,
    corpus_cfg: CorpusConfig,
    *,
    seed: int,
) -> tuple[tf.data.Dataset, int]:
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=seed)
    split = ds_train_full.train_test_split(test_size=corpus_cfg.valid_split, seed=seed)
    ds_val = split["test"]

    clean_kwargs = dict(
        lowercase=corpus_cfg.lowercase,
        end_inline_sep=resolve_end_inline_sep(corpus_cfg.end_inline_sep),
        min_line_chars=40,
        min_alpha_ratio=0.4,
        normalize_numbers=corpus_cfg.normalize_numbers,
        drop_uppercase_metadata=corpus_cfg.drop_uppercase_metadata,
    )

    max_val_docs = min(int(corpus_cfg.max_docs * corpus_cfg.valid_split), len(ds_val))
    val_tokens, val_stats = builder.encode_split(
        ds_val,
        max_val_docs,
        min_len=corpus_cfg.min_len,
        clean_kwargs=clean_kwargs,
    )
    ds_val = builder.make_dataset(val_tokens, shuffle=False, repeat=False)
    return ds_val, val_stats.batches


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, steps: int) -> Dict[str, float]:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    iterator = iter(dataset)
    for _ in range(steps):
        try:
            batch_inputs, batch_targets = next(iterator)
        except StopIteration:
            break
        logits = model(batch_inputs, training=False)
        batch_loss = loss_fn(batch_targets, logits).numpy()
        predictions = tf.argmax(logits, axis=-1, output_type=batch_targets.dtype)
        matches = tf.equal(predictions, batch_targets)
        total_correct += int(tf.reduce_sum(tf.cast(matches, tf.int64)))
        total_tokens += int(np.prod(batch_targets.shape))
        total_loss += float(batch_loss)

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    accuracy = total_correct / total_tokens
    return {"loss": average_loss, "perplexity": perplexity, "accuracy": accuracy}


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)

    tokenizer_path = Path(mapping["tokenizer_model"])
    seq_len = int(mapping["sequence_length"])
    stride = int(mapping["stride"])
    batch_size = args.batch_size or int(mapping.get("batch_size", 32))
    add_bos = bool(mapping.get("add_bos", True))
    add_eos = bool(mapping.get("add_eos", True))

    data_cfg = DataConfig(
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        add_bos=add_bos,
        add_eos=add_eos,
        seed=args.seed,
    )
    corpus_cfg = CorpusConfig(
        max_docs=args.max_docs,
        min_len=args.min_len,
        valid_split=args.valid_split,
        lowercase=True,
        normalize_numbers=True,
        drop_uppercase_metadata=True,
        end_inline_sep="newline",
    )

    set_seeds(args.seed)
    builder = TransformerDataBuilder(data_cfg)
    val_dataset, val_steps = build_validation_dataset(builder, corpus_cfg, seed=args.seed)

    model_cfg = mapping["model"]
    model = build_transformer_model(
        vocab_size=int(mapping["tokenizer_vocab_size"]),
        seq_len=seq_len,
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

    ckpt_candidates = [
        args.model.with_name(f"{args.model.stem}_checkpoint.keras"),
        args.model,
    ]
    stage_name = args.model.stem.split("_")[-1]
    stage_dir = ROOT / "versions" / "v5-transformer" / "checkpoints" / stage_name
    if stage_dir.is_dir():
        ckpt_candidates.extend(sorted(stage_dir.glob("epoch_*.weights.h5"), reverse=True))

    weights_loaded = False
    for candidate in ckpt_candidates:
        if not candidate.exists():
            continue
        try:
            model.load_weights(str(candidate))
            print(f"[INFO] Pesos carregados de {candidate}")
            weights_loaded = True
            break
        except Exception as exc:
            print(f"[WARN] Não foi possível carregar pesos de {candidate}: {exc}")
    if not weights_loaded:
        raise RuntimeError("Nenhum checkpoint compatível foi encontrado para carregar os pesos do modelo.")

    metrics = evaluate_model(model, val_dataset, val_steps)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
