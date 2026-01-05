#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de dados para o Transformer v5 (decoder-only).

Nesta etapa focamos em:
  * carregar o BrWaC com a mesma limpeza da v4;
  * aplicar o tokenizer SentencePiece (4k + byte fallback);
  * gerar janelas autoregressivas (seq_len + 1) com stride configurável;
  * construir datasets tf.data prontos para treino/validação.

O treino completo e o modelo Transformer serão adicionados em etapas
subsequentes. Por ora, o script facilita validar estatísticas e salvar
um arquivo JSON com contagens de documentos/tokens/janelas.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in os.sys.path:
    os.sys.path.append(str(_ROOT))

from tcc_llm.sentencepiece_utils import encode_text, load_processor  # noqa: E402
from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep  # noqa: E402


@dataclass
class DataConfig:
    tokenizer_path: Path
    seq_len: int = 256
    stride: int = 2
    batch_size: int = 64
    shuffle_buffer: int = 200_000
    add_bos: bool = True
    add_eos: bool = True
    seed: int = 42
    drop_remainder: bool = True


@dataclass
class CorpusConfig:
    max_docs: int = 20_000
    min_len: int = 200
    valid_split: float = 0.1
    lowercase: bool = True
    normalize_numbers: bool = False
    drop_uppercase_metadata: bool = True
    end_inline_sep: str = "newline"
    min_line_chars: int = 40
    min_alpha_ratio: float = 0.4


@dataclass
class DatasetStats:
    docs: int
    tokens: int
    windows: int
    batches: int
    avg_tokens_per_doc: float


@tf.keras.utils.register_keras_serializable(name="WarmupCosineSchedule")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        base_lr = tf.cast(self.base_lr, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        def warmup():
            return base_lr * (step / warmup_steps)

        def cosine_decay():
            progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
            return base_lr * 0.5 * (1.0 + tf.cos(tf.constant(np.pi) * progress))

        return tf.cond(step < warmup_steps, warmup, cosine_decay)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


@tf.keras.utils.register_keras_serializable(name="WarmupCosineRestartSchedule")
class WarmupCosineRestartSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup followed by cosine decay with restarts."""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        first_restart_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
    ):
        super().__init__()
        if first_restart_steps <= 0:
            raise ValueError("first_restart_steps must be > 0")
        self.base_lr = float(base_lr)
        self.warmup_steps = max(1, warmup_steps)
        self.cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=float(base_lr),
            first_decay_steps=int(first_restart_steps),
            t_mul=float(t_mul),
            m_mul=float(m_mul),
            alpha=float(alpha),
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        def warmup():
            return self.base_lr * (step / warmup_steps)

        def decay():
            shifted = tf.maximum(0.0, step - warmup_steps)
            return self.cosine(shifted)

        return tf.cond(step < warmup_steps, warmup, decay)

    def get_config(self):
        cfg = self.cosine.get_config()
        cfg.update({"base_lr": self.base_lr, "warmup_steps": self.warmup_steps})
        return cfg


@tf.keras.utils.register_keras_serializable(name="OutputProjection")
class OutputProjection(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_layer: tf.keras.layers.Embedding | None = None,
        *,
        vocab_size: int | None = None,
        proj_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            vocab_size = int(embedding_layer.input_dim)
            proj_dim = int(embedding_layer.output_dim)
        if vocab_size is None or proj_dim is None:
            raise ValueError("vocab_size and proj_dim must be provided when embedding_layer is None.")
        self.vocab_size = int(vocab_size)
        self.proj_dim = int(proj_dim)
        self._tied = embedding_layer is not None
        self.kernel: tf.Variable | None = None
        self.bias: tf.Variable | None = None

    def build(self, input_shape):
        if self._tied and self.embedding_layer is not None:
            self.kernel = self.embedding_layer.embeddings
            self.embeddings = self.kernel
        else:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(self.vocab_size, self.proj_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        logits = tf.matmul(inputs, self.kernel, transpose_b=True)
        return logits + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "proj_dim": self.proj_dim,
                "tied": bool(self.embedding_layer is not None),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        # embedding layer reference cannot be restored automatically; create untied projection
        config.pop("tied", None)
        return cls(embedding_layer=None, **config)


class TokensPerSecondCallback(tf.keras.callbacks.Callback):
    def __init__(self, seq_len: int, batch_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.start_time = 0.0
        self.total_tokens = 0
        self.tokens_per_sec = 0.0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.total_tokens = 0

    def on_train_batch_end(self, batch, logs=None):
        self.total_tokens += self.seq_len * self.batch_size
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.tokens_per_sec = self.total_tokens / elapsed


class JsonHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path: Optional[Path], tokens_per_epoch: Optional[int] = None):
        super().__init__()
        self.output_path = output_path
        self.history: List[Dict[str, float]] = []
        self.start_time = 0.0
        self.epoch_start_time = 0.0
        self.tokens_per_epoch = int(tokens_per_epoch) if tokens_per_epoch else 0
        self.tokens_seen_total = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.tokens_seen_total = 0
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        entry = {"epoch": int(epoch) + 1}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                entry[key] = float(value)
        loss_value = logs.get("loss")
        if isinstance(loss_value, (int, float)):
            entry["perplexity"] = float(math.exp(float(loss_value)))
        val_loss_value = logs.get("val_loss")
        if isinstance(val_loss_value, (int, float)):
            entry["val_perplexity"] = float(math.exp(float(val_loss_value)))
        if self.tokens_per_epoch:
            self.tokens_seen_total += self.tokens_per_epoch
            entry["tokens_epoch"] = self.tokens_per_epoch
            entry["tokens_seen_total"] = self.tokens_seen_total
        if self.epoch_start_time:
            entry["epoch_time_sec"] = float(time.time() - self.epoch_start_time)
        self.history.append(entry)

    def on_train_end(self, logs=None):
        if not self.output_path:
            return
        payload = {
            "history": self.history,
            "total_time_sec": time.time() - self.start_time,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class MetricsDumpCallback(tf.keras.callbacks.Callback):
    """Write per-epoch metrics to individual JSON files (for dashboards/CI)."""

    def __init__(self, output_dir: Optional[Path], tokens_per_epoch: Optional[int] = None):
        super().__init__()
        self.output_dir = output_dir
        self.tokens_per_epoch = int(tokens_per_epoch) if tokens_per_epoch else 0
        self.tokens_seen_total = 0
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs=None):
        self.tokens_seen_total = 0
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if not self.output_dir:
            return
        logs = logs or {}
        record = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        record["epoch"] = int(epoch) + 1
        if "loss" in record:
            record["perplexity"] = float(math.exp(record["loss"]))
        if "val_loss" in record:
            record["val_perplexity"] = float(math.exp(record["val_loss"]))
        if self.tokens_per_epoch:
            self.tokens_seen_total += self.tokens_per_epoch
            record["tokens_epoch"] = self.tokens_per_epoch
            record["tokens_seen_total"] = self.tokens_seen_total
        if self.epoch_start_time:
            record["epoch_time_sec"] = float(time.time() - self.epoch_start_time)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"metrics_epoch_{int(epoch)+1:02d}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


class CheckpointPruner(tf.keras.callbacks.Callback):
    """Keep only the most recent N checkpoints in a directory."""

    def __init__(self, directory: Path, keep_last: int):
        super().__init__()
        self.directory = directory
        self.keep_last = max(0, int(keep_last))

    def on_epoch_end(self, epoch, logs=None):
        if self.keep_last <= 0:
            return
        checkpoints = sorted(self.directory.glob("epoch_*.weights.h5"))
        if len(checkpoints) <= self.keep_last:
            return
        obsolete = checkpoints[: len(checkpoints) - self.keep_last]
        for ckpt in obsolete:
            try:
                ckpt.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[WARN] Falha ao remover checkpoint antigo {ckpt}: {exc}")


class TransformerDataBuilder:
    def __init__(self, data_cfg: DataConfig) -> None:
        self.cfg = data_cfg
        self.processor = load_processor(str(data_cfg.tokenizer_path))
        self.vocab_size = int(self.processor.GetPieceSize())
        self.pad_id = self.processor.pad_id()
        if self.pad_id < 0:
            alt = self.processor.unk_id()
            self.pad_id = alt if alt >= 0 else 0
        self.bos_id = self.processor.bos_id() if data_cfg.add_bos else -1
        self.eos_id = self.processor.eos_id() if data_cfg.add_eos else -1

    def encode_split(
        self,
        dataset_split: Iterable[Dict[str, str]],
        limit_docs: int,
        *,
        min_len: int,
        clean_kwargs: Dict[str, object],
    ) -> Tuple[np.ndarray, DatasetStats]:
        tokens: list[int] = []
        docs = 0
        for example in dataset_split:
            if docs >= limit_docs:
                break
            text = preparar_texto(example["text"], **clean_kwargs)
            if len(text) < min_len:
                continue
            ids = encode_text(text, self.processor, add_bos=False, add_eos=False)
            if not ids:
                continue
            if self.bos_id >= 0:
                tokens.append(int(self.bos_id))
            tokens.extend(int(idx) for idx in ids)
            if self.eos_id >= 0:
                tokens.append(int(self.eos_id))
            docs += 1
            if docs % 5000 == 0:
                print(f"[INFO] {docs} documentos processados...")

        arr = np.asarray(tokens, dtype=np.int32)
        total_tokens = int(arr.size)
        seq_len = self.cfg.seq_len
        stride = max(1, self.cfg.stride)
        windows = max(0, (total_tokens - seq_len) // stride)
        batches = windows // max(1, self.cfg.batch_size)
        avg_tokens = float(total_tokens / docs) if docs else 0.0
        stats = DatasetStats(
            docs=docs,
            tokens=total_tokens,
            windows=windows,
            batches=batches,
            avg_tokens_per_doc=avg_tokens,
        )
        return arr, stats

    def make_dataset(self, tokens: np.ndarray, *, shuffle: bool, repeat: bool = True) -> tf.data.Dataset:
        seq_plus_one = self.cfg.seq_len + 1
        stride = max(1, self.cfg.stride)
        batch_size = self.cfg.batch_size

        ds = tf.data.Dataset.from_tensor_slices(tokens)
        ds = ds.window(seq_plus_one, shift=stride, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(seq_plus_one))
        ds = ds.map(
            lambda w: (w[:-1], w[1:]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if shuffle:
            ds = ds.shuffle(self.cfg.shuffle_buffer, seed=self.cfg.seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=self.cfg.drop_remainder)
        # repeat the dataset so Keras can consume multiple epochs without running out of data
        # For validation, we don't repeat so validation_steps=None works correctly
        if repeat:
            ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preparar datasets subword para o Transformer v5.")
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        required=True,
        help="Caminho para o modelo SentencePiece (.model).",
    )
    parser.add_argument("--max_docs", type=int, default=20_000)
    parser.add_argument("--min_len", type=int, default=200)
    parser.add_argument("--valid_split", type=float, default=0.1)
    parser.add_argument("--min_line_chars", type=int, default=40, help="Minimum chars per line after cleaning.")
    parser.add_argument("--min_alpha_ratio", type=float, default=0.4, help="Minimum alpha ratio per line after cleaning.")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shuffle_buffer", type=int, default=200_000)
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--add_eos", action="store_true", default=True)
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--keep_numbers", action="store_true")
    parser.add_argument("--keep_upper_metadata", action="store_true")
    parser.add_argument("--end_inline_sep", choices=["space", "newline"], default="newline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_output", type=Path, default=None, help="Arquivo JSON para salvar estatísticas.")
    parser.add_argument("--dry_run_batches", type=int, default=0, help="Quantidade de batches para inspecionar (opcional).")
    parser.add_argument("--train", action="store_true", help="Executa treino completo do Transformer.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="Se >0, usa valor fixo; caso contrário usa contagem do dataset.")
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=0, help="Se 0, calcula automaticamente (epochs * steps_per_epoch).")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for loss (0.0-0.2, default 0.1)")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm (default 1.0)")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument(
        "--precision",
        choices=["mixed_float16", "float32"],
        default="mixed_float16",
        help="Define a politica global de precisao (default: mixed_float16).",
    )
    parser.add_argument(
        "--jit_compile",
        action="store_true",
        help="Habilita XLA JIT durante o model.compile (desabilitado por padrao).",
    )
    parser.add_argument("--fallback_penalty", type=float, default=0.0, help="Penalty weight for assigning probability to byte fallback tokens.")
    parser.add_argument("--model_output", type=Path, default=Path("versions/v5-transformer/models/modelo_v5.keras"))
    parser.add_argument("--mappings_output", type=Path, default=Path("versions/v5-transformer/mappings/mapeamentos_v5.pkl"))
    parser.add_argument("--log_json_output", type=Path, default=Path("versions/v5-transformer/logs/train_v5.json"))
    parser.add_argument("--csv_log_output", type=Path, default=Path("versions/v5-transformer/logs/history_v5.csv"))
    parser.add_argument("--tensorboard_logdir", type=Path, default=None)
    parser.add_argument(
        "--lr_schedule",
        choices=["cosine", "cosine_restart"],
        default="cosine",
        help="Learning-rate schedule to use (default: cosine).",
    )
    parser.add_argument(
        "--restart_steps",
        type=int,
        default=0,
        help="First restart interval in steps (only used when --lr_schedule=cosine_restart).",
    )
    parser.add_argument("--restart_t_mul", type=float, default=2.0, help="Cycle-length multiplier for cosine restarts.")
    parser.add_argument("--restart_m_mul", type=float, default=1.0, help="Learning-rate multiplier for cosine restarts.")
    parser.add_argument("--metrics_dir", type=Path, default=None, help="Dump per-epoch metrics as JSON files.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Directory for saving per-epoch model checkpoints (weights-only).",
    )
    parser.add_argument(
        "--keep_n_checkpoints",
        type=int,
        default=5,
        help="How many recent checkpoints to keep in --checkpoint_dir (0 keeps all).",
    )
    parser.add_argument(
        "--initial_checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint (weights) to warm-start training.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_cfg = DataConfig(
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        add_bos=bool(args.add_bos),
        add_eos=bool(args.add_eos),
        seed=args.seed,
    )
    corpus_cfg = CorpusConfig(
        max_docs=args.max_docs,
        min_len=args.min_len,
        valid_split=args.valid_split,
        lowercase=not args.no_lowercase,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
        end_inline_sep=args.end_inline_sep,
        min_line_chars=args.min_line_chars,
        min_alpha_ratio=args.min_alpha_ratio,
    )

    builder = TransformerDataBuilder(data_cfg)

    print("[INFO] Carregando dataset BrWaC...")
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=args.seed)
    split = ds_train_full.train_test_split(test_size=corpus_cfg.valid_split, seed=args.seed)
    ds_train = split["train"]
    ds_val = split["test"]
    print(f"[INFO] Split -> train: {len(ds_train)} | val: {len(ds_val)}")

    clean_kwargs = dict(
        lowercase=corpus_cfg.lowercase,
        end_inline_sep=resolve_end_inline_sep(corpus_cfg.end_inline_sep),
        min_line_chars=corpus_cfg.min_line_chars,
        min_alpha_ratio=corpus_cfg.min_alpha_ratio,
        normalize_numbers=corpus_cfg.normalize_numbers,
        drop_uppercase_metadata=corpus_cfg.drop_uppercase_metadata,
    )

    max_train = min(corpus_cfg.max_docs, len(ds_train))
    max_val = min(int(corpus_cfg.max_docs * corpus_cfg.valid_split), len(ds_val))

    started = time.time()
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
    elapsed_encode = time.time() - started

    print(f"[INFO] Tokens treino: {train_stats.tokens:,} | docs: {train_stats.docs} | janelas: {train_stats.windows}")
    print(f"[INFO] Tokens validação: {val_stats.tokens:,} | docs: {val_stats.docs} | janelas: {val_stats.windows}")
    print(f"[INFO] Tempo de encode: {elapsed_encode:.2f}s")

    ds_train_tf = builder.make_dataset(train_tokens, shuffle=True, repeat=True)
    ds_val_tf = builder.make_dataset(val_tokens, shuffle=False, repeat=True)

    # Libertar memória
    train_tokens = None
    val_tokens = None

    # Inspeção opcional
    if args.dry_run_batches > 0:
        print(f"[INFO] Inspecionando {args.dry_run_batches} batches de treino...")
        for idx, (x_batch, y_batch) in enumerate(ds_train_tf.take(args.dry_run_batches), 1):
            print(f"  Batch {idx}: inputs={x_batch.shape}, targets={y_batch.shape}")

    data_cfg_dict = asdict(data_cfg)
    data_cfg_dict["tokenizer_path"] = str(data_cfg.tokenizer_path)
    summary = {
        "config": {
            "data": data_cfg_dict,
            "corpus": asdict(corpus_cfg),
            "tokenizer_vocab": builder.vocab_size,
        },
        "train": asdict(train_stats),
        "validation": asdict(val_stats),
        "encode_elapsed_sec": elapsed_encode,
    }

    stats_path = args.stats_output
    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Estatísticas salvas em {stats_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.train:
        return

    precision_policy = args.precision
    if args.no_mixed_precision and args.precision != "float32":
        precision_policy = "float32"
        print("[WARN] --no_mixed_precision esta obsoleto; utilizando precision=float32.")
    try:
        tf.keras.mixed_precision.set_global_policy(precision_policy)
        print(f"[INFO] Politica de precisao ativa: {precision_policy}.")
    except Exception as exc:
        print(f"[WARN] Nao foi possivel configurar politica de precisao '{precision_policy}': {exc}")

    steps_per_epoch = args.steps_per_epoch or max(1, train_stats.batches)
    # Set validation_steps to None to run through full validation set each epoch
    validation_steps = args.validation_steps if args.validation_steps > 0 else val_stats.batches
    total_steps = args.total_steps or (steps_per_epoch * args.epochs)
    tokens_per_step = data_cfg.batch_size * data_cfg.seq_len
    tokens_per_epoch = tokens_per_step * steps_per_epoch

    print(
        f"[INFO] Treinando Transformer v5 -> epochs={args.epochs}, steps/epoch={steps_per_epoch}, "
        f"tokens/epoch={tokens_per_epoch:,}, warmup={args.warmup_steps}, total_steps={total_steps}"
    )

    model = build_transformer_model(
        vocab_size=builder.vocab_size,
        seq_len=data_cfg.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    if args.lr_schedule == "cosine_restart":
        restart_steps = args.restart_steps or steps_per_epoch
        lr_schedule = WarmupCosineRestartSchedule(
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            first_restart_steps=restart_steps,
            t_mul=args.restart_t_mul,
            m_mul=args.restart_m_mul,
        )
    else:
        lr_schedule = WarmupCosineSchedule(
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
        )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8,
        weight_decay=args.weight_decay,
        clipnorm=args.gradient_clip_norm,
    )
    fallback_penalty = max(0.0, args.fallback_penalty)
    fallback_start = builder.vocab_size - 256 if fallback_penalty > 0.0 and builder.vocab_size >= 512 else None

    def apply_penalties(loss_value, y_pred):
        if fallback_penalty > 0.0 and fallback_start is not None:
            probs = tf.nn.softmax(y_pred)
            fallback_prob = tf.reduce_mean(tf.reduce_sum(probs[..., fallback_start:], axis=-1))
            loss_value += fallback_penalty * tf.cast(fallback_prob, loss_value.dtype)
        return loss_value

    if args.label_smoothing > 0:
        smoothing = args.label_smoothing

        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            vocab_size = tf.shape(y_pred)[-1]
            y_true_one_hot = tf.one_hot(y_true, depth=vocab_size)
            y_true_smooth = y_true_one_hot * (1.0 - smoothing) + (smoothing / tf.cast(vocab_size, tf.float32))
            xe = tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred, from_logits=True)
            loss_value = tf.reduce_mean(xe)
            return apply_penalties(loss_value, y_pred)
    else:
        def loss_fn(y_true, y_pred):
            xe = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            loss_value = tf.reduce_mean(xe)
            return apply_penalties(loss_value, y_pred)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=False,
        jit_compile=bool(args.jit_compile),
    )

    if args.initial_checkpoint:
        initial_checkpoint = Path(args.initial_checkpoint)
        if initial_checkpoint.exists():
            try:
                model.load_weights(str(initial_checkpoint), by_name=True, skip_mismatch=True)
                print(f"[INFO] Pesos iniciais carregados de {initial_checkpoint} (skip_mismatch=True).")
            except Exception as exc:
                print(f"[WARN] Falha ao carregar checkpoint inicial {initial_checkpoint}: {exc}")
        else:
            print(f"[WARN] Checkpoint inicial não encontrado: {initial_checkpoint}")

    callbacks: List[tf.keras.callbacks.Callback] = []
    if args.tensorboard_logdir:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(args.tensorboard_logdir), profile_batch=0))
    if args.csv_log_output:
        args.csv_log_output.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(str(args.csv_log_output)))

    # Early stopping para prevenir overfitting
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Checkpoint para salvar melhor modelo
    model_output_path = Path(args.model_output)
    checkpoint_path = model_output_path.parent / f"{model_output_path.stem}_checkpoint.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        per_epoch_template = checkpoint_dir / "epoch_{epoch:02d}.weights.h5"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(per_epoch_template),
                save_weights_only=True,
                save_freq="epoch",
            )
        )
        if args.keep_n_checkpoints > 0:
            callbacks.append(CheckpointPruner(checkpoint_dir, args.keep_n_checkpoints))
        print(f"[INFO] Salvando checkpoints por época em {checkpoint_dir}")

    json_logger = JsonHistoryCallback(args.log_json_output, tokens_per_epoch=tokens_per_epoch)
    callbacks.append(json_logger)
    perf_callback = TokensPerSecondCallback(seq_len=data_cfg.seq_len, batch_size=data_cfg.batch_size)
    callbacks.append(perf_callback)
    if args.metrics_dir:
        callbacks.append(MetricsDumpCallback(args.metrics_dir, tokens_per_epoch=tokens_per_epoch))

    start_training = time.time()
    history = model.fit(
        ds_train_tf,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val_tf,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    total_time = time.time() - start_training

    model_output_path = Path(args.model_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    print(f"[INFO] Modelo salvo em {model_output_path}")

    mappings_output_path = Path(args.mappings_output)
    mappings_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_metadata(
        mappings_output_path,
        tokenizer=str(args.tokenizer_path.resolve()),
        vocab_size=builder.vocab_size,
        data_cfg=data_cfg,
        model_cfg=dict(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ),
    )
    print(f"[INFO] Metadados salvos em {mappings_output_path}")

    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history.get("val_loss", [float("nan")])[-1])
    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history.get("val_accuracy", [float("nan")])[-1])
    tokens_per_sec = perf_callback.tokens_per_sec
    final_ppl = math.exp(final_loss) if math.isfinite(final_loss) else float("nan")
    final_val_ppl = math.exp(final_val_loss) if math.isfinite(final_val_loss) else float("nan")

    print(
        f"[INFO] Treino concluído em {total_time/3600:.2f} h | tokens/s ~ {tokens_per_sec:.0f} "
        f"| loss={final_loss:.4f} (ppl={final_ppl:.2f}) "
        f"| val_loss={final_val_loss:.4f} (val_ppl={final_val_ppl:.2f}) "
        f"| acc={final_acc:.4f} | val_acc={final_val_acc:.4f}"
    )


def build_transformer_model(
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    dropout: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="inputs")
    token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
    position_embedding = tf.keras.layers.Embedding(seq_len, d_model, name="position_embedding")

    position_indices = tf.keras.layers.Lambda(
        lambda tensor: tf.tile(
            tf.expand_dims(tf.range(tf.shape(tensor)[1]), axis=0),
            [tf.shape(tensor)[0], 1],
        ),
        name="position_indices",
    )(inputs)
    position_embeddings = position_embedding(position_indices)
    token_embeddings = token_embedding(inputs)

    x = token_embeddings + position_embeddings
    x = tf.keras.layers.Dropout(dropout)(x)

    for layer_idx in range(num_layers):
        residual = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"mha_{layer_idx}",
        )(x, x, use_causal_mask=True)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        x = residual + attn

        residual_ffn = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        ffn = tf.keras.layers.Dense(d_ff, activation="gelu", name=f"ffn_{layer_idx}_1")(x)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f"ffn_{layer_idx}_2")(ffn)
        x = residual_ffn + ffn

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    logits = OutputProjection(token_embedding, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="transformer_decoder_v5")


def save_metadata(path: Path, *, tokenizer: str, vocab_size: int, data_cfg: DataConfig, model_cfg: Dict[str, object]) -> None:
    payload = {
        "tokenization": "sentencepiece",
        "tokenizer_model": tokenizer,
        "tokenizer_vocab_size": vocab_size,
        "sequence_length": data_cfg.seq_len,
        "stride": data_cfg.stride,
        "batch_size": data_cfg.batch_size,
        "add_bos": data_cfg.add_bos,
        "add_eos": data_cfg.add_eos,
        "model": model_cfg,
    }
    import pickle

    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
