#!/usr/bin/env python3
"""Generate qualitative samples for the v5b Transformer."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras

keras.config.enable_unsafe_deserialization()

D_MODEL = 320
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 1280
DROPOUT = 0.05

DEFAULT_PROMPTS = [
    "O setor de infraestrutura logistica brasileira debate novas concessoes ferroviarias, cronogramas de duplicacao, metas de produtividade e integracao com portos para desafogar corredores de exportacao.",
    "Analistas do mercado financeiro reavaliam previsoes trimestrais para bancos listados, discutem juros, carteira de credito, inadimplencia corporativa e estrategias de hedge diante de volatilidade externa.",
    "Engenheiros de software de uma fintech planejam migracao para microservicos, definem SLAs, monitoramento proativo, politicas de rollback continuo e treinamentos para times de suporte e compliance.",
    "A diretoria de um clube de futebol negocia patrocinio master, reforcos na janela de transferencia, metas de bilheteria, programa de socios e auditoria das contas apos uma temporada irregular nos gramados.",
    "Uma rede de hospitais privados avalia expansao para telemedicina, protocolos integrados de prontuario, parcerias com seguradoras, capacidade de leitos, gestao de estoque farmacologico e contratacao medica.",
]


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
        config.pop("tied", None)
        return cls(embedding_layer=None, **config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerar samples para o v5b_long.")
    parser.add_argument("--model", type=Path, default=Path("versions/v5b-transformer/models/modelo_v5b_long_checkpoint.keras"))
    parser.add_argument("--tokenizer", type=Path, default=Path("versions/v5b-transformer/tokenizers/unigram_12k/spm_v5b.model"))
    parser.add_argument("--output", type=Path, default=Path("analysis/samples_v5b_long.json"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--prompts-file", type=Path, help="Arquivo com prompts (um por linha).")
    parser.add_argument("--decode", choices=["nucleus", "beam"], default="nucleus")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--beam-alpha", type=float, default=0.7, help="Penaliza beams longos.")
    return parser.parse_args()


def extract_weights_archive(model_path: Path, weights_path: Path) -> Path:
    if weights_path.exists():
        return weights_path
    if not model_path.exists():
        raise SystemExit(f"Modelo nao encontrado: {model_path}")
    with zipfile.ZipFile(model_path, "r") as zf:
        with zf.open("model.weights.h5") as src, weights_path.open("wb") as dst:
            dst.write(src.read())
    return weights_path


def build_transformer_model(vocab_size: int, seq_len: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="inputs")
    token_embedding = tf.keras.layers.Embedding(vocab_size, D_MODEL, name="token_embedding")
    position_embedding = tf.keras.layers.Embedding(seq_len, D_MODEL, name="position_embedding")

    def position_indices(tensor):
        seq_length = tf.shape(tensor)[1]
        batch_size = tf.shape(tensor)[0]
        positions = tf.range(seq_length)[tf.newaxis, :]
        return tf.tile(positions, [batch_size, 1])

    indices = tf.keras.layers.Lambda(position_indices, name="position_indices")(inputs)
    x = token_embedding(inputs) + position_embedding(indices)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    for layer_idx in range(NUM_LAYERS):
        residual = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_attn_{layer_idx}")(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=D_MODEL // NUM_HEADS,
            dropout=DROPOUT,
            name=f"mha_{layer_idx}",
        )(x, x, use_causal_mask=True)
        attn = tf.keras.layers.Dropout(DROPOUT)(attn)
        x = residual + attn

        residual_ffn = x
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_ffn_{layer_idx}")(x)
        ffn = tf.keras.layers.Dense(D_FF, activation="gelu", name=f"ffn_{layer_idx}_1")(x)
        ffn = tf.keras.layers.Dropout(DROPOUT)(ffn)
        ffn = tf.keras.layers.Dense(D_MODEL, name=f"ffn_{layer_idx}_2")(ffn)
        x = residual_ffn + ffn

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    logits = OutputProjection(token_embedding, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="transformer_decoder_v5")


def nucleus_sample(logits: np.ndarray, top_p: float) -> int:
    probs = tf.nn.softmax(logits).numpy()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = cumulative <= top_p
    cutoff_idx = np.where(cutoff)[0]
    if cutoff_idx.size == 0:
        cutoff_idx = np.array([0])
    limit = cutoff_idx[-1] + 1
    filtered_idx = sorted_idx[:limit]
    filtered_probs = sorted_probs[:limit]
    filtered_probs /= filtered_probs.sum()
    choice = np.random.choice(filtered_idx, p=filtered_probs)
    return int(choice)


def beam_search(
    model: tf.keras.Model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    *,
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    beam_size: int,
    alpha: float,
) -> str:
    eos_id = sp.eos_id()
    start_ids = sp.EncodeAsIds(prompt)
    beams: List[Tuple[List[int], float]] = [(start_ids, 0.0)]
    completed: List[Tuple[List[int], float]] = []

    for _ in range(max_new_tokens):
        new_beams: List[Tuple[List[int], float]] = []
        for seq, score in beams:
            window = seq[-seq_len:]
            logits = model(tf.constant([window], dtype=tf.int32), training=False)[0, -1]
            logits = logits / max(temperature, 1e-5)
            log_probs = tf.nn.log_softmax(logits).numpy()
            top_indices = np.argpartition(-log_probs, beam_size)[:beam_size]
            for idx in top_indices:
                new_seq = seq + [int(idx)]
                new_score = score + float(log_probs[idx])
                if idx == eos_id:
                    length_norm = (len(new_seq) ** alpha) / ((1 + len(new_seq)) ** alpha)
                    completed.append((new_seq, new_score / max(length_norm, 1e-6)))
                else:
                    new_beams.append((new_seq, new_score))
        if not new_beams:
            break
        new_beams.sort(key=lambda item: item[1], reverse=True)
        beams = new_beams[:beam_size]

    if not completed:
        completed = beams
    best_seq = max(completed, key=lambda item: item[1])[0]
    return sp.DecodeIds(best_seq)


def generate_sequence(
    model: tf.keras.Model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    *,
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    top_p: float,
    decode_mode: str,
    beam_size: int,
    beam_alpha: float,
) -> str:
    if decode_mode == "beam":
        return beam_search(
            model,
            sp,
            prompt,
            max_new_tokens=max_new_tokens,
            seq_len=seq_len,
            temperature=temperature,
            beam_size=beam_size,
            alpha=beam_alpha,
        )

    ids: List[int] = sp.EncodeAsIds(prompt)
    eos_id = sp.eos_id()
    for _ in range(max_new_tokens):
        window = ids[-seq_len:]
        logits = model(tf.constant([window], dtype=tf.int32), training=False)[0]
        next_logits = logits[-1] / max(temperature, 1e-5)
        next_token = nucleus_sample(next_logits.numpy(), top_p)
        if next_token == eos_id:
            break
        ids.append(next_token)
    return sp.DecodeIds(ids)


def main() -> None:
    args = parse_args()
    weights_path = extract_weights_archive(args.model, args.model.with_suffix(".weights.h5"))
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer nao encontrado: {args.tokenizer}")

    print("[V5B] Carregando tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))
    vocab_size = sp.GetPieceSize()

    print("[V5B] Construindo modelo...")
    model = build_transformer_model(vocab_size, args.seq_len)
    print(f"[V5B] Carregando pesos de {weights_path.name}...")
    model.load_weights(str(weights_path))

    if args.prompts_file:
        prompts = [
            line.strip()
            for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        prompts = DEFAULT_PROMPTS

    print("[V5B] Gerando amostras...")
    results = []
    for idx, prompt in enumerate(prompts, start=1):
        text = generate_sequence(
            model,
            sp,
            prompt,
            max_new_tokens=args.max_new_tokens,
            seq_len=args.seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            decode_mode=args.decode,
            beam_size=args.beam_size,
            beam_alpha=args.beam_alpha,
        )
        preview = text.replace("\n", " ")[:150]
        print(f"  ({idx}/{len(prompts)}) {preview}...")
        results.append({"prompt": prompt, "output": text})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[V5B] Samples salvos em {args.output}")


if __name__ == "__main__":
    main()
