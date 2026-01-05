# -*- coding: utf-8 -*-
"""
Avaliacao comparativa de modelos char/subword.

Calcula perplexidade, loss medio e acuracia (argmax) em um subconjunto
limpo do BrWaC e salva os resultados em JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from datasets import load_dataset

# Importa utilitarios de limpeza diretamente do script da v3
_SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.append(str(_SCRIPT_DIR))

from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep


@dataclass
class TokenizerAssets:
    mode: str
    seq_len: int
    vocab_size: int
    char_to_idx: Dict[str, int] | None = None
    idx_to_token: Dict[int, str] | None = None
    unk_idx: int | None = None
    unk_token: str | None = None
    processor: spm.SentencePieceProcessor | None = None
    add_bos: bool = False
    add_eos: bool = False
    tokenizer_model: Path | None = None


@dataclass
class ModelSpec:
    name: str
    model_path: Path
    mapping_path: Path
    tokenizer: TokenizerAssets | None = None


def _load_mapping(path: Path) -> TokenizerAssets:
    with path.open("rb") as fh:
        data = pickle.load(fh)

    seq_len = data.get("tamanho_sequencia") or data.get("sequence_length")
    if not isinstance(seq_len, int):
        raise ValueError(f"tamanho_sequencia ausente em {path}")

    tokenization = data.get("tokenization", "char")
    if tokenization == "sentencepiece":
        model_raw = data.get("tokenizer_model")
        if not model_raw:
            raise ValueError(f"tokenizer_model ausente em {path}")
        model_path = Path(model_raw)
        if not model_path.exists():
            raise ValueError(f"Arquivo SentencePiece nao encontrado: {model_path}")
        processor = spm.SentencePieceProcessor()
        if not processor.Load(str(model_path)):
            raise ValueError(f"Falha ao carregar SentencePiece: {model_path}")
        vocab_size = int(data.get("tokenizer_vocab_size") or processor.GetPieceSize())
        add_bos = bool(data.get("add_bos", False))
        add_eos = bool(data.get("add_eos", False))
        return TokenizerAssets(
            mode="sentencepiece",
            seq_len=seq_len,
            vocab_size=vocab_size,
            processor=processor,
            add_bos=add_bos,
            add_eos=add_eos,
            tokenizer_model=model_path,
        )

    if "char_to_idx" in data and "idx_to_char" in data:
        c2i = data["char_to_idx"]
        i2c = data["idx_to_char"]
    elif "char_to_id" in data and "id_to_char" in data:
        c2i = data["char_to_id"]
        i2c = data["id_to_char"]
    elif "char_para_int" in data and "int_para_char" in data:
        c2i = data["char_para_int"]
        i2c = data["int_para_char"]
    else:
        raise ValueError(f"Formato de mapeamentos nao reconhecido em {path}")

    try:
        c2i = {str(k): int(v) for k, v in c2i.items()}
        i2c = {int(k): str(v) for k, v in i2c.items()}
    except Exception:
        pass

    unk_token = data.get("unk_token")
    if isinstance(unk_token, bytes):
        unk_token = unk_token.decode("utf-8", errors="ignore") or None
    unk_idx = None
    if isinstance(unk_token, str) and unk_token in c2i:
        unk_idx = c2i[unk_token]
    elif " " in c2i:
        unk_idx = c2i[" "]

    return TokenizerAssets(
        mode="char",
        seq_len=seq_len,
        vocab_size=len(c2i),
        char_to_idx=c2i,
        idx_to_token=i2c,
        unk_idx=unk_idx,
        unk_token=unk_token if isinstance(unk_token, str) else None,
    )


def _texto_para_indices_char(texto: str, assets: TokenizerAssets) -> List[int]:
    assert assets.char_to_idx is not None
    seq: List[int] = []
    for ch in texto:
        if ch in assets.char_to_idx:
            seq.append(assets.char_to_idx[ch])
        elif assets.unk_idx is not None:
            seq.append(assets.unk_idx)
    return seq


def _encode_docs_sentencepiece(
    docs: Sequence[str],
    assets: TokenizerAssets,
) -> List[int]:
    assert assets.processor is not None
    seq: List[int] = []
    bos_id = assets.processor.bos_id() if assets.add_bos else -1
    eos_id = assets.processor.eos_id() if assets.add_eos else -1
    for doc in docs:
        ids = assets.processor.EncodeAsIds(doc)
        if not ids:
            continue
        if assets.add_bos and bos_id >= 0:
            seq.append(int(bos_id))
        seq.extend(int(idx) for idx in ids)
        if assets.add_eos and eos_id >= 0:
            seq.append(int(eos_id))
    return seq


def _windows(
    seq: Iterable[int],
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seq = list(seq)
    if len(seq) <= seq_len:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0,), dtype=np.int32)
    xs: List[List[int]] = []
    ys: List[int] = []
    for i in range(0, len(seq) - seq_len, stride):
        xs.append(seq[i : i + seq_len])
        ys.append(seq[i + seq_len])
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def _evaluate_model(
    model: tf.keras.Model,
    windows: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    input_mode: str,
    vocab_size: int,
) -> Dict[str, float]:
    ds = tf.data.Dataset.from_tensor_slices((windows, targets)).batch(batch_size)
    total_log_prob = 0.0
    total_n = 0
    total_correct = 0
    for x_batch, y_batch in ds:
        if input_mode == "onehot":
            x_batch = tf.one_hot(x_batch, depth=vocab_size, dtype=tf.float32)
        probs = model(x_batch, training=False)
        probs = tf.clip_by_value(tf.convert_to_tensor(probs, dtype=tf.float32), 1e-9, 1.0)
        log_probs = tf.math.log(probs)
        batch_indices = tf.stack(
            [tf.range(tf.shape(y_batch)[0], dtype=tf.int32), tf.cast(y_batch, tf.int32)],
            axis=1,
        )
        gathered = tf.gather_nd(log_probs, batch_indices)
        total_log_prob += float(tf.reduce_sum(gathered))
        total_n += int(y_batch.shape[0])
        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        total_correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, y_batch), tf.int32)))
    avg_loss = float(-total_log_prob / total_n) if total_n > 0 else math.nan
    perplexity = math.exp(-total_log_prob / total_n) if total_n > 0 else math.inf
    accuracy = total_correct / total_n if total_n > 0 else math.nan
    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "windows": total_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliar modelos char-level")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Especificacao no formato nome:modelo_path:mapeamentos_path",
    )
    parser.add_argument("--max_docs", type=int, default=2000, help="Documentos do BrWaC para avaliacao")
    parser.add_argument("--min_len", type=int, default=200, help="Comprimento minimo apos limpeza")
    parser.add_argument("--stride_eval", type=int, default=1, help="Stride para janelas de avaliacao")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size para inferencia")
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/artifacts/results/results.json",
        help="Arquivo JSON de saida",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=250_000,
        help="Limite maximo de janelas por modelo para manter avaliacao manejavel",
    )
    args = parser.parse_args()

    specs: List[ModelSpec] = []
    for spec in args.model:
        try:
            name, model_path, mapping_path = spec.split(":", 2)
        except ValueError:
            raise SystemExit(f"Formato invalido para --model '{spec}'")
        specs.append(ModelSpec(name=name, model_path=Path(model_path), mapping_path=Path(mapping_path)))

    end_inline_sep = resolve_end_inline_sep("newline")

    print(f"Carregando subset do BrWaC ({args.max_docs} documentos)...")
    dataset = load_dataset("nlpufg/brwac", split=f"train[:{args.max_docs}]")

    textos: List[str] = []
    for item in dataset:
        texto = preparar_texto(
            item["text"],
            lowercase=True,
            end_inline_sep=end_inline_sep,
            min_line_chars=40,
            min_alpha_ratio=0.4,
            normalize_numbers=True,
            drop_uppercase_metadata=True,
        )
        if len(texto) >= args.min_len:
            textos.append(texto)
    avaliacao_texto = "\n\n".join(textos)
    print(f"Documentos validos: {len(textos)} | caracteres totais: {len(avaliacao_texto)}")

    resultados: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        print(f"\n--- Avaliando {spec.name} ---")
        assets = _load_mapping(spec.mapping_path)
        spec.tokenizer = assets
        if assets.mode == "char":
            idx_seq = _texto_para_indices_char(avaliacao_texto, assets)
        else:
            idx_seq = _encode_docs_sentencepiece(textos, assets)
        if not idx_seq:
            raise SystemExit("Falha na conversao de texto para tokens.")
        seq_len = assets.seq_len
        windows, targets = _windows(idx_seq, seq_len, args.stride_eval)
        if windows.shape[0] == 0:
            raise SystemExit("Nenhuma janela de avaliacao gerada.")
        if windows.shape[0] > args.max_windows:
            windows = windows[: args.max_windows]
            targets = targets[: args.max_windows]
        print(f"Janelas geradas: {windows.shape[0]} (seq_len={seq_len}, tokens={len(idx_seq)})")
        model = tf.keras.models.load_model(spec.model_path)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if len(input_shape) == 2:
            input_mode = "embedding"
        elif len(input_shape) == 3:
            input_mode = "onehot"
        else:
            raise ValueError(f"Forma de entrada inesperada {input_shape} para {spec.name}")
        metrics = _evaluate_model(
            model,
            windows,
            targets,
            args.batch_size,
            input_mode,
            assets.vocab_size,
        )
        resultados[spec.name] = {
            **metrics,
            "seq_len": seq_len,
            "dataset_windows": int(windows.shape[0]),
            "model_path": str(spec.model_path),
            "mapping_path": str(spec.mapping_path),
            "tokenization": assets.mode,
            "vocab_size": assets.vocab_size,
            "dataset_tokens": int(len(idx_seq)),
            "tokenizer_model": str(assets.tokenizer_model) if assets.tokenizer_model else "",
        }
        print(
            f"{spec.name}: loss={metrics['avg_loss']:.4f} | ppl={metrics['perplexity']:.2f} | acc={metrics['accuracy']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "max_docs": args.max_docs,
                    "min_len": args.min_len,
                    "stride_eval": args.stride_eval,
                    "batch_size": args.batch_size,
                    "max_windows": args.max_windows,
                },
                "results": resultados,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResultados salvos em {output_path}")


if __name__ == "__main__":
    main()
