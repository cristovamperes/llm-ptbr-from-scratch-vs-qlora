"""
Subword language model (v4) built on top of SentencePiece tokenization.

The model consumes integer token IDs produced by a SentencePiece tokenizer and
trains a stack of recurrent layers (GRU or LSTM) for next-token prediction.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from tcc_llm.sentencepiece_utils import encode_text, load_processor


def _parse_units(units: Sequence[int] | int | None, fallback: Sequence[int]) -> Tuple[int, ...]:
    if units is None:
        return tuple(fallback)
    if isinstance(units, int):
        return (units,)
    parsed: List[int] = []
    for value in units:
        parsed.append(int(value))
    return tuple(parsed) if parsed else tuple(fallback)


@dataclass
class V4Config:
    sequence_length: int = 192
    embedding_dim: int = 512
    rnn_units: Tuple[int, ...] = (512, 512)
    rnn_type: str = "gru"  # "gru" or "lstm"
    dropout: float = 0.1
    recurrent_dropout: float = 0.0
    final_dropout: float = 0.1
    stride: int = 2
    shuffle_buffer: int = 200000
    batch_size: int = 256
    epochs: int = 2
    learning_rate: float = 1e-3
    clipnorm: float | None = 1.0
    add_bos: bool = True
    add_eos: bool = True


class LLMSubwordV4:
    """
    Train a stacked recurrent model (GRU/LSTM) over SentencePiece tokens.
    """

    def __init__(
        self,
        *,
        sequence_length: int = 192,
        embedding_dim: int = 512,
        rnn_units: Iterable[int] | int | None = None,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        recurrent_dropout: float = 0.0,
        final_dropout: float = 0.1,
        stride: int = 2,
        shuffle_buffer: int = 200000,
        batch_size: int = 256,
        epochs: int = 2,
        learning_rate: float = 1e-3,
        clipnorm: float | None = 1.0,
        add_bos: bool = True,
        add_eos: bool = True,
        tokenizer_path: str | Path = "",
    ) -> None:
        self.cfg = V4Config(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            rnn_units=_parse_units(rnn_units, fallback=(512, 512)),
            rnn_type=rnn_type.lower(),
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            final_dropout=final_dropout,
            stride=stride,
            shuffle_buffer=shuffle_buffer,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        if self.cfg.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        if not tokenizer_path:
            raise ValueError("tokenizer_path must be provided for subword training.")
        self.tokenizer_path = str(tokenizer_path)
        self.processor = load_processor(self.tokenizer_path)
        self.vocab_size = int(self.processor.GetPieceSize())
        self.model: tf.keras.Model | None = None
        self.train_windows: int = 0
        self.val_windows: int = 0
        self.train_batches: int = 0
        self.val_batches: int = 0
        self.train_tokens: int = 0
        self.val_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def treinar(
        self,
        caminho_texto: str,
        nome_arquivo_modelo: str,
        nome_arquivo_maps: str,
        *,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        caminho_texto_validacao: Optional[str] = None,
    ) -> None:
        texto_train = self._ler(caminho_texto)
        ids_train = self._encode(texto_train)
        self.train_tokens = int(ids_train.size)
        ds_train = self._seq_para_dataset(ids_train, treino=True)
        self.train_windows = self._estimate_windows(ids_train.size)
        self.train_batches = int(np.ceil(self.train_windows / max(1, self.cfg.batch_size)))

        ds_val = None
        if caminho_texto_validacao:
            texto_val = self._ler(caminho_texto_validacao)
            ids_val = self._encode(texto_val)
            self.val_tokens = int(ids_val.size)
            ds_val = self._seq_para_dataset(ids_val, treino=False)
            self.val_windows = self._estimate_windows(ids_val.size)
            self.val_batches = int(np.ceil(self.val_windows / max(1, self.cfg.batch_size)))

        self.model = self._construir_modelo(self.vocab_size)
        self.model.fit(
            ds_train,
            epochs=self.cfg.epochs,
            validation_data=ds_val,
            callbacks=callbacks if callbacks else None,
            verbose=2,
        )

        os.makedirs(os.path.dirname(nome_arquivo_modelo) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(nome_arquivo_maps) or ".", exist_ok=True)
        self.model.save(nome_arquivo_modelo)
        self._salvar_metadata(nome_arquivo_maps)

    def stats(self) -> Dict[str, object]:
        params = int(self.model.count_params()) if self.model is not None else 0
        return {
            "vocab_size": self.vocab_size,
            "sequence_length": self.cfg.sequence_length,
            "tamanho_sequencia": self.cfg.sequence_length,
            "rnn_units": list(self.cfg.rnn_units),
            "rnn_type": self.cfg.rnn_type,
            "batch_size": self.cfg.batch_size,
            "stride": self.cfg.stride,
            "train_windows": self.train_windows,
            "val_windows": self.val_windows,
            "train_windows_est": self.train_windows,
            "val_windows_est": self.val_windows,
            "tokens_per_epoch_est": self.train_windows * self.cfg.sequence_length,
            "train_batches": self.train_batches,
            "val_batches": self.val_batches,
            "dropout": self.cfg.dropout,
            "recurrent_dropout": self.cfg.recurrent_dropout,
            "final_dropout": self.cfg.final_dropout,
            "params": params,
            "tokenizer_model": self.tokenizer_path,
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ler(caminho: str) -> str:
        with open(caminho, "r", encoding="utf-8") as fh:
            return fh.read()

    def _encode(self, texto: str) -> np.ndarray:
        docs = [doc.strip() for doc in texto.split("\n\n") if doc.strip()]
        all_ids: List[int] = []
        bos_id = self.processor.bos_id() if self.cfg.add_bos else -1
        eos_id = self.processor.eos_id() if self.cfg.add_eos else -1
        for doc in docs:
            doc_ids = encode_text(doc, self.processor, add_bos=False, add_eos=False)
            if not doc_ids:
                continue
            if bos_id >= 0:
                all_ids.append(int(bos_id))
            all_ids.extend(int(idx) for idx in doc_ids)
            if eos_id >= 0:
                all_ids.append(int(eos_id))
        return np.asarray(all_ids, dtype=np.int32)

    def _estimate_windows(self, token_count: int) -> int:
        seq_len = self.cfg.sequence_length
        stride = max(1, self.cfg.stride)
        if token_count <= seq_len:
            return 0
        return int((token_count - seq_len) // stride)

    def _seq_para_dataset(self, ids: np.ndarray, *, treino: bool) -> tf.data.Dataset:
        if ids.size <= self.cfg.sequence_length:
            raise ValueError("Sequencia de tokens muito curta para o tamanho definido.")
        ds = tf.data.Dataset.from_tensor_slices(ids)
        ds = ds.window(self.cfg.sequence_length + 1, shift=self.cfg.stride, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.cfg.sequence_length + 1))
        ds = ds.map(lambda w: (w[:-1], w[-1]), num_parallel_calls=tf.data.AUTOTUNE)
        if treino:
            ds = ds.shuffle(self.cfg.shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _construir_modelo(self, vocab_size: int) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.cfg.sequence_length,), name="inputs")
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.cfg.embedding_dim,
            name="embedding",
        )(inputs)
        for idx, units in enumerate(self.cfg.rnn_units):
            return_sequences = idx < len(self.cfg.rnn_units) - 1
            if self.cfg.rnn_type == "gru":
                x = layers.GRU(
                    units,
                    dropout=self.cfg.dropout,
                    recurrent_dropout=self.cfg.recurrent_dropout,
                    return_sequences=return_sequences,
                    name=f"gru_{idx + 1}",
                )(x)
            else:
                x = layers.LSTM(
                    units,
                    dropout=self.cfg.dropout,
                    recurrent_dropout=self.cfg.recurrent_dropout,
                    return_sequences=return_sequences,
                    name=f"lstm_{idx + 1}",
                )(x)
        if self.cfg.final_dropout and self.cfg.final_dropout > 0:
            x = layers.Dropout(self.cfg.final_dropout, name="final_dropout")(x)
        outputs = layers.Dense(vocab_size, activation="softmax", name="logits")(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="llm_subword_v4")
        if self.cfg.clipnorm is not None and self.cfg.clipnorm > 0:
            opt = optimizers.Adam(learning_rate=self.cfg.learning_rate, clipnorm=self.cfg.clipnorm)
        else:
            opt = optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _salvar_metadata(self, caminho_maps: str) -> None:
        metadata = {
            "tokenization": "sentencepiece",
            "tokenizer_model": os.path.abspath(self.tokenizer_path),
            "tokenizer_vocab_size": self.vocab_size,
            "sequence_length": self.cfg.sequence_length,
            "stride": self.cfg.stride,
            "rnn_units": list(self.cfg.rnn_units),
            "rnn_type": self.cfg.rnn_type,
            "embedding_dim": self.cfg.embedding_dim,
            "dropout": self.cfg.dropout,
            "recurrent_dropout": self.cfg.recurrent_dropout,
            "final_dropout": self.cfg.final_dropout,
            "add_bos": self.cfg.add_bos,
            "add_eos": self.cfg.add_eos,
        }
        with open(caminho_maps, "wb") as fh:
            pickle.dump(metadata, fh)
        # Provide JSON copy to ease inspection
        json_path = Path(caminho_maps).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as jh:
            json.dump(metadata, jh, ensure_ascii=False, indent=2)
