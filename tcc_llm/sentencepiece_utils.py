"""
Utilities for training and using SentencePiece tokenizers within the project.

This module centralises common helpers so that both CLI scripts and training
pipelines can rely on the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm


@dataclass
class SentencePiecePaths:
    model: Path
    vocab: Path


def train_sentencepiece(
    input_file: Path,
    output_dir: Path,
    *,
    vocab_size: int = 3200,
    model_prefix: str = "spm",
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    input_sentence_size: int | None = None,
    shuffle_input_sentence: bool = True,
    seed: int = 42,
    extra_options: Sequence[str] | None = None,
) -> SentencePiecePaths:
    """
    Train a SentencePiece model using the provided cleaned corpus.

    Parameters
    ----------
    input_file:
        Path to a UTF-8 text file containing one or more documents separated by
        blank lines. The file must already be preprocessed.
    output_dir:
        Directory where the `.model` and `.vocab` files will be stored.
    vocab_size:
        Target vocabulary size (including reserved tokens).
    model_prefix:
        Prefix for generated files (defaults to ``spm``). The full paths will be
        `<output_dir>/<model_prefix>.model` and `<output_dir>/<model_prefix>.vocab`.
    model_type:
        SentencePiece model type (e.g. ``bpe``, ``unigram``, ``word``, ``char``).
    character_coverage:
        Desired character coverage. For Portuguese corpora 0.9995 generally works
        well to keep rare glyphs.
    input_sentence_size:
        Optional subsampling size. When set, SentencePiece will sample sentences
        up to this limit.
    shuffle_input_sentence:
        Whether to shuffle sentences during sampling (recommended).
    seed:
        Random seed for SentencePiece trainers.
    extra_options:
        Additional SentencePiece parameters passed as raw strings (e.g.
        ``["--hard_vocab_limit=false"]``).

    Returns
    -------
    SentencePiecePaths
        Dataclass containing the resulting `.model` and `.vocab` paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_file}")

    model_prefix_path = output_dir / model_prefix
    spm_args: list[str] = [
        f"--input={input_file}",
        f"--model_prefix={model_prefix_path}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--pad_id=0",
        f"--pad_piece=<pad>",
        f"--unk_id=1",
        f"--unk_piece=<unk>",
        f"--bos_id=2",
        f"--bos_piece=<s>",
        f"--eos_id=3",
        f"--eos_piece=</s>",
        f"--user_defined_symbols=<sep>,<mask>",
        f"--train_extremely_large_corpus=true",
        f"--seed_sentencepiece_size={vocab_size}",
        "--hard_vocab_limit=false",
    ]
    if input_sentence_size:
        spm_args.append(f"--input_sentence_size={input_sentence_size}")
        spm_args.append(f"--shuffle_input_sentence={'true' if shuffle_input_sentence else 'false'}")
    if extra_options:
        spm_args.extend(extra_options)

    spm.SentencePieceTrainer.Train(" ".join(spm_args))
    model_path = Path(f"{model_prefix_path}.model")
    vocab_path = Path(f"{model_prefix_path}.vocab")
    if not model_path.exists() or not vocab_path.exists():
        raise RuntimeError("SentencePiece training finished but model files were not created.")
    return SentencePiecePaths(model=model_path, vocab=vocab_path)


def load_processor(model_path: Path | str) -> spm.SentencePieceProcessor:
    """Load a SentencePieceProcessor from a `.model` file."""

    processor = spm.SentencePieceProcessor()
    model_path = str(model_path)
    if not processor.Load(model_path):
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return processor


def encode_text(
    text: str,
    processor: spm.SentencePieceProcessor,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    """
    Encode text into a list of token IDs.

    Parameters
    ----------
    text:
        Cleaned input string.
    processor:
        Loaded SentencePieceProcessor.
    add_bos / add_eos:
        Whether to add BOS/EOS tokens to the resulting sequence.
    """

    return processor.EncodeAsIds(text, add_bos=add_bos, add_eos=add_eos)


def encode_batch(
    texts: Iterable[str],
    processor: spm.SentencePieceProcessor,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[List[int]]:
    """Encode an iterable of texts, returning a list of ID sequences."""

    return [encode_text(text, processor, add_bos=add_bos, add_eos=add_eos) for text in texts]


def decode_ids(
    ids: Sequence[int],
    processor: spm.SentencePieceProcessor,
    *,
    skip_special: bool = True,
) -> str:
    """
    Decode token IDs back into text.

    Parameters
    ----------
    ids:
        Sequence of integer IDs.
    processor:
        Loaded SentencePieceProcessor.
    skip_special:
        When True the output string removes padding tokens.
    """

    if skip_special:
        ids = [idx for idx in ids if idx not in (processor.pad_id(),)]
    return processor.DecodeIds(list(ids))
