"""
Helper utilities for working with the BrWaC dataset across scripts.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from scripts.brwac_preprocess import preparar_texto


@dataclass
class CorpusExportResult:
    count: int
    path: Path
    total_chars: int


def iter_clean_documents(
    dataset_split: Iterable[dict],
    limit: int,
    *,
    min_len: int,
    clean_kwargs: dict,
) -> Iterator[str]:
    """
    Yield cleaned documents from the BrWaC split respecting the length filter.
    """

    count = 0
    for exemplo in dataset_split:
        if count >= limit:
            break
        texto_p = preparar_texto(exemplo["text"], **clean_kwargs)
        if len(texto_p) >= min_len:
            yield texto_p
            count += 1


def export_clean_corpus(
    dataset_split: Iterable[dict],
    limit: int,
    *,
    min_len: int,
    clean_kwargs: dict,
    output_path: Path | None = None,
    separator: str = "\n\n",
) -> CorpusExportResult:
    """
    Clean and export up to ``limit`` documents into a temporary text file.

    Returns a dataclass containing the document count, resulting path and total
    number of characters.
    """

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        output_path = Path(tmp.name)
        tmp.close()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for texto in iter_clean_documents(dataset_split, limit, min_len=min_len, clean_kwargs=clean_kwargs):
            handle.write(texto + separator)
            total_chars += len(texto)
            count += 1
            if count % 5000 == 0:
                print(f"[INFO] Processados {count} textos...")
    return CorpusExportResult(count=count, path=output_path, total_chars=total_chars)
