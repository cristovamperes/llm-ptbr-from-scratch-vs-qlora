from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cria subsets extrínsecos (QA/sumarização/reescrita) a partir do Canarim.")
    p.add_argument("--dataset", default="dominguesm/Canarim-Instruct-PTBR-Dataset")
    p.add_argument("--split", default=None, help="Split a usar (default: primeiro split disponível).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_per_task", type=int, default=200)
    p.add_argument("--out_dir", type=Path, default=Path("versions/trilha2-lora/analysis/extrinsic"))
    p.add_argument(
        "--exclude_jsonl",
        type=Path,
        action="append",
        default=[
            Path("versions/trilha2-lora/datasets/canarim_sft_10k_train.jsonl"),
            Path("versions/trilha2-lora/datasets/canarim_sft_10k_val.jsonl"),
            Path("versions/trilha2-lora/datasets/canarim_sft_10k_test.jsonl"),
        ],
        help="JSONLs usados no treino (para excluir overlap). Pode repetir.",
    )
    return p.parse_args()


def _pick_fields(example: Dict[str, Any]) -> Tuple[str, str, str]:
    for ins_key, inp_key, out_key in (
        ("instruction", "input", "output"),
        ("prompt", "input", "response"),
        ("instruction", "context", "output"),
    ):
        if ins_key in example and out_key in example:
            instruction = (example.get(ins_key) or "").strip()
            inp = (example.get(inp_key) or "").strip() if inp_key in example else ""
            output = (example.get(out_key) or "").strip()
            return instruction, inp, output
    raise KeyError(f"Campos não reconhecidos no exemplo: {list(example.keys())}")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _row_hash(instruction: str, inp: str, output: str) -> str:
    payload = f"{instruction}\n---\n{inp}\n---\n{output}".strip()
    return _sha1_text(payload)


def _load_exclude_hashes(paths: List[Path]) -> set[str]:
    hashes: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instruction = (row.get("instruction") or "").strip()
                inp = (row.get("input") or "").strip()
                output = (row.get("output") or "").strip()
                hashes.add(_row_hash(instruction, inp, output))
    return hashes


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _classify_task(instruction: str, inp: str, output: str) -> Optional[str]:
    text = f"{instruction}\n{inp}".strip().lower()
    in_len = len(_word_tokens(inp))
    out_len = len(_word_tokens(output))

    if out_len == 0:
        return None

    qa_hint = ("?" in text) or ("opções de resposta" in text) or ("(a)" in text) or ("(b)" in text)
    if out_len <= 12 and qa_hint:
        return "qa"

    summ_kw = re.search(r"\b(resuma|resumo|sumarize|sumarização|sumariza|sumário|sintetize|síntese)\b", text)
    if summ_kw:
        return "summarization"
    if in_len >= 120 and 30 <= out_len <= 140 and out_len <= int(0.55 * max(1, in_len)):
        return "summarization"

    rew_kw = re.search(
        r"\b(reescreva|reformule|corrija|melhore|parafrase|paráfrase|revise|revisão|simplifique|transforme)\b", text
    )
    if rew_kw:
        return "rewriting"
    if 20 <= in_len <= 220 and 20 <= out_len <= 220:
        ratio = out_len / max(1, in_len)
        if 0.6 <= ratio <= 1.6:
            return "rewriting"

    return None


def _iter_examples(ds: Dataset) -> Iterable[Dict[str, Any]]:
    for ex in ds:
        yield ex


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    exclude_hashes = _load_exclude_hashes([Path(p) for p in args.exclude_jsonl])

    dataset_dict = load_dataset(args.dataset)
    split_name = args.split or next(iter(dataset_dict.keys()))
    ds = dataset_dict[split_name].shuffle(seed=int(args.seed))

    targets = {"qa": int(args.n_per_task), "summarization": int(args.n_per_task), "rewriting": int(args.n_per_task)}
    picked: Dict[str, List[Dict[str, Any]]] = {k: [] for k in targets}

    scanned = 0
    for ex in _iter_examples(ds):
        scanned += 1
        instruction, inp, output = _pick_fields(ex)
        h = _row_hash(instruction, inp, output)
        if h in exclude_hashes:
            continue

        task = _classify_task(instruction, inp, output)
        if task is None or task not in targets:
            continue
        if len(picked[task]) >= targets[task]:
            continue

        picked[task].append(
            {
                "task": task,
                "instruction": instruction,
                "input": inp,
                "reference": output,
                "hash": h,
            }
        )

        if all(len(picked[k]) >= targets[k] for k in targets):
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "split": split_name,
        "seed": int(args.seed),
        "n_per_task_requested": int(args.n_per_task),
        "n_per_task_obtained": {k: len(v) for k, v in picked.items()},
        "scanned_examples": scanned,
        "exclude_jsonl": [str(p) for p in args.exclude_jsonl],
        "out_dir": str(args.out_dir),
        "filters": {
            "qa": "out_len<=12 and (contains '?' or options)",
            "summarization": "keywords OR long input with shorter output",
            "rewriting": "keywords OR similar length (heuristic)",
        },
        "files": {
            "qa": str(args.out_dir / "canarim_extrinsic_qa.jsonl"),
            "summarization": str(args.out_dir / "canarim_extrinsic_summarization.jsonl"),
            "rewriting": str(args.out_dir / "canarim_extrinsic_rewriting.jsonl"),
        },
    }

    _write_jsonl(args.out_dir / "canarim_extrinsic_qa.jsonl", picked["qa"])
    _write_jsonl(args.out_dir / "canarim_extrinsic_summarization.jsonl", picked["summarization"])
    _write_jsonl(args.out_dir / "canarim_extrinsic_rewriting.jsonl", picked["rewriting"])
    (args.out_dir / "canarim_extrinsic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[OK] Wrote:")
    for k in ("qa", "summarization", "rewriting"):
        print(f"  - {manifest['files'][k]} ({len(picked[k])} exemplos)")
    print(f"  - {args.out_dir / 'canarim_extrinsic_manifest.json'}")


if __name__ == "__main__":
    main()

