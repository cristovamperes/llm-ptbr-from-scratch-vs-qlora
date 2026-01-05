from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta Canarim (10k) para SFT (JSONL) com split 80/10/10.")
    p.add_argument("--dataset", default="dominguesm/Canarim-Instruct-PTBR-Dataset")
    p.add_argument("--split", default=None, help="Split a usar (default: primeiro split disponível).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--out_dir", type=Path, required=True)
    return p.parse_args()


def _pick_fields(example: Dict[str, object]) -> Tuple[str, str, str]:
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


def _to_jsonl(ds: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in ds:
            instruction, inp, output = _pick_fields(ex)
            row = {"instruction": instruction, "input": inp, "output": output}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset(args.dataset)
    split_name = args.split or next(iter(dataset_dict.keys()))
    ds = dataset_dict[split_name].shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n_samples, len(ds))))

    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))
    test_size = len(ds) - train_size - val_size

    ds_train = ds.select(range(0, train_size))
    ds_val = ds.select(range(train_size, train_size + val_size))
    ds_test = ds.select(range(train_size + val_size, train_size + val_size + test_size))

    out_train = args.out_dir / "canarim_sft_10k_train.jsonl"
    out_val = args.out_dir / "canarim_sft_10k_val.jsonl"
    out_test = args.out_dir / "canarim_sft_10k_test.jsonl"
    _to_jsonl(ds_train, out_train)
    _to_jsonl(ds_val, out_val)
    _to_jsonl(ds_test, out_test)

    stats: Dict[str, object] = {
        "dataset": args.dataset,
        "split": split_name,
        "seed": args.seed,
        "requested_samples": args.n_samples,
        "total_samples": len(ds),
        "splits": {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)},
        "artifacts": {
            "train_jsonl": str(out_train),
            "val_jsonl": str(out_val),
            "test_jsonl": str(out_test),
        },
    }
    stats_path = args.out_dir / "canarim_sft_10k_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {out_train}")
    print(f"[OK] Wrote: {out_val}")
    print(f"[OK] Wrote: {out_test}")
    print(f"[OK] Stats: {stats_path}")


if __name__ == "__main__":
    main()

