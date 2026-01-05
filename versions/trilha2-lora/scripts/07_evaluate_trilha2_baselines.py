from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from common import collect_env_info, run_timers, set_seeds, try_get_git_commit, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avalia baselines vs adapters (CPT e SFT) da Trilha 2.")
    p.add_argument(
        "--cpt-config",
        type=Path,
        default=Path("versions/trilha2-lora/configs/cpt_qlora_llama31_8b_brwac10k.json"),
    )
    p.add_argument(
        "--sft-config",
        type=Path,
        default=Path("versions/trilha2-lora/configs/sft_qlora_llama31_8b_instruct_canarim10k.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("versions/trilha2-lora/analysis/eval_trilha2_baselines.json"),
    )
    p.add_argument("--limit-cpt-eval", type=int, default=0, help="0 = sem limite (avalia tudo).")
    p.add_argument("--limit-sft-eval", type=int, default=0, help="0 = sem limite (avalia tudo).")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name.lower() == "bfloat16":
        return torch.bfloat16
    if name.lower() == "float16":
        return torch.float16
    raise ValueError(f"torch dtype inválido: {name}")


def _resolve(repo_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p)


def _build_bnb_config(cfg_model: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    if not cfg_model.get("use_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(cfg_model["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(cfg_model["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=_dtype(str(cfg_model["bnb_4bit_compute_dtype"])),
    )


def _evaluate(
    model: Any,
    dataset: Dataset,
    data_collator: Any,
    per_device_eval_batch_size: int,
    bf16: bool,
) -> Dict[str, Any]:
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    total_loss_weighted = 0.0
    total_label_tokens = 0

    model.eval()
    for batch in loader:
        labels = batch.get("labels")
        if isinstance(labels, torch.Tensor):
            label_tokens = int((labels != -100).sum().item())
        else:
            label_tokens = 0

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            if device.type == "cuda" and bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
            else:
                out = model(**batch)
        loss = float(out.loss.detach().cpu())
        total_loss_weighted += loss * label_tokens
        total_label_tokens += label_tokens

    end = time.time()
    timers = run_timers(start, end)

    eval_loss = total_loss_weighted / max(1, total_label_tokens)
    ppl = float("inf") if eval_loss > 20 else float(math.exp(eval_loss))
    return {
        "eval_loss": float(eval_loss),
        "perplexity": ppl,
        "timers": timers.__dict__,
        "label_tokens": total_label_tokens,
        "batches": len(loader),
    }


def _load_cpt_eval_dataset(repo_root: Path, cfg: Dict[str, Any], tokenizer: Any) -> Dataset:
    seed = int(cfg["experiment"]["seed"])
    ds = load_dataset("json", data_files=str(_resolve(repo_root, str(cfg["data"]["dataset_jsonl"]))))["train"]
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=float(cfg["data"]["valid_split"]), seed=seed)

    seq_len = int(cfg["data"]["seq_len"])

    def tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples[str(cfg["data"]["text_field"])], truncation=True, max_length=seq_len)

    tokenized = split.map(tokenize, batched=True, remove_columns=split["train"].column_names)

    block_size = seq_len

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    return lm_datasets["test"]


def _format_prompt(instruction: str, inp: str) -> str:
    if inp:
        return f"{instruction}\n\nEntrada:\n{inp}".strip()
    return instruction.strip()


def _load_sft_eval_dataset(repo_root: Path, cfg: Dict[str, Any], tokenizer: Any) -> Dataset:
    seed = int(cfg["experiment"]["seed"])
    ds_val = load_dataset("json", data_files=str(_resolve(repo_root, str(cfg["data"]["val_jsonl"]))))["train"]
    seq_len = int(cfg["data"]["seq_len"])

    def encode(ex: Dict[str, Any]) -> Dict[str, Any]:
        instruction = str(ex.get("instruction", "") or "")
        inp = str(ex.get("input", "") or "")
        output = str(ex.get("output", "") or "")
        user_msg = _format_prompt(instruction, inp)

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        else:
            prompt_only = f"### Instrução:\n{user_msg}\n\n### Resposta:\n"
            full_text = prompt_only + output

        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, truncation=True, max_length=seq_len, add_special_tokens=False)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}

    return ds_val.shuffle(seed=seed + 1).map(encode, remove_columns=ds_val.column_names)


def _sft_data_collator(tokenizer: Any, pad_to_multiple_of: int = 8):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = int(batch["input_ids"].shape[1])
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            seq = f["labels"][:max_len]
            if seq:
                labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        batch["labels"] = labels
        return batch

    return collate


def _try_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg_cpt: Dict[str, Any] = json.loads(args.cpt_config.read_text(encoding="utf-8"))
    cfg_sft: Dict[str, Any] = json.loads(args.sft_config.read_text(encoding="utf-8"))

    seed = int(cfg_cpt["experiment"]["seed"])
    set_seeds(seed)

    repo_root = Path(__file__).resolve().parents[3]
    env = collect_env_info()
    env["git_commit"] = try_get_git_commit(repo_root)

    summary: Dict[str, Any] = {"env": env, "configs": {"cpt": str(args.cpt_config), "sft": str(args.sft_config)}}

    # --- CPT (BrWaC)
    cpt_model_name = str(cfg_cpt["model"]["name_or_path"])
    cpt_tokenizer = AutoTokenizer.from_pretrained(cpt_model_name, use_fast=True)
    if cpt_tokenizer.pad_token is None:
        cpt_tokenizer.pad_token = cpt_tokenizer.eos_token

    cpt_eval_ds = _load_cpt_eval_dataset(repo_root, cfg_cpt, cpt_tokenizer)
    if args.limit_cpt_eval and args.limit_cpt_eval > 0:
        cpt_eval_ds = cpt_eval_ds.select(range(min(args.limit_cpt_eval, len(cpt_eval_ds))))

    cpt_bnb = _build_bnb_config(cfg_cpt["model"])
    cpt_base = AutoModelForCausalLM.from_pretrained(
        cpt_model_name,
        quantization_config=cpt_bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    cpt_collator = DataCollatorForLanguageModeling(tokenizer=cpt_tokenizer, mlm=False)
    cpt_base_metrics = _evaluate(
        model=cpt_base,
        dataset=cpt_eval_ds,
        data_collator=cpt_collator,
        per_device_eval_batch_size=int(cfg_cpt["training"]["per_device_eval_batch_size"]),
        bf16=bool(cfg_cpt["training"].get("bf16", True)),
    )

    cpt_adapter_dir = _resolve(repo_root, str(cfg_cpt["training"]["output_dir"])) / "adapter"
    cpt_with_adapter = PeftModel.from_pretrained(cpt_base, str(cpt_adapter_dir), is_trainable=False)
    cpt_adapter_metrics = _evaluate(
        model=cpt_with_adapter,
        dataset=cpt_eval_ds,
        data_collator=cpt_collator,
        per_device_eval_batch_size=int(cfg_cpt["training"]["per_device_eval_batch_size"]),
        bf16=bool(cfg_cpt["training"].get("bf16", True)),
    )

    summary["cpt"] = {
        "dataset": str(cfg_cpt["data"]["dataset_jsonl"]),
        "eval_samples": len(cpt_eval_ds),
        "base": cpt_base_metrics,
        "adapter": cpt_adapter_metrics,
        "delta": {
            "eval_loss": cpt_base_metrics["eval_loss"] - cpt_adapter_metrics["eval_loss"],
            "perplexity": cpt_base_metrics["perplexity"] - cpt_adapter_metrics["perplexity"],
            "perplexity_reduction_ratio": (
                0.0
                if cpt_base_metrics["perplexity"] in (0.0, float("inf"))
                else 1.0 - (cpt_adapter_metrics["perplexity"] / cpt_base_metrics["perplexity"])
            ),
        },
        "training_log": _try_load_json(repo_root / "versions" / "trilha2-lora" / "logs" / "train_cpt_qlora.json"),
    }

    del cpt_with_adapter
    del cpt_base
    torch.cuda.empty_cache()

    # --- SFT (Canarim)
    sft_model_name = str(cfg_sft["model"]["name_or_path"])
    sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name, use_fast=True)
    sft_tokenizer.padding_side = "right"
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token

    sft_eval_ds = _load_sft_eval_dataset(repo_root, cfg_sft, sft_tokenizer)
    if args.limit_sft_eval and args.limit_sft_eval > 0:
        sft_eval_ds = sft_eval_ds.select(range(min(args.limit_sft_eval, len(sft_eval_ds))))

    sft_bnb = _build_bnb_config(cfg_sft["model"])
    sft_base = AutoModelForCausalLM.from_pretrained(
        sft_model_name,
        quantization_config=sft_bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    sft_collator = _sft_data_collator(sft_tokenizer)
    sft_base_metrics = _evaluate(
        model=sft_base,
        dataset=sft_eval_ds,
        data_collator=sft_collator,
        per_device_eval_batch_size=int(cfg_sft["training"]["per_device_eval_batch_size"]),
        bf16=bool(cfg_sft["training"].get("bf16", True)),
    )

    sft_adapter_dir = _resolve(repo_root, str(cfg_sft["training"]["output_dir"])) / "adapter"
    sft_with_adapter = PeftModel.from_pretrained(sft_base, str(sft_adapter_dir), is_trainable=False)
    sft_adapter_metrics = _evaluate(
        model=sft_with_adapter,
        dataset=sft_eval_ds,
        data_collator=sft_collator,
        per_device_eval_batch_size=int(cfg_sft["training"]["per_device_eval_batch_size"]),
        bf16=bool(cfg_sft["training"].get("bf16", True)),
    )

    summary["sft"] = {
        "dataset": str(cfg_sft["data"]["val_jsonl"]),
        "eval_samples": len(sft_eval_ds),
        "base": sft_base_metrics,
        "adapter": sft_adapter_metrics,
        "delta": {
            "eval_loss": sft_base_metrics["eval_loss"] - sft_adapter_metrics["eval_loss"],
            "perplexity": sft_base_metrics["perplexity"] - sft_adapter_metrics["perplexity"],
            "perplexity_reduction_ratio": (
                0.0
                if sft_base_metrics["perplexity"] in (0.0, float("inf"))
                else 1.0 - (sft_adapter_metrics["perplexity"] / sft_base_metrics["perplexity"])
            ),
        },
        "training_log": _try_load_json(repo_root / "versions" / "trilha2-lora" / "logs" / "train_sft_qlora.json"),
    }

    # Top-level timing
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_path = _resolve(repo_root, str(args.out))
    write_json(out_path, summary)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
