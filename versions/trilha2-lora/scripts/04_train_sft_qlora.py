from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from common import collect_env_info, run_timers, set_seeds, try_get_git_commit, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT com QLoRA (Llama 3.1-8B Instruct) em Canarim (10k).")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name.lower() == "bfloat16":
        return torch.bfloat16
    if name.lower() == "float16":
        return torch.float16
    raise ValueError(f"torch dtype inválido: {name}")


def _format_prompt(instruction: str, inp: str) -> str:
    if inp:
        return f"{instruction}\n\nEntrada:\n{inp}".strip()
    return instruction.strip()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["experiment"]["seed"])
    set_seeds(seed)

    repo_root = Path(__file__).resolve().parents[3]
    env = collect_env_info()
    env["git_commit"] = try_get_git_commit(repo_root)

    model_name = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if cfg["model"].get("use_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg["model"]["bnb_4bit_quant_type"]),
            bnb_4bit_use_double_quant=bool(cfg["model"]["bnb_4bit_use_double_quant"]),
            bnb_4bit_compute_dtype=_dtype(str(cfg["model"]["bnb_4bit_compute_dtype"])),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        bias=str(cfg["lora"]["bias"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds_train = load_dataset("json", data_files=str(cfg["data"]["train_jsonl"]))["train"]
    ds_val = load_dataset("json", data_files=str(cfg["data"]["val_jsonl"]))["train"]

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

    train_tok = ds_train.shuffle(seed=seed).map(encode, remove_columns=ds_train.column_names)
    val_tok = ds_val.shuffle(seed=seed + 1).map(encode, remove_columns=ds_val.column_names)

    train_args_kwargs: Dict[str, Any] = {
        "output_dir": str(cfg["training"]["output_dir"]),
        "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
        "per_device_train_batch_size": int(cfg["training"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["training"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["training"]["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "warmup_steps": int(cfg["training"]["warmup_steps"]),
        "lr_scheduler_type": str(cfg["training"]["lr_scheduler_type"]),
        "logging_steps": int(cfg["training"]["logging_steps"]),
        "evaluation_strategy": "steps",
        "eval_steps": int(cfg["training"]["eval_steps"]),
        "save_steps": int(cfg["training"]["save_steps"]),
        "save_total_limit": int(cfg["training"]["save_total_limit"]),
        "bf16": bool(cfg["training"].get("bf16", True)),
        "report_to": [],
    }

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "eval_strategy" in ta_params and "evaluation_strategy" in train_args_kwargs:
        train_args_kwargs["eval_strategy"] = train_args_kwargs.pop("evaluation_strategy")

    train_args = TrainingArguments(**train_args_kwargs)

    pad_to_multiple_of = 8

    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    start = time.time()
    train_out = trainer.train()
    end = time.time()

    timers = run_timers(start, end)
    tokens_seen = int(len(train_tok) * seq_len * float(cfg["training"]["num_train_epochs"]))
    tokens_per_sec = tokens_seen / max(1.0, timers.duration_sec)
    cost_per_hour = float(cfg["experiment"]["cost_usd_per_hour"])
    total_cost = (timers.duration_sec / 3600.0) * cost_per_hour

    best_eval_loss = None
    for row in trainer.state.log_history:
        if "eval_loss" in row:
            val = float(row["eval_loss"])
            best_eval_loss = val if best_eval_loss is None else min(best_eval_loss, val)

    out_dir = Path(str(cfg["training"]["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    trainer.save_model(out_dir / "adapter")

    log: Dict[str, Any] = {
        "experiment": cfg["experiment"]["name"],
        "config_path": str(args.config),
        "timers": timers.__dict__,
        "env": env,
        "model": {"name_or_path": model_name, "use_4bit": cfg["model"].get("use_4bit", False)},
        "data": {
            "train_jsonl": str(cfg["data"]["train_jsonl"]),
            "val_jsonl": str(cfg["data"]["val_jsonl"]),
            "seq_len": seq_len,
        },
        "training": {
            "train_samples": len(train_tok),
            "eval_samples": len(val_tok),
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tokens_per_sec,
            "best_eval_loss": best_eval_loss,
            "trainer_metrics": train_out.metrics,
        },
        "cost": {
            "cost_usd_per_hour": cost_per_hour,
            "total_cost_usd": total_cost,
        },
        "artifacts": {
            "output_dir": str(out_dir),
            "adapter_dir": str(out_dir / "adapter"),
        },
    }

    log_path = repo_root / "versions" / "trilha2-lora" / "logs" / "train_sft_qlora.json"
    write_json(log_path, log)
    print(f"[OK] Log: {log_path}")


if __name__ == "__main__":
    main()