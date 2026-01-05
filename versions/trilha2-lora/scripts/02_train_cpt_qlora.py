from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from common import collect_env_info, run_timers, set_seeds, try_get_git_commit, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPT com QLoRA (Llama 3.1-8B) em BrWaC (10k).")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name.lower() == "bfloat16":
        return torch.bfloat16
    if name.lower() == "float16":
        return torch.float16
    raise ValueError(f"torch dtype inválido: {name}")


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

    ds = load_dataset("json", data_files=str(cfg["data"]["dataset_jsonl"]))["train"].shuffle(seed=seed)
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    start = time.time()
    train_out = trainer.train()
    end = time.time()

    timers = run_timers(start, end)
    train_tokens_per_epoch = len(lm_datasets["train"]) * seq_len
    tokens_seen = int(train_tokens_per_epoch * float(cfg["training"]["num_train_epochs"]))
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
        "data": {"dataset_jsonl": str(cfg["data"]["dataset_jsonl"]), "seq_len": seq_len},
        "training": {
            "train_samples": len(lm_datasets["train"]),
            "eval_samples": len(lm_datasets["test"]),
            "train_tokens_per_epoch": int(train_tokens_per_epoch),
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

    log_path = repo_root / "versions" / "trilha2-lora" / "logs" / "train_cpt_qlora.json"
    write_json(log_path, log)
    print(f"[OK] Log: {log_path}")


if __name__ == "__main__":
    main()
