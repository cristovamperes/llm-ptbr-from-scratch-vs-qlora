from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from common import collect_env_info, run_timers, set_seeds, try_get_git_commit, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação extrínseca (QA/sumarização/reescrita) — baseline vs SFT-QLoRA.")
    p.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("versions/trilha2-lora/analysis/extrinsic"),
        help="Saída do 09_prepare_extrinsic_eval_sets.py",
    )
    p.add_argument("--out", type=Path, default=Path("versions/trilha2-lora/analysis/eval_trilha2_extrinsic.json"))
    p.add_argument("--details_out", type=Path, default=Path("versions/trilha2-lora/analysis/eval_trilha2_extrinsic_details.jsonl"))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cost_usd_per_hour", type=float, default=0.30)

    p.add_argument("--model_base", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path("versions/trilha2-lora/outputs/sft_qlora_llama31_8b_instruct_canarim10k/adapter"),
    )

    p.add_argument("--max_prompt_tokens", type=int, default=1536)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens_qa", type=int, default=32)
    p.add_argument("--max_new_tokens_summarization", type=int, default=160)
    p.add_argument("--max_new_tokens_rewriting", type=int, default=160)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"invalid torch dtype: {name}")


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype("bfloat16"),
    )


def _format_user_prompt(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    if inp:
        return f"{instruction}\n\nEntrada:\n{inp}".strip()
    return instruction


def _build_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{user_prompt}\n\nResposta:"


@torch.inference_mode()
def _generate_batches(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[List[str], int]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: List[str] = []
    total_generated_tokens = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        gen = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        attn = batch.get("attention_mask")
        for j in range(gen.shape[0]):
            prompt_len = int(attn[j].sum().item()) if attn is not None else batch["input_ids"].shape[1]
            out_ids = gen[j][prompt_len:]
            total_generated_tokens += int(out_ids.shape[0])
            outputs.append(tokenizer.decode(out_ids, skip_special_tokens=True).strip())

    return outputs, total_generated_tokens


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _normalize_answer(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\b(o|a|os|as|um|uma|uns|umas)\\b", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return (text or "").strip()


def _em_f1(pred: str, ref: str) -> Tuple[float, float]:
    pred_n = _normalize_answer(_first_line(pred))
    ref_n = _normalize_answer(_first_line(ref))
    em = 1.0 if pred_n == ref_n and ref_n != "" else 0.0

    pred_toks = pred_n.split() if pred_n else []
    ref_toks = ref_n.split() if ref_n else []
    if not pred_toks or not ref_toks:
        return em, 0.0

    common: Dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in ref_toks:
        c = common.get(t, 0)
        if c > 0:
            num_same += 1
            common[t] = c - 1
    if num_same == 0:
        return em, 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    ref_toks = _normalize_answer(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _write_details(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))

    repo_root = Path(__file__).resolve().parents[3]
    env = collect_env_info()
    env["git_commit"] = try_get_git_commit(repo_root)

    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else (repo_root / args.dataset_dir)
    qa_path = dataset_dir / "canarim_extrinsic_qa.jsonl"
    summ_path = dataset_dir / "canarim_extrinsic_summarization.jsonl"
    rew_path = dataset_dir / "canarim_extrinsic_rewriting.jsonl"
    manifest_path = dataset_dir / "canarim_extrinsic_manifest.json"

    qa_rows = _read_jsonl(qa_path)
    summ_rows = _read_jsonl(summ_path)
    rew_rows = _read_jsonl(rew_path)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_base), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = _build_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_base),
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    adapter_dir = args.adapter_dir if args.adapter_dir.is_absolute() else (repo_root / args.adapter_dir)
    sft_base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_base),
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    sft_base_model.eval()
    sft_model = PeftModel.from_pretrained(sft_base_model, str(adapter_dir), is_trainable=False)
    sft_model.eval()

    def run_task(task_name: str, rows: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        prompts = [_build_chat_prompt(tokenizer, _format_user_prompt(r["instruction"], r.get("input", ""))) for r in rows]
        refs = [r["reference"] for r in rows]

        start = time.time()
        base_outs, base_gen_tokens = _generate_batches(
            base_model,
            tokenizer,
            prompts,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(args.batch_size),
        )
        mid = time.time()

        sft_outs, sft_gen_tokens = _generate_batches(
            sft_model,
            tokenizer,
            prompts,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(args.batch_size),
        )
        end = time.time()

        base_metrics: List[float] = []
        sft_metrics: List[float] = []
        base_em: List[float] = []
        sft_em: List[float] = []

        details: List[Dict[str, Any]] = []
        for r, ref, b, s in zip(rows, refs, base_outs, sft_outs):
            if task_name == "qa":
                bem, bf1 = _em_f1(b, ref)
                sem, sf1 = _em_f1(s, ref)
                base_em.append(bem)
                sft_em.append(sem)
                base_metrics.append(bf1)
                sft_metrics.append(sf1)
                metrics_row = {
                    "base": {"em": bem, "f1": bf1},
                    "sft": {"em": sem, "f1": sf1},
                }
            else:
                b_rouge = _rouge_l_f1(b, ref)
                s_rouge = _rouge_l_f1(s, ref)
                base_metrics.append(b_rouge)
                sft_metrics.append(s_rouge)
                metrics_row = {
                    "base": {"rouge_l_f1": b_rouge},
                    "sft": {"rouge_l_f1": s_rouge},
                }

            details.append(
                {
                    "task": task_name,
                    "hash": r.get("hash") or _sha1(r["instruction"] + "\n" + (r.get("input", "") or "") + "\n" + ref),
                    "instruction": r["instruction"],
                    "input": r.get("input", ""),
                    "reference": ref,
                    "base_output": b,
                    "sft_output": s,
                    "metrics": metrics_row,
                }
            )

        def mean(xs: List[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))

        summary: Dict[str, Any] = {
            "task": task_name,
            "n": len(rows),
            "max_new_tokens": int(max_new_tokens),
            "timers": {
                "base_sec": mid - start,
                "sft_sec": end - mid,
                "total_sec": end - start,
            },
            "generation_tokens": {
                "base_generated_tokens": int(base_gen_tokens),
                "sft_generated_tokens": int(sft_gen_tokens),
            },
        }

        if task_name == "qa":
            summary["metrics"] = {
                "base": {"em": mean(base_em), "f1": mean(base_metrics)},
                "sft": {"em": mean(sft_em), "f1": mean(sft_metrics)},
            }
        else:
            summary["metrics"] = {
                "base": {"rouge_l_f1": mean(base_metrics)},
                "sft": {"rouge_l_f1": mean(sft_metrics)},
            }

        return summary, details

    qa_sum, qa_details = run_task("qa", qa_rows, int(args.max_new_tokens_qa))
    summ_sum, summ_details = run_task("summarization", summ_rows, int(args.max_new_tokens_summarization))
    rew_sum, rew_details = run_task("rewriting", rew_rows, int(args.max_new_tokens_rewriting))

    all_details = qa_details + summ_details + rew_details
    _write_details(repo_root / args.details_out, all_details)

    total_duration_sec = qa_sum["timers"]["total_sec"] + summ_sum["timers"]["total_sec"] + rew_sum["timers"]["total_sec"]
    total_cost_usd = (total_duration_sec / 3600.0) * float(args.cost_usd_per_hour)

    out: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env,
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "models": {
            "baseline": {"model": str(args.model_base), "use_4bit": True},
            "sft": {"model": str(args.model_base), "adapter_dir": str(args.adapter_dir), "use_4bit": True},
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "batch_size": int(args.batch_size),
        },
        "tasks": {
            "qa": qa_sum,
            "summarization": summ_sum,
            "rewriting": rew_sum,
        },
        "cost": {
            "cost_usd_per_hour": float(args.cost_usd_per_hour),
            "total_duration_sec": float(total_duration_sec),
            "total_cost_usd": float(total_cost_usd),
        },
    }

    write_json(repo_root / args.out, out)
    print(f"[OK] Wrote: {repo_root / args.out}")
    print(f"[OK] Details: {repo_root / args.details_out}")


if __name__ == "__main__":
    main()
