from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from common import collect_env_info, set_seeds, try_get_git_commit, write_json  # noqa: E402


@dataclass(frozen=True)
class GenerationCfg:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera qualitativos alinhados aos datasets (CPT e SFT).")
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
        default=Path("versions/trilha2-lora/analysis/qualitative_trilha2_dataset_aligned.json"),
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-cpt-samples", type=int, default=2)
    p.add_argument("--cpt-prompt-tokens", type=int, default=256)
    p.add_argument("--cpt-max-new-tokens", type=int, default=160)

    p.add_argument("--n-sft-eval", type=int, default=200)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--sft-max-new-tokens", type=int, default=192)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"invalid torch dtype: {name}")


def _resolve(repo_root: Path, path: str | Path) -> Path:
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


def _load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model(model_name: str, *, bnb_config: Optional[BitsAndBytesConfig]) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


def _format_prompt(instruction: str, inp: str) -> str:
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
def _generate(model: Any, tokenizer: Any, prompt: str, cfg: GenerationCfg) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen = model.generate(
        **inputs,
        max_new_tokens=int(cfg.max_new_tokens),
        do_sample=bool(cfg.do_sample),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    out_ids = gen[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _normalize_for_f1(text: str) -> List[str]:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text.split() if text else []


def _f1(pred: str, ref: str) -> float:
    pred_toks = _normalize_for_f1(pred)
    ref_toks = _normalize_for_f1(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in ref_toks:
        c = common.get(t, 0)
        if c > 0:
            num_same += 1
            common[t] = c - 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return (2 * precision * recall) / (precision + recall)


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _load_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pick_cpt_prompts(
    *, docs: List[Dict[str, Any]], seed: int, n_samples: int, tokenizer: Any, prompt_tokens: int
) -> List[str]:
    rnd = random.Random(seed)
    texts = [str(d.get("text", "") or "") for d in docs]
    texts = [t for t in texts if t.strip()]
    chosen = rnd.sample(texts, k=min(n_samples, len(texts)))
    prompts: List[str] = []
    for t in chosen:
        ids = tokenizer(t, truncation=True, max_length=prompt_tokens, add_special_tokens=False)["input_ids"]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(prompt.strip())
    return prompts


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))

    repo_root = Path(__file__).resolve().parents[3]
    env = collect_env_info()
    env["git_commit"] = try_get_git_commit(repo_root)

    cfg_cpt: Dict[str, Any] = json.loads(args.cpt_config.read_text(encoding="utf-8"))
    cfg_sft: Dict[str, Any] = json.loads(args.sft_config.read_text(encoding="utf-8"))

    out: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env,
        "configs": {"cpt": str(args.cpt_config), "sft": str(args.sft_config)},
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }

    # --- CPT qualitative (text continuation)
    cpt_model_name = str(cfg_cpt["model"]["name_or_path"])
    cpt_tok = _load_tokenizer(cpt_model_name)
    cpt_bnb = _build_bnb_config(cfg_cpt["model"])

    cpt_dataset_path = _resolve(repo_root, str(cfg_cpt["data"]["dataset_jsonl"]))
    cpt_docs = _load_jsonl_all(cpt_dataset_path)
    cpt_prompts = _pick_cpt_prompts(
        docs=cpt_docs,
        seed=int(args.seed),
        n_samples=int(args.n_cpt_samples),
        tokenizer=cpt_tok,
        prompt_tokens=int(args.cpt_prompt_tokens),
    )

    cpt_base = _load_model(cpt_model_name, bnb_config=cpt_bnb)
    cpt_cfg = GenerationCfg(max_new_tokens=int(args.cpt_max_new_tokens), do_sample=False, temperature=0.0, top_p=1.0)
    cpt_base_outs = [_generate(cpt_base, cpt_tok, prompt, cpt_cfg) for prompt in cpt_prompts]

    cpt_adapter_dir = _resolve(repo_root, str(cfg_cpt["training"]["output_dir"])) / "adapter"
    cpt_adapted = PeftModel.from_pretrained(cpt_base, str(cpt_adapter_dir), is_trainable=False)
    cpt_adapted.eval()
    cpt_samples: List[Dict[str, Any]] = []
    for i, prompt in enumerate(cpt_prompts):
        adapted_out = _generate(cpt_adapted, cpt_tok, prompt, cpt_cfg)
        cpt_samples.append(
            {
                "sample_id": f"cpt_{i+1:02d}",
                "prompt": prompt,
                "base_continuation": cpt_base_outs[i],
                "cpt_continuation": adapted_out,
            }
        )

    out["cpt"] = {
        "dataset_jsonl": str(cfg_cpt["data"]["dataset_jsonl"]),
        "n_samples": len(cpt_samples),
        "prompt_tokens": int(args.cpt_prompt_tokens),
        "max_new_tokens": int(args.cpt_max_new_tokens),
        "samples": cpt_samples,
    }

    del cpt_adapted
    del cpt_base
    torch.cuda.empty_cache()

    # --- SFT qualitative (aligned with Canarim test split)
    sft_model_name = str(cfg_sft["model"]["name_or_path"])
    sft_tok = _load_tokenizer(sft_model_name)
    sft_tok.padding_side = "right"
    sft_bnb = _build_bnb_config(cfg_sft["model"])

    sft_test_path = _resolve(repo_root, str(cfg_sft["data"]["val_jsonl"]))
    sft_test_rows = _load_jsonl_all(sft_test_path)
    rnd = random.Random(int(args.seed))
    rnd.shuffle(sft_test_rows)
    sft_eval_rows = sft_test_rows[: max(1, min(int(args.n_sft_eval), len(sft_test_rows)))]

    sft_base = _load_model(sft_model_name, bnb_config=sft_bnb)
    sft_cfg = GenerationCfg(max_new_tokens=int(args.sft_max_new_tokens), do_sample=False, temperature=0.0, top_p=1.0)
    eval_entries: List[Tuple[str, str, str, str]] = []
    for row in sft_eval_rows:
        instruction = str(row.get("instruction", "") or "")
        inp = str(row.get("input", "") or "")
        ref = str(row.get("output", "") or "")
        user_prompt = _format_prompt(instruction, inp)
        prompt = _build_chat_prompt(sft_tok, user_prompt)
        eval_entries.append((instruction, inp, ref, prompt))

    base_outs = [_generate(sft_base, sft_tok, prompt, sft_cfg) for _, _, _, prompt in eval_entries]

    sft_adapter_dir = _resolve(repo_root, str(cfg_sft["training"]["output_dir"])) / "adapter"
    sft_adapted = PeftModel.from_pretrained(sft_base, str(sft_adapter_dir), is_trainable=False)
    sft_adapted.eval()
    scored: List[Tuple[float, Dict[str, Any]]] = []

    started = time.time()
    for idx, (instruction, inp, ref, prompt) in enumerate(eval_entries):
        base_out = base_outs[idx]
        sft_out = _generate(sft_adapted, sft_tok, prompt, sft_cfg)

        base_f1 = _f1(base_out, ref)
        sft_f1 = _f1(sft_out, ref)
        delta = sft_f1 - base_f1

        scored.append(
            (
                delta,
                {
                    "row_idx": idx,
                    "instruction": instruction,
                    "input": inp,
                    "reference": ref,
                    "base_output": base_out,
                    "sft_output": sft_out,
                    "base_f1": base_f1,
                    "sft_f1": sft_f1,
                    "delta_f1": delta,
                },
            )
        )

    ended = time.time()
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [row for _, row in scored[: int(args.top_k)]]
    bottom = [row for _, row in scored[-min(int(args.top_k), len(scored)) :]]

    out["sft"] = {
        "dataset_jsonl": str(cfg_sft["data"]["val_jsonl"]),
        "n_eval": len(sft_eval_rows),
        "max_new_tokens": int(args.sft_max_new_tokens),
        "timing_sec": ended - started,
        "mean_delta_f1": float(sum(d for d, _ in scored) / max(1, len(scored))),
        "top_k": top,
        "bottom_k": bottom,
    }

    # write
    out_path = _resolve(repo_root, args.out)
    write_json(out_path, out)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
