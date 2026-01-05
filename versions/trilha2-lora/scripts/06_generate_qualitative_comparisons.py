from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"invalid torch dtype: {name}")


@dataclass(frozen=True)
class GenerationCfg:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


def build_prompt(tokenizer: Any, user_prompt: str, *, system_prompt: Optional[str], use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}\n\nResposta:"
    return f"{user_prompt}\n\nResposta:"


@torch.inference_mode()
def generate_one(model: Any, tokenizer: Any, prompt: str, cfg: GenerationCfg) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": bool(cfg.do_sample),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        gen_kwargs["temperature"] = float(cfg.temperature)
        gen_kwargs["top_p"] = float(cfg.top_p)

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def load_base_model(model_name: str, *, use_4bit: bool, compute_dtype: str) -> tuple[Any, Any, Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype(compute_dtype),
        )

    # Force everything on GPU 0 (avoids auto offload decisions).
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    meta = {
        "model_name": model_name,
        "use_4bit": bool(use_4bit),
        "bnb_4bit_compute_dtype": compute_dtype,
    }
    return model, tokenizer, meta


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "utc": now_utc_iso(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except Exception:
        pass
    try:
        import bitsandbytes as bnb

        info["bitsandbytes"] = getattr(bnb, "__version__", None)
    except Exception:
        pass
    try:
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = {
                "device": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
    except Exception:
        pass
    return info


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "versions" / "trilha2-lora" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts: List[Dict[str, Any]] = [
        {
            "id": "p01_lora",
            "category": "conceito",
            "prompt": "Explique de forma simples o que é LoRA e o que é QLoRA. Responda em até 6 frases e em PT-BR.",
        },
        {
            "id": "p02_cpt",
            "category": "conceito",
            "prompt": "O que é Continued Pretraining (CPT) e quando ele faz sentido? Responda em PT-BR.",
        },
        {
            "id": "p03_resumo",
            "category": "sumarizacao",
            "prompt": "Resuma o texto a seguir em 3 bullets:\n\n"
            + "A soberania digital envolve a capacidade de um país controlar infraestrutura, dados e tecnologias críticas. "
            + "No caso de modelos de linguagem, treinar do zero exige grandes volumes de dados, computação e know-how, "
            + "o que torna difícil competir com modelos de fronteira. Uma alternativa pragmática é adaptar modelos abertos "
            + "por meio de pós-treinamento eficiente, reduzindo custo e tempo.",
        },
        {
            "id": "p04_expressao",
            "category": "cultura",
            "prompt": "O que significa a expressão brasileira 'ficar de molho'? Dê 2 exemplos de uso.",
        },
        {
            "id": "p05_email",
            "category": "redacao",
            "prompt": "Escreva um e-mail formal (8 a 12 linhas) solicitando acesso a um dataset acadêmico, em PT-BR.",
        },
        {
            "id": "p06_qa",
            "category": "qa_factual",
            "prompt": "Pergunta: Qual é a capital do Brasil? Responda de forma direta.",
        },
        {
            "id": "p07_math",
            "category": "raciocinio",
            "prompt": "Se eu treinar por 3 horas a $0.30/h, qual é o custo total em dólares? Mostre a conta.",
        },
        {
            "id": "p08_instrucao",
            "category": "instrucao",
            "prompt": "Dada a lista [maçã, banana, aveia, leite], gere uma receita simples em PT-BR com modo de preparo.",
        },
    ]

    cfg = GenerationCfg(max_new_tokens=256, do_sample=False)
    system_pt = "Você é um assistente útil. Responda em português brasileiro (PT-BR)."

    results: Dict[str, Any] = {
        "run_at": now_utc_iso(),
        "env": env_info(),
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        "prompts": prompts,
        "comparisons": [],
    }

    def run_pair(
        *,
        model_name: str,
        base_key: str,
        adapter_key: str,
        adapter_dir: Path,
        use_chat_template: bool,
    ) -> None:
        model, tokenizer, meta = load_base_model(model_name, use_4bit=True, compute_dtype="bfloat16")

        def gen_for_model(m: Any) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for p in prompts:
                prompt_txt = build_prompt(
                    tokenizer,
                    p["prompt"],
                    system_prompt=system_pt,
                    use_chat_template=use_chat_template,
                )
                out[p["id"]] = generate_one(m, tokenizer, prompt_txt, cfg)
            return out

        base_out = gen_for_model(model)

        adapted = PeftModel.from_pretrained(model, str(adapter_dir))
        adapted.eval()
        adapter_out = gen_for_model(adapted)

        pair: Dict[str, Any] = {
            "base": {"key": base_key, "model": meta},
            "adapter": {"key": adapter_key, "adapter_dir": str(adapter_dir)},
            "outputs": [],
        }
        for p in prompts:
            pid = p["id"]
            pair["outputs"].append(
                {
                    "prompt_id": pid,
                    "base": base_out.get(pid, ""),
                    "adapted": adapter_out.get(pid, ""),
                }
            )

        results["comparisons"].append(pair)

        del adapted
        del model
        torch.cuda.empty_cache()
        time.sleep(1.0)

    run_pair(
        model_name="meta-llama/Llama-3.1-8B",
        base_key="llama31_8b_base",
        adapter_key="llama31_8b_base+cpt_qlora",
        adapter_dir=repo_root / "versions" / "trilha2-lora" / "outputs" / "cpt_qlora_llama31_8b_brwac10k" / "adapter",
        use_chat_template=False,
    )

    run_pair(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_key="llama31_8b_instruct_base",
        adapter_key="llama31_8b_instruct_base+sft_qlora",
        adapter_dir=repo_root
        / "versions"
        / "trilha2-lora"
        / "outputs"
        / "sft_qlora_llama31_8b_instruct_canarim10k"
        / "adapter",
        use_chat_template=True,
    )

    out_path = out_dir / "qualitative_trilha2_llama31_8b_cpt_sft.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()