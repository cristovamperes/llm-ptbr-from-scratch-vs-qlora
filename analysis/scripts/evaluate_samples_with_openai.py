"""
Avalia amostras de texto usando um LLM da OpenAI, retornando notas e comentários.

O script lê o arquivo JSON de amostras geradas (`analysis/artifacts/samples/samples_brwac_custom_prompts.json`
por padrão), embaralha os itens para evitar viés, envia cada par (prompt, saída)
para o modelo especificado e salva as avaliações em JSON (`analysis/artifacts/results/evaluation_llm_review.json`).

Uso básico:
    python analysis/evaluate_samples_with_openai.py \
        --input analysis/artifacts/samples/samples_brwac_custom_prompts.json \
        --output analysis/artifacts/results/evaluation_llm_review.json \
        --model gpt-4o-mini

Antes de executar, defina a variável de ambiente OPENAI_API_KEY
com a chave de API (por exemplo, export OPENAI_API_KEY="sk-...").
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

DEFAULT_INPUT = Path("analysis/artifacts/samples/samples_brwac_custom_prompts.json")
DEFAULT_OUTPUT = Path("analysis/artifacts/results/evaluation_llm_review.json")
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
Você é um avaliador independente de qualidade textual. Sempre responda
apenas com JSON válido, sem comentários externos.

Critérios de avaliação (nota inteira de 0 a 5 para cada item):
- relevancia: quanto o texto aborda o tema e os detalhes do prompt? (0=irrelevante, 5=totalmente alinhado)
- coerencia: fluidez, consistência lógica, ausência de contradições ou quebras abruptas
- estilo: naturalidade do português, adequação de vocabulário, ausência de repetições artificiais

Também gere um campo `observacoes` com 2 ou 3 frases destacando pontos fortes
e fraquezas, sem tentar adivinhar modelo, arquitetura ou detalhes de treinamento.

Formato de resposta:
{
  "relevancia": int,
  "coerencia": int,
  "estilo": int,
  "observacoes": "..."
}
"""

USER_TEMPLATE = """\
Prompt original (contexto fornecido ao gerador):
{prompt}

Texto gerado para avaliação:
{output}

Forneça sua análise segundo os critérios indicados e retorne JSON.
"""


@dataclass
class Sample:
    model: str
    seed: int
    prompt: str
    output: str


def load_samples(path: Path) -> List[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: List[Sample] = []
    for model, entries in data.items():
        for entry in entries:
            samples.append(
                Sample(
                    model=model,
                    seed=int(entry["seed"]),
                    prompt=entry["prompt"],
                    output=entry["output"],
                )
            )
    return samples


def build_client(api_key: Optional[str]) -> OpenAI:
    if not api_key:
        raise RuntimeError(
            "Variável de ambiente OPENAI_API_KEY não definida. "
            "Defina a chave antes de rodar o script."
        )
    return OpenAI(api_key=api_key)


def request_evaluation(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> Dict[str, Any]:
    """Chama o modelo da OpenAI com política de retry exponencial simples."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Resposta inválida (não-JSON) após {max_retries} tentativas."
                    ) from exc
                time.sleep(1.5)
                continue
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Falha da API após {max_retries} tentativas: {exc}") from exc
            backoff = min(15.0, 2 ** attempt)
            time.sleep(backoff)
    raise RuntimeError("Falha inesperada ao solicitar avaliação.")


def evaluate_samples(
    samples: Iterable[Sample],
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, 1):
        user_prompt = USER_TEMPLATE.format(prompt=sample.prompt, output=sample.output)
        evaluation = request_evaluation(
            client,
            model=model,
            prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        results.append(
            {
                "model": sample.model,
                "seed": sample.seed,
                "prompt": sample.prompt,
                "output": sample.output,
                "evaluation": evaluation,
            }
        )
        print(f"[INFO] Avaliação concluída ({idx}) {sample.model} seed={sample.seed}")
    return results


def summarize_scores(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    aggregates: Dict[str, Dict[str, Any]] = {}
    for item in results:
        model = item["model"]
        evaluation = item["evaluation"]
        if model not in aggregates:
            aggregates[model] = {
                "count": 0,
                "relevancia_total": 0,
                "coerencia_total": 0,
                "estilo_total": 0,
            }
        agg = aggregates[model]
        agg["count"] += 1
        agg["relevancia_total"] += int(evaluation.get("relevancia", 0))
        agg["coerencia_total"] += int(evaluation.get("coerencia", 0))
        agg["estilo_total"] += int(evaluation.get("estilo", 0))

    summary: Dict[str, Any] = {}
    for model, agg in aggregates.items():
        count = max(1, agg["count"])
        summary[model] = {
            "amostras": count,
            "relevancia_media": agg["relevancia_total"] / count,
            "coerencia_media": agg["coerencia_total"] / count,
            "estilo_media": agg["estilo_total"] / count,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia saídas geradas com um modelo da OpenAI.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Arquivo JSON de amostras.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Arquivo de saída com avaliações.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="ID do modelo OpenAI a ser usado.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura do avaliador.")
    parser.add_argument("--max-tokens", type=int, default=400, help="Limite de tokens de resposta.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout por requisição (segundos).")
    parser.add_argument("--max-retries", type=int, default=4, help="Tentativas em caso de falha.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed para embaralhar a ordem das amostras.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de amostras inexistente: {args.input}")

    samples = load_samples(args.input)
    random.Random(args.seed).shuffle(samples)

    api_key = os.environ.get("OPENAI_API_KEY") or ""
    client = build_client(api_key)

    evaluations = evaluate_samples(
        samples,
        client=client,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    summary = summarize_scores(evaluations)

    payload = {
        "input_file": str(args.input),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "evaluations": evaluations,
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Avaliações salvas em {args.output}")


if __name__ == "__main__":
    main()
