# Trilha 2 — QLoRA em Llama 3.1-8B (CPT + SFT em PT-BR)

Esta trilha executa pós-treinamento eficiente (LoRA/QLoRA) em um modelo base open-source de maior capacidade, em **GPU única (RTX 4090, 24 GB)**, com rastreabilidade de tempo/tokens/custo.

> Snapshot público: os **adapters** (pesos LoRA/QLoRA) não são versionados aqui. A auditoria é feita via logs estruturados, datasets exportados e resultados de avaliação.

## O que foi executado

1) **CPT (continued pretraining)** em PT-BR (BrWaC 10k), sobre `meta-llama/Llama-3.1-8B` (QLoRA 4-bit)  
2) **SFT (supervised fine-tuning)** instrucional em PT-BR (Canarim 10k), sobre `meta-llama/Llama-3.1-8B-Instruct` (QLoRA 4-bit)  
3) **Avaliação extrínseca** (QA/sumarização/reescrita) repetida em **3 seeds**, em subconjunto do Canarim filtrado

## Resultados de treino (intrínsecos)

Extraído de:

- `logs/train_cpt_qlora.json`
- `logs/train_sft_qlora.json`

| Execução | Modelo base | Dados | seq_len | Épocas | Melhor eval loss | Tempo (h) | Custo (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| CPT (QLoRA) | Llama 3.1-8B | BrWaC 10k | 2048 | 3 | 2.1395 | 2.35 | 0.71 |
| SFT (QLoRA) | Llama 3.1-8B-Instruct | Canarim 10k | 2048 | 2 | 1.1035 | 1.06 | 0.32 |

## Avaliação extrínseca (evidência principal)

Resumo agregado (média ± desvio padrão em 3 seeds) de `analysis/eval_trilha2_extrinsic_multiseed.json`:

- **QA (F1)**: base `0.0630 ± 0.0179` → SFT `0.1133 ± 0.0179`
- **Sumarização (ROUGE-L F1)**: base `0.1537 ± 0.0035` → SFT `0.2037 ± 0.0103`
- **Reescrita (ROUGE-L F1)**: base `0.1729 ± 0.0212` → SFT `0.2955 ± 0.0205`

Arquivos:

- Resultado agregado: `analysis/eval_trilha2_extrinsic_multiseed.json`
- Resultados por seed: `analysis/eval_trilha2_extrinsic*.json`
- Detalhes por item: `analysis/eval_trilha2_extrinsic_details*.jsonl`
- Manifests dos conjuntos: `analysis/extrinsic*/canarim_extrinsic_manifest.json`

## Datasets e scripts

- CPT (BrWaC 10k): `datasets/brwac_cpt_10k.jsonl`
- SFT (Canarim 10k): `datasets/canarim_sft_10k_{train,val,test}.jsonl`
- Pipeline: `scripts/` (prepare → train → evaluate)
- Configs (hiperparâmetros/seeds): `configs/`

## Reprodutibilidade (observação)

Para reproduzir os treinos, é necessário:

- Python + PyTorch + Transformers/PEFT/BitsAndBytes
- Acesso aos modelos `meta-llama/Llama-3.1-8B*` no Hugging Face (gated)
