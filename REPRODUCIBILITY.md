# Reprodutibilidade (visão geral)

## O que este snapshot contém

- Logs, análises e scripts suficientes para auditoria dos números reportados no texto.
- Datasets pequenos/splits usados em Trilha 2 quando couberem no limite de tamanho.

## O que este snapshot NÃO contém

- Pesos/checkpoints (TensorFlow/Keras, adapters LoRA/QLoRA, etc.).
- Segredos/credenciais.
- Corpora grandes exportados (ex.: `corpus_v6.txt`), quando excedem o limite de tamanho.

## Como reproduzir (alto nível)

1. Crie um ambiente Python e instale dependências:
   - `pip install -r requirements.txt`
2. Execute os scripts em `scripts/` e use os artefatos em `analysis/` e `versions/` como referência.

Detalhes por versão e trilha estão documentados nos `README.md` dentro de `versions/`.