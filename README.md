# LLM PT-BR: treino do zero (v1–v6) vs QLoRA (CPT/SFT)

Este repositório reúne os artefatos do meu Trabalho de Conclusão de Curso (TCC) do **MBA em Ciência de Dados (ICMC/USP)**.

- **Título:** Um caminho estratégico para a soberania digital através de LLMs nacionais
- **Autor:** Cristovam Belizário Peres
- **Ano:** 2026

## O que foi investigado

Sob restrições realistas (GPU única), o trabalho compara duas estratégias para obter um modelo em PT-BR:

- **Trilha 1 — Treinar do zero (v1–v6):** protótipos em escala reduzida para evidenciar custos, limitações e descobertas (com foco em limpeza/tokenização e trade-offs de arquitetura).
- **Trilha 2 — Pós-treinamento eficiente (QLoRA):** (i) *continued pretraining* (CPT) em PT-BR (BrWaC 10k) e (ii) *supervised fine-tuning* (SFT) instrucional (Canarim 10k) sobre **LLaMA 3.1-8B**, com avaliação extrínseca (QA/sumarização/reescrita) repetida em múltiplas seeds.

## Documento

- PDF (versão oficial): [documento/pdf/tcc.pdf](documento/pdf/tcc.pdf) (exportado do Overleaf)
- Markdown (conversão best-effort): [documento/markdown/tcc.md](documento/markdown/tcc.md)

> Observação: o Markdown é uma conversão automática para facilitar leitura rápida e busca; o PDF é a referência.

## Onde estão as evidências

- [versions/](versions/) — versões/experimentos com `README.md`, logs e artefatos.
- [analysis/](analysis/) — métricas agregadas, guardrails, amostras e relatórios.
- [scripts/](scripts/) — scripts de preparo, treino e avaliação.
- [tcc_llm/](tcc_llm/) — módulos Python compartilhados (dataset/tokenização/modelos) usados pelos scripts.
- [docs/](docs/) — relatórios técnicos e notas (ex.: `DIAGNOSTICO_V5.md`, `RELATORIO_COMPARATIVO_V5_VS_V5B.md`).

No PDF, as referências a artefatos e logs são links clicáveis que apontam para estes caminhos no repositório.

## Reprodutibilidade (alto nível)

Veja [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Em resumo:

- este snapshot contém logs, análises e scripts suficientes para auditoria e replicação dos números reportados;
- **não** inclui pesos/checkpoints nem segredos/credenciais;
- alguns arquivos grandes podem ser omitidos quando excedem o limite de tamanho do GitHub.

## O que não está incluído

- O **fonte LaTeX** do TCC (o PDF é exportado do Overleaf).
- Pesos/checkpoints (TensorFlow/Keras e adapters LoRA/QLoRA).
- Arquivos temporários, credenciais e corpora grandes fora do limite de tamanho.
