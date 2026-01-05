#!/usr/bin/env bash
set -euo pipefail

# Gera amostras para os modelos char-level usando prompts padronizados.
# Requer ambiente virtual com dependências já ativado.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python analysis/generate_samples.py
