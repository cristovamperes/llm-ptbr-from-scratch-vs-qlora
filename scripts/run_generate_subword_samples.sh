#!/usr/bin/env bash
set -euo pipefail

# Gera amostras apenas para o modelo subword (v4).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export MODEL_KEYS="${MODEL_KEYS:-v4_brwac_subword}"
export OUTPUT_PATH="${OUTPUT_PATH:-analysis/artifacts/samples/samples_brwac_v4.json}"

python analysis/generate_samples.py
