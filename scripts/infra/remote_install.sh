#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tcc_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt