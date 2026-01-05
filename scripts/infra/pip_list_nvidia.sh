#!/usr/bin/env bash
source tcc_venv/bin/activate
pip list | grep nvidia || true