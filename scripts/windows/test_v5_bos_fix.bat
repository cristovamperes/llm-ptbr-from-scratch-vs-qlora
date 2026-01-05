@echo off
set MODEL_KEYS=v5_brwac_transformer_fixed
set OUTPUT_PATH=analysis/artifacts/samples/samples_v5_test_bos.json
python analysis/generate_samples.py
