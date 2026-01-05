#!/usr/bin/env bash
set -e
if [ -d "$HOME/.cache/huggingface/datasets" ]; then
  du -sh "$HOME/.cache/huggingface/datasets"
else
  echo "datasets cache not present"
fi