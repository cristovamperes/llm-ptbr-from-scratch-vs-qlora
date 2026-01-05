#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nohup bash run_train_v3.sh > train_v3.log 2>&1 &
echo $! > train_v3.pid
echo "started $!"