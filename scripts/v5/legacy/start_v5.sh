#!/usr/bin/env bash
set -euo pipefail
cd /root/TCC
timestamp=$(date +%Y%m%d_%H%M%S)
log="versions/v5-transformer/logs/train_v5_${timestamp}_b192.log"
nohup ./run_v5_transformer.sh > "$log" 2>&1 &
echo "started: $log"
