#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

exec "$PYTHON" -u -m server._launch_server serve \
    --model "Qwen/Qwen3-ASR-1.7B" \
    --port 8000 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 3072 \
    --max-num-batched-tokens 3072 \
    --max-num-seqs 16 \
    --max-concurrent-streams 50 \
    --health-port 8080
