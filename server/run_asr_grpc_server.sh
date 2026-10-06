#!/usr/bin/env bash
# ASR Concurrent Stream Server v1.0.6
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
    --kv-cache-memory-bytes 0 \
    --max-concurrent-streams 50 \
    --health-port 8080
