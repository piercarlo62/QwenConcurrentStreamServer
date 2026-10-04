#!/usr/bin/env bash
set -euo pipefail

# Launcher for ASR_Concurrent_Stream/server/asr_grpc_server.py
# All startup arguments are hard-coded below.

# --- Hard-coded startup arguments ---
PORT=8000
MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
GPU_MEMORY_UTILIZATION=0.30
MAX_MODEL_LEN=3072
MAX_NUM_BATCHED_TOKENS=3072
MAX_NUM_SEQS=16
MAX_CONCURRENT_STREAMS=50
HEALTH_PORT=8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

# Export args as env vars so _launch_server.py (a real file) can read them.
# vLLM's spawn multiprocessing requires __main__ to be a real file on disk.
export ASR_PORT="$PORT"
export ASR_MODEL_PATH="$MODEL_PATH"
export ASR_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
export ASR_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export ASR_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
export ASR_MAX_NUM_SEQS="$MAX_NUM_SEQS"
export ASR_MAX_CONCURRENT_STREAMS="$MAX_CONCURRENT_STREAMS"
export ASR_HEALTH_PORT="$HEALTH_PORT"

exec "$PYTHON" -u "${SCRIPT_DIR}/_launch_server.py"