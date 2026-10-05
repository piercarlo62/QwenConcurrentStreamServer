# ASR Concurrent Stream Server

Concurrent streaming gRPC server and client for Qwen3-ASR with vLLM continuous batching.

## Installation

```bash
pip install asr-concurrent-stream
```

For client-side audio loading and VAD segmentation:

```bash
pip install asr-concurrent-stream[client]
```

For development (regenerating protobuf code):

```bash
pip install asr-concurrent-stream[dev]
```

**Note:** The server requires `vllm` and `transformers` to run the Qwen3-ASR model. The model is loaded directly via vLLM's `AsyncLLMEngine` — no separate `qwen-asr` package needed.

## Quick Start

### Start the server

```bash
asr-concurrent-server serve
```

All configuration via CLI arguments:

```bash
asr-concurrent-server serve \
    --model Qwen/Qwen3-ASR-1.7B \
    --port 8000 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 16 \
    --max-concurrent-streams 15 \
    --health-port 8080
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-ASR-1.7B` | Model path or HuggingFace ID |
| `--port` | `8000` | gRPC server port |
| `--gpu-memory-utilization` | `0.30` | GPU memory fraction |
| `--max-model-len` | `4096` | Maximum model sequence length |
| `--max-num-batched-tokens` | `2048` | Max tokens per vLLM batch |
| `--max-num-seqs` | `16` | Max concurrent sequences in vLLM |
| `--max-concurrent-streams` | `15` | Maximum concurrent streams (excess queued) |
| `--health-port` | `8080` | HTTP health check port |

Environment variables (`ASR_PORT`, `ASR_MODEL_PATH`, etc.) are also supported as fallback.

### Run the client

```bash
asr-concurrent-client --audio path/to/audio.wav --language Italian
```

## Architecture

### vLLM Continuous Batching

The server uses vLLM's `AsyncLLMEngine` for GPU-level continuous batching. Each stream has its own worker coroutine that submits inference requests to the shared async engine. vLLM's scheduler dynamically batches prefill and decode across all active streams.

```
Client gRPC bidir stream
  → Per-stream asyncio.Queue + worker
    → AsyncLLMEngine.generate() (shared across streams)
      → vLLM scheduler batches at GPU level
        → Results → per-stream queue → gRPC response yield
```

### Sliding Window ASR

Instead of re-encoding the full accumulated audio for each chunk (quadratic cost), the server uses a fixed sliding window:

- **Past context**: 2.0s before current chunk
- **Present**: 0.5s current chunk
- **Future context**: 0.5s lookahead from buffer

This bounds prefill cost to ~3 seconds of audio regardless of total utterance length.

### Stream Queuing

When `MAX_CONCURRENT_STREAMS` is reached, new streams are queued instead of rejected. As soon as one stream finishes, the next queued stream starts automatically.

## Performance

| Concurrent Streams | Last chunk → Final (avg) | Wall time |
|--------------------|-------------------------|-----------|
| 1 | 60ms | — |
| 10 | 83ms | 5.5s |
| 50 | 171ms | 5.7s |

## Dependencies

| Package | Purpose |
|---------|---------|
| `grpcio` | gRPC framework |
| `numpy` | Audio array handling |
| `protobuf` | Protocol buffer serialization |
| `vllm` | vLLM async inference engine with continuous batching |
| `transformers` | Model processor and tokenizer |

Optional client dependencies: `soundfile` or `scipy` (for audio loading/resampling), `torch` (for Silero VAD segmentation).

---

*Docker image for the Qwen3-ASR concurrent streaming gRPC server.*

## Image Composition

| Layer | Detail |
|-------|--------|
| **Base OS** | Ubuntu 24.04 LTS |
| **CUDA** | NVIDIA CUDA 12.8.0 (`nvidia/cuda:12.8.0-runtime-ubuntu24.04`) |
| **Python** | 3.12 (deadsnakes PPA), installed into `/opt/venv` |
| **Model** | `Qwen/Qwen3-ASR-1.7B` baked into the image |
| **Server port** | Container `8002` (map to any host port at runtime) |

## Installation Sequence

1. **System dependencies** — `curl`, `git`, `wget`, `build-essential`, `ffmpeg`, `libsox-*`, `libsndfile1`, `libopus0`, `libffi-dev`.
2. **Python 3.12** — installed via deadsnakes PPA, venv created at `/opt/venv`, binaries symlinked to `/usr/local/bin`.
3. **pip** — upgraded to latest `pip`, `setuptools`, `wheel`.
4. **`vllm`** — installs vLLM with all required dependencies (transformers, torch, etc.). The server uses vLLM's `AsyncLLMEngine` directly to load and run the Qwen3-ASR model.
5. **flash-attention** — prebuilt wheel for CUDA 12.8 + PyTorch 2.9 + Python 3.12.
6. **Project files** — `ASR_Concurrent_Stream/` copied to `/app/ASR_Concurrent_Stream/`.
7. **Model download** — `Qwen/Qwen3-ASR-1.7B` downloaded from HuggingFace to `/app/models/Qwen3-ASR-1.7B`.
8. **Launcher patch** — `run_asr_grpc_server.sh` is updated:
   - `PORT` → `8002`
   - `MODEL_PATH` → `/app/models/Qwen3-ASR-1.7B`
   - `QWEN3_ASR_PATH` is no longer needed (vLLM loads model directly)

## Build & Run

```bash
# Build
docker build -t qwen3-asr-server:latest .

# Run (map host port 8002 to container port 8002)
docker run -d \
    --name qwen3-asr-server \
    --gpus all \
    -p 8002:8002 \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --restart unless-stopped \
    qwen3-asr-server:latest

# View logs
docker logs -f qwen3-asr-server

# Stop & remove
docker stop qwen3-asr-server && docker rm qwen3-asr-server
```

## Server Configuration

Hard-coded defaults in `server/run_asr_grpc_server.sh`:

| Parameter | Value |
|-----------|-------|
| Port | `8002` |
| Model | `/app/models/Qwen3-ASR-1.7B` |
| GPU memory utilization | `0.35` |
| Max model length | `4096` |
| Max concurrent streams | `50` |

## Project Structure (in image)

```
/app/
├── ASR_Concurrent_Stream/
│   ├── proto/              # gRPC protobuf definitions
│   ├── server/
│   │   ├── asr_grpc_server.py
│   │   ├── _launch_server.py
│   │   ├── stream_manager.py
│   │   ├── inference_coordinator.py
│   │   ├── asr_model.py
│   │   ├── asr_processor.py
│   │   ├── asr_utils.py
│   │   └── run_asr_grpc_server.sh
│   └── client/
│       └── example_client.py
└── models/
    └── Qwen3-ASR-1.7B/      # Pre-downloaded model
```
