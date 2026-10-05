"""
Launcher for asr_grpc_server.py v1.0.6 — called by run_asr_grpc_server.sh
and via the `asr-concurrent-server` entry point.

Usage:
    asr-concurrent-server serve --model Qwen/Qwen3-ASR-1.7B --port 8000

A real file on disk is required so vLLM's multiprocessing spawn can re-import __main__.
"""

import argparse
import asyncio
import os
import signal
import sys

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)
for _p in (_PROJECT_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.asr_grpc_server import serve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASR Concurrent Stream Server — Qwen3-ASR with vLLM continuous batching"
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the gRPC server")
    subparsers.add_parser("version", help="Print server version")

    serve_parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B"),
        help="Model path or HuggingFace ID (default: Qwen/Qwen3-ASR-1.7B)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ASR_PORT", "8000")),
        help="gRPC server port (default: 8000)",
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("ASR_GPU_MEMORY_UTILIZATION", "0.30")),
        help="GPU memory fraction (default: 0.30)",
    )
    serve_parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("ASR_MAX_MODEL_LEN", "4096")),
        help="Maximum model sequence length (default: 4096)",
    )
    serve_parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("ASR_MAX_NUM_BATCHED_TOKENS", "2048")),
        help="Max tokens per vLLM batch (default: 2048)",
    )
    serve_parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=int(os.environ.get("ASR_MAX_NUM_SEQS", "16")),
        help="Max concurrent sequences in vLLM (default: 16)",
    )
    serve_parser.add_argument(
        "--max-concurrent-streams",
        type=int,
        default=int(os.environ.get("ASR_MAX_CONCURRENT_STREAMS", "15")),
        help="Maximum concurrent streams (excess queued, default: 15)",
    )
    serve_parser.add_argument(
        "--health-port",
        type=int,
        default=int(os.environ.get("ASR_HEALTH_PORT", "8080")),
        help="HTTP health check port (default: 8080)",
    )
    serve_parser.add_argument(
        "--punctuate",
        action="store_true",
        default=False,
        help="Enable punctuation in transcription output",
    )

    return parser.parse_args()


async def _run_serve(args: argparse.Namespace) -> None:
    server, stream_manager, coordinator, http_health = await serve(
        port=args.port,
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_concurrent_streams=args.max_concurrent_streams,
        health_port=args.health_port,
        punctuate=args.punctuate,
    )

    print(f"ASR gRPC server running on port {args.port}", flush=True)

    stop_future: asyncio.Future = asyncio.Future()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: stop_future.set_result(None) if not stop_future.done() else None)

    try:
        await stop_future
    finally:
        http_health.shutdown()
        await stream_manager.stop_cleanup_task()
        await coordinator.stop()
        await server.stop(0)


def main() -> None:
    args = parse_args()

    if args.command == "serve":
        asyncio.run(_run_serve(args))
    elif args.command == "version":
        print("asr-concurrent-stream 1.0.8")
    elif args.command is None:
        print("Error: missing subcommand. Usage: asr-concurrent-server serve [args]", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: unknown subcommand '{args.command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
