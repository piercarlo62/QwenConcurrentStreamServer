"""
Launcher for asr_grpc_server.py — called by run_asr_grpc_server.sh.
All arguments are read from environment variables set by the shell script.
A real file on disk is required so vLLM's multiprocessing spawn can re-import __main__.
"""

import asyncio
import os
import signal
import sys

# Ensure the project root (parent of server/) is on sys.path so that
# `server`, `proto`, and `client` packages can be imported when this
# script is run directly (not via the installed entry point).
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)
for _p in (_PROJECT_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.asr_grpc_server import serve  # noqa: E402


async def _main() -> None:
    port                  = int(os.environ["ASR_PORT"])
    model_path            = os.environ["ASR_MODEL_PATH"]
    gpu_memory_util       = float(os.environ["ASR_GPU_MEMORY_UTILIZATION"])
    max_model_len         = int(os.environ["ASR_MAX_MODEL_LEN"])
    max_num_batched_tokens= int(os.environ.get("ASR_MAX_NUM_BATCHED_TOKENS", "2048"))
    max_num_seqs          = int(os.environ.get("ASR_MAX_NUM_SEQS", "16"))
    max_concurrent_streams= int(os.environ["ASR_MAX_CONCURRENT_STREAMS"])
    health_port           = int(os.environ.get("ASR_HEALTH_PORT", "8080"))

    server, stream_manager, coordinator, http_health = await serve(
        port=port,
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_concurrent_streams=max_concurrent_streams,
        health_port=health_port,
    )

    print(f"ASR gRPC server running on port {port}", flush=True)

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
    asyncio.run(_main())


if __name__ == "__main__":
    main()
