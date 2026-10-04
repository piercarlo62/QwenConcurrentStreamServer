"""
gRPC Server for ASR Concurrent Stream

Uses generated protobuf code from proto/asr.proto
"""

import asyncio
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, AsyncIterator

import grpc
import grpc.aio

import sys
import os

# Ensure the project root is on sys.path for direct execution
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# Import generated protobuf code
from proto import asr_pb2
from proto import asr_pb2_grpc

# Import server modules (same directory)
from server.stream_manager import StreamManager, StreamConfig
from server.inference_coordinator import InferenceCoordinator
from server.asr_model import Qwen3ASRModel

logger = logging.getLogger(__name__)


class HealthHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that responds to /health requests."""

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "healthy", "version": "1.0.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.debug(f"HTTP health: {format % args}")


def start_http_health_server(port: int) -> HTTPServer:
    """Start an HTTP health server in a background thread."""
    httpd = HTTPServer(("", port), HealthHTTPHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="http-health")
    thread.start()
    logger.info(f"HTTP health server started on port {port}")
    return httpd


class ASRServiceImpl(asr_pb2_grpc.ASRServiceServicer):
    """Implementation of ASRService gRPC service"""

    def __init__(
        self,
        stream_manager: StreamManager,
        inference_coordinator: InferenceCoordinator,
    ):
        self.stream_manager = stream_manager
        self.inference_coordinator = inference_coordinator
        self._response_queues = {}

    async def StreamTranscribe(
        self,
        request_iterator: AsyncIterator[asr_pb2.AudioChunkRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[asr_pb2.TranscriptionResponse]:
        """Bidirectional streaming RPC for real-time transcription"""
        stream_id = None
        response_queue = asyncio.Queue()
        request_task = None

        async def process_requests():
            nonlocal stream_id
            finalization_requested = False
            try:
                async for request in request_iterator:
                    if not request.stream_id:
                        logger.warning("Received request without stream_id")
                        continue

                    if stream_id is None:
                        stream_id = request.stream_id
                        logger.info(f"Stream {stream_id}: Starting transcription")

                        await self.inference_coordinator.register_result_queue(
                            stream_id, response_queue
                        )

                        stream = await self.stream_manager.get_stream(stream_id)
                        if stream is None:
                            # No prior StartStream call — auto-initialize with defaults
                            await self.stream_manager.create_stream(
                                stream_id=stream_id,
                                context="",
                                language=None,
                            )
                            await self.inference_coordinator.initialize_stream(stream_id)
                            logger.info(
                                f"Stream {stream_id}: Auto-initialized (no StartStream called)"
                            )
                        else:
                            logger.debug(
                                f"Stream {stream_id}: Using configuration from StartStream"
                            )

                    if request.stream_id != stream_id:
                        logger.warning(f"Mismatched stream_id")
                        continue

                    is_final = request.is_final

                    await self.inference_coordinator.submit_chunk(
                        stream_id=request.stream_id,
                        chunk_id=request.chunk_id,
                        audio_data=request.audio_data,
                        is_final=is_final,
                    )
                    logger.debug(
                        f"Stream {stream_id}: Submitted chunk {request.chunk_id}, is_final={is_final}"
                    )

                    await self.stream_manager.update_metrics(
                        stream_id=request.stream_id,
                        chunks_received=1,
                    )

                    if is_final and not finalization_requested:
                        finalization_requested = True
                        logger.info(
                            f"Stream {stream_id}: is_final triggered, requesting finalization"
                        )
                        await self.inference_coordinator.request_finalization(stream_id)

            except grpc.aio.RpcError as e:
                logger.error(f"Stream {stream_id}: RPC error: {e}")
            except Exception as e:
                logger.error(f"Stream {stream_id}: Error: {e}", exc_info=True)

        # Start request processing as background task
        request_task = asyncio.create_task(process_requests())

        try:
            while True:
                result = await response_queue.get()
                if result is None:
                    break
                response = asr_pb2.TranscriptionResponse(
                    stream_id=result.stream_id,
                    chunk_id=result.chunk_id,
                    partial_transcript=result.text,
                    is_final=result.is_final,
                    language=result.language,
                    confidence=0.0,
                    latency_ms=int(result.latency_ms),
                    processing_timestamp=result.timestamp_ms,
                )
                logger.debug(
                    f"Stream {result.stream_id}: yielding chunk {result.chunk_id}, "
                    f"is_final={result.is_final}"
                )
                yield response
                if result.is_final:
                    break
        except Exception as e:
            logger.error(f"Error in response yielding: {e}")
        finally:
            # Wait for request processor to complete
            if request_task:
                try:
                    await request_task
                except asyncio.CancelledError:
                    pass

            # Drain any results that arrived after the loop exited
            drained = 0
            while True:
                try:
                    result = await asyncio.wait_for(response_queue.get(), timeout=0.1)
                    if result is None:
                        break
                    drained += 1
                    response = asr_pb2.TranscriptionResponse(
                        stream_id=result.stream_id,
                        chunk_id=result.chunk_id,
                        partial_transcript=result.text,
                        is_final=result.is_final,
                        language=result.language,
                        confidence=0.0,
                        latency_ms=int(result.latency_ms),
                        processing_timestamp=result.timestamp_ms,
                    )
                    yield response
                except asyncio.TimeoutError:
                    break

            if drained:
                logger.debug(f"Stream {stream_id}: drained {drained} remaining results")

            if stream_id:
                await self.inference_coordinator.unregister_result_queue(stream_id)
                await self.inference_coordinator.cleanup_stream(stream_id)
                await self.stream_manager.close_stream(stream_id)
                logger.info(f"Stream {stream_id}: ended")

    async def StartStream(
        self,
        request: asr_pb2.StreamStartRequest,
        context: grpc.aio.ServicerContext,
    ) -> asr_pb2.StreamStartResponse:
        """Initialize a new stream"""
        try:
            stream_id = request.stream_id

            config = None
            if request.config:
                config = StreamConfig(
                    chunk_size_sec=request.config.chunk_size_sec or 0.5,
                    unfixed_chunk_num=request.config.unfixed_chunk_num or 2,
                    unfixed_token_num=request.config.unfixed_token_num or 5,
                )

            stream = await self.stream_manager.create_stream(
                stream_id=stream_id,
                context=request.context or "",
                language=request.language or None,
                config=config,
            )

            await self.inference_coordinator.initialize_stream(
                stream_id=stream_id,
                context=request.context or "",
                language=request.language or None,
                chunk_size_sec=stream.config.chunk_size_sec,
                unfixed_chunk_num=stream.config.unfixed_chunk_num,
                unfixed_token_num=stream.config.unfixed_token_num,
            )

            logger.info(f"Stream {stream_id}: Started")
            return asr_pb2.StreamStartResponse(
                stream_id=stream_id,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error starting stream: {e}", exc_info=True)
            return asr_pb2.StreamStartResponse(
                stream_id=request.stream_id, success=False, error=str(e)
            )

    async def FinalizeStream(
        self,
        request: asr_pb2.StreamFinalizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> asr_pb2.StreamFinalizeResponse:
        """Signal finalization with priority processing"""
        stream_id = request.stream_id
        start_time = time.perf_counter()

        try:
            stream = await self.stream_manager.get_stream(stream_id)
            if not stream:
                return asr_pb2.StreamFinalizeResponse(
                    stream_id=stream_id,
                    final_transcript="",
                    language="",
                    success=False,
                    error=f"Stream {stream_id} not found",
                )

            # Ensure finalization event exists before requesting finalization
            event = await self.inference_coordinator.get_finalization_event(stream_id)

            # Request priority finalization
            await self.stream_manager.request_finalization(stream_id)
            await self.inference_coordinator.request_finalization(stream_id)

            # Wait for finalization to complete (event-based, with timeout)
            try:
                await asyncio.wait_for(event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Stream {stream_id}: Finalization timed out after 10s"
                )

            final_transcript = (
                await self.inference_coordinator.get_final_transcript(stream_id) or ""
            )
            language = ""

            finalization_latency_ms = int((time.perf_counter() - start_time) * 1000)

            await self.stream_manager.complete_finalization(
                stream_id=stream_id,
                final_transcript=final_transcript,
                language=language,
            )

            await self.inference_coordinator.cleanup_stream(stream_id)
            await self.stream_manager.close_stream(stream_id)

            logger.info(
                f"Stream {stream_id}: finalized, "
                f"transcript='{final_transcript[:50]}...', "
                f"latency={finalization_latency_ms}ms"
            )

            return asr_pb2.StreamFinalizeResponse(
                stream_id=stream_id,
                final_transcript=final_transcript,
                language=language,
                success=True,
                finalization_latency_ms=finalization_latency_ms,
            )

        except Exception as e:
            logger.error(f"Error finalizing stream: {e}", exc_info=True)
            return asr_pb2.StreamFinalizeResponse(
                stream_id=stream_id,
                final_transcript="",
                language="",
                success=False,
                error=str(e),
            )

    async def GetStreamStatus(
        self,
        request: asr_pb2.StreamStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> asr_pb2.StreamStatusResponse:
        """Get stream status"""
        try:
            stream = await self.stream_manager.get_stream(request.stream_id)
            if not stream:
                return asr_pb2.StreamStatusResponse(
                    stream_id=request.stream_id,
                    status="not_found",
                    chunks_received=0,
                    last_chunk_time_ms=0,
                )

            return asr_pb2.StreamStatusResponse(
                stream_id=request.stream_id,
                status=stream.state.value,
                chunks_received=stream.metrics.chunks_received,
                last_chunk_time_ms=stream.metrics.last_chunk_time_ms,
            )

        except Exception as e:
            return asr_pb2.StreamStatusResponse(
                stream_id=request.stream_id,
                status="error",
                chunks_received=0,
                last_chunk_time_ms=0,
            )

    async def HealthCheck(
        self,
        request: asr_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> asr_pb2.HealthCheckResponse:
        """Health check endpoint"""
        return asr_pb2.HealthCheckResponse(
            serving=True,
            version="1.0.0",
        )


async def serve(
    port: int = 50051,
    stream_manager: Optional[StreamManager] = None,
    inference_coordinator: Optional[InferenceCoordinator] = None,
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 16,
    max_concurrent_streams: int = 30,
    max_batch_size: int = 8,
    batch_timeout_ms: int = 50,
    worker_threads: int = 4,
    health_port: int = 8080,
):
    """Start the gRPC server and HTTP health endpoint"""
    logger.info("Starting ASR Concurrent Stream server...")

    if stream_manager is None:
        stream_manager = StreamManager(max_concurrent_streams=max_concurrent_streams)

    if inference_coordinator is None:
        logger.info(f"Loading model from {model_path}...")
        model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_new_tokens=32,
        )
        logger.info("Model loaded")

        inference_coordinator = InferenceCoordinator(
            model=model,
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            worker_threads=worker_threads,
        )
        await inference_coordinator.start()

    servicer = ASRServiceImpl(stream_manager, inference_coordinator)

    server = grpc.aio.server()
    asr_pb2_grpc.add_ASRServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    await server.start()
    await stream_manager.start_cleanup_task()

    http_health_server = start_http_health_server(health_port)

    logger.info(f"Server started on port {port}")

    return server, stream_manager, inference_coordinator, http_health_server


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        server, stream_manager, coordinator, http_health = await serve()
        logger.info("Server running. Press Ctrl+C to stop.")
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            http_health.shutdown()
            await stream_manager.stop_cleanup_task()
            await coordinator.stop()
            await server.stop(0)

    asyncio.run(main())
