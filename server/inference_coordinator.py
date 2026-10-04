"""
Inference Coordinator for ASR Concurrent Stream Server

Manages audio chunk processing with:
- Dynamic batching for throughput
- Priority queue for finalization (lowest latency)
- Worker pool for parallel processing
- Per-stream state management
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class ChunkRequest:
    """Represents an audio chunk waiting to be processed"""

    stream_id: str
    chunk_id: int
    audio_data: np.ndarray
    is_final: bool
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    processing_start_ms: Optional[int] = None


@dataclass
class TranscriptionResult:
    """Result from processing an audio chunk"""

    stream_id: str
    chunk_id: int
    text: str
    language: str
    is_final: bool
    latency_ms: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class InferenceCoordinator:
    """
    Coordinates inference across multiple concurrent streams.

    Features:
    - Dynamic batching (process multiple streams together)
    - Priority queue for finalization requests
    - Worker pool for parallel inference
    - Result callbacks for streaming results back to clients
    """

    def __init__(
        self,
        model: Any,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 50,
        worker_threads: int = 4,
    ):
        """
        Initialize the inference coordinator.

        Args:
            model: Qwen3ASRModel instance (vLLM backend)
            max_batch_size: Maximum streams to batch together
            batch_timeout_ms: Max time to wait for batch to fill
            worker_threads: Number of worker threads for inference
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.worker_threads = worker_threads

        # Per-stream ASR states
        self._stream_states: Dict[str, Any] = {}
        self._states_lock = asyncio.Lock()

        # Single FIFO queue — ordering guarantees each stream's chunks are
        # processed before its finalization marker.
        self._normal_queue: asyncio.Queue = asyncio.Queue()

        # Worker tasks
        self._worker_tasks: List[asyncio.Task] = []
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False

        # Thread pool for blocking inference
        self._thread_pool = ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="asr-inference",
        )

        # Per-stream result queues (stream_id -> asyncio.Queue)
        self._result_queues: Dict[str, asyncio.Queue] = {}
        self._result_queues_lock = asyncio.Lock()

        # Legacy callbacks (kept for compatibility)
        self._result_callback: Optional[Callable] = None
        self._finalize_callback: Optional[Callable] = None

        # Metrics
        self._total_chunks_processed = 0
        self._total_batches_processed = 0
        self._lock = threading.Lock()

        # Per-stream finalization events for FinalizeStream RPC synchronization
        self._finalization_events: Dict[str, asyncio.Event] = {}
        self._finalization_events_lock = asyncio.Lock()

    async def register_result_queue(self, stream_id: str, queue: asyncio.Queue):
        """Register a result queue for a stream"""
        async with self._result_queues_lock:
            self._result_queues[stream_id] = queue
            logger.info(f"Registered result queue for stream {stream_id}")

    async def unregister_result_queue(self, stream_id: str):
        """Unregister a result queue for a stream"""
        async with self._result_queues_lock:
            if stream_id in self._result_queues:
                del self._result_queues[stream_id]
                logger.info(f"Unregistered result queue for stream {stream_id}")

    def set_result_callback(self, callback: Callable):
        """Set result callback for legacy compatibility"""
        self._result_callback = callback

    def set_finalize_callback(self, callback: Callable):
        """Set finalize callback for legacy compatibility"""
        self._finalize_callback = callback

    async def initialize_stream(
        self,
        stream_id: str,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 0.5,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
    ) -> None:
        """Initialize streaming state for a new stream (idempotent)."""
        async with self._states_lock:
            if stream_id in self._stream_states:
                logger.debug(f"Stream {stream_id} already initialized, skipping")
                return
            asr_state = self.model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )
            self._stream_states[stream_id] = asr_state
            logger.debug(
                f"Initialized stream {stream_id} with chunk_size={chunk_size_sec}s"
            )

    async def submit_chunk(
        self,
        stream_id: str,
        chunk_id: int,
        audio_data: bytes,
        is_final: bool = False,
    ) -> None:
        """
        Submit an audio chunk for processing.

        Args:
            stream_id: Stream identifier
            chunk_id: Chunk sequence number
            audio_data: Raw audio bytes (16-bit PCM)
            is_final: Whether this is a final chunk (for auto-finalization)
        """
        # Convert bytes to numpy array
        pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        logger.info(
            f"submit_chunk: stream={stream_id}, chunk_id={chunk_id}, audio_len={len(pcm_float32)}, is_final={is_final}"
        )

        request = ChunkRequest(
            stream_id=stream_id,
            chunk_id=chunk_id,
            audio_data=pcm_float32,
            is_final=is_final,
        )

        await self._normal_queue.put(request)
        logger.info(f"Submitted chunk {chunk_id} for stream {stream_id}")

    async def request_finalization(self, stream_id: str) -> bool:
        """
        Request finalization for a stream (flush remaining buffer).

        Args:
            stream_id: Stream to finalize

        Returns:
            True if finalization was queued
        """
        async with self._states_lock:
            if stream_id not in self._stream_states:
                logger.warning(f"Cannot finalize unknown stream {stream_id}")
                return False

        # Signal finalization by putting a special marker in priority queue
        request = ChunkRequest(
            stream_id=stream_id,
            chunk_id=-1,  # Marker for finalization
            audio_data=np.array([], dtype=np.float32),
            is_final=True,
        )
        await self._normal_queue.put(request)
        logger.debug(f"Finalization requested for stream {stream_id}")
        return True

    async def _process_chunk(
        self, request: ChunkRequest
    ) -> Optional[TranscriptionResult]:
        """Process a single audio chunk via the thread pool."""
        start_time = time.perf_counter()

        async with self._states_lock:
            asr_state = self._stream_states.get(request.stream_id)

        if asr_state is None:
            logger.warning(f"Stream state not found for {request.stream_id}")
            return None

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._thread_pool,
                self.model.streaming_transcribe,
                request.audio_data,
                asr_state,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            async with self._states_lock:
                text = asr_state.text
                language = asr_state.language

            logger.debug(
                f"Stream {request.stream_id}: chunk {request.chunk_id} done, "
                f"latency={latency_ms:.1f}ms, text='{text[:30]}'"
            )

            result = TranscriptionResult(
                stream_id=request.stream_id,
                chunk_id=request.chunk_id,
                text=text,
                language=language,
                # Audio chunks are always partial; only _finalize_stream sets is_final=True
                is_final=False,
                latency_ms=latency_ms,
            )

            with self._lock:
                self._total_chunks_processed += 1

            return result

        except Exception as e:
            logger.error(
                f"Error processing chunk for {request.stream_id}: {e}", exc_info=True
            )
            return None

    async def _finalize_stream(self, stream_id: str) -> Optional[TranscriptionResult]:
        """Finalize a stream (flush remaining buffer)"""
        start_time = time.perf_counter()

        try:
            async with self._states_lock:
                asr_state = self._stream_states.get(stream_id)

            if asr_state is None:
                logger.warning(f"Stream state not found for {stream_id}")
                return None

            # Run blocking finalization in thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._thread_pool,
                self.model.finish_streaming_transcribe,
                asr_state,
            )

            finalization_latency_ms = (time.perf_counter() - start_time) * 1000

            # Get final result
            async with self._states_lock:
                text = asr_state.text
                language = asr_state.language

            result = TranscriptionResult(
                stream_id=stream_id,
                chunk_id=-1,  # Final chunk marker
                text=text,
                language=language,
                is_final=True,
                latency_ms=finalization_latency_ms,
            )

            # Call finalize callback
            if self._finalize_callback:
                self._finalize_callback(
                    stream_id, text, language, finalization_latency_ms
                )

            logger.debug(
                f"Finalized stream {stream_id}: "
                f"transcript='{text[:50]}...', latency={finalization_latency_ms:.1f}ms"
            )

            # Signal finalization completion so FinalizeStream RPC can unblock
            async with self._finalization_events_lock:
                event = self._finalization_events.get(stream_id)
                if event:
                    event.set()

            return result

        except Exception as e:
            logger.error(f"Error finalizing stream {stream_id}: {e}", exc_info=True)
            return None

    async def _batch_processor(self):
        """Process chunks in batches for better throughput."""
        logger.info("Batch processor started")

        while self._running:
            try:
                # Collect a batch from the single FIFO queue.
                # FIFO ordering ensures each stream's audio chunks are processed
                # before their finalization marker (chunk_id == -1).
                batch: List[ChunkRequest] = []
                try:
                    first_chunk = await asyncio.wait_for(
                        self._normal_queue.get(), timeout=self.batch_timeout_ms / 1000.0
                    )
                    batch.append(first_chunk)

                    while len(batch) < self.max_batch_size:
                        try:
                            chunk = await asyncio.wait_for(
                                self._normal_queue.get(), timeout=0.005
                            )
                            batch.append(chunk)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                logger.debug(f"Processing batch of {len(batch)} chunks")
                with self._lock:
                    self._total_batches_processed += 1

                # Group by stream_id: sequential within a stream, parallel across streams.
                # Finalization markers (chunk_id == -1) are handled in stream order.
                stream_batches: Dict[str, List[ChunkRequest]] = {}
                for req in batch:
                    stream_batches.setdefault(req.stream_id, []).append(req)

                async def _process_stream_batch(
                    reqs: List[ChunkRequest],
                ) -> List[TranscriptionResult]:
                    results: List[TranscriptionResult] = []
                    for req in reqs:
                        if req.chunk_id == -1:
                            result = await self._finalize_stream(req.stream_id)
                        else:
                            result = await self._process_chunk(req)
                        if result:
                            results.append(result)
                    return results

                batched_results = await asyncio.gather(
                    *[_process_stream_batch(reqs) for reqs in stream_batches.values()]
                )
                for stream_results in batched_results:
                    for result in stream_results:
                        await self._send_to_stream_queue(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)

        logger.info("Batch processor stopped")

    async def _send_to_stream_queue(self, result: TranscriptionResult):
        """Send result to the registered stream's response queue."""
        try:
            async with self._result_queues_lock:
                queue = self._result_queues.get(result.stream_id)
            if queue:
                await queue.put(result)
                logger.debug(
                    f"Stream {result.stream_id}: queued chunk {result.chunk_id}, "
                    f"is_final={result.is_final}, text='{result.text[:30]}'"
                )
                # Enqueue None sentinel so the response loop exits cleanly
                if result.is_final:
                    await queue.put(None)
            else:
                logger.warning(f"No queue registered for stream {result.stream_id}")

            if self._result_callback:
                self._result_callback(result)
        except Exception as e:
            logger.error(f"Error sending to stream queue: {e}")

    async def get_finalization_event(self, stream_id: str) -> asyncio.Event:
        """Get (or create) the finalization asyncio.Event for a stream."""
        async with self._finalization_events_lock:
            if stream_id not in self._finalization_events:
                self._finalization_events[stream_id] = asyncio.Event()
            return self._finalization_events[stream_id]

    async def start(self):
        """Start the inference coordinator"""
        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info(
            f"Inference coordinator started with {self.worker_threads} workers, _running={self._running}"
        )

    async def stop(self):
        """Stop the inference coordinator"""
        self._running = False

        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        self._thread_pool.shutdown(wait=True)
        logger.info("Inference coordinator stopped")

    async def get_final_transcript(self, stream_id: str) -> Optional[str]:
        """Get the final transcript for a stream"""
        async with self._states_lock:
            asr_state = self._stream_states.get(stream_id)
            if asr_state:
                text = getattr(asr_state, "text", "") or ""
                return text
        return None

    async def cleanup_stream(self, stream_id: str) -> None:
        """Clean up stream state and finalization event."""
        async with self._states_lock:
            self._stream_states.pop(stream_id, None)
        async with self._finalization_events_lock:
            self._finalization_events.pop(stream_id, None)
        logger.debug(f"Cleaned up stream {stream_id}")

    def get_metrics(self) -> dict:
        """Get coordinator metrics"""
        with self._lock:
            return {
                "total_chunks_processed": self._total_chunks_processed,
                "total_batches_processed": self._total_batches_processed,
                "max_batch_size": self.max_batch_size,
                "batch_timeout_ms": self.batch_timeout_ms,
                "worker_threads": self.worker_threads,
            }
