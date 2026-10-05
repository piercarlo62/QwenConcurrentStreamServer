"""
Inference Coordinator for ASR Concurrent Stream Server v1.0.6

Uses vLLM's AsyncLLMEngine for continuous batching at the GPU level.
Each stream has its own asyncio.Queue and worker coroutine. The vLLM
scheduler batches requests across all active streams dynamically.
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkRequest:
    stream_id: str
    chunk_id: int
    audio_data: np.ndarray
    is_final: bool


@dataclass
class TranscriptionResult:
    stream_id: str
    chunk_id: int
    text: str
    language: str
    is_final: bool
    latency_ms: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class InferenceCoordinator:

    def __init__(self, model: Any):
        self.model = model
        self.engine = model.engine
        self.sampling_params = model.sampling_params

        self._stream_states: Dict[str, Any] = {}
        self._states_lock = asyncio.Lock()

        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._worker_tasks: Dict[str, asyncio.Task] = {}

        self._result_queues: Dict[str, asyncio.Queue] = {}
        self._result_queues_lock = asyncio.Lock()

        self._finalization_requested: set = set()
        self._finalization_events: Dict[str, asyncio.Event] = {}
        self._finalization_events_lock = asyncio.Lock()

        self._running = False

        self._total_chunks_processed = 0
        self._lock = asyncio.Lock()

    async def register_result_queue(self, stream_id: str, queue: asyncio.Queue):
        async with self._result_queues_lock:
            self._result_queues[stream_id] = queue

    async def unregister_result_queue(self, stream_id: str):
        async with self._result_queues_lock:
            self._result_queues.pop(stream_id, None)

    async def initialize_stream(
        self,
        stream_id: str,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 0.5,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        context_before_sec: float = 5.0,
        context_after_sec: float = 0.5,
    ) -> None:
        async with self._states_lock:
            if stream_id in self._stream_states:
                return
            asr_state = self.model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
                context_before_sec=context_before_sec,
                context_after_sec=context_after_sec,
                stream_id=stream_id,
            )
            self._stream_states[stream_id] = asr_state

        queue = asyncio.Queue()
        self._stream_queues[stream_id] = queue
        self._worker_tasks[stream_id] = asyncio.create_task(
            self._stream_worker(stream_id, queue)
        )

    async def submit_chunk(
        self,
        stream_id: str,
        chunk_id: int,
        audio_data: bytes,
        is_final: bool = False,
    ) -> None:
        pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        request = ChunkRequest(
            stream_id=stream_id,
            chunk_id=chunk_id,
            audio_data=pcm_float32,
            is_final=is_final,
        )

        await self._stream_queues[stream_id].put(request)

        if is_final:
            await self.request_finalization(stream_id)

    async def request_finalization(self, stream_id: str) -> bool:
        async with self._states_lock:
            if stream_id not in self._stream_states:
                return False

        if stream_id in self._finalization_requested:
            return True
        self._finalization_requested.add(stream_id)

        request = ChunkRequest(
            stream_id=stream_id,
            chunk_id=-1,
            audio_data=np.array([], dtype=np.float32),
            is_final=True,
        )
        await self._stream_queues[stream_id].put(request)
        return True

    async def _stream_worker(self, stream_id: str, queue: asyncio.Queue):
        sentinel_sent = False
        try:
            while True:
                req = await queue.get()

                if req.chunk_id == -1:
                    result = await self._do_finalize(stream_id)
                    if result:
                        await self._send_to_stream_queue(result)
                    sentinel_sent = True
                    break

                pending = [req]
                while not queue.empty():
                    try:
                        pending.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                final_present = any(r.chunk_id == -1 for r in pending)
                if final_present:
                    audio_parts = [r.audio_data for r in pending if r.chunk_id >= 0]
                else:
                    audio_parts = [r.audio_data for r in pending]

                if audio_parts:
                    audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
                else:
                    audio = np.zeros(0, dtype=np.float32)

                async with self._states_lock:
                    state = self._stream_states.get(stream_id)

                if state is None:
                    break

                last_chunk_id = max(r.chunk_id for r in pending if r.chunk_id >= 0)

                inp = self.model.prepare_chunk(state, audio)
                while inp is not None:
                    request_id = f"{stream_id}-{state.chunk_id}"
                    gen_text = ""
                    async for output in self.engine.generate(inp, self.sampling_params, request_id):
                        gen_text = output.outputs[0].text
                    self.model.apply_output(state, gen_text)
                    inp = self.model.prepare_chunk(state, np.zeros(0, dtype=np.float32))

                result = TranscriptionResult(
                    stream_id=stream_id,
                    chunk_id=last_chunk_id,
                    text=state.text,
                    language=state.language,
                    is_final=False,
                    latency_ms=0.0,
                )
                await self._send_to_stream_queue(result)

                if final_present:
                    result = await self._do_finalize(stream_id)
                    if result:
                        await self._send_to_stream_queue(result)
                    sentinel_sent = True
                    break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stream {stream_id} worker error: {e}", exc_info=True)
            result = TranscriptionResult(
                stream_id=stream_id,
                chunk_id=-1,
                text="",
                language="",
                is_final=True,
                latency_ms=0.0,
            )
            await self._send_to_stream_queue(result)
            sentinel_sent = True
        finally:
            if not sentinel_sent:
                await self._send_sentinel(stream_id)

    async def _do_finalize(self, stream_id: str) -> Optional[TranscriptionResult]:
        start_time = time.perf_counter()

        try:
            async with self._states_lock:
                state = self._stream_states.get(stream_id)

            if state is None:
                return None

            text_before = state.text

            self.model._save_partials_debug(state.partials_list, stream_id)

            inp = self.model.finish_streaming_transcribe(state)
            if inp is not None:
                request_id = f"{stream_id}-finalize"
                gen_text = ""
                async for output in self.engine.generate(inp, self.sampling_params, request_id):
                    gen_text = output.outputs[0].text
                self.model.apply_finalize_output(state, gen_text)

            if len(state.text) < len(text_before):
                state.text = text_before

            latency_ms = (time.perf_counter() - start_time) * 1000

            result = TranscriptionResult(
                stream_id=stream_id,
                chunk_id=-1,
                text=state.text,
                language=state.language,
                is_final=True,
                latency_ms=latency_ms,
            )

            async with self._finalization_events_lock:
                event = self._finalization_events.get(stream_id)
                if event:
                    event.set()

            async with self._lock:
                self._total_chunks_processed += 1

            return result

        except Exception as e:
            logger.error(f"Error finalizing stream {stream_id}: {e}", exc_info=True)
            return None

    async def _send_to_stream_queue(self, result: Optional[TranscriptionResult]):
        if result is None:
            return
        try:
            async with self._result_queues_lock:
                queue = self._result_queues.get(result.stream_id)
            if queue:
                await queue.put(result)
                if result.is_final:
                    await queue.put(None)
        except Exception as e:
            logger.error(f"Error sending to stream queue: {e}")

    async def _send_sentinel(self, stream_id: str):
        """Send None sentinel to signal stream end."""
        try:
            async with self._result_queues_lock:
                queue = self._result_queues.get(stream_id)
            if queue:
                await queue.put(None)
        except Exception as e:
            logger.error(f"Error sending sentinel to stream {stream_id}: {e}")

    async def get_finalization_event(self, stream_id: str) -> asyncio.Event:
        async with self._finalization_events_lock:
            if stream_id not in self._finalization_events:
                self._finalization_events[stream_id] = asyncio.Event()
            return self._finalization_events[stream_id]

    async def start(self):
        self._running = True
        logger.info("Inference coordinator started")

    async def stop(self):
        self._running = False

        for stream_id, task in list(self._worker_tasks.items()):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self._worker_tasks.clear()
        self._stream_queues.clear()
        self._stream_states.clear()
        self._finalization_requested.clear()
        self._finalization_events.clear()

        logger.info("Inference coordinator stopped")

    async def cleanup_stream(self, stream_id: str) -> None:
        async with self._states_lock:
            self._stream_states.pop(stream_id, None)

        async with self._finalization_events_lock:
            self._finalization_events.pop(stream_id, None)

        self._finalization_requested.discard(stream_id)

        task = self._worker_tasks.pop(stream_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self._stream_queues.pop(stream_id, None)

    async def get_final_transcript(self, stream_id: str) -> Optional[str]:
        async with self._states_lock:
            state = self._stream_states.get(stream_id)
            if state:
                return getattr(state, "text", "") or ""
        return None
