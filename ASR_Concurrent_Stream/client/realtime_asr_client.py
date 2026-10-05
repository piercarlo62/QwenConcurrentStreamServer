import asyncio
import queue
import sys
import os
import uuid
import threading
from typing import Callable, Optional

import numpy as np
import grpc.aio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from proto import asr_pb2
from proto import asr_pb2_grpc

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 200
PAD_MS = 150
VAD_WINDOW = 512


class RealtimeASRClient:
    """Event-based realtime ASR client.

    Usage:
        client = RealtimeASRClient(host="localhost", port=8001)
        client.on_partial = lambda text: print("partial:", text)
        client.on_final = lambda text: print("final:", text)
        await client.start()

        # feed audio from any source (mic, file, websocket, ...)
        client.feed_audio(bytes)

        await client.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        language: str = "Italian",
        silence_duration_ms: int = SILENCE_DURATION_MS,
    ):
        self.host = host
        self.port = port
        self.language = language
        self.silence_duration_ms = silence_duration_ms

        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None

        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: list = []
        self._segment_idx = 0

    async def start(self):
        channel = grpc.aio.insecure_channel(
            f"{self.host}:{self.port}",
            options=[("grpc.max_receive_message_length", 10 * 1024 * 1024)],
        )
        self._stub = asr_pb2_grpc.ASRServiceStub(channel)
        self._loop = asyncio.get_event_loop()
        self._running = True
        self._tasks.append(asyncio.create_task(self._stream_loop())

    async def stop(self):
        self._running = False
        self._audio_queue.put(None)
        for t in self._tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw int16 mono PCM audio from any source."""
        if self._running:
            self._audio_queue.put(audio_bytes)

    async def _stream_loop(self):
        while self._running:
            segment_queue = asyncio.Queue()
            self._vad_result_queue = segment_queue

            vad_task = asyncio.create_task(
                self._vad_collector(self._audio_queue, segment_queue)
            )
            await self._stream_segment(segment_queue, self._segment_idx)
            vad_task.cancel()
            try:
                await vad_task
            except asyncio.CancelledError:
                pass
            self._segment_idx += 1

    async def _vad_collector(self, audio_q: asyncio.Queue, out_q: asyncio.Queue):
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
            trust_repo="true",
        )
        _, _, _, VADIterator, _ = utils
        vad_iter = VADIterator(
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=self.silence_duration_ms,
        )

        buffer = np.array([], dtype=np.float32)
        in_speech = False
        chunk_id = 0
        pad_samples = int(PAD_MS * SAMPLE_RATE / 1000)
        pre_speech_chunks = []

        while True:
            raw = await self._loop.run_in_executor(None, audio_q.get)
            if raw is None:
                if in_speech:
                    await out_q.put(("final", chunk_id, None))
                await out_q.put((None,))
                return

            chunk_f = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            buffer = np.concatenate([buffer, chunk_f])

            while len(buffer) >= VAD_WINDOW:
                window = buffer[:VAD_WINDOW]
                buffer = buffer[VAD_WINDOW:]
                int16_bytes = (window * 32767).astype(np.int16).tobytes()

                if not in_speech:
                    pre_speech_chunks.append(int16_bytes)
                    if len(pre_speech_chunks) * VAD_WINDOW > pad_samples:
                        pre_speech_chunks.pop(0)

                ev = vad_iter(torch.from_numpy(window), return_seconds=False)

                if ev is not None and "start" in ev and not in_speech:
                    in_speech = True
                    for pc in pre_speech_chunks:
                        await out_q.put(("audio", chunk_id, pc))
                        chunk_id += 1
                    pre_speech_chunks = []
                    await out_q.put(("audio", chunk_id, int16_bytes))
                    chunk_id += 1
                elif ev is not None and "end" in ev and in_speech:
                    await out_q.put(("final", chunk_id, int16_bytes))
                    in_speech = False
                    chunk_id = 0
                elif in_speech:
                    await out_q.put(("audio", chunk_id, int16_bytes))
                    chunk_id += 1

    async def _stream_segment(self, segment_q: asyncio.Queue, segment_idx: int):
        stream_id = str(uuid.uuid4())

        config = asr_pb2.StreamConfig(
            chunk_size_sec=0.5, unfixed_chunk_num=2, unfixed_token_num=5
        )
        start_resp = await self._stub.StartStream(
            asr_pb2.StreamStartRequest(
                stream_id=stream_id, language=self.language, config=config
            )
        )
        if not start_resp.success:
            return

        async def chunk_generator():
            while True:
                item = await segment_q.get()
                if item[0] is None:
                    return
                if item[0] == "audio":
                    _, cid, data = item
                    yield asr_pb2.AudioChunkRequest(
                        stream_id=stream_id,
                        chunk_id=cid,
                        audio_data=data,
                        sample_rate=SAMPLE_RATE,
                        is_final=False,
                    )
                elif item[0] == "final":
                    _, cid, data = item
                    if data is not None:
                        yield asr_pb2.AudioChunkRequest(
                            stream_id=stream_id,
                            chunk_id=cid,
                            audio_data=data,
                            sample_rate=SAMPLE_RATE,
                            is_final=True,
                        )
                    return

        try:
            async for response in self._stub.StreamTranscribe(chunk_generator()):
                if response.is_final:
                    if self.on_final:
                        self.on_final(response.partial_transcript)
                else:
                    if self.on_partial:
                        self.on_partial(response.partial_transcript)
        except Exception:
            pass
