import asyncio
import queue
import sys
import os
import uuid
import threading

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


def load_silero_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
        trust_repo="true",
    )
    _, _, _, VADIterator, _ = utils
    return VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=SILENCE_DURATION_MS,
    )


async def mic_stream_producer(audio_queue, mic_queue):
    import torch

    vad_iter = load_silero_vad()
    buffer = np.array([], dtype=np.float32)
    in_speech = False
    chunk_id = 0
    pad_samples = int(PAD_MS * SAMPLE_RATE / 1000)
    pre_speech_chunks = []

    while True:
        raw = await asyncio.get_event_loop().run_in_executor(None, audio_queue.get)
        if raw is None:
            if in_speech:
                mic_queue.put_nowait(("final", chunk_id, raw))
            mic_queue.put_nowait((None,))
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
                    mic_queue.put_nowait(("audio", chunk_id, pc))
                    chunk_id += 1
                pre_speech_chunks = []
                mic_queue.put_nowait(("audio", chunk_id, int16_bytes))
                chunk_id += 1
            elif ev is not None and "end" in ev and in_speech:
                mic_queue.put_nowait(("final", chunk_id, int16_bytes))
                in_speech = False
                chunk_id = 0
            elif in_speech:
                mic_queue.put_nowait(("audio", chunk_id, int16_bytes))
                chunk_id += 1


async def stream_session(stub, mic_queue, segment_idx):
    stream_id = str(uuid.uuid4())
    prefix = f"seg {segment_idx}"

    config = asr_pb2.StreamConfig(
        chunk_size_sec=0.5, unfixed_chunk_num=2, unfixed_token_num=5
    )
    start_resp = await stub.StartStream(
        asr_pb2.StreamStartRequest(
            stream_id=stream_id, language="Italian", config=config
        )
    )
    if not start_resp.success:
        print(f"[{prefix}] StartStream failed: {start_resp.error}")
        return

    async def chunk_generator():
        while True:
            item = await mic_queue.get()
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
        async for response in stub.StreamTranscribe(chunk_generator()):
            text = response.partial_transcript
            status = "FINAL  " if response.is_final else "partial"
            display = (text[:72] + "…") if len(text) > 73 else text
            print(
                f"\r[{prefix}] [{status}] {display:<76} | {response.latency_ms}ms",
                end="",
                flush=True,
            )
            if response.is_final:
                print()
                break
    except Exception as e:
        print(f"\n[{prefix}] Error: {e}")


async def run_mic_stream(host="localhost", port=8001):
    print(f"Connecting to ASR server at {host}:{port}...")

    channel = grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[("grpc.max_receive_message_length", 10 * 1024 * 1024)],
    )
    stub = asr_pb2_grpc.ASRServiceStub(channel)

    audio_queue = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print(status)
        audio_queue.put(bytes(indata))

    import sounddevice as sd

    segment_idx = 0
    print("Listening... (Ctrl+C to stop)\n")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=cb,
    ):
        try:
            while True:
                mic_queue = asyncio.Queue()
                producer_task = asyncio.create_task(mic_stream_producer(audio_queue, mic_queue))
                await stream_session(stub, mic_queue, segment_idx)
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
                segment_idx += 1
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await channel.close()


if __name__ == "__main__":
    asyncio.run(run_mic_stream())
