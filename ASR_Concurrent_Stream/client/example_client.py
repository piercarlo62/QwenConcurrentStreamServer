"""
gRPC Client for ASR Concurrent Stream

Multi-segment continuous streaming with Silero VAD:
- Loads one or more audio files
- Segments audio using Silero VAD (configurable silence threshold)
- Streams each segment to a dedicated gRPC stream (new stream_id per segment)
- Prints partial transcripts in real-time as they arrive from the server
- Optional concurrent mode: all segments processed in parallel across streams

Usage examples:
  # Single file, sequential segments
  python example_client.py --audio my_audio.wav

  # Single file, custom silence gap and language
  python example_client.py --audio my_audio.wav --silence-ms 600 --language Italian

  # Multiple files, all segments in parallel
  python example_client.py --audio a.wav b.wav --concurrent

  # Faster-than-realtime (e.g. 4x) for stress testing
  python example_client.py --audio my_audio.wav --realtime 4.0
"""

import asyncio
import logging
import math
import os
import sys
import time
import uuid
from typing import List, Optional, Tuple

import grpc
import grpc.aio
import numpy as np

CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLIENT_DIR)
sys.path.insert(0, BASE_DIR)

from proto import asr_pb2
from proto import asr_pb2_grpc

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600   # 100 ms per chunk at 16 kHz
VAD_WINDOW = 512    # Silero VAD internal window size at 16 kHz


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(audio_file: str) -> np.ndarray:
    """Load audio as float32 mono at 16 kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_file, dtype="float32")
    except Exception:
        import librosa
        return librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)[0]

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples)
    return audio.astype(np.float32)


def segment_audio_with_vad(
    audio: np.ndarray,
    silence_duration_ms: int = 800,
    speech_threshold: float = 0.5,
    pad_ms: int = 150,
) -> List[np.ndarray]:
    """
    Segment float32 audio into speech chunks using Silero VAD.

    Each returned segment starts `pad_ms` before the detected speech onset
    (for model context) and ends at the detected speech offset.  If no
    segments are found (all silence, or threshold too strict), the entire
    audio is returned as a single segment.
    """
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
        threshold=speech_threshold,
        min_silence_duration_ms=silence_duration_ms,
    )

    pad_samples = int(pad_ms * SAMPLE_RATE / 1000)
    segments: List[Tuple[int, int]] = []
    current_start: Optional[int] = None

    for i in range(0, len(audio), VAD_WINDOW):
        window = audio[i : i + VAD_WINDOW].astype(np.float32)
        if len(window) < VAD_WINDOW:
            window = np.pad(window, (0, VAD_WINDOW - len(window)))

        ev = vad_iter(torch.from_numpy(window), return_seconds=False)
        if ev is not None:
            if "start" in ev:
                current_start = max(0, i - pad_samples)
            if "end" in ev and current_start is not None:
                segments.append((current_start, min(i + VAD_WINDOW, len(audio))))
                current_start = None

    # Audio ends while still in speech
    if current_start is not None:
        segments.append((current_start, len(audio)))

    if not segments:
        logger.warning("VAD found no speech; treating entire audio as one segment")
        return [audio]

    return [audio[s:e] for s, e in segments]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check(
    host: str = "localhost",
    port: int = 50051,
) -> bool:
    """Call the server HealthCheck RPC and return True if serving."""
    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    try:
        stub = asr_pb2_grpc.ASRServiceStub(channel)
        response = await stub.HealthCheck(asr_pb2.HealthCheckRequest())
        print(f"Health check: serving={response.serving}, version={response.version}")
        return response.serving
    except grpc.aio.AioRpcError as e:
        print(f"Health check failed: {e.code()} – {e.details()}")
        return False
    finally:
        await channel.close()


# ---------------------------------------------------------------------------
# Per-segment streaming
# ---------------------------------------------------------------------------

async def stream_segment(
    stub: asr_pb2_grpc.ASRServiceStub,
    audio_segment: np.ndarray,
    segment_idx: int,
    language: str = "Italian",
    realtime_factor: float = 1.0,
    label: str = "",
) -> Tuple[int, str, float]:
    """
    Stream a single audio segment and return (segment_idx, final_transcript, total_ms).

    A new stream_id is created for every call.  Partial transcripts are
    printed to stdout as they arrive.  The function returns once the server
    sends is_final=True (triggered when the last chunk's is_final flag
    flows through the coordinator and calls finish_streaming_transcribe).
    """
    stream_id = str(uuid.uuid4())
    prefix = label or f"seg {segment_idx}"
    sleep_per_chunk = (
        CHUNK_SIZE / SAMPLE_RATE / max(realtime_factor, 1e-3)
        if realtime_factor > 0
        else 0.0
    )

    # Initialise the stream (sets language and chunking config on the server)
    config = asr_pb2.StreamConfig(
        chunk_size_sec=0.5, unfixed_chunk_num=2, unfixed_token_num=5
    )
    start_resp = await stub.StartStream(
        asr_pb2.StreamStartRequest(
            stream_id=stream_id, language=language, config=config
        )
    )
    if not start_resp.success:
        logger.error(f"[{prefix}] StartStream failed: {start_resp.error}")
        return segment_idx, "", 0.0

    audio_int16 = (audio_segment * 32767).astype(np.int16)
    num_chunks = math.ceil(len(audio_int16) / CHUNK_SIZE)
    t_start = time.perf_counter()
    final_text = ""

    async def chunk_generator():
        for i in range(num_chunks):
            s = i * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, len(audio_int16))
            is_final = i == num_chunks - 1
            yield asr_pb2.AudioChunkRequest(
                stream_id=stream_id,
                chunk_id=i,
                audio_data=audio_int16[s:e].tobytes(),
                sample_rate=SAMPLE_RATE,
                is_final=is_final,
            )
            if is_final:
                return
            if sleep_per_chunk > 0:
                await asyncio.sleep(sleep_per_chunk)

    try:
        async for response in stub.StreamTranscribe(chunk_generator()):
            text = response.partial_transcript
            status = "FINAL  " if response.is_final else "partial"
            display = (text[:72] + "\u2026") if len(text) > 73 else text
            print(
                f"\r[{prefix}] [{status}] {display:<76} | {response.latency_ms}ms",
                end="",
                flush=True,
            )
            if response.is_final:
                final_text = text
                print()   # newline after final
                break
    except grpc.aio.AioRpcError as e:
        logger.error(f"[{prefix}] gRPC error: {e.code()} \u2013 {e.details()}")
        print()
    except Exception as e:
        logger.error(f"[{prefix}] Error: {e}", exc_info=True)
        print()

    total_ms = (time.perf_counter() - t_start) * 1000
    return segment_idx, final_text, total_ms


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_continuous_streaming(
    audio_files: List[str],
    host: str = "localhost",
    port: int = 50051,
    language: str = "Italian",
    silence_duration_ms: int = 800,
    concurrent: bool = False,
    realtime_factor: float = 1.0,
) -> None:
    """
    Load one or more audio files, segment them with Silero VAD, and stream
    each segment to a dedicated gRPC stream on the ASR server.

    concurrent=False  — segments are processed one after another (default)
    concurrent=True   — all segments fire simultaneously (multi-stream load test)
    """
    print("\n" + "=" * 65)
    print("ASR Concurrent Stream \u2014 Continuous Streaming with Silero VAD")
    print("=" * 65)
    print(f"  Files:       {len(audio_files)}")
    print(f"  Language:    {language}")
    print(f"  Silence gap: {silence_duration_ms} ms")
    print(f"  Mode:        {'Concurrent (parallel streams)' if concurrent else 'Sequential'}")
    print(f"  Realtime:    {realtime_factor}\u00d7")
    print("=" * 65 + "\n")

    channel = grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[("grpc.max_receive_message_length", 10 * 1024 * 1024)],
    )
    stub = asr_pb2_grpc.ASRServiceStub(channel)
    logger.info(f"Connected to {host}:{port}")

    try:
        # Build task list: (label, segment_audio, global_idx)
        tasks: List[Tuple[str, np.ndarray, int]] = []
        global_idx = 0

        for f_idx, audio_file in enumerate(audio_files):
            print(f"\u25ba Loading {os.path.basename(audio_file)}")
            audio = load_audio(audio_file)
            duration_s = len(audio) / SAMPLE_RATE
            print(f"  Duration : {duration_s:.2f}s")

            print(f"  Segmenting with Silero VAD (silence={silence_duration_ms}ms)...")
            segments = segment_audio_with_vad(
                audio, silence_duration_ms=silence_duration_ms
            )
            print(f"  Segments : {len(segments)}\n")

            for s_idx, seg in enumerate(segments):
                seg_dur = len(seg) / SAMPLE_RATE
                lbl = (
                    f"f{f_idx}/s{s_idx}" if len(audio_files) > 1 else f"seg {s_idx}"
                )
                print(f"  [{lbl}] {seg_dur:.2f}s")
                tasks.append((lbl, seg, global_idx))
                global_idx += 1

        if not tasks:
            print("No segments to process.")
            return

        print(f"\nStreaming {len(tasks)} segment(s)...\n")

        if concurrent:
            coros = [
                stream_segment(stub, seg, idx, language, realtime_factor, lbl)
                for lbl, seg, idx in tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
        else:
            results = []
            for lbl, seg, idx in tasks:
                r = await stream_segment(stub, seg, idx, language, realtime_factor, lbl)
                results.append(r)

        print("\n" + "=" * 65)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 65)
        for res in results:
            if isinstance(res, Exception):
                print(f"  ERROR: {res}")
            else:
                seg_idx, text, lat = res
                print(f"  [{seg_idx:>2}] {(text or '(empty)'):<60}  ({lat:.0f} ms)")
        print("=" * 65)

    finally:
        await channel.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ASR Concurrent Stream Client \u2014 multi-segment Silero VAD streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=["/mnt/c/Users/pierc/Qwen3_asr/audios/Audio 1-2.wav"],
        metavar="FILE",
        help="Path(s) to audio file(s) to transcribe",
    )
    parser.add_argument("--host", default="localhost", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument(
        "--language",
        default="Italian",
        help="Transcription language (e.g. Italian, English)",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=400,
        help="Minimum silence (ms) between speech segments for VAD segmentation",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Process all segments in parallel (multi-stream concurrency test)",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=2.0,
        help=(
            "Realtime streaming factor: 1.0 = real-time, "
            "2.0 = 2x faster, 0.0 = no delay (stress test)"
        ),
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run a health check against the server and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.health:
        healthy = asyncio.run(health_check(host=args.host, port=args.port))
        sys.exit(0 if healthy else 1)

    asyncio.run(
        run_continuous_streaming(
            audio_files=args.audio,
            host=args.host,
            port=args.port,
            language=args.language,
            silence_duration_ms=args.silence_ms,
            concurrent=args.concurrent,
            realtime_factor=args.realtime,
        )
    )


if __name__ == "__main__":
    main()
