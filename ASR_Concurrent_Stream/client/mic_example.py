import asyncio
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_asr_client import RealtimeASRClient

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice")
    sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(description="ASR Microphone Streaming Client")
    parser.add_argument("--host", default="localhost", help="gRPC server host")
    parser.add_argument("--port", type=int, default=8001, help="gRPC server port")
    parser.add_argument("--language", default="Italian", help="Transcription language")
    parser.add_argument("--silence-ms", type=int, default=200, help="VAD silence threshold (ms)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD speech threshold (0.0-1.0)")
    parser.add_argument("--pad-ms", type=int, default=150, help="Audio padding before speech start (ms)")
    args = parser.parse_args()

    client = RealtimeASRClient(
        host=args.host,
        port=args.port,
        language=args.language,
        silence_duration_ms=args.silence_ms,
        vad_threshold=args.vad_threshold,
        pad_ms=args.pad_ms,
    )

    def on_partial(text):
        print(f"\r[partial] {text}", end="", flush=True)

    def on_final(text):
        print(f"\n[final]   {text}\n")

    client.on_partial = on_partial
    client.on_final = on_final

    await client.start()
    print("Listening... (Ctrl+C to stop)")
    print(f"  VAD: silence={args.silence_ms}ms, threshold={args.vad_threshold}, pad={args.pad_ms}ms\n")

    def cb(indata, frames, t, status):
        if status:
            print(status)
        client.feed_audio(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=1600,
        callback=cb,
    ):
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            print("\nStopping...")

    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
