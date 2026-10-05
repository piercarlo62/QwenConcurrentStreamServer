import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_asr_client import RealtimeASRClient

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice")
    sys.exit(1)


async def main():
    client = RealtimeASRClient(host="localhost", port=8001, language="Italian")

    def on_partial(text):
        print(f"\r[partial] {text}", end="", flush=True)

    def on_final(text):
        print(f"\n[final]   {text}\n")

    client.on_partial = on_partial
    client.on_final = on_final

    await client.start()
    print("Listening... (Ctrl+C to stop)\n")

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
