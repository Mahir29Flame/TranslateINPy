# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai==2.8.0 pyaudio
```

This script also requires `ffmpeg` to be installed on your system. For example, on macOS:

```
brew install ffmpeg
```

Before running this script, ensure the `GEMINI_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

"""

import time
import argparse
import array
import asyncio
import os
import traceback

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

from google import genai
from google.genai import types


# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1

SEND_SAMPLE_RATE = 16_000
RECEIVE_SAMPLE_RATE = 24_000

# 100 ms at 16 kHz
CHUNK_SIZE = 1_600


# Queue markers
_INPUT_END = object()
_OUTPUT_END = object()


def adjust_volume(pcm_data: bytes, volume: float) -> bytes:
    """Reduce the volume of signed 16-bit PCM audio."""

    if volume >= 1.0:
        return pcm_data

    if volume <= 0.0:
        return b"\x00" * len(pcm_data)

    samples = array.array("h", pcm_data)

    for index in range(len(samples)):
        samples[index] = int(samples[index] * volume)

    return samples.tobytes()


async def stream_audio_url(
    url: str,
    audio_queue: asyncio.Queue,
    original_playback_queue: asyncio.Queue | None = None,
    volume: float = 0.08,
) -> None:
    """
    Decode an audio file/URL with FFmpeg and place PCM chunks in the queue.
    """

    print(f"\n[Info] Starting audio stream from: {url}", flush=True)

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        url,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SEND_SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    chunk_size_bytes = CHUNK_SIZE * 2
    bytes_per_second = SEND_SAMPLE_RATE * 2

    start_time = asyncio.get_running_loop().time()
    bytes_sent = 0

    try:
        while True:
            data = await process.stdout.read(chunk_size_bytes)

            if not data:
                break

            await audio_queue.put(data)

            if (
                original_playback_queue is not None
                and volume > 0.0
            ):
                quiet_data = adjust_volume(data, volume)
                await original_playback_queue.put(quiet_data)

            bytes_sent += len(data)

            expected_elapsed = bytes_sent / bytes_per_second
            actual_elapsed = (
                asyncio.get_running_loop().time() - start_time
            )

            sleep_time = expected_elapsed - actual_elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # FFmpeg has reached EOF. Now wait for its exit status.
        stderr_data = await process.stderr.read()
        return_code = await process.wait()

        if return_code != 0:
            error_text = stderr_data.decode(
                errors="replace"
            ).strip()

            raise RuntimeError(
                f"FFmpeg failed with exit code {return_code}"
                + (f": {error_text}" if error_text else "")
            )

    except asyncio.CancelledError:
        raise

    finally:
        if process.returncode is None:
            try:
                process.terminate()
                await process.wait()
            except Exception:
                pass

        print("\n[Info] Audio stream finished.", flush=True)


async def send_realtime(
    session,
    audio_queue: asyncio.Queue,
) -> None:
    """Send PCM audio chunks to Gemini."""

    try:
        while True:
            chunk = await audio_queue.get()

            try:
                if chunk is _INPUT_END:
                    await session.send_realtime_input(
                        audio_stream_end=True
                    )
                    return

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=(
                            f"audio/pcm;rate={SEND_SAMPLE_RATE}"
                        ),
                    )
                )

            finally:
                audio_queue.task_done()

    except asyncio.CancelledError:
        raise


async def receive_responses(
    session,
    audio_queue_output: asyncio.Queue,
    response_complete_event: asyncio.Event,
    last_activity: list,
) -> None:
    """Receive translated audio and transcription responses."""

    try:
        async for response in session.receive():
            server_content = getattr(
                response,
                "server_content",
                None,
            )

            last_activity[0] = time.monotonic()

            if server_content is None:
                continue

            model_turn = getattr(
                server_content,
                "model_turn",
                None,
            )

            if model_turn is not None:
                for part in model_turn.parts:
                    inline_data = getattr(
                        part,
                        "inline_data",
                        None,
                    )

                    if inline_data is None:
                        continue

                    audio_data = getattr(
                        inline_data,
                        "data",
                        None,
                    )

                    if isinstance(audio_data, bytes):
                        last_activity[0] = time.monotonic()
                        await audio_queue_output.put(audio_data)

            input_transcription = getattr(
                server_content,
                "input_transcription",
                None,
            )

            if (
                input_transcription is not None
                and input_transcription.text
            ):
                language_code = getattr(
                    input_transcription,
                    "language_code",
                    None,
                )

                language = (
                    f" ({language_code})"
                    if language_code
                    else ""
                )

                print(
                    f"\n[Source{language}] "
                    f"{input_transcription.text}",
                    flush=True,
                )

            output_transcription = getattr(
                server_content,
                "output_transcription",
                None,
            )

            if (
                output_transcription is not None
                and output_transcription.text
            ):
                language_code = getattr(
                    output_transcription,
                    "language_code",
                    None,
                )

                language = (
                    f" ({language_code})"
                    if language_code
                    else ""
                )

                print(
                    f"\n[Translation{language}] "
                    f"{output_transcription.text}",
                    flush=True,
                )

            turn_complete = getattr(
                server_content,
                "turn_complete",
                False,
            )

            generation_complete = getattr(
                server_content,
                "generation_complete",
                False,
            )

            if turn_complete or generation_complete:
                print(
                    "\n[Info] Gemini response complete.",
                    flush=True,
                )

                response_complete_event.set()

                # All audio parts from this response were processed
                # before this marker was inserted.
                await audio_queue_output.put(_OUTPUT_END)

                return

    except asyncio.CancelledError:
        raise

    except Exception:
        print(
            "\n[Error] Receiver crashed:",
            flush=True,
        )

        traceback.print_exc()

        response_complete_event.set()

        # Allow the player to shut down if receiving fails.
        await audio_queue_output.put(_OUTPUT_END)


async def play_audio(
    audio_queue_output: asyncio.Queue,
) -> None:
    """Play translated audio until _OUTPUT_END is received."""

    pya = pyaudio.PyAudio()
    stream = None

    try:
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while True:
            audio_data = await audio_queue_output.get()

            try:
                if audio_data is _OUTPUT_END:
                    return

                await asyncio.to_thread(
                    stream.write,
                    audio_data,
                )

            finally:
                audio_queue_output.task_done()

    except asyncio.CancelledError:
        raise

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        pya.terminate()


async def play_original_audio(
    original_playback_queue: asyncio.Queue,
) -> None:
    """Play the quiet original audio."""

    pya = pyaudio.PyAudio()
    stream = None

    try:
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            output=True,
        )

        while True:
            audio_data = await original_playback_queue.get()

            try:
                await asyncio.to_thread(
                    stream.write,
                    audio_data,
                )

            finally:
                original_playback_queue.task_done()

    except asyncio.CancelledError:
        raise

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        pya.terminate()


async def run(
    url: str,
    target_lang: str,
    api_key: str,
    original_volume: float = 0.08,
) -> None:
    """Run the live audio translation."""

    if not api_key:
        raise ValueError(
            "A Gemini API key is required."
        )

    client = genai.Client(api_key=api_key)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        translation_config=types.TranslationConfig(
            echo_target_language=True,
            target_language_code=target_lang,
        ),
        input_audio_transcription=(
            types.AudioTranscriptionConfig()
        ),
        output_audio_transcription=(
            types.AudioTranscriptionConfig()
        ),
    )

    audio_queue_input = asyncio.Queue(maxsize=50)
    audio_queue_output = asyncio.Queue()
    original_playback_queue = asyncio.Queue()

    response_complete_event = asyncio.Event()
    last_activity = [time.monotonic()]

    model = "gemini-3.5-live-translate-preview"

    print(
        f"[Info] Connecting to Gemini Live ({model})...",
        flush=True,
    )

    async with client.aio.live.connect(
        model=model,
        config=config,
    ) as session:
        print(
            "[Info] Connected successfully. Ready to stream!",
            flush=True,
        )

        async with asyncio.TaskGroup() as task_group:
            stream_task = task_group.create_task(
                stream_audio_url(
                    url=url,
                    audio_queue=audio_queue_input,
                    original_playback_queue=(
                        original_playback_queue
                    ),
                    volume=original_volume,
                )
            )

            send_task = task_group.create_task(
                send_realtime(
                    session=session,
                    audio_queue=audio_queue_input,
                )
            )

            receive_task = task_group.create_task(
                receive_responses(
                    session=session,
                    audio_queue_output=audio_queue_output,
                    response_complete_event=response_complete_event,
                    last_activity=last_activity,
                )
            )

            play_task = task_group.create_task(
                play_audio(
                    audio_queue_output=audio_queue_output,
                )
            )

            original_play_task = task_group.create_task(
                play_original_audio(
                    original_playback_queue=(
                        original_playback_queue
                    ),
                )
            )

            # Wait until FFmpeg has completely finished reading
            # and queueing the input file.
            await stream_task

            print(
                "[Info] Input finished. "
                "Sending end-of-stream marker...",
                flush=True,
            )

            # The sender is still alive here.
            await audio_queue_input.put(_INPUT_END)

            # Wait until every input chunk and the end marker
            # have been sent to Gemini.
            await audio_queue_input.join()

            # send_realtime() returns after sending
            # audio_stream_end=True.
            await send_task

            # Wait for the quiet original audio to finish.
            await original_playback_queue.join()

            original_play_task.cancel()

            await asyncio.gather(
                original_play_task,
                return_exceptions=True,
            )

            # Wait for Gemini's final response signal.
            # Gemini may not send generation_complete or turn_complete
            # for this translation stream. Wait until the response becomes
            # quiet instead.
            last_activity[0] = time.monotonic()

            quiet_period = 5.0
            maximum_wait = 45.0
            deadline = time.monotonic() + maximum_wait

            while time.monotonic() < deadline:
                quiet_for = time.monotonic() - last_activity[0]

                if quiet_for >= quiet_period:
                    break

                await asyncio.sleep(0.25)

            if time.monotonic() >= deadline:
                print(
                    "[Warning] Gemini response timeout reached.",
                    flush=True,
                )
            else:
                print(
                    "[Info] Gemini response is quiet. "
                    "Finishing playback.",
                    flush=True,
                )

            # This is now safe because Gemini has been quiet for 5 seconds.
            await audio_queue_output.put(_OUTPUT_END)

# Let the player finish every queued audio chunk.
            try:
                await asyncio.wait_for(
                    play_task,
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                print(
                    "[Warning] Final audio playback timed out.",
                    flush=True,
                )
                play_task.cancel()

            # The receiver should already have ended after
            # generation_complete or turn_complete.
            if not receive_task.done():
                receive_task.cancel()

            await asyncio.gather(
                receive_task,
                play_task,
                return_exceptions=True,
            )


def Translator(
    url: str,
    target_lang: str,
    api_key: str,
    original_volume: float = 0.08,
):
    """Public synchronous API."""

    if not url:
        print("[Error] URL cannot be empty.")
        return

    if not target_lang:
        print("[Error] Target language cannot be empty.")
        return

    if not api_key:
        print("[Error] API key cannot be empty.")
        return

    if not 0.0 <= original_volume <= 1.0:
        print(
            "[Error] Original volume must be between 0 and 1."
        )
        return

    return asyncio.run(
        run(
            url=url,
            target_lang=target_lang,
            api_key=api_key,
            original_volume=original_volume,
        )
    )


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Stream audio from a file or URL and translate it "
            "using Gemini Live."
        )
    )

    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Audio file path or URL (positional).",
    )

    parser.add_argument(
        "--url",
        default=None,
        help="Audio file path or URL.",
    )

    parser.add_argument(
        "--target",
        default="bn",
        help="Target language code (e.g., 'bn', 'es', 'fr', 'en').",
    )

    parser.add_argument(
        "--api_key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key.",
    )

    parser.add_argument(
        "--original-volume",
        type=float,
        default=0.08,
        help="Original audio volume from 0.0 to 1.0.",
    )

    args = parser.parse_args()

    audio_source = (
        args.audio
        or args.url
        or (
            "https://storage.googleapis.com/"
            "generativeai-downloads/gemini-cookbook/audio/"
            "gemini-live-translate-sample.wav"
        )
    )

    if any(sep in args.target for sep in ("/", "\\", ".")):
        print(
            f"[Error] '{args.target}' does not look like a language code (e.g., 'bn', 'es', 'en'). "
            "Did you pass the audio file to --target instead of as the audio file?",
            flush=True,
        )
        return

    try:
        asyncio.run(
            run(
                url=audio_source,
                target_lang=args.target,
                api_key=args.api_key,
                original_volume=args.original_volume,
            )
        )

    except KeyboardInterrupt:
        print(
            "\n[Info] Interrupted by user.",
            flush=True,
        )

    except Exception as error:
        print(
            f"\n[Error] {error}",
            flush=True,
        )

        traceback.print_exc()


if __name__ == "__main__":
    cli()        