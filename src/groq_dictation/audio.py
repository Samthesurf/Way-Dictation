"""Microphone capture with voice activity detection (VAD).

Records audio from the default input device, splits it into phrases using
an energy-based VAD, and yields each phrase as a 16 kHz mono WAV file ready
for the Groq API.

The VAD here is a simple energy threshold to keep the scaffold dependency-free.
For better accuracy, swap in Silero VAD (torch) later, see README.
"""

from __future__ import annotations

import io
import struct
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Groq downsamples to 16k mono anyway; capture native is fine
CHANNELS = 1
DTYPE = np.int16


@dataclass
class VADConfig:
    """Energy-based voice activity detection parameters."""

    sample_rate: int = SAMPLE_RATE
    enabled: bool = True
    # Frame length in ms used for energy analysis.
    frame_ms: int = 20
    # Silence threshold (RMS). Lower = more sensitive to quiet speech.
    threshold: float = 0.010
    # Frames of sustained silence before the phrase is considered finished.
    silence_frames_trigger: int = 20  # ~0.5s at 20ms frames
    # Minimum silence (ms) required to end a phrase.
    min_phrase_silence_ms: float = 450.0
    # Maximum phrase length (ms) before we cut it off regardless.
    max_phrase_ms: float = 30000.0
    # Optional fixed recording length when VAD is disabled (ms).
    fixed_record_ms: float = 5000.0
    # Trim leading/trailing silence before sending to the API. Whisper tolerates
    # padding, but LLM transcribers (Gemini) hallucinate text over silence.
    trim_silence: bool = True
    # Frames of speech-like padding kept around detected speech after trimming.
    trim_pad_ms: float = 150.0


class EnergyVAD:
    """Simple RMS-energy based VAD over 20ms frames."""

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.frame_ms = cfg.frame_ms
        self.frame_samples = int(cfg.sample_rate * self.frame_ms / 1000)
        self._silence_runs = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0
        return rms > self.cfg.threshold

    def process_frame(self, frame: np.ndarray) -> str:
        """Returns 'speech', 'silence', or 'end'."""
        if self.is_speech(frame):
            self._silence_runs = 0
            return "speech"
        self._silence_runs += 1
        if self._silence_runs >= self.cfg.silence_frames_trigger:
            return "end"
        return "silence"


def _frames(data: bytes, frame_samples: int):
    """Yield 20ms frames from a raw int16 byte buffer."""
    n = len(data) // 2
    arr = np.frombuffer(data, dtype=DTYPE)[: n - (n % frame_samples)]
    for i in range(0, len(arr), frame_samples):
        yield arr[i : i + frame_samples]


def _wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Wrap raw int16 PCM into a WAV byte buffer (16-bit mono)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype(DTYPE).tobytes())
    return buf.getvalue()


def trim_silence(
    pcm: np.ndarray,
    sample_rate: int,
    threshold: float = 0.010,
    frame_ms: int = 20,
    pad_ms: float = 150.0,
) -> np.ndarray:
    """Trim leading/trailing silence from int16 PCM, keeping a small pad.

    Returns the central span bracketing detected speech. If nothing is above
    the threshold, returns an empty array (the caller treats <10ms as 'no
    speech' and skips). This removes the silence tails that LLM transcribers
    tend to hallucinate over.
    """
    if len(pcm) == 0:
        return pcm
    frame = int(sample_rate * frame_ms / 1000)
    n = len(pcm) // frame
    if n == 0:
        return pcm
    arr = pcm[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean((arr.astype(np.float32) / 32768.0) ** 2, axis=1))
    speech = rms > threshold
    if not speech.any():
        return np.zeros(0, dtype=DTYPE)
    first = int(np.nonzero(speech)[0][0])
    last = int(np.nonzero(speech)[0][-1])
    pad_frames = int(pad_ms / 1000.0 * sample_rate / frame)
    first = max(0, first - pad_frames)
    last = min(n - 1, last + pad_frames)
    return pcm[first * frame : (last + 1) * frame]


def record_phrase(
    cfg: VADConfig,
    stream_callback=None,
    cancel_event: threading.Event | None = None,
) -> tuple[bytes, float]:
    """Records one phrase.

    Press/release or hold-to-talk semantics are handled by the caller. This
    function blocks until a phrase boundary is detected (or max length).

    If `cancel_event` is set (from any thread), the stream is aborted and the
    function returns promptly; callers should re-check the event to discard
    the partial audio.

    Returns (wav_bytes, duration_seconds).
    """
    use_vad = cfg.enabled
    raw = bytearray()
    frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
    vad = EnergyVAD(cfg)

    max_samples = int(cfg.sample_rate * cfg.max_phrase_ms / 1000)
    fixed_samples = int(cfg.sample_rate * cfg.fixed_record_ms / 1000)
    target = max_samples if use_vad else fixed_samples

    start = time.time()

    def cb(indata, frames, t, status):
        raw.extend(indata[:, 0].astype(DTYPE).tobytes())
        if stream_callback:
            stream_callback(indata[:, 0])

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=max(1024, frame_samples),
        callback=cb,
    ) as stream:
        if not use_vad:
            # Fixed-length recording: sleep until target samples reached.
            while len(raw) // 2 < target:
                if cancel_event is not None and cancel_event.is_set():
                    stream.abort()
                    break
                time.sleep(0.05)
        else:
            # VAD phrase detection loop.
            speaking = False
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    stream.abort()
                    break
                time.sleep(0.02)
                n_frames = len(raw) // 2 // frame_samples
                if n_frames == 0:
                    continue
                # Process the newest complete frame.
                frame = np.frombuffer(raw[: n_frames * frame_samples * 2], dtype=DTYPE)[
                    -frame_samples:
                ]
                state = vad.process_frame(frame)
                if state == "speech":
                    speaking = True
                if state == "end" and speaking:
                    break
                if len(raw) // 2 >= max_samples:
                    break

    duration = time.time() - start
    pcm = np.frombuffer(bytes(raw), dtype=DTYPE)
    if cfg.trim_silence:
        pcm = trim_silence(
            pcm,
            cfg.sample_rate,
            threshold=cfg.threshold,
            frame_ms=cfg.frame_ms,
            pad_ms=cfg.trim_pad_ms,
        )
    return _wav_bytes(pcm, cfg.sample_rate), duration