"""Transcription clients.

Four backends:
- `GroqTranscriber`: Groq's hosted `whisper-large-v3-turbo` (free tier) via the
  official SDK. A purpose-built STT model, extremely fast on Groq's LPU.
- `GeminiTranscriber`: a general purpose LLM (Gemini 3.1 Flash Lite) routed
  through OpenRouter chat completions, fed the WAV as inline audio. Used to
  compare a general-purpose audio-aware LLM against dedicated STT models.
- `OpenRouterSttTranscriber`: dedicated STT models on OpenRouter's
  `/audio/transcriptions` endpoint (e.g. `openai/gpt-transcribe`). Proper STT,
  avoids the LLM hallucination/audio-drop problems.
- `GeminiLiveTranscriber`: Google's `gemini-3.5-transcribe-live` over the Live
  API WebSocket. Low-latency dedicated STT with utterance-level language
  detection. Uses GEMINI_API_KEY (https://aistudio.google.com/apikey).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import wave

import httpx
from groq import Groq

# Free-tier Groq STT model. Accurate and fast on Groq's LPU hardware.
DEFAULT_GROQ_MODEL = "whisper-large-v3-turbo"
# OpenRouter slug for Gemini 3.1 Flash Lite (GA, accepts audio input).
DEFAULT_GEMINI_MODEL = "google/gemini-3.1-flash-lite"
# OpenRouter slug for GPT Transcribe (dedicated STT, 3.3% WER, $0.0045/min).
DEFAULT_GPT_TRANSCRIBE_MODEL = "openai/gpt-transcribe"
# Google's dedicated low-latency streaming STT model (Live API).
DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.5-transcribe-live"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEMINI_LIVE_WS = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Prompt that asks Gemini to return only the transcript, verbatim, and to stay
# silent when there is no intelligible speech (LLM transcribers hallucinate
# filler over silence/noise otherwise).
_TRANSCRIBE_SYSTEM = (
    "You are a speech-to-text engine. Transcribe ONLY the actual words spoken "
    "in the audio, verbatim, with no preamble, no commentary, and no markdown. "
    "If the audio is silent, contains no intelligible speech, or is just noise, "
    "return an empty string. Never invent, repeat, or add words that were not "
    "spoken. Do not describe the audio; do not add filler."
)


def _require_key(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} not set. Add it to .env or export it in the environment."
        )
    return value


def _wav_duration_ms(wav_bytes: bytes) -> float:
    """Return WAV duration in ms by reading its header (no decode)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate()) * 1000.0


def _wav_to_mp3(wav_bytes: bytes, sample_rate: int | None = None) -> bytes:
    """Convert raw WAV to a high-quality mp3 via ffmpeg.

    LLM audio encoders (Gemini) decode compressed mp3 far more completely than
    bare raw-PCM WAV, which they tend to mis-segment and drop chunks from.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wavf:
        wavf.write(wav_bytes)
        wav_path = wavf.name
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3f:
            mp3_path = mp3f.name
        cmd = ["ffmpeg", "-y", "-i", wav_path]
        if sample_rate:
            cmd += ["-ar", str(sample_rate)]
        cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", mp3_path]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        for p in (wav_path, mp3_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _split_wav(wav_bytes: bytes, chunk_ms: int, overlap_ms: int = 500) -> list[bytes]:
    """Split a WAV into overlapping byte chunks.

    Returns a list of WAV byte buffers. Overlap prevents a word being cut at a
    chunk boundary. Used to transcribe long audio segment-by-segment so an LLM
    encoder never drops trailing parts in a single pass.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        nframes = w.getnframes()
        nch = w.getnchannels()
        sampw = w.getsampwidth()
        raw = w.readframes(nframes)
    total = nframes
    step = int(sr * chunk_ms / 1000)
    overlap = int(sr * overlap_ms / 1000)
    if total <= step:
        return [wav_bytes]
    byte_per_frame = nch * sampw
    chunks: list[bytes] = []
    start = 0
    while start < total:
        end = min(start + step, total)
        chunk_frames = raw[start * byte_per_frame : end * byte_per_frame]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nch)
            out.setsampwidth(sampw)
            out.setframerate(sr)
            out.writeframes(chunk_frames)
        chunks.append(buf.getvalue())
        if end >= total:
            break
        start = end - overlap
    return chunks


def _stitch(parts: list[str]) -> str:
    """Merge overlapping transcript chunks, deduping shared boundary words.

    Adjacent chunks overlap in audio, so the tail of one chunk appears at the
    head of the next. Build the full text incrementally: for each new part,
    drop the longest suffix of the already-built text that matches the new
    part's prefix.
    """
    if not parts:
        return ""
    built_words = parts[0].strip().split()
    for part in parts[1:]:
        p = part.strip()
        if not p:
            continue
        pw = p.split()
        max_overlap = min(len(built_words), len(pw))
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if built_words[-k:] == pw[:k]:
                overlap = k
                break
        built_words = built_words[:-overlap] if overlap else built_words
        built_words = built_words + pw
    return " ".join(built_words).strip()


class GroqTranscriber:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_GROQ_MODEL):
        self.api_key = _require_key("GROQ_API_KEY", api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.client = Groq(api_key=self.api_key)

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """Transcribe WAV bytes and return the text."""
        resp = self.client.audio.transcriptions.create(
            file=("phrase.wav", wav_bytes, "audio/wav"),
            model=self.model,
            language=language,
        )
        return resp.text.strip()

    def transcribe_file(self, path: str, language: str | None = None) -> str:
        """Transcribe a WAV file on disk."""
        with open(path, "rb") as f:
            return self.transcribe(f.read(), language=language)


class GeminiTranscriber:
    """Send raw WAV audio to a general-purpose LLM via OpenRouter.

    Gemini accepts audio as an inline multimodal part (base64, no data-URI
    prefix). This is the "LLM that can understand audio" comparison path.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        base_url: str = OPENROUTER_BASE,
        *,
        use_mp3: bool = True,
        max_chunk_ms: int = 20000,
    ):
        self.api_key = _require_key(
            "OPENROUTER_API_KEY", api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Convert WAV -> mp3 before sending (LLM encoders decode compressed
        # audio far more completely than raw PCM).
        self.use_mp3 = use_mp3
        # Transcribe audio longer than this by splitting into overlapping
        # chunks and stitching, so a single-pass encoder never drops audio.
        self.max_chunk_ms = max_chunk_ms

    def _transcribe_audio(self, audio: bytes, fmt: str, language: str | None) -> str:
        debug_dir = os.environ.get("GROQ_DICTATION_DEBUG_DIR")
        if debug_dir:
            try:
                from pathlib import Path
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                Path(debug_dir, f"req_{int(time.time() * 1000)}.{fmt}").write_bytes(audio)
            except OSError:
                pass
        b64 = base64.b64encode(audio).decode("ascii")
        content: list[dict] = [
            {"type": "text", "text": _TRANSCRIBE_SYSTEM},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"OpenRouter chat HTTP {e.response.status_code}: "
                    f"{e.response.text[:400]}"
                ) from e
            data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from e
        return text.strip()

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """Transcribe WAV bytes, converting to mp3 and segmenting if long."""
        duration = _wav_duration_ms(wav_bytes)
        if self.use_mp3:
            audio, fmt = _wav_to_mp3(wav_bytes), "mp3"
        else:
            audio, fmt = wav_bytes, "wav"

        if duration <= self.max_chunk_ms:
            return self._transcribe_audio(audio, fmt, language)

        # Long audio: split the ORIGINAL WAV into overlapping chunks, transcribe
        # each independently, then stitch. Overlap avoids cutting words.
        chunks = _split_wav(wav_bytes, self.max_chunk_ms)
        parts: list[str] = []
        for i, chunk in enumerate(chunks):
            c_audio = _wav_to_mp3(chunk) if self.use_mp3 else chunk
            c_fmt = "mp3" if self.use_mp3 else "wav"
            t = self._transcribe_audio(c_audio, c_fmt, language)
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    def transcribe_file(self, path: str, language: str | None = None) -> str:
        with open(path, "rb") as f:
            return self.transcribe(f.read(), language=language)


class OpenRouterSttTranscriber:
    """Dedicated STT models via OpenRouter's `/audio/transcriptions` endpoint.

    Unlike the Gemini chat path, this is a proper speech-to-text endpoint
    (base64 audio in, text out). Model slug e.g. `openai/gpt-transcribe`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GPT_TRANSCRIBE_MODEL,
        base_url: str = OPENROUTER_BASE,
        *,
        use_mp3: bool = True,
        max_chunk_ms: int = 20000,
    ):
        self.api_key = _require_key(
            "OPENROUTER_API_KEY", api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Convert WAV -> mp3 before sending (typically faster and decoded more
        # reliably than raw PCM by the upstream STT encoder).
        self.use_mp3 = use_mp3
        # Split audio longer than this into overlapping chunks and stitch, so a
        # single request never drops trailing segments.
        self.max_chunk_ms = max_chunk_ms

    def _transcribe_audio(self, audio: bytes, fmt: str, language: str | None) -> str:
        debug_dir = os.environ.get("GROQ_DICTATION_DEBUG_DIR")
        if debug_dir:
            try:
                from pathlib import Path
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                Path(debug_dir, f"req_{int(time.time() * 1000)}.{fmt}").write_bytes(audio)
            except OSError:
                pass
        b64 = base64.b64encode(audio).decode("ascii")
        payload: dict = {
            "model": self.model,
            "input_audio": {"data": b64, "format": fmt},
            # GPT transcription models drop/truncate segments around pauses and
            # trailing speech. A prompt forbidding omission plus low temperature
            # is the community-verified fix (OpenAI forum). See README.
            "prompt": (
                "Transcribe every spoken word exactly as spoken. Do NOT omit, "
                "summarize, clean up, or truncate anything. Output all words, "
                "including any that follow pauses or occur at the very start or "
                "end of the audio. Do not combine or drop sentences."
            ),
            "temperature": 0.2,
        }
        if language:
            payload["language"] = language
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.base_url}/audio/transcriptions",
                json=payload,
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"OpenRouter STT HTTP {e.response.status_code}: "
                    f"{e.response.text[:400]}"
                ) from e
            data = resp.json()
        try:
            text = data["text"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Unexpected OpenRouter STT response: {data}") from e
        return text.strip()

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """Transcribe WAV bytes, converting to mp3 and segmenting if long."""
        duration = _wav_duration_ms(wav_bytes)
        if self.use_mp3:
            audio, fmt = _wav_to_mp3(wav_bytes), "mp3"
        else:
            audio, fmt = wav_bytes, "wav"

        if duration <= self.max_chunk_ms:
            return self._transcribe_audio(audio, fmt, language)

        # Long audio: split into overlapping chunks, transcribe each, then
        # stitch with overlap dedup so boundary words aren't duplicated.
        chunks = _split_wav(wav_bytes, self.max_chunk_ms)
        parts: list[str] = []
        for chunk in chunks:
            c_audio = _wav_to_mp3(chunk) if self.use_mp3 else chunk
            c_fmt = "mp3" if self.use_mp3 else "wav"
            t = self._transcribe_audio(c_audio, c_fmt, language)
            if t:
                parts.append(t)
        return _stitch(parts)

    def transcribe_file(self, path: str, language: str | None = None) -> str:
        with open(path, "rb") as f:
            return self.transcribe(f.read(), language=language)


class GeminiLiveTranscriber:
    """Google `gemini-3.5-transcribe-live` over the Live API WebSocket.

    Dedicated low-latency STT. Requires GEMINI_API_KEY
    (https://aistudio.google.com/apikey).

    Protocol (per ai.google.dev/gemini-api/docs/live-api/live-transcribe):
    1. Open the BidiGenerateContent WebSocket with ?key=API_KEY.
    2. Send a setup message (responseModalities: ["TEXT"], optional
       languageCodes).
    3. Stream raw 16-bit mono 16 kHz PCM in ~100 ms chunks, paced at real
       time; the API only finalizes a turn when audio arrives at its natural
       cadence.
    4. Signal end of audio with realtimeInput.audioStreamEnd.
    5. Collect finalized `serverContent.inputTranscription.text` events.
       All server frames (including setupComplete) arrive as *binary*
       WebSocket frames, not text (verified Aug 2026).
    """

    def __init__(self, model: str = DEFAULT_GEMINI_LIVE_MODEL):
        self.api_key = _require_key(
            "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY")
        )
        self.model = model

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            if (
                w.getframerate() != 16000
                or w.getnchannels() != 1
                or w.getsampwidth() != 2
            ):
                raise RuntimeError(
                    "Gemini Live requires 16 kHz mono 16-bit PCM audio, got "
                    f"{w.getframerate()} Hz / {w.getnchannels()} ch / "
                    f"{w.getsampwidth() * 8}-bit. Re-record or convert: "
                    "ffmpeg -i in.wav -ar 16000 -ac 1 -acodec pcm_s16le out.wav"
                )
            pcm = w.readframes(w.getnframes())
        return asyncio.run(self._run(pcm, language))

    async def _run(self, pcm: bytes, language: str | None) -> str:
        import websockets

        ws_url = f"{GEMINI_LIVE_WS}?key={self.api_key}"
        # Prefer IPv4: the Live API never answers over IPv6 (verified Aug 2026:
        # the IPv6 TLS handshake completes, then the server stays silent).
        import socket
        infos = socket.getaddrinfo(
            "generativelanguage.googleapis.com", 443, socket.AF_INET
        )
        addrs: list[tuple[str, int]] = sorted(
            {info[4] for info in infos}  # type: ignore[arg-type]
        )
        if not addrs:
            addrs = [
                info[4]  # type: ignore[misc]
                for info in socket.getaddrinfo(
                    "generativelanguage.googleapis.com", 443
                )
            ]
        last_err: Exception | None = None
        sock = None
        for addr in addrs:
            try:
                sock = socket.create_connection(addr, timeout=20)
                break
            except OSError as e:
                last_err = e
        if sock is None:
            raise RuntimeError(
                f"Could not connect to Gemini Live API: {last_err}"
            )

        async with websockets.connect(
            ws_url, sock=sock, max_size=10_000_000, open_timeout=20
        ) as ws:
            setup: dict = {
                "setup": {
                    "model": f"models/{self.model}",
                    "generationConfig": {"responseModalities": ["TEXT"]},
                    "inputAudioTranscription": {
                        "languageCodes": [language] if language else []
                    },
                }
            }
            await ws.send(json.dumps(setup))

            chunk = 3200  # 100 ms of 16 kHz / 16-bit mono
            for i in range(0, len(pcm), chunk):
                seg = pcm[i : i + chunk]
                await ws.send(
                    json.dumps(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "data": base64.b64encode(seg).decode("ascii"),
                                    "mimeType": "audio/pcm;rate=16000",
                                }
                            }
                        }
                    )
                )
                await asyncio.sleep(0.10)
            await ws.send(
                json.dumps({"realtimeInput": {"audioStreamEnd": True}})
            )

            parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + 20
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    if parts:
                        break
                    continue
                data = json.loads(raw)
                if "error" in data:
                    raise RuntimeError(f"Gemini Live API error: {data['error']}")
                sc = data.get("serverContent") or {}
                it = sc.get("inputTranscription") or {}
                text = (it.get("text") or "").strip()
                if text:
                    parts.append(text)
                if sc.get("generationComplete") or sc.get("turnComplete"):
                    break
            if not parts:
                raise RuntimeError(
                    "Gemini Live returned no transcript for this audio"
                )
            return " ".join(parts)

    def transcribe_file(self, path: str, language: str | None = None) -> str:
        with open(path, "rb") as f:
            return self.transcribe(f.read(), language=language)


def build_transcriber(
    provider: str,
    model: str | None = None,
) -> GroqTranscriber | GeminiTranscriber | OpenRouterSttTranscriber | GeminiLiveTranscriber:
    """Return the transcriber for the given provider.

    Providers: 'groq', 'gemini', 'gpt-transcribe', 'gemini-live'.
    """
    provider = provider.lower()
    if provider == "groq":
        return GroqTranscriber(model=model or DEFAULT_GROQ_MODEL)
    if provider in ("gemini", "openrouter"):
        return GeminiTranscriber(model=model or DEFAULT_GEMINI_MODEL)
    if provider in ("gpt-transcribe", "gpttranscribe", "gpt"):
        return OpenRouterSttTranscriber(model=model or DEFAULT_GPT_TRANSCRIBE_MODEL)
    if provider in ("gemini-live", "gemini-live-transcribe", "glive"):
        return GeminiLiveTranscriber(model=model or DEFAULT_GEMINI_LIVE_MODEL)
    raise RuntimeError(
        f"Unknown provider: {provider!r} "
        "(expected 'groq', 'gemini', 'gemini-live', or 'gpt-transcribe')"
    )