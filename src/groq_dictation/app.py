"""Dictation engine: capture -> transcribe -> inject."""

from __future__ import annotations

import logging
import time

from .audio import VADConfig, record_phrase
from .client import GroqTranscriber
from .injector import Injector

log = logging.getLogger("groq_dictation")


class DictationApp:
    def __init__(
        self,
        transcriber: GroqTranscriber,
        injector: Injector,
        vad: VADConfig | None = None,
        language: str | None = None,
        auto_inject: bool = True,
    ):
        self.transcriber = transcriber
        self.injector = injector
        self.vad = vad or VADConfig()
        self.language = language
        self.auto_inject = auto_inject

    def transcribe_live(self) -> None:
        """Record one phrase, transcribe it, and (optionally) inject it."""
        log.info("Recording... (speak now)")
        wav, dur = record_phrase(self.vad)
        if not wav or len(wav) < 44 + 160:  # <10ms of audio
            log.info("No speech detected, skipping.")
            return
        log.info(f"Recorded {dur:.1f}s, transcribing...")
        t0 = time.time()
        text = self.transcriber.transcribe(wav, language=self.language)
        latency = (time.time() - t0) * 1000
        log.info(f"Transcribed in {latency:.0f}ms: {text!r}")
        if text and self.auto_inject:
            log.info("Injecting text.")
            self.injector.type_text(text)
        return text if not self.auto_inject else None

    def run_loop(self, count: int | None = None) -> None:
        """Run continuous dictation. Each phrase is a separate transcription."""
        n = 0
        while count is None or n < count:
            try:
                self.transcribe_live()
                n += 1
            except KeyboardInterrupt:
                log.info("Stopped.")
                break
            except Exception as e:  # noqa: BLE001
                log.error(f"Error in dictation loop: {e}")
                time.sleep(1)