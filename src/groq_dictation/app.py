"""Dictation engine: capture -> transcribe -> inject."""

from __future__ import annotations

import logging
import queue
import threading
import time

from .audio import VADConfig, record_phrase
from .client import GroqTranscriber
from .injector import Injector

log = logging.getLogger("groq_dictation")

MIN_WAV_BYTES = 44 + 160  # below this there is no meaningful speech


class DictationPipeline:
    """Continuous capture + transcription pipeline.

    A recorder thread captures phrases back to back into a bounded queue
    while a consumer thread transcribes and injects them in order. Speech
    spoken during a cloud round-trip keeps being recorded, so audio is not
    lost while the previous phrase is being transcribed.

    Callbacks (all optional, invoked from the worker threads):
      on_status(status): "listening" | "transcribing" | "paused"
      on_phrase(text, latency_ms): a phrase was transcribed and injected
      on_error(message, fatal): mic errors are fatal, API errors are not
    """

    QUEUE_SIZE = 16

    def __init__(
        self,
        transcriber: GroqTranscriber,
        injector: Injector,
        vad: VADConfig | None = None,
        language: str | None = None,
        on_status=None,
        on_phrase=None,
        on_error=None,
    ):
        self.transcriber = transcriber
        self.injector = injector
        self.vad = vad or VADConfig()
        self.language = language
        self.on_status = on_status or (lambda status: None)
        self.on_phrase = on_phrase or (lambda text, latency: None)
        self.on_error = on_error or (lambda message, fatal: None)

        self._pause = threading.Event()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._phrases: queue.Queue[bytes] = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._consumer: threading.Thread | None = None
        self._recorder: threading.Thread | None = None

    # -- control (callable from any thread; pause/resume from the UI) --

    def start(self) -> None:
        """Spawn the recorder and consumer threads."""
        if self._recorder is not None and self._recorder.is_alive():
            return
        self._stop.clear()
        self._pause.clear()
        self._cancel.clear()
        self._consumer = threading.Thread(
            target=self._consume_loop, name="dictation-consumer", daemon=True
        )
        self._recorder = threading.Thread(
            target=self._record_loop, name="dictation-recorder", daemon=True
        )
        self._consumer.start()
        self._recorder.start()

    def pause(self) -> None:
        """Stop capturing and abort an in-flight recording.

        Phrases already queued are dropped by the consumer.
        """
        self._pause.set()
        self._cancel.set()  # abort an in-flight recording promptly

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        """Stop everything and wait briefly for the threads to wind down."""
        self._stop.set()
        self._cancel.set()
        for thread in (self._recorder, self._consumer):
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the recorder thread exits; returns True if it did."""
        if self._recorder is None:
            return True
        self._recorder.join(timeout)
        return not self._recorder.is_alive()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    # -- worker threads --

    def _record_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self._pause.is_set():
                    self.on_status("paused")
                    while self._pause.is_set() and not self._stop.is_set():
                        time.sleep(0.05)
                    continue
                self._cancel.clear()
                self.on_status("listening")
                try:
                    wav, _dur = record_phrase(self.vad, cancel_event=self._cancel)
                except Exception as e:  # noqa: BLE001 - mic errors are fatal
                    self.on_error(f"Microphone error: {e}", True)
                    break
                if self._cancel.is_set() or self._stop.is_set():
                    continue  # paused or stopped mid-recording
                if not wav or len(wav) < MIN_WAV_BYTES:
                    continue  # no speech detected, keep listening
                try:
                    self._phrases.put_nowait(wav)
                except queue.Full:
                    log.warning("transcription queue full, dropping phrase")
        finally:
            # any recorder exit means no more injections may happen; this
            # also wakes the consumer so it can wind down
            self._stop.set()

    def _consume_loop(self) -> None:
        while True:
            try:
                wav = self._phrases.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if self._pause.is_set():
                continue  # dropped while paused
            self.on_status("transcribing")
            t0 = time.time()
            try:
                text = self.transcriber.transcribe(wav, language=self.language)
            except Exception as e:  # noqa: BLE001 - retry with the next phrase
                self.on_error(f"Transcription error: {e}", False)
                log.error("Transcription error: %s", e)
                self.on_status("listening")
                continue
            latency = (time.time() - t0) * 1000
            log.info("Transcribed in %.0fms: %r", latency, text)
            if text and not self._pause.is_set() and not self._stop.is_set():
                try:
                    self.injector.type_text(text)
                except Exception as e:  # noqa: BLE001
                    self.on_error(f"Injection error: {e}", False)
                else:
                    self.on_phrase(text, latency)
            self.on_status("listening")


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
        if not wav or len(wav) < MIN_WAV_BYTES:
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
        """Run continuous pipelined dictation.

        The recorder keeps capturing while phrases are being transcribed,
        so speech is not lost during cloud round-trips.
        """
        if count is not None and count < 1:
            return
        remaining = count
        done = threading.Event()
        fatal = threading.Event()

        def on_phrase(text: str, latency: float) -> None:
            nonlocal remaining
            log.info("Phrase injected (%.0fms): %r", latency, text)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    done.set()

        def on_error(message: str, fatal_flag: bool) -> None:
            log.error(message)
            if fatal_flag:
                fatal.set()

        pipeline = DictationPipeline(
            transcriber=self.transcriber,
            injector=self.injector,
            vad=self.vad,
            language=self.language,
            on_phrase=on_phrase,
            on_error=on_error,
        )
        pipeline.start()
        try:
            while not done.is_set() and not fatal.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("Stopped.")
        finally:
            pipeline.stop()
