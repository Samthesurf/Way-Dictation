"""CLI entry point for groq-dictate."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from .app import DictationApp
from .audio import VADConfig
from .client import build_transcriber
from .injector import Injector


def _load_api_key() -> str:
    load_dotenv()  # reads .env in CWD
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.stderr.write(
            "GROQ_API_KEY not found.\n"
            "Create a free key at https://console.groq.com/keys, then either:\n"
            "  echo 'GROQ_API_KEY=...' > .env\n"
            "  # or export GROQ_API_KEY=...\n"
        )
        sys.exit(1)
    return key


def _load_openrouter_key() -> str:
    load_dotenv()  # reads .env in CWD
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.stderr.write(
            "OPENROUTER_API_KEY not found.\n"
            "Get a key at https://openrouter.ai/keys, then either:\n"
            "  echo 'OPENROUTER_API_KEY=...' > .env\n"
            "  # or export OPENROUTER_API_KEY=...\n"
        )
        sys.exit(1)
    return key


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="groq-dictate", description="Groq dictation app")
    p.add_argument("--once", action="store_true", help="Transcribe a single phrase and exit")
    p.add_argument("--count", type=int, default=None, help="Transcribe N phrases then exit")
    p.add_argument("--file", type=str, help="Transcribe an existing WAV file instead of mic")
    p.add_argument("--model", default=None, help="Overrides the model ID for the chosen provider")
    p.add_argument("--provider", default="gpt-transcribe",
                   help="Transcription backend: gpt-transcribe | groq | gemini")
    p.add_argument("--lang", default=None, help="Optional ISO-639-1 language code (e.g. en)")
    p.add_argument("--method", default="auto", help="Injection: auto|wtype|ydotool|xdotool")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("groq_dictation")

    # Normalize provider aliases and load the key it needs.
    provider = args.provider.lower()
    if provider in ("openrouter", "gemini"):
        provider = "gemini"
    elif provider in ("gpt-transcribe", "gpttranscribe", "gpt"):
        provider = "gpt-transcribe"
    if provider == "groq":
        _load_api_key()
    else:
        _load_openrouter_key()
    transcriber = build_transcriber(provider, model=args.model)

    if args.file:
        text = transcriber.transcribe_file(args.file, language=args.lang)
        print(text)
        return

    injector = Injector(method=args.method)
    app = DictationApp(
        transcriber=transcriber,
        injector=injector,
        vad=VADConfig(),
        language=args.lang,
    )

    if args.once:
        app.transcribe_live()
    elif args.count:
        app.run_loop(count=args.count)
    else:
        log.info("Continuous dictation. Ctrl+C to stop. Type into your target app.")
        app.run_loop()


if __name__ == "__main__":
    main()