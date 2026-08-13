#!/usr/bin/env bash
# Hold-to-talk / tap-to-dictate helper for way-dictate.
#
# Bind this to a global hotkey in KDE (System Settings > Shortcuts >
# Custom Shortcuts) or via sxhkd / keyd.
#
# Usage:
#   ./scripts/way-dictate-hotkey.sh                 # one phrase, inject into focused app
#   DICTATION_LANG=es ./scripts/...                 # force a language
#   DICTATION_PROVIDER=gpt-transcribe ./scripts/... # OpenRouter STT backend
#
# Requires GROQ_API_KEY (or OPENROUTER_API_KEY for gpt-transcribe/gemini).

set -euo pipefail
cd "$(dirname "$0")/.."

# Path to the release binary (build with: cargo build --release).
BIN="${WAY_DICTATE_BIN:-$PWD/target/release/way-dictate}"

if [ ! -x "$BIN" ]; then
    echo "way-dictate binary not found at $BIN" >&2
    echo "Build it with: cargo build --release" >&2
    exit 1
fi

# Optional visual cue that recording started.
if command -v notify-send >/dev/null 2>&1; then
    notify-send -t 800 "Way Dictation" "Recording..." >/dev/null 2>&1 &
fi

exec "$BIN" --once \
    --provider "${DICTATION_PROVIDER:-gpt-transcribe}" \
    --lang "${DICTATION_LANG:-}"
