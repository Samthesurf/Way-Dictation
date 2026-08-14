#!/usr/bin/env bash
# Hold-to-talk / tap-to-dictate helper for way-dictate.
#
# Bind this to a global hotkey (KDE System Settings > Shortcuts > Custom
# Shortcuts, or sxhkd / keyd).
#
# Usage:
#   way-dictate-hotkey                        # one phrase, inject into focused app
#   DICTATION_LANG=es way-dictate-hotkey      # force a language
#   DICTATION_PROVIDER=gpt-transcribe way-dictate-hotkey
#
# Requires GROQ_API_KEY (or OPENROUTER_API_KEY for gpt-transcribe/gemini).

set -euo pipefail

BIN="${WAY_DICTATE_BIN:-/usr/bin/way-dictate}"

if [ ! -x "$BIN" ]; then
    echo "way-dictate binary not found at $BIN" >&2
    exit 1
fi

# Optional visual cue that recording started.
if command -v notify-send >/dev/null 2>&1; then
    notify-send -t 800 "Way Dictation" "Recording..." >/dev/null 2>&1 &
fi

exec "$BIN" --once \
    --provider "${DICTATION_PROVIDER:-gpt-transcribe}" \
    --lang "${DICTATION_LANG:-}"
