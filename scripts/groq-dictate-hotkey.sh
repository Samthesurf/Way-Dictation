#!/usr/bin/env bash
# Hold-to-talk / tap-to-dictate helper for groq-dictate.
#
# This is a thin wrapper. For a real global hotkey you bind this to a key
# combo in KDE (System Settings > Shortcuts > Custom Shortcuts) or via a tool
# like sxhkd / keyd. See README section "Binding a global hotkey".
#
# Usage:
#   ./scripts/groq-dictate-hotkey.sh          # one phrase, inject into focused app
#   DICTATION_LANG=es ./scripts/...           # force a language
#   DICTATION_PROVIDER=gpt-transcribe ./scripts/...  # OpenRouter STT backend
#
# Requires GROQ_API_KEY (or OPENROUTER_API_KEY for gpt-transcribe/gemini).

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate the venv non-interactively.
source .venv/bin/activate

# Optionally show a brief visual cue that recording started.
if command -v notify-send >/dev/null 2>&1; then
    notify-send -t 800 "Groq Dictation" "Recording..." >/dev/null 2>&1 &
fi

exec python -m groq_dictation.cli --once \
    --provider "${DICTATION_PROVIDER:-gpt-transcribe}" \
    --lang "${DICTATION_LANG:-}"