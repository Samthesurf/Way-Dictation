# groq-dictation

Native Linux speech-to-text dictation app. Speaks into the mic, transcribes with
Groq's hosted (free) Whisper, and types the result into whatever app currently
has keyboard focus.

Works on Wayland (KDE Plasma) and X11.

---

## Why this stack (research summary, verified Feb/Mar 2026)

This project's backend choice was made after actually testing on this machine
(i5-8350U, 8 threads, 23 GiB RAM, no dGPU). The conclusion: **use Groq's hosted
free Whisper, not local models.**

### The local-model reality check (why we did NOT go local)

Measured on this exact hardware with the latest whisper.cpp built from source:

| Model | Time for 3.78s clip | RTF | Usable for dictation? |
|---|---|---|---|
| large-v3-turbo q5 (whisper.cpp) | ~199s (under load) | ~52x slower than real-time | No |
| tiny (whisper.cpp) | ~5.5s | ~1.5x slower | Marginal, low accuracy |

The i5-8350U is a 1.7 GHz 4-core low-power laptop chip. It cannot run
large-v3-turbo fast enough for real-time dictation. Groq runs the same model on
LPU hardware at ~216x real-time speed, with far better accuracy than any local
small model.

### Groq free tier (confirmed from console.groq.com/docs/rate-limits)

Whisper models on the **free plan** (no card required, separate from any
OpenRouter credits):

| Limit | value |
|---|---|
| Requests / minute | 20 |
| Requests / day | 2,000 |
| Audio seconds / hour | 7,200 (2 hrs of audio) |
| Audio seconds / day | 28,800 (8 hrs of audio) |
| Max file size | 25 MB |
| Cost | $0 |

For dictation (short phrases), you will use a tiny fraction of these. Free tier
is more than enough.

### OpenRouter alternatives (researched)

- `openai/gpt-transcribe` (the exact model asked about): $0.0045/min, very
  accurate, but **not free** and no published latency figures. Brand new (Aug
  2026). Not ideal for low-latency live dictation.
- `openai/gpt-4o-transcribe`: $2.50/$10 per 1M tokens, also paid.
- Verdict: Groq free tier wins for a $0 budget + low latency.

### Python version locked in: 3.12

Verified working on this machine: faster-whisper, whisper-cpp-python, latest
whisper.cpp source build, and the Groq SDK all run on Python 3.12.13.

**Gotcha found:** the PyPI `whisper-cpp-python` 0.2.0 binding is stale and
crashes (GGML assertion) on the large-v3-turbo model; you must build whisper.cpp
from source to use turbo. Moot here since we use Groq.

---

## Architecture

```
mic (sounddevice, ALSA)
   -> record_phrase()  (energy-based VAD, splits speech into phrases)
   -> WAV bytes (16 kHz mono)
   -> GroqTranscriber (whisper-large-v3-turbo, free API)
   -> text
   -> Injector (wtype / ydotool / xdotool) -> focused app
```

Modules (`src/groq_dictation/`):

| File | Role |
|---|---|
| `audio.py` | Mic capture + energy VAD, yields WAV for one phrase |
| `client.py` | Groq Whisper API wrapper |
| `injector.py` | Auto-detects text injection method for Wayland/X11 |
| `app.py` | `DictationApp` orchestration (capture->transcribe->inject) |
| `cli.py` | `groq-dictate` CLI entry point |
| `gui.py` | `groq-dictation-gui` floating desktop widget (PySide6) |

---

## Setup

```bash
cd ~/Documents/python_stuff/groq-dictation

# 1. Get a free API key (no card): https://console.groq.com/keys
cp .env.example .env
# edit .env, set GROQ_API_KEY=your_key

# 2. Create the venv (Python 3.12) and install
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .

# 3. (Wayland/KDE) injection tools
sudo pacman -S --needed wtype ydotool   # wtype works on KDE via virtual-keyboard protocol
```

### ydotool daemon (only if wtype is not usable)

ydotool needs the `ydotoold` daemon running with uinput access:

```bash
sudo systemctl enable --now uinput.service ydotool.service
```
(If using a distro without packaged services, run `ydotoold` as root.)

---

## Usage

```bash
# One phrase, then inject into the focused app
groq-dictate --once

# Continuous loop (each phrase = one transcription), Ctrl+C to stop
groq-dictate

# Transcribe an existing WAV file instead of the mic
groq-dictate --file path/to/audio.wav

# Force a language
groq-dictate --once --lang en

# Transcribe via OpenRouter STT models (dedicated speech-to-text, not an LLM)
# instead of the default Groq Whisper backend. Requires OPENROUTER_API_KEY.
groq-dictate --once --provider gpt-transcribe

# Pick any model ID for either backend
groq-dictate --once --provider gpt-transcribe --model openai/gpt-transcribe
groq-dictate --once --provider gemini --model google/gemini-3.1-flash-lite
groq-dictate --once --provider groq --model whisper-large-v3-turbo

# Choose injection method explicitly
groq-dictate --once --method wtype   # or ydotool | xdotool
```

### Providers

| Flag | Backend | Model (default) | Notes |
|---|---|---|---|
| `--provider gpt-transcribe` (**default**) | OpenRouter STT | `openai/gpt-transcribe` | Best on this user's voice; accurate + complete. Paid (~$0.0045/min) |
| `--provider groq` | Groq Whisper | `whisper-large-v3-turbo` | Free, fast, Whisper-class; but **hallucinates "thank you" on background noise/keystrokes** |
| `--provider gemini` (alias: `openrouter`) | OpenRouter Gemini | `google/gemini-3.1-flash-lite` | General LLM that understands audio; hallucinates over silence, can drop audio |

Real-world A/B on this machine (same passage read aloud): `gpt-transcribe` gave a
coherent, near-complete transcript, while `groq` repeated "thank you" / "thank
you" many times over keyboard/background noise. gpt-transcribe is the default
for everyday dictation; pass `--provider groq` for the free tier.

`gpt-transcribe` and `gemini` use `OPENROUTER_API_KEY`; `groq` uses
`GROQ_API_KEY`. Both are read from `.env`. Measured on this machine for a 2 s
test clip: Groq ~1.7 s, gpt-transcribe ~2.3 s, Gemini ~3.3 s.

### Improving completeness (audio loss)

**Known gpt-transcribe behavior:** GPT-family transcription models
(`openai/gpt-transcribe`, like the older `gpt-4o-transcribe`) are documented by
users to **drop/truncate segments around pauses and at the end of audio**,
even though what they do transcribe is very accurate. This is a known model
quirk, NOT a mic or code bug. Whisper-class models do not have it.

The community-verified fix, applied here by default, is to send a `prompt` that
forbids omission plus a low `temperature` (0.2):

```json
{
  "model": "openai/gpt-transcribe",
  "input_audio": { "data": "...", "format": "mp3" },
  "prompt": "Transcribe every spoken word exactly as spoken. Do NOT omit, summarize, clean up, or truncate anything. Output all words, including any that follow pauses or occur at the very start or end of the audio. Do not combine or drop sentences.",
  "temperature": 0.2
}
```

Additional code-level mitigations (on by default):

- **mp3 conversion**: audio is converted to high-quality mp3 before sending.
  Compressed audio is decoded more completely and faster than raw PCM WAV.
- **Segmented transcription** (`_split_wav` + `_stitch`): audio longer than
  `max_chunk_ms` (default 20 s) is split into overlapping chunks, each
  transcribed independently, then stitched with overlap dedup. Note: keep
  chunks reasonably large (>=15 s). Very short chunks (e.g. 8 s) give
  gpt-transcribe too little speech and it hallucinates words in near-empty
  chunks. OpenRouter's gpt-transcribe does NOT support `verbose_json`/word
  timestamps (returns 400), so finer-grained gap detection isn't available.

If a backend still drops audio, the reliable choice is a different backend
(`groq` is free and fast; Whisper-class models are truncation-resistant).

## Desktop GUI (`groq-dictation-gui`)

A small floating dictation widget (PySide6): frameless rounded card, one big
play/pause button, an X that quits, and a gear for settings. It stays on top
and does not steal keyboard focus, so you can press play, click into your
editor/chat, and speak; each phrase is typed into the focused window.

```bash
# the gui entry point is installed with the package (PySide6 dependency)
uv pip install --python .venv/bin/python -e .
groq-dictation-gui
```

| Control | Action |
|---|---|
| Play button | Starts the dictation loop (record -> transcribe -> inject) |
| Pause (same button) | Stops listening between phrases; aborts the in-progress phrase |
| X | Stops everything and quits |
| Gear | Provider (gpt-transcribe / groq / gemini), language, model override, always-on-top |

- Same `.env` keys as the CLI; provider and language apply on the next play.
- Settings persist to `~/.config/groq-dictation/config.json`; window position
  is remembered.
- While listening, pulse rings animate around the button; the last injected
  phrase is previewed under the status.
- Errors surface in the widget: missing API key, mic failure, transcription
  or injection problems (non-fatal errors keep the loop running).

### Desktop launcher (optional)

```bash
cp scripts/groq-dictation-gui.desktop ~/.local/share/applications/
```

The widget then appears in the KDE launcher as "Groq Dictation".

---

Use the wrapper script `scripts/groq-dictate-hotkey.sh`:

1. System Settings > Shortcuts > Custom Shortcuts > New > Global Shortcut >
   Command/URL.
2. Set the trigger (e.g. `Meta+Space` or a dedicated key).
3. Set the command to the absolute path of the script, e.g.
   `/home/samuelsurf/Documents/python_stuff/groq-dictation/scripts/groq-dictate-hotkey.sh`

This gives press-a-key-to-dictate-one-phrase behavior that types into the
focused window.

---

## Testing without a real mic/key

- `groq-dictate --help` — verify CLI parses.
- Missing key gives a clear error (see `cli.py`).
- Mic detection: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- To test the network path, you need a real `GROQ_API_KEY`. The free tier is
  enough for development.

---

## Troubleshooting

- **"no input device"**: check `sounddevice.query_devices()` pick a different
  input in `audio.py` (`sd.default.device`).
- **Text not injected on Wayland**: `wtype` needs the compositor's
  virtual-keyboard protocol (KDE supports it). If it fails, switch to
  `--method ydotool` and ensure `ydotoold` is running.
- **429 rate limit**: you hit Groq free limits. Wait a minute (RPM) or a day
  (RPD). Unlikely for normal dictation.
- **Environment leak when run under Hermes terminal**: run the app from a normal
  terminal or systemd service, not the Hermes wrapper, to avoid the Hermes venv
  polluting `sys.path`.

---

## Next steps / ideas

- [ ] Real user sends a real Groq key to test transcription round-trip.
- [ ] Swap energy VAD for Silero VAD (better accuracy on noisy audio).
- [ ] Add punctuation/navigation commands (spoken "new line", "period", "question mark").
- [ ] Streaming transcription for faster first-token feedback (Groq supports it).
- [ ] Hold-to-talk vs tap-to-toggle UX decision.
- [ ] Systemd user service for always-on hotkey.
- [ ] Word-level timestamps for live inline correction.