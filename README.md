# Way Dictation (Rust)

Native Linux speech-to-text dictation app. Speak into the mic, transcribe with
Groq's hosted (free) Whisper or OpenRouter's STT models, and type the result
into whatever app currently has keyboard focus.

This is a from-scratch Rust rewrite of the original Python project. Works on
Wayland (KDE Plasma) and X11.

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

Whisper models on the **free plan** (no card required):

| Limit | value |
|---|---|
| Requests / minute | 20 |
| Requests / day | 2,000 |
| Audio seconds / hour | 7,200 (2 hrs) |
| Audio seconds / day | 28,800 (8 hrs) |
| Max file size | 25 MB |
| Cost | $0 |

For dictation (short phrases) you use a tiny fraction of these.

### OpenRouter alternatives (researched)

- `openai/gpt-transcribe`: $0.0045/min, very accurate, paid. Brand new
  (Aug 2026). Best on this user's voice; accurate and complete.
- `openai/gpt-4o-transcribe`: $2.50/$10 per 1M tokens, also paid.
- Verdict: Groq free tier wins for a $0 budget; gpt-transcribe wins on accuracy.

---

## Architecture

```
mic (cpal, ALSA)
   -> record_phrase()  (energy-based VAD, splits speech into phrases)
   -> WAV bytes (16 kHz mono)
   -> Transcriber (Groq Whisper / OpenRouter gpt-transcribe / Gemini)
   -> text
   -> Injector (wtype / ydotool / xdotool) -> focused app
```

Modules (`src/`):

| File | Role |
|---|---|
| `audio.rs` | Mic capture + energy VAD, yields WAV for one phrase |
| `wav.rs` | WAV parse/write/split (hand-rolled, no deps) |
| `client.rs` | Groq + OpenRouter (STT and Gemini) transcription backends |
| `injector.rs` | Auto-detects text injection method for Wayland/X11 |
| `app.rs` | `DictationApp` orchestration (capture -> transcribe -> inject) |
| `main.rs` | `way-dictate` CLI entry point |
| `gui.rs` | `way-dictation-gui` floating desktop widget (iced) |

---

## Setup

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. System dependencies

Build deps (ALSA for audio capture, plus X11/Wayland libs only if you build the GUI):

```bash
# Debian/Ubuntu
sudo apt install -y pkg-config libasound2-dev
# GUI only (optional):
sudo apt install -y libxkbcommon-dev libwayland-dev libx11-dev libxrandr-dev \
    libxi-dev libxcursor-dev libxinerama-dev libgl1-mesa-dev

# Arch (KDE)
sudo pacman -S --needed alsa-lib
# GUI only (optional):
sudo pacman -S --needed libxkbcommon wayland libx11 libxrandr libxi \
    libxcursor libxinerama mesa
```

`ffmpeg` is required for the gpt-transcribe and gemini backends (they convert
WAV to mp3 before upload):

```bash
sudo apt install -y ffmpeg   # or: sudo pacman -S --needed ffmpeg
```

### 3. Get an API key

Free Groq key (no card): <https://console.groq.com/keys>.
OpenRouter key (for gpt-transcribe / gemini): <https://openrouter.ai/keys>.

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY and/or OPENROUTER_API_KEY
```

### 4. Build

```bash
cargo build --release            # CLI only
cargo build --release --features gui   # CLI + desktop widget
```

Binaries land in `target/release/`:

- `way-dictate` — the dictation CLI
- `way-dictation-gui` — the floating desktop widget (with `--features gui`)

### 5. Injection tools (Wayland/KDE)

```bash
sudo pacman -S --needed wtype ydotool   # Arch
# or: sudo apt install -y wtype ydotool
```

ydotool needs the `ydotoold` daemon running with uinput access:

```bash
sudo systemctl enable --now uinput.service ydotool.service
```

---

## Usage

```bash
# One phrase, then inject into the focused app
way-dictate --once

# Continuous loop (each phrase = one transcription), Ctrl+C to stop
way-dictate

# Transcribe an existing audio file instead of the mic
way-dictate --file path/to/audio.wav

# Force a language
way-dictate --once --lang en

# Transcribe via OpenRouter STT (default) instead of Groq
way-dictate --once --provider gpt-transcribe

# Pick any model ID for either backend
way-dictate --once --provider gpt-transcribe --model openai/gpt-transcribe
way-dictate --once --provider gemini --model google/gemini-3.1-flash-lite
way-dictate --once --provider groq --model whisper-large-v3-turbo

# Choose injection method explicitly
way-dictate --once --method wtype   # or ydotool | xdotool
```

### Providers

| Flag | Backend | Model (default) | Notes |
|---|---|---|---|
| `--provider gpt-transcribe` (**default**) | OpenRouter STT | `openai/gpt-transcribe` | Best on this user's voice; accurate + complete. Paid (~$0.0045/min) |
| `--provider groq` | Groq Whisper | `whisper-large-v3-turbo` | Free, fast; but hallucinates "thank you" on background noise |
| `--provider gemini` (alias: `openrouter`) | OpenRouter Gemini | `google/gemini-3.1-flash-lite` | General LLM; hallucinates over silence, can drop audio |
| `--provider gemini-live` (alias: `glive`) | Google Gemini Live API | `gemini-3.5-transcribe-live` | Low-latency WebSocket STT, accurate on this user's voice, free tier available |

`gpt-transcribe` and `gemini` use `OPENROUTER_API_KEY`; `groq` uses
`GROQ_API_KEY`; `gemini-live` uses `GEMINI_API_KEY` (from
<https://aistudio.google.com/apikey>). All are read from `.env` (or the GUI's
saved keys).

### Improving completeness (audio loss)

GPT-family transcription models are documented to drop/truncate segments around
pauses and at the end of audio. The community-verified fix (applied by default)
is a `prompt` forbidding omission plus a low `temperature` (0.2), and segmenting
long audio into overlapping chunks that are stitched back together.

---

## Desktop GUI (`way-dictation-gui`)

A small floating widget: one big play/pause button, an X that quits, and a gear
for settings. It stays on top and does not steal keyboard focus, so you can
press play, click into your editor/chat, and speak; each phrase is typed into
the focused window.

```bash
cargo build --release --features gui
./target/release/way-dictation-gui
```

| Control | Action |
|---|---|
| Play button | Starts the dictation loop |
| Pause (same button) | Stops listening between phrases; aborts the in-progress phrase |
| X | Stops everything and quits |
| Gear | Opens settings (API keys, provider, language, model, always-on-top) |

- API keys saved to `~/.config/way-dictation/keys.env` (0600), which take
  priority over the `.env` file; real environment variables still win.
- Settings persist to `~/.config/way-dictation/config.json`; the window
  position is remembered and restored on the next launch (compositors that
  forbid client positioning, like Wayland, ignore the restore silently).
- While listening, pulse rings animate around the button; the last injected
  phrase is previewed under the status.

### Desktop launcher (optional)

```bash
cp scripts/way-dictation-gui.desktop ~/.local/share/applications/
# edit the Exec= path to point at your built binary
```

### Binding a global hotkey

Use `scripts/way-dictate-hotkey.sh`:

1. System Settings > Shortcuts > Custom Shortcuts > New > Global Shortcut >
   Command/URL.
2. Set the trigger (e.g. `Meta+Space`).
3. Set the command to the absolute path of the script.

This gives press-a-key-to-dictate-one-phrase behavior.

---

## Testing

```bash
cargo test                 # unit tests (VAD, WAV, stitch, injector)
way-dictate --help         # verify CLI parses
way-dictate --file test.wav --provider groq   # network path (needs a key)
```

- Missing key gives a clear error.
- Mic detection: the app reports "no default input device found" if it can't
  open one.

---

## Troubleshooting

- **"no default input device found"**: no mic, or ALSA can't see it. Check with
  `arecord -l`.
- **Text not injected on Wayland**: `wtype` needs the compositor's
  virtual-keyboard protocol. On KDE use `--method ydotool` and ensure `ydotoold`
  is running.
- **429 rate limit**: you hit Groq free limits. Wait a minute (RPM) or a day (RPD).
- **ffmpeg not found**: the gpt-transcribe/gemini backends need it; install it
  or use `--provider groq`.

---

## Next steps / ideas

- [ ] Silero VAD (better accuracy on noisy audio).
- [ ] Punctuation/navigation commands (spoken "new line", "period").
- [ ] Streaming transcription for faster first-token feedback.
- [ ] Hold-to-talk vs tap-to-toggle UX decision.
- [ ] Systemd user service for always-on hotkey.
- [ ] Word-level timestamps for live inline correction.
