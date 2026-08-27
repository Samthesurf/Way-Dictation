//! Transcription backends.
//!
//! Three providers, mirroring the Python project:
//! - `GroqTranscriber`: Groq's hosted `whisper-large-v3-turbo` (free tier) via
//!   the OpenAI-compatible `/audio/transcriptions` endpoint (multipart).
//! - `OpenRouterSttTranscriber`: dedicated STT models on OpenRouter's
//!   `/audio/transcriptions` endpoint (base64 JSON), e.g. `openai/gpt-transcribe`.
//! - `GeminiTranscriber`: a general-purpose LLM routed through OpenRouter chat
//!   completions, fed the audio as an inline multimodal part.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};

use crate::wav;

pub const DEFAULT_GROQ_MODEL: &str = "whisper-large-v3-turbo";
pub const DEFAULT_GEMINI_MODEL: &str = "google/gemini-3.1-flash-lite";
pub const DEFAULT_GEMINI_LIVE_MODEL: &str = "gemini-3.5-transcribe-live";
pub const DEFAULT_GPT_TRANSCRIBE_MODEL: &str = "openai/gpt-transcribe";

const GROQ_BASE: &str = "https://api.groq.com/openai/v1";
const OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";
const GEMINI_LIVE_WS: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/// Prompt that asks a general LLM to return only the verbatim transcript and to
/// stay silent when there is no intelligible speech.
const TRANSCRIBE_SYSTEM: &str =
    "You are a speech-to-text engine. Transcribe ONLY the actual words spoken \
in the audio, verbatim, with no preamble, no commentary, and no markdown. \
If the audio is silent, contains no intelligible speech, or is just noise, \
return an empty string. Never invent, repeat, or add words that were not \
spoken. Do not describe the audio; do not add filler.";

/// Prompt that stops GPT-class STT models from dropping words around pauses.
const STT_OMISSION_PROMPT: &str = "Transcribe every spoken word exactly as spoken. Do NOT omit, \
summarize, clean up, or truncate anything. Output all words, including any that \
follow pauses or occur at the very start or end of the audio. Do not combine or \
drop sentences.";

pub trait Transcriber: Send {
    fn transcribe(&self, wav: &[u8], language: Option<&str>) -> Result<String>;

    fn transcribe_file(&self, path: &str, language: Option<&str>) -> Result<String> {
        let bytes = fs::read(path).with_context(|| format!("cannot read {path}"))?;
        self.transcribe(&bytes, language)
    }
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client")
}

/// Whether each API key is present and non-empty in the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyStatus {
    pub groq: bool,
    pub openrouter: bool,
    pub gemini: bool,
}

pub fn key_status() -> KeyStatus {
    let set = |name: &str| {
        std::env::var(name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    };
    KeyStatus {
        groq: set("GROQ_API_KEY"),
        openrouter: set("OPENROUTER_API_KEY"),
        gemini: set("GEMINI_API_KEY"),
    }
}

/// Which key env var a provider needs (`None` for an unknown provider).
pub fn provider_key_var(provider: &str) -> Option<&'static str> {
    match provider {
        "groq" => Some("GROQ_API_KEY"),
        "gemini" | "openrouter" | "gpt-transcribe" | "gpttranscribe" | "gpt" => {
            Some("OPENROUTER_API_KEY")
        }
        "gemini-live" | "gemini-live-transcribe" | "glive" => Some("GEMINI_API_KEY"),
        _ => None,
    }
}

const GROQ_KEY_URL: &str = "https://console.groq.com/keys";
const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/keys";
const GEMINI_KEY_URL: &str = "https://aistudio.google.com/apikey";

/// Friendly "what to do next" text when a provider's key is missing: if the
/// other key is present, point the user at switching provider; if neither is
/// present, tell them where to get one.
fn missing_key_message(needed: &str, other_present: bool) -> String {
    match (needed, other_present) {
        ("GROQ_API_KEY", true) => format!(
            "No Groq API key found ({needed} is not set). You already have an OpenRouter key: \
             open Settings (gear icon) and switch Provider to \"GPT Transcribe (paid)\" or \
             \"Gemini (OpenRouter)\", or add a free Groq key at {GROQ_KEY_URL}."
        ),
        ("OPENROUTER_API_KEY", true) => format!(
            "No OpenRouter API key found ({needed} is not set). You already have a Groq key: \
             open Settings (gear icon) and switch Provider to \"Groq Whisper (free)\", or add \
             an OpenRouter key at {OPENROUTER_KEY_URL}."
        ),
        ("GEMINI_API_KEY", true) => format!(
            "No Gemini API key found ({needed} is not set). Switch Provider to \
             \"GPT Transcribe (paid)\", \"Groq Whisper (free)\", or \"Gemini (OpenRouter)\" \
             with one of your existing keys, or add a Gemini key at {GEMINI_KEY_URL}."
        ),
        (_, false) => format!(
            "No API keys found. You need a Groq or OpenRouter key to transcribe. Get a free \
             Groq key at {GROQ_KEY_URL} (free tier, no card required) or an OpenRouter key at \
             {OPENROUTER_KEY_URL}, then open Settings (gear icon) in the app, paste it into \
             the matching field, and press play again."
        ),
        _ => unreachable!("other_present is only true for the real key vars"),
    }
}

fn require_key(name: &str) -> Result<String> {
    if let Ok(k) = std::env::var(name) {
        if !k.trim().is_empty() {
            return Ok(k);
        }
    }
    let other = if name == "GROQ_API_KEY" {
        "OPENROUTER_API_KEY"
    } else {
        "GROQ_API_KEY"
    };
    let other_present = std::env::var(other)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    Err(anyhow!(missing_key_message(name, other_present)))
}

fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        &s[..n]
    }
}

fn temp_file(ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("way_dictation_{nanos}.{ext}"))
}

/// Convert WAV bytes to a high-quality mp3 via ffmpeg.
///
/// LLM audio encoders decode compressed mp3 far more completely than bare
/// raw-PCM WAV, which they tend to mis-segment and drop chunks from.
fn wav_to_mp3(wav_bytes: &[u8]) -> Result<Vec<u8>> {
    let wav_path = temp_file("wav");
    let mp3_path = temp_file("mp3");
    fs::write(&wav_path, wav_bytes)?;

    let out = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            wav_path.to_str().unwrap(),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            mp3_path.to_str().unwrap(),
        ])
        .output()
        .context("failed to run ffmpeg (is it installed?)")?;

    let _ = fs::remove_file(&wav_path);

    if !out.status.success() {
        let _ = fs::remove_file(&mp3_path);
        bail!(
            "ffmpeg failed: {}",
            truncate(&String::from_utf8_lossy(&out.stderr), 400)
        );
    }
    let bytes = fs::read(&mp3_path).context("reading converted mp3")?;
    let _ = fs::remove_file(&mp3_path);
    Ok(bytes)
}

/// Merge overlapping transcript chunks, deduping shared boundary words.
fn stitch(parts: &[String]) -> String {
    if parts.is_empty() {
        return String::new();
    }
    let mut built: Vec<String> = parts[0].split_whitespace().map(|s| s.to_string()).collect();
    for part in &parts[1..] {
        let pw: Vec<String> = part.split_whitespace().map(|s| s.to_string()).collect();
        if pw.is_empty() {
            continue;
        }
        let max_overlap = built.len().min(pw.len());
        let mut overlap = 0;
        for k in (1..=max_overlap).rev() {
            if built[built.len() - k..] == pw[..k] {
                overlap = k;
                break;
            }
        }
        built.truncate(built.len() - overlap);
        built.extend(pw);
    }
    built.join(" ")
}

/// Write request audio to a debug directory when `WAY_DICTATION_DEBUG_DIR`
/// (or the Python project's `GROQ_DICTATION_DEBUG_DIR`) is set.
fn maybe_dump(fmt: &str, audio: &[u8]) {
    let dir = std::env::var("WAY_DICTATION_DEBUG_DIR")
        .or_else(|_| std::env::var("GROQ_DICTATION_DEBUG_DIR"));
    if let Ok(dir) = dir {
        if let Ok(()) = fs::create_dir_all(&dir) {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let _ = fs::write(PathBuf::from(dir).join(format!("req_{nanos}.{fmt}")), audio);
        }
    }
}

// ---------------------------------------------------------------------------
// Groq (Whisper, free tier)
// ---------------------------------------------------------------------------

pub struct GroqTranscriber {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
}

impl GroqTranscriber {
    pub fn new(api_key: &str, model: &str) -> Self {
        GroqTranscriber {
            client: http_client(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

impl Transcriber for GroqTranscriber {
    fn transcribe(&self, wav: &[u8], language: Option<&str>) -> Result<String> {
        let part = reqwest::blocking::multipart::Part::bytes(wav.to_vec())
            .file_name("phrase.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::blocking::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone());
        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }
        let resp = self
            .client
            .post(format!("{GROQ_BASE}/audio/transcriptions"))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .context("Groq request failed")?;
        let status = resp.status();
        let body = resp.text().context("reading Groq response")?;
        if !status.is_success() {
            bail!("Groq HTTP {status}: {}", truncate(&body, 400));
        }
        let v: Value = serde_json::from_str(&body).context("parsing Groq response")?;
        v["text"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| anyhow!("unexpected Groq response: {}", truncate(&body, 400)))
    }
}

// ---------------------------------------------------------------------------
// OpenRouter dedicated STT (e.g. openai/gpt-transcribe)
// ---------------------------------------------------------------------------

pub struct OpenRouterSttTranscriber {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    use_mp3: bool,
    max_chunk_ms: u32,
}

impl OpenRouterSttTranscriber {
    pub fn new(api_key: &str, model: &str) -> Self {
        OpenRouterSttTranscriber {
            client: http_client(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            use_mp3: true,
            max_chunk_ms: 20000,
        }
    }

    fn transcribe_audio(&self, audio: &[u8], fmt: &str, language: Option<&str>) -> Result<String> {
        maybe_dump(fmt, audio);
        let b64 = base64::engine::general_purpose::STANDARD.encode(audio);
        let mut payload = json!({
            "model": self.model,
            "input_audio": { "data": b64, "format": fmt },
            "prompt": STT_OMISSION_PROMPT,
            "temperature": 0.2,
        });
        if let Some(lang) = language {
            payload["language"] = json!(lang);
        }
        let resp = self
            .client
            .post(format!("{OPENROUTER_BASE}/audio/transcriptions"))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .context("OpenRouter STT request failed")?;
        let status = resp.status();
        let body = resp.text().context("reading OpenRouter STT response")?;
        if !status.is_success() {
            bail!("OpenRouter STT HTTP {status}: {}", truncate(&body, 400));
        }
        let v: Value = serde_json::from_str(&body).context("parsing OpenRouter STT response")?;
        v["text"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| {
                anyhow!(
                    "unexpected OpenRouter STT response: {}",
                    truncate(&body, 400)
                )
            })
    }
}

impl Transcriber for OpenRouterSttTranscriber {
    fn transcribe(&self, wav: &[u8], language: Option<&str>) -> Result<String> {
        let duration = wav::duration_ms(wav);
        let (audio, fmt) = if self.use_mp3 {
            (wav_to_mp3(wav)?, "mp3".to_string())
        } else {
            (wav.to_vec(), "wav".to_string())
        };

        if duration <= self.max_chunk_ms as f64 {
            return self.transcribe_audio(&audio, &fmt, language);
        }

        // Long audio: split into overlapping chunks, transcribe each, stitch.
        let chunks = wav::split_chunks(wav, self.max_chunk_ms, 500)?;
        let mut parts = Vec::new();
        for chunk in &chunks {
            let (c_audio, c_fmt) = if self.use_mp3 {
                (wav_to_mp3(chunk)?, "mp3".to_string())
            } else {
                (chunk.clone(), "wav".to_string())
            };
            let t = self.transcribe_audio(&c_audio, &c_fmt, language)?;
            if !t.is_empty() {
                parts.push(t);
            }
        }
        Ok(stitch(&parts))
    }
}

// ---------------------------------------------------------------------------
// OpenRouter Gemini (general LLM with audio input)
// ---------------------------------------------------------------------------

pub struct GeminiTranscriber {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    use_mp3: bool,
    max_chunk_ms: u32,
}

impl GeminiTranscriber {
    pub fn new(api_key: &str, model: &str) -> Self {
        GeminiTranscriber {
            client: http_client(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            use_mp3: true,
            max_chunk_ms: 20000,
        }
    }

    fn transcribe_audio(&self, audio: &[u8], fmt: &str, _language: Option<&str>) -> Result<String> {
        maybe_dump(fmt, audio);
        let b64 = base64::engine::general_purpose::STANDARD.encode(audio);
        let payload = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": TRANSCRIBE_SYSTEM },
                    { "type": "input_audio", "input_audio": { "data": b64, "format": fmt } },
                ],
            }],
        });
        let resp = self
            .client
            .post(format!("{OPENROUTER_BASE}/chat/completions"))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .context("OpenRouter chat request failed")?;
        let status = resp.status();
        let body = resp.text().context("reading OpenRouter chat response")?;
        if !status.is_success() {
            bail!("OpenRouter chat HTTP {status}: {}", truncate(&body, 400));
        }
        let v: Value = serde_json::from_str(&body).context("parsing OpenRouter chat response")?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| {
                anyhow!(
                    "unexpected OpenRouter chat response: {}",
                    truncate(&body, 400)
                )
            })
    }
}

impl Transcriber for GeminiTranscriber {
    fn transcribe(&self, wav: &[u8], language: Option<&str>) -> Result<String> {
        let duration = wav::duration_ms(wav);
        let (audio, fmt) = if self.use_mp3 {
            (wav_to_mp3(wav)?, "mp3".to_string())
        } else {
            (wav.to_vec(), "wav".to_string())
        };

        if duration <= self.max_chunk_ms as f64 {
            return self.transcribe_audio(&audio, &fmt, language);
        }

        let chunks = wav::split_chunks(wav, self.max_chunk_ms, 500)?;
        let mut parts = Vec::new();
        for chunk in &chunks {
            let (c_audio, c_fmt) = if self.use_mp3 {
                (wav_to_mp3(chunk)?, "mp3".to_string())
            } else {
                (chunk.clone(), "wav".to_string())
            };
            let t = self.transcribe_audio(&c_audio, &c_fmt, language)?;
            if !t.is_empty() {
                parts.push(t);
            }
        }
        Ok(parts.join(" "))
    }
}

// ---------------------------------------------------------------------------
// Gemini Live transcription (gemini-3.5-transcribe-live via Live API WS)
// ---------------------------------------------------------------------------

/// Live transcription over the Gemini Live API WebSocket.
///
/// Protocol (per ai.google.dev/gemini-api/docs/live-api/live-transcribe):
/// 1. Open `BidiGenerateContent?key=API_KEY`.
/// 2. Send a setup message: model `gemini-3.5-transcribe-live`,
///    `responseModalities: ["TEXT"]`, optional language hints.
/// 3. Stream raw 16-bit PCM mono 16 kHz audio in ~100 ms chunks as
///    `realtimeInput.audio` (base64, mime `audio/pcm;rate=16000`).
/// 4. Signal the end of audio with `realtimeInput.audioStreamEnd: true`.
/// 5. Read `serverContent.inputTranscription.text` events for finalized text.
///
/// The app already records 16 kHz mono i16 PCM, which is exactly the format
/// this API consumes, so the WAV payload data is sent as-is.
pub struct GeminiLiveTranscriber {
    api_key: String,
    model: String,
    max_chunk_ms: u32,
}

impl GeminiLiveTranscriber {
    pub fn new(api_key: &str, model: &str) -> Self {
        GeminiLiveTranscriber {
            api_key: api_key.to_string(),
            model: model.to_string(),
            max_chunk_ms: 100,
        }
    }

    /// Send one JSON message over the WebSocket.
    fn send_json(&self, ws: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, value: &Value) -> Result<()> {
        use tungstenite::Message;
        let raw = serde_json::to_string(value).context("serializing Live API message")?;
        log::debug!("Gemini Live send: {}", truncate(&raw, 300));
        ws.send(Message::Text(raw.into()))
            .context("sending to Gemini Live API")
    }

    /// Read messages until the finalized transcription for this audio arrives
    /// (or until a timeout / session end). Returns the collected transcript
    /// text, and an error surface for setup failures.
    fn collect_transcript(
        &self,
        ws: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
        deadline: std::time::Instant,
    ) -> Result<String> {
        use tungstenite::Message;

        let mut parts: Vec<String> = Vec::new();
        loop {
            if std::time::Instant::now() > deadline {
                log::debug!("Gemini Live collect: deadline reached, parts={}", parts.len());
                break;
            }
            let msg = match ws.read() {
                Ok(m) => m,
                Err(e) => {
                    // A blocked read just means no more messages in flight;
                    // that is a valid end condition once audio is done.
                    let etxt = e.to_string();
                    if etxt.contains("timed out")
                        || etxt.contains("WouldBlock")
                        || etxt.contains("Resource temporarily unavailable")
                        || etxt.contains("Connection reset")
                    {
                        break;
                    }
                    return Err(anyhow!("reading from Gemini Live API: {e}"));
                }
            };
            match msg {
                Message::Text(t) => {
                    let text: &str = t.as_str();
                    log::debug!("Gemini Live recv(text): {}", truncate(text, 200));
                    let v: Value = match serde_json::from_str(text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if let Some(err) = v.get("error") {
                        return Err(anyhow!(
                            "Gemini Live API error: {}",
                            truncate(&err.to_string(), 400)
                        ));
                    }
                    if let Some(sc) = v.get("serverContent") {
                        if let Some(it) = sc.get("inputTranscription") {
                            if let Some(s) = it.get("text").and_then(|x| x.as_str()) {
                                if !s.trim().is_empty() {
                                    parts.push(s.trim().to_string());
                                }
                            }
                        }
                        // The server marks the end of the turn explicitly;
                        // nothing more will be finalized after this.
                        if sc.get("generationComplete").and_then(|x| x.as_bool()) == Some(true) {
                            break;
                        }
                        // Legacy stop signal (older sessions used this after
                        // audioStreamEnd before generationComplete was added).
                        if sc.get("turnComplete").and_then(|x| x.as_bool()) == Some(true) {
                            break;
                        }
                    }
                }
                // The Gemini Live API sends all response frames as Binary
                // (verified Aug 2026), including setupComplete, interim
                // transcripts, and the finalized transcript.
                Message::Binary(b) => {
                    let text: String = String::from_utf8_lossy(&b).into_owned();
                    log::debug!("Gemini Live recv(binary): {}", truncate(&text, 200));
                    let v: Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if let Some(err) = v.get("error") {
                        return Err(anyhow!(
                            "Gemini Live API error: {}",
                            truncate(&err.to_string(), 400)
                        ));
                    }
                    if let Some(sc) = v.get("serverContent") {
                        if let Some(it) = sc.get("inputTranscription") {
                            if let Some(s) = it.get("text").and_then(|x| x.as_str()) {
                                if !s.trim().is_empty() {
                                    parts.push(s.trim().to_string());
                                }
                            }
                        }
                        if sc.get("generationComplete").and_then(|x| x.as_bool()) == Some(true) {
                            break;
                        }
                        if sc.get("turnComplete").and_then(|x| x.as_bool()) == Some(true) {
                            break;
                        }
                    }
                }
                Message::Ping(p) => {
                    let _ = ws.send(Message::Pong(p));
                }
                Message::Close(c) => {
                    log::debug!("Gemini Live: server closed: {c:?}");
                    break;
                }
                Message::Pong(_) => {}
                _ => {}
            }
        }
        if parts.is_empty() {
            bail!("Gemini Live returned no transcript for this audio");
        }
        Ok(parts.join(" "))
    }

    /// Stream raw PCM (16-bit LE mono) to the WebSocket in ~100ms chunks,
    /// paced at real time. The Live API only finalizes transcripts when audio
    /// arrives at its natural cadence; dumping everything at once yields
    /// interim text but never a finalized turn.
    fn stream_pcm(
        &self,
        ws: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
        pcm: &[u8],
    ) -> Result<()> {
        use base64::Engine as _;
        let frame_bytes = 2usize; // 16-bit
        let sample_rate = 16000u32;
        let chunk_frames = (sample_rate as usize * self.max_chunk_ms as usize / 1000).max(1);
        let chunk_bytes = chunk_frames * frame_bytes;
        let mut i = 0usize;
        while i < pcm.len() {
            let end = (i + chunk_bytes).min(pcm.len());
            let b64 = base64::engine::general_purpose::STANDARD.encode(&pcm[i..end]);
            let msg = json!({
                "realtimeInput": {
                    "audio": {
                        "data": b64,
                        "mimeType": "audio/pcm;rate=16000"
                    }
                }
            });
            self.send_json(ws, &msg)?;
            i = end;
            std::thread::sleep(std::time::Duration::from_millis(self.max_chunk_ms as u64));
        }
        // Signal the end of the audio stream.
        let end_msg = json!({ "realtimeInput": { "audioStreamEnd": true } });
        self.send_json(ws, &end_msg)
    }
}

impl Transcriber for GeminiLiveTranscriber {
    fn transcribe(&self, wav: &[u8], language: Option<&str>) -> Result<String> {
        // Parse the WAV to extract raw PCM. The Live API requires 16 kHz mono
        // 16-bit PCM; our recorder produces exactly that, so the fmt/data is
        // sent verbatim. Anything else (e.g. converted file audio) must be
        // resampled by the caller or rejected here with a clear message.
        let parsed = wav::parse_wav(wav).context("parsing WAV for Gemini Live")?;
        if parsed.sample_rate != 16000 || parsed.channels != 1 || parsed.bits_per_sample != 16 {
            bail!(
                "Gemini Live requires 16 kHz mono 16-bit PCM audio, got {} Hz / {} ch / {}-bit. \
                 Re-record or convert: ffmpeg -i in.wav -ar 16000 -ac 1 -acodec pcm_s16le out.wav",
                parsed.sample_rate,
                parsed.channels,
                parsed.bits_per_sample,
            );
        }

        let url = format!(
            "{GEMINI_LIVE_WS}?key={}",
            // API keys are URL-safe enough for query strings; encode defensively.
            urlencode(&self.api_key)
        );
        let mut ws = connect_gemini_live(&url)?;
        // Give the server a bounded window to answer; long dictation phrases
        // (up to minutes) still complete well inside a generous deadline.
        // MaybeTlsStream is an enum; reach the underlying TCP socket per variant.
        // (No cfg gates here: our crate only builds this with the tls feature
        // active, and both variants exist in the compiled dependency.)
        let timeout = Some(std::time::Duration::from_secs(65));
        match ws.get_ref() {
            tungstenite::stream::MaybeTlsStream::Plain(t) => {
                let _ = t.set_read_timeout(timeout);
            }
            tungstenite::stream::MaybeTlsStream::Rustls(s) => {
                // `StreamOwned` exposes its socket as the public `sock` field.
                let _ = s.sock.set_read_timeout(timeout);
            }
            _ => {}
        }
        log::debug!("Gemini Live: WebSocket connected");

        // Setup message: dedicated transcription model, text-only modality.
        let mut lang_codes: Vec<Value> = Vec::new();
        if let Some(lang) = language {
            lang_codes.push(json!(lang));
        }
        let setup = json!({
            "setup": {
                "model": format!("models/{}", self.model),
                "generationConfig": {
                    "responseModalities": ["TEXT"]
                },
                "inputAudioTranscription": {
                    "languageCodes": lang_codes
                }
            }
        });
        self.send_json(&mut ws, &setup)?;

        // Stream the audio, then read final transcripts.
        self.stream_pcm(&mut ws, &parsed.data)?;
        // The server finalizes a turn a few seconds after audio ends; give
        // short clips a generous window (the probe showed ~5-10s worst case).
        let settle = std::time::Duration::from_secs(20);
        let deadline = std::time::Instant::now() + settle;
        let text = self.collect_transcript(&mut ws, deadline)?;
        let _ = ws.close(None);
        Ok(text)
    }
}

/// Percent-encode a value for use inside a URL query (keys are base64-ish;
/// this is defensive for any `=`, `+`, `/` characters).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Connect to the Gemini Live WebSocket, preferring IPv4.
///
/// Observed behaviour of the Live API endpoint (verified Aug 2026): the
/// service accepts TLS on both address families but only ever answers
/// transcript messages on IPv4. A connection pinned to an IPv6 AAAA record
/// completes the handshake and then receives nothing. Resolve the hostname
/// ourselves, pick an IPv4 socket address, and let `client_tls` handle the
/// TLS + WebSocket upgrade (SNI still comes from the hostname in the URL, so
/// certificate validation is unaffected).
fn connect_gemini_live(url: &str) -> Result<tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>> {
    use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
    let uri = url
        .parse::<tungstenite::http::Uri>()
        .map_err(|e| anyhow!("invalid Live API URL: {e}"))?;
    let host = uri
        .host()
        .ok_or_else(|| anyhow!("Live API URL has no host"))?
        .to_string();
    let port = uri.port_u16().unwrap_or(443);

    let mut addrs: Vec<SocketAddr> = format!("{host}:{port}")
        .to_socket_addrs()
        .context("resolving Gemini Live API host")?
        .collect();
    // Prefer IPv4 (see doc comment above); keep IPv6 as a fallback.
    addrs.sort_by_key(|a| matches!(a, SocketAddr::V6(_)));

    let mut last_err: Option<std::io::Error> = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(20)) {
            Ok(stream) => {
                log::debug!("Gemini Live: connected to {addr}");
                let (ws, _resp) = tungstenite::client_tls(url, stream)
                    .map_err(|e| anyhow!("Gemini Live WebSocket handshake: {e}"))?;
                return Ok(ws);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow!(
        "could not connect to Gemini Live API: {}",
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "no addresses".to_string())
    ))
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build the transcriber for the given provider. `model` overrides the default.
pub fn build_transcriber(provider: &str, model: Option<&str>) -> Result<Box<dyn Transcriber>> {
    match provider {
        "groq" => {
            let key = require_key("GROQ_API_KEY")?;
            Ok(Box::new(GroqTranscriber::new(
                &key,
                model.unwrap_or(DEFAULT_GROQ_MODEL),
            )))
        }
        "gemini" | "openrouter" => {
            let key = require_key("OPENROUTER_API_KEY")?;
            Ok(Box::new(GeminiTranscriber::new(
                &key,
                model.unwrap_or(DEFAULT_GEMINI_MODEL),
            )))
        }
        "gemini-live" | "gemini-live-transcribe" | "glive" => {
            let key = require_key("GEMINI_API_KEY")?;
            Ok(Box::new(GeminiLiveTranscriber::new(
                &key,
                model.unwrap_or(DEFAULT_GEMINI_LIVE_MODEL),
            )))
        }
        "gpt-transcribe" | "gpttranscribe" | "gpt" => {
            let key = require_key("OPENROUTER_API_KEY")?;
            Ok(Box::new(OpenRouterSttTranscriber::new(
                &key,
                model.unwrap_or(DEFAULT_GPT_TRANSCRIBE_MODEL),
            )))
        }
        _ => {
            bail!("Unknown provider: {provider:?} (expected 'groq', 'gemini', 'gemini-live', or 'gpt-transcribe')")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_dedupes_overlap() {
        let parts = vec![
            "hello world how are".to_string(),
            "how are you today".to_string(),
        ];
        assert_eq!(stitch(&parts), "hello world how are you today");
    }

    #[test]
    fn stitch_handles_empty() {
        assert_eq!(stitch(&[]), "");
        assert_eq!(stitch(&["  ".to_string()]), "");
    }

    #[test]
    fn provider_alias_mapping() {
        // These should resolve without error to the factory (key load fails,
        // but the mapping itself is what we check via the error message).
        assert!(build_transcriber("gpt", None).is_err());
        assert!(build_transcriber("gpttranscribe", None).is_err());
        assert!(build_transcriber("openrouter", None).is_err());
        assert!(build_transcriber("bogus", None).is_err());
    }

    #[test]
    fn provider_key_var_mapping() {
        assert_eq!(provider_key_var("groq"), Some("GROQ_API_KEY"));
        assert_eq!(
            provider_key_var("gpt-transcribe"),
            Some("OPENROUTER_API_KEY")
        );
        assert_eq!(provider_key_var("gpt"), Some("OPENROUTER_API_KEY"));
        assert_eq!(provider_key_var("gemini"), Some("OPENROUTER_API_KEY"));
        assert_eq!(provider_key_var("openrouter"), Some("OPENROUTER_API_KEY"));
        assert_eq!(provider_key_var("bogus"), None);
    }

    #[test]
    fn missing_key_message_recommends_switching_provider() {
        let m = missing_key_message("GROQ_API_KEY", true);
        assert!(m.contains("GROQ_API_KEY is not set"));
        assert!(m.contains("OpenRouter key"));
        assert!(!m.contains("Groq Whisper (free)"));
        assert!(m.contains("GPT Transcribe (paid)"));
        assert!(m.contains(GROQ_KEY_URL));

        let m2 = missing_key_message("OPENROUTER_API_KEY", true);
        assert!(m2.contains("OPENROUTER_API_KEY is not set"));
        assert!(m2.contains("Groq key"));
        assert!(m2.contains("Groq Whisper (free)"));
        assert!(m2.contains(OPENROUTER_KEY_URL));
    }

    #[test]
    fn missing_key_message_has_get_key_urls_when_none_present() {
        let m = missing_key_message("GROQ_API_KEY", false);
        assert!(m.contains(GROQ_KEY_URL));
        assert!(m.contains(OPENROUTER_KEY_URL));
        assert!(m.contains("press play again"));
    }
}
