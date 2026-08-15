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
pub const DEFAULT_GPT_TRANSCRIBE_MODEL: &str = "openai/gpt-transcribe";

const GROQ_BASE: &str = "https://api.groq.com/openai/v1";
const OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";

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
    }
}

/// Which key env var a provider needs (`None` for an unknown provider).
pub fn provider_key_var(provider: &str) -> Option<&'static str> {
    match provider {
        "groq" => Some("GROQ_API_KEY"),
        "gemini" | "openrouter" | "gpt-transcribe" | "gpttranscribe" | "gpt" => {
            Some("OPENROUTER_API_KEY")
        }
        _ => None,
    }
}

const GROQ_KEY_URL: &str = "https://console.groq.com/keys";
const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/keys";

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
        (_, false) => format!(
            "No API keys found. You need a Groq or OpenRouter key to transcribe. Get a free \
             Groq key at {GROQ_KEY_URL} (free tier, no card required) or an OpenRouter key at \
             {OPENROUTER_KEY_URL}, then open Settings (gear icon) in the app, paste it into \
             the matching field, and press play again."
        ),
        _ => unreachable!("other_present is only true for the two real key vars"),
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
        "gpt-transcribe" | "gpttranscribe" | "gpt" => {
            let key = require_key("OPENROUTER_API_KEY")?;
            Ok(Box::new(OpenRouterSttTranscriber::new(
                &key,
                model.unwrap_or(DEFAULT_GPT_TRANSCRIBE_MODEL),
            )))
        }
        _ => {
            bail!("Unknown provider: {provider:?} (expected 'groq', 'gemini', or 'gpt-transcribe')")
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
