//! Config directories, environment loading, and persisted settings.

use std::env;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// `~/.config/way-dictation/` (or `$XDG_CONFIG_HOME/way-dictation`).
pub fn config_dir() -> PathBuf {
    if let Some(x) = env::var_os("XDG_CONFIG_HOME") {
        if !x.is_empty() {
            return PathBuf::from(x).join("way-dictation");
        }
    }
    if let Some(h) = env::var_os("HOME") {
        return PathBuf::from(h).join(".config").join("way-dictation");
    }
    PathBuf::from(".config").join("way-dictation")
}

pub fn key_file() -> PathBuf {
    config_dir().join("keys.env")
}

pub fn config_file() -> PathBuf {
    config_dir().join("config.json")
}

/// Load API keys. Priority: real env vars > saved keys.env > project .env > CWD .env.
///
/// The Python app saved keys to `~/.config/groq-dictation/keys.env`; keep
/// reading that file as a fallback so migrating users do not lose their keys.
/// `dotenvy` never overrides a variable that is already present in the
/// environment, so loading in this order yields exactly that precedence.
pub fn load_env(project_root: Option<&Path>) {
    if key_file().is_file() {
        let _ = dotenvy::from_path(key_file());
    }
    let legacy = config_dir()
        .parent()
        .map(|p| p.join("groq-dictation").join("keys.env"));
    if let Some(legacy) = legacy {
        if legacy.is_file() {
            let _ = dotenvy::from_path(legacy);
        }
    }
    if let Some(root) = project_root {
        let env_file = root.join(".env");
        if env_file.is_file() {
            let _ = dotenvy::from_path(&env_file);
        }
    }
    let _ = dotenvy::dotenv();
}

fn default_provider() -> String {
    "gpt-transcribe".to_string()
}

fn default_true() -> bool {
    true
}

/// Persisted GUI settings (mirrors the Python `config.json` shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_true")]
    pub always_on_top: bool,
    #[serde(default)]
    pub pos: Option<[i32; 2]>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            provider: default_provider(),
            language: String::new(),
            model: String::new(),
            always_on_top: true,
            pos: None,
        }
    }
}

pub fn load_settings() -> Settings {
    match std::fs::read_to_string(config_file()) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => Settings::default(),
    }
}

pub fn save_settings(settings: &Settings) {
    if let Err(e) = std::fs::create_dir_all(config_dir()) {
        log::warn!("could not create config dir: {e}");
        return;
    }
    if let Ok(text) = serde_json::to_string_pretty(settings) {
        if let Err(e) = std::fs::write(config_file(), text) {
            log::warn!("could not save settings: {e}");
        }
    }
}

/// Persist API keys from the settings dialog to keys.env (0600).
pub fn save_keys(groq: &str, openrouter: &str) {
    if let Err(e) = std::fs::create_dir_all(config_dir()) {
        log::warn!("could not create config dir: {e}");
        return;
    }
    let mut lines = String::new();
    if !groq.is_empty() {
        lines.push_str(&format!("GROQ_API_KEY={groq}\n"));
    }
    if !openrouter.is_empty() {
        lines.push_str(&format!("OPENROUTER_API_KEY={openrouter}\n"));
    }
    let path = key_file();
    if let Err(e) = write_private(&path, lines.as_bytes()) {
        log::warn!("could not save API keys: {e}");
    }
}

/// Write a file with mode 0600 (best-effort on Unix).
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true).mode(0o600);
    use std::io::Write;
    let mut f = opts.open(path)?;
    f.write_all(bytes)
}
