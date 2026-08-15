//! `way-dictate` CLI: native Linux speech-to-text dictation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::Parser;
use log::{error, info};

use way_dictation::app::DictationApp;
use way_dictation::audio::VadConfig;
use way_dictation::client::build_transcriber;
use way_dictation::config;
use way_dictation::injector::{InjectMethod, Injector};

#[derive(Parser, Debug)]
#[command(
    name = "way-dictate",
    version,
    about = "Native Linux dictation: speak, transcribe (Groq / OpenRouter), type into the focused app."
)]
struct Args {
    /// Transcribe a single phrase and exit.
    #[arg(long)]
    once: bool,

    /// Transcribe N phrases then exit.
    #[arg(long)]
    count: Option<usize>,

    /// Transcribe an existing audio file instead of the mic.
    #[arg(long)]
    file: Option<String>,

    /// Override the model ID for the chosen provider.
    #[arg(long)]
    model: Option<String>,

    /// Transcription backend: gpt-transcribe | groq | gemini.
    #[arg(long, default_value = "gpt-transcribe")]
    provider: String,

    /// Optional ISO-639-1 language code (e.g. en).
    #[arg(long)]
    lang: Option<String>,

    /// Injection method: auto | wtype | ydotool | xdotool.
    #[arg(long, default_value = "auto")]
    method: String,

    /// Verbose logging.
    #[arg(short, long)]
    verbose: bool,
}

fn normalize_provider(p: &str) -> String {
    match p.to_ascii_lowercase().as_str() {
        "openrouter" | "gemini" => "gemini".to_string(),
        "gpt-transcribe" | "gpttranscribe" | "gpt" => "gpt-transcribe".to_string(),
        "groq" => "groq".to_string(),
        other => other.to_string(),
    }
}

fn main() {
    let args = Args::parse();

    env_logger::Builder::new()
        .filter_level(if args.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    config::load_env(None);

    let provider = normalize_provider(&args.provider);

    // File mode: no mic, no injection; print the transcript and exit.
    if let Some(path) = &args.file {
        match build_transcriber(&provider, args.model.as_deref()) {
            Ok(t) => match t.transcribe_file(path, args.lang.as_deref()) {
                Ok(text) => println!("{text}"),
                Err(e) => {
                    error!("{e}");
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let transcriber = match build_transcriber(&provider, args.model.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let method = match InjectMethod::parse(&args.method) {
        Ok(m) => m,
        Err(e) => {
            error!("{e}");
            std::process::exit(1);
        }
    };

    let injector = Injector::new(method);
    let app = DictationApp::new(
        transcriber,
        injector,
        VadConfig::default(),
        args.lang.clone(),
        true,
    );

    // Install a Ctrl+C handler so recordings abort promptly.
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = Arc::clone(&cancel);
    if let Err(e) = ctrlc::set_handler(move || {
        c2.store(true, Ordering::SeqCst);
    }) {
        error!("could not install Ctrl+C handler: {e}");
    }

    if args.once {
        app.transcribe_live(&cancel);
    } else if let Some(count) = args.count {
        app.run_loop(Some(count), &cancel);
    } else {
        info!("Continuous dictation. Ctrl+C to stop. Type into your target app.");
        app.run_loop(None, &cancel);
    }
}
