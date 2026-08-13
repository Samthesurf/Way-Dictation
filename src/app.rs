//! Dictation engine: capture -> transcribe -> inject.

use std::sync::atomic::{AtomicBool, Ordering};

use log::{error, info};

use crate::audio::{record_phrase, VadConfig};
use crate::client::Transcriber;
use crate::injector::Injector;

const MIN_WAV_BYTES: usize = 44 + 160; // below this there is no meaningful speech

pub struct DictationApp {
    transcriber: Box<dyn Transcriber>,
    injector: Injector,
    vad: VadConfig,
    language: Option<String>,
    auto_inject: bool,
}

impl DictationApp {
    pub fn new(
        transcriber: Box<dyn Transcriber>,
        injector: Injector,
        vad: VadConfig,
        language: Option<String>,
        auto_inject: bool,
    ) -> Self {
        DictationApp {
            transcriber,
            injector,
            vad,
            language,
            auto_inject,
        }
    }

    /// Record one phrase, transcribe it, and (optionally) inject it.
    ///
    /// Returns the transcribed text (or `None` if no speech was detected or an
    /// error occurred). `cancel` lets another thread abort the recording.
    pub fn transcribe_live(&self, cancel: &AtomicBool) -> Option<String> {
        info!("Recording... (speak now)");
        let (wav, dur) = match record_phrase(&self.vad, cancel) {
            Ok(x) => x,
            Err(e) => {
                error!("Microphone error: {e}");
                return None;
            }
        };
        if cancel.load(Ordering::SeqCst) {
            return None;
        }
        if wav.len() < MIN_WAV_BYTES {
            info!("No speech detected, skipping.");
            return None;
        }
        info!("Recorded {dur:.1}s, transcribing...");
        let t0 = std::time::Instant::now();
        let text = match self.transcriber.transcribe(&wav, self.language.as_deref()) {
            Ok(t) => t,
            Err(e) => {
                error!("Transcription error: {e}");
                return None;
            }
        };
        let latency = t0.elapsed().as_secs_f64() * 1000.0;
        info!("Transcribed in {latency:.0}ms: {text:?}");
        if !text.is_empty() && self.auto_inject {
            info!("Injecting text.");
            if let Err(e) = self.injector.type_text(&text) {
                error!("Injection error: {e}");
            }
        }
        Some(text)
    }

    /// Run continuous dictation until cancelled or `count` phrases are done.
    pub fn run_loop(&self, count: Option<usize>, cancel: &AtomicBool) {
        let mut n = 0usize;
        while !cancel.load(Ordering::SeqCst) {
            if let Some(c) = count {
                if n >= c {
                    break;
                }
            }
            self.transcribe_live(cancel);
            n += 1;
        }
        info!("Stopped.");
    }
}
