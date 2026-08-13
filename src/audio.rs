//! Microphone capture with an energy-based voice activity detector (VAD).
//!
//! Records from the default input device, splits audio into phrases using RMS
//! energy over 20 ms frames, and returns each phrase as 16-bit mono PCM wrapped
//! in a WAV buffer ready for the transcription API.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::wav;

pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Energy-based voice activity detection parameters.
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub enabled: bool,
    /// Frame length in ms used for energy analysis.
    pub frame_ms: u32,
    /// Silence threshold (RMS). Lower = more sensitive to quiet speech.
    pub threshold: f32,
    /// Frames of sustained silence before a phrase is considered finished.
    pub silence_frames_trigger: u32,
    /// Maximum phrase length (ms) before we cut it off regardless.
    pub max_phrase_ms: f64,
    /// Fixed recording length when VAD is disabled (ms).
    pub fixed_record_ms: f64,
    /// Trim leading/trailing silence before sending to the API.
    pub trim_silence: bool,
    /// Speech-like padding kept around detected speech after trimming (ms).
    pub trim_pad_ms: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            sample_rate: DEFAULT_SAMPLE_RATE,
            enabled: true,
            frame_ms: 20,
            threshold: 0.010,
            silence_frames_trigger: 20,
            max_phrase_ms: 30000.0,
            fixed_record_ms: 5000.0,
            trim_silence: true,
            trim_pad_ms: 150.0,
        }
    }
}

/// Simple RMS-energy VAD over fixed-size frames.
struct EnergyVad {
    cfg: VadConfig,
    silence_runs: u32,
}

impl EnergyVad {
    fn new(cfg: &VadConfig) -> Self {
        EnergyVad {
            cfg: cfg.clone(),
            silence_runs: 0,
        }
    }

    fn rms(&self, frame: &[i16]) -> f64 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum: f64 = frame
            .iter()
            .map(|&s| {
                let x = s as f64 / 32768.0;
                x * x
            })
            .sum();
        (sum / frame.len() as f64).sqrt()
    }

    fn is_speech(&self, frame: &[i16]) -> bool {
        self.rms(frame) > self.cfg.threshold as f64
    }

    /// Returns "speech", "silence", or "end".
    fn process_frame(&mut self, frame: &[i16]) -> &'static str {
        if self.is_speech(frame) {
            self.silence_runs = 0;
            return "speech";
        }
        self.silence_runs += 1;
        if self.silence_runs >= self.cfg.silence_frames_trigger {
            return "end";
        }
        "silence"
    }
}

/// Trim leading/trailing silence from i16 PCM, keeping a small pad.
///
/// Returns the central span bracketing detected speech. If nothing is above
/// the threshold, returns an empty vector (the caller treats that as "no
/// speech").
pub fn trim_silence(
    pcm: &[i16],
    sample_rate: u32,
    threshold: f32,
    frame_ms: u32,
    pad_ms: f64,
) -> Vec<i16> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let frame = (sample_rate as usize * frame_ms as usize / 1000).max(1);
    let n = pcm.len() / frame;
    if n == 0 {
        return pcm.to_vec();
    }
    let mut first = 0usize;
    let mut last = 0usize;
    let mut any = false;
    for i in 0..n {
        let chunk = &pcm[i * frame..(i + 1) * frame];
        let sum: f64 = chunk
            .iter()
            .map(|&s| {
                let x = s as f64 / 32768.0;
                x * x
            })
            .sum();
        let rms = (sum / frame as f64).sqrt();
        if rms > threshold as f64 {
            if !any {
                first = i;
            }
            last = i;
            any = true;
        }
    }
    if !any {
        return Vec::new();
    }
    let pad = (pad_ms / 1000.0 * sample_rate as f64 / frame as f64) as usize;
    first = first.saturating_sub(pad);
    last = (last + pad).min(n - 1);
    pcm[first * frame..(last + 1) * frame].to_vec()
}

/// Open an input stream at the given sample rate, converting to i16 mono.
fn open_stream(
    device: &cpal::Device,
    sample_rate: u32,
    format: cpal::SampleFormat,
    buf: &Arc<Mutex<Vec<i16>>>,
    err: &Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream> {
    let cfg = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let mk_err = {
        let e = Arc::clone(err);
        move |stream_err: cpal::StreamError| {
            *e.lock().unwrap() = Some(stream_err.to_string());
        }
    };

    let stream = match format {
        cpal::SampleFormat::I16 => {
            let b = Arc::clone(buf);
            device.build_input_stream(
                &cfg,
                move |data: &[i16], _| {
                    b.lock().unwrap().extend_from_slice(data);
                },
                mk_err,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let b = Arc::clone(buf);
            device.build_input_stream(
                &cfg,
                move |data: &[u16], _| {
                    let mut g = b.lock().unwrap();
                    g.extend(data.iter().map(|&s| (s as i32 - 32768) as i16));
                },
                mk_err,
                None,
            )
        }
        cpal::SampleFormat::F32 => {
            let b = Arc::clone(buf);
            device.build_input_stream(
                &cfg,
                move |data: &[f32], _| {
                    let mut g = b.lock().unwrap();
                    g.extend(data.iter().map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16));
                },
                mk_err,
                None,
            )
        }
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    };

    stream.context("failed to build input stream")
}

/// Record one phrase.
///
/// Blocks until a phrase boundary is detected (or max length). If `cancel` is
/// set from another thread, the stream is aborted and the function returns
/// promptly; callers should re-check `cancel` and discard the partial audio.
///
/// Returns `(wav_bytes, duration_seconds)`.
pub fn record_phrase(cfg: &VadConfig, cancel: &AtomicBool) -> Result<(Vec<u8>, f64)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device found"))?;
    let default_cfg = device
        .default_input_config()
        .context("no default input config")?;
    let format = default_cfg.sample_format();
    let default_rate = default_cfg.sample_rate().0;

    let buffer: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    let err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    // Prefer 16 kHz (matches the Python build and the ASR services); fall back
    // to the device's native rate when the mic doesn't support it.
    let (stream, sample_rate) = match open_stream(&device, cfg.sample_rate, format, &buffer, &err) {
        Ok(s) => (s, cfg.sample_rate),
        Err(_) => {
            let s = open_stream(&device, default_rate, format, &buffer, &err)
                .context("could not open the microphone (is one connected and permitted?)")?;
            (s, default_rate)
        }
    };

    stream.play().context("failed to start audio stream")?;

    let frame_samples = (sample_rate as usize * cfg.frame_ms as usize / 1000).max(1);
    let max_samples = (sample_rate as f64 * cfg.max_phrase_ms / 1000.0) as usize;
    let fixed_samples = (sample_rate as f64 * cfg.fixed_record_ms / 1000.0) as usize;
    let target = if cfg.enabled {
        max_samples
    } else {
        fixed_samples
    };

    let start = Instant::now();
    let mut vad = EnergyVad::new(cfg);
    let mut speaking = false;

    loop {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        // Surface a stream error promptly.
        let stream_err = err.lock().unwrap().clone();
        if let Some(msg) = stream_err {
            drop(stream);
            return Err(anyhow!("audio stream error: {msg}"));
        }
        std::thread::sleep(Duration::from_millis(20));

        let len = buffer.lock().unwrap().len();

        if !cfg.enabled {
            if len >= target {
                break;
            }
            continue;
        }

        let n_frames = len / frame_samples;
        if n_frames == 0 {
            continue;
        }
        let frame: Vec<i16> = {
            let b = buffer.lock().unwrap();
            b[(n_frames - 1) * frame_samples..n_frames * frame_samples].to_vec()
        };
        let state = vad.process_frame(&frame);
        if state == "speech" {
            speaking = true;
        }
        if state == "end" && speaking {
            break;
        }
        if len >= max_samples {
            break;
        }
    }

    drop(stream);

    let samples: Vec<i16> = buffer.lock().unwrap().clone();
    let duration = start.elapsed().as_secs_f64();

    let samples = if cfg.trim_silence {
        trim_silence(
            &samples,
            sample_rate,
            cfg.threshold,
            cfg.frame_ms,
            cfg.trim_pad_ms,
        )
    } else {
        samples
    };

    let pcm = wav::i16_to_pcm(&samples);
    let wav = wav::wav_from_pcm(1, 16, sample_rate, &pcm);
    Ok((wav, duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_detects_speech() {
        let cfg = VadConfig {
            threshold: 0.01,
            ..Default::default()
        };
        let mut vad = EnergyVad::new(&cfg);
        let loud = vec![12000i16; 320];
        let quiet = vec![0i16; 320];
        assert!(vad.is_speech(&loud));
        assert!(!vad.is_speech(&quiet));
        assert_eq!(vad.process_frame(&loud), "speech");
        // 20 consecutive silent frames -> "end"
        let mut state = "silence";
        for _ in 0..20 {
            state = vad.process_frame(&quiet);
        }
        assert_eq!(state, "end");
    }

    #[test]
    fn trim_removes_silence() {
        let mut pcm = vec![0i16; 1600]; // 100ms silence
        pcm.extend(vec![8000i16; 1600]); // 100ms speech
        pcm.extend(vec![0i16; 1600]); // 100ms silence
        let trimmed = trim_silence(&pcm, 16000, 0.01, 20, 0.0);
        assert!(!trimmed.is_empty());
        assert!(trimmed.len() < pcm.len());
    }
}
