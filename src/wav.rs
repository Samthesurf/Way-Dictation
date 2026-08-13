//! Minimal RIFF/WAVE helpers: parse, write, measure, and split.
//!
//! We hand-roll WAV handling instead of pulling in a crate because the code is
//! small and we need precise control over chunk splitting. Only standard PCM
//! (8/16/24/32-bit, any channel count) is supported; compressed or float WAV
//! files are rejected with a clear message telling the user to run ffmpeg.

use anyhow::{anyhow, bail, Result};

/// Parsed WAV metadata plus the raw `data` chunk bytes (PCM).
#[derive(Debug, Clone)]
pub struct Wav {
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_rate: u32,
    pub byte_rate: u32,
    pub block_align: u16,
    /// Raw PCM sample bytes (the `data` chunk).
    pub data: Vec<u8>,
}

impl Wav {
    /// Number of frames (samples per channel).
    pub fn frames(&self) -> usize {
        if self.block_align == 0 {
            0
        } else {
            self.data.len() / self.block_align as usize
        }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        if self.byte_rate == 0 {
            0.0
        } else {
            self.data.len() as f64 / self.byte_rate as f64 * 1000.0
        }
    }
}

/// Wrap raw PCM bytes into a WAV buffer (RIFF + fmt + data).
pub fn wav_from_pcm(channels: u16, bits_per_sample: u16, sample_rate: u32, data: &[u8]) -> Vec<u8> {
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = sample_rate * block_align as u32;
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// Encode i16 samples as little-endian PCM bytes.
pub fn i16_to_pcm(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Parse a WAV buffer. Returns an error for non-PCM or malformed input.
pub fn parse_wav(bytes: &[u8]) -> Result<Wav> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }

    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u32, u16, u16)> = None;
    let mut data: Vec<u8> = Vec::new();

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let start = pos + 8;
        let end = start + size;
        if end > bytes.len() {
            bail!("truncated WAV chunk '{}'", String::from_utf8_lossy(id));
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    bail!("fmt chunk too small");
                }
                let audio_format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
                let sample_rate =
                    u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                let byte_rate =
                    u32::from_le_bytes(bytes[start + 8..start + 12].try_into().unwrap());
                let block_align =
                    u16::from_le_bytes(bytes[start + 12..start + 14].try_into().unwrap());
                let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
                fmt = Some((
                    audio_format,
                    channels,
                    sample_rate,
                    byte_rate,
                    block_align,
                    bits,
                ));
            }
            b"data" => {
                data = bytes[start..end].to_vec();
            }
            _ => {}
        }
        pos = end + (size & 1); // chunks are word-aligned
    }

    let (audio_format, channels, sample_rate, byte_rate, block_align, bits) =
        fmt.ok_or_else(|| anyhow!("WAV file has no fmt chunk"))?;

    if audio_format == 0xFFFE {
        bail!("WAVE_FORMAT_EXTENSIBLE is not supported; convert with ffmpeg: ffmpeg -i in.wav out.wav");
    }
    if audio_format != 1 {
        bail!(
            "unsupported WAV encoding (audio_format={audio_format}); convert with ffmpeg: ffmpeg -i in.wav out.wav"
        );
    }
    if !matches!(bits, 8 | 16 | 24 | 32) {
        bail!("unsupported bit depth {bits}; convert with ffmpeg");
    }
    if data.is_empty() {
        bail!("WAV file has no data chunk");
    }

    Ok(Wav {
        channels,
        bits_per_sample: bits,
        sample_rate,
        byte_rate,
        block_align,
        data,
    })
}

/// Duration of a WAV buffer in ms, or 0.0 if it can't be parsed.
pub fn duration_ms(bytes: &[u8]) -> f64 {
    parse_wav(bytes).map(|w| w.duration_ms()).unwrap_or(0.0)
}

/// Split a WAV buffer into overlapping chunks, each re-wrapped as a WAV file.
///
/// Overlap (default 500 ms) prevents a word being cut at a chunk boundary.
pub fn split_chunks(bytes: &[u8], chunk_ms: u32, overlap_ms: u32) -> Result<Vec<Vec<u8>>> {
    let w = parse_wav(bytes)?;
    if w.block_align == 0 {
        bail!("WAV block_align is 0");
    }
    let step = (w.sample_rate as u64 * chunk_ms as u64 / 1000) as usize * w.block_align as usize;
    let overlap =
        (w.sample_rate as u64 * overlap_ms as u64 / 1000) as usize * w.block_align as usize;
    if step == 0 {
        return Ok(vec![bytes.to_vec()]);
    }
    let total = w.data.len();
    if total <= step {
        return Ok(vec![bytes.to_vec()]);
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = (start + step).min(total);
        let chunk = wav_from_pcm(
            w.channels,
            w.bits_per_sample,
            w.sample_rate,
            &w.data[start..end],
        );
        chunks.push(chunk);
        if end >= total {
            break;
        }
        start = end - overlap;
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_i16_mono() {
        let samples: Vec<i16> = (0..1600).map(|i| (i % 100 - 50) as i16).collect();
        let pcm = i16_to_pcm(&samples);
        let wav = wav_from_pcm(1, 16, 16000, &pcm);
        let parsed = parse_wav(&wav).unwrap();
        assert_eq!(parsed.sample_rate, 16000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.bits_per_sample, 16);
        assert_eq!(parsed.frames(), 1600);
        assert!((parsed.duration_ms() - 100.0).abs() < 0.01);
        assert_eq!(parsed.data, pcm);
    }

    #[test]
    fn splits_with_overlap() {
        // 1 second of 16k mono i16 = 16000 frames, 32000 bytes
        let samples = vec![100i16; 16000];
        let pcm = i16_to_pcm(&samples);
        let wav = wav_from_pcm(1, 16, 16000, &pcm);
        // 300ms chunks, 100ms overlap -> 5 chunks (0-300, 200-500, 400-700, 600-900, 800-1000)
        let chunks = split_chunks(&wav, 300, 100).unwrap();
        assert_eq!(chunks.len(), 5);
        for c in &chunks {
            assert!(parse_wav(c).is_ok());
        }
    }

    #[test]
    fn rejects_non_pcm() {
        // fake a float32 wav
        let wav = wav_from_pcm(1, 16, 16000, &[0u8; 320]);
        // flip audio_format to 3 (float)
        let mut bad = wav;
        bad[20] = 3; // audio_format low byte at fmt data offset (byte 20)
        assert!(parse_wav(&bad).is_err());
    }
}
