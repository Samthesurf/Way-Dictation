//! Way Dictation: native Linux speech-to-text that types into any focused app.
//!
//! Speak into the mic, transcribe with Groq's free Whisper or OpenRouter's STT
//! models, then inject the text into whichever window currently has keyboard
//! focus. A Rust rewrite of the original Python project.

pub mod app;
pub mod audio;
pub mod client;
pub mod config;
pub mod injector;
pub mod wav;

#[cfg(feature = "gui")]
pub mod gui;
