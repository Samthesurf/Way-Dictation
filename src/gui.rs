//! Desktop GUI: a small frameless floating dictation widget, built on iced.
//!
//! One big play/pause button drives the dictation engine (mic -> cloud
//! transcription -> keystroke injection) in a background thread; an X quits and
//! a gear opens settings. The visuals mirror the Python widget: gradient disc
//! with canvas-painted glyphs, stacked-layer drop shadow under the card, and
//! press-to-drag window movement. Build with `--features gui`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iced::futures::channel::mpsc::{UnboundedReceiver, UnboundedSender};
use iced::futures::{Stream, StreamExt};
use iced::gradient::Linear;
use iced::mouse;
use iced::widget::canvas::{Cache, Geometry, LineCap, LineJoin, Path, Program, Stroke};
use iced::widget::{
    button, center, column, container, mouse_area, pick_list, row, stack, text, text_input, tooltip,
    Canvas,
};
use iced::window::{self, Level, Position};
use iced::{
    Alignment, Background, Border, Center, Color, Element, Fill, Font, Gradient, Padding, Point,
    Radians, Rectangle, Renderer, Shadow, Size, Subscription, Task, Theme,
};
use iced_winit::program::Program as WinitProgram;
use log::{error, info, warn};

use crate::audio::{record_phrase, VadConfig};
use crate::client::{build_transcriber, Transcriber};
use crate::config::{self, Settings};
use crate::injector::{InjectMethod, Injector};

const MIN_WAV_BYTES: usize = 44 + 160;

/// One full pulse-ring loop in seconds (matches the Python QVariantAnimation).
const PULSE_PERIOD: f32 = 1.6;

// --- palette (mirrors the Python widget) ---
const ACCENT: Color = Color::from_rgb(0.41, 0.62, 0.39);
const ACCENT_HOVER: Color = Color::from_rgb(0.47, 0.70, 0.46);
const ACCENT_PRESSED: Color = Color::from_rgb(0.25, 0.39, 0.25);
const ACCENT_DARK: Color = Color::from_rgb(0.31, 0.47, 0.29);
const TEXT_C: Color = Color::from_rgb(0.91, 0.92, 0.95);
const DIM: Color = Color::from_rgb(0.55, 0.58, 0.65);
const FAINT: Color = Color::from_rgb(0.43, 0.46, 0.53);
const AMBER: Color = Color::from_rgb(1.0, 0.71, 0.33);
const RED: Color = Color::from_rgb(1.0, 0.37, 0.42);
const FIELD_BG: Color = Color::from_rgb(0.09, 0.10, 0.11);
const BORDER: Color = Color::from_rgb(0.20, 0.22, 0.26);
const BORDER_HOVER: Color = Color::from_rgb(0.27, 0.30, 0.35);

// --- geometry (mirrors the Python widget) ---
const CARD_W: f32 = 342.0 - 52.0;
const CARD_H: f32 = 420.0 - 52.0;
const CARD_ROUNDING: f32 = 22.0;

// ---------------------------------------------------------------------------
// Provider / language options for the settings pick lists
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProviderOpt {
    label: &'static str,
    value: &'static str,
}

impl std::fmt::Display for ProviderOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

const PROVIDERS: [ProviderOpt; 3] = [
    ProviderOpt {
        label: "GPT Transcribe (paid)",
        value: "gpt-transcribe",
    },
    ProviderOpt {
        label: "Groq Whisper (free)",
        value: "groq",
    },
    ProviderOpt {
        label: "Gemini (OpenRouter)",
        value: "gemini",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LanguageOpt {
    label: &'static str,
    value: &'static str,
}

impl std::fmt::Display for LanguageOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

const LANGUAGE_OPTS: [LanguageOpt; 15] = [
    LanguageOpt {
        label: "Auto-detect",
        value: "",
    },
    LanguageOpt {
        label: "English",
        value: "en",
    },
    LanguageOpt {
        label: "Igbo",
        value: "ig",
    },
    LanguageOpt {
        label: "Yoruba",
        value: "yo",
    },
    LanguageOpt {
        label: "Hausa",
        value: "ha",
    },
    LanguageOpt {
        label: "French",
        value: "fr",
    },
    LanguageOpt {
        label: "Spanish",
        value: "es",
    },
    LanguageOpt {
        label: "German",
        value: "de",
    },
    LanguageOpt {
        label: "Portuguese",
        value: "pt",
    },
    LanguageOpt {
        label: "Italian",
        value: "it",
    },
    LanguageOpt {
        label: "Dutch",
        value: "nl",
    },
    LanguageOpt {
        label: "Arabic",
        value: "ar",
    },
    LanguageOpt {
        label: "Japanese",
        value: "ja",
    },
    LanguageOpt {
        label: "Chinese",
        value: "zh",
    },
    LanguageOpt {
        label: "Russian",
        value: "ru",
    },
];

fn provider_opt(value: &str) -> ProviderOpt {
    PROVIDERS
        .iter()
        .copied()
        .find(|p| p.value == value)
        .unwrap_or(PROVIDERS[0])
}

fn language_opt(code: &str) -> LanguageOpt {
    LANGUAGE_OPTS
        .iter()
        .copied()
        .find(|l| l.value == code)
        .unwrap_or(LANGUAGE_OPTS[0])
}

// ---------------------------------------------------------------------------
// Worker thread (capture -> transcribe -> inject), mirrors the Python QThread.
// UI events flow to the app over a futures mpsc channel that the iced
// subscription polls; commands flow the other way over a std mpsc channel.
// ---------------------------------------------------------------------------

enum Cmd {
    Pause,
    Resume,
    Stop,
}

#[derive(Debug, Clone)]
enum UiEvent {
    Status(&'static str),
    Phrase { text: String },
    Error { message: String, fatal: bool },
}

struct WorkerHandle {
    cmd_tx: Sender<Cmd>,
}

fn start_worker(
    transcriber: Box<dyn Transcriber>,
    injector: Injector,
    vad: VadConfig,
    language: Option<String>,
    ev_tx: UnboundedSender<UiEvent>,
) -> WorkerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();

    // Pipelined design: the recorder thread never waits for the cloud, so
    // speech spoken while a phrase is being transcribed is still captured.
    // Phrases flow through a bounded queue to the consumer thread, which
    // transcribes and injects them in order.
    let (phrase_tx, phrase_rx) = mpsc::sync_channel::<Vec<u8>>(16);
    let paused = Arc::new(AtomicBool::new(false));

    // consumer: transcribe + inject phrases as they arrive
    {
        let paused = Arc::clone(&paused);
        let ev_tx = ev_tx.clone();
        std::thread::spawn(move || {
            while let Ok(wav) = phrase_rx.recv() {
                if paused.load(Ordering::SeqCst) {
                    continue; // dropped while paused
                }
                if wav.len() < MIN_WAV_BYTES {
                    continue;
                }
                let _ = ev_tx.unbounded_send(UiEvent::Status("transcribing"));
                let t0 = Instant::now();
                match transcriber.transcribe(&wav, language.as_deref()) {
                    Ok(text) => {
                        let latency = t0.elapsed().as_secs_f32() * 1000.0;
                        info!("Transcribed in {latency:.0}ms: {text:?}");
                        if !text.is_empty() && !paused.load(Ordering::SeqCst) {
                            match injector.type_text(&text) {
                                Ok(()) => {
                                    let _ = ev_tx.unbounded_send(UiEvent::Phrase { text });
                                }
                                Err(e) => {
                                    let _ = ev_tx.unbounded_send(UiEvent::Error {
                                        message: format!("Injection error: {e}"),
                                        fatal: false,
                                    });
                                }
                            }
                        }
                        let _ = ev_tx.unbounded_send(UiEvent::Status("listening"));
                    }
                    Err(e) => {
                        error!("Transcription error: {e}");
                        let _ = ev_tx.unbounded_send(UiEvent::Error {
                            message: format!("Transcription error: {e}"),
                            fatal: false,
                        });
                        let _ = ev_tx.unbounded_send(UiEvent::Status("listening"));
                    }
                }
            }
        });
    }

    // recorder + command coordinator: keeps capturing audio while the
    // consumer transcribes; pauses only stop the capture of new phrases
    std::thread::spawn(move || {
        let cancel = Arc::new(AtomicBool::new(false));
        let _ = ev_tx.unbounded_send(UiEvent::Status("listening"));

        loop {
            // drain pending commands
            loop {
                match cmd_rx.try_recv() {
                    Ok(Cmd::Pause) => {
                        paused.store(true, Ordering::SeqCst);
                        cancel.store(true, Ordering::SeqCst);
                    }
                    Ok(Cmd::Resume) => {
                        paused.store(false, Ordering::SeqCst);
                        cancel.store(false, Ordering::SeqCst);
                    }
                    Ok(Cmd::Stop) => return,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }

            if paused.load(Ordering::SeqCst) {
                let _ = ev_tx.unbounded_send(UiEvent::Status("paused"));
                loop {
                    match cmd_rx.recv() {
                        Ok(Cmd::Resume) => {
                            paused.store(false, Ordering::SeqCst);
                            cancel.store(false, Ordering::SeqCst);
                            let _ = ev_tx.unbounded_send(UiEvent::Status("listening"));
                            break;
                        }
                        Ok(Cmd::Stop) | Err(_) => return,
                        Ok(Cmd::Pause) => {}
                    }
                }
                continue;
            }

            cancel.store(false, Ordering::SeqCst);
            match record_phrase(&vad, &cancel) {
                Ok((wav, _)) => {
                    if cancel.load(Ordering::SeqCst) {
                        continue; // paused mid-recording: drop the partial phrase
                    }
                    match phrase_tx.try_send(wav) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            warn!("transcription queue full, dropping phrase");
                        }
                        Err(TrySendError::Disconnected(_)) => return, // consumer gone
                    }
                }
                Err(e) => {
                    let _ = ev_tx.unbounded_send(UiEvent::Error {
                        message: format!("Microphone error: {e}"),
                        fatal: true,
                    });
                    return;
                }
            }
        }
    });

    WorkerHandle { cmd_tx }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Message {
    WindowId(Option<window::Id>),
    Tick,
    Drag,
    Toggle,
    Quit,
    OpenSettings,
    CancelSettings,
    SaveSettings,
    Worker(UiEvent),
    Position(Option<Point>),
    KeyGroq(String),
    KeyOpenrouter(String),
    ToggleRevealGroq,
    ToggleRevealOpenrouter,
    DraftProvider(ProviderOpt),
    DraftLanguage(LanguageOpt),
    DraftModel(String),
    DraftOntop(bool),
    CloseRequested(window::Id),
}

/// Global slot for the worker-event receiver, installed before the app runs.
/// `Subscription::run` takes a plain function pointer (no captures), so the
/// stream builder pulls the receiver from here exactly once at startup.
static EVENT_RX: Mutex<Option<UnboundedReceiver<UiEvent>>> = Mutex::new(None);

/// True while the recorder is listening. The tick stream reads it to pick its
/// interval (33ms while the pulse ring animates, 250ms while idle).
static PULSE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// True while the play button's click-bounce runs (260ms); the tick stream
/// must also run fast so the pop animates smoothly.
static BOUNCE_ACTIVE: AtomicBool = AtomicBool::new(false);

fn worker_stream() -> impl Stream<Item = Message> {
    let rx = EVENT_RX
        .lock()
        .expect("event receiver lock poisoned")
        .take()
        .expect("worker event receiver installed before run");
    rx.map(Message::Worker)
}

/// Heartbeat stream for the pulse ring (33ms while listening) and for
/// transcript dimming / window-position saving (250ms while idle). Backed by
/// a plain thread: iced 0.14's default executor is the futures thread pool,
/// so timer streams must not assume a tokio reactor.
fn tick_stream() -> impl Stream<Item = Message> {
    let (tx, rx) = iced::futures::channel::mpsc::unbounded::<()>();
    std::thread::spawn(move || loop {
        let fast = PULSE_ACTIVE.load(Ordering::Relaxed) || BOUNCE_ACTIVE.load(Ordering::Relaxed);
        let interval = if fast {
            Duration::from_millis(33)
        } else {
            Duration::from_millis(250)
        };
        std::thread::sleep(interval);
        if tx.unbounded_send(()).is_err() {
            break;
        }
    });
    rx.map(|_| Message::Tick)
}

struct WayDictationApp {
    win: Option<window::Id>,
    settings: Settings,
    worker: Option<WorkerHandle>,
    ev_tx: UnboundedSender<UiEvent>,
    state: &'static str, // ready | listening | transcribing | paused | error
    status: String,
    hint: String,
    transcript: String,
    transcript_error: bool,
    transcript_set: Option<Instant>,
    fatal: bool,
    settings_win: Option<window::Id>,
    // settings dialog fields
    key_groq: String,
    key_openrouter: String,
    reveal_groq: bool,
    reveal_openrouter: bool,
    draft_provider: ProviderOpt,
    draft_language: LanguageOpt,
    draft_model: String,
    draft_ontop: bool,
    // window position persistence (debounced, mirrors the Python 400ms timer)
    last_pos: Option<Point>,
    pos_save_due: Option<Instant>,
    // current window height, tracked so the widget can stretch to fit text
    win_h: f32,
    // pulse-ring animation (mirrors the Python PulseRing)
    pulse_t: f32,
    last_tick: Option<Instant>,
    last_slow: Instant,
    // click bounce on the play button (mirrors the Python QVariantAnimation)
    bounce_at: Option<Instant>,
}

impl WayDictationApp {
    fn new(ev_tx: UnboundedSender<UiEvent>) -> (Self, Task<Message>) {
        let settings = config::load_settings();
        let draft_provider = provider_opt(&settings.provider);
        let draft_language = language_opt(&settings.language);
        let draft_model = settings.model.clone();
        let draft_ontop = settings.always_on_top;

        (
            WayDictationApp {
                win: None,
                settings,
                worker: None,
                ev_tx,
                state: "ready",
                status: "Ready".to_string(),
                hint: "Press play, click the app you want to type into, and speak".to_string(),
                transcript: String::new(),
                transcript_error: false,
                transcript_set: None,
                fatal: false,
                settings_win: None,
                key_groq: String::new(),
                key_openrouter: String::new(),
                reveal_groq: false,
                reveal_openrouter: false,
                draft_provider,
                draft_language,
                draft_model,
                draft_ontop,
                last_pos: None,
                pos_save_due: None,
                win_h: 420.0,
                pulse_t: 0.0,
                last_tick: None,
                last_slow: Instant::now(),
                bounce_at: None,
            },
            window::latest().map(Message::WindowId),
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::WindowId(id) => {
                self.win = id;
                Task::none()
            }
            Message::Tick => self.on_tick(),
            Message::Drag => match self.win {
                Some(id) => window::drag(id),
                None => Task::none(),
            },
            Message::Toggle => {
                self.toggle();
                Task::none()
            }
            Message::Quit => {
                self.stop_worker();
                match self.win {
                    Some(id) => window::close(id),
                    None => Task::none(),
                }
            }
            Message::OpenSettings => match self.settings_win {
                Some(id) => window::gain_focus(id),
                None => {
                    self.snapshot_settings();
                    // the open task must be executed or the window never
                    // appears; the id itself is usable immediately
                    let (id, open_task) = window::open(settings_window());
                    self.settings_win = Some(id);
                    open_task.discard()
                }
            },
            Message::CancelSettings => match self.settings_win.take() {
                Some(id) => window::close(id),
                None => Task::none(),
            },
            Message::SaveSettings => self.save_settings(),
            Message::CloseRequested(id) => {
                if self.settings_win == Some(id) {
                    self.settings_win = None;
                } else if self.win == Some(id) {
                    self.stop_worker();
                }
                Task::none()
            }
            Message::Worker(ev) => {
                self.handle_worker_event(ev);
                Task::none()
            }
            Message::Position(pos) => {
                if let Some(p) = pos {
                    if self.last_pos != Some(p) {
                        self.last_pos = Some(p);
                        self.pos_save_due = Some(Instant::now() + Duration::from_millis(400));
                    }
                }
                Task::none()
            }
            Message::KeyGroq(s) => {
                self.key_groq = s;
                Task::none()
            }
            Message::KeyOpenrouter(s) => {
                self.key_openrouter = s;
                Task::none()
            }
            Message::ToggleRevealGroq => {
                self.reveal_groq = !self.reveal_groq;
                Task::none()
            }
            Message::ToggleRevealOpenrouter => {
                self.reveal_openrouter = !self.reveal_openrouter;
                Task::none()
            }
            Message::DraftProvider(p) => {
                self.draft_provider = p;
                Task::none()
            }
            Message::DraftLanguage(l) => {
                self.draft_language = l;
                Task::none()
            }
            Message::DraftModel(s) => {
                self.draft_model = s;
                Task::none()
            }
            Message::DraftOntop(b) => {
                self.draft_ontop = b;
                Task::none()
            }
        }
    }

    fn on_tick(&mut self) -> Task<Message> {
        let now = Instant::now();
        // advance the pulse-ring phase; frozen at zero when not listening
        let pulsing = PULSE_ACTIVE.load(Ordering::Relaxed);
        if let Some(last) = self.last_tick {
            let dt = (now - last).as_secs_f32().min(0.25);
            self.pulse_t = if pulsing {
                (self.pulse_t + dt) % PULSE_PERIOD
            } else {
                0.0
            };
        }
        self.last_tick = Some(now);
        // the click bounce runs for 260ms, then the fast tick can idle
        if let Some(t0) = self.bounce_at {
            if t0.elapsed() >= Duration::from_millis(260) {
                self.bounce_at = None;
                BOUNCE_ACTIVE.store(false, Ordering::Relaxed);
            }
        }
        // housekeeping below stays on the ~250ms cadence while the pulse
        // stream ticks fast
        if now - self.last_slow < Duration::from_millis(250) {
            return Task::none();
        }
        self.last_slow = now;
        // transcript settles after its bright window
        if let Some(t0) = self.transcript_set {
            let secs = if self.transcript_error { 2.6 } else { 2.2 };
            if t0.elapsed().as_secs_f32() >= secs {
                self.transcript_set = None;
            }
        }
        // persist the window position once it has been stable for a moment
        if self.pos_save_due.is_some_and(|due| Instant::now() >= due) {
            self.pos_save_due = None;
            self.persist_pos();
        }
        let Some(id) = self.win else {
            return Task::none();
        };
        let mut tasks: Vec<Task<Message>> = Vec::new();
        // auto-size: stretch the window so wrapped text never overflows the
        // card (the estimate errs on the tall side; clipping is what we avoid)
        let desired = self.desired_height();
        if (desired - self.win_h).abs() > 0.5 {
            self.win_h = desired;
            tasks.push(window::resize(id, Size::new(342.0, desired)));
        }
        tasks.push(window::position(id).map(Message::Position));
        Task::batch(tasks)
    }

    /// Natural window height for the current status/hint/transcript text.
    fn desired_height(&self) -> f32 {
        // 13px text in the 290px card column: ~42 chars per line
        const CHARS_PER_LINE: f32 = 42.0;
        const LINE_H: f32 = 18.0;
        let hint_lines = (self.hint.chars().count() as f32 / CHARS_PER_LINE).ceil().max(1.0);
        let transcript_lines = if self.transcript.is_empty() {
            1.0 // the reserved preview slot
        } else {
            (self.transcript.chars().count() as f32 / CHARS_PER_LINE).ceil().max(1.0)
        };
        // margins (52) + card padding (36) + header (30) + play area (206)
        // + status block (35) + column spacing (24) + text lines + slack (6)
        52.0 + 36.0 + 30.0 + 206.0 + 35.0 + 24.0 + 6.0
            + (hint_lines + transcript_lines) * LINE_H
    }

    fn persist_pos(&mut self) {
        if let Some(p) = self.last_pos {
            self.settings.pos = Some([p.x as i32, p.y as i32]);
            config::save_settings(&self.settings);
        }
    }

    fn start(&mut self) {
        config::load_env(None);
        let provider = self.settings.provider.clone();
        let model = {
            let m = self.settings.model.trim();
            if m.is_empty() {
                None
            } else {
                Some(m.to_string())
            }
        };
        let language = {
            let l = self.settings.language.trim();
            if l.is_empty() {
                None
            } else {
                Some(l.to_string())
            }
        };

        match build_transcriber(&provider, model.as_deref()) {
            Ok(transcriber) => {
                let injector = Injector::new(InjectMethod::Auto);
                let handle = start_worker(
                    transcriber,
                    injector,
                    VadConfig::default(),
                    language,
                    self.ev_tx.clone(),
                );
                self.worker = Some(handle);
                self.fatal = false;
                self.set_state("listening");
            }
            Err(e) => {
                warn!("{e}");
                self.fatal = true;
                self.state = "error";
                PULSE_ACTIVE.store(false, Ordering::Relaxed);
                // Mirror the Python key-missing wording.
                let msg = e.to_string();
                let key = msg.split(" not set").next().unwrap_or("the API key");
                self.status = "API key missing".to_string();
                self.hint =
                    format!("Add {key} to the .env file in the project folder, then press play again.");
            }
        }
    }

    fn stop_worker(&mut self) {
        if let Some(w) = self.worker.take() {
            // Pause first so a recording in progress aborts instead of
            // transcribing and injecting one last phrase.
            let _ = w.cmd_tx.send(Cmd::Pause);
            let _ = w.cmd_tx.send(Cmd::Stop);
        }
        self.set_state("ready");
    }

    fn toggle(&mut self) {
        // click feedback: pop the disc, like the Python bounce animation
        self.bounce_at = Some(Instant::now());
        BOUNCE_ACTIVE.store(true, Ordering::Relaxed);
        match self.state {
            "listening" | "transcribing" => {
                if let Some(w) = &self.worker {
                    let _ = w.cmd_tx.send(Cmd::Pause);
                    self.set_state("paused");
                }
            }
            "paused" => {
                if let Some(w) = &self.worker {
                    let _ = w.cmd_tx.send(Cmd::Resume);
                    self.set_state("listening");
                }
            }
            _ => self.start(),
        }
    }

    fn set_state(&mut self, state: &'static str) {
        self.state = state;
        // the pulse ring animates only while listening
        PULSE_ACTIVE.store(state == "listening", Ordering::Relaxed);
        match state {
            "ready" => {
                self.status = "Ready".to_string();
                self.hint =
                    "Press play, click the app you want to type into, and speak".to_string();
            }
            "listening" => {
                self.status = "Listening…".to_string();
                self.hint = "Speak now, a short pause ends each phrase".to_string();
            }
            "transcribing" => {
                self.status = "Transcribing…".to_string();
                self.hint = "Sending audio to the cloud".to_string();
            }
            "paused" => {
                self.status = "Paused".to_string();
                self.hint = "Press play to resume".to_string();
            }
            "error" => {
                self.status = "Error".to_string();
            }
            _ => {}
        }
    }

    fn handle_worker_event(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Status(s) => match s {
                // The status line stays "Listening…" the whole time the
                // recorder is running; transcribing only nudges the hint.
                "listening" => {
                    if self.state == "listening" {
                        self.hint = "Speak now, a short pause ends each phrase".to_string();
                    }
                }
                "transcribing" => {
                    if self.state == "listening" {
                        self.hint = "Sending audio to the cloud…".to_string();
                    }
                }
                "paused" => {
                    if self.state == "listening" {
                        self.set_state("paused");
                    }
                }
                _ => {}
            },
            UiEvent::Phrase { text } => {
                info!("Phrase injected: {text:?}");
                self.set_transcript(text, false);
            }
            UiEvent::Error { message, fatal } => {
                if fatal {
                    self.fatal = true;
                    self.state = "error";
                    PULSE_ACTIVE.store(false, Ordering::Relaxed);
                    self.status = "Stopped".to_string();
                    self.hint = message;
                    self.worker = None;
                } else {
                    self.set_transcript(message, true);
                }
            }
        }
    }

    fn set_transcript(&mut self, text: String, error: bool) {
        self.transcript = text;
        self.transcript_error = error;
        self.transcript_set = Some(Instant::now());
    }

    /// Bright for 2.2s (or 2.6s for errors), then dimmed, like the Python one.
    fn transcript_color(&self) -> Color {
        let Some(t0) = self.transcript_set else {
            return FAINT;
        };
        let (secs, bright) = if self.transcript_error {
            (2.6, RED)
        } else {
            (2.2, TEXT_C)
        };
        if t0.elapsed().as_secs_f32() < secs {
            bright
        } else {
            FAINT
        }
    }

    fn status_color(&self) -> Color {
        match self.state {
            "listening" => ACCENT,
            "transcribing" => AMBER,
            "error" => RED,
            _ => DIM,
        }
    }

    /// Dynamic tooltip for the play button (mirrors the Python one).
    fn play_tooltip(&self) -> &'static str {
        match self.state {
            "listening" | "transcribing" => "Pause",
            "paused" => "Resume",
            _ => "Start dictation",
        }
    }

    fn snapshot_settings(&mut self) {
        self.key_groq = std::env::var("GROQ_API_KEY").unwrap_or_default();
        self.key_openrouter = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        self.draft_provider = provider_opt(&self.settings.provider);
        self.draft_language = language_opt(&self.settings.language);
        self.draft_model = self.settings.model.clone();
        self.draft_ontop = self.settings.always_on_top;
    }

    fn save_settings(&mut self) -> Task<Message> {
        let changed_ontop = self.settings.always_on_top != self.draft_ontop;
        self.settings.provider = self.draft_provider.value.to_string();
        self.settings.language = self.draft_language.value.to_string();
        self.settings.model = self.draft_model.clone();
        self.settings.always_on_top = self.draft_ontop;
        config::save_settings(&self.settings);
        config::save_keys(&self.key_groq, &self.key_openrouter);
        // Make freshly saved keys visible to the next `build_transcriber`.
        set_env_or_remove("GROQ_API_KEY", &self.key_groq);
        set_env_or_remove("OPENROUTER_API_KEY", &self.key_openrouter);

        let mut tasks: Vec<Task<Message>> = Vec::new();
        if changed_ontop {
            if let Some(win) = self.win {
                tasks.push(window::set_level(
                    win,
                    if self.draft_ontop {
                        Level::AlwaysOnTop
                    } else {
                        Level::Normal
                    },
                ));
            }
        }
        if let Some(id) = self.settings_win.take() {
            tasks.push(window::close(id));
        }
        Task::batch(tasks)
    }

    // ------------------------------------------------------------- painting

    fn view(&self, window: window::Id) -> Element<'_, Message> {
        if self.settings_win == Some(window) {
            self.settings_window_view()
        } else {
            let body = card_shell(self.main_content());
            // press-to-drag anywhere on the card background; interactive
            // children (buttons, inputs) capture their own clicks so they
            // never start a drag
            mouse_area(body).on_press(Message::Drag).into()
        }
    }

    /// The settings window paints its own opaque background because the
    /// app-wide style keeps windows transparent for the floating widget.
    fn settings_window_view(&self) -> Element<'_, Message> {
        container(self.settings_content())
            .width(Fill)
            .height(Fill)
            .padding(20)
            .style(|_t| container::Style {
                background: Some(Background::Color(Color::from_rgb(0.0, 0.0, 0.0))),
                border: Border::default(),
                text_color: None,
                ..Default::default()
            })
            .into()
    }

    fn main_content(&self) -> Element<'_, Message> {
        // symmetric 30px header buttons so the title is optically centered
        let gear = tooltip(
            icon_button(
                Canvas::new(GearGlyph).width(30).height(30),
                Color::from_rgba(1.0, 1.0, 1.0, 0.09),
                Message::OpenSettings,
            ),
            text("Settings").size(11),
            tooltip::Position::FollowCursor,
        )
        .style(tooltip_style);
        let close = tooltip(
            icon_button(
                Canvas::new(CloseGlyph).width(30).height(30),
                Color::from_rgba(1.0, 0.37, 0.42, 0.18),
                Message::Quit,
            ),
            text("Stop and quit").size(11),
            tooltip::Position::FollowCursor,
        )
        .style(tooltip_style);

        let header = row![
            gear,
            text("DICTATION").size(16).color(DIM).width(Fill).center(),
            close
        ]
        .align_y(Center);

        // canvas-painted glyph: exact geometry, rounded stroke, optically
        // nudged right (a triangle's visual mass sits on the left)
        let playing = matches!(self.state, "listening" | "transcribing");
        // click bounce: the disc pops to 107% and settles, mirroring the
        // Python QVariantAnimation keyframes (1.0 -> 1.07 @ 45% -> 1.0 over
        // 260ms; QVariantAnimation interpolates linearly between keyframes)
        let bounce_scale = match self.bounce_at {
            Some(t0) => {
                let t = (t0.elapsed().as_secs_f32() / 0.26).min(1.0);
                if t < 0.45 {
                    1.0 + 0.07 * (t / 0.45)
                } else {
                    1.0 + 0.07 * ((1.0 - t) / 0.55)
                }
            }
            None => 1.0,
        };
        let btn = 126.0 * bounce_scale;
        let play_btn = button(Canvas::new(GlyphProgram { playing }).width(btn).height(btn))
            .on_press(Message::Toggle)
            .padding(0)
            .width(btn)
            .height(btn)
            .style(move |_theme, status| {
                let (top, bottom) = match status {
                    button::Status::Hovered => (ACCENT_HOVER, ACCENT_DARK),
                    button::Status::Pressed => (ACCENT, ACCENT_PRESSED),
                    _ => (ACCENT, ACCENT_DARK),
                };
                button::Style {
                    background: Some(Background::Gradient(Gradient::Linear(
                        Linear::new(Radians(0.8))
                            .add_stop(0.0, top)
                            .add_stop(1.0, bottom),
                    ))),
                    text_color: Color::from_rgba(1.0, 1.0, 1.0, 0.96),
                    border: Border {
                        radius: (63.0 * bounce_scale).into(),
                        width: 1.0,
                        color: Color::from_rgba(1.0, 1.0, 1.0, 0.20),
                    },
                    shadow: Shadow::default(),
                    snap: false,
                }
            });
        let play_btn = tooltip(
            play_btn,
            text(self.play_tooltip()).size(11),
            tooltip::Position::FollowCursor,
        )
        .style(tooltip_style);

        // pulse ring behind the button: expanding, fading rings while
        // listening (the Python PulseRing), plus the recording glow
        let ring = Canvas::new(PulseRingProgram {
            t: self.pulse_t,
            active: matches!(self.state, "listening"),
            glowing: playing,
        })
        .width(206)
        .height(206);

        // iced Stack positions later children by their own layout, so the
        // button must expand into the full 206px box (center()) or it would
        // sit at the stack's top-left, 40px off the ring's center.
        let center_block = container(center(stack![ring, center(play_btn)])).height(206).width(Fill);

        let status = text(&self.status)
            .size(18)
            .font(Font {
                weight: iced::font::Weight::Bold,
                ..Default::default()
            })
            .color(self.status_color())
            .center()
            .width(Fill);

        // breathing room between the status line and the hint below it
        let status_block = container(status).padding(Padding {
            top: 2.0,
            right: 0.0,
            bottom: 8.0,
            left: 0.0,
        });

        let hint = text(&self.hint).size(13).color(FAINT).center().width(Fill);

        // reserved space for the last transcribed phrase (dimmed preview)
        let preview = if !self.transcript.is_empty() {
            self.transcript.as_str()
        } else if playing {
            "…"
        } else {
            ""
        };
        let transcript = container(
            text(preview)
                .size(13)
                .color(self.transcript_color())
                .center()
                .width(Fill),
        )
        .width(Fill);

        container(column![header, center_block, status_block, hint, transcript].spacing(6))
            .width(Fill)
            .height(Fill)
            .into()
    }

    fn settings_content(&self) -> Element<'_, Message> {
        // Mirrors the Python dialog: labels sit left of their fields, every
        // control shares one dark field style with rounded 8px borders, and
        // the app's green is the only accent. All text is 13px so nothing
        // shouts over the rest of the form.
        let header = text("Settings")
            .size(15)
            .font(Font {
                weight: iced::font::Weight::Bold,
                ..Default::default()
            })
            .color(TEXT_C);

        let groq_field = row![
            text_input("Not set", &self.key_groq)
                .on_input(Message::KeyGroq)
                .secure(!self.reveal_groq)
                .size(13)
                .padding(8)
                .width(Fill)
                .style(field_style),
            tooltip(
                eye_button(self.reveal_groq, Message::ToggleRevealGroq),
                text("Show / hide").size(11),
                tooltip::Position::FollowCursor,
            )
            .style(tooltip_style),
        ]
        .spacing(6)
        .align_y(Center);

        let or_field = row![
            text_input("Not set", &self.key_openrouter)
                .on_input(Message::KeyOpenrouter)
                .secure(!self.reveal_openrouter)
                .size(13)
                .padding(8)
                .width(Fill)
                .style(field_style),
            tooltip(
                eye_button(self.reveal_openrouter, Message::ToggleRevealOpenrouter),
                text("Show / hide").size(11),
                tooltip::Position::FollowCursor,
            )
            .style(tooltip_style),
        ]
        .spacing(6)
        .align_y(Center);

        let provider_field = pick_list(&PROVIDERS[..], Some(self.draft_provider), Message::DraftProvider)
            .width(Fill)
            .text_size(13)
            .padding(8)
            .style(picklist_style)
            .menu_style(menu_style);

        let language_field =
            pick_list(&LANGUAGE_OPTS[..], Some(self.draft_language), Message::DraftLanguage)
                .width(Fill)
                .text_size(13)
                .padding(8)
                .style(picklist_style)
                .menu_style(menu_style);

        let model_field = text_input("Provider default", &self.draft_model)
            .on_input(Message::DraftModel)
            .size(13)
            .padding(8)
            .width(Fill)
            .style(field_style);

        // custom-painted checkbox (the stock iced check glyph sits
        // off-center); left-aligned like the Python dialog
        let ontop_field = row![
            check_button(self.draft_ontop, Message::DraftOntop),
            text("Always on top").size(12).color(TEXT_C),
        ]
        .spacing(8)
        .align_y(Center);

        let keys_note = text(
            "Keys are saved to ~/.config/way-dictation/keys.env and take priority over the .env file.",
        )
        .size(11)
        .color(FAINT)
        .width(Fill);

        let engine_note = text(
            "GPT Transcribe and Gemini use the OpenRouter key; Groq Whisper uses the Groq key. \
             Changes apply the next time you press play.",
        )
        .size(11)
        .color(FAINT)
        .width(Fill);

        let buttons = container(
            row![
                settings_button("Cancel", false, Message::CancelSettings),
                settings_button("Save", true, Message::SaveSettings),
            ]
            .spacing(10)
            .align_y(Center),
        )
        .width(Fill)
        .padding(Padding {
            top: 8.0,
            right: 0.0,
            bottom: 0.0,
            left: 0.0,
        })
        .align_x(Alignment::End);

        // vertical rhythm: generous air before each section header, even
        // gaps between the rows inside a section
        let fields = column![
            section_label("API KEYS", 0.0),
            label_row("Groq API key", groq_field),
            label_row("OpenRouter API key", or_field),
            keys_note,
            section_label("ENGINE", 12.0),
            label_row("Provider", provider_field),
            label_row("Language", language_field),
            label_row("Model", model_field),
            container(ontop_field).padding(Padding {
                top: 2.0,
                right: 0.0,
                bottom: 2.0,
                left: 0.0,
            }),
            engine_note,
            buttons,
        ]
        .spacing(10)
        .width(Fill);

        let body = column![header, fields].spacing(14);

        container(body).width(Fill).height(Fill).into()
    }
}

// ---------------------------------------------------------------------------
// Small widgets and painting helpers
// ---------------------------------------------------------------------------

/// 30x30 header button with a custom-painted glyph and circular hover tint.
fn icon_button<'a>(
    glyph: impl Into<Element<'a, Message>>,
    hover: Color,
    msg: Message,
) -> Element<'a, Message> {
    button(glyph)
        .on_press(msg)
        .padding(0)
        .width(30)
        .height(30)
        .style(move |_t, s| button::Style {
            background: Some(Background::Color(if matches!(s, button::Status::Hovered) {
                hover
            } else {
                Color::TRANSPARENT
            })),
            text_color: TEXT_C,
            border: Border {
                radius: 15.0.into(),
                ..Default::default()
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

/// Tiny eye toggle for the API-key fields (painted, like Qt's).
fn eye_button<'a>(open: bool, msg: Message) -> Element<'a, Message> {
    button(Canvas::new(EyeGlyph { open }).width(24).height(24))
        .on_press(msg)
        .padding(0)
        .width(24)
        .height(24)
        .style(|_t, s| button::Style {
            background: Some(Background::Color(if matches!(s, button::Status::Hovered) {
                Color::from_rgba(1.0, 1.0, 1.0, 0.09)
            } else {
                Color::TRANSPARENT
            })),
            text_color: TEXT_C,
            border: Border {
                radius: 12.0.into(),
                ..Default::default()
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

/// Primary (green) or ghost (outlined) settings button. Sized by its text
/// and padding so the label is always optically centered.
fn settings_button<'a>(label: &'a str, primary: bool, msg: Message) -> Element<'a, Message> {
    button(text(label).size(13).color(if primary { Color::WHITE } else { DIM }))
        .on_press(msg)
        .padding(Padding {
            top: 8.0,
            right: 20.0,
            bottom: 8.0,
            left: 20.0,
        })
        .style(move |_t, s| {
            if primary {
                let (top, bottom) = match s {
                    button::Status::Hovered => (ACCENT_HOVER, ACCENT_DARK),
                    button::Status::Pressed => (ACCENT, ACCENT_PRESSED),
                    _ => (ACCENT, ACCENT_DARK),
                };
                button::Style {
                    background: Some(Background::Gradient(Gradient::Linear(
                        Linear::new(Radians(0.8))
                            .add_stop(0.0, top)
                            .add_stop(1.0, bottom),
                    ))),
                    text_color: Color::from_rgba(1.0, 1.0, 1.0, 0.96),
                    border: Border {
                        radius: 8.0.into(),
                        ..Default::default()
                    },
                    shadow: Shadow::default(),
                    snap: false,
                }
            } else {
                button::Style {
                    background: Some(Background::Color(Color::TRANSPARENT)),
                    text_color: if matches!(s, button::Status::Hovered) {
                        TEXT_C
                    } else {
                        DIM
                    },
                    border: Border {
                        radius: 8.0.into(),
                        width: 1.0,
                        color: if matches!(s, button::Status::Hovered) {
                            BORDER_HOVER
                        } else {
                            BORDER
                        },
                    },
                    shadow: Shadow::default(),
                    snap: false,
                }
            }
        })
        .into()
}

/// A settings row: label on the left, vertically centered against its field.
fn label_row<'a>(label: &'a str, field: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    row![
        text(label).size(12).color(DIM).width(118),
        container(field).width(Fill).align_y(Center),
    ]
    .spacing(10)
    .align_y(Center)
    .into()
}

/// Section header: white bold to contrast with the muted labels, with
/// controllable top air so sections read as distinct groups.
fn section_label(label: &str, top: f32) -> Element<'_, Message> {
    container(
        text(label)
            .size(12)
            .font(Font {
                weight: iced::font::Weight::Bold,
                ..Default::default()
            })
            .color(TEXT_C),
    )
    .padding(Padding {
        top,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    })
    .into()
}

/// Painted checkbox: rounded dark box, green fill and a hand-drawn white
/// check centered precisely when checked. Pressing sends the toggled value.
fn check_button<'a>(checked: bool, on_press: fn(bool) -> Message) -> Element<'a, Message> {
    button(Canvas::new(CheckGlyph { checked }).width(20).height(20))
        .on_press(on_press(!checked))
        .padding(0)
        .width(20)
        .height(20)
        .style(|_t, s| button::Style {
            background: Some(Background::Color(if matches!(s, button::Status::Hovered) {
                Color::from_rgba(1.0, 1.0, 1.0, 0.06)
            } else {
                Color::TRANSPARENT
            })),
            text_color: TEXT_C,
            border: Border {
                radius: 4.0.into(),
                ..Default::default()
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

fn field_style(_theme: &Theme, status: text_input::Status) -> text_input::Style {
    let border_color = match status {
        text_input::Status::Focused { .. } => ACCENT,
        text_input::Status::Hovered => BORDER_HOVER,
        _ => BORDER,
    };
    text_input::Style {
        background: Background::Color(FIELD_BG),
        border: Border {
            radius: 8.0.into(),
            width: 1.0,
            color: border_color,
        },
        icon: DIM,
        placeholder: FAINT,
        value: TEXT_C,
        selection: Color::from_rgba(0.41, 0.62, 0.39, 0.35),
    }
}

/// Tooltip bubble style. iced 0.14's default container style is transparent,
/// so an unstyled tooltip renders as bare white text floating over the
/// window; this gives it the app's dark field background and border.
fn tooltip_style(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(FIELD_BG)),
        border: Border {
            radius: 6.0.into(),
            width: 1.0,
            color: BORDER,
        },
        text_color: Some(TEXT_C),
        shadow: Shadow::default(),
        snap: false,
    }
}

/// Pick lists share the exact field style of the text inputs (same dark
/// background, 8px radius, green focus border) so the dialog reads as one
/// design instead of mixing framework defaults.
fn picklist_style(_theme: &Theme, status: pick_list::Status) -> pick_list::Style {
    let border_color = match status {
        pick_list::Status::Opened { .. } => ACCENT,
        pick_list::Status::Hovered => BORDER_HOVER,
        _ => BORDER,
    };
    pick_list::Style {
        text_color: TEXT_C,
        placeholder_color: FAINT,
        handle_color: Color::from_rgb(0.62, 0.66, 0.74),
        background: Background::Color(FIELD_BG),
        border: Border {
            radius: 8.0.into(),
            width: 1.0,
            color: border_color,
        },
    }
}

/// Dropdown menu: dark panel with the app's green as the selection accent.
fn menu_style(_theme: &Theme) -> iced::overlay::menu::Style {
    iced::overlay::menu::Style {
        background: Background::Color(Color::from_rgb(0.13, 0.14, 0.17)),
        border: Border {
            radius: 8.0.into(),
            width: 1.0,
            color: BORDER_HOVER,
        },
        text_color: TEXT_C,
        selected_text_color: Color::WHITE,
        selected_background: Background::Color(ACCENT),
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.5),
            offset: iced::Vector::new(0.0, 6.0),
            blur_radius: 18.0,
        },
    }
}

fn card_style(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb(0.0, 0.0, 0.0))),
        border: Border {
            radius: CARD_ROUNDING.into(),
            ..Default::default()
        },
        text_color: None,
        ..Default::default()
    }
}

/// Soft drop shadow: stacked rounded rects with decaying alpha, offset 6px
/// down (wgpu's native shadow blur misrenders on this transparent-window
/// setup, so the blur is faked deterministically). The layer alphas are
/// tuned so the cumulative falloff mirrors the Python QGraphicsDropShadowEffect
/// (blur 24, offset 0/8, black at 0.63): ~0.38 just outside the card edge,
/// fading to zero about 18px out.
fn shadow_layer(exp: f32, alpha: f32) -> Element<'static, Message> {
    container(
        container(column![])
            .width(CARD_W + 2.0 * exp)
            .height(CARD_H + 2.0 * exp)
            .style(move |_t| container::Style {
                background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, alpha))),
                border: Border {
                    radius: (CARD_ROUNDING + exp).into(),
                    ..Default::default()
                },
                text_color: None,
                ..Default::default()
            }),
    )
    .width(Fill)
    .height(Fill)
    .center_x(Fill)
    .align_y(Alignment::Start)
    .padding(Padding {
        top: 32.0 - exp,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    })
    .into()
}

/// The black rounded card floating on the shadow stack, with 26px margins.
/// The last stack child is the base (top); earlier children go under.
fn card_shell<'a>(content: Element<'a, Message>) -> Element<'a, Message> {
    let card = container(content)
        .width(Fill)
        .height(Fill)
        .padding(18)
        .style(card_style);
    let padded_card = container(card).padding(26).width(Fill).height(Fill);

    stack![
        shadow_layer(18.0, 0.006),
        shadow_layer(16.0, 0.010),
        shadow_layer(14.0, 0.016),
        shadow_layer(12.0, 0.024),
        shadow_layer(10.0, 0.035),
        shadow_layer(8.0, 0.050),
        shadow_layer(6.0, 0.070),
        shadow_layer(4.0, 0.100),
        shadow_layer(2.0, 0.140),
        container(padded_card)
            .width(Fill)
            .height(Fill)
            .center_x(Fill)
            .center_y(Fill),
    ]
    .into()
}

// ---------------------------------------------------------------------------
// Canvas glyphs (no font dependency, exact geometry, rounded caps/joins)
// ---------------------------------------------------------------------------

/// Paints the play/pause glyph on a canvas.
struct GlyphProgram {
    playing: bool,
}

/// Three slider lines with knobs: the settings gear.
struct GearGlyph;

/// Two crossed strokes with round caps: a proper close-button X.
struct CloseGlyph;

/// Open eye (ring + pupil) or closed eye (dimmed ring + slash).
struct EyeGlyph {
    open: bool,
}

/// Settings checkbox: rounded dark box; green fill with a hand-drawn white
/// check, precisely centered, when checked.
struct CheckGlyph {
    checked: bool,
}

impl<Message> Program<Message> for GearGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            let c = frame.center();
            let color = Color::from_rgb(0.545, 0.576, 0.647);
            let stroke = Stroke::default()
                .with_color(color)
                .with_width(2.2)
                .with_line_cap(LineCap::Round);
            for (row, knob_x) in [(-1i32, -4.0f32), (0, 4.0), (1, 0.0)] {
                let y = c.y + row as f32 * 6.5;
                frame.stroke(
                    &Path::line(Point::new(c.x - 6.5, y), Point::new(c.x + 6.5, y)),
                    stroke,
                );
                frame.fill(&Path::circle(Point::new(c.x + knob_x, y), 2.4), color);
            }
        });
        vec![geometry]
    }
}

impl<Message> Program<Message> for CloseGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            let c = frame.center();
            let d = 3.8f32;
            let stroke = Stroke::default()
                .with_color(Color::from_rgb(0.867, 0.890, 0.933))
                .with_width(2.2)
                .with_line_cap(LineCap::Round);
            frame.stroke(
                &Path::line(
                    Point::new(c.x - d, c.y - d),
                    Point::new(c.x + d, c.y + d),
                ),
                stroke,
            );
            frame.stroke(
                &Path::line(
                    Point::new(c.x + d, c.y - d),
                    Point::new(c.x - d, c.y + d),
                ),
                stroke,
            );
        });
        vec![geometry]
    }
}

impl<Message> Program<Message> for EyeGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            let c = frame.center();
            let stroke = Stroke::default().with_width(1.6).with_line_cap(LineCap::Round);
            if self.open {
                let color = Color::from_rgb(0.855, 0.878, 0.922);
                frame.stroke(&Path::circle(c, 6.0), stroke.with_color(color));
                frame.fill(&Path::circle(c, 2.2), color);
            } else {
                let color = Color::from_rgba(0.545, 0.576, 0.647, 0.6);
                frame.stroke(&Path::circle(c, 6.0), stroke.with_color(color));
                let d = 7.5f32;
                frame.stroke(
                    &Path::line(
                        Point::new(c.x - d, c.y + d),
                        Point::new(c.x + d, c.y - d),
                    ),
                    stroke.with_color(color),
                );
            }
        });
        vec![geometry]
    }
}

impl<Message> Program<Message> for CheckGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            // 20x20 canvas: the box leaves a 1px border for the stroke
            let box_path = Path::rounded_rectangle(
                Point::new(1.0, 1.0),
                Size::new(18.0, 18.0),
                iced::border::Radius::new(4.0),
            );
            if self.checked {
                frame.fill(&box_path, ACCENT);
                // check mark, centered by its bounding box
                let c = frame.center();
                let stroke = Stroke::default()
                    .with_color(Color::WHITE)
                    .with_width(2.2)
                    .with_line_cap(LineCap::Round)
                    .with_line_join(LineJoin::Round);
                let path = Path::new(|b| {
                    b.move_to(Point::new(c.x - 4.6, c.y + 0.4));
                    b.line_to(Point::new(c.x - 1.7, c.y + 3.3));
                    b.line_to(Point::new(c.x + 4.6, c.y - 3.4));
                });
                frame.stroke(&path, stroke);
            } else {
                frame.fill(&box_path, FIELD_BG);
                frame.stroke(&box_path, Stroke::default().with_color(BORDER).with_width(1.0));
            }
        });
        vec![geometry]
    }
}

impl<Message> Program<Message> for GlyphProgram {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            let c = frame.center();
            let scale = bounds.width / 126.0;
            let white = Color::from_rgba(1.0, 1.0, 1.0, 0.96);
            if self.playing {
                // two rounded pause bars
                let bar_w = 13.0 * scale;
                let bar_h = 46.0 * scale;
                let gap = 15.0 * scale;
                let r = 6.5 * scale;
                let y = c.y - bar_h / 2.0;
                for x in [c.x - bar_w - gap / 2.0, c.x + gap / 2.0] {
                    let path = Path::rounded_rectangle(
                        Point::new(x, y),
                        Size::new(bar_w, bar_h),
                        iced::border::Radius::new(r),
                    );
                    frame.fill(&path, white);
                }
            } else {
                // play triangle, nudged right: its visual mass is on the left
                let nudge = 4.0 * scale;
                let p1 = Point::new(c.x - 14.0 * scale + nudge, c.y - 24.0 * scale);
                let p2 = Point::new(c.x + 22.0 * scale + nudge, c.y);
                let p3 = Point::new(c.x - 14.0 * scale + nudge, c.y + 24.0 * scale);
                let path = Path::new(|b| {
                    b.move_to(p1);
                    b.line_to(p2);
                    b.line_to(p3);
                    b.close();
                });
                let stroke = Stroke::default()
                    .with_color(white)
                    .with_width(11.0 * scale)
                    .with_line_cap(LineCap::Round)
                    .with_line_join(LineJoin::Round);
                frame.fill(&path, white);
                frame.stroke(&path, stroke);
            }
        });
        vec![geometry]
    }
}

/// Expanding, fading rings shown around the play button while listening, plus
/// the soft green glow the Qt button paints while recording. Mirrors the
/// Python PulseRing: a 1.6s loop with two rings half a phase apart, each
/// growing from 0.58x to 1.0x of the ring box while its alpha fades out.
struct PulseRingProgram {
    t: f32, // seconds since the pulse started
    active: bool, // rings animate only while listening
    glowing: bool, // static glow while listening or transcribing
}

impl<Message> Program<Message> for PulseRingProgram {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let cache = Cache::new();
        let geometry = cache.draw(renderer, bounds.size(), |frame| {
            let c = frame.center();
            let green = |a: f32| Color::from_rgba(ACCENT.r, ACCENT.g, ACCENT.b, a);

            if self.glowing {
                // Python's radial glow fades to zero just past the disc edge;
                // layered concentric strokes approximate the falloff (iced's
                // canvas Gradient is linear-only).
                for (r, a) in [(63.5, 0.07), (64.5, 0.04), (65.5, 0.02)] {
                    frame.stroke(
                        &Path::circle(c, r),
                        Stroke::default().with_color(green(a)).with_width(1.8),
                    );
                }
            }

            if !self.active {
                return;
            }
            let max_r = bounds.width.min(bounds.height) / 2.0 - 4.0;
            let t = (self.t / PULSE_PERIOD).fract();
            for phase in [0.0, 0.5] {
                let tt = (t + phase).fract();
                let radius = max_r * (0.58 + 0.42 * tt);
                let alpha = (1.0 - tt) * 75.0 / 255.0;
                frame.stroke(
                    &Path::circle(c, radius),
                    Stroke::default().with_color(green(alpha)).with_width(2.0),
                );
            }
        });
        vec![geometry]
    }
}

// ---------------------------------------------------------------------------
// Window icon (tiny software rasterizer; no asset files)
// ---------------------------------------------------------------------------

fn make_icon() -> window::icon::Icon {
    const N: usize = 64;
    let mut rgba = vec![0u8; N * N * 4];

    let accent_top = [ACCENT.r, ACCENT.g, ACCENT.b];
    let accent_bot = [ACCENT_DARK.r, ACCENT_DARK.g, ACCENT_DARK.b];
    let acc = |t: f32| {
        let l = |a: f32, b: f32| a + (b - a) * t;
        [
            l(accent_top[0], accent_bot[0]),
            l(accent_top[1], accent_bot[1]),
            l(accent_top[2], accent_bot[2]),
        ]
    };
    let white = [1.0f32, 1.0, 1.0];

    // signed distance to a rounded rect (centers of pixels)
    let sd_rrect = |px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32, r: f32| -> f32 {
        let cx = (x0 + x1) / 2.0;
        let cy = (y0 + y1) / 2.0;
        let hw = (x1 - x0) / 2.0 - r;
        let hh = (y1 - y0) / 2.0 - r;
        let qx = (px - cx).abs() - hw;
        let qy = (py - cy).abs() - hh;
        qx.max(qy).min(0.0) + (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() - r
    };
    let cover = |sd: f32| (0.5f32 - sd).clamp(0.0, 1.0);

    for y in 0..N {
        for x in 0..N {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // premultiplied accumulators
            let mut cr = 0.0f32;
            let mut cg = 0.0f32;
            let mut cb = 0.0f32;
            let mut ca = 0.0f32;

            // over-composite `color` at coverage `a` (premultiplied source-over)
            let paint = |cr: &mut f32, cg: &mut f32, cb: &mut f32, ca: &mut f32, a: f32, color: [f32; 3]| {
                if a > 0.0 {
                    let inv = 1.0 - a;
                    *cr = *cr * inv + color[0] * a;
                    *cg = *cg * inv + color[1] * a;
                    *cb = *cb * inv + color[2] * a;
                    *ca = *ca * inv + a;
                }
            };

            // green rounded-square base with a diagonal gradient
            let t = (((px - 2.0) + (py - 2.0)) / 120.0f32).clamp(0.0, 1.0);
            paint(
                &mut cr, &mut cg, &mut cb, &mut ca,
                cover(sd_rrect(px, py, 2.0, 2.0, 62.0, 62.0, 16.0)),
                acc(t),
            );

            // mic capsule
            paint(
                &mut cr, &mut cg, &mut cb, &mut ca,
                cover(sd_rrect(px, py, 26.0, 12.0, 38.0, 36.0, 6.0)),
                white,
            );

            // arc (ring 20..160 degrees, like the Python icon)
            let dx = px - 32.0;
            let dy = py - 24.0;
            let d = (dx * dx + dy * dy).sqrt();
            let ang = dy.atan2(dx).to_degrees();
            if d > 0.1 && (20.0..=160.0).contains(&ang) {
                // signed distance to the ring band [12.5, 15.5]; only the band
                // itself paints, not the whole wedge
                let a = cover(((d - 14.0).abs()) - 1.5);
                paint(&mut cr, &mut cg, &mut cb, &mut ca, a, white);
            }

            // stem
            paint(
                &mut cr, &mut cg, &mut cb, &mut ca,
                cover(sd_rrect(px, py, 30.5, 38.0, 33.5, 46.0, 1.5)),
                white,
            );

            // base
            paint(
                &mut cr, &mut cg, &mut cb, &mut ca,
                cover(sd_rrect(px, py, 24.0, 48.5, 40.0, 51.5, 1.5)),
                white,
            );

            let i = (y * N + x) * 4;
            if ca > 0.0 {
                rgba[i] = (cr / ca) as u8;
                rgba[i + 1] = (cg / ca) as u8;
                rgba[i + 2] = (cb / ca) as u8;
                rgba[i + 3] = (ca * 255.0) as u8;
            }
        }
    }

    window::icon::from_rgba(rgba, N as u32, N as u32).expect("valid icon data")
}

// ---------------------------------------------------------------------------
// Program (multi-window) + entry point
// ---------------------------------------------------------------------------

/// The app implements the `Program` trait directly (instead of the closure
/// builder) because per-window views are required: the settings dialog is a
/// separate decorated window.
struct GuiProgram {
    ev_tx: UnboundedSender<UiEvent>,
    main_window: window::Settings,
}

impl WinitProgram for GuiProgram {
    type State = WayDictationApp;
    type Message = Message;
    type Theme = Theme;
    type Renderer = Renderer;
    type Executor = iced::executor::Default;

    fn name() -> &'static str {
        "Way Dictation"
    }

    fn settings(&self) -> iced::Settings {
        iced::Settings {
            default_font: Font {
                family: iced::font::Family::Name("Noto Sans"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn window(&self) -> Option<window::Settings> {
        Some(self.main_window.clone())
    }

    fn boot(&self) -> (Self::State, Task<Message>) {
        WayDictationApp::new(self.ev_tx.clone())
    }

    fn update(&self, state: &mut Self::State, message: Message) -> Task<Message> {
        state.update(message)
    }

    fn view<'a>(
        &self,
        state: &'a Self::State,
        window: window::Id,
    ) -> Element<'a, Message, Self::Theme, Self::Renderer> {
        state.view(window)
    }

    fn title(&self, state: &Self::State, window: window::Id) -> String {
        if state.settings_win == Some(window) {
            "Way Dictation - Settings".to_string()
        } else {
            "Way Dictation".to_string()
        }
    }

    fn subscription(&self, _state: &Self::State) -> Subscription<Message> {
        Subscription::batch([
            Subscription::run(worker_stream),
            Subscription::run(tick_stream),
            window::close_requests().map(Message::CloseRequested),
        ])
    }

    fn theme(&self, _state: &Self::State, _window: window::Id) -> Option<Theme> {
        Some(Theme::Dark)
    }

    fn style(&self, _state: &Self::State, _theme: &Theme) -> iced::theme::Style {
        // transparent window: the card paints itself, the margins show the
        // desktop (the settings window paints its own opaque background)
        iced::theme::Style {
            background_color: Color::TRANSPARENT,
            text_color: TEXT_C,
        }
    }
}

fn settings_window() -> window::Settings {
    window::Settings {
        size: Size::new(380.0, 600.0),
        position: Position::Centered,
        decorations: true,
        // opaque: a decorated dialog must never let the widget behind it
        // bleed through, which transparent surfaces can on some compositors
        transparent: false,
        resizable: false,
        level: Level::AlwaysOnTop,
        icon: Some(make_icon()),
        ..Default::default()
    }
}

pub fn run() -> Result<(), iced_winit::Error> {
    config::load_env(None);
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let settings = config::load_settings();
    let (ev_tx, ev_rx) = iced::futures::channel::mpsc::unbounded::<UiEvent>();
    *EVENT_RX.lock().expect("event slot poisoned") = Some(ev_rx);

    let main_window = window::Settings {
        size: Size::new(342.0, 420.0),
        position: match settings.pos {
            Some([x, y]) => Position::Specific(Point::new(x as f32, y as f32)),
            None => Position::Centered,
        },
        decorations: false,
        transparent: true,
        resizable: true,
        level: if settings.always_on_top {
            Level::AlwaysOnTop
        } else {
            Level::Normal
        },
        icon: Some(make_icon()),
        ..Default::default()
    };

    iced_winit::run(GuiProgram { ev_tx, main_window })
}

fn set_env_or_remove(name: &str, value: &str) {
    if value.is_empty() {
        std::env::remove_var(name);
    } else {
        std::env::set_var(name, value);
    }
}
