//! Desktop GUI: a small frameless floating dictation widget.
//!
//! One big play/pause button drives the dictation engine (mic -> cloud
//! transcription -> keystroke injection) in a background thread; an X quits and
//! a gear opens settings. Build with `--features gui`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{self, Color32, RichText, Stroke};
use log::{error, info, warn};

use crate::audio::{record_phrase, VadConfig};
use crate::client::{build_transcriber, Transcriber};
use crate::config::{self, Settings};
use crate::injector::{InjectMethod, Injector};

const MIN_WAV_BYTES: usize = 44 + 160;

// --- palette (mirrors the Python widget) ---
const ACCENT: Color32 = Color32::from_rgb(0x68, 0x9f, 0x63);
const ACCENT_HOVER: Color32 = Color32::from_rgb(0x77, 0xb2, 0x76);
const CARD_BG: Color32 = Color32::from_rgb(0x00, 0x00, 0x00);
const DIM: Color32 = Color32::from_rgb(0x8b, 0x93, 0xa5);
const FAINT: Color32 = Color32::from_rgb(0x6d, 0x76, 0x86);
const AMBER: Color32 = Color32::from_rgb(0xff, 0xb4, 0x54);
const RED: Color32 = Color32::from_rgb(0xff, 0x5f, 0x6b);

enum Cmd {
    Pause,
    Resume,
    Stop,
}

enum UiEvent {
    Status(&'static str),
    Phrase { text: String, latency: f32 },
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
) -> (WorkerHandle, Receiver<UiEvent>) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
    let (ev_tx, ev_rx) = mpsc::channel::<UiEvent>();

    std::thread::spawn(move || {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut paused = false;

        loop {
            while let Ok(c) = cmd_rx.try_recv() {
                match c {
                    Cmd::Pause => {
                        paused = true;
                        cancel.store(true, Ordering::SeqCst);
                    }
                    Cmd::Resume => paused = false,
                    Cmd::Stop => return,
                }
            }

            if paused {
                let _ = ev_tx.send(UiEvent::Status("paused"));
                loop {
                    match cmd_rx.recv() {
                        Ok(Cmd::Resume) => {
                            paused = false;
                            break;
                        }
                        Ok(Cmd::Stop) | Err(_) => return,
                        Ok(Cmd::Pause) => {}
                    }
                }
                continue;
            }

            cancel.store(false, Ordering::SeqCst);
            let _ = ev_tx.send(UiEvent::Status("listening"));

            let wav = match record_phrase(&vad, &cancel) {
                Ok((wav, _)) => wav,
                Err(e) => {
                    let _ = ev_tx.send(UiEvent::Error {
                        message: format!("Microphone error: {e}"),
                        fatal: true,
                    });
                    return;
                }
            };

            if cancel.load(Ordering::SeqCst) {
                continue; // paused or stopped mid-recording
            }
            if wav.len() < MIN_WAV_BYTES {
                continue;
            }

            let _ = ev_tx.send(UiEvent::Status("transcribing"));
            let t0 = Instant::now();
            match transcriber.transcribe(&wav, language.as_deref()) {
                Ok(text) => {
                    let latency = t0.elapsed().as_secs_f32() * 1000.0;
                    info!("Transcribed in {latency:.0}ms: {text:?}");
                    if !text.is_empty() {
                        match injector.type_text(&text) {
                            Ok(()) => {
                                let _ = ev_tx.send(UiEvent::Phrase { text, latency });
                            }
                            Err(e) => {
                                let _ = ev_tx.send(UiEvent::Error {
                                    message: format!("Injection error: {e}"),
                                    fatal: false,
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Transcription error: {e}");
                    let _ = ev_tx.send(UiEvent::Error {
                        message: format!("Transcription error: {e}"),
                        fatal: false,
                    });
                }
            }
        }
    });

    (WorkerHandle { cmd_tx }, ev_rx)
}

struct WayDictationApp {
    settings: Settings,
    worker: Option<WorkerHandle>,
    ev_rx: Receiver<UiEvent>,
    state: &'static str, // ready | listening | transcribing | paused | error
    status: String,
    hint: String,
    transcript: String,
    fatal: bool,
    show_settings: bool,
    // settings dialog fields
    key_groq: String,
    key_openrouter: String,
    reveal_groq: bool,
    reveal_openrouter: bool,
}

impl WayDictationApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let settings = config::load_settings();
        let (ev_tx, ev_rx) = mpsc::channel();
        drop(ev_tx);
        WayDictationApp {
            settings,
            worker: None,
            ev_rx,
            state: "ready",
            status: "Ready".to_string(),
            hint: "Press play, click the app you want to type into, and speak".to_string(),
            transcript: String::new(),
            fatal: false,
            show_settings: false,
            key_groq: std::env::var("GROQ_API_KEY").unwrap_or_default(),
            key_openrouter: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            reveal_groq: false,
            reveal_openrouter: false,
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
                let (handle, rx) =
                    start_worker(transcriber, injector, VadConfig::default(), language);
                self.worker = Some(handle);
                self.ev_rx = rx;
                self.fatal = false;
                self.set_state("listening");
            }
            Err(e) => {
                warn!("{e}");
                self.fatal = true;
                self.status = "API key missing".to_string();
                self.hint = e.to_string();
                self.state = "error";
            }
        }
    }

    fn stop_worker(&mut self) {
        if let Some(w) = self.worker.take() {
            let _ = w.cmd_tx.send(Cmd::Stop);
        }
        self.worker = None;
        self.set_state("ready");
    }

    fn toggle(&mut self) {
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
        match state {
            "ready" => {
                self.status = "Ready".to_string();
                self.hint =
                    "Press play, click the app you want to type into, and speak".to_string();
            }
            "listening" => {
                self.status = "Listening…".to_string();
                self.hint = "Speak now — a short pause ends each phrase".to_string();
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

    fn poll_events(&mut self) {
        while let Ok(ev) = self.ev_rx.try_recv() {
            match ev {
                UiEvent::Status(s) => {
                    if self.state != "error" {
                        self.set_state(s);
                    }
                }
                UiEvent::Phrase { text, latency } => {
                    info!("Phrase injected ({latency:.0}ms)");
                    self.transcript = format!("{text} · {latency:.0}ms");
                }
                UiEvent::Error { message, fatal } => {
                    if fatal {
                        self.fatal = true;
                        self.state = "error";
                        self.status = "Error".to_string();
                        self.hint = message;
                        self.worker = None;
                    } else {
                        self.transcript = message;
                    }
                }
            }
        }
    }

    fn status_color(&self) -> Color32 {
        match self.state {
            "listening" => ACCENT,
            "transcribing" => AMBER,
            "error" => RED,
            _ => DIM,
        }
    }

    fn apply_ontop(&self, ctx: &egui::Context, on: bool) {
        let level = if on {
            egui::WindowLevel::AlwaysOnTop
        } else {
            egui::WindowLevel::Normal
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
    }

    fn save_settings(&mut self) {
        config::save_settings(&self.settings);
        config::save_keys(&self.key_groq, &self.key_openrouter);
        // Make freshly saved keys visible to the next `build_transcriber`.
        set_env_or_remove("GROQ_API_KEY", &self.key_groq);
        set_env_or_remove("OPENROUTER_API_KEY", &self.key_openrouter);
    }

    fn draw_play_glyph(&self, painter: &egui::Painter, center: egui::Pos2, playing: bool) {
        let color = Color32::from_rgba_unmultiplied(255, 255, 255, 245);
        if playing {
            let bar_w = 13.0f32;
            let bar_h = 46.0f32;
            let gap = 15.0f32;
            let r = 6.5f32;
            let x1 = center.x - bar_w - gap / 2.0;
            let x2 = center.x + gap / 2.0;
            let y = center.y - bar_h / 2.0;
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x1, y), egui::vec2(bar_w, bar_h)),
                r,
                color,
            );
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x2, y), egui::vec2(bar_w, bar_h)),
                r,
                color,
            );
        } else {
            // play triangle, nudged right so it reads centered
            let p1 = egui::pos2(center.x - 12.0, center.y - 24.0);
            let p2 = egui::pos2(center.x + 24.0, center.y);
            let p3 = egui::pos2(center.x - 12.0, center.y + 24.0);
            painter.add(egui::Shape::convex_polygon(
                vec![p1, p2, p3],
                color,
                Stroke::NONE,
            ));
        }
    }

    fn show_main(&mut self, ctx: &egui::Context) {
        let listening = self.state == "listening";
        let playing = listening || self.state == "transcribing";

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(CARD_BG)
                    .rounding(22.0)
                    .inner_margin(egui::Margin::same(18.0))
                    .stroke(egui::Stroke::new(
                        1.0_f32,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 26),
                    )),
            )
            .show(ctx, |ui| {
                ui.set_min_width(250.0);

                // header: gear | title | close
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new("⚙").frame(false))
                        .on_hover_text("Settings")
                        .clicked()
                    {
                        self.show_settings = true;
                    }
                    ui.with_layout(
                        egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                        |ui| {
                            ui.label(RichText::new("DICTATION").color(DIM).strong());
                        },
                    );
                    if ui
                        .add(egui::Button::new("✕").frame(false))
                        .on_hover_text("Stop and quit")
                        .clicked()
                    {
                        self.stop_worker();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.add_space(6.0);

                // center: pulse rings + play button
                let desired = egui::vec2(206.0, 206.0);
                ui.vertical_centered(|ui| {
                    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
                    let painter = ui.painter();
                    let center = rect.center();

                    if listening {
                        let t = ctx.input(|i| i.time) as f32;
                        for phase in [0.0f32, 0.5] {
                            let tt = (t * 0.6 + phase).fract();
                            let r = (rect.width() / 2.0 - 4.0) * (0.58 + 0.42 * tt);
                            let alpha = ((1.0 - tt) * 75.0) as u8;
                            painter.circle_stroke(
                                center,
                                r,
                                Stroke::new(
                                    2.0_f32,
                                    Color32::from_rgba_unmultiplied(104, 159, 99, alpha),
                                ),
                            );
                        }
                        ctx.request_repaint();
                    }

                    let disc_r = 63.0f32;
                    let disc_color = if response.hovered() && !playing {
                        ACCENT_HOVER
                    } else if playing {
                        ACCENT
                    } else {
                        ACCENT
                    };
                    painter.circle_filled(center, disc_r, disc_color);
                    painter.circle_stroke(
                        center,
                        disc_r,
                        Stroke::new(1.4_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 55)),
                    );
                    self.draw_play_glyph(painter, center, playing);

                    if response.clicked() {
                        self.toggle();
                    }
                });

                ui.add_space(4.0);

                // status + hint + transcript
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new(&self.status)
                            .color(self.status_color())
                            .size(13.5)
                            .strong(),
                    );
                    ui.add(
                        egui::Label::new(RichText::new(&self.hint).color(FAINT).size(11.0)).wrap(),
                    );
                    if !self.transcript.is_empty() {
                        let t = self.transcript.clone();
                        ui.add(
                            egui::Label::new(RichText::new(t).color(FAINT).size(11.0).italics())
                                .wrap(),
                        );
                    }
                });
            });
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        let mut changed_ontop: Option<bool> = None;
        let mut close = false;
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label(RichText::new("API KEYS").color(FAINT).strong().size(10.0));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Groq key");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.key_groq)
                            .password(!self.reveal_groq)
                            .desired_width(220.0),
                    );
                    ui.checkbox(&mut self.reveal_groq, "👁");
                });
                ui.horizontal(|ui| {
                    ui.label("OpenRouter key");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.key_openrouter)
                            .password(!self.reveal_openrouter)
                            .desired_width(220.0),
                    );
                    ui.checkbox(&mut self.reveal_openrouter, "👁");
                });
                ui.add_space(6.0);

                ui.label(RichText::new("ENGINE").color(FAINT).strong().size(10.0));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Provider");
                    egui::ComboBox::from_id_salt("provider")
                        .selected_text(provider_label(&self.settings.provider))
                        .show_ui(ui, |ui| {
                            for (label, value) in [
                                ("GPT Transcribe (paid)", "gpt-transcribe"),
                                ("Groq Whisper (free)", "groq"),
                                ("Gemini (OpenRouter)", "gemini"),
                            ] {
                                ui.selectable_value(
                                    &mut self.settings.provider,
                                    value.to_string(),
                                    label,
                                );
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Language");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.settings.language)
                            .hint_text("auto")
                            .desired_width(120.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Model");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.settings.model)
                            .hint_text("provider default")
                            .desired_width(180.0),
                    );
                });
                ui.add_space(6.0);
                ui.checkbox(&mut self.settings.always_on_top, "Always on top");
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    if ui.button("Save").clicked() {
                        self.save_settings();
                        changed_ontop = Some(self.settings.always_on_top);
                        close = true;
                    }
                });
                ui.label(
                    RichText::new("Changes apply the next time you press play.")
                        .color(FAINT)
                        .size(11.0),
                );
            });
        if close {
            open = false;
        }
        self.show_settings = open;
        if let Some(on) = changed_ontop {
            self.apply_ontop(ctx, on);
        }
    }
}

fn set_env_or_remove(name: &str, value: &str) {
    if value.is_empty() {
        std::env::remove_var(name);
    } else {
        std::env::set_var(name, value);
    }
}

fn provider_label(p: &str) -> &str {
    match p {
        "groq" => "Groq Whisper (free)",
        "gemini" => "Gemini (OpenRouter)",
        _ => "GPT Transcribe (paid)",
    }
}

impl eframe::App for WayDictationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();
        self.show_main(ctx);
        if self.show_settings {
            self.show_settings_window(ctx);
        }
    }
}

pub fn run() -> eframe::Result<()> {
    config::load_env(None);
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([290.0, 360.0])
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top()
        .with_resizable(false)
        .with_title("Way Dictation");

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "way-dictation",
        options,
        Box::new(|cc| Ok(Box::new(WayDictationApp::new(cc)))),
    )
}
