//! Text injection into the focused app.
//!
//! Auto-detects the best method based on the session type:
//! - `WAYLAND_DISPLAY` set -> prefer wtype (KDE prefers ydotool because KWin
//!   lacks the virtual-keyboard protocol), fall back to ydotool.
//! - X11 -> xdotool.
//!
//! ydotool requires the `ydotoold` daemon running (uinput access). See README.

use std::env;
use std::process::Command;

use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectMethod {
    Auto,
    WType,
    Ydotool,
    Xdotool,
}

impl InjectMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(InjectMethod::Auto),
            "wtype" => Ok(InjectMethod::WType),
            "ydotool" => Ok(InjectMethod::Ydotool),
            "xdotool" => Ok(InjectMethod::Xdotool),
            other => Err(anyhow!(
                "unknown inject method {other:?} (expected auto|wtype|ydotool|xdotool)"
            )),
        }
    }
}

pub struct Injector {
    pub method: InjectMethod,
}

impl Injector {
    pub fn new(method: InjectMethod) -> Self {
        Injector { method }
    }

    fn is_kde(&self) -> bool {
        let desktop = env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
        desktop.to_lowercase().contains("kde")
            || env::var("KDE_FULL_SESSION")
                .map(|v| !v.is_empty())
                .unwrap_or(false)
    }

    fn which(cmd: &str) -> bool {
        env::var_os("PATH")
            .map(|p| env::split_paths(&p).any(|d| d.join(cmd).is_file()))
            .unwrap_or(false)
    }

    fn resolve(&self) -> Result<&'static str> {
        match self.method {
            InjectMethod::WType => Ok("wtype"),
            InjectMethod::Ydotool => Ok("ydotool"),
            InjectMethod::Xdotool => Ok("xdotool"),
            InjectMethod::Auto => {
                if env::var("WAYLAND_DISPLAY")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
                {
                    // KWin does not implement the Wayland virtual-keyboard
                    // protocol, so wtype fails there; ydotool works everywhere.
                    if self.is_kde() {
                        if Self::which("ydotool") {
                            return Ok("ydotool");
                        }
                        if Self::which("wtype") {
                            return Ok("wtype");
                        }
                        return Err(anyhow!(
                            "KDE/Wayland session but neither ydotool nor wtype found. \
                             Install ydotool (and start the ydotoold user service)."
                        ));
                    }
                    if Self::which("wtype") {
                        return Ok("wtype");
                    }
                    if Self::which("ydotool") {
                        return Ok("ydotool");
                    }
                    return Err(anyhow!(
                        "Wayland session but neither wtype nor ydotool found. \
                         Install wtype (non-KDE) or ydotool."
                    ));
                }
                if Self::which("xdotool") {
                    return Ok("xdotool");
                }
                Err(anyhow!("No injection tool found (wtype/ydotool/xdotool)."))
            }
        }
    }

    fn run(&self, cmd: &mut Command) -> bool {
        cmd.status().map(|s| s.success()).unwrap_or(false)
    }

    pub fn type_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let method = self.resolve()?;
        let ok = match method {
            "wtype" => {
                let spaced = text.replace(' ', "  ");
                self.run(Command::new("wtype").args(["-s", "20", &spaced]))
            }
            "ydotool" => {
                self.run(Command::new("ydotool").args(["type", "--key-delay", "20", text]))
            }
            "xdotool" => self.run(Command::new("xdotool").args(["type", "--delay", "20", text])),
            _ => unreachable!(),
        };
        // KWin lacks the virtual-keyboard protocol; fall back to ydotool.
        if !ok && method == "wtype" && Self::which("ydotool") {
            self.run(Command::new("ydotool").args(["type", "--key-delay", "20", text]));
        }
        Ok(())
    }

    pub fn type_command(&self, key: &str) -> Result<()> {
        let method = self.resolve()?;
        let ok = match method {
            "wtype" => self.run(Command::new("wtype").args(["-k", key])),
            "ydotool" => self.run(Command::new("ydotool").args(["key", key])),
            "xdotool" => self.run(Command::new("xdotool").args(["key", key])),
            _ => unreachable!(),
        };
        if !ok && method == "wtype" && Self::which("ydotool") {
            self.run(Command::new("ydotool").args(["key", key]));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_parsing() {
        assert_eq!(InjectMethod::parse("auto").unwrap(), InjectMethod::Auto);
        assert_eq!(InjectMethod::parse("wtype").unwrap(), InjectMethod::WType);
        assert!(InjectMethod::parse("nope").is_err());
    }

    #[test]
    fn auto_resolves_on_headless_box() {
        // Simulate a box with no injection tools by clearing PATH and the
        // Wayland display. No other test reads PATH, so mutating it here is
        // safe (all tests share one process).
        std::env::set_var("PATH", "/nonexistent-way-dictation-test-dir");
        std::env::remove_var("WAYLAND_DISPLAY");
        // On a box with no Wayland/X11 and no tools, resolve() must error (not
        // panic) and carry a helpful message.
        let inj = Injector::new(InjectMethod::Auto);
        let err = inj.type_text("hello").unwrap_err();
        assert!(err.to_string().contains("injection tool"), "{err}");
    }
}
