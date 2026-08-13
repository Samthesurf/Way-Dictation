//! `way-dictation-gui` binary entry point.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    way_dictation::gui::run()?;
    Ok(())
}
