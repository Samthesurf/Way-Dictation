Name:           way-dictation
Version:        0.2.0
Release:        1%{?dist}
Summary:        Native Linux speech-to-text dictation (Groq / OpenRouter)
License:        MIT
URL:            https://github.com/Samthesurf/Way-Dictation
Requires:       alsa-lib
Requires:       libwayland-client
Requires:       libxkbcommon
Requires:       libX11
Requires:       libglvnd-egl
Requires:       libglvnd-glx
Requires:       vulkan-loader
Requires:       fontconfig

%define srcroot /home/samuelsurf/Documents/python_stuff/groq-dictation

%description
Speak into the mic, transcribe with Groq or OpenRouter, and type into any
focused app. Ships a floating GUI widget and a CLI (way-dictate).

%prep
# Binaries are prebuilt (cargo build --release --features gui); nothing to prepare.

%build
# Nothing to build here.

%install
install -Dm755 %{srcroot}/target/release/way-dictate %{buildroot}/usr/bin/way-dictate
install -Dm755 %{srcroot}/target/release/way-dictation-gui %{buildroot}/usr/bin/way-dictation-gui
install -Dm755 %{srcroot}/packaging/way-dictate-hotkey.sh %{buildroot}/usr/bin/way-dictate-hotkey
install -Dm644 %{srcroot}/packaging/way-dictation.desktop %{buildroot}/usr/share/applications/way-dictation.desktop
install -Dm644 %{srcroot}/packaging/icons/way-dictation-128.png %{buildroot}/usr/share/icons/hicolor/128x128/apps/way-dictation.png
install -Dm644 %{srcroot}/packaging/icons/way-dictation-256.png %{buildroot}/usr/share/icons/hicolor/256x256/apps/way-dictation.png
install -Dm644 %{srcroot}/packaging/icons/way-dictation-512.png %{buildroot}/usr/share/icons/hicolor/512x512/apps/way-dictation.png

%files
/usr/bin/way-dictate
/usr/bin/way-dictation-gui
/usr/bin/way-dictate-hotkey
/usr/share/applications/way-dictation.desktop
/usr/share/icons/hicolor/128x128/apps/way-dictation.png
/usr/share/icons/hicolor/256x256/apps/way-dictation.png
/usr/share/icons/hicolor/512x512/apps/way-dictation.png

%changelog
* Fri Aug 14 2026 Samuel Ukpai <Samthesurf@users.noreply.github.com> - 0.2.0-1
- Initial packaging.
