#!/usr/bin/env bash
# Assemble a pacman-installable pkg.tar.zst from the prebuilt release binaries.
#
# Fast local path; packaging/PKGBUILD is the rebuild-from-source path (makepkg
# fetches the v0.2.0 tag, recompiles, and runs the same installs).
set -euo pipefail
cd "$(dirname "$0")/.."

PKG="way-dictation"
VER="0.2.0"
REL="1"
OUT="packaging/dist/${PKG}-${VER}-${REL}-x86_64.pkg.tar.zst"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/usr/bin" \
         "$STAGE/usr/share/applications" \
         "$STAGE/usr/share/icons/hicolor/128x128/apps" \
         "$STAGE/usr/share/icons/hicolor/256x256/apps" \
         "$STAGE/usr/share/icons/hicolor/512x512/apps"

install -m755 target/release/way-dictate "$STAGE/usr/bin/way-dictate"
install -m755 target/release/way-dictation-gui "$STAGE/usr/bin/way-dictation-gui"
install -m755 packaging/way-dictate-hotkey.sh "$STAGE/usr/bin/way-dictate-hotkey"
install -m644 packaging/way-dictation.desktop "$STAGE/usr/share/applications/way-dictation.desktop"
install -m644 packaging/icons/way-dictation-128.png "$STAGE/usr/share/icons/hicolor/128x128/apps/way-dictation.png"
install -m644 packaging/icons/way-dictation-256.png "$STAGE/usr/share/icons/hicolor/256x256/apps/way-dictation.png"
install -m644 packaging/icons/way-dictation-512.png "$STAGE/usr/share/icons/hicolor/512x512/apps/way-dictation.png"

SIZE=$(du -sb "$STAGE/usr" | cut -f1)
BUILDDATE=$(date +%s)

cat > "$STAGE/.PKGINFO" <<EOF
pkgname = $PKG
pkgbase = $PKG
pkgver = $VER-$REL
pkgdesc = Native Linux speech-to-text dictation that types into any focused app (Groq / OpenRouter)
url = https://github.com/Samthesurf/Way-Dictation
builddate = $BUILDDATE
packager = Samuel Ukpai <Samthesurf@users.noreply.github.com>
size = $SIZE
arch = x86_64
license = MIT
depend = libxkbcommon
depend = fontconfig
depend = wayland
depend = libx11
depend = libgl
depend = vulkan-icd-loader
depend = alsa-lib
EOF

mkdir -p packaging/dist
bsdtar -C "$STAGE" -cf - .PKGINFO usr | zstd -q -c > "$OUT"
echo "built $OUT ($(stat -c%s "$OUT") bytes)"
