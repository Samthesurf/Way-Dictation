"""Text injection into the focused app.

Auto-detects the best method based on the session type:
- WAYLAND_DISPLAY set -> prefer wtype (KDE Plasma supports the virtual-keyboard
  protocol), fall back to ydotool (uinput, works everywhere).
- X11 -> xdotool.

ydotool requires the `ydotoold` daemon running (root/uinput access). See README
for setup.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass
class Injector:
    method: str = "auto"  # auto | wtype | ydotool | xdotool

    def _is_kde(self) -> bool:
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        return "kde" in desktop or bool(os.environ.get("KDE_FULL_SESSION"))

    def _resolve(self) -> str:
        if self.method != "auto":
            return self.method
        if os.environ.get("WAYLAND_DISPLAY"):
            # KWin does not implement the Wayland virtual-keyboard protocol, so
            # wtype fails there. ydotool (uinput) works on any compositor.
            if self._is_kde():
                if shutil.which("ydotool"):
                    return "ydotool"
                if shutil.which("wtype"):
                    return "wtype"
                raise RuntimeError(
                    "KDE/Wayland session but neither ydotool nor wtype found. "
                    "Install ydotool (and start the ydotoold user service)."
                )
            if shutil.which("wtype"):
                return "wtype"
            if shutil.which("ydotool"):
                return "ydotool"
            raise RuntimeError(
                "Wayland session but neither wtype nor ydotool found. "
                "Install wtype (non-KDE) or ydotool."
            )
        if shutil.which("xdotool"):
            return "xdotool"
        raise RuntimeError("No injection tool found (wtype/ydotool/xdotool).")

    @staticmethod
    def _run(cmd: list[str]) -> bool:
        """Run an inject command; return True on success, False on failure."""
        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def type_text(self, text: str) -> None:
        if not text:
            return
        method = self._resolve()
        if method == "wtype":
            # wtype types a string directly; '  ' inserts a space.
            ok = self._run(["wtype", "-s", "20", text.replace(" ", "  ")])
            # KWin lacks the virtual-keyboard protocol; fall back to ydotool.
            if not ok and shutil.which("ydotool"):
                self._run(["ydotool", "type", "--key-delay", "20", text])
        elif method == "ydotool":
            self._run(["ydotool", "type", "--key-delay", "20", text])
        elif method == "xdotool":
            self._run(["xdotool", "type", "--delay", "20", text])
        else:
            raise RuntimeError(f"Unknown inject method: {method}")

    def type_command(self, key: str) -> None:
        """Send a key command (e.g. 'Return', 'BackSpace')."""
        method = self._resolve()
        if method == "wtype":
            self._run(["wtype", "-k", key])
        elif method == "ydotool":
            self._run(["ydotool", "key", key])
        elif method == "xdotool":
            self._run(["xdotool", "key", key])