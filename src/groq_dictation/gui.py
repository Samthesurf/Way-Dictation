"""Desktop GUI for groq-dictation.

A small frameless floating widget: one big play/pause button drives the
dictation engine, an X button quits, and a gear opens settings. Dictation
runs in a worker thread (mic capture -> cloud transcription -> keystroke
injection into the focused app), so the UI never blocks.

Run with:  groq-dictation-gui
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from PySide6 import QtCore, QtGui, QtWidgets

from .app import DictationApp
from .audio import VADConfig, record_phrase
from .client import build_transcriber
from .injector import Injector

log = logging.getLogger("groq_dictation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "groq-dictation"
)
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_SETTINGS = {
    "provider": "gpt-transcribe",
    "language": "",  # ISO-639-1, empty = auto-detect
    "model": "",  # empty = provider default
    "always_on_top": True,
    "pos": None,  # last window position [x, y]
}

# --- palette ---
CARD_TOP = "#000000"
CARD_BOTTOM = "#000000"
PANEL = "#0d0e10"
FIELD_BG = "#17191d"
BORDER = "#343842"
BORDER_HOVER = "#454b58"
BORDER_FOCUS = "#4d5566"
TEXT = "#e8ebf2"
DIM = "#8b93a5"
FAINT = "#6d7686"
ACCENT = "#689f63"
ACCENT_DARK = "#4e774a"
ACCENT_HOVER = "#77b276"
ACCENT_PRESSED = "#3f6340"
AMBER = "#ffb454"
RED = "#ff5f6b"
GREEN = "#3ecf8e"

PROVIDERS = [
    ("GPT Transcribe (paid)", "gpt-transcribe"),
    ("Groq Whisper (free)", "groq"),
    ("Gemini (OpenRouter)", "gemini"),
]
LANGUAGES = [
    ("Auto-detect", ""),
    ("English", "en"),
    ("Igbo", "ig"),
    ("Yoruba", "yo"),
    ("Hausa", "ha"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("German", "de"),
    ("Portuguese", "pt"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Arabic", "ar"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
]

MIN_WAV_BYTES = 44 + 160  # below this there is no meaningful speech
STATUS_MAX_W = 240
TRANSCRIPT_MAX_W = 250
FIELD_MAX_W = 178


def _load_env() -> None:
    """Load API keys from the project .env and the CWD .env (fallback)."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        settings.update(json.loads(CONFIG_FILE.read_text()))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(settings, indent=2))
    except OSError as e:  # pragma: no cover
        log.warning("Could not save settings: %s", e)


def _elide(text: str, font: QtGui.QFont, width: int) -> str:
    return QtGui.QFontMetrics(font).elidedText(text, QtCore.Qt.TextElideMode.ElideRight, width)


def _paint_chevron(p: QtGui.QPainter) -> None:
    pen = QtGui.QPen(QtGui.QColor("#8f97a5"), 2.6, QtCore.Qt.PenStyle.SolidLine,
                     QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QtCore.QPointF(8, 9.5), QtCore.QPointF(12, 13.5), QtCore.QPointF(16, 9.5)])


def _paint_check(p: QtGui.QPainter) -> None:
    pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 240), 2.8, QtCore.Qt.PenStyle.SolidLine,
                     QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QtCore.QPointF(6.5, 12.5), QtCore.QPointF(10, 16), QtCore.QPointF(17.5, 8)])


def _write_glyph_assets() -> tuple[str | None, str | None]:
    """Paint chevron/check PNGs used by QSS; return their paths (or None)."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        chevron = CONFIG_DIR / "chevron.png"
        check = CONFIG_DIR / "check.png"
    except OSError:
        return None, None
    try:
        for path, fn in ((chevron, _paint_chevron), (check, _paint_check)):
            pm = QtGui.QPixmap(24, 24)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            fn(p)
            p.end()
            pm.save(str(path))
    except OSError:
        return None, None
    return str(chevron), str(check)


class DictationWorker(QtCore.QThread):
    """Runs the capture -> transcribe -> inject loop off the GUI thread."""

    statusChanged = QtCore.Signal(str)  # "listening" | "transcribing" | "paused"
    phraseDone = QtCore.Signal(str, float)  # text, latency_ms
    errorOccurred = QtCore.Signal(str, bool)  # message, fatal

    def __init__(self, app: DictationApp, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.app = app
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._cancel = threading.Event()

    # -- control (callable from the GUI thread) --
    def pause(self) -> None:
        self._pause.set()
        self._cancel.set()  # abort an in-flight recording promptly

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def run(self) -> None:  # noqa: D102
        vad = self.app.vad
        transcriber = self.app.transcriber
        while not self._stop.is_set():
            if self._pause.is_set():
                self.statusChanged.emit("paused")
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                continue
            self.statusChanged.emit("listening")
            try:
                wav, _dur = record_phrase(vad, cancel_event=self._cancel)
            except Exception as e:  # noqa: BLE001 - mic errors are fatal here
                self.errorOccurred.emit(f"Microphone error: {e}", True)
                break
            if self._cancel.is_set():
                self._cancel.clear()
                continue
            if not wav or len(wav) < MIN_WAV_BYTES:
                continue  # no speech detected, keep listening
            self.statusChanged.emit("transcribing")
            t0 = time.time()
            try:
                text = transcriber.transcribe(wav, language=self.app.language)
            except Exception as e:  # noqa: BLE001 - retry next phrase
                self.errorOccurred.emit(f"Transcription error: {e}", False)
                continue
            latency = (time.time() - t0) * 1000
            log.info("Transcribed in %.0fms: %r", latency, text)
            if text:
                try:
                    self.app.injector.type_text(text)
                except Exception as e:  # noqa: BLE001
                    self.errorOccurred.emit(f"Injection error: {e}", False)
                else:
                    self.phraseDone.emit(text, latency)


class PulseRing(QtWidgets.QWidget):
    """Expanding, fading rings shown around the play button while listening."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._t = 0.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1600)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_tick)

    def _on_tick(self, value: float) -> None:
        self._t = value
        self.update()

    def set_active(self, active: bool) -> None:
        running = self._anim.state() == QtCore.QAbstractAnimation.State.Running
        if active and not running:
            self._anim.start()
        elif not active and running:
            self._anim.stop()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D102
        if self._anim.state() != QtCore.QAbstractAnimation.State.Running:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        max_r = min(self.width(), self.height()) / 2 - 4
        for phase in (0.0, 0.5):
            tt = (self._t + phase) % 1.0
            radius = max_r * (0.58 + 0.42 * tt)
            alpha = int((1.0 - tt) * 75)
            p.setPen(QtGui.QPen(QtGui.QColor(104, 159, 99, alpha), 2.0))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(center, radius, radius)


class PlayPauseButton(QtWidgets.QAbstractButton):
    """Big round gradient button with a painted play/pause glyph.

    Doubles as a drag handle: press-and-move drags the window (emitting
    dragBy), press-and-release within a small threshold is a normal click.
    """

    SIZE = 132
    DRAG_THRESHOLD = 8  # px of movement before a press becomes a drag
    dragBy = QtCore.Signal(int, int)  # window delta x, y

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._playing = False
        self._hover = False
        self._scale = 1.0
        self._dragging = False
        self._press_global = QtCore.QPoint()
        self._last_global = QtCore.QPoint()
        self._bounce = QtCore.QVariantAnimation(self)
        self._bounce.setDuration(260)
        self._bounce.setStartValue(1.0)
        self._bounce.setKeyValueAt(0.0, 1.0)
        self._bounce.setKeyValueAt(0.45, 1.07)
        self._bounce.setEndValue(1.0)
        self._bounce.valueChanged.connect(self._on_bounce)
        self.clicked.connect(self._on_click)

    # ------------------------------------------------------- drag vs. click
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            self._press_global = event.globalPosition().toPoint()
            self._last_global = self._press_global
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            cur = event.globalPosition().toPoint()
            if not self._dragging and (
                (cur - self._press_global).manhattanLength() > self.DRAG_THRESHOLD
            ):
                self._dragging = True
                self.setDown(False)  # leave the pressed visual, suppress click
                # Wayland does not allow clients to position windows, so hand
                # the drag to the compositor (xdg_toplevel::move). Returns
                # False on platforms without support; fall back to manual
                # delta moves below.
                handle = self.window().windowHandle()
                if handle is not None and handle.startSystemMove():
                    event.accept()
                    return
            if self._dragging:
                self.dragBy.emit(cur.x() - self._last_global.x(),
                                 cur.y() - self._last_global.y())
                self._last_global = cur
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        if self._dragging:
            self._dragging = False
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _on_bounce(self, value: float) -> None:
        self._scale = value
        self.update()

    def _on_click(self) -> None:
        self._bounce.stop()
        self._bounce.start()

    def set_playing(self, playing: bool) -> None:
        if self._playing != playing:
            self._playing = playing
            self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:  # noqa: D102
        self._hover = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: D102
        self._hover = False
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D102
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        center = rect.center()

        # transform for press/hover/bounce scale
        p.translate(center)
        p.scale(self._scale, self._scale)
        p.translate(-center)

        disc = rect.adjusted(3, 3, -3, -3)

        # soft glow while playing; fade to alpha 0 exactly at the widget edge
        # (0.87 of the glow radius) so the square clip boundary never shows
        if self._playing:
            glow_r = disc.width() * 0.603  # 76px, inside the 66px half-width
            glow = QtGui.QRadialGradient(center, glow_r)
            glow.setColorAt(0.0, QtGui.QColor(104, 159, 99, 70))
            glow.setColorAt(0.8, QtGui.QColor(104, 159, 99, 28))
            glow.setColorAt(0.87, QtGui.QColor(104, 159, 99, 0))
            glow.setColorAt(1.0, QtGui.QColor(104, 159, 99, 0))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawEllipse(center, glow_r, glow_r)

        # main disc
        top = ACCENT_HOVER if self._hover else ACCENT
        bottom = ACCENT_PRESSED if self.isDown() else ACCENT_DARK
        grad = QtGui.QLinearGradient(disc.topLeft(), disc.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(top))
        grad.setColorAt(1.0, QtGui.QColor(bottom))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 55), 1.4))
        p.setBrush(grad)
        p.drawEllipse(disc)

        # top highlight
        hl = QtGui.QLinearGradient(disc.topLeft(), disc.bottomLeft())
        hl.setColorAt(0.0, QtGui.QColor(255, 255, 255, 42))
        hl.setColorAt(0.45, QtGui.QColor(255, 255, 255, 0))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(hl)
        p.drawEllipse(disc)

        # glyph (supersampled 4x for smooth edges at 1x scale)
        glyph_rect = QtCore.QRectF(center.x() - 40, center.y() - 44, 80, 88)
        pm = self._render_glyph(self._playing, glyph_rect)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(glyph_rect, pm, QtCore.QRectF(pm.rect()))

    def _render_glyph(self, playing: bool, rect: QtCore.QRectF) -> QtGui.QPixmap:
        """Paint the play/pause glyph at 4x into a pixmap, then scale down."""
        ss = 4
        pm = QtGui.QPixmap(int(rect.width() * ss), int(rect.height() * ss))
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        gp = QtGui.QPainter(pm)
        gp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        gp.scale(ss, ss)
        gp.translate(-rect.topLeft())
        self._paint_glyph(gp, playing)
        gp.end()
        return pm

    def _paint_glyph(self, p: QtGui.QPainter, playing: bool) -> None:
        center = self.rect().center()
        icon_color = QtGui.QColor(255, 255, 255, 245)
        if playing:
            bar_w, bar_h, gap, r = 13.0, 46.0, 15.0, 6.5
            x1 = center.x() - bar_w - gap / 2
            x2 = center.x() + gap / 2
            y = center.y() - bar_h / 2
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(icon_color)
            p.drawRoundedRect(QtCore.QRectF(x1, y, bar_w, bar_h), r, r)
            p.drawRoundedRect(QtCore.QRectF(x2, y, bar_w, bar_h), r, r)
        else:
            # nudge right: a play triangle's visual mass is on the left, so a
            # geometrically centered glyph reads as shifted left
            r = QtCore.QRectF(center.x() - 14, center.y() - 24, 36, 48)
            path = QtGui.QPainterPath()
            path.moveTo(r.left() + 2, r.top())
            path.lineTo(r.right(), r.center().y())
            path.lineTo(r.left() + 2, r.bottom())
            path.closeSubpath()
            p.setPen(
                QtGui.QPen(
                    icon_color,
                    11.0,
                    QtCore.Qt.PenStyle.SolidLine,
                    QtCore.Qt.PenCapStyle.RoundCap,
                    QtCore.Qt.PenJoinStyle.RoundJoin,
                )
            )
            p.setBrush(icon_color)
            p.drawPath(path)


class IconButton(QtWidgets.QAbstractButton):
    """Small round header button: 'close' (X) or 'gear' (settings sliders)."""

    def __init__(self, kind: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(30, 30)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._hover = False

    def enterEvent(self, event: QtCore.QEvent) -> None:  # noqa: D102
        self._hover = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: D102
        self._hover = False
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D102
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self._hover:
            if self.kind == "close":
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(255, 95, 107, 45))
            else:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(255, 255, 255, 22))
            p.drawEllipse(rect)
        color = QtGui.QColor("#dde3ee" if self._hover else "#8b93a5")
        c = rect.center()
        if self.kind == "close":
            pen = QtGui.QPen(color, 2.2, QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            d = 3.8
            p.drawLine(QtCore.QPointF(c.x() - d, c.y() - d),
                       QtCore.QPointF(c.x() + d, c.y() + d))
            p.drawLine(QtCore.QPointF(c.x() + d, c.y() - d),
                       QtCore.QPointF(c.x() - d, c.y() + d))
        else:  # settings sliders
            pen = QtGui.QPen(color, 2.2, QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            knob_r = 2.4
            for row, knob_x in enumerate((-4.0, 4.0, 0.0)):
                y = c.y() + (row - 1) * 6.5
                p.drawLine(QtCore.QPointF(c.x() - 6.5, y),
                           QtCore.QPointF(c.x() + 6.5, y))
                p.setBrush(color)
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawEllipse(QtCore.QPointF(c.x() + knob_x, y), knob_r, knob_r)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)


class DictationWindow(QtWidgets.QWidget):
    """The main frameless widget."""

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self._worker: DictationWorker | None = None
        self._state = "ready"
        self._fatal_error: str | None = None
        self._quitting = False
        self._drag_offset: QtCore.QPoint | None = None
        self._placed = False

        flags = QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool
        if self.settings.get("always_on_top", True):
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Groq Dictation")
        self.setWindowIcon(self._make_icon())

        self._build_ui()
        self._build_settings_panel()
        self._pos_save_timer = QtCore.QTimer(self)
        self._pos_save_timer.setSingleShot(True)
        self._pos_save_timer.setInterval(400)
        self._pos_save_timer.timeout.connect(self._save_pos)
        self._transcript_timer = QtCore.QTimer(self)
        self._transcript_timer.setSingleShot(True)
        self._transcript_timer.timeout.connect(self._dim_transcript)

    # ---------------------------------------------------------------- UI build

    def _make_icon(self) -> QtGui.QIcon:
        pm = QtGui.QPixmap(64, 64)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        grad = QtGui.QLinearGradient(0, 0, 64, 64)
        grad.setColorAt(0.0, QtGui.QColor(ACCENT))
        grad.setColorAt(1.0, QtGui.QColor(ACCENT_DARK))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(2, 2, 60, 60, 16, 16)
        # microphone glyph
        p.setBrush(QtGui.QColor(255, 255, 255, 245))
        p.drawRoundedRect(26, 12, 12, 24, 6, 6)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 245), 3.0,
                            QtCore.Qt.PenStyle.SolidLine,
                            QtCore.Qt.PenCapStyle.RoundCap))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawArc(QtCore.QRectF(18, 12, 28, 24), 20 * 16, 140 * 16)
        p.drawLine(QtCore.QPointF(32, 38), QtCore.QPointF(32, 46))
        p.drawLine(QtCore.QPointF(24, 50), QtCore.QPointF(40, 50))
        p.end()
        return QtGui.QIcon(pm)

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(26, 26, 26, 26)
        outer.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        self.card.setGraphicsEffect(shadow)
        outer.addWidget(self.card)

        lay = QtWidgets.QVBoxLayout(self.card)
        lay.setContentsMargins(18, 12, 18, 16)
        lay.setSpacing(8)

        # header
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(0)
        self.gear_btn = IconButton("gear")
        self.gear_btn.setToolTip("Settings")
        self.gear_btn.clicked.connect(self._toggle_settings)
        header.addWidget(self.gear_btn)
        header.addStretch(1)
        title = QtWidgets.QLabel("DICTATION")
        title.setObjectName("titleLabel")
        title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # letter spacing adds a trailing gap that skews optical centering
        title.setContentsMargins(0, 0, 3, 0)
        f = title.font()
        f.setPointSizeF(10.5)
        f.setWeight(QtGui.QFont.Weight.DemiBold)
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 2.2)
        title.setFont(f)
        header.addWidget(title)
        header.addStretch(1)
        self.close_btn = IconButton("close")
        self.close_btn.setToolTip("Stop and quit")
        self.close_btn.clicked.connect(self.close)
        header.addWidget(self.close_btn)
        lay.addLayout(header)

        # center: pulse ring + play button
        center = QtWidgets.QWidget()
        center.setFixedSize(206, 206)
        self.ring = PulseRing(center)
        self.ring.setGeometry(0, 0, 206, 206)
        self.play_btn = PlayPauseButton(center)
        self.play_btn.setGeometry(37, 37, PlayPauseButton.SIZE, PlayPauseButton.SIZE)
        self.play_btn.clicked.connect(self._on_toggle)
        self.play_btn.dragBy.connect(self._drag_by)
        center_layout = QtWidgets.QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(center)
        center_layout.addStretch(1)
        lay.addLayout(center_layout)

        # status + hint
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(STATUS_MAX_W)
        f = self.status_label.font()
        f.setPointSizeF(13.5)
        f.setWeight(QtGui.QFont.Weight.DemiBold)
        self.status_label.setFont(f)
        lay.addWidget(self.status_label)

        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setMaximumWidth(TRANSCRIPT_MAX_W)
        lay.addWidget(self.hint_label)

        # transcript preview
        self.transcript_label = QtWidgets.QLabel()
        self.transcript_label.setObjectName("transcriptLabel")
        self.transcript_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.transcript_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.transcript_label.setFixedHeight(18)
        self.transcript_label.setMaximumWidth(TRANSCRIPT_MAX_W)
        lay.addWidget(self.transcript_label)

        # settings panel (hidden)
        self.settings_panel = QtWidgets.QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        self.settings_panel.setMaximumHeight(0)
        self._panel_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        self._panel_layout.setContentsMargins(10, 6, 10, 12)
        self._panel_layout.setSpacing(7)
        lay.addWidget(self.settings_panel)

        chevron_path, check_path = _write_glyph_assets()
        chevron_css = (
            f"QComboBox::down-arrow {{ image: url({chevron_path}); width: 13px; height: 13px; }}"
            if chevron_path else ""
        )
        check_css = (
            f"QCheckBox::indicator:checked {{ image: url({check_path}); }}"
            if check_path else ""
        )

        self.setStyleSheet(
            f"""
            QFrame#card {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {CARD_TOP}, stop:1 {CARD_BOTTOM});
                border-radius: 22px;
                border: 1px solid #1AFFFFFF;
            }}
            QFrame#settingsPanel {{
                background: {PANEL};
                border-radius: 14px;
                border: 1px solid #131518;
            }}
            QLabel#titleLabel {{ color: {DIM}; }}
            QLabel#statusLabel {{ color: {DIM}; }}
            QLabel#hintLabel {{ color: {FAINT}; font-size: 11px; }}
            QLabel#transcriptLabel {{ color: {FAINT}; font-size: 11px; font-style: italic; }}
            QLabel#fieldLabel {{ color: {DIM}; font-size: 11px; }}
            QComboBox {{
                background: {FIELD_BG}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 5px 26px 5px 10px; font-size: 11px;
            }}
            QComboBox:hover {{ border-color: {BORDER_HOVER}; }}
            QComboBox:focus {{ border-color: {BORDER_FOCUS}; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            {chevron_css}
            QComboBox QAbstractItemView {{
                background: {FIELD_BG}; color: {TEXT};
                selection-background-color: {ACCENT};
                selection-color: #ffffff;
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 4px; outline: 0;
            }}
            QComboBox QAbstractItemView::item {{ padding: 5px 8px; border-radius: 4px; }}
            QComboBox QLineEdit {{
                background: transparent; border: none; color: {TEXT};
                padding: 0; font-size: 11px;
            }}
            QLineEdit {{
                background: {FIELD_BG}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 5px 10px; font-size: 11px;
            }}
            QLineEdit:focus {{ border-color: {BORDER_FOCUS}; }}
            QCheckBox {{ color: {TEXT}; font-size: 11px; }}
            QCheckBox::indicator {{
                width: 15px; height: 15px; border-radius: 4px;
                border: 1px solid {BORDER}; background: {FIELD_BG};
            }}
            QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
            QCheckBox::indicator:checked {{
                background: {ACCENT}; border-color: {ACCENT};
            }}
            {check_css}
            QToolTip {{
                background: {FIELD_BG}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding: 4px 8px;
            }}
            """
        )

    def _build_settings_panel(self) -> None:
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        self._panel_layout.addLayout(grid)

        def field_label(text: str) -> QtWidgets.QLabel:
            lab = QtWidgets.QLabel(text)
            lab.setObjectName("fieldLabel")
            return lab

        self.provider_combo = QtWidgets.QComboBox()
        for label, value in PROVIDERS:
            self.provider_combo.addItem(label, value)
        self.provider_combo.setToolTip("Transcription backend")
        self.provider_combo.setMaximumWidth(FIELD_MAX_W)
        idx = self.provider_combo.findData(self.settings["provider"])
        self.provider_combo.setCurrentIndex(max(idx, 0))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        grid.addWidget(field_label("Provider"), 0, 0)
        grid.addWidget(self.provider_combo, 0, 1)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.setEditable(True)
        self.language_combo.setMaximumWidth(FIELD_MAX_W)
        self.language_combo.lineEdit().setPlaceholderText("Auto-detect")
        self.language_combo.setToolTip("Language code (e.g. en, ig, fr)")
        for label, value in LANGUAGES:
            self.language_combo.addItem(label, value)
        if self.settings["language"]:
            i = self.language_combo.findData(self.settings["language"])
            if i >= 0:
                self.language_combo.setCurrentIndex(i)
            else:
                self.language_combo.setEditText(self.settings["language"])
        self.language_combo.activated.connect(self._on_language_changed)
        self.language_combo.lineEdit().editingFinished.connect(self._on_language_changed)
        grid.addWidget(field_label("Language"), 1, 0)
        grid.addWidget(self.language_combo, 1, 1)

        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("Provider default")
        self.model_edit.setMaximumWidth(FIELD_MAX_W)
        self.model_edit.setText(self.settings["model"])
        self.model_edit.setToolTip("Override the model ID")
        self.model_edit.editingFinished.connect(self._on_model_changed)
        grid.addWidget(field_label("Model"), 2, 0)
        grid.addWidget(self.model_edit, 2, 1)

        self.ontop_check = QtWidgets.QCheckBox("Always on top")
        self.ontop_check.setChecked(bool(self.settings.get("always_on_top", True)))
        self.ontop_check.toggled.connect(self._on_ontop_changed)
        grid.addWidget(self.ontop_check, 3, 0, 1, 2)

        note = QtWidgets.QLabel("Provider and language apply on the next play.")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)
        self._panel_layout.addWidget(note)

    # --------------------------------------------------------------- settings

    def _save(self) -> None:
        save_settings(self.settings)

    def _on_provider_changed(self) -> None:
        self.settings["provider"] = self.provider_combo.currentData()
        self._save()

    def _on_language_changed(self) -> None:
        i = self.language_combo.currentIndex()
        text = self.language_combo.currentText().strip()
        if i >= 0 and text == self.language_combo.itemText(i):
            lang = self.language_combo.itemData(i) or ""
        else:
            lang = text
        self.settings["language"] = lang
        self._save()

    def _on_model_changed(self) -> None:
        self.settings["model"] = self.model_edit.text().strip()
        self._save()

    def _on_ontop_changed(self, on: bool) -> None:
        self.settings["always_on_top"] = on
        self._save()
        flag = QtCore.Qt.WindowType.WindowStaysOnTopHint
        if bool(self.windowFlags() & flag) != on:
            self.setWindowFlag(flag, on)
            self.show()

    def _toggle_settings(self) -> None:
        if self.settings_panel.maximumHeight() == 0:
            self._animate_panel(True)
        else:
            self._animate_panel(False)

    def _animate_panel(self, open_: bool) -> None:
        if hasattr(self, "_panel_anim") and self._panel_anim is not None:
            self._panel_anim.stop()
        target = self.settings_panel.sizeHint().height() if open_ else 0
        if open_:
            self.settings_panel.setVisible(True)
        self._panel_anim = QtCore.QVariantAnimation(self)
        self._panel_anim.setDuration(200)
        self._panel_anim.setStartValue(self.settings_panel.maximumHeight())
        self._panel_anim.setEndValue(target)
        self._panel_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._panel_anim.valueChanged.connect(
            lambda v: self.settings_panel.setMaximumHeight(int(v)))
        self._panel_anim.finished.connect(
            lambda: self.settings_panel.setVisible(open_))
        self._panel_anim.start()

    # ----------------------------------------------------------------- states

    def _set_state(self, state: str, status: str | None = None,
                   hint: str | None = None) -> None:
        self._state = state
        colors = {
            "ready": DIM,
            "listening": ACCENT,
            "transcribing": AMBER,
            "paused": DIM,
            "error": RED,
        }
        default_status = {
            "ready": "Ready",
            "listening": "Listening\u2026",
            "transcribing": "Transcribing\u2026",
            "paused": "Paused",
            "error": "Error",
        }
        default_hint = {
            "ready": "Press play, click the app you want to type into, and speak",
            "listening": "Speak now \u2013 a short pause ends each phrase",
            "transcribing": "Sending audio to the cloud",
            "paused": "Press play to resume",
            "error": "Check settings and the API key in .env",
        }
        text = status or default_status[state]
        self.status_label.setText(
            _elide(text, self.status_label.font(), STATUS_MAX_W - 6))
        self.status_label.setStyleSheet(f"color: {colors[state]};")
        self.hint_label.setText(hint or default_hint[state])
        self.ring.set_active(state == "listening")
        # glyph shows the NEXT action: bars while dictating, triangle when
        # paused or ready (standard media-button convention)
        self.play_btn.set_playing(state in ("listening", "transcribing"))
        self.play_btn.setToolTip(
            {
                "ready": "Start dictation",
                "listening": "Pause",
                "transcribing": "Pause",
                "paused": "Resume",
                "error": "Start dictation",
            }[state]
        )

    def _show_transcript(self, text: str, color: str, timeout_ms: int) -> None:
        self.transcript_label.setText(
            _elide(text, self.transcript_label.font(), TRANSCRIPT_MAX_W - 10))
        self.transcript_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-style: italic;")
        self._transcript_timer.start(timeout_ms)

    def _dim_transcript(self) -> None:
        self.transcript_label.setStyleSheet(
            f"color: {FAINT}; font-size: 11px; font-style: italic;")

    def _on_worker_status(self, status: str) -> None:
        self._set_state(status if status in ("listening", "transcribing", "paused") else "listening")

    def _on_phrase(self, text: str, latency: float) -> None:
        self._show_transcript(text, TEXT, 2200)
        log.info("Phrase injected (%0.fms): %r", latency, text)

    def _on_worker_error(self, message: str, fatal: bool) -> None:
        if fatal:
            self._fatal_error = message
            self._set_state("error", "Error", message)
        else:
            self._show_transcript(message, RED, 2600)

    def _on_worker_finished(self) -> None:
        if self._quitting:
            return
        if self._fatal_error:
            self._set_state("error", "Stopped", self._fatal_error)
        else:
            self._set_state("ready")

    # ------------------------------------------------------------- play/pause

    def _on_toggle(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            if self._worker.is_paused():
                self._worker.resume()
            else:
                self._worker.pause()
            return
        self._start()

    def _start(self) -> None:
        _load_env()
        provider = self.settings["provider"]
        model = (self.settings.get("model") or "").strip() or None
        language = (self.settings.get("language") or "").strip() or None
        try:
            transcriber = build_transcriber(provider, model=model)
        except RuntimeError as e:
            msg = str(e)
            if " not set" in msg:
                key = msg.split(" not set")[0]
                self._set_state(
                    "error",
                    "API key missing",
                    f"Add {key} to the .env file in the project folder, then press play again.",
                )
            else:
                self._set_state("error", "Error", msg)
            return
        engine = DictationApp(
            transcriber=transcriber,
            injector=Injector(method="auto"),
            vad=VADConfig(),
            language=language,
            auto_inject=False,  # the worker injects so it can report phrases
        )
        worker = DictationWorker(engine, self)
        worker.statusChanged.connect(self._on_worker_status)
        worker.phraseDone.connect(self._on_phrase)
        worker.errorOccurred.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._fatal_error = None
        worker.start()
        self._set_state("listening")

    def _stop_worker(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            if not self._worker.wait(8000):
                log.warning("Worker did not stop in time; terminating.")
                self._worker.terminate()
                self._worker.wait(1000)

    # ----------------------------------------------------------- window events

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: D102
        super().showEvent(event)
        if not self._placed:
            self._placed = True
            self._place_window()
            # No fade-in animation: the Wayland QPA ignores windowOpacity
            # ("This plugin does not support setting window opacity"), so the
            # widget simply appears - the only cross-platform-safe behavior.

    def _place_window(self) -> None:
        pos = self.settings.get("pos")
        screen = QtGui.QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        if pos and any(
            s.availableGeometry().contains(QtCore.QPoint(pos[0] + 40, pos[1] + 40))
            for s in QtGui.QGuiApplication.screens()
        ):
            self.move(pos[0], pos[1])
        else:
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)

    def _save_pos(self) -> None:
        if self._quitting:
            return
        self.settings["pos"] = [self.x(), self.y()]
        self._save()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:  # noqa: D102
        super().moveEvent(event)
        if self._placed and not self._quitting:
            self._pos_save_timer.start()

    def _drag_by(self, dx: int, dy: int) -> None:
        """Move the window by a delta (driven by play-button drags)."""
        self.move(self.x() + dx, self.y() + dy)
        self._pos_save_timer.start()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Wayland: compositor-driven move (clients cannot self-position).
            # Falls back to manual tracking where unsupported.
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        if self._drag_offset is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D102
        self._drag_offset = None
        self._save_pos()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: D102
        self._quitting = True
        self._save_pos()
        self._stop_worker()
        self.settings["pos"] = [self.x(), self.y()]
        save_settings(self.settings)
        event.accept()


def main(argv: list[str] | None = None) -> int:
    _load_env()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("groq-dictation")
    app.setStyle("Fusion")
    window = DictationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
