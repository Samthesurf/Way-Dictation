#!/usr/bin/env python3
"""Generate the Way Dictation app icon PNGs (no dependencies).

Mirrors the runtime icon in src/gui.rs `make_icon()`: a green rounded
square with a diagonal gradient, a white microphone capsule, and an arc
ring. Renders at 4x supersampling and box-averages down for crisp edges.

Usage: python3 scripts/gen_icon.py [out_dir]
"""
import math
import os
import struct
import sys
import zlib

ACCENT_TOP = (0.41, 0.62, 0.39)   # #689F63
ACCENT_BOTTOM = (0.31, 0.47, 0.29)  # #4E774A
WHITE = (1.0, 1.0, 1.0)

# Geometry in the 64-unit design space of make_icon()
BASE = (2.0, 2.0, 62.0, 62.0, 16.0)     # rounded square: x0 y0 x1 y1 r
CAPSULE = (26.0, 12.0, 38.0, 36.0, 6.0)  # mic capsule
STEM = (30.5, 38.0, 33.5, 46.0, 1.5)     # mic stem
MIC_BASE = (24.0, 48.5, 40.0, 51.5, 1.5) # mic base bar
ARC_CENTER = (32.0, 24.0)                # arc ring center
ARC_R0, ARC_R1 = 12.5, 15.5
ARC_ANG = (20.0, 160.0)                  # degrees


def sd_rrect(px, py, x0, y0, x1, y1, r):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) / 2.0 - r, (y1 - y0) / 2.0 - r
    qx, qy = abs(px - cx) - hw, abs(py - cy) - hh
    return min(max(qx, qy), 0.0) + math.hypot(max(qx, 0.0), max(qy, 0.0)) - r


def cover(sd):
    return max(0.0, min(1.0, 0.5 - sd))


def sample(px, py):
    """Composite the icon at one point; returns (r, g, b, a) premultiplied."""
    cr = cg = cb = ca = 0.0

    def paint(a, color):
        nonlocal cr, cg, cb, ca
        if a > 0.0:
            inv = 1.0 - a
            cr = cr * inv + color[0] * a
            cg = cg * inv + color[1] * a
            cb = cb * inv + color[2] * a
            ca = ca * inv + a

    # green rounded square with a diagonal gradient
    t = min(1.0, max(0.0, ((px - 2.0) + (py - 2.0)) / 120.0))
    acc = tuple(
        ACCENT_TOP[i] + (ACCENT_BOTTOM[i] - ACCENT_TOP[i]) * t for i in range(3)
    )
    paint(cover(sd_rrect(px, py, *BASE)), acc)

    # mic capsule
    paint(cover(sd_rrect(px, py, *CAPSULE)), WHITE)
    # mic stem and base bar
    paint(cover(sd_rrect(px, py, *STEM)), WHITE)
    paint(cover(sd_rrect(px, py, *MIC_BASE)), WHITE)

    # arc ring (signed distance to the band [R0, R1]; only the band paints)
    dx, dy = px - ARC_CENTER[0], py - ARC_CENTER[1]
    d = math.hypot(dx, dy)
    if d > 0.1:
        ang = math.degrees(math.atan2(dy, dx))
        if ARC_ANG[0] <= ang <= ARC_ANG[1]:
            mid = (ARC_R0 + ARC_R1) / 2.0
            half = (ARC_R1 - ARC_R0) / 2.0
            paint(cover(abs(d - mid) - half), WHITE)

    return cr, cg, cb, ca


def render(size):
    """Render `size` px icon (1x; the SDF cover() already antialiases)."""
    scale = 64.0 / size  # design space is 64 units wide
    out = bytearray()
    for y in range(size):
        for x in range(size):
            px = (x + 0.5) * scale
            py = (y + 0.5) * scale
            cr, cg, cb, ca = sample(px, py)
            if ca > 0.0:
                out += bytes(
                    (
                        round(min(1.0, cr / ca) * 255.0),
                        round(min(1.0, cg / ca) * 255.0),
                        round(min(1.0, cb / ca) * 255.0),
                        round(ca * 255.0),
                    )
                )
            else:
                out += b"\x00\x00\x00\x00"
    return bytes(out)


def png(size, rgba):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter: none
        raw += rgba[y * stride : (y + 1) * stride]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "packaging/icons"
    os.makedirs(out_dir, exist_ok=True)
    for size in (64, 128, 256, 512):
        path = os.path.join(out_dir, f"way-dictation-{size}.png")
        with open(path, "wb") as f:
            f.write(png(size, render(size)))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
