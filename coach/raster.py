"""Rasterize the SVG board to a PNG so it can be a clickable `gr.Image`.

We keep `board_svg` as the single renderer and rasterize it with cairosvg, at the
board's native WIDTH x HEIGHT -- so a pixel (x, y) in the image maps 1:1 onto the
SVG coordinate space, and `board_svg.point_at` can turn a click into a point.

cairosvg needs the native libcairo. On macOS + Homebrew, ctypes' find_library
doesn't search /opt/homebrew/lib, so we add it before importing cairosvg (a no-op
on Linux, where libcairo2 is on the default path -- see packages.txt for Spaces).
"""
import io
import os
import sys

if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/lib"):
    _existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if "/opt/homebrew/lib" not in _existing.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            p for p in (_existing, "/opt/homebrew/lib") if p)

import cairosvg
from PIL import Image

from engine.board import Board
from coach.board_svg import board_svg, WIDTH, HEIGHT

SCALE = 1   # render at native size: a bigger image reloads slowly and flashes white
            # on every gr.Image update. Crispness comes from displaying at native
            # size (no upscaling), not from more pixels. Callers divide clicks by SCALE.


def board_image(board: Board, dice: tuple[int, int] | None = None,
                highlight: set[int] | None = None,
                used: list[int] | None = None) -> Image.Image:
    """The board as a (WIDTH*SCALE) x (HEIGHT*SCALE) PIL image for a `gr.Image`.
    `.select` returns coords in this scaled space; divide by SCALE before
    `board_svg.point_at`."""
    png = cairosvg.svg2png(bytestring=board_svg(board, dice, highlight, used).encode(),
                           output_width=WIDTH * SCALE, output_height=HEIGHT * SCALE)
    return Image.open(io.BytesIO(png))
