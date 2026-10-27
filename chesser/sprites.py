"""Scalable multi-tone piece sprites, colour themes and tile building.

Each piece is defined once as vector shapes in a unit square (x right, y down).
`sprite(piece_type, n)` rasterizes it at any size n into a tone map:

    0 transparent  1 outline  2 fill  3 shadow  4 highlight  5 detail

* the silhouette is drawn supersampled and thresholded on coverage;
* the outline is the silhouette's outer ring, about n/12 pixels thick;
* light comes from the top left: interior pixels next to the right/bottom edge
  become shadow, those next to the top/left edge highlight (from n >= 12, where
  there is room for a lit side without eating the fill);
* detail shapes (the bishop's slit, the knight's eye, the king's cross...) are
  drawn in their own tone inside the silhouette.

Below 12 pixels no rasterizer makes readable pieces, so (like hinted pixel
fonts) small sizes use the hand-drawn 8x8 masters in PIXEL_8, scaled.
"""

from functools import lru_cache

import chess
import numpy as np
from PIL import Image, ImageDraw

TRANSPARENT, OUTLINE, FILL, SHADOW, HIGHLIGHT, DETAIL = range(6)

# Shapes: (kind, params, add). kind: "poly" [(x, y), ...], "ellipse" (cx, cy, rx, ry),
# "rect" (x0, y0, x1, y1), "line" (x0, y0, x1, y1, width). add=False cuts the shape out.
_BASE = ("rect", (0.16, 0.80, 0.84, 0.94), True)

SHAPES = {
    chess.PAWN: {
        "body": [
            ("ellipse", (0.50, 0.30, 0.15, 0.15), True),
            ("poly", [(0.40, 0.40), (0.60, 0.40), (0.70, 0.80), (0.30, 0.80)], True),
            ("rect", (0.22, 0.74, 0.78, 0.94), True),
        ],
        "detail": [],
    },
    chess.KNIGHT: {
        "body": [
            ("poly", [(0.24, 0.94), (0.84, 0.94), (0.82, 0.58), (0.74, 0.30), (0.60, 0.15),
                      (0.52, 0.04), (0.44, 0.16), (0.30, 0.26), (0.12, 0.50), (0.18, 0.62),
                      (0.30, 0.60), (0.46, 0.50), (0.32, 0.80), (0.24, 0.82)], True),
        ],
        "detail": [
            ("ellipse", (0.44, 0.30, 0.07, 0.07), True),       # eye
            ("line", (0.64, 0.22, 0.76, 0.62, 0.07), True),     # mane
        ],
    },
    chess.BISHOP: {
        "body": [
            ("ellipse", (0.50, 0.10, 0.08, 0.08), True),
            ("ellipse", (0.50, 0.44, 0.22, 0.28), True),
            ("rect", (0.32, 0.66, 0.68, 0.80), True),
            _BASE,
        ],
        "detail": [
            ("line", (0.60, 0.28, 0.44, 0.50, 0.09), True),     # mitre slit
            ("rect", (0.30, 0.74, 0.70, 0.80), True),           # collar
        ],
    },
    chess.ROOK: {
        "body": [
            ("rect", (0.18, 0.12, 0.82, 0.38), True),
            ("rect", (0.34, 0.12, 0.44, 0.24), False),          # crenels
            ("rect", (0.56, 0.12, 0.66, 0.24), False),
            ("rect", (0.28, 0.36, 0.72, 0.82), True),
            _BASE,
        ],
        "detail": [
            ("rect", (0.28, 0.38, 0.72, 0.44), True),           # rim under the battlements
        ],
    },
    chess.QUEEN: {
        "body": [
            ("poly", [(0.10, 0.20), (0.32, 0.50), (0.36, 0.12), (0.50, 0.44), (0.64, 0.12),
                      (0.68, 0.50), (0.90, 0.20), (0.78, 0.80), (0.22, 0.80)], True),
            ("ellipse", (0.10, 0.18, 0.07, 0.07), True),
            ("ellipse", (0.36, 0.10, 0.07, 0.07), True),
            ("ellipse", (0.64, 0.10, 0.07, 0.07), True),
            ("ellipse", (0.90, 0.18, 0.07, 0.07), True),
            _BASE,
        ],
        "detail": [
            ("rect", (0.22, 0.66, 0.78, 0.73), True),           # crown band
        ],
    },
    chess.KING: {
        "body": [
            ("rect", (0.44, 0.02, 0.56, 0.30), True),           # cross
            ("rect", (0.34, 0.08, 0.66, 0.19), True),
            ("ellipse", (0.50, 0.46, 0.34, 0.20), True),
            ("poly", [(0.18, 0.46), (0.82, 0.46), (0.72, 0.80), (0.28, 0.80)], True),
            _BASE,
        ],
        "detail": [
            ("rect", (0.47, 0.04, 0.53, 0.28), True),
            ("rect", (0.37, 0.11, 0.63, 0.16), True),
            ("rect", (0.26, 0.66, 0.74, 0.72), True),           # girdle
        ],
    },
}

# Hand-drawn 8x8 masters: '.' transparent, 'o' outline, '#' fill, 's' shadow, 'd' detail.
PIXEL_8 = {
    chess.PAWN: ["........", "...oo...", "..o#so..", "..o#so..", "...oo...", "..o#so..", ".o###so.", ".oooooo."],
    chess.KNIGHT: ["....o...", "...o#o..", "..o#d#o.", ".o####so", "o##oo#so", ".oo.o#so", "..o###so", ".ooooooo"],
    chess.BISHOP: ["...oo...", "..o#so..", ".o#d#so.", ".o##dso.", "..o#so..", "..oddo..", ".o###so.", ".oooooo."],
    chess.ROOK: ["........", "oo.oo.oo", "o#####so", ".oddddo.", ".o###so.", ".o###so.", "o#####so", "oooooooo"],
    chess.QUEEN: [".o.oo.o.", "o#o##o#o", "o#####so", ".o###so.", ".oddddo.", "..o#so..", ".o###so.", "oooooooo"],
    chess.KING: ["...oo...", "..oooo..", "...oo...", ".oo##oo.", "o#####so", "o#ddddso", ".o###so.", ".oooooo."],
}
_PIXEL_TONES = {".": TRANSPARENT, "o": OUTLINE, "#": FILL, "s": SHADOW, "d": DETAIL}

SUPERSAMPLE = 8


def _coverage(shapes, n):
    s = n * SUPERSAMPLE
    img = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(img)
    for kind, p, add in shapes:
        val = 255 if add else 0
        if kind == "poly":
            draw.polygon([(x * s, y * s) for x, y in p], fill=val)
        elif kind == "ellipse":
            cx, cy, rx, ry = p
            draw.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=val)
        elif kind == "rect":
            x0, y0, x1, y1 = p
            draw.rectangle([x0 * s, y0 * s, x1 * s - 1, y1 * s - 1], fill=val)
        elif kind == "line":
            x0, y0, x1, y1, w = p
            draw.line([(x0 * s, y0 * s), (x1 * s, y1 * s)], fill=val, width=max(1, round(w * s)))
    a = np.asarray(img, dtype=np.float64) / 255
    return a.reshape(n, SUPERSAMPLE, n, SUPERSAMPLE).mean((1, 3))


def _erode(mask):
    """4-neighbour erosion; pixels outside the canvas count as empty."""
    p = np.pad(mask, 1)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


@lru_cache(maxsize=None)
def sprite(piece_type, n):
    """Tone map (n, n) of uint8 for a piece at square size n."""
    if n < 12:
        master = np.array([[_PIXEL_TONES[c] for c in row] for row in PIXEL_8[piece_type]], dtype=np.uint8)
        idx = np.arange(n) * 8 // n
        return master[np.ix_(idx, idx)]
    spec = SHAPES[piece_type]
    body = _coverage(spec["body"], n) >= 0.5
    inner = body
    for _ in range(max(1, round(n / 12))):
        inner = _erode(inner)
    tones = np.where(body, OUTLINE, TRANSPARENT).astype(np.uint8)
    tones[inner] = FILL

    t = max(1, round(n / 10))
    p = np.pad(inner, t)
    right_edge = np.zeros_like(inner)
    left_edge = np.zeros_like(inner)
    for d in range(1, t + 1):
        right_edge |= ~p[t:-t or None, t + d:p.shape[1] - t + d] | ~p[t + d:p.shape[0] - t + d, t:-t or None]
        left_edge |= ~p[t:-t or None, t - d:p.shape[1] - t - d] | ~p[t - d:p.shape[0] - t - d, t:-t or None]
    # Only shade pixels that have fill on the other side, so thin parts stay solid.
    tones[inner & right_edge & ~left_edge] = SHADOW
    if n >= 12:
        tones[inner & left_edge & ~right_edge] = HIGHLIGHT

    if spec["detail"]:
        detail = _coverage(spec["detail"], n) >= 0.5
        tones[detail & inner] = DETAIL
    return tones


# Each theme: light square, dark square, white piece base, black piece base.
THEMES = {
    "mono": {"light": (205, 205, 205), "dark": (85, 85, 85), "white": (255, 255, 255), "black": (0, 0, 0)},
    "classic": {"light": (238, 238, 210), "dark": (118, 150, 86), "white": (255, 255, 255), "black": (0, 0, 0)},
    "wood": {"light": (240, 217, 181), "dark": (181, 136, 99), "white": (255, 250, 240), "black": (20, 15, 10)},
}

# Option index for a square: 0 = empty, 1..6 = white P N B R Q K, 7..12 = black p n b r q k.
NUM_OPTIONS = 13


def option_piece(opt):
    if opt == 0:
        return None
    color = chess.WHITE if opt <= 6 else chess.BLACK
    return chess.Piece((opt - 1) % 6 + 1, color)


def piece_option(piece):
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


def luminance(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def shade(rgb, lum):
    """Tint colour(s) toward white or black until they have the given luminance.

    Broadcasts: rgb (..., 3) and lum (...) may be arrays.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    lum = np.asarray(lum, dtype=np.float64)[..., None]
    base = luminance(rgb)[..., None]
    up = rgb + (255.0 - rgb) * (lum - base) / np.maximum(255.0 - base, 1e-9)
    down = rgb * lum / np.maximum(base, 1e-9)
    return np.clip(np.round(np.where(lum >= base, up, down)), 0, 255).astype(np.uint8)


def piece_tones(color, fill_lum):
    """Luminance of fill, shadow, highlight and detail for a piece of the given fill."""
    if color == chess.WHITE:
        return {FILL: fill_lum, SHADOW: fill_lum - 55, HIGHLIGHT: min(255, fill_lum + 40),
                DETAIL: max(0, fill_lum - 130)}
    return {FILL: fill_lum, SHADOW: max(0, fill_lum - 30), HIGHLIGHT: fill_lum + 55,
            DETAIL: min(255, fill_lum + 120)}


# Outlines are always dark: the contrast rule already keeps every piece's fill
# clear of its square, and black pieces get their shape from light details.
OUTLINE_LUM = 15


def piece_layers(theme, n, piece_shades, white_band, black_band):
    """Piece pixels at square size n, separate from the square behind them.

    Returns rgb (P, 13, n, n, 3) and lum (P, 13, n, n) per piece shade and
    option, mask (13, n, n) marking piece pixels (option 0, empty, has none),
    and fill (P, 13), each piece's fill luminance.
    """
    t = THEMES[theme]
    rgb = np.zeros((piece_shades, NUM_OPTIONS, n, n, 3), dtype=np.uint8)
    mask = np.zeros((NUM_OPTIONS, n, n), dtype=bool)
    fill = np.zeros((piece_shades, NUM_OPTIONS))
    fills = {chess.WHITE: np.linspace(*white_band, piece_shades), chess.BLACK: np.linspace(*black_band, piece_shades)}
    for opt in range(1, NUM_OPTIONS):
        piece = option_piece(opt)
        tm = sprite(piece.piece_type, n)
        mask[opt] = tm != TRANSPARENT
        pbase = t["white"] if piece.color == chess.WHITE else t["black"]
        for ps in range(piece_shades):
            fill[ps, opt] = fills[piece.color][ps]
            tones = piece_tones(piece.color, fill[ps, opt])
            tones[OUTLINE] = OUTLINE_LUM
            for tone, lum in tones.items():
                rgb[ps, opt][tm == tone] = shade(pbase, lum)
    return rgb, luminance(rgb), mask, fill
