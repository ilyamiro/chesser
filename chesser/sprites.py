"""Piece sprites, colour themes and piece layers.

`sprite(piece_type, color, n)` gives a piece at square size n as a tone map:

    0 transparent  1 outline  2 fill  3 shadow  4 highlight  5 detail

From 10 pixels up it is the Lichess cburnett set (Colin M.L. Burnett, GPLv2+,
pieces/cburnett/) rasterized from SVG, with a guaranteed outline ring. Smaller
squares use hand-drawn 8x8 masters, like hinted pixel fonts, since no
rasterizer keeps pieces readable that small.
"""

import io
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image

TRANSPARENT, OUTLINE, FILL, SHADOW, HIGHLIGHT, DETAIL = range(6)

SVG_MIN_SIZE = 10

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

def _erode(mask):
    """4-neighbour erosion; pixels outside the canvas count as empty."""
    p = np.pad(mask, 1)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


PIECE_DIR = Path(__file__).parent / "pieces" / "cburnett"
_LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


@lru_cache(maxsize=None)
def svg_sprite(piece_type, color, n):
    """Tone map (n, n) of the Lichess cburnett piece, rasterized at size n.

    Rendered supersampled; a pixel belongs to the piece when it is at least 40%
    covered. The silhouette always gets a dark outline ring (Lichess's own stroke
    is thinner than a pixel at small sizes). Inside, white pieces keep their dark
    lines as outline colour, black pieces their light lines as detail.
    """
    import cairosvg

    ss = 4
    name = ("w" if color == chess.WHITE else "b") + _LETTER[piece_type] + ".svg"
    png = cairosvg.svg2png(url=str(PIECE_DIR / name), output_width=n * ss, output_height=n * ss)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float64) / 255
    alpha = rgba[..., 3].reshape(n, ss, n, ss).mean((1, 3))
    lum = (luminance(rgba[..., :3]) * rgba[..., 3]).reshape(n, ss, n, ss).mean((1, 3)) / np.maximum(alpha, 1e-9)
    body = alpha >= 0.4
    inner = body
    for _ in range(max(1, round(n / 24))):  # Lichess's stroke vanishes at small sizes: draw our own ring
        inner = _erode(inner)
    tones = np.where(body, OUTLINE, TRANSPARENT).astype(np.uint8)
    if color == chess.WHITE:
        tones[inner] = np.where(lum[inner] >= 0.5, FILL, OUTLINE)
    else:
        tones[inner] = np.where(lum[inner] < 0.5, FILL, DETAIL)
    return tones


@lru_cache(maxsize=None)
def sprite(piece_type, color, n):
    """Tone map (n, n) of uint8 for a piece at square size n."""
    if n >= SVG_MIN_SIZE:
        return svg_sprite(piece_type, color, n)
    master = np.array([[_PIXEL_TONES[c] for c in row] for row in PIXEL_8[piece_type]], dtype=np.uint8)
    idx = np.arange(n) * 8 // n
    return master[np.ix_(idx, idx)]


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
        tm = sprite(piece.piece_type, piece.color, n)
        mask[opt] = tm != TRANSPARENT
        pbase = t["white"] if piece.color == chess.WHITE else t["black"]
        for ps in range(piece_shades):
            fill[ps, opt] = fills[piece.color][ps]
            tones = piece_tones(piece.color, fill[ps, opt])
            tones[OUTLINE] = OUTLINE_LUM
            for tone, lum in tones.items():
                rgb[ps, opt][tm == tone] = shade(pbase, lum)
    return rgb, luminance(rgb), mask, fill
