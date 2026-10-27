"""Image -> grid of chessboards, each square an n x n tile (8 by default)."""

import math
import random
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np
from PIL import Image

from . import legality
from .sprites import NUM_OPTIONS, OUTLINE_LUM, THEMES, luminance, option_piece, piece_layers, shade

KINGS = (6, 12)
NON_KING = [o for o in range(NUM_OPTIONS) if o not in KINGS]


def quadrant_map(n):
    """(n, n) quadrant index of each pixel: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right."""
    half = np.arange(n) >= n // 2
    return half[:, None] * 2 + half[None, :]


class Palette:
    """Pieces and square shades for a theme, plus what matching needs to know about them.

    A square's background is a base shade (one of `shades` levels in its band)
    with each 2x2 quadrant shifted by up to +-spread, so squares carry a little
    gradient while staying inside their band. Pieces are drawn on top.
    """

    def __init__(self, theme="mono", square_size=8, shades=12, light_band=(150, 235), dark_band=(25, 110),
                 piece_shades=5, white_band=(170, 255), black_band=(0, 85), min_contrast=30, spread=10):
        n = self.n = square_size
        self.theme = THEMES[theme]
        self.spread = spread
        self.bands = np.array([dark_band, light_band], dtype=np.float64)  # [square is light]
        self.bases = np.stack([np.linspace(*dark_band, shades), np.linspace(*light_band, shades)])
        self.rgb, lum, self.mask, fill = piece_layers(theme, n, piece_shades, white_band, black_band)
        self.qmap = quadrant_map(n)
        npix = np.bincount(self.qmap.ravel(), minlength=4)
        q1h = np.eye(4)[self.qmap]  # (n, n, 4)
        # Quadrant mean = a * background + c, per (piece shade, option, quadrant).
        self.a = np.einsum("oij,ijq->oq", ~self.mask, q1h) / npix
        self.c = np.einsum("poij,ijq->poq", lum * self.mask, q1h) / npix

        # Readability: the silhouette (outline or fill) must stand out from the square.
        b = self.bases[:, :, None, None]
        contrast = np.maximum(abs(OUTLINE_LUM - b), abs(fill[None, None] - b))  # (2, shades, P, 13)
        contrast[..., 0] = np.inf
        valid = contrast >= min_contrast
        for light in (0, 1):
            for opt in range(1, NUM_OPTIONS):
                if not valid[light, ..., opt].any():
                    valid[light, ..., opt].flat[contrast[light, ..., opt].argmax()] = True
        self.valid = valid

        # Darkest and brightest square averages the tiles can produce.
        am, cm = self.a.mean(1), self.c.mean(2)  # (13,), (P, 13)
        means = am[None, None, None, :, None] * self.bands[:, None, None, None, :] + cm[None, None, :, :, None]
        ok = valid.any(1)[:, None, :, :, None]
        self.tone_range = means[np.broadcast_to(ok, means.shape)].min(), means[np.broadcast_to(ok, means.shape)].max()


def cropped_fraction(w, h, boards_wide, boards_high):
    """Share of the picture lost when centre-cropping it to the grid's aspect ratio."""
    a, t = w / h, boards_wide / boards_high
    return 1 - min(a, t) / max(a, t)


def load_target(path, boards_wide, boards_high=None, square_size=8):
    """Load an image cropped (never stretched) to a whole number of boards.

    Without boards_high, both neighbouring board counts are tried and the one
    that crops away the least of the picture wins. The crop is centred and done
    at full resolution, before downscaling.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if boards_high is None:
        exact = boards_wide * h / w
        options = {max(1, math.floor(exact)), max(1, math.ceil(exact))}
        boards_high = min(options, key=lambda bh: cropped_fraction(w, h, boards_wide, bh))
    t = boards_wide / boards_high
    cw, ch = (round(h * t), h) if w / h > t else (w, round(w / t))
    left, top = (w - cw) // 2, (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    board_px = 8 * square_size
    img = img.resize((boards_wide * board_px, boards_high * board_px), Image.LANCZOS)
    return luminance(np.asarray(img)), boards_high, cropped_fraction(w, h, boards_wide, boards_high)


def fit_contrast(target, palette, low_pct=1.0, high_pct=99.0):
    """Stretch the image's tones onto the range the tiles can actually reproduce."""
    lo, hi = np.percentile(target, [low_pct, high_pct])
    t = np.clip((target - lo) / max(hi - lo, 1e-9), 0, 1)
    lo, hi = palette.tone_range
    return lo + t * (hi - lo)


def light_mask(rows, cols):
    """1 where an image-grid square is a light square (a1, bottom-left of each board, is dark)."""
    r, c = np.mgrid[0:rows, 0:cols]
    return ((c % 8) + (7 - r % 8)) % 2


class Choice:
    """Per image square and option: best piece shade and per-quadrant background luminance."""

    def __init__(self, piece_shade, background):
        self.piece_shade = piece_shade  # (H, W, 13)
        self.background = background    # (H, W, 13, 4)


def board_blur(values, sigma):
    """Gaussian blur of a (H, W) per-square map, separately inside each 8x8 board."""
    if sigma <= 0:
        return values
    H, W = values.shape
    r = max(1, int(np.ceil(2.5 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    v = values.reshape(H // 8, 8, W // 8, 8)
    for axis in (1, 3):
        pad = [(0, 0)] * 4
        pad[axis] = (r, r)
        p = np.pad(v, pad, mode="reflect" if r < 8 else "edge")
        v = sum(w * np.take(p, np.arange(i, i + 8), axis=axis) for i, w in enumerate(k))
    return v.reshape(H, W)


def square_costs(target, palette, smooth=0.5, smooth_radius=1.5):
    """Per-square cost of every option, already minimised over all shading.

    Returns cost (H, W, 13), a Choice, and the light mask (H, W), where H, W
    count squares. Cost is the mean squared error of the square's four quadrant
    means. For a given piece, piece shade and base shade, each quadrant's mean is
    linear in its background, so the best background per quadrant is solved in
    closed form and clipped to base +- spread (and to the square's band).
    Tiles that break the readability rule (palette.valid) are never chosen.

    Smoothing: each base shade also costs smooth * (base - reference)^2, where
    the reference is the photo's tone blurred over smooth_radius squares within
    the board and clamped to the square's band. Nearby squares of one colour
    therefore stay close unless the photo itself changes there.
    """
    n = palette.n
    H, W = target.shape[0] // n, target.shape[1] // n
    blocks = target.reshape(H, n, W, n).transpose(0, 2, 1, 3)
    q1h = np.eye(4)[palette.qmap]
    T = np.einsum("hwij,ijq->hwq", blocks, q1h) / q1h.sum((0, 1))  # (H, W, 4)
    light = light_mask(H, W)
    band = palette.bands[light]  # (H, W, 2)
    reference = np.clip(board_blur(T.mean(-1), smooth_radius), band[..., 0], band[..., 1])
    a = palette.a[None, None]  # (1, 1, 13, 4)
    has_bg = a > 0

    best = np.full((H, W, NUM_OPTIONS), np.inf)
    best_ps = np.zeros((H, W, NUM_OPTIONS), dtype=np.int64)
    best_bg = np.zeros((H, W, NUM_OPTIONS, 4))
    for si in range(palette.bases.shape[1]):
        base = palette.bases[light, si]  # (H, W)
        lo = np.maximum(base - palette.spread, band[..., 0])[..., None, None]
        hi = np.minimum(base + palette.spread, band[..., 1])[..., None, None]
        for ps in range(len(palette.c)):
            c = palette.c[ps][None, None]  # (1, 1, 13, 4)
            ideal = np.where(has_bg, (T[:, :, None, :] - c) / np.where(has_bg, a, 1), base[..., None, None])
            bg = np.clip(ideal, lo, hi)
            err = ((T[:, :, None, :] - c - a * bg) ** 2).mean(-1) + smooth * ((base - reference) ** 2)[..., None]
            err = np.where(palette.valid[light, si, ps], err, np.inf)
            better = err < best
            best[better] = err[better]
            best_ps[better] = ps
            best_bg[better] = bg[better]
    return best, Choice(best_ps, best_bg), light


def board_cost(cost, br, bc):
    """Slice one board's (64, 13) cost table, indexed by chess square (a1 = 0)."""
    block = cost[br * 8:(br + 1) * 8, bc * 8:(bc + 1) * 8]
    return block[::-1].reshape(64, NUM_OPTIONS)


def opts_to_board(opts, turn=None):
    board = chess.Board(None)
    for sq, opt in enumerate(opts):
        if opt:
            board.set_piece_at(sq, option_piece(opt))
    if turn is None:
        legality.set_turn(board)
    else:
        board.turn = turn
    return board


class BoardSearch:
    """Local search over one board that only ever visits legal positions."""

    def __init__(self, cost, seed, max_promotions=8):
        self.cost = cost.tolist()
        self.max_promotions = max_promotions
        self.rng = random.Random(seed)
        self.board = chess.Board(None)
        self.opts = [0] * 64
        # For each square, options ranked best-first (kings are only moved by swaps).
        self.ranked = [sorted(NON_KING, key=lambda o, sq=sq: self.cost[sq][o]) for sq in range(64)]

    def total(self):
        return sum(self.cost[sq][o] for sq, o in enumerate(self.opts))

    def _set(self, sq, opt):
        self.opts[sq] = opt
        if opt:
            self.board.set_piece_at(sq, option_piece(opt))
        else:
            self.board.remove_piece_at(sq)

    def try_changes(self, changes):
        """Apply [(square, option)] if the result is legal; return whether it was kept."""
        old = [(sq, self.opts[sq]) for sq, _ in changes]
        for sq, opt in changes:
            self._set(sq, opt)
        legality.set_turn(self.board)
        if legality.realistic(self.board, self.max_promotions) and legality.is_legal(self.board):
            return True
        for sq, opt in reversed(old):
            self._set(sq, opt)
        return False

    def place_kings(self):
        c = self.cost
        wk = sorted(range(64), key=lambda sq: c[sq][6] - c[sq][0])[:16]
        bk = sorted(range(64), key=lambda sq: c[sq][12] - c[sq][0])[:16]
        pairs = [(c[w][6] - c[w][0] + c[b][12] - c[b][0], w, b)
                 for w in wk for b in bk if chess.square_distance(w, b) > 1]
        _, w, b = min(pairs)
        self._set(w, 6)
        self._set(b, 12)

    def delta(self, changes):
        return sum(self.cost[sq][o] - self.cost[sq][self.opts[sq]] for sq, o in changes)

    def greedy(self):
        while True:
            cands = sorted(
                (self.cost[sq][o] - self.cost[sq][self.opts[sq]], sq, o)
                for sq in range(64) if self.opts[sq] not in KINGS
                for o in self.ranked[sq][:6] if self.cost[sq][o] < self.cost[sq][self.opts[sq]])
            improved = False
            for _, sq, o in cands:
                if self.opts[sq] not in KINGS and self.cost[sq][o] < self.cost[sq][self.opts[sq]]:
                    improved |= self.try_changes([(sq, o)])
            if not improved:
                return

    def propose(self):
        rng, opts = self.rng, self.opts
        r = rng.random()
        if r < 0.6:
            sq = rng.randrange(64)
            if opts[sq] in KINGS:
                return None
            o = rng.choice(self.ranked[sq][:4])
            return [(sq, o)] if o != opts[sq] else None
        if r < 0.8:
            a, b = rng.randrange(64), rng.randrange(64)
            return [(a, opts[b]), (b, opts[a])] if opts[a] != opts[b] else None
        # Move material: empty one square and fill another (frees piece budget).
        occupied = [sq for sq in range(64) if opts[sq] and opts[sq] not in KINGS]
        if not occupied:
            return None
        a, b = rng.choice(occupied), rng.randrange(64)
        if a == b or opts[b] in KINGS:
            return None
        return [(a, 0), (b, rng.choice(self.ranked[b][:4]))]

    def anneal(self, iters):
        gaps = sorted(self.cost[sq][self.ranked[sq][1]] - self.cost[sq][self.ranked[sq][0]] for sq in range(64))
        t0 = max(gaps[32] * 2, 1e-6)
        best, best_opts = self.total(), list(self.opts)
        current = best
        for i in range(iters):
            temp = t0 * (0.01 ** (i / max(iters - 1, 1)))
            changes = self.propose()
            if not changes:
                continue
            d = self.delta(changes)
            if d > 0 and self.rng.random() >= math.exp(-d / temp):
                continue
            if self.try_changes(changes):
                current += d
                if current < best - 1e-9:
                    best, best_opts = current, list(self.opts)
        self.board = opts_to_board(best_opts)
        self.opts = list(best_opts)


def synthesize(args):
    cost, seed, iters, max_promotions = args
    search = BoardSearch(cost, seed, max_promotions)
    search.place_kings()
    search.greedy()
    search.anneal(iters)
    search.greedy()
    return search.opts, search.total()


def synthesize_all(cost, boards_high, boards_wide, iters, seed, workers, max_promotions=1):
    jobs = [(board_cost(cost, br, bc), seed * 100003 + br * 1000 + bc, iters, max_promotions)
            for br in range(boards_high) for bc in range(boards_wide)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(synthesize, jobs, chunksize=1))


def render(palette, grid, choice, light, scale=1, grid_px=0, grid_color=(20, 20, 20), grid_opacity=0.45):
    """grid: (H, W) option per image square. Returns an RGB PIL image.

    grid_px > 0 inserts separator lines of that many (output) pixels between
    boards and around the edge. They are added, never drawn over squares, and
    are semi-transparent: grid_color blended over the average of the pixels on
    either side of the line.
    """
    H, W = grid.shape
    n = palette.n
    sel = grid[..., None]
    ps = np.take_along_axis(choice.piece_shade, sel, 2)[..., 0]  # (H, W)
    bg = np.take_along_axis(choice.background, sel[..., None], 2)[:, :, 0]  # (H, W, 4)
    bg_lum = bg[:, :, palette.qmap]  # (H, W, n, n)
    base_rgb = np.where(light[..., None], 1, 0)
    colors = np.array([palette.theme["dark"], palette.theme["light"]])[base_rgb[..., 0]]  # (H, W, 3)
    tiles = shade(colors[:, :, None, None, :], bg_lum)
    piece = palette.mask[grid][..., None]  # (H, W, n, n, 1)
    tiles = np.where(piece, palette.rgb[ps, grid], tiles)
    img = tiles.transpose(0, 2, 1, 3, 4).reshape(H * n, W * n, 3)
    if scale > 1:
        img = img.repeat(scale, 0).repeat(scale, 1)
    if grid_px <= 0:
        return Image.fromarray(img)
    size, bh, bw, g = 8 * n * scale, H // 8, W // 8, grid_px
    out = np.zeros((bh * size + (bh + 1) * g, bw * size + (bw + 1) * g, 3))
    for br in range(bh):
        for bc in range(bw):
            y, x = g + br * (size + g), g + bc * (size + g)
            out[y:y + size, x:x + size] = img[br * size:(br + 1) * size, bc * size:(bc + 1) * size]
    color = np.asarray(grid_color, dtype=np.float64)
    for i in range(bw + 1):  # vertical lines first; horizontal ones then blend over the crossings
        x = i * (size + g)
        sides = ([out[:, x - 1]] if i > 0 else []) + ([out[:, x + g]] if i < bw else [])
        out[:, x:x + g] = ((1 - grid_opacity) * np.mean(sides, 0) + grid_opacity * color)[:, None]
    for i in range(bh + 1):
        y = i * (size + g)
        sides = ([out[y - 1]] if i > 0 else []) + ([out[y + g]] if i < bh else [])
        out[y:y + g] = ((1 - grid_opacity) * np.mean(sides, 0) + grid_opacity * color)[None]
    return Image.fromarray(np.round(out).astype(np.uint8))


def place_board(grid, br, bc, opts):
    """Write a board's 64 options (a1 = 0) into the image-square grid."""
    grid[br * 8:(br + 1) * 8, bc * 8:(bc + 1) * 8] = np.asarray(opts).reshape(8, 8)[::-1]
