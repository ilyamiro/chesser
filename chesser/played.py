"""Played generation: every board is reached by an actual game from the start.

The search runs in native/ (chesser-play); every returned line is replayed
here with python-chess, so each board's move list is independently checked.
"""

import subprocess

import chess
import numpy as np

from .games import native_binary
from .sprites import piece_option


def synthesize(tables, beam=256, max_plies=120, min_plies=20, plausibility=2, max_promotions=1,
               seed=0, threads=None):
    """tables: (R, 64, 13) costs. Returns per board (opts, error, board, SAN moves)."""
    cmd = [native_binary("chesser-play"), "--beam", str(beam), "--max-plies", str(max_plies),
           "--min-plies", str(min_plies), "--plausibility", str(plausibility),
           "--max-promotions", str(max_promotions), "--seed", str(seed)]
    if threads:
        cmd += ["--threads", str(threads)]
    data = np.uint32(len(tables)).tobytes() + np.ascontiguousarray(tables, dtype="<f4").tobytes()
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout.decode()

    results = [None] * len(tables)
    for line in out.splitlines():
        idx, _, moves = line.split("\t")
        board = chess.Board()
        san = []
        for uci in moves.split():
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise RuntimeError(f"board {idx}: illegal move {uci} in returned line")
            san.append(board.san(move))
            board.push(move)
        opts = [piece_option(board.piece_at(sq)) for sq in range(64)]
        table = tables[int(idx)]
        err = float(table[np.arange(64), opts].sum())
        results[int(idx)] = (opts, err, board, san)
    return results


def pgn_moves(san):
    """SAN moves as numbered movetext, e.g. '1. e4 e5 2. Nf3'."""
    parts = []
    for i, m in enumerate(san):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(m)
    return " ".join(parts)
