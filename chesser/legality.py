"""Chess position legality checks.

python-chess's Board.status() covers the basic rules (one king each, kings not
adjacent, no pawns on the back ranks, side not to move not in check, possible
checks, at most 16 pieces / 8 pawns per side). On top of that we enforce
retrograde-style accounting that a position reachable from the initial setup
must satisfy:

* promotions: every piece beyond the starting set (counting light- and
  dark-squared bishops separately) needs a promoted pawn, so
  pawns + promotions <= 8;
* trapped bishops: a bishop of the c1/f1/c8/f8 colour that is not on its home
  square while both pawns guarding that home are unmoved must be promoted;
* pawn captures: pawns change file only by capturing, so the minimum number of
  file shifts needed to explain the pawn structure must not exceed the number
  of enemy pieces missing from the board.

These are necessary conditions, not a full proof of reachability (that would
need complete retrograde analysis), but they reject the usual "impossible
position" tells.
"""

import chess

# (home bishop square, square colour mask, the two pawn squares that wall it in)
_BISHOP_HOMES = {
    chess.WHITE: [
        (chess.C1, chess.BB_DARK_SQUARES, (chess.B2, chess.D2)),
        (chess.F1, chess.BB_LIGHT_SQUARES, (chess.E2, chess.G2)),
    ],
    chess.BLACK: [
        (chess.C8, chess.BB_LIGHT_SQUARES, (chess.B7, chess.D7)),
        (chess.F8, chess.BB_DARK_SQUARES, (chess.E7, chess.G7)),
    ],
}


def min_pawn_file_shifts(files):
    """Minimum total file distance to assign each pawn a distinct starting file."""
    files = sorted(files)
    inf = float("inf")
    prev = [0] * 9  # prev[j]: first i pawns matched within start files 0..j-1
    for f in files:
        cur = [inf] * 9
        for j in range(1, 9):
            cur[j] = min(cur[j - 1], prev[j - 1] + abs(f - (j - 1)))
        prev = cur
    return prev[8]


def promotions_needed(board, color):
    pieces = board.occupied_co[color]
    queens = chess.popcount(board.queens & pieces)
    rooks = chess.popcount(board.rooks & pieces)
    knights = chess.popcount(board.knights & pieces)
    bishops = board.bishops & pieces
    light_b = chess.popcount(bishops & chess.BB_LIGHT_SQUARES)
    dark_b = chess.popcount(bishops & chess.BB_DARK_SQUARES)

    # A bishop can't have left a home walled in by two unmoved pawns.
    pawns = board.pawns & pieces
    for home, mask, (p1, p2) in _BISHOP_HOMES[color]:
        walled = pawns & chess.BB_SQUARES[p1] and pawns & chess.BB_SQUARES[p2]
        if walled and not bishops & chess.BB_SQUARES[home]:
            # The original bishop is gone; every bishop of this colour is promoted.
            if mask == chess.BB_LIGHT_SQUARES:
                light_b += 1
            else:
                dark_b += 1

    return (max(0, queens - 1) + max(0, rooks - 2) + max(0, knights - 2)
            + max(0, light_b - 1) + max(0, dark_b - 1))


def set_turn(board):
    """Make the side in check (if any) the side to move."""
    king = board.king(chess.BLACK)
    board.turn = chess.BLACK if king is not None and board.is_attacked_by(chess.WHITE, king) else chess.WHITE


def problems(board):
    """Return a list of reasons the position is illegal (empty if it passes)."""
    reasons = []
    status = board.status()
    if status != chess.STATUS_VALID:
        reasons.append(f"python-chess status {status!r}")
        return reasons

    for color in (chess.WHITE, chess.BLACK):
        name = chess.COLOR_NAMES[color]
        pawns = board.pawns & board.occupied_co[color]
        n_pawns = chess.popcount(pawns)
        promos = promotions_needed(board, color)
        if n_pawns + promos > 8:
            reasons.append(f"{name}: {n_pawns} pawns + {promos} promotions > 8")
        shifts = min_pawn_file_shifts([chess.square_file(sq) for sq in chess.scan_forward(pawns)])
        missing = 16 - chess.popcount(board.occupied_co[not color])
        if shifts > missing:
            reasons.append(f"{name}: pawns need {shifts} captures but only {missing} enemy pieces are missing")
    return reasons


def is_legal(board):
    return not problems(board)


def realistic(board, max_promotions):
    """Stricter than legal: each side may have at most `max_promotions` pieces
    beyond its starting set (so no armies of eight rooks)."""
    return all(promotions_needed(board, color) <= max_promotions for color in (chess.WHITE, chess.BLACK))
