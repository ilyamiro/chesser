//! Plays games from the starting position, steered toward an image.
//!
//! Usage: chesser-play [--beam N] [--max-plies N] [--min-plies N] [--plausibility N]
//!                     [--max-promotions N] [--threads N] [--seed N]  < tables
//!
//! stdin: u32 board count, then count * 64 * 13 f32 (little-endian) cost tables,
//! [board][square, a1 = 0][option: 0 empty, 1-6 white PNBRQK, 7-12 black].
//!
//! For each board a beam search over legal moves keeps the `beam` lines whose
//! positions match the table best, for up to `max_plies` half-moves. The best
//! position seen from `min_plies` on is printed with the moves reaching it:
//!
//!     board index \t cost \t UCI moves separated by spaces
//!
//! Plausibility, checked on every position of every line:
//!   0  legal moves only
//!   1  the side to move can never mate in one, so no mate is ever missed or
//!      allowed, and no game ends before the final position
//!   2  also no piece can be won outright (taken undefended, or by a cheaper
//!      piece)
//! Extra pieces from promotion are capped at `max_promotions` per side.

use std::{
    collections::HashSet,
    env,
    io::{self, Read, Write},
};

use rayon::prelude::*;
use shakmaty::{
    Bitboard, CastlingMode, Chess, Color, EnPassantMode, Move, Position, Role, uci::UciMove,
    zobrist::Zobrist64,
};

const OPTIONS: usize = 13;

struct Params {
    beam: usize,
    max_plies: usize,
    min_plies: usize,
    plausibility: u32,
    max_promotions: usize,
    seed: u64,
}

fn code(role: Role, color: Color) -> usize {
    role as usize + if color == Color::White { 0 } else { 6 }
}

/// Table cost of a position: the empty-board cost plus each piece's change to its square.
fn cost(pos: &Chess, table: &[f32], empty: f32) -> f32 {
    let mut c = empty;
    for (sq, piece) in pos.board().iter() {
        let s = sq.to_usize() * OPTIONS;
        c += table[s + code(piece.role, piece.color)] - table[s];
    }
    c
}

fn value(role: Role) -> u32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 100,
    }
}

fn has_mate_in_one(pos: &Chess) -> bool {
    pos.legal_moves().into_iter().any(|m| {
        let mut p = pos.clone();
        p.play_unchecked(m);
        p.is_checkmate()
    })
}

/// Can the side to move win a piece outright (taken undefended, or by a cheaper piece)?
fn piece_hangs(pos: &Chess) -> bool {
    pos.legal_moves().into_iter().any(|m| {
        let Some(captured) = m.capture() else { return false };
        if value(m.role()) < value(captured) {
            return true;
        }
        let mut p = pos.clone();
        p.play_unchecked(m);
        // After the capture it is the victim's side to move: can it take back?
        p.board().attacks_to(m.to(), p.turn(), p.board().occupied()).is_empty()
    })
}

fn promotions(pos: &Chess, color: Color) -> usize {
    let b = pos.board();
    let mine = b.by_color(color);
    let n = |bb: Bitboard| (bb & mine).count();
    let bishops = b.bishops() & mine;
    n(b.queens()).saturating_sub(1)
        + n(b.rooks()).saturating_sub(2)
        + n(b.knights()).saturating_sub(2)
        + (bishops & Bitboard::LIGHT_SQUARES).count().saturating_sub(1)
        + (bishops & Bitboard::DARK_SQUARES).count().saturating_sub(1)
}

/// Whether a position may appear in a line (the game goes on from it).
fn plausible(pos: &Chess, p: &Params) -> bool {
    if pos.legal_moves().is_empty() {
        return false; // checkmate or stalemate: the game would be over
    }
    if promotions(pos, Color::White) > p.max_promotions || promotions(pos, Color::Black) > p.max_promotions {
        return false;
    }
    if p.plausibility >= 1 && has_mate_in_one(pos) {
        return false;
    }
    if p.plausibility >= 2 && piece_hangs(pos) {
        return false;
    }
    true
}

struct Node {
    pos: Chess,
    id: usize, // index into the history of (parent id, move)
}

fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32
}

/// Best (cost, moves) for one board.
fn play(table: &[f32], p: &Params, seed: u64) -> (f32, Vec<Move>) {
    let empty: f32 = (0..64).map(|sq| table[sq * OPTIONS]).sum();
    let start = Chess::default();
    let mut history: Vec<(usize, Option<Move>)> = vec![(0, None)];
    let mut beam = vec![Node { pos: start, id: 0 }];
    let mut best = (f32::INFINITY, 0usize);
    let mut best_any = (f32::INFINITY, 0usize); // fallback if the beam dies before min_plies
    let mut rng = seed | 1;

    for ply in 1..=p.max_plies {
        // Expand every line by every legal move, then accept the best-scoring
        // distinct children; the costly plausibility checks run only on those.
        let mut children: Vec<(f32, usize, Move, Chess, f32)> = Vec::with_capacity(beam.len() * 40);
        for (bi, node) in beam.iter().enumerate() {
            for m in node.pos.legal_moves() {
                let mut child = node.pos.clone();
                child.play_unchecked(m);
                let c = cost(&child, table, empty);
                // A little noise keeps the beam from collapsing onto near-identical lines.
                let score = c * (1.0 + 0.02 * (xorshift(&mut rng) - 0.5));
                children.push((score, bi, m, child, c));
            }
        }
        children.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut seen: HashSet<Zobrist64> = HashSet::with_capacity(p.beam * 2);
        let mut next = Vec::with_capacity(p.beam);
        for (_, bi, m, child, c) in children {
            if next.len() == p.beam {
                break;
            }
            if !seen.insert(child.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)) || !plausible(&child, p) {
                continue;
            }
            history.push((beam[bi].id, Some(m)));
            let id = history.len() - 1;
            if ply >= p.min_plies && c < best.0 {
                best = (c, id);
            }
            if c < best_any.0 {
                best_any = (c, id);
            }
            next.push(Node { pos: child, id });
        }
        if next.is_empty() {
            break;
        }
        beam = next;
    }

    if best.0.is_infinite() {
        best = best_any;
    }
    let mut moves = Vec::new();
    let mut id = best.1;
    while let (parent, Some(m)) = history[id] {
        moves.push(m);
        id = parent;
    }
    moves.reverse();
    (best.0, moves)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut p = Params { beam: 256, max_plies: 120, min_plies: 20, plausibility: 2, max_promotions: 1, seed: 0 };
    let mut threads = 0usize;
    let mut i = 1;
    while i + 1 < args.len() {
        let v: u64 = args[i + 1].parse().expect("numeric option value");
        match args[i].as_str() {
            "--beam" => p.beam = v as usize,
            "--max-plies" => p.max_plies = v as usize,
            "--min-plies" => p.min_plies = v as usize,
            "--plausibility" => p.plausibility = v as u32,
            "--max-promotions" => p.max_promotions = v as usize,
            "--threads" => threads = v as usize,
            "--seed" => p.seed = v,
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }

    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let count = u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;
    let floats: Vec<f32> = input[4..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(floats.len(), count * 64 * OPTIONS, "table size mismatch");

    let results: Vec<(f32, Vec<Move>)> = (0..count)
        .into_par_iter()
        .map(|b| play(&floats[b * 64 * OPTIONS..(b + 1) * 64 * OPTIONS], &p, p.seed.wrapping_mul(1_000_003) + b as u64 + 1))
        .collect();

    let mut out = io::BufWriter::new(io::stdout().lock());
    for (b, (c, moves)) in results.iter().enumerate() {
        let uci: Vec<String> = moves.iter().map(|m| UciMove::from_move(*m, CastlingMode::Standard).to_string()).collect();
        writeln!(out, "{b}\t{c}\t{}", uci.join(" "))?;
    }
    Ok(())
}
