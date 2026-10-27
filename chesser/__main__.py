"""Command line: convert images, download and index Lichess games, verify an output."""

import argparse
import json
import os
import shutil
import time

import chess
import numpy as np

from . import download, engine, legality, played
from .config import data_dir
from .games import GameIndex, build_shard
from .sprites import THEMES


def band(text):
    lo, hi = (float(x) for x in text.split(","))
    return lo, hi


def prepare(path, palette, a):
    """Load, score and synthesize one image."""
    t0 = time.time()
    target, boards_high, cropped = engine.load_target(path, a.boards, a.boards_high, a.square_size)
    target = engine.fit_contrast(target, palette)
    cost, choice, light = engine.square_costs(target, palette, a.smooth, a.smooth_radius)
    coords = [(br, bc) for br in range(boards_high) for bc in range(a.boards)]
    board_px = 8 * a.square_size
    entries = []
    if a.generate == "played":
        tables = np.stack([engine.board_cost(cost, br, bc) for br, bc in coords])
        results = played.synthesize(tables, a.beam, a.max_plies, a.min_plies, a.plausibility,
                                    a.max_promotions, a.seed, a.workers)
        for (br, bc), (opts, err, board, san) in zip(coords, results):
            entries.append({"row": br, "col": bc, "source": "played", "opts": opts, "turn": board.turn,
                            "synthetic_error": err, "error": err, "moves": played.pgn_moves(san)})
    else:
        results = engine.synthesize_all(cost, boards_high, a.boards, a.iters, a.seed, a.workers, a.max_promotions)
        for (br, bc), (opts, err) in zip(coords, results):
            board = engine.opts_to_board(opts)
            entries.append({"row": br, "col": bc, "source": "synthetic", "opts": opts, "turn": board.turn,
                            "synthetic_error": err, "error": err})
    print(f"{path}: {a.boards}x{boards_high} boards ({a.boards * board_px}x{boards_high * board_px} px), "
          f"{cropped:.1%} cropped, synthesized in {time.time() - t0:.1f}s")
    return {"path": path, "boards_high": boards_high, "cost": cost, "choice": choice, "light": light,
            "coords": coords, "entries": entries}


def assign_games(img, matches, index, a):
    """Replace synthetic boards with real game positions that reach the threshold."""
    # Fit = synthetic error / game error (1.0 = as good as the free-form optimum).
    # A small floor keeps near-flat regions from punishing every game.
    floor = 64 * a.fit_floor
    entries = img["entries"]
    synth = np.array([e["synthetic_error"] for e in entries])
    best_fit = np.array([(synth[r] + floor) / (m[0][0] + floor) if m else 0.0 for r, m in enumerate(matches)])
    used = set()  # positions by content: the same position can appear in several shards
    for r in np.argsort(-best_fit):
        for err, (si, slot) in matches[r]:
            if (synth[r] + floor) / (err + floor) < a.threshold:
                break
            shard = index.shards[si]
            codes = shard.codes([slot])[0]
            if codes.tobytes() in used or np.count_nonzero(codes) < a.game_min_pieces:
                continue
            used.add(codes.tobytes())
            entries[r].update(source="game", opts=codes.tolist(), turn=shard.turn(slot),
                              error=err, game=shard.info(slot))
            entries[r].pop("moves", None)
            break
    print(f"{img['path']}: best game fit per board median {np.median(best_fit):.3f}, max {best_fit.max():.3f}; "
          f"{len(used)}/{len(entries)} boards from real games (threshold {a.threshold})")


def output_paths(path, a):
    stem = os.path.splitext(os.path.basename(path))[0]
    out = a.output.format(stem=stem)
    return out, (a.json.format(stem=stem) if a.json else os.path.splitext(out)[0] + ".json")


def write(img, palette, a):
    out_path, json_path = output_paths(img["path"], a)
    grid = np.zeros((img["boards_high"] * 8, a.boards * 8), dtype=np.int64)
    out_boards = []
    for e in img["entries"]:
        board = engine.opts_to_board(e["opts"], e["turn"])
        reasons = legality.problems(board)
        if reasons:
            raise RuntimeError(f"board {e['row']},{e['col']} failed legality: {reasons}")
        engine.place_board(grid, e["row"], e["col"], e["opts"])
        rec = {"row": e["row"], "col": e["col"], "fen": board.fen(), "source": e["source"],
               "rmse": round(float(np.sqrt(e["error"] / 64)), 2)}
        if "moves" in e:
            rec["moves"] = e["moves"]
        if e["source"] == "game":
            rec["fit"] = round(float((e["synthetic_error"] + 64 * a.fit_floor) / (e["error"] + 64 * a.fit_floor)), 3)
            rec["game"] = e["game"]
        out_boards.append(rec)

    engine.render(palette, grid, img["choice"], img["light"], a.scale,
                  a.grid, a.grid_color, a.grid_opacity).save(out_path)
    with open(json_path, "w") as f:
        json.dump({"boards_wide": a.boards, "boards_high": img["boards_high"], "boards": out_boards}, f, indent=1)
    n_game = sum(b["source"] == "game" for b in out_boards)
    print(f"wrote {out_path} and {json_path}: {len(out_boards)} legal boards ({n_game} from real games)")


def cmd_convert(a):
    t0 = time.time()
    if len(a.input) > 1 and ("{stem}" not in a.output or (a.json and "{stem}" not in a.json)):
        raise SystemExit("with several inputs, --output (and --json) must contain {stem}")
    palette = engine.Palette(a.theme, a.square_size, a.shades, a.light_band, a.dark_band,
                             a.piece_shades, a.white_band, a.black_band, a.min_contrast, a.spread)
    images = [prepare(path, palette, a) for path in a.input]

    if a.games is not None:
        # One pass over the index serves every board of every image.
        index = GameIndex(a.games or data_dir() / "index")
        tables = np.concatenate([np.stack([engine.board_cost(img["cost"], br, bc) for br, bc in img["coords"]])
                                 for img in images])
        t1 = time.time()
        matches = index.top_matches(tables, k=a.candidates, min_pieces=a.game_min_pieces)
        print(f"searched {len(index):,} game positions in {len(index.shards)} shards for {len(tables)} boards "
              f"in {time.time() - t1:.1f}s")
        start = 0
        for img in images:
            n = len(img["coords"])
            assign_games(img, matches[start:start + n], index, a)
            start += n

    for img in images:
        write(img, palette, a)
    print(f"done in {time.time() - t0:.1f}s")


def cmd_index(a):
    out = a.output or data_dir() / "index"
    for pgn in a.pgn:
        name = os.path.basename(pgn).split(".")[0].removeprefix("lichess_db_standard_rated_")
        build_shard(pgn, os.path.join(out, name), a.min_ply, a.stride, a.threads)


def cmd_download(a):
    root = data_dir()
    months = download.available()
    if a.list:
        sizes = download.remote_sizes(months)
        for m in months:
            print(f"{m}  {download.fmt_size(sizes[m]):>9}  index ~{download.fmt_size(download.estimate_index(sizes[m], a.stride)):>9}")
        print(f"{len(months)} months, {download.fmt_size(sum(sizes.values()))} compressed")
        return
    if a.budget:
        sizes = download.remote_sizes(months)
        chosen = download.fit_budget(months, sizes, a.budget * 1e9, not a.delete_pgn, a.stride)
    else:
        chosen = download.select(months, a)
        sizes = download.remote_sizes(chosen)
    if not chosen:
        raise SystemExit("nothing selected: give months, --range, --latest or --budget (see --list)")

    pgn_dir, index_dir = root / "pgn", root / "index"
    pgn_dir.mkdir(parents=True, exist_ok=True)
    pending = sum(sizes[m] - ((pgn_dir / download.filename(m)).stat().st_size
                              if (pgn_dir / download.filename(m)).exists() else 0) for m in chosen)
    indexing = 0 if a.no_index else sum(download.estimate_index(sizes[m], a.stride) for m in chosen
                                        if not (index_dir / m / "meta.json").exists())
    free = shutil.disk_usage(pgn_dir).free
    print(f"{len(chosen)} months ({chosen[0]} .. {chosen[-1]}) into {root}: "
          f"{download.fmt_size(pending)} to download, ~{download.fmt_size(indexing)} of index, "
          f"{download.fmt_size(free)} free")
    if pending + indexing > free and not a.force:
        raise SystemExit("not enough free space (use --budget, --delete-pgn, a larger --stride, or --force)")

    sums = download.checksums() if not a.no_verify else {}
    for m in chosen:
        path = download.fetch(m, pgn_dir, sizes[m])
        ok_marker = path.with_name(path.name + ".sha256-ok")
        if sums and not ok_marker.exists():
            print(f"{m}: verifying checksum")
            if download.sha256(path) != sums.get(path.name):
                raise SystemExit(f"{m}: checksum mismatch; delete {path} and download again")
            ok_marker.touch()
        if a.no_index:
            continue
        shard = index_dir / m
        if (shard / "meta.json").exists() and not a.reindex:
            print(f"{m}: already indexed")
        else:
            build_shard(path, shard, a.min_ply, a.stride, a.threads)
        if a.delete_pgn:
            path.unlink()
            ok_marker.unlink(missing_ok=True)


def cmd_verify(a):
    with open(a.json) as f:
        boards = json.load(f)["boards"]
    bad = 0
    for b in boards:
        reasons = legality.problems(chess.Board(b["fen"]))
        if "moves" in b:  # played boards: the game must really reach the position
            game = chess.Board()
            for token in b["moves"].split():
                if not token.endswith("."):
                    game.push_san(token)
            if game.board_fen() != chess.Board(b["fen"]).board_fen():
                reasons.append("move list does not reach the position")
        if reasons:
            bad += 1
            print(f"board {b['row']},{b['col']} {b['fen']}: {'; '.join(reasons)}")
    print(f"{len(boards) - bad}/{len(boards)} boards legal")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(prog="chesser", description="Turn images into grids of legal chess positions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert one or more images (several share one game-index search)")
    c.add_argument("input", nargs="+")
    c.add_argument("-o", "--output", default="{stem}-chess.png",
                   help="output image; {stem} is replaced by the input's name (required with several inputs)")
    c.add_argument("--json", help="board/FEN listing (default: output name with .json)")
    c.add_argument("-b", "--boards", type=int, default=8, help="boards across")
    c.add_argument("--square-size", type=int, default=8, help="pixels per square (a board is 8x that)")
    c.add_argument("--min-contrast", type=float, default=30,
                   help="min luminance difference between a piece's silhouette (outline or fill) and its square")
    c.add_argument("--boards-high", type=int, help="boards down (default: keep aspect ratio)")
    c.add_argument("--theme", choices=sorted(THEMES), default="mono")
    c.add_argument("--shades", type=int, default=12, help="shade levels per square colour")
    c.add_argument("--light-band", type=band, default=(150, 235), help="light square luminance range, e.g. 150,235")
    c.add_argument("--dark-band", type=band, default=(25, 110), help="dark square luminance range, e.g. 25,110")
    c.add_argument("--piece-shades", type=int, default=5, help="shade levels per piece colour")
    c.add_argument("--white-band", type=band, default=(170, 255), help="white piece fill luminance range")
    c.add_argument("--black-band", type=band, default=(0, 85), help="black piece fill luminance range")
    c.add_argument("--spread", type=float, default=10,
                   help="how far each quadrant of a square may shift from the square's base shade")
    c.add_argument("--smooth", type=float, default=0.5,
                   help="pull of each square's base shade toward the locally smoothed photo (0 = off)")
    c.add_argument("--smooth-radius", type=float, default=1.5, help="smoothing radius in squares (within a board)")
    c.add_argument("--generate", choices=("free", "played"), default="free",
                   help="free: pieces placed directly; played: every board reached by a real move sequence")
    c.add_argument("--beam", type=int, default=256, help="played: lines kept per half-move")
    c.add_argument("--max-plies", type=int, default=120, help="played: longest game, in half-moves")
    c.add_argument("--min-plies", type=int, default=20, help="played: shortest game, in half-moves")
    c.add_argument("--plausibility", type=int, choices=(0, 1, 2), default=2,
                   help="played: 0 legal only, 1 no mate in one ever available, 2 also no piece can be won outright")
    c.add_argument("--max-promotions", type=int, default=1,
                   help="generated boards: max pieces per side beyond the starting set (0 = normal armies only)")
    c.add_argument("--iters", type=int, default=4000, help="annealing steps per board")
    c.add_argument("--scale", type=int, default=1, help="nearest-neighbour upscale of the output")
    c.add_argument("--grid", type=int, default=2, help="separator line width between boards in output pixels (0 = none)")
    c.add_argument("--grid-color", type=lambda t: tuple(int(x) for x in t.split(",")), default=(20, 20, 20),
                   help="separator colour, e.g. 20,20,20")
    c.add_argument("--grid-opacity", type=float, default=0.45, help="separator opacity, 0..1")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=int)
    c.add_argument("--games", nargs="?", const="",
                   help="use real game positions; optional index directory (default: <data>/index)")
    c.add_argument("--threshold", type=float, default=0.9,
                   help="min fit (synthetic error / game error) for a real game position to be used")
    c.add_argument("--game-min-pieces", type=int, default=16,
                   help="skip real game positions with fewer pieces (sparse endgames read as empty boards)")
    c.add_argument("--fit-floor", type=float, default=4.0, help="per-pixel error floor used in the fit ratio")
    c.add_argument("--candidates", type=int, default=32, help="game candidates kept per board and shard")
    c.set_defaults(func=cmd_convert)

    d = sub.add_parser("download", help="download Lichess months and index them (you choose how much)")
    d.add_argument("months", nargs="*", help="months as YYYY-MM")
    d.add_argument("--list", action="store_true", help="show every published month with its size")
    d.add_argument("--range", help="months FROM:TO, e.g. 2022-01:2022-06")
    d.add_argument("--latest", type=int, help="the N most recent months")
    d.add_argument("--budget", type=float, help="newest months that fit in this many GB of disk")
    d.add_argument("--stride", type=int, default=4, help="keep every Nth position of each game")
    d.add_argument("--min-ply", type=int, default=8, help="skip the first N plies of every game")
    d.add_argument("--threads", type=int)
    d.add_argument("--delete-pgn", action="store_true", help="delete each download once indexed")
    d.add_argument("--no-index", action="store_true", help="only download")
    d.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    d.add_argument("--reindex", action="store_true", help="rebuild shards that already exist")
    d.add_argument("--force", action="store_true", help="ignore the free-space check")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("index", help="build index shards from PGN files (.pgn or .pgn.zst), one per file")
    i.add_argument("pgn", nargs="+")
    i.add_argument("-o", "--output", help="directory to put shards in (default: <data>/index)")
    i.add_argument("--min-ply", type=int, default=8, help="skip the first N plies of every game")
    i.add_argument("--stride", type=int, default=4, help="keep every Nth position of each game")
    i.add_argument("--threads", type=int)
    i.set_defaults(func=cmd_index)

    v = sub.add_parser("verify", help="re-check every FEN in a convert output")
    v.add_argument("json")
    v.set_defaults(func=cmd_verify)

    a = p.parse_args()
    raise SystemExit(a.func(a) or 0)


if __name__ == "__main__":
    main()
