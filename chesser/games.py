"""Index of positions from real games, and matching them against image regions.

Shards are built by the Rust indexer (native/, one shard per PGN file) and
memory-mapped here, so the index can be far larger than RAM. Search runs the
AVX2 kernel in match.c over every shard, keeping the best candidates per
region by quantized score; those candidates are then rescored exactly.
"""

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import chess
import numpy as np

from .config import cache_dir
from .sprites import NUM_OPTIONS

HERE = Path(__file__).parent
NATIVE = HERE.parent / "native"


def _load_kernel():
    src, lib = HERE / "match.c", cache_dir() / "_match.so"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["cc", "-O3", "-mavx2", "-fopenmp", "-shared", "-fPIC",
                        str(src), "-o", str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    so.top_matches.restype = ctypes.c_int
    so.top_matches.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return so


def native_binary(name):
    """A native tool: $CHESSER_<NAME>, on PATH, or a release build of the repo's native/ crate."""
    found = os.environ.get("CHESSER_" + name.removeprefix("chesser-").upper()) or shutil.which(name)
    if found:
        return found
    built = NATIVE / "target" / "release" / name
    if not built.exists():
        if not (NATIVE / "Cargo.toml").exists():
            raise FileNotFoundError(f"{name} not found: run `cargo install --path native` from the repository")
        subprocess.run(["cargo", "build", "--release"], cwd=NATIVE, check=True,
                       env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"})
    return str(built)


def build_shard(pgn, out_dir, min_ply=8, stride=4, threads=None):
    """Run the Rust indexer on one PGN file."""
    cmd = [native_binary("chesser-indexer"), str(pgn), str(out_dir), "--min-ply", str(min_ply), "--stride", str(stride)]
    if threads:
        cmd += ["--threads", str(threads)]
    subprocess.run(cmd, check=True)


def quantize(tables):
    """(R, 64, 13) float costs -> (R, 64, 16) uint8 lookup tables in 0..255.

    Each square's minimum is subtracted (a per-region constant, so rankings are
    unchanged) and one scale per region maps the largest remaining cost to 255,
    so nothing is clipped. Unused codes 13-15 never occur in an index.
    """
    d = tables - tables.min(2, keepdims=True)
    scale = 255.0 / np.maximum(d.max((1, 2)), 1e-9)
    q = np.full((len(tables), 64, 16), 255, dtype=np.uint8)
    q[:, :, :NUM_OPTIONS] = np.round(d * scale[:, None, None])
    return q


class Shard:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        nb = self.meta["blocks"]
        self.n = self.meta["positions"]
        self.blocks = np.memmap(self.path / "blocks.bin", dtype=np.uint8, mode="r", shape=(nb, 32, 32))
        self.game = np.memmap(self.path / "game.bin", dtype=np.uint32, mode="r")
        self.ply = np.memmap(self.path / "ply.bin", dtype=np.uint16, mode="r")
        self.offsets = np.memmap(self.path / "offsets.bin", dtype=np.uint64, mode="r")

    def codes(self, slots):
        """Option codes (len(slots), 64) for the given slot indices."""
        slots = np.asarray(slots)
        packed = self.blocks[slots // 32, :, slots % 32]  # (n, 32)
        out = np.empty((len(slots), 64), dtype=np.uint8)
        out[:, 0::2] = packed & 15
        out[:, 1::2] = packed >> 4
        return out

    def info(self, slot):
        gid = int(self.game[slot])
        with open(self.path / "games.jsonl", "rb") as f:
            f.seek(int(self.offsets[gid]))
            headers = json.loads(f.readline())
        ply = int(self.ply[slot])
        site = headers.get("Site", "")
        if site.startswith("https://lichess.org/"):
            headers["url"] = f"{site}#{ply}"
        headers["ply"] = ply
        headers["shard"] = self.path.name
        return headers

    def turn(self, slot):
        return chess.WHITE if self.ply[slot] % 2 == 0 else chess.BLACK


class GameIndex:
    """All shards under a directory (each subdirectory with a meta.json)."""

    def __init__(self, root):
        root = Path(root)
        dirs = [root] if (root / "meta.json").exists() else sorted(p.parent for p in root.glob("*/meta.json"))
        if not dirs:
            raise FileNotFoundError(f"no index shards under {root}")
        self.shards = [Shard(d) for d in dirs]
        self.kernel = _load_kernel()

    def __len__(self):
        return sum(s.n for s in self.shards)

    def top_matches(self, tables, k=32, min_pieces=0):
        """Best k (exact error, (shard, slot)) per region, sorted ascending.

        Positions with fewer than min_pieces pieces are skipped inside the search,
        so they never crowd better qualifying candidates out of the shortlist.
        """
        R = len(tables)
        q = np.ascontiguousarray(quantize(tables))
        cands = [[] for _ in range(R)]
        for si, shard in enumerate(self.shards):
            out_s = np.empty((R, k), dtype=np.uint16)
            out_i = np.empty((R, k), dtype=np.int64)
            rc = self.kernel.top_matches(shard.blocks.ctypes.data, shard.meta["blocks"], shard.n,
                                         q.ctypes.data, R, k, min_pieces, out_s.ctypes.data, out_i.ctypes.data)
            if rc:
                raise MemoryError("match kernel allocation failed")
            for r in range(R):
                slots = out_i[r][out_i[r] >= 0]
                if len(slots) == 0:
                    continue
                codes = shard.codes(slots).astype(np.int64)
                exact = tables[r][np.arange(64), codes].sum(1)
                cands[r].extend(zip(exact.tolist(), [(si, int(s)) for s in slots]))
        return [sorted(c)[:k] for c in cands]
