"""Fetch Lichess monthly databases (standard rated games) and index them.

Downloads resume after interruption and are checked against Lichess's
published SHA-256 sums.
"""

import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://database.lichess.org/standard/"
CHUNK = 1 << 20


def filename(month):
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def _get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60)


def available():
    """All published months, oldest first, as 'YYYY-MM'."""
    with _get(BASE + "list.txt") as r:
        return sorted(set(re.findall(r"rated_(\d{4}-\d{2})\.pgn\.zst", r.read().decode())))


def remote_size(month, attempts=6):
    """Size in bytes, retrying with backoff when Lichess rate-limits (HTTP 429)."""
    req = urllib.request.Request(BASE + filename(month), method="HEAD")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))


def remote_sizes(months):
    # Few parallel requests: the server rate-limits bursts.
    with ThreadPoolExecutor(3) as pool:
        return dict(zip(months, pool.map(remote_size, months)))


def checksums():
    with _get(BASE + "sha256sums.txt") as r:
        return {name: digest for digest, name in (line.split() for line in r.read().decode().splitlines() if line)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 * CHUNK):
            h.update(block)
    return h.hexdigest()


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000


def fetch(month, dest_dir, expected_size, log=print):
    """Download one month into dest_dir, resuming a partial file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename(month)
    have = path.stat().st_size if path.exists() else 0
    if have == expected_size:
        log(f"{month}: already downloaded")
        return path
    if have > expected_size:
        path.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    started, last = time.time(), 0.0
    with _get(BASE + filename(month), headers) as r, open(path, "ab" if have else "wb") as f:
        if have and r.status != 206:  # server ignored the range: start over
            f.truncate(0)
            have = 0
        done = have
        while block := r.read(CHUNK):
            f.write(block)
            done += len(block)
            now = time.time()
            if now - last > 2 or done == expected_size:
                speed = (done - have) / max(now - started, 1e-9)
                eta = (expected_size - done) / max(speed, 1)
                print(f"\r{month}: {fmt_size(done)} / {fmt_size(expected_size)}  {fmt_size(speed)}/s  "
                      f"eta {eta / 60:.0f} min   ", end="", file=sys.stderr, flush=True)
                last = now
    print(file=sys.stderr)
    if path.stat().st_size != expected_size:
        raise IOError(f"{month}: download incomplete ({path.stat().st_size} of {expected_size} bytes); run again to resume")
    return path


def select(months, a):
    """Months chosen by the command-line options (explicit list, --range, --latest)."""
    chosen = list(a.months)
    if a.range:
        lo, _, hi = a.range.partition(":")
        chosen += [m for m in months if lo <= m <= (hi or lo)]
    if a.latest:
        chosen += months[-a.latest:]
    unknown = sorted(set(chosen) - set(months))
    if unknown:
        raise SystemExit(f"not published by Lichess: {', '.join(unknown)}")
    return sorted(set(chosen))


def estimate_index(size, stride):
    """Rough index size: ~1.9x the compressed PGN at stride 4 (measured on 2022 data)."""
    return int(size * 1.9 * 4 / stride)


def fit_budget(months, sizes, budget, keep_pgn, stride):
    """Newest months whose estimated footprint fits in `budget` bytes."""
    chosen, used = [], 0
    for m in reversed(months):
        need = estimate_index(sizes[m], stride) + (sizes[m] if keep_pgn else 0)
        if used + need > budget:
            continue
        chosen.append(m)
        used += need
    return sorted(chosen)
