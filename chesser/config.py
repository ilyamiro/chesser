"""Where chesser keeps its data and build artifacts."""

import os
from pathlib import Path


def data_dir():
    """$CHESSER_DATA, else ./data if it exists, else ~/.local/share/chesser."""
    env = os.environ.get("CHESSER_DATA")
    if env:
        return Path(env).expanduser()
    if Path("data").is_dir():
        return Path("data")
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "chesser"


def cache_dir():
    path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "chesser"
    path.mkdir(parents=True, exist_ok=True)
    return path
