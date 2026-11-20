"""Load model/lru.py from the authors' pinned source or a verified local cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from urllib.error import URLError
from urllib.request import urlopen


UPSTREAM_COMMIT = "e02a559bd8c42342f5c53d69cefea2589becb22c"
SOURCE_SHA256 = "149e0ccfe03c7eba7a51bf405d179f8a3f395011c0f730169aec2b59c9c156b1"
SOURCE_URL = f"https://raw.githubusercontent.com/yueqirex/LRURec/{UPSTREAM_COMMIT}/model/lru.py"


def _verify(source, location):
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"LRURec source {location} has SHA-256 {actual}; expected {SOURCE_SHA256}.")
    return source


def read_source(cache_dir=None, *, offline=False):
    """Read verified bytes. Only the first cache population requires the authors' server."""
    override = os.environ.get("LRUREC_REFERENCE_FILE")
    if override:
        path = Path(override).expanduser()
        return path, _verify(path.read_bytes(), path)

    root = cache_dir or os.environ.get("LRUREC_CACHE_DIR") or Path.home() / ".cache" / "online-retail" / "lrurec"
    path = Path(root).expanduser() / UPSTREAM_COMMIT / "lru.py"
    if path.is_file():
        return path, _verify(path.read_bytes(), path)
    if offline:
        raise FileNotFoundError(f"LRURec source missing at {path}. Populate the cache or set LRUREC_REFERENCE_FILE.")

    try:
        with urlopen(SOURCE_URL, timeout=30) as response:
            source = _verify(response.read(), SOURCE_URL)
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not fetch {SOURCE_URL}. Supply a cached copy through LRUREC_REFERENCE_FILE: {error}"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / "lru.py"
        temporary.write_bytes(source)
        temporary.replace(path)
    return path, source


def load_lru_class():
    path, source = read_source()
    name = "yueqirex_lrurec_" + UPSTREAM_COMMIT
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__file__ = str(path)
        # Execute the bytes whose digest was checked, without loading a stale bytecode cache.
        exec(compile(source, str(path), "exec"), module.__dict__)
        sys.modules[name] = module
    return module.LRU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Defaults to LRUREC_CACHE_DIR or the user cache.")
    parser.add_argument("--offline", action="store_true", help="Read a verified local file without downloading.")
    args = parser.parse_args()
    path, _ = read_source(args.cache_dir, offline=args.offline)
    print(json.dumps({"commit": UPSTREAM_COMMIT, "sha256": SOURCE_SHA256, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
