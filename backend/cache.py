"""
Persistent file-based cache with version-based invalidation.

Cache persists across server restarts. Validity is controlled by a .version
file — when the GitHub Actions scrape workflow updates the DB, it bumps
this version, atomically invalidating all cached entries.

Thread-safe. Uses an in-memory hot cache for frequently accessed entries.

Design:
- Version-based invalidation: All entries are tagged with the current cache
  version. When the DB updates, the version is bumped -> all old entries are
  invalidated atomically without needing to delete files.
- Data-level caching: Cache keys are derived from the DATA FILTERS only
  (not pagination/sort params). This means changing page or sort order
  doesn't require a new DB query.
- Two-tier: Hot cache (RAM) + File cache (disk, survives restarts)
"""
import hashlib
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable


# --- Paths ---
CACHE_DATA_DIR = Path(__file__).parent / "cache_data"
VERSION_FILE = CACHE_DATA_DIR / ".version"


def _get_cache_version() -> str:
    """Read the current cache version from the .version file.

    Bumped by the GH Actions scrape workflow after a successful DB update.
    Returns "0" if the file doesn't exist (first deploy / local dev).
    """
    try:
        if VERSION_FILE.exists():
            return VERSION_FILE.read_text().strip()
    except OSError:
        pass
    return "0"


def _make_cache_key(*args, **kwargs) -> str:
    """Generate a deterministic SHA-256 cache key from args."""
    raw = ":".join([str(args), str(sorted(kwargs.items()))])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def function_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from a function's qualified name + args."""
    key_parts = [
        func.__module__ or "",
        func.__qualname__ or func.__name__,
        str(args),
        str(sorted(kwargs.items())),
    ]
    raw = ":".join(key_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cache_path(key: str) -> Path:
    """Get filesystem path for a cache key, sharded by first 2 hex chars."""
    subdir = CACHE_DATA_DIR / key[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key}.json"


class PersistentCache:
    """File-based persistent cache with version-based invalidation.

    Features:
    - JSON-serialized values stored on disk (survives restarts)
    - Version-driven invalidation (bumped by GH Action after DB update)
    - In-memory hot cache for frequently accessed entries
    - Thread-safe via RLock
    - Configurable TTL as safety net (default: 7 days)
    - Stats tracking for monitoring
    """

    def __init__(self, ttl: int = 604800, hot_cache_size: int = 512):
        self._ttl = ttl
        self._hot_cache_size = hot_cache_size
        self._lock = threading.RLock()
        self._hot_cache: dict[str, tuple[Any, float, str]] = {}
        self._hits = 0
        self._misses = 0
        CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._hot_cache)

    def get(self, key: str) -> Any | None:
        """Retrieve a value from cache (hot -> disk). Returns None on miss/stale."""
        current_version = _get_cache_version()
        now = time.time()

        with self._lock:
            # 1. Hot cache check
            if key in self._hot_cache:
                value, ts, ver = self._hot_cache[key]
                if ver == current_version and (now - ts) < self._ttl:
                    self._hits += 1
                    return value
                del self._hot_cache[key]

            # 2. Disk cache check
            cache_path = _get_cache_path(key)
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    sv = cached.get("version", "0")
                    st = cached.get("timestamp", 0)
                    sd = cached.get("data")
                    if sv == current_version and (now - st) < self._ttl:
                        self._hot_cache[key] = (sd, st, sv)
                        self._hits += 1
                        return sd
                    cache_path.unlink(missing_ok=True)
                except (json.JSONDecodeError, OSError, KeyError):
                    try:
                        cache_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            self._misses += 1
            return None

    def set(self, key: str, value: Any):
        """Store a value in cache (hot + disk)."""
        current_version = _get_cache_version()
        now = time.time()
        cache_path = _get_cache_path(key)

        payload = {"version": current_version, "timestamp": now, "data": value}

        with self._lock:
            self._hot_cache[key] = (value, now, current_version)
            # Trim hot cache if needed (oldest 25%)
            if len(self._hot_cache) > self._hot_cache_size:
                sorted_keys = sorted(
                    self._hot_cache.keys(),
                    key=lambda k: self._hot_cache[k][1],
                )
                for k in sorted_keys[: self._hot_cache_size // 4]:
                    del self._hot_cache[k]

            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(payload, f, default=str, separators=(",", ":"))
            except (OSError, TypeError) as e:
                pass  # Degrade gracefully - entry stays in hot cache only

    def invalidate(self, key: str):
        """Remove a specific key from both caches."""
        with self._lock:
            self._hot_cache.pop(key, None)
            p = _get_cache_path(key)
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self):
        """Clear ALL cached data (hot cache + all disk files)."""
        with self._lock:
            self._hot_cache.clear()
            self._hits = 0
            self._misses = 0
            try:
                for f in CACHE_DATA_DIR.glob("*/*.json"):
                    f.unlink(missing_ok=True)
            except OSError:
                pass

    def cached(self, ttl: int | None = None,
               key_func: Callable | None = None) -> Callable:
        """Decorator: caches a function's return value.

        Args:
            ttl: Override default TTL for this specific function.
            key_func: Custom key function (func, args, kwargs) -> str.
                      Default uses function name + all args.
        """
        effective_ttl = ttl if ttl is not None else self._ttl

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = (
                    key_func(func, args, kwargs)
                    if key_func
                    else function_cache_key(func, args, kwargs)
                )
                cached = self.get(cache_key)
                if cached is not None:
                    return cached
                result = func(*args, **kwargs)
                self.set(cache_key, result)
                return result

            return wrapper

        return decorator


# Global singleton - shared across the entire app
persistent_cache = PersistentCache()
