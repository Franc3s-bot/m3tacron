"""
Simple in-memory TTL cache with zero external dependencies.

Provides a decorator-based cache for expensive function calls,with time-based expiration. Thread-safe via reentrant lock."""
import time
import threading
from functools import wraps
from typing import Any, Callable


class TTLCache:
    """
    A simple time-to-live (TTL) cache for storing function results in memory.

    Features:
    - Configurable TTL per instance (seconds)
    - Optional custom key function for advanced cache key generation
    - Thread-safe read/write via RLock
    - Automatic cleanup of expired entries on access
    - Stats tracking (hits, misses, size)

    Usage:
        cache = TTLCache(ttl=600)  # 10 minutes

        @cache.decorator
        def expensive_func(arg1, arg2):
            ...
            return result

        # Or manually:
        cache.set("my_key", some_value)
        value = cache.get("my_key")
    """

    def __init__(self, ttl: int = 600, key_func: Callable | None = None):
        """
        Initialize the cache.

        Args:
            ttl: Time-to-live in seconds. Default 600 (10 minutes).
            key_func: Optional function to generate cache keys from
                      function arguments. If None, uses str(args, kwargs).
        """
        self._ttl = ttl
        self._key_func = key_func
        self._store: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function arguments."""
        if self._key_func:
            return self._key_func(*args, **kwargs)
        return str((args, tuple(sorted(kwargs.items()))))

    def _evict_expired(self):
        """Remove all expired entries from the store."""
        now = time.time()
        expired = [
            key for key, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for key in expired:
            del self._store[key]
            del self._timestamps[key]

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None if key doesn't exist or is expired."""
        with self._lock:
            self._evict_expired()
            if key in self._store:
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any):
        """Set a value in cache with the current timestamp."""
        with self._lock:
            self._store[key] = value
            self._timestamps[key] = time.time()

    def invalidate(self, key: str):
        """Remove a specific key from cache."""
        with self._lock:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)

    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            self._store.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0

    def decorator(self, func: Callable) -> Callable:
        """
        Decorator that caches the return value of the wrapped function.

        The cache key is derived from the function name and arguments.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__qualname__}:{self._make_key(args, kwargs)}"
            cached = self.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        return wrapper


# Global singleton caches for different endpoint types
# This allows shared caching across requests within the same worker process
meta_snapshot_cache = TTLCache(ttl=600)  # 10 minutes for dashboard data
