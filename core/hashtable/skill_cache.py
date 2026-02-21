"""
Skill Cache — System 2→1 Compression via Structural Hashing

Standing on Giants:
  Kahneman (2011) — System 1 (fast/automatic) vs System 2 (slow/deliberate)
  Anderson (1982) — ACT-R skill compilation theory

When a thought chain is computed (System 2), its structural hash is
stored alongside the result. Future queries with the same structural
pattern bypass deliberation (System 1) — "skill compilation."

Thread-safe via threading.Lock, matching the pattern established in
core/proof_engine/evidence_ledger.py and core/token/ledger.py.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.integration.constants import (
    SKILL_CACHE_DEFAULT_TTL,
    SKILL_CACHE_MAX_SIZE,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.proof_engine.canonical import blake3_digest, canonical_bytes


@dataclass(frozen=True)
class CachedSkillResult:
    """Immutable cached skill entry."""

    structural_hash: str
    query_pattern: str
    result: Dict[str, Any]
    snr_score: float
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    last_hit: float = 0.0


class SkillCache:
    """
    LRU cache for compiled thought patterns.

    Keys are structural hashes of thought chains. Results below the
    Ihsan floor are auto-evicted on retrieval — the system refuses
    to serve cached mediocrity.

    Thread-safe: all public methods acquire self._lock.
    """

    __slots__ = (
        "_cache",
        "_lock",
        "_max_size",
        "_default_ttl",
        "_ihsan_floor",
        "_hits",
        "_misses",
        "_evictions",
    )

    def __init__(
        self,
        max_size: int = SKILL_CACHE_MAX_SIZE,
        default_ttl: int = SKILL_CACHE_DEFAULT_TTL,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    ) -> None:
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max(1, max_size)
        self._default_ttl = default_ttl
        self._ihsan_floor = ihsan_floor
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def structural_hash(self, thought_chain: list[dict]) -> str:
        """
        Compute a deterministic structural hash of a thought chain.

        Uses canonical_bytes for stable serialization (sorted keys,
        NFC unicode, no whitespace variance) then BLAKE3 for the hash.
        Returns the first 16 hex characters (64 bits — collision-safe
        for cache-sized namespaces).
        """
        raw = canonical_bytes(thought_chain)
        digest = blake3_digest(raw)
        return digest.hex()[:16]

    def get(self, key: str) -> Optional[CachedSkillResult]:
        """
        Retrieve a cached skill result.

        Returns None on miss, TTL expiry, or Ihsan floor violation.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            now = time.monotonic()

            # TTL check
            if now - entry.created_at > entry.ttl_seconds:
                del self._cache[key]
                self._evictions += 1
                self._misses += 1
                return None

            # Ihsan floor check — refuse to serve sub-threshold results
            if entry.snr_score < self._ihsan_floor:
                del self._cache[key]
                self._evictions += 1
                self._misses += 1
                return None

            # Cache hit — update stats and move to end (most recent)
            entry.hit_count += 1
            entry.last_hit = now
            self._cache.move_to_end(key)
            self._hits += 1

            return CachedSkillResult(
                structural_hash=key,
                query_pattern=entry.query_pattern,
                result=entry.result,
                snr_score=entry.snr_score,
                created_at=entry.created_at,
                ttl_seconds=entry.ttl_seconds,
                hit_count=entry.hit_count,
                last_hit=entry.last_hit,
            )

    def put(
        self,
        key: str,
        result: Dict[str, Any],
        snr_score: float,
        query_pattern: str = "",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a compiled skill result.

        Items below the Ihsan floor are silently rejected —
        the system never caches mediocrity.
        """
        if snr_score < self._ihsan_floor:
            return

        with self._lock:
            now = time.monotonic()
            entry = _CacheEntry(
                query_pattern=query_pattern,
                result=result,
                snr_score=snr_score,
                created_at=now,
                ttl_seconds=ttl if ttl is not None else self._default_ttl,
                hit_count=0,
                last_hit=0.0,
            )

            # Insert/update
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # LRU eviction
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": len(self._cache),
                "max_size": self._max_size,
                "fill_ratio": (
                    len(self._cache) / self._max_size if self._max_size else 0.0
                ),
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"SkillCache(size={len(self._cache)}/{self._max_size}, "
                f"hits={self._hits}, misses={self._misses})"
            )


class _CacheEntry:
    """Mutable internal cache entry (not exposed externally)."""

    __slots__ = (
        "query_pattern",
        "result",
        "snr_score",
        "created_at",
        "ttl_seconds",
        "hit_count",
        "last_hit",
    )

    def __init__(
        self,
        query_pattern: str,
        result: Dict[str, Any],
        snr_score: float,
        created_at: float,
        ttl_seconds: int,
        hit_count: int,
        last_hit: float,
    ) -> None:
        self.query_pattern = query_pattern
        self.result = result
        self.snr_score = snr_score
        self.created_at = created_at
        self.ttl_seconds = ttl_seconds
        self.hit_count = hit_count
        self.last_hit = last_hit
