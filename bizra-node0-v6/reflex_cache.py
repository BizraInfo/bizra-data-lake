"""
BIZRA Reflex Cache — System 1 / System 2 Bridge
════════════════════════════════════════════════

HashMap-based O(1) pattern cache. Replaces SQLite.

When a mission pattern repeats 3+ times with Ihsan ≥ 0.90,
the response PRECIPITATES into the reflex cache. Future identical
queries hit S1 (cache) instead of S2 (full LLM pipeline).

This is the performance engine behind BIZRA's latency targets:
  TRIVIAL tier: <100ms (reflex hit)
  vs COMPLEX tier: <15,000ms (full PAT pipeline)

Precipitation model (Theorem 2.2):
  After K successful repetitions with Ihsan ≥ threshold,
  the pattern crystallizes into a reflex.
  Convergence: S1 hit rate increases monotonically with usage.

Constitution reference: §9 [reflex]
"""

from __future__ import annotations

import hashlib
import json
import time
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
from pathlib import Path

try:
    from generated.generated_constants import (
        REFLEX_MAX_ENTRIES,
        REFLEX_PRECIPITATION_HITS,
        REFLEX_PRECIPITATION_IHSAN,
        REFLEX_SIMILARITY_THRESHOLD,
        REFLEX_INVALIDATION_INTERVAL,
        REFLEX_INVALIDATION_DELTA,
        REFLEX_STALENESS_DAYS,
    )
except ImportError:
    REFLEX_MAX_ENTRIES = 500
    REFLEX_PRECIPITATION_HITS = 3
    REFLEX_PRECIPITATION_IHSAN = 0.90
    REFLEX_SIMILARITY_THRESHOLD = 0.95
    REFLEX_INVALIDATION_INTERVAL = 100
    REFLEX_INVALIDATION_DELTA = 0.05
    REFLEX_STALENESS_DAYS = 30

logger = logging.getLogger("bizra.reflex_cache")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReflexEntry:
    """A single cached reflex pattern."""
    pattern_hash: str           # SHA-256 of normalized input
    input_template: str         # The input pattern (stripped of specifics)
    output_template: str        # The cached response
    ihsan_composite: float      # Ihsan score when precipitated
    ihsan_tensor: dict[str, float]
    hit_count: int = 0          # Times this reflex was served
    precipitation_count: int = 0  # Times pattern repeated before precipitation
    created_at: float = 0.0     # Unix timestamp
    last_hit_at: float = 0.0    # Last time this reflex was used
    last_validated_at: float = 0.0  # Last freshness check
    validation_hits_since: int = 0  # Hits since last validation
    stale: bool = False         # Marked for invalidation

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def needs_validation(self) -> bool:
        """Check if this reflex needs a freshness validation."""
        if self.stale:
            return True
        if self.validation_hits_since >= REFLEX_INVALIDATION_INTERVAL:
            return True
        if self.age_days() > REFLEX_STALENESS_DAYS:
            return True
        return False


@dataclass
class PrecipitationCandidate:
    """Tracks a pattern that may precipitate into a reflex."""
    pattern_hash: str
    input_template: str
    observations: list[dict] = field(default_factory=list)
    # Each observation: {"output": str, "ihsan_composite": float, "ihsan_tensor": dict, "timestamp": float}

    def consecutive_high_quality(self) -> int:
        """Count consecutive observations with Ihsan ≥ threshold."""
        count = 0
        for obs in reversed(self.observations):
            if obs["ihsan_composite"] >= REFLEX_PRECIPITATION_IHSAN:
                count += 1
            else:
                break
        return count

    def ready_to_precipitate(self) -> bool:
        return self.consecutive_high_quality() >= REFLEX_PRECIPITATION_HITS

    def best_output(self) -> dict | None:
        """Return the observation with highest Ihsan."""
        if not self.observations:
            return None
        return max(self.observations, key=lambda o: o["ihsan_composite"])


@dataclass
class CacheStats:
    """Runtime statistics for the reflex cache."""
    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    precipitations: int = 0
    invalidations: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    def as_dict(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "precipitations": self.precipitations,
            "invalidations": self.invalidations,
            "evictions": self.evictions,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX CACHE — Thread-safe HashMap with LRU eviction
# ═══════════════════════════════════════════════════════════════════════════════


class ReflexCache:
    """
    O(1) pattern cache with precipitation and invalidation.

    Thread-safe via threading.Lock. Uses OrderedDict for LRU eviction.
    All thresholds sourced from constitution.toml.

    Lifecycle:
      1. Mission executes via full PAT pipeline (S2)
      2. Pattern hash recorded in precipitation candidates
      3. After 3 consecutive high-Ihsan completions → PRECIPITATE
      4. Future matching queries → O(1) cache hit (S1)
      5. Every 100th hit → freshness validation against LLM
      6. If delta > 0.05 → INVALIDATE
      7. After 30 days → force invalidation regardless
    """

    def __init__(
        self,
        max_entries: int = REFLEX_MAX_ENTRIES,
        persistence_path: Path | None = None,
    ):
        self._cache: OrderedDict[str, ReflexEntry] = OrderedDict()
        self._candidates: dict[str, PrecipitationCandidate] = {}
        self._lock = threading.Lock()
        self._stats = CacheStats()
        self._max_entries = max_entries
        self._persistence_path = persistence_path

        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    # ── Core Operations ──

    def lookup(self, input_text: str) -> ReflexEntry | None:
        """
        O(1) cache lookup. Returns entry if found and not stale.

        Args:
            input_text: The user's input text.

        Returns:
            ReflexEntry if cache hit, None if miss.
        """
        pattern_hash = self._hash_input(input_text)

        with self._lock:
            self._stats.total_lookups += 1

            entry = self._cache.get(pattern_hash)
            if entry is None:
                self._stats.cache_misses += 1
                return None

            if entry.stale:
                self._stats.cache_misses += 1
                return None

            # LRU: move to end
            self._cache.move_to_end(pattern_hash)

            # Update hit stats
            entry.hit_count += 1
            entry.last_hit_at = time.time()
            entry.validation_hits_since += 1

            self._stats.cache_hits += 1

            # Check if validation needed (but still return the cached result)
            if entry.needs_validation():
                logger.info(
                    f"Reflex {pattern_hash[:12]}... needs validation "
                    f"(hits_since={entry.validation_hits_since}, "
                    f"age={entry.age_days():.1f}d)"
                )

            return entry

    def record_observation(
        self,
        input_text: str,
        output_text: str,
        ihsan_composite: float,
        ihsan_tensor: dict[str, float],
    ) -> ReflexEntry | None:
        """
        Record a mission observation for potential precipitation.

        Called after every S2 (full pipeline) completion.
        If the pattern has been seen 3+ times with high Ihsan,
        it precipitates into a reflex.

        Args:
            input_text: The input that produced this output.
            output_text: The PAT pipeline output.
            ihsan_composite: Ihsan gate composite score.
            ihsan_tensor: Full 6-dim tensor.

        Returns:
            ReflexEntry if precipitation occurred, None otherwise.
        """
        pattern_hash = self._hash_input(input_text)

        with self._lock:
            # Get or create candidate
            if pattern_hash not in self._candidates:
                self._candidates[pattern_hash] = PrecipitationCandidate(
                    pattern_hash=pattern_hash,
                    input_template=input_text,
                )

            candidate = self._candidates[pattern_hash]
            candidate.observations.append({
                "output": output_text,
                "ihsan_composite": ihsan_composite,
                "ihsan_tensor": dict(ihsan_tensor),
                "timestamp": time.time(),
            })

            # Keep only last 10 observations to bound memory
            if len(candidate.observations) > 10:
                candidate.observations = candidate.observations[-10:]

            # Check for precipitation
            if candidate.ready_to_precipitate():
                return self._precipitate(candidate)

        return None

    def invalidate(self, pattern_hash: str) -> bool:
        """Mark a reflex as stale. It will not be served on next lookup."""
        with self._lock:
            entry = self._cache.get(pattern_hash)
            if entry is None:
                return False
            entry.stale = True
            self._stats.invalidations += 1
            logger.info(f"Invalidated reflex {pattern_hash[:12]}...")
            return True

    def validate_entry(
        self,
        pattern_hash: str,
        fresh_ihsan_composite: float,
    ) -> bool:
        """
        Compare fresh LLM output Ihsan against cached.
        If delta > threshold, invalidate.

        Returns True if entry is still valid.
        """
        with self._lock:
            entry = self._cache.get(pattern_hash)
            if entry is None:
                return False

            delta = abs(fresh_ihsan_composite - entry.ihsan_composite)
            entry.last_validated_at = time.time()
            entry.validation_hits_since = 0

            if delta > REFLEX_INVALIDATION_DELTA:
                entry.stale = True
                self._stats.invalidations += 1
                logger.info(
                    f"Reflex {pattern_hash[:12]}... invalidated: "
                    f"delta={delta:.3f} > {REFLEX_INVALIDATION_DELTA}"
                )
                return False

            return True

    # ── Precipitation ──

    def _precipitate(self, candidate: PrecipitationCandidate) -> ReflexEntry:
        """Crystallize a precipitation candidate into a reflex entry."""
        best = candidate.best_output()

        entry = ReflexEntry(
            pattern_hash=candidate.pattern_hash,
            input_template=candidate.input_template,
            output_template=best["output"],
            ihsan_composite=best["ihsan_composite"],
            ihsan_tensor=best["ihsan_tensor"],
            precipitation_count=len(candidate.observations),
            created_at=time.time(),
            last_hit_at=time.time(),
            last_validated_at=time.time(),
        )

        # Evict LRU if at capacity
        while len(self._cache) >= self._max_entries:
            evicted_hash, _ = self._cache.popitem(last=False)
            self._stats.evictions += 1
            logger.debug(f"Evicted LRU reflex {evicted_hash[:12]}...")

        self._cache[candidate.pattern_hash] = entry
        self._stats.precipitations += 1

        # Remove from candidates
        del self._candidates[candidate.pattern_hash]

        logger.info(
            f"Precipitated reflex {candidate.pattern_hash[:12]}... "
            f"(ihsan={entry.ihsan_composite:.3f}, "
            f"observations={entry.precipitation_count})"
        )

        return entry

    # ── Hashing ──

    @staticmethod
    def _hash_input(input_text: str) -> str:
        """
        Normalize and hash input for pattern matching.
        Normalization: lowercase, strip whitespace, collapse spaces.
        """
        normalized = " ".join(input_text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    # ── Persistence ──

    def save_to_disk(self):
        """Serialize cache to disk for shutdown persistence."""
        if self._persistence_path is None:
            return

        with self._lock:
            data = {
                "entries": {
                    k: asdict(v) for k, v in self._cache.items()
                },
                "stats": self._stats.as_dict(),
                "saved_at": time.time(),
            }

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persistence_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._cache)} reflexes to {self._persistence_path}")

    def _load_from_disk(self):
        """Deserialize cache from disk on startup."""
        try:
            with open(self._persistence_path) as f:
                data = json.load(f)

            for k, v in data.get("entries", {}).items():
                self._cache[k] = ReflexEntry(**v)

            logger.info(f"Loaded {len(self._cache)} reflexes from {self._persistence_path}")
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load reflex cache: {e}")

    # ── Introspection ──

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._cache)

    def entries_needing_validation(self) -> list[ReflexEntry]:
        """Return entries that need freshness checks."""
        with self._lock:
            return [e for e in self._cache.values() if e.needs_validation()]

    def get_all_entries(self) -> list[ReflexEntry]:
        """Return all entries (for telescript publishing scan)."""
        with self._lock:
            return list(self._cache.values())

    def clear(self):
        """Clear all entries and candidates."""
        with self._lock:
            self._cache.clear()
            self._candidates.clear()
