"""
BIZRA Reflex Compiler — System-1 O(1) Cache with Precipitation
═══════════════════════════════════════════════════════════════

HashMap-based pattern cache. When a mission pattern repeats K times
with Ihsan >= threshold, the response PRECIPITATES into the reflex
cache. Future identical queries hit S1 (cache, <100ms) instead of
S2 (full LLM pipeline, <15,000ms).

Precipitation model (Theorem 2.2):
  After K successful repetitions with Ihsan >= threshold,
  the pattern crystallizes into a reflex.
  Convergence: S1 hit rate increases monotonically with usage.

Standing on Giants:
  Kahneman (2011) — System 1/2 dual-process theory
  Shannon (1948) — Information entropy + SNR
  Deming (1950) — PDCA closed-loop quality ratchet

Constitutional reference: §9 [reflex], constants.py (SSOT)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.integration.constants import (
    REFLEX_INVALIDATION_DELTA,
    REFLEX_INVALIDATION_INTERVAL,
    REFLEX_MAX_ENTRIES,
    REFLEX_PRECIPITATION_HITS,
    REFLEX_PRECIPITATION_IHSAN,
    REFLEX_STALENESS_DAYS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReflexEntry:
    """A single cached reflex pattern — the crystallized System-1 response."""

    pattern_hash: str
    input_template: str
    output_template: str
    ihsan_composite: float
    ihsan_tensor: dict[str, float] = field(default_factory=dict)
    hit_count: int = 0
    precipitation_count: int = 0
    created_at: float = 0.0
    last_hit_at: float = 0.0
    last_validated_at: float = 0.0
    validation_hits_since: int = 0
    stale: bool = False

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
    observations: list[dict[str, Any]] = field(default_factory=list)

    def consecutive_high_quality(self) -> int:
        """Count consecutive observations with Ihsan >= threshold."""
        count = 0
        for obs in reversed(self.observations):
            if obs.get("ihsan_composite", 0.0) >= REFLEX_PRECIPITATION_IHSAN:
                count += 1
            else:
                break
        return count

    def ready_to_precipitate(self) -> bool:
        return self.consecutive_high_quality() >= REFLEX_PRECIPITATION_HITS

    def best_output(self) -> Optional[dict[str, Any]]:
        """Return the observation with highest Ihsan."""
        if not self.observations:
            return None
        return max(self.observations, key=lambda o: o.get("ihsan_composite", 0.0))


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

    def as_dict(self) -> dict[str, Any]:
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
# REFLEX COMPILER — Thread-safe HashMap with LRU eviction + precipitation
# ═══════════════════════════════════════════════════════════════════════════════


class ReflexCompiler:
    """
    O(1) pattern cache with precipitation and constitutional invalidation.

    Thread-safe via threading.Lock. Uses OrderedDict for LRU eviction.
    All thresholds sourced from core/integration/constants.py (SSOT).

    Lifecycle:
      1. Mission executes via full PAT pipeline (System-2)
      2. Pattern hash recorded in precipitation candidates
      3. After K consecutive high-Ihsan completions -> PRECIPITATE
      4. Future matching queries -> O(1) cache hit (System-1)
      5. Every Nth hit -> freshness validation against LLM
      6. If delta > threshold -> INVALIDATE
      7. After staleness_days -> force invalidation regardless

    Usage::

        compiler = ReflexCompiler(persistence_path=state_dir / "reflexes.json")

        # System-2 path: check cache first
        entry = compiler.lookup("what is autopoiesis?")
        if entry:
            return entry.output_template  # <100ms

        # Full pipeline result
        result = await mission_orchestrator.execute(request)

        # Record for precipitation
        compiler.record_observation(
            input_text=request.description,
            output_text=result.synthesis,
            ihsan_composite=result.ihsan_score,
        )
    """

    def __init__(
        self,
        max_entries: int = REFLEX_MAX_ENTRIES,
        persistence_path: Optional[Path] = None,
    ) -> None:
        self._cache: OrderedDict[str, ReflexEntry] = OrderedDict()
        self._candidates: OrderedDict[str, PrecipitationCandidate] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()
        self._max_entries = max_entries
        self._max_candidates = max_entries * 2
        self._persistence_path = persistence_path

        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    # ── Core Operations ──────────────────────────────────────────────────────

    def lookup(self, input_text: str) -> Optional[ReflexEntry]:
        """
        O(1) cache lookup. Returns entry if found and not stale.

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

            if entry.needs_validation():
                logger.info(
                    "Reflex %s needs validation (hits_since=%d, age=%.1fd)",
                    pattern_hash[:12],
                    entry.validation_hits_since,
                    entry.age_days(),
                )

            return entry

    def record_observation(
        self,
        input_text: str,
        output_text: str,
        ihsan_composite: float,
        ihsan_tensor: Optional[dict[str, float]] = None,
    ) -> Optional[ReflexEntry]:
        """
        Record a mission observation for potential precipitation.

        Called after every System-2 (full pipeline) completion.
        If the pattern has been seen K+ times with high Ihsan,
        it precipitates into a reflex.

        Returns:
            ReflexEntry if precipitation occurred, None otherwise.
        """
        pattern_hash = self._hash_input(input_text)

        with self._lock:
            if pattern_hash not in self._candidates:
                # Evict oldest candidate if at capacity
                while len(self._candidates) >= self._max_candidates:
                    self._candidates.popitem(last=False)
                self._candidates[pattern_hash] = PrecipitationCandidate(
                    pattern_hash=pattern_hash,
                    input_template=input_text,
                )

            candidate = self._candidates[pattern_hash]
            candidate.observations.append({
                "output": output_text,
                "ihsan_composite": ihsan_composite,
                "ihsan_tensor": dict(ihsan_tensor or {}),
                "timestamp": time.time(),
            })

            # Bound memory: keep only last 10 observations
            if len(candidate.observations) > 10:
                candidate.observations = candidate.observations[-10:]

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
            logger.info("Invalidated reflex %s", pattern_hash[:12])
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
                    "Reflex %s invalidated: delta=%.3f > %.3f",
                    pattern_hash[:12],
                    delta,
                    REFLEX_INVALIDATION_DELTA,
                )
                return False

            return True

    # ── Compilation from SDPO Bridge ─────────────────────────────────────────

    def compile_from_candidate(
        self,
        pattern_id: str,
        input_template: str,
        output_template: str,
        ihsan_score: float,
        observation_count: int,
    ) -> ReflexEntry:
        """
        Directly compile a ReflexCandidate from the SDPO bridge.

        This bypasses the precipitation mechanism — the SDPO bridge
        has already verified eligibility (Ihsan >= 0.98, reproducibility
        >= 0.90, observations >= 5, impact > 0.01).
        """
        with self._lock:
            # Evict LRU if at capacity
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            entry = ReflexEntry(
                pattern_hash=pattern_id,
                input_template=input_template,
                output_template=output_template,
                ihsan_composite=ihsan_score,
                precipitation_count=observation_count,
                created_at=time.time(),
                last_hit_at=time.time(),
                last_validated_at=time.time(),
            )

            self._cache[pattern_id] = entry
            self._stats.precipitations += 1

            logger.info(
                "Compiled reflex %s from SDPO (ihsan=%.3f, obs=%d)",
                pattern_id[:12],
                ihsan_score,
                observation_count,
            )

            return entry

    # ── Precipitation ────────────────────────────────────────────────────────

    def _precipitate(self, candidate: PrecipitationCandidate) -> ReflexEntry:
        """Crystallize a precipitation candidate into a reflex entry."""
        best = candidate.best_output()
        if best is None:
            raise RuntimeError("Cannot precipitate: no observations available")

        entry = ReflexEntry(
            pattern_hash=candidate.pattern_hash,
            input_template=candidate.input_template,
            output_template=best["output"],
            ihsan_composite=best["ihsan_composite"],
            ihsan_tensor=best.get("ihsan_tensor", {}),
            precipitation_count=len(candidate.observations),
            created_at=time.time(),
            last_hit_at=time.time(),
            last_validated_at=time.time(),
        )

        # Evict LRU if at capacity
        while len(self._cache) >= self._max_entries:
            evicted_hash, _ = self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._cache[candidate.pattern_hash] = entry
        self._stats.precipitations += 1
        del self._candidates[candidate.pattern_hash]

        logger.info(
            "Precipitated reflex %s (ihsan=%.3f, observations=%d)",
            candidate.pattern_hash[:12],
            entry.ihsan_composite,
            entry.precipitation_count,
        )

        return entry

    # ── Hashing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_input(input_text: str) -> str:
        """Normalize and hash input for pattern matching."""
        normalized = " ".join(input_text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_to_disk(self) -> None:
        """Serialize cache to disk for shutdown persistence."""
        if self._persistence_path is None:
            return

        with self._lock:
            data = {
                "entries": {k: asdict(v) for k, v in self._cache.items()},
                "stats": self._stats.as_dict(),
                "saved_at": time.time(),
            }

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d reflexes to %s", len(self._cache), self._persistence_path)

    def _load_from_disk(self) -> None:
        """Deserialize cache from disk on startup."""
        try:
            data = json.loads(self._persistence_path.read_text())
            for k, v in data.get("entries", {}).items():
                self._cache[k] = ReflexEntry(**v)
            logger.info(
                "Loaded %d reflexes from %s",
                len(self._cache),
                self._persistence_path,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Failed to load reflex cache: %s", e)

    # ── Introspection ────────────────────────────────────────────────────────

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

    def get_status(self) -> dict[str, Any]:
        """Return status dict for telemetry and /v1/status."""
        with self._lock:
            return {
                "size": len(self._cache),
                "candidates": len(self._candidates),
                "max_entries": self._max_entries,
                **self._stats.as_dict(),
            }

    def clear(self) -> None:
        """Clear all entries and candidates."""
        with self._lock:
            self._cache.clear()
            self._candidates.clear()
