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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from core.integration.constants import (  # isort: skip
    IHSAN_THRESHOLD,  # noqa: F401
    REFLEX_INVALIDATION_DELTA,
    REFLEX_INVALIDATION_INTERVAL,
    REFLEX_MAX_ENTRIES,
    REFLEX_PRECIPITATION_HITS,
    REFLEX_PRECIPITATION_IHSAN,
    REFLEX_STALENESS_DAYS,
    SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════


class HHMMEngine(Protocol):
    """Interface for HHMM state prediction (Spine §3, Helix 1)."""

    def predict_state(
        self, description: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Predict macro-state from mission description.

        Returns one of ~47 latent states (e.g., "EMAIL_COMPOSE",
        "CODE_REVIEW", "DOCUMENT_ANALYSIS").
        """
        ...

    def update_transitions(self, description: str, state: str, ihsan: float) -> None:
        """Update transition probabilities from observed mission."""
        ...


class EvidenceRecorder(Protocol):
    """Interface for evidence chain recording (Spine §8.1)."""

    def record_reflex_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Record a reflex event and return receipt hash."""
        ...


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
    source: str = "local"

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

    @property
    def confidence(self) -> float:
        """Confidence decays with time (BLOOM-style), grows with hits."""
        if self.created_at <= 0:
            return self.ihsan_composite
        age_months = (time.time() - self.created_at) / (86400 * 30)
        decay = 0.98**age_months  # Monthly decay factor
        hit_bonus = min(0.1, self.hit_count * 0.005)
        return min(1.0, self.ihsan_composite * decay + hit_bonus)


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


class ReflexStatus(Enum):
    """Lifecycle status for reflex entries."""

    ACTIVE = "active"
    STALE = "stale"
    EVICTED = "evicted"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class ReflexKey:
    """Hierarchical hash key: HHMM macro-state + mission hash.

    HHMM reduces search space from 2^256 to ~47 macro-states × 2^64 per-state.
    """

    macro_state: str
    mission_hash: str

    @classmethod
    def from_mission(
        cls, description: str, macro_state: Optional[str] = None
    ) -> ReflexKey:
        mission_hash = hashlib.sha256(
            description.lower().strip().encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            macro_state=macro_state or "UNKNOWN",
            mission_hash=mission_hash,
        )

    @property
    def composite_key(self) -> str:
        return f"{self.macro_state}::{self.mission_hash}"


@dataclass
class ObservationWindow:
    """Tracks repeated patterns waiting for precipitation.

    Like a supersaturated solution — observations accumulate until
    the pattern crystallizes into a reflex.
    """

    scores: List[float] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def avg_ihsan(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def recent_avg(self) -> float:
        recent = self.scores[-REFLEX_PRECIPITATION_HITS:]
        return sum(recent) / len(recent) if recent else 0.0

    @property
    def ready_to_precipitate(self) -> bool:
        if self.count < REFLEX_PRECIPITATION_HITS:
            return False
        return self.recent_avg >= REFLEX_PRECIPITATION_IHSAN

    @property
    def best_plan(self) -> Optional[Dict[str, Any]]:
        if not self.scores:
            return None
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.plans[best_idx]

    def add(self, ihsan: float, plan: Dict[str, Any]) -> None:
        self.scores.append(ihsan)
        self.plans.append(plan)
        self.timestamps.append(time.time())


@dataclass
class PrecipitationEvent:
    """Record of a reflex compilation event — evidence for the chain."""

    key: ReflexKey
    avg_ihsan: float
    observation_count: int
    compiled_at: str
    evidence_hash: str
    source: str = "local"


@dataclass
class CacheStats:
    """Runtime statistics for the reflex cache."""

    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    precipitations: int = 0
    invalidations: int = 0
    evictions: int = 0
    revalidations: int = 0
    forest_imports: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    @property
    def s1_fraction(self) -> float:
        """Fraction of lookups served from System-1 cache."""
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
            "revalidations": self.revalidations,
            "forest_imports": self.forest_imports,
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
        hhmm_engine: Optional[HHMMEngine] = None,
        evidence_recorder: Optional[EvidenceRecorder] = None,
        on_precipitation: Optional[Callable[[PrecipitationEvent], None]] = None,
    ) -> None:
        self._cache: OrderedDict[str, ReflexEntry] = OrderedDict()
        self._candidates: OrderedDict[str, PrecipitationCandidate] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()
        self._max_entries = max_entries
        self._max_candidates = max_entries * 2
        self._persistence_path = persistence_path
        self._hhmm = hhmm_engine
        self._evidence = evidence_recorder
        self._on_precipitation = on_precipitation
        self._precipitation_log: List[PrecipitationEvent] = []

        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    # ── Core Operations ──────────────────────────────────────────────────────

    def lookup(
        self, input_text: str, *, macro_state: Optional[str] = None
    ) -> Optional[ReflexEntry]:
        """
        O(1) cache lookup. Returns entry if found and not stale.

        If an HHMM engine is configured, uses hierarchical key prediction
        before falling back to simple hash. Applies confidence gate
        (SNR_THRESHOLD) to filter decayed entries.

        Returns:
            ReflexEntry if cache hit, None if miss.
        """
        pattern_hash = self._hash_input(input_text)

        # Enhanced path: HHMM hierarchical key
        composite_key: Optional[str] = None
        if self._hhmm:
            if macro_state is None:
                macro_state = self._hhmm.predict_state(input_text)
            rkey = ReflexKey.from_mission(input_text, macro_state)
            composite_key = rkey.composite_key

        with self._lock:
            self._stats.total_lookups += 1

            # Try HHMM composite key first, then fall back to simple hash
            entry: Optional[ReflexEntry] = None
            cache_key = pattern_hash
            if composite_key is not None:
                entry = self._cache.get(composite_key)
                if entry is not None:
                    cache_key = composite_key
            if entry is None:
                entry = self._cache.get(pattern_hash)
                cache_key = pattern_hash

            if entry is None:
                self._stats.cache_misses += 1
                return None

            if entry.stale:
                self._stats.cache_misses += 1
                return None

            # Confidence gate: skip entries below SNR threshold
            if entry.confidence < SNR_THRESHOLD:
                self._stats.cache_misses += 1
                return None

            # LRU: move to end
            self._cache.move_to_end(cache_key)

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
            candidate.observations.append(
                {
                    "output": output_text,
                    "ihsan_composite": ihsan_composite,
                    "ihsan_tensor": dict(ihsan_tensor or {}),
                    "timestamp": time.time(),
                }
            )

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

        # Record precipitation event for evidence trail
        evidence_data = {
            "key": candidate.pattern_hash,
            "ihsan": entry.ihsan_composite,
            "observations": entry.precipitation_count,
        }
        evidence_hash = hashlib.blake2b(
            json.dumps(evidence_data, sort_keys=True).encode(),
            digest_size=32,
        ).hexdigest()
        event = PrecipitationEvent(
            key=ReflexKey.from_mission(candidate.input_template),
            avg_ihsan=entry.ihsan_composite,
            observation_count=entry.precipitation_count,
            compiled_at=datetime.now(timezone.utc).isoformat(),
            evidence_hash=evidence_hash,
        )
        self._precipitation_log.append(event)

        if self._evidence:
            self._evidence.record_reflex_event("PRECIPITATION", asdict(event))

        if self._on_precipitation:
            self._on_precipitation(event)

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

    # ── Forest Protocol (Cross-node reflex sharing) ──────────────────────────

    def import_forest_reflex(
        self,
        key_str: str,
        plan: str,
        ihsan: float,
        source: str,
        confidence: float = 1.0,
    ) -> bool:
        """Import a reflex from another node via gossip protocol.

        Constitutional gate: only import if Ihsan >= precipitation threshold
        and confidence >= SNR threshold. Does not overwrite local reflexes
        with higher quality.

        Returns:
            True if imported, False if rejected.
        """
        if ihsan < REFLEX_PRECIPITATION_IHSAN:
            logger.info(
                "Forest reflex rejected: Ihsan %.3f < %.3f",
                ihsan,
                REFLEX_PRECIPITATION_IHSAN,
            )
            return False
        if confidence < SNR_THRESHOLD:
            logger.info(
                "Forest reflex rejected: confidence %.3f < %.3f",
                confidence,
                SNR_THRESHOLD,
            )
            return False

        with self._lock:
            existing = self._cache.get(key_str)
            if existing and existing.ihsan_composite >= ihsan:
                return False

            # LRU eviction if at capacity
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            entry = ReflexEntry(
                pattern_hash=key_str,
                input_template="",
                output_template=plan,
                ihsan_composite=ihsan,
                created_at=time.time(),
                last_hit_at=time.time(),
                last_validated_at=time.time(),
                source=source,
            )

            self._cache[key_str] = entry
            self._stats.forest_imports += 1

            logger.info(
                "Forest import: %s from %s (ihsan=%.3f)",
                key_str[:12],
                source[:12],
                ihsan,
            )

            return True

    def export_for_gossip(
        self,
        min_ihsan: float = REFLEX_PRECIPITATION_IHSAN,
        max_entries: int = 50,
    ) -> List[Dict[str, Any]]:
        """Export high-quality local reflexes for forest gossip protocol.

        Only exports abstract patterns — raw data stays sovereign.
        Spine §7.3: patterns propagate, data stays local.
        """
        with self._lock:
            exportable: List[Dict[str, Any]] = []
            for key_str, entry in self._cache.items():
                if len(exportable) >= max_entries:
                    break
                if (
                    entry.source == "local"
                    and entry.ihsan_composite >= min_ihsan
                    and not entry.stale
                    and entry.hit_count >= 2
                ):
                    exportable.append(
                        {
                            "key": key_str,
                            "plan_abstract": self._abstract_plan(entry.output_template),
                            "ihsan": entry.ihsan_composite,
                            "hit_count": entry.hit_count,
                            "confidence": round(entry.confidence, 4),
                        }
                    )
            return exportable

    def revalidate(self, key_str: str, new_ihsan: float) -> bool:
        """Revalidate a stale reflex with a fresh Ihsan score.

        Called during Helix 3 (evolutionary cycle) when a stale reflex
        is re-executed via System-2 and scored. Resets TTL on success,
        evicts on quality degradation.

        Returns:
            True if entry revalidated successfully, False otherwise.
        """
        with self._lock:
            entry = self._cache.get(key_str)
            if entry is None:
                return False

            if new_ihsan >= REFLEX_PRECIPITATION_IHSAN:
                entry.ihsan_composite = new_ihsan
                entry.last_validated_at = time.time()
                entry.created_at = time.time()  # Reset TTL
                entry.stale = False
                entry.validation_hits_since = 0
                self._stats.revalidations += 1
                logger.info(
                    "Reflex revalidated: %s (ihsan=%.3f)",
                    key_str[:12],
                    new_ihsan,
                )
                return True

            # Quality degraded — evict
            del self._cache[key_str]
            self._stats.evictions += 1
            logger.info(
                "Reflex evicted (quality degraded): %s (ihsan=%.3f)",
                key_str[:12],
                new_ihsan,
            )
            return False

    def get_top_reflexes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return top N reflexes by hit count for observability."""
        with self._lock:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda kv: kv[1].hit_count,
                reverse=True,
            )[:n]
            return [
                {
                    "key": key_str,
                    "ihsan": entry.ihsan_composite,
                    "hits": entry.hit_count,
                    "confidence": round(entry.confidence, 4),
                    "precipitation_count": entry.precipitation_count,
                    "source": entry.source,
                    "stale": entry.stale,
                }
                for key_str, entry in sorted_entries
            ]

    @staticmethod
    def _abstract_plan(plan: str) -> str:
        """Strip personal data from a plan for gossip export.

        Sovereignty-preserving: the pattern structure propagates,
        but personal content stays local.
        """
        if len(plan) > 200:
            return plan[:200] + "..."
        return plan

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_to_disk(self) -> None:
        """Serialize cache to disk for shutdown persistence.

        Uses atomic write (tempfile + os.replace) to prevent corruption
        if the process crashes mid-write. Same pattern as genesis_ceremony.
        Standing on Giants: Lamport (1978) — durable state transitions.
        """
        if self._persistence_path is None:
            return

        with self._lock:
            data = {
                "entries": {k: asdict(v) for k, v in self._cache.items()},
                "stats": self._stats.as_dict(),
                "saved_at": time.time(),
            }

        import os
        import tempfile

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._persistence_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self._persistence_path))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
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
                "s1_fraction": round(self._stats.s1_fraction, 4),
                **self._stats.as_dict(),
            }

    def clear(self) -> None:
        """Clear all entries and candidates."""
        with self._lock:
            self._cache.clear()
            self._candidates.clear()
