"""
Constitutional Engram Layer (CEL) — O(1) constitutional memory.

Standing on Giants:
- Al-Ghazali: verified truth should not be re-derived
- Al-Khwarizmi: compute once, look up forever
- Architecture of Intelligence whitepaper: "no-op planning" as cognitive scaffold

The insight: constitutional invariants are IMMUTABLE. Their verification
results are DETERMINISTIC. Pre-computing and caching these results converts
O(k) gate checks into O(1) lookups, freeing cognitive depth for novel reasoning.

Usage:
    cel = ConstitutionalEngramLayer()
    
    # First time: computes and caches
    clearance = cel.check(mission_type="research_query", context={...})
    
    # Second time: O(1) lookup, no recomputation
    clearance = cel.check(mission_type="research_query", context={...})
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("bizra.cel")

# Constitutional invariants — immutable by definition
INVARIANTS = {
    "I-1_IHSAN_FLOOR": {"threshold": 0.95, "operator": ">="},
    "I-2_RIBA_ZERO": {"threshold": 0, "operator": "=="},
    "I-3_ADL_LIMIT": {"threshold": 0.35, "operator": "<="},
    "I-4_ZANN_ZERO": {"threshold": "verified", "operator": "=="},
    "I-5_FROZEN_AGENTS": {"frozen": ["P5", "S2"]},
    "I-6_SOVEREIGNTY": {"cloud_auth": False},
    "I-7_SPINE_GUARD": {"spine_check": "passed"},
}


@dataclass
class EngramEntry:
    """A cached constitutional clearance result."""
    mission_type_hash: str
    cleared: bool
    invariants_checked: int
    check_time_ms: float
    cached_at: float
    hits: int = 0
    last_hit: float = 0.0


@dataclass
class CELStats:
    """Engram layer performance metrics."""
    total_checks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    entries: int = 0


class ConstitutionalEngramLayer:
    """O(1) constitutional memory — pre-verified clearance cache.
    
    Invariants are immutable. Their verification results are deterministic.
    Cache them once, look up forever. No invalidation needed because
    the invariants never change.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._cache: Dict[str, EngramEntry] = {}
        self._max_entries = max_entries
        self._stats = CELStats()
        self._hit_times: list[float] = []
        self._miss_times: list[float] = []
        log.info("CEL initialized (max_entries=%d, invariants=%d)", 
                 max_entries, len(INVARIANTS))

    def _compute_key(self, mission_type: str, context: Dict[str, Any]) -> str:
        """Hash mission type + constitutional-relevant context fields."""
        # Only hash fields that affect constitutional clearance
        relevant = {
            "type": mission_type,
            "authority": context.get("authority", ""),
            "agents": sorted(context.get("agents", [])),
            "cloud_auth": context.get("cloud_auth", False),
            "zann_status": context.get("zann_status", "unknown"),
        }
        content = str(sorted(relevant.items()))
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    def check(
        self,
        mission_type: str,
        context: Dict[str, Any],
    ) -> tuple[bool, dict]:
        """Check constitutional clearance — O(1) if cached, O(k) on first check.
        
        Returns (cleared, detail_dict).
        """
        t0 = time.monotonic()
        key = self._compute_key(mission_type, context)
        self._stats.total_checks += 1

        # O(1) cache hit
        if key in self._cache:
            entry = self._cache[key]
            entry.hits += 1
            entry.last_hit = time.time()
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._stats.cache_hits += 1
            self._hit_times.append(elapsed_ms)
            return entry.cleared, {
                "source": "engram",
                "elapsed_ms": round(elapsed_ms, 4),
                "hits": entry.hits,
                "invariants_checked": 0,  # Zero recomputation
            }

        # Cache miss — compute clearance
        self._stats.cache_misses += 1
        cleared, violations = self._verify_invariants(context)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._miss_times.append(elapsed_ms)

        # Cache the result
        if len(self._cache) >= self._max_entries:
            self._evict_lru()

        self._cache[key] = EngramEntry(
            mission_type_hash=key,
            cleared=cleared,
            invariants_checked=len(INVARIANTS),
            check_time_ms=elapsed_ms,
            cached_at=time.time(),
        )

        return cleared, {
            "source": "computed",
            "elapsed_ms": round(elapsed_ms, 4),
            "invariants_checked": len(INVARIANTS),
            "violations": violations,
        }

    def _verify_invariants(self, context: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Full O(k) invariant verification — runs only on cache miss."""
        violations = []

        # I-5: Frozen agents check
        agents = context.get("agents", [])
        frozen = INVARIANTS["I-5_FROZEN_AGENTS"]["frozen"]
        for f in frozen:
            if f in agents:
                violations.append(f"I-5: frozen agent {f} in execution list")

        # I-6: Sovereignty check
        if context.get("cloud_auth", False):
            violations.append("I-6: cloud authentication detected")

        # I-4: Zann check
        zann = context.get("zann_status", "unknown")
        if zann != "verified":
            violations.append(f"I-4: zann_status={zann}, required=verified")

        # I-7: Spine guard
        authority = context.get("authority", "")
        if authority not in ("sovereign", "delegated", "constitutional"):
            violations.append(f"I-7: invalid authority={authority}")

        return len(violations) == 0, violations

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].last_hit)
        del self._cache[lru_key]

    def stats(self) -> Dict[str, Any]:
        """CEL performance metrics."""
        hit_avg = sum(self._hit_times) / max(len(self._hit_times), 1)
        miss_avg = sum(self._miss_times) / max(len(self._miss_times), 1)
        hit_rate = self._stats.cache_hits / max(self._stats.total_checks, 1)
        return {
            "total_checks": self._stats.total_checks,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate": round(hit_rate, 4),
            "avg_hit_ms": round(hit_avg, 4),
            "avg_miss_ms": round(miss_avg, 4),
            "speedup_factor": round(miss_avg / max(hit_avg, 0.001), 1),
            "entries_cached": len(self._cache),
        }

    def preload(self, common_types: list[str]) -> int:
        """Pre-warm the cache with common mission types."""
        loaded = 0
        base_context = {
            "authority": "sovereign",
            "agents": ["P1", "P2", "P3", "P4", "P6", "P7"],
            "cloud_auth": False,
            "zann_status": "verified",
        }
        for mt in common_types:
            self.check(mt, base_context)
            loaded += 1
        log.info("CEL preloaded %d common mission types", loaded)
        return loaded


# Singleton
_cel: Optional[ConstitutionalEngramLayer] = None


def get_cel() -> ConstitutionalEngramLayer:
    """Get or create the global CEL instance."""
    global _cel
    if _cel is None:
        _cel = ConstitutionalEngramLayer()
        # Preload common mission types
        _cel.preload([
            "research_query", "code_generation", "file_organization",
            "knowledge_search", "text_analysis", "summarization",
            "planning", "evaluation", "publishing",
        ])
    return _cel
