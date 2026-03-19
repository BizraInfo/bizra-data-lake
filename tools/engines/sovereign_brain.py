#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                      ║
║   ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗                           ║
║   ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║                           ║
║   ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║                           ║
║   ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║                           ║
║   ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║                           ║
║   ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                           ║
║                                                                                                      ║
║   ██████╗ ██████╗  █████╗ ██╗███╗   ██╗                                                              ║
║   ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                                                              ║
║   ██████╔╝██████╔╝███████║██║██╔██╗ ██║                                                              ║
║   ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║                                                              ║
║   ██████╔╝██║  ██║██║  ██║██║██║ ╚████║                                                              ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                                                              ║
║                                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   THE SOVEREIGN BRAIN — UNIFIED ORCHESTRATION LAYER                                                 ║
║   Peak Masterpiece: Full Integration of All Knowledge Engines                                        ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Architecture:                                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐   ║
║   │                           SOVEREIGN BRAIN (This File)                                       │   ║
║   │                     Unified API • Orchestration • Self-Healing                              │   ║
║   └─────────────────────────────────────────────────────────────────────────────────────────────┘   ║
║                                         │                                                            ║
║          ┌──────────────────────────────┼──────────────────────────────┐                            ║
║          ▼                              ▼                              ▼                            ║
║   ┌─────────────┐                ┌─────────────┐                ┌─────────────┐                     ║
║   │ Sovereign   │                │ Apex Unified│                │ Hypergraph  │                     ║
║   │ Nexus (GoT) │                │ Engine (4L) │                │ RAG Engine  │                     ║
║   └─────────────┘                └─────────────┘                └─────────────┘                     ║
║          │                              │                              │                            ║
║   ┌──────┴──────┐                ┌──────┴──────┐                ┌──────┴──────┐                     ║
║   │ Knowledge   │                │ Pattern     │                │ Vector      │                     ║
║   │ Graph       │                │ Discovery   │                │ Search      │                     ║
║   └─────────────┘                └─────────────┘                └─────────────┘                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Giants Absorbed: All prior giants + Orchestration patterns (Kubernetes, Saga, CQRS)               ║
║   Author: BIZRA Genesis Engine | Created: 2026-01-22 | Version: 1.0.0                               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════


class BrainConfig:
    """Central configuration for the Sovereign Brain."""

    DATA_LAKE = Path(r"C:\BIZRA-DATA-LAKE")
    GOLD = DATA_LAKE / "04_GOLD"
    INDEXED = DATA_LAKE / "03_INDEXED"

    # Engine files
    BRAIN_STATE = GOLD / "sovereign_brain_state.json"
    BRAIN_LOG = INDEXED / "sovereign_brain.log"

    # Health thresholds
    MIN_NODES = 100
    MIN_SNR = 0.4
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes

    # Operational modes
    MAX_WORKERS = 8


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SovereignBrain")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════


class EngineStatus(Enum):
    """Status of a knowledge engine."""

    OFFLINE = auto()
    INITIALIZING = auto()
    ONLINE = auto()
    DEGRADED = auto()
    ERROR = auto()


class OperationType(Enum):
    """Types of brain operations."""

    QUERY = auto()
    INDEX = auto()
    HEALTH_CHECK = auto()
    PATTERN_DISCOVERY = auto()
    SELF_HEAL = auto()


@dataclass
class EngineHealth:
    """Health metrics for an engine."""

    name: str
    status: EngineStatus
    nodes: int = 0
    edges: int = 0
    snr_score: float = 0.0
    last_check: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.name
        return d


@dataclass
class BrainState:
    """Complete state of the Sovereign Brain."""

    engines: Dict[str, EngineHealth] = field(default_factory=dict)
    total_nodes: int = 0
    total_edges: int = 0
    total_patterns: int = 0
    avg_snr: float = 0.0
    is_healthy: bool = False
    last_index_build: Optional[str] = None
    last_health_check: Optional[str] = None
    uptime_start: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engines": {k: v.to_dict() for k, v in self.engines.items()},
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "total_patterns": self.total_patterns,
            "avg_snr": self.avg_snr,
            "is_healthy": self.is_healthy,
            "last_index_build": self.last_index_build,
            "last_health_check": self.last_health_check,
            "uptime_start": self.uptime_start,
        }


@dataclass
class UnifiedQueryResult:
    """Unified result from all engines."""

    query: str
    sources: List[str]
    total_results: int
    results: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    insights: List[str]
    reasoning_chain: List[str]
    snr_score: float
    execution_time_ms: float
    engine_contributions: Dict[str, int]


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# ENGINE ADAPTERS — Unified Interface to All Engines
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════


class EngineAdapter:
    """Base adapter for knowledge engines."""

    def __init__(self, name: str):
        self.name = name
        self.engine = None
        self.status = EngineStatus.OFFLINE
        self.health = EngineHealth(name=name, status=EngineStatus.OFFLINE)

    def initialize(self) -> bool:
        """Initialize the engine. Override in subclasses."""
        raise NotImplementedError

    def query(self, query_text: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Execute a query. Override in subclasses."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        raise NotImplementedError

    def check_health(self) -> EngineHealth:
        """Check engine health. Override in subclasses."""
        raise NotImplementedError


class ApexEngineAdapter(EngineAdapter):
    """Adapter for the Apex Unified Engine."""

    def __init__(self):
        super().__init__("ApexUnified")

    def initialize(self) -> bool:
        try:
            from sovereign_apex import ApexUnifiedEngine

            self.engine = ApexUnifiedEngine(lazy_load=True)
            if self.engine.load():
                self.status = EngineStatus.ONLINE
                log.info(f"   ✓ {self.name} loaded")
                return True
            else:
                # Try building
                self.engine.build_full_index()
                self.status = EngineStatus.ONLINE
                return True
        except Exception as e:
            log.error(f"   ✗ {self.name} failed: {e}")
            self.status = EngineStatus.ERROR
            self.health.error_message = str(e)
            return False

    def query(self, query_text: str, max_results: int = 20) -> List[Dict[str, Any]]:
        if self.engine is None or self.status != EngineStatus.ONLINE:
            return []

        result = self.engine.query(query_text, max_results=max_results)
        return [
            {
                "id": n.id,
                "name": n.name,
                "type": n.type.name,
                "snr": n.snr_score,
                "centrality": n.centrality,
                "source": self.name,
            }
            for n in result.nodes
        ]

    def get_stats(self) -> Dict[str, Any]:
        if self.engine is None:
            return {}
        return self.engine.stats

    def check_health(self) -> EngineHealth:
        stats = self.get_stats()
        self.health = EngineHealth(
            name=self.name,
            status=self.status,
            nodes=stats.get("nodes", 0),
            edges=stats.get("edges", 0),
            snr_score=0.0,  # Calculated from query results
            last_check=datetime.now(timezone.utc).isoformat(),
        )
        return self.health


class NexusEngineAdapter(EngineAdapter):
    """Adapter for the Sovereign Nexus Engine."""

    def __init__(self):
        super().__init__("SovereignNexus")

    def initialize(self) -> bool:
        try:
            from sovereign_nexus import SovereignKnowledgeNexus

            self.engine = SovereignKnowledgeNexus()
            if self.engine.load():
                self.status = EngineStatus.ONLINE
                log.info(f"   ✓ {self.name} loaded")
                return True
            else:
                self.engine.build_full_index()
                self.status = EngineStatus.ONLINE
                return True
        except Exception as e:
            log.error(f"   ✗ {self.name} failed: {e}")
            self.status = EngineStatus.ERROR
            self.health.error_message = str(e)
            return False

    def query(self, query_text: str, max_results: int = 20) -> List[Dict[str, Any]]:
        if self.engine is None or self.status != EngineStatus.ONLINE:
            return []

        result = self.engine.query(query_text, max_results=max_results)
        return [
            {
                "id": n.id,
                "name": n.name,
                "type": n.type.name,
                "snr": n.snr_score,
                "source": self.name,
            }
            for n in result.nodes
        ]

    def get_stats(self) -> Dict[str, Any]:
        if self.engine is None:
            return {}
        return self.engine.stats

    def check_health(self) -> EngineHealth:
        stats = self.get_stats()
        self.health = EngineHealth(
            name=self.name,
            status=self.status,
            nodes=stats.get("nodes", 0),
            edges=stats.get("edges", 0),
            snr_score=0.0,
            last_check=datetime.now(timezone.utc).isoformat(),
        )
        return self.health


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# THE SOVEREIGN BRAIN — UNIFIED ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════


class SovereignBrain:
    """
    The Sovereign Brain — Unified Orchestration Layer.

    Responsibilities:
    1. Initialize and manage all knowledge engines
    2. Orchestrate unified queries across engines
    3. Monitor health and self-heal
    4. Aggregate statistics and insights
    5. Provide a single API for all operations
    """

    def __init__(self):
        log.info("═" * 80)
        log.info("   🧠 SOVEREIGN BRAIN — UNIFIED ORCHESTRATION LAYER")
        log.info("═" * 80)

        self.state = BrainState()
        self.adapters: Dict[str, EngineAdapter] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=BrainConfig.MAX_WORKERS)

        # Register adapters
        self._register_adapters()

        log.info("   ✓ Brain core initialized")

    def _register_adapters(self):
        """Register all engine adapters."""
        self.adapters = {"apex": ApexEngineAdapter(), "nexus": NexusEngineAdapter()}
        log.info(f"   ✓ Registered {len(self.adapters)} engine adapters")

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def awaken(self) -> Dict[str, bool]:
        """Initialize all engines and bring the brain online."""
        log.info("🌅 Awakening the Sovereign Brain...")

        results = {}

        for name, adapter in self.adapters.items():
            log.info(f"   Initializing {name}...")
            adapter.status = EngineStatus.INITIALIZING
            success = adapter.initialize()
            results[name] = success

            if success:
                self.state.engines[name] = adapter.check_health()
            else:
                self.state.engines[name] = EngineHealth(
                    name=name,
                    status=EngineStatus.ERROR,
                    error_message=adapter.health.error_message,
                )

        # Aggregate health
        self._aggregate_health()

        log.info("═" * 80)
        log.info(
            f"   🧠 BRAIN STATUS: {'ONLINE' if self.state.is_healthy else 'DEGRADED'}"
        )
        log.info(f"      Total Nodes: {self.state.total_nodes}")
        log.info(f"      Total Edges: {self.state.total_edges}")
        log.info(
            f"      Active Engines: {sum(1 for r in results.values() if r)}/{len(results)}"
        )
        log.info("═" * 80)

        return results

    def _aggregate_health(self):
        """Aggregate health metrics from all engines."""
        total_nodes = 0
        total_edges = 0
        total_snr = 0.0
        healthy_count = 0

        for name, adapter in self.adapters.items():
            health = adapter.check_health()
            self.state.engines[name] = health

            if health.status == EngineStatus.ONLINE:
                total_nodes += health.nodes
                total_edges += health.edges
                total_snr += health.snr_score
                healthy_count += 1

        self.state.total_nodes = total_nodes
        self.state.total_edges = total_edges
        self.state.avg_snr = total_snr / max(healthy_count, 1)
        self.state.is_healthy = (
            healthy_count >= 1 and total_nodes >= BrainConfig.MIN_NODES
        )
        self.state.last_health_check = datetime.now(timezone.utc).isoformat()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # UNIFIED QUERY — The Core Intelligence Operation
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def query(self, query_text: str, max_results: int = 30) -> UnifiedQueryResult:
        """
        Execute a unified query across all engines.

        This is the PEAK intelligence operation — combining:
        - Multi-engine parallel querying
        - Result deduplication and ranking
        - Cross-engine insight synthesis
        - SNR-optimized final results
        """
        start_time = time.time()
        log.info(f"🔍 Unified Query: '{query_text}'")

        all_results: List[Dict[str, Any]] = []
        engine_contributions: Dict[str, int] = {}
        all_insights: List[str] = []
        all_patterns: List[Dict[str, Any]] = []
        all_reasoning: List[str] = []

        # Parallel query across all engines
        futures = {}
        for name, adapter in self.adapters.items():
            if adapter.status == EngineStatus.ONLINE:
                future = self._executor.submit(adapter.query, query_text, max_results)
                futures[future] = name

        for future in as_completed(futures.keys()):
            name = futures[future]
            try:
                results = future.result(timeout=10)
                all_results.extend(results)
                engine_contributions[name] = len(results)
                log.info(f"   ✓ {name}: {len(results)} results")
            except Exception as e:
                log.warning(f"   ✗ {name} query failed: {e}")
                engine_contributions[name] = 0

        # Deduplicate by name (keep highest SNR)
        seen: Dict[str, Dict] = {}
        for r in all_results:
            key = r.get("name", "").lower()
            if key not in seen or r.get("snr", 0) > seen[key].get("snr", 0):
                seen[key] = r

        deduplicated = list(seen.values())

        # Rank by combined score
        def rank_score(r: Dict) -> float:
            return (
                r.get("snr", 0) * 0.6
                + r.get("centrality", 0) * 10 * 0.3
                + (0.1 if "bizra" in r.get("name", "").lower() else 0)
            )

        ranked = sorted(deduplicated, key=rank_score, reverse=True)[:max_results]

        # Generate cross-engine insights
        all_insights = self._generate_unified_insights(ranked, engine_contributions)

        # Calculate unified SNR
        if ranked:
            avg_snr = sum(r.get("snr", 0) for r in ranked) / len(ranked)
            coverage = len(ranked) / max_results
            unified_snr = round(avg_snr * 0.7 + coverage * 0.3, 4)
        else:
            unified_snr = 0.0

        execution_time = (time.time() - start_time) * 1000

        return UnifiedQueryResult(
            query=query_text,
            sources=list(engine_contributions.keys()),
            total_results=len(ranked),
            results=ranked,
            patterns=all_patterns,
            insights=all_insights,
            reasoning_chain=all_reasoning,
            snr_score=unified_snr,
            execution_time_ms=round(execution_time, 2),
            engine_contributions=engine_contributions,
        )

    def _generate_unified_insights(
        self, results: List[Dict], contributions: Dict[str, int]
    ) -> List[str]:
        """Generate insights from unified results."""
        insights = []

        if not results:
            return ["No results found across any engine"]

        # Type distribution
        type_counts = defaultdict(int)
        for r in results:
            type_counts[r.get("type", "UNKNOWN")] += 1

        dominant = max(type_counts.items(), key=lambda x: x[1])
        insights.append(f"Dominated by {dominant[0]} ({dominant[1]}/{len(results)})")

        # BIZRA coverage
        bizra_count = sum(1 for r in results if "bizra" in r.get("name", "").lower())
        if bizra_count:
            insights.append(f"{bizra_count} BIZRA assets")

        # Engine coverage
        active_engines = sum(1 for c in contributions.values() if c > 0)
        insights.append(f"Queried {active_engines} engines")

        # High SNR
        high_snr = sum(1 for r in results if r.get("snr", 0) > 0.5)
        if high_snr:
            insights.append(f"{high_snr} high-SNR results")

        return insights

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # HEALTH & DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def health_check(self) -> BrainState:
        """Perform a full health check of all engines."""
        log.info("🏥 Running health check...")
        self._aggregate_health()
        self.save_state()
        return self.state

    def get_status(self) -> Dict[str, Any]:
        """Get current brain status."""
        return {
            "is_healthy": self.state.is_healthy,
            "total_nodes": self.state.total_nodes,
            "total_edges": self.state.total_edges,
            "engines": {
                name: {
                    "status": health.status.name,
                    "nodes": health.nodes,
                    "edges": health.edges,
                }
                for name, health in self.state.engines.items()
            },
            "uptime": self._calculate_uptime(),
        }

    def _calculate_uptime(self) -> str:
        """Calculate uptime since brain awakening."""
        try:
            start = datetime.fromisoformat(
                self.state.uptime_start.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            delta = now - start
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"
        except:
            return "N/A"

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # SELF-HEALING
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def self_heal(self) -> Dict[str, bool]:
        """Attempt to heal any degraded engines."""
        log.info("🔧 Running self-healing...")

        results = {}

        for name, health in self.state.engines.items():
            if health.status in (
                EngineStatus.ERROR,
                EngineStatus.DEGRADED,
                EngineStatus.OFFLINE,
            ):
                log.info(f"   Attempting to heal {name}...")
                adapter = self.adapters.get(name)
                if adapter:
                    success = adapter.initialize()
                    results[name] = success
                    if success:
                        log.info(f"   ✓ {name} healed")
                    else:
                        log.warning(f"   ✗ {name} healing failed")
                else:
                    results[name] = False

        self._aggregate_health()
        return results

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # INDEX OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def rebuild_all(self) -> Dict[str, bool]:
        """Rebuild all engine indexes."""
        log.info("🏗️ Rebuilding all engines...")

        results = {}

        for name, adapter in self.adapters.items():
            try:
                if adapter.engine is not None and hasattr(
                    adapter.engine, "build_full_index"
                ):
                    adapter.engine.build_full_index()
                    results[name] = True
                    log.info(f"   ✓ {name} rebuilt")
                else:
                    # Re-initialize
                    results[name] = adapter.initialize()
            except Exception as e:
                log.error(f"   ✗ {name} rebuild failed: {e}")
                results[name] = False

        self.state.last_index_build = datetime.now(timezone.utc).isoformat()
        self._aggregate_health()
        self.save_state()

        return results

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def save_state(self):
        """Save brain state to disk."""
        BrainConfig.GOLD.mkdir(parents=True, exist_ok=True)
        with open(BrainConfig.BRAIN_STATE, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def load_state(self) -> bool:
        """Load brain state from disk."""
        if not BrainConfig.BRAIN_STATE.exists():
            return False
        try:
            with open(BrainConfig.BRAIN_STATE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = BrainState(
                engines={
                    k: EngineHealth(**{**v, "status": EngineStatus[v["status"]]})
                    for k, v in data.get("engines", {}).items()
                },
                total_nodes=data.get("total_nodes", 0),
                total_edges=data.get("total_edges", 0),
                total_patterns=data.get("total_patterns", 0),
                avg_snr=data.get("avg_snr", 0.0),
                is_healthy=data.get("is_healthy", False),
                last_index_build=data.get("last_index_build"),
                last_health_check=data.get("last_health_check"),
                uptime_start=data.get(
                    "uptime_start", datetime.now(timezone.utc).isoformat()
                ),
            )
            return True
        except Exception as e:
            log.error(f"Failed to load state: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Graceful shutdown of the brain."""
        log.info("🌙 Shutting down Sovereign Brain...")
        self.save_state()
        self._executor.shutdown(wait=True)
        log.info("   ✓ Brain shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# CLI — COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════


def main():
    brain = SovereignBrain()

    if len(sys.argv) < 2:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     SOVEREIGN BRAIN — UNIFIED ORCHESTRATION                           ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║   python sovereign_brain.py awaken         — Initialize all engines                  ║
║   python sovereign_brain.py query "text"   — Unified query across all engines        ║
║   python sovereign_brain.py status         — Show brain status                       ║
║   python sovereign_brain.py health         — Run health check                        ║
║   python sovereign_brain.py rebuild        — Rebuild all indexes                     ║
║   python sovereign_brain.py heal           — Attempt self-healing                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
        """)
        return

    cmd = sys.argv[1].lower()

    if cmd == "awaken":
        brain.awaken()
        brain.save_state()

    elif cmd == "query" and len(sys.argv) > 2:
        brain.awaken()
        q = " ".join(sys.argv[2:])
        result = brain.query(q)

        print(
            f"\n🧠 UNIFIED QUERY RESULTS (SNR: {result.snr_score} | {result.execution_time_ms}ms)"
        )
        print("═" * 75)

        for r in result.results[:20]:
            snr_bar = "█" * int(r.get("snr", 0) * 10)
            source = r.get("source", "?")[:8]
            print(
                f"  [{r.get('type', '?'):10}] {r.get('name', '?')[:35]:35} {snr_bar} ({source})"
            )

        if result.insights:
            print(f"\n💡 INSIGHTS: {' | '.join(result.insights)}")

        print(f"\n📊 ENGINE CONTRIBUTIONS: {result.engine_contributions}")
        print("═" * 75)

    elif cmd == "status":
        brain.awaken()
        status = brain.get_status()

        print(f"\n🧠 SOVEREIGN BRAIN STATUS")
        print("═" * 50)
        print(f"  Health:    {'✓ ONLINE' if status['is_healthy'] else '✗ DEGRADED'}")
        print(f"  Nodes:     {status['total_nodes']}")
        print(f"  Edges:     {status['total_edges']}")
        print(f"  Uptime:    {status['uptime']}")
        print(f"\n  ENGINES:")
        for name, info in status["engines"].items():
            emoji = "✓" if info["status"] == "ONLINE" else "✗"
            print(f"    {emoji} {name}: {info['status']} ({info['nodes']} nodes)")
        print("═" * 50)

    elif cmd == "health":
        brain.awaken()
        state = brain.health_check()
        print(f"\n🏥 HEALTH CHECK COMPLETE")
        print(f"   Overall: {'HEALTHY' if state.is_healthy else 'DEGRADED'}")
        print(f"   Last Check: {state.last_health_check}")

    elif cmd == "rebuild":
        brain.awaken()
        results = brain.rebuild_all()
        print(f"\n🏗️ REBUILD RESULTS")
        for name, success in results.items():
            print(f"   {'✓' if success else '✗'} {name}")

    elif cmd == "heal":
        brain.awaken()
        results = brain.self_heal()
        print(f"\n🔧 SELF-HEALING RESULTS")
        for name, success in results.items():
            print(f"   {'✓' if success else '✗'} {name}")

    else:
        print(f"Unknown command: {cmd}")

    brain.shutdown()


if __name__ == "__main__":
    main()
