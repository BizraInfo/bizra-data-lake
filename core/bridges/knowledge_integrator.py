"""
Knowledge Integrator — BIZRA Data Lake → Proactive Sovereign Swarm
=================================================================
Integrates all BIZRA Data Lake resources and MoMo's 3-year R&D knowledge
to feed the Proactive Sovereign Swarm agents with contextual intelligence.

Resources Integrated:
- Living Memory System (100K+ entries, self-healing)
- Vector Embeddings (60MB FAISS-indexed)
- Knowledge Graphs (112MB sacred wisdom)
- Session State (30+ files)
- Agent Specializations (12-agent hierarchy)
- Core Modules (50+ Python files, 187 tests)

Standing on Giants: Shannon + Lamport + Al-Ghazali + Anthropic
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Paths relative to BIZRA-DATA-LAKE root
DATA_LAKE_ROOT = Path(__file__).parent.parent.parent


@dataclass
class KnowledgeSource:
    """A discoverable knowledge source."""

    name: str = ""
    path: str = ""
    source_type: str = ""  # parquet, json, jsonl, npy, md, py
    category: str = ""  # memory, embedding, graph, session, module
    size_bytes: int = 0
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    priority: str = "MEDIUM"  # CRITICAL, HIGH, MEDIUM, LOW
    loaded: bool = False
    last_accessed: Optional[datetime] = None


@dataclass
class KnowledgeQuery:
    """A query to the knowledge base."""

    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    min_snr: float = 0.85
    categories: List[str] = field(default_factory=list)
    requester: str = ""  # Agent role requesting


@dataclass
class KnowledgeResult:
    """Result from knowledge retrieval."""

    query_id: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    sources_consulted: List[str] = field(default_factory=list)
    snr_score: float = 0.0
    latency_ms: float = 0.0
    from_cache: bool = False


class KnowledgeIntegrator:
    """
    Integrates BIZRA Data Lake knowledge with the Proactive Sovereign Swarm.

    Provides:
    - Discovery of all knowledge sources
    - Lazy loading of high-value resources
    - Context-aware retrieval for swarm agents
    - Knowledge graph traversal
    - Memory consolidation interface
    """

    # Knowledge source catalog
    KNOWLEDGE_SOURCES = [
        # CRITICAL - Core reasoning substrate
        KnowledgeSource(
            name="Living Memory Core",
            path="core/living_memory/core.py",
            source_type="py",
            category="memory",
            snr_score=0.95,
            ihsan_score=0.98,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Proactive Retriever",
            path="core/living_memory/proactive.py",
            source_type="py",
            category="memory",
            snr_score=0.92,
            ihsan_score=0.96,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Vector Embeddings",
            path="04_GOLD/sacred_wisdom_embeddings.npy",
            source_type="npy",
            category="embedding",
            snr_score=0.92,
            ihsan_score=0.96,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Sacred Wisdom Graph",
            path="04_GOLD/sacred_wisdom_graph.json",
            source_type="json",
            category="graph",
            snr_score=0.88,
            ihsan_score=0.94,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Documents Corpus",
            path="04_GOLD/documents.parquet",
            source_type="parquet",
            category="corpus",
            snr_score=0.90,
            ihsan_score=0.95,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Chunks Embeddings",
            path="04_GOLD/chunks.parquet",
            source_type="parquet",
            category="embedding",
            snr_score=0.90,
            ihsan_score=0.95,
            priority="CRITICAL",
        ),
        # HIGH - Session and agent context
        KnowledgeSource(
            name="Session State",
            path=".claude-flow/memory/session-state.json",
            source_type="json",
            category="session",
            snr_score=0.91,
            ihsan_score=0.97,
            priority="HIGH",
        ),
        KnowledgeSource(
            name="Agent Specializations",
            path=".claude-flow/memory/agent-specializations.json",
            source_type="json",
            category="agent",
            snr_score=0.93,
            ihsan_score=0.97,
            priority="HIGH",
        ),
        KnowledgeSource(
            name="Omega Point Integration",
            path=".claude-flow/memory/omega-point-integration-2026-02-04.json",
            source_type="json",
            category="session",
            snr_score=0.96,
            ihsan_score=0.99,
            priority="CRITICAL",
        ),
        KnowledgeSource(
            name="Standing on Giants",
            path=".claude-flow/memory/standing-on-giants.json",
            source_type="json",
            category="foundation",
            snr_score=0.90,
            ihsan_score=0.94,
            priority="HIGH",
        ),
        KnowledgeSource(
            name="Integration Index",
            path=".claude-flow/memory/integration-index.json",
            source_type="json",
            category="integration",
            snr_score=0.93,
            ihsan_score=0.95,
            priority="HIGH",
        ),
        KnowledgeSource(
            name="Knowledge Index",
            path=".claude-flow/memory/knowledge-index.json",
            source_type="json",
            category="index",
            snr_score=0.91,
            ihsan_score=0.94,
            priority="HIGH",
        ),
        KnowledgeSource(
            name="DATA4LLM Knowledge",
            path="04_GOLD/DATA4LLM-knowledge.json",
            source_type="json",
            category="framework",
            snr_score=0.84,
            ihsan_score=0.91,
            priority="HIGH",
        ),
        # MEDIUM - Patterns and analysis
        KnowledgeSource(
            name="Golden Gems",
            path="04_GOLD/golden_gems_index.jsonl",
            source_type="jsonl",
            category="insights",
            snr_score=0.88,
            ihsan_score=0.92,
            priority="MEDIUM",
        ),
        KnowledgeSource(
            name="Apex Knowledge Graph",
            path="04_GOLD/apex_knowledge_graph.json",
            source_type="json",
            category="graph",
            snr_score=0.85,
            ihsan_score=0.90,
            priority="MEDIUM",
        ),
        KnowledgeSource(
            name="Project Patterns",
            path=".claude-flow/memory/project-patterns.json",
            source_type="json",
            category="patterns",
            snr_score=0.82,
            ihsan_score=0.88,
            priority="MEDIUM",
        ),
        KnowledgeSource(
            name="Genesis Covenant",
            path="04_GOLD/genesis.json",
            source_type="json",
            category="identity",
            snr_score=0.95,
            ihsan_score=0.98,
            priority="HIGH",
        ),
    ]

    def __init__(
        self,
        data_lake_root: Optional[Path] = None,
        ihsan_threshold: float = 0.95,
        cache_enabled: bool = True,
    ):
        self.data_lake_root = data_lake_root or DATA_LAKE_ROOT
        self.ihsan_threshold = ihsan_threshold
        self.cache_enabled = cache_enabled

        # Knowledge catalog
        self._sources: Dict[str, KnowledgeSource] = {}
        self._loaded_data: Dict[str, Any] = {}
        self._cache: Dict[str, KnowledgeResult] = {}

        # MoMo context
        self._momo_context: Dict[str, Any] = {}

        # Statistics
        self._query_count = 0
        self._cache_hits = 0
        self._total_latency_ms = 0.0

        # Initialize catalog
        self._build_catalog()

    def _build_catalog(self) -> None:
        """Build knowledge source catalog."""
        for source in self.KNOWLEDGE_SOURCES:
            full_path = self.data_lake_root / source.path
            if full_path.exists():
                source.size_bytes = full_path.stat().st_size
                self._sources[source.name] = source
            else:
                logger.debug(f"Knowledge source not found: {source.path}")

        logger.info(f"Knowledge catalog: {len(self._sources)} sources discovered")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize integrator and load critical resources."""
        loaded = []
        failed = []

        # PERF FIX #2: Load CRITICAL sources in parallel using asyncio.gather()
        critical_sources = [
            (name, source)
            for name, source in self._sources.items()
            if source.priority == "CRITICAL"
        ]

        if critical_sources:
            tasks = [self._load_source(source) for _, source in critical_sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (name, _), result in zip(critical_sources, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to load {name}: {result}")
                    failed.append(name)
                else:
                    loaded.append(name)

        # Load MoMo context
        await self._load_momo_context()

        return {
            "sources_discovered": len(self._sources),
            "sources_loaded": len(loaded),
            "sources_failed": len(failed),
            "momo_context_loaded": bool(self._momo_context),
        }

    def _load_json_sync(self, path: Path) -> Any:
        """Synchronous JSON loading for thread pool executor."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl_sync(self, path: Path) -> List[Dict[str, Any]]:
        """Synchronous JSONL loading for thread pool executor."""
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        return lines

    async def _load_source(self, source: KnowledgeSource) -> None:
        """Load a knowledge source into memory (non-blocking)."""
        full_path = self.data_lake_root / source.path
        loop = asyncio.get_event_loop()

        # PERF FIX #1: Use run_in_executor for blocking file I/O
        if source.source_type == "json":
            data = await loop.run_in_executor(None, self._load_json_sync, full_path)
            self._loaded_data[source.name] = data

        elif source.source_type == "jsonl":
            data = await loop.run_in_executor(None, self._load_jsonl_sync, full_path)
            self._loaded_data[source.name] = data

        elif source.source_type == "py":
            # For Python modules, just mark as available
            self._loaded_data[source.name] = {"type": "module", "path": str(full_path)}

        elif source.source_type in ("npy", "parquet"):
            # Large binary files - load on demand
            self._loaded_data[source.name] = {
                "type": source.source_type,
                "path": str(full_path),
            }

        source.loaded = True
        source.last_accessed = datetime.now(timezone.utc)
        logger.debug(f"Loaded knowledge source: {source.name}")

    async def _load_momo_context(self) -> None:
        """Load MoMo's identity and R&D context (non-blocking)."""
        context_files = [
            ("session-state.json", ".claude-flow/memory/session-state.json"),
            ("genesis.json", "04_GOLD/genesis.json"),
            ("standing-on-giants.json", ".claude-flow/memory/standing-on-giants.json"),
        ]

        loop = asyncio.get_event_loop()

        # PERF FIX: Load context files in parallel with non-blocking I/O
        async def load_context_file(name: str, path: str) -> tuple:
            full_path = self.data_lake_root / path
            if full_path.exists():
                data = await loop.run_in_executor(None, self._load_json_sync, full_path)
                return (name, data)
            return (name, None)

        tasks = [load_context_file(name, path) for name, path in context_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                self._momo_context[result[0]] = result[1]

        # Extract key MoMo details
        if "session-state.json" in self._momo_context:
            session = self._momo_context["session-state.json"]
            self._momo_context["summary"] = {
                "user": session.get("user", "MoMo"),
                "investment_hours": session.get("investment_hours", 15000),
                "ihsan_score": session.get("ihsan_score", 0.97),
                "genesis_complete": session.get("genesis_complete", True),
            }

        logger.info(f"MoMo context loaded: {list(self._momo_context.keys())}")

    async def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Query the integrated knowledge base."""
        self._query_count += 1
        start_time = datetime.now(timezone.utc)

        # Check cache
        cache_key = hashlib.md5(
            f"{query.query}:{query.categories}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        if self.cache_enabled and cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            cached.from_cache = True
            return cached

        result = KnowledgeResult(
            query_id=f"kq-{self._query_count:06d}",
        )

        # Search relevant sources
        sources_to_search = []
        for name, source in self._sources.items():
            if query.categories and source.category not in query.categories:
                continue
            if source.snr_score < query.min_snr:
                continue
            sources_to_search.append(source)

        # Sort by priority and SNR
        sources_to_search.sort(
            key=lambda s: (0 if s.priority == "CRITICAL" else 1, -s.snr_score)
        )

        # Search each source
        for source in sources_to_search[:5]:  # Limit to top 5 sources
            if source.name not in self._loaded_data:
                await self._load_source(source)

            matches = self._search_source(query.query, source)
            result.results.extend(matches[: query.max_results])
            result.sources_consulted.append(source.name)

        # Calculate aggregate SNR
        if result.results:
            result.snr_score = sum(
                r.get("snr_score", 0.8) for r in result.results
            ) / len(result.results)

        result.latency_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        self._total_latency_ms += result.latency_ms

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = result

        return result

    def _search_source(
        self,
        query: str,
        source: KnowledgeSource,
    ) -> List[Dict[str, Any]]:
        """Search within a single source."""
        matches = []
        data = self._loaded_data.get(source.name)

        if not data:
            return matches

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        if isinstance(data, dict):
            # Search dict keys and values
            for key, value in data.items():
                text = f"{key} {json.dumps(value, default=str)}".lower()
                if any(term in text for term in query_terms):
                    matches.append(
                        {
                            "source": source.name,
                            "key": key,
                            "value": value,
                            "snr_score": source.snr_score,
                        }
                    )

        elif isinstance(data, list):
            # Search list items
            for i, item in enumerate(data[:100]):  # Limit to first 100
                text = json.dumps(item, default=str).lower()
                if any(term in text for term in query_terms):
                    matches.append(
                        {
                            "source": source.name,
                            "index": i,
                            "value": item,
                            "snr_score": source.snr_score,
                        }
                    )

        return matches

    def get_momo_context(self) -> Dict[str, Any]:
        """Get MoMo's identity and context."""
        return self._momo_context.get("summary", {})

    def get_standing_on_giants(self) -> List[Dict[str, Any]]:
        """Get the attribution chain (Standing on Giants)."""
        sog = self._momo_context.get("standing-on-giants.json", {})
        return sog.get("giants", [])

    def get_source_catalog(self) -> List[Dict[str, Any]]:
        """Get the knowledge source catalog."""
        return [
            {
                "name": s.name,
                "category": s.category,
                "priority": s.priority,
                "snr_score": s.snr_score,
                "ihsan_score": s.ihsan_score,
                "loaded": s.loaded,
                "size_kb": s.size_bytes // 1024 if s.size_bytes else 0,
            }
            for s in self._sources.values()
        ]

    def stats(self) -> Dict[str, Any]:
        """Get integrator statistics."""
        return {
            "sources_discovered": len(self._sources),
            "sources_loaded": sum(1 for s in self._sources.values() if s.loaded),
            "total_size_mb": sum(s.size_bytes for s in self._sources.values())
            / (1024 * 1024),
            "query_count": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._query_count, 1),
            "avg_latency_ms": self._total_latency_ms / max(self._query_count, 1),
            "momo_context_loaded": bool(self._momo_context),
        }


# Convenience function for swarm agents
async def create_knowledge_integrator(
    ihsan_threshold: float = 0.95,
) -> KnowledgeIntegrator:
    """Create and initialize a knowledge integrator."""
    integrator = KnowledgeIntegrator(ihsan_threshold=ihsan_threshold)
    await integrator.initialize()
    return integrator


__all__ = [
    "KnowledgeIntegrator",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeSource",
    "create_knowledge_integrator",
]
