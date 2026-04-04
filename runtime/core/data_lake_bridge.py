"""
BIZRA Data Lake Bridge
Provides unified memory access to BIZRA-DATA-LAKE via MCP protocol.

This module bridges the Dual-Agentic system with the Data Lake's
709k-node hypergraph for cross-domain knowledge retrieval.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependencies - graceful degradation if not installed
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log missing dependencies
if not HTTPX_AVAILABLE:
    logger.warning("httpx not installed - Data Lake MCP bridge will be disabled")
if not YAML_AVAILABLE:
    logger.warning("PyYAML not installed - config loading will be disabled")

# Environment configuration
DATA_LAKE_MCP_URL = os.getenv("DATA_LAKE_MCP_URL", "https://localhost:8443")
DATA_LAKE_PATH = os.getenv("DATA_LAKE_PATH", "/mnt/c/BIZRA-DATA-LAKE")
DATA_LAKE_TIMEOUT = int(os.getenv("DATA_LAKE_TIMEOUT", "30"))


class MemoryTier(Enum):
    """Unified memory tiers (M1-M6 from Data Lake, L1-L5 from Dual-Agentic)."""

    # Data Lake tiers
    M1_TASKMASTER = "M1"  # Current session context
    M2_SHORT_TERM = "M2"  # Short-term episodic
    M3_SEMANTIC = "M3"  # Medium-term semantic
    M4_PROCEDURAL = "M4"  # Long-term procedural
    M5_HISTORICAL = "M5"  # Historical archive
    M6_SOVEREIGN = "M6"  # Cross-domain omniscient

    # Dual-Agentic tiers (mapped)
    L1_IMMEDIATE = "L1"  # → M1
    L2_WORKING = "L2"  # → M2
    L3_EPISODIC = "L3"  # → M3
    L4_SEMANTIC = "L4"  # → M4
    L5_PROCEDURAL = "L5"  # → M5


# Tier mapping: Dual-Agentic → Data Lake
TIER_MAPPING = {
    MemoryTier.L1_IMMEDIATE: MemoryTier.M1_TASKMASTER,
    MemoryTier.L2_WORKING: MemoryTier.M2_SHORT_TERM,
    MemoryTier.L3_EPISODIC: MemoryTier.M3_SEMANTIC,
    MemoryTier.L4_SEMANTIC: MemoryTier.M4_PROCEDURAL,
    MemoryTier.L5_PROCEDURAL: MemoryTier.M5_HISTORICAL,
}


@dataclass
class KnowledgeResult:
    """Result from a knowledge retrieval query."""

    query: str
    results: List[Dict[str, Any]]
    tier: MemoryTier
    timestamp: str
    latency_ms: float
    total_count: int
    source: str = "data_lake"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": self.results,
            "tier": self.tier.value,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "total_count": self.total_count,
            "source": self.source,
        }


@dataclass
class DataLakeStatus:
    """Status of the Data Lake connection."""

    online: bool
    url: str
    nodes: int = 709000
    edges: int = 1400000
    last_check: str = ""
    error: Optional[str] = None


class DataLakeBridge:
    """
    Bridge to BIZRA-DATA-LAKE for unified memory access.

    Provides:
    - MCP protocol communication (JSON-RPC 2.0 over HTTPS)
    - Knowledge retrieval from 709k-node hypergraph
    - Memory tier mapping (L1-L5 ↔ M1-M6)
    - Sovereign (M6) cross-domain queries
    """

    def __init__(
        self,
        url: str = DATA_LAKE_MCP_URL,
        timeout: int = DATA_LAKE_TIMEOUT,
        verify_ssl: bool = False,  # Self-signed cert
    ):
        self.url = url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._client: Optional[httpx.AsyncClient] = None
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load data lake configuration from YAML."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed - using default config")
            self._config = {}
            return

        config_path = Path(__file__).parent.parent / "config" / "data_lake.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)
                logger.info("Loaded data lake config from %s", config_path)
        else:
            logger.warning("Data lake config not found at %s", config_path)
            self._config = {}

    async def _get_client(self):
        """Get or create async HTTP client."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Install with: pip install httpx")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> DataLakeStatus:
        """Check if the Data Lake MCP bridge is online."""
        try:
            client = await self._get_client()
            # Simple JSON-RPC ping
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            }
            response = await client.post(self.url, json=payload)

            if response.status_code == 200:
                return DataLakeStatus(
                    online=True,
                    url=self.url,
                    last_check=datetime.utcnow().isoformat(),
                )
            else:
                return DataLakeStatus(
                    online=False,
                    url=self.url,
                    last_check=datetime.utcnow().isoformat(),
                    error=f"HTTP {response.status_code}",
                )
        except Exception as e:
            logger.error("Data Lake health check failed: %s", e)
            return DataLakeStatus(
                online=False,
                url=self.url,
                last_check=datetime.utcnow().isoformat(),
                error=str(e),
            )

    async def knowledge_retrieve(
        self,
        query: str,
        tier: MemoryTier = MemoryTier.M6_SOVEREIGN,
        limit: int = 10,
    ) -> KnowledgeResult:
        """
        Query the Data Lake hypergraph for knowledge.

        Args:
            query: Semantic search query
            tier: Memory tier to query (default: M6 Sovereign for cross-domain)
            limit: Maximum results to return

        Returns:
            KnowledgeResult with matching knowledge nodes
        """
        start_time = datetime.utcnow()

        try:
            client = await self._get_client()

            # MCP JSON-RPC 2.0 request
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "knowledge_retrieve",
                    "arguments": {
                        "query": query,
                        "tier": tier.value,
                        "limit": limit,
                    },
                },
                "id": 1,
            }

            response = await client.post(self.url, json=payload)
            response.raise_for_status()

            data = response.json()

            # Parse MCP response
            if "result" in data:
                result_data = data["result"]
                results = result_data.get("content", [])
                if isinstance(results, str):
                    # Text response - wrap in list
                    results = [{"text": results}]
                elif isinstance(results, list) and len(results) > 0:
                    # Extract text from content blocks
                    parsed_results = []
                    for item in results:
                        if isinstance(item, dict) and "text" in item:
                            try:
                                # Try parsing as JSON
                                parsed = json.loads(item["text"])
                                if isinstance(parsed, list):
                                    parsed_results.extend(parsed)
                                else:
                                    parsed_results.append(parsed)
                            except json.JSONDecodeError:
                                parsed_results.append({"text": item["text"]})
                        else:
                            parsed_results.append(item)
                    results = parsed_results
            elif "error" in data:
                logger.error("MCP error: %s", data["error"])
                results = []
            else:
                results = []

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            return KnowledgeResult(
                query=query,
                results=results,
                tier=tier,
                timestamp=datetime.utcnow().isoformat(),
                latency_ms=latency,
                total_count=len(results),
            )

        except httpx.ConnectError as e:
            logger.error("Failed to connect to Data Lake: %s", e)
            return KnowledgeResult(
                query=query,
                results=[],
                tier=tier,
                timestamp=datetime.utcnow().isoformat(),
                latency_ms=0,
                total_count=0,
            )
        except Exception as e:
            logger.error("Knowledge retrieval failed: %s", e)
            raise

    async def query_sovereign(
        self,
        query: str,
        limit: int = 10,
    ) -> KnowledgeResult:
        """
        Query the M6 Sovereign tier for cross-domain knowledge.

        This provides the "God view" across the entire 1.37TB data lake.
        """
        return await self.knowledge_retrieve(
            query=query,
            tier=MemoryTier.M6_SOVEREIGN,
            limit=limit,
        )

    async def query_tier(
        self,
        query: str,
        tier: MemoryTier,
        limit: int = 10,
    ) -> KnowledgeResult:
        """
        Query a specific memory tier.

        If a Dual-Agentic tier (L1-L5) is provided, it will be mapped
        to the corresponding Data Lake tier (M1-M5).
        """
        # Map L-tiers to M-tiers
        mapped_tier = TIER_MAPPING.get(tier, tier)
        return await self.knowledge_retrieve(
            query=query,
            tier=mapped_tier,
            limit=limit,
        )

    def get_gold_path(self) -> Path:
        """Get the path to the Gold layer (curated data)."""
        if self._config and "paths" in self._config:
            return Path(self._config["paths"]["gold"])
        return Path(DATA_LAKE_PATH) / "04_GOLD"

    def get_poi_ledger_path(self) -> Path:
        """Get the path to the Proof-of-Impact ledger."""
        return self.get_gold_path() / "poi_ledger.jsonl"


# Singleton instance
_bridge: Optional[DataLakeBridge] = None


def get_data_lake_bridge() -> DataLakeBridge:
    """Get the singleton Data Lake bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = DataLakeBridge()
    return _bridge


async def query_lake(query: str, limit: int = 10) -> KnowledgeResult:
    """
    Convenience function to query the Data Lake.

    Uses M6 Sovereign tier by default for cross-domain search.

    Example:
        result = await query_lake("SAPE probes architecture")
        for item in result.results:
            print(item)
    """
    bridge = get_data_lake_bridge()
    return await bridge.query_sovereign(query, limit=limit)


# Synchronous wrapper for simple usage
def query_lake_sync(query: str, limit: int = 10) -> KnowledgeResult:
    """
    Synchronous wrapper for query_lake.

    Usage:
        python -c "from core.data_lake_bridge import query_lake_sync; print(query_lake_sync('SAPE'))"
    """
    import asyncio

    return asyncio.run(query_lake(query, limit))


if __name__ == "__main__":
    # Test the bridge
    import asyncio

    async def test():
        bridge = DataLakeBridge()

        # Health check
        status = await bridge.health_check()
        print(f"Data Lake Status: {'ONLINE' if status.online else 'OFFLINE'}")
        print(f"URL: {status.url}")

        if status.online:
            # Query test
            result = await bridge.query_sovereign("BIZRA architecture")
            print(f"\nQuery: {result.query}")
            print(f"Results: {result.total_count}")
            print(f"Latency: {result.latency_ms:.2f}ms")
            for item in result.results[:3]:
                print(f"  - {item}")

        await bridge.close()

    asyncio.run(test())
