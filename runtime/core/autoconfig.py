"""
BIZRA Kernel Auto-Configuration and Self-Healing System.

Probes all backend services, detects available models, and generates runtime config.
Handles graceful degradation when services are unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ServiceProbeResult:
    reachable: bool
    latency_ms: Optional[float] = None
    version: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ModelInfo:
    name: str
    provider: str
    size: Optional[int] = None
    capabilities: List[str] = field(default_factory=list)


class AutoConfigurator:
    """Auto-detects and configures BIZRA kernel on startup."""

    def __init__(self):
        self.node_id = "node0-genesis"
        self.config_dir = Path.home() / ".bizra"
        self.config_path = self.config_dir / "autoconfig.json"
        self._singleton_instance = None

    async def probe_ollama(self, url: str, timeout: float = 2.0) -> ServiceProbeResult:
        """Probe Ollama API for available models."""
        if not aiohttp:
            return ServiceProbeResult(False, error="aiohttp not installed")

        start = time.monotonic()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(f"{url}/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        latency = round((time.monotonic() - start) * 1000, 2)
                        return ServiceProbeResult(
                            reachable=True,
                            latency_ms=latency,
                            info={"models": models, "url": url},
                        )
                    return ServiceProbeResult(False, error=f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def probe_lmstudio(
        self, url: str, timeout: float = 2.0
    ) -> ServiceProbeResult:
        """Probe LM Studio API for available models."""
        if not aiohttp:
            return ServiceProbeResult(False, error="aiohttp not installed")

        start = time.monotonic()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(f"{url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["id"] for m in data.get("data", [])]
                        latency = round((time.monotonic() - start) * 1000, 2)
                        return ServiceProbeResult(
                            reachable=True,
                            latency_ms=latency,
                            info={"models": models, "url": url},
                        )
                    return ServiceProbeResult(False, error=f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def _tcp_check(self, host: str, port: int, timeout: float) -> bool:
        """Fast TCP port check before expensive driver probe."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def probe_redis(self, url: str, timeout: float = 1.5) -> ServiceProbeResult:
        """Probe Redis/Synapse connectivity."""
        start = time.monotonic()
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        password = parsed.password

        # Fast TCP check first
        if not await self._tcp_check(host, port, min(timeout, 0.5)):
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="port unreachable",
            )

        try:
            import redis.asyncio as aioredis
        except ImportError:
            return ServiceProbeResult(False, error="redis.asyncio not installed")

        try:
            conn = aioredis.Redis(
                host=host,
                port=port,
                password=password,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
            )
            await asyncio.wait_for(conn.ping(), timeout=timeout)
            info = await conn.info()
            latency = round((time.monotonic() - start) * 1000, 2)
            await conn.aclose()
            return ServiceProbeResult(
                reachable=True,
                latency_ms=latency,
                version=info.get("redis_version"),
                info={"version": info.get("redis_version"), "url": f"{host}:{port}"},
            )
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def probe_neo4j(
        self, url: str, auth: tuple[str, str], timeout: float = 1.5
    ) -> ServiceProbeResult:
        """Probe Neo4j connectivity."""
        start = time.monotonic()
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 7687

        if not await self._tcp_check(host, port, min(timeout, 0.5)):
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="port unreachable",
            )

        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            return ServiceProbeResult(False, error="neo4j driver not installed")

        try:
            driver = AsyncGraphDatabase.driver(
                url, auth=auth, connection_timeout=timeout
            )
            await driver.verify_connectivity()
            latency = round((time.monotonic() - start) * 1000, 2)
            await driver.close()
            return ServiceProbeResult(
                reachable=True, latency_ms=latency, info={"url": url}
            )
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def probe_chromadb(
        self, url: str, timeout: float = 2.0
    ) -> ServiceProbeResult:
        """Probe ChromaDB connectivity."""
        if not aiohttp:
            return ServiceProbeResult(False, error="aiohttp not installed")

        start = time.monotonic()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(f"{url}/api/v1/heartbeat") as resp:
                    if resp.status == 200:
                        latency = round((time.monotonic() - start) * 1000, 2)
                        return ServiceProbeResult(
                            reachable=True, latency_ms=latency, info={"url": url}
                        )
                    return ServiceProbeResult(False, error=f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def probe_postgres(
        self, url: str, timeout: float = 1.5
    ) -> ServiceProbeResult:
        """Probe PostgreSQL connectivity."""
        start = time.monotonic()
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432

        if not await self._tcp_check(host, port, min(timeout, 0.5)):
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="port unreachable",
            )

        try:
            import asyncpg
        except ImportError:
            return ServiceProbeResult(False, error="asyncpg not installed")

        try:
            conn = await asyncio.wait_for(asyncpg.connect(url), timeout=timeout)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            latency = round((time.monotonic() - start) * 1000, 2)
            return ServiceProbeResult(
                reachable=True,
                latency_ms=latency,
                version=version,
                info={"version": version},
            )
        except asyncio.TimeoutError:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error="timeout",
            )
        except Exception as e:
            return ServiceProbeResult(
                False,
                latency_ms=round((time.monotonic() - start) * 1000, 2),
                error=str(e),
            )

    async def probe_services(
        self, timeout: float = 1.5
    ) -> Dict[str, ServiceProbeResult]:
        """Probe all backend services concurrently with fast timeouts."""
        ollama_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_URL")
            or "http://localhost:11434"
        )
        lmstudio_url = os.getenv("LMSTUDIO_URL") or "http://localhost:1234"
        # Use host-mapped port (6380), not Docker-internal synapse:6379
        redis_url = (
            os.getenv("BIZRA_REDIS_URL")
            or "redis://:bizra_synapse_secure@localhost:6380"
        )
        neo4j_url = os.getenv("WISDOM_URL") or "bolt://localhost:7474"
        neo4j_auth_raw = os.getenv("NEO4J_AUTH") or "neo4j/bizra"
        chromadb_url = os.getenv("VECTORS_URL") or "http://localhost:8001"
        postgres_url = (
            os.getenv("DATABASE_URL") or "postgresql://bizra:bizra@localhost:5433/bizra"
        )

        neo4j_auth = None
        if neo4j_auth_raw and neo4j_auth_raw.lower() != "none":
            parts = neo4j_auth_raw.replace("/", ":").split(":", 1)
            if len(parts) == 2:
                neo4j_auth = (parts[0], parts[1])

        tasks = {
            "ollama": self.probe_ollama(ollama_url, timeout),
            "lmstudio": self.probe_lmstudio(lmstudio_url, timeout),
            "redis": self.probe_redis(redis_url, timeout),
            "chromadb": self.probe_chromadb(chromadb_url, timeout),
            "postgres": self.probe_postgres(postgres_url, timeout),
        }

        if neo4j_auth:
            tasks["neo4j"] = self.probe_neo4j(neo4j_url, neo4j_auth, timeout)

        # Global 5s timeout on all probes combined
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            results = [ServiceProbeResult(False, error="global timeout")] * len(tasks)

        return {
            k: (
                v
                if isinstance(v, ServiceProbeResult)
                else ServiceProbeResult(False, error=str(v))
            )
            for k, v in zip(tasks.keys(), results)
        }

    def _find_model(self, wanted: str, available: List[str]) -> Optional[str]:
        """Find a model in available list, with family-level fuzzy matching.
        e.g. 'deepseek-r1:8b' matches 'deepseek-r1:14b' (same family, different size).
        """
        if wanted in available:
            return wanted
        # Fuzzy: match model family (everything before the colon/size tag)
        family = wanted.split(":")[0] if ":" in wanted else wanted
        for m in available:
            if m.split(":")[0] == family:
                return m
        return None

    def configure_model_routing(
        self, services: Dict[str, ServiceProbeResult]
    ) -> Dict[str, Dict[str, str]]:
        """Generate model routing table based on available models."""
        routing = {}

        ollama_models = (
            services.get("ollama", ServiceProbeResult(False)).info.get("models", [])
            if services.get("ollama", ServiceProbeResult(False)).reachable
            else []
        )
        lmstudio_models = (
            services.get("lmstudio", ServiceProbeResult(False)).info.get("models", [])
            if services.get("lmstudio", ServiceProbeResult(False)).reachable
            else []
        )

        all_available = ollama_models + lmstudio_models
        canonical_routing = self._load_canonical_routing()

        for slot_name, slot_config in canonical_routing.items():
            candidates = [
                slot_config.get("primary"),
                slot_config.get("fallback"),
            ] + slot_config.get("allowed", [])

            selected = None
            provider = None

            for candidate in candidates:
                if not candidate:
                    continue
                # Try Ollama first (local-first)
                match = self._find_model(candidate, ollama_models)
                if match:
                    selected, provider = match, "ollama"
                    break
                # Then LM Studio
                match = self._find_model(candidate, lmstudio_models)
                if match:
                    selected, provider = match, "lmstudio"
                    break

            if selected and provider:
                routing[slot_name] = {"model": selected, "provider": provider}

        return routing

    def _load_canonical_routing(self) -> Dict[str, Dict[str, Any]]:
        """Load canonical model routing from sealed YAML."""
        if not yaml:
            return {}

        repo_root = Path(__file__).resolve().parents[1]
        yaml_path = repo_root / "model-family-genesis-v1-SEALED.yaml"

        if not yaml_path.exists():
            return {}

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            slots = data.get("capability_slots", {})
            return {
                slot_name: {
                    "primary": slot_config.get("routing", {}).get("primary"),
                    "fallback": slot_config.get("routing", {}).get("fallback"),
                    "allowed": slot_config.get("allowed_models", []),
                }
                for slot_name, slot_config in slots.items()
            }
        except Exception:
            return {}

    def auto_heal(
        self,
        services: Dict[str, ServiceProbeResult],
        routing: Dict[str, Dict[str, str]],
    ) -> str:
        """Determine operational mode based on service availability."""
        llm_available = any(
            services.get(k, ServiceProbeResult(False)).reachable
            for k in ["ollama", "lmstudio"]
        )
        redis_available = services.get("redis", ServiceProbeResult(False)).reachable
        neo4j_available = services.get("neo4j", ServiceProbeResult(False)).reachable

        if not llm_available:
            return "simulated"
        elif not redis_available or not neo4j_available:
            return "degraded"
        elif not routing:
            return "degraded"
        return "real"

    def detect_capabilities(
        self,
        services: Dict[str, ServiceProbeResult],
        routing: Dict[str, Dict[str, str]],
    ) -> List[str]:
        """Detect available capabilities based on models and services."""
        capabilities = []

        if "cold_core" in routing or "primary_reasoning" in routing:
            capabilities.append("reasoning")
        if "embeddings" in routing:
            capabilities.append("embeddings")
        if "vision" in routing:
            capabilities.append("vision")
        if services.get("neo4j", ServiceProbeResult(False)).reachable:
            capabilities.append("graph_memory")
        if services.get("chromadb", ServiceProbeResult(False)).reachable:
            capabilities.append("vector_search")

        return capabilities

    async def auto_configure(self) -> Dict[str, Any]:
        """Run full auto-configuration: probe, configure, persist."""
        services = await self.probe_services()
        routing = self.configure_model_routing(services)
        mode = self.auto_heal(services, routing)
        capabilities = self.detect_capabilities(services, routing)

        config = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "node_id": self.node_id,
            "services": {
                name: {
                    "reachable": result.reachable,
                    "latency_ms": result.latency_ms,
                    "version": result.version,
                    "info": result.info,
                    "error": result.error,
                }
                for name, result in services.items()
            },
            "model_routing": routing,
            "mode": mode,
            "capabilities": capabilities,
        }

        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return config


_AUTOCONFIGURATOR: Optional[AutoConfigurator] = None


def get_autoconfigurator() -> AutoConfigurator:
    """Get singleton AutoConfigurator instance."""
    global _AUTOCONFIGURATOR
    if _AUTOCONFIGURATOR is None:
        _AUTOCONFIGURATOR = AutoConfigurator()
    return _AUTOCONFIGURATOR


async def auto_configure() -> Dict[str, Any]:
    """Entry point: run full auto-configuration."""
    configurator = get_autoconfigurator()
    return await configurator.auto_configure()
