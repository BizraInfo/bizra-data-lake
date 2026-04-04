#!/usr/bin/env python3
"""
BIZRA PAT Memory - Persistent Memory for Personal Agentic Team
==============================================================
Dual-layer persistence: Redis (hot) + JSON (cold)
Survives full system restarts with automatic restoration.

Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                    PAT MEMORY STORE                        │
    ├───────────────────────────────────────────────────────────┤
    │                                                            │
    │   HOT LAYER (Redis)          COLD LAYER (JSON)            │
    │   ┌─────────────────┐        ┌─────────────────┐          │
    │   │ user_preferences│        │ .bizra/          │          │
    │   │ session_history │   ←→   │ pat_memory.json │          │
    │   │ learned_patterns│        │                 │          │
    │   │ model_routing   │        │ (persistent)    │          │
    │   │ system_config   │        │                 │          │
    │   └─────────────────┘        └─────────────────┘          │
    │          ↕                            ↕                    │
    │   bizra:pat:memory:*        ~/.bizra/ or C:/Users/...    │
    │   (fast, session)            (survives restarts)          │
    │                                                            │
    └───────────────────────────────────────────────────────────┘

Memory Categories:
    - user_preferences: UI settings, favorite models, display prefs
    - session_history: Compressed history of recent sessions (last 50)
    - learned_patterns: Patterns PAT has learned about user workflow
    - model_routing: Which models work best for which tasks (learned)
    - system_config: Auto-detected system capabilities (GPU, RAM, etc)
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("pat_memory")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Use host-mapped port (6380), not Docker-internal synapse:6379 with TLS
REDIS_URL = os.getenv("BIZRA_REDIS_URL", "redis://:bizra_synapse_secure@localhost:6380")
REDIS_KEY_PREFIX = "bizra:pat:memory"
SESSION_HISTORY_LIMIT = 50
EVIDENCE_PATH = Path("docs/evidence/receipts/pat_memory")


def _get_cold_storage_path() -> Path:
    """Determine cold storage path based on platform."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("USERPROFILE", "C:/Users/Default"))
    else:
        base = Path.home()

    bizra_dir = base / ".bizra"
    bizra_dir.mkdir(exist_ok=True)
    return bizra_dir / "pat_memory.json"


COLD_STORAGE_PATH = _get_cold_storage_path()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_dict(data: Dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════


async def _emit_receipt(
    operation: str,
    category: str,
    key: str,
    value: Any,
    success: bool,
    error: Optional[str] = None,
) -> str:
    """Emit a receipt for memory operations."""
    EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)

    receipt = {
        "schema": "bizra.pat_memory.v1",
        "receipt_type": "MemoryOperation",
        "receipt_id": f"MEM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}",
        "timestamp": utc_now_iso(),
        "operation": operation,
        "category": category,
        "key": key,
        "value_hash": sha256_dict({"value": value}) if value else None,
        "success": success,
        "error": error,
        "integrity_hash": "",
    }

    receipt["integrity_hash"] = sha256_dict(receipt)

    receipt_file = EVIDENCE_PATH / "operations.jsonl"
    try:
        with open(receipt_file, "a") as f:
            f.write(json.dumps(receipt) + "\n")
    except Exception as e:
        logger.warning(f"Failed to emit receipt: {e}")

    return receipt["receipt_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAT MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════════════


class PATMemoryStore:
    """
    Persistent memory store for PAT with dual-layer persistence.

    Hot layer: Redis (fast, session-based)
    Cold layer: JSON file (survives restarts)

    Usage:
        memory = await get_pat_memory()
        await memory.store("user_preferences", "theme", "dark")
        theme = await memory.retrieve("user_preferences", "theme", default="light")
        context = await memory.get_user_context()
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._cold_path = COLD_STORAGE_PATH
        self._categories = [
            "user_preferences",
            "session_history",
            "learned_patterns",
            "model_routing",
            "system_config",
        ]

    async def initialize(self) -> bool:
        """Initialize hot and cold storage layers."""
        if self._connected:
            return True

        # Connect to Redis (hot layer)
        if REDIS_AVAILABLE:
            try:
                self._redis = aioredis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                )
                await self._redis.ping()
                self._connected = True
                logger.info(f"PAT Memory connected to Redis: {REDIS_URL}")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}. Using cold storage only.")
                self._connected = False
        else:
            logger.warning("redis-py not installed. Using cold storage only.")

        # Load cold storage into hot layer
        await self._load_from_disk()

        return True

    async def close(self) -> None:
        """Close connections and sync to disk."""
        await self.sync_to_disk()
        if self._redis:
            try:
                await self._redis.aclose()
            except AttributeError:
                await self._redis.close()

    def _key(self, category: str, key: str) -> str:
        """Build Redis key."""
        return f"{REDIS_KEY_PREFIX}:{category}:{key}"

    async def store(
        self, category: str, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Store a value in both hot and cold layers."""
        if category not in self._categories:
            logger.warning(f"Unknown category: {category}")

        try:
            # Hot layer (Redis)
            if self._connected and self._redis:
                redis_key = self._key(category, key)
                value_json = json.dumps(value, default=str)

                if ttl:
                    await self._redis.setex(redis_key, ttl, value_json)
                else:
                    await self._redis.set(redis_key, value_json)

            # Cold layer (JSON) - no TTL for persistence
            if not ttl:
                await self._update_cold_storage(category, key, value)

            await _emit_receipt("store", category, key, value, success=True)
            return True

        except Exception as e:
            logger.error(f"Failed to store {category}:{key}: {e}")
            await _emit_receipt(
                "store", category, key, value, success=False, error=str(e)
            )
            return False

    async def retrieve(self, category: str, key: str, default: Any = None) -> Any:
        """Retrieve a value from hot or cold layer."""
        try:
            # Try hot layer first (Redis)
            if self._connected and self._redis:
                redis_key = self._key(category, key)
                value_json = await self._redis.get(redis_key)
                if value_json:
                    return json.loads(value_json)

            # Fallback to cold layer
            cold_data = await self._read_cold_storage()
            value = cold_data.get(category, {}).get(key, default)

            await _emit_receipt("retrieve", category, key, value, success=True)
            return value

        except Exception as e:
            logger.error(f"Failed to retrieve {category}:{key}: {e}")
            await _emit_receipt(
                "retrieve", category, key, None, success=False, error=str(e)
            )
            return default

    async def retrieve_all(self, category: str) -> Dict[str, Any]:
        """Retrieve all entries in a category."""
        try:
            result = {}

            # Hot layer (Redis)
            if self._connected and self._redis:
                pattern = self._key(category, "*")
                async for redis_key in self._redis.scan_iter(match=pattern):
                    value_json = await self._redis.get(redis_key)
                    if value_json:
                        # Extract key from redis_key
                        key = redis_key.split(":")[-1]
                        result[key] = json.loads(value_json)

            # Merge with cold layer
            cold_data = await self._read_cold_storage()
            cold_category = cold_data.get(category, {})
            for key, value in cold_category.items():
                if key not in result:
                    result[key] = value

            return result

        except Exception as e:
            logger.error(f"Failed to retrieve_all for {category}: {e}")
            return {}

    async def learn_pattern(
        self, pattern_name: str, pattern_data: Dict[str, Any]
    ) -> bool:
        """Store a learned pattern about user workflow."""
        pattern_entry = {
            "name": pattern_name,
            "data": pattern_data,
            "learned_at": utc_now_iso(),
            "confidence": pattern_data.get("confidence", 1.0),
        }
        return await self.store("learned_patterns", pattern_name, pattern_entry)

    async def get_user_context(self) -> Dict[str, Any]:
        """Get full user context for LLM system prompts."""
        context = {
            "timestamp": utc_now_iso(),
            "user_preferences": await self.retrieve_all("user_preferences"),
            "learned_patterns": await self.retrieve_all("learned_patterns"),
            "model_routing": await self.retrieve_all("model_routing"),
            "system_config": await self.retrieve_all("system_config"),
            "recent_sessions": await self._get_recent_sessions(limit=10),
        }
        return context

    async def _get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history."""
        all_sessions = await self.retrieve_all("session_history")
        sessions = list(all_sessions.values())
        sessions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return sessions[:limit]

    async def sync_to_disk(self) -> bool:
        """Flush hot layer to cold storage."""
        try:
            cold_data = {}
            for category in self._categories:
                cold_data[category] = await self.retrieve_all(category)

            # Write atomically
            temp_path = self._cold_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(cold_data, f, indent=2, default=str)
            temp_path.replace(self._cold_path)

            logger.info(f"Synced PAT memory to disk: {self._cold_path}")
            await _emit_receipt("sync_to_disk", "all", "all", cold_data, success=True)
            return True

        except Exception as e:
            logger.error(f"Failed to sync to disk: {e}")
            await _emit_receipt(
                "sync_to_disk", "all", "all", None, success=False, error=str(e)
            )
            return False

    async def load_from_disk(self) -> bool:
        """Restore cold storage to hot layer."""
        return await self._load_from_disk()

    async def _load_from_disk(self) -> bool:
        """Internal: Load cold storage into Redis."""
        try:
            if not self._cold_path.exists():
                logger.info("No cold storage found. Starting fresh.")
                return True

            cold_data = await self._read_cold_storage()

            if not self._connected or not self._redis:
                logger.info("Redis unavailable. Cold data loaded for fallback.")
                return True

            # Load into Redis (hot layer)
            for category, entries in cold_data.items():
                if category not in self._categories:
                    continue
                for key, value in entries.items():
                    redis_key = self._key(category, key)
                    value_json = json.dumps(value, default=str)
                    await self._redis.set(redis_key, value_json)

            logger.info(f"Loaded PAT memory from disk: {self._cold_path}")
            await _emit_receipt("load_from_disk", "all", "all", cold_data, success=True)
            return True

        except Exception as e:
            logger.error(f"Failed to load from disk: {e}")
            await _emit_receipt(
                "load_from_disk", "all", "all", None, success=False, error=str(e)
            )
            return False

    async def _read_cold_storage(self) -> Dict[str, Any]:
        """Read cold storage JSON."""
        if not self._cold_path.exists():
            return {}

        try:
            with open(self._cold_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cold storage: {e}")
            return {}

    async def _update_cold_storage(self, category: str, key: str, value: Any) -> None:
        """Update a single entry in cold storage."""
        cold_data = await self._read_cold_storage()

        if category not in cold_data:
            cold_data[category] = {}

        cold_data[category][key] = value

        # Write atomically
        temp_path = self._cold_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(cold_data, f, indent=2, default=str)
        temp_path.replace(self._cold_path)

    async def detect_system(self) -> Dict[str, Any]:
        """Detect system capabilities and store in system_config."""
        system_info = {
            "detected_at": utc_now_iso(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }

        # Detect GPU
        gpu_info = await self._detect_gpu()
        if gpu_info:
            system_info["gpu"] = gpu_info

        # Detect RAM
        ram_info = await self._detect_ram()
        if ram_info:
            system_info["ram"] = ram_info

        # Detect Ollama models
        ollama_models = await self._detect_ollama_models()
        if ollama_models:
            system_info["ollama_models"] = ollama_models

        # Detect LM Studio models
        lmstudio_models = await self._detect_lmstudio_models()
        if lmstudio_models:
            system_info["lmstudio_models"] = lmstudio_models

        # Store in system_config
        await self.store("system_config", "detected", system_info)

        logger.info(
            f"System detection complete: {len(ollama_models)} Ollama models, GPU={gpu_info is not None}"
        )
        return system_info

    async def _detect_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect GPU via nvidia-smi."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0:
                output = stdout.decode().strip()
                if output:
                    parts = [p.strip() for p in output.split(",")]
                    if len(parts) >= 2:
                        return {
                            "name": parts[0],
                            "memory_total": parts[1],
                            "driver_version": parts[2] if len(parts) > 2 else "unknown",
                        }
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")

        return None

    async def _detect_ram(self) -> Optional[Dict[str, Any]]:
        """Detect RAM via psutil if available."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
            }
        except ImportError:
            logger.debug("psutil not installed, skipping RAM detection")
        except Exception as e:
            logger.debug(f"RAM detection failed: {e}")

        return None

    async def _detect_ollama_models(self) -> List[str]:
        """Detect Ollama models via API."""
        try:
            # Check if we have httpx or aiohttp
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        return [m["name"] for m in data.get("models", [])]
            except ImportError:
                pass

            try:
                import aiohttp

                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(
                        "http://localhost:11434/api/tags"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return [m["name"] for m in data.get("models", [])]
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"Ollama detection failed: {e}")

        return []

    async def _detect_lmstudio_models(self) -> List[str]:
        """Detect LM Studio models via API."""
        try:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:1234/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        return [m["id"] for m in data.get("data", [])]
            except ImportError:
                pass

            try:
                import aiohttp

                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(
                        "http://localhost:1234/v1/models"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return [m["id"] for m in data.get("data", [])]
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"LM Studio detection failed: {e}")

        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_pat_memory: Optional[PATMemoryStore] = None


async def get_pat_memory() -> PATMemoryStore:
    """Get the singleton PAT memory instance."""
    global _pat_memory
    if _pat_memory is None:
        _pat_memory = PATMemoryStore()
        await _pat_memory.initialize()
    return _pat_memory


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Test PAT memory system."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA PAT Memory")
    parser.add_argument(
        "--detect", action="store_true", help="Detect system capabilities"
    )
    parser.add_argument("--sync", action="store_true", help="Sync to disk")
    parser.add_argument("--load", action="store_true", help="Load from disk")
    parser.add_argument("--context", action="store_true", help="Show user context")
    parser.add_argument("--test", action="store_true", help="Run test scenario")

    args = parser.parse_args()

    memory = await get_pat_memory()

    if args.detect:
        print("\n🔍 Detecting system capabilities...")
        info = await memory.detect_system()
        print("\n" + "═" * 60)
        print("  SYSTEM DETECTION")
        print("═" * 60)
        print(f"  Platform: {info.get('platform')}")
        print(f"  Python: {info.get('python_version')}")

        if "gpu" in info:
            gpu = info["gpu"]
            print(f"  GPU: {gpu['name']} ({gpu['memory_total']})")
        else:
            print("  GPU: Not detected")

        if "ram" in info:
            ram = info["ram"]
            print(
                f"  RAM: {ram['total_gb']} GB total, {ram['available_gb']} GB available"
            )

        ollama = info.get("ollama_models", [])
        print(f"  Ollama Models: {len(ollama)}")
        for model in ollama[:5]:
            print(f"    - {model}")

        lmstudio = info.get("lmstudio_models", [])
        print(f"  LM Studio Models: {len(lmstudio)}")
        for model in lmstudio[:5]:
            print(f"    - {model}")

        print("═" * 60 + "\n")
        await memory.close()
        return

    if args.sync:
        print("\n💾 Syncing to disk...")
        success = await memory.sync_to_disk()
        if success:
            print(f"✅ Synced to: {COLD_STORAGE_PATH}")
        else:
            print("❌ Sync failed")
        await memory.close()
        return

    if args.load:
        print("\n📂 Loading from disk...")
        success = await memory.load_from_disk()
        if success:
            print(f"✅ Loaded from: {COLD_STORAGE_PATH}")
        else:
            print("❌ Load failed")
        await memory.close()
        return

    if args.context:
        print("\n🧠 Retrieving user context...")
        context = await memory.get_user_context()
        print("\n" + "═" * 60)
        print("  USER CONTEXT")
        print("═" * 60)
        print(json.dumps(context, indent=2, default=str))
        print("═" * 60 + "\n")
        await memory.close()
        return

    if args.test:
        print("\n🧪 PAT MEMORY TEST SCENARIO\n")

        # Test 1: Store user preferences
        print("1. Storing user preferences...")
        await memory.store("user_preferences", "theme", "dark")
        await memory.store("user_preferences", "language", "en")
        await memory.store("user_preferences", "favorite_model", "deepseek-r1:8b")
        print("   ✅ Stored 3 preferences")

        # Test 2: Store session history
        print("\n2. Storing session history...")
        session = {
            "session_id": "sess-001",
            "timestamp": utc_now_iso(),
            "task": "Test memory system",
            "outcome": "success",
        }
        await memory.store("session_history", "sess-001", session)
        print("   ✅ Stored 1 session")

        # Test 3: Learn a pattern
        print("\n3. Learning a pattern...")
        pattern = {
            "pattern": "user prefers reasoning models for technical tasks",
            "confidence": 0.85,
            "observations": 12,
        }
        await memory.learn_pattern("technical_model_preference", pattern)
        print("   ✅ Learned 1 pattern")

        # Test 4: Retrieve
        print("\n4. Retrieving values...")
        theme = await memory.retrieve("user_preferences", "theme")
        print(f"   Theme: {theme}")

        # Test 5: Retrieve all
        print("\n5. Retrieving all preferences...")
        prefs = await memory.retrieve_all("user_preferences")
        print(f"   ✅ Retrieved {len(prefs)} preferences")
        for key, value in prefs.items():
            print(f"      {key}: {value}")

        # Test 6: Sync to disk
        print("\n6. Syncing to disk...")
        await memory.sync_to_disk()
        print(f"   ✅ Synced to {COLD_STORAGE_PATH}")

        # Test 7: Get user context
        print("\n7. Getting user context...")
        context = await memory.get_user_context()
        print(f"   ✅ Context includes {len(context)} categories")

        # Test 8: Detect system
        print("\n8. Detecting system capabilities...")
        info = await memory.detect_system()
        ollama_count = len(info.get("ollama_models", []))
        print(f"   ✅ Detected {ollama_count} Ollama models")

        await memory.close()
        print("\n✅ PAT memory test complete!\n")
        return

    # Default: Show status
    print("\n📊 PAT MEMORY STATUS\n")
    print(f"Cold storage: {COLD_STORAGE_PATH}")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Connected: {memory._connected}")
    print(f"Categories: {', '.join(memory._categories)}")
    print("\nRun with --help for options")

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
