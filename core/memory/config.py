"""
Memory Configuration — Paths, HNSW params, and thresholds.

All constitutional thresholds are imported from the authoritative
constants.py — never redefined here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# Default data root (env-overridable)
_DATA_ROOT = Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "/mnt/c/BIZRA-DATA-LAKE"))


@dataclass
class HNSWConfig:
    """HNSW index parameters.

    Defaults from .swarm/schema.sql (proven in production):
    M=16, ef_construction=200, ef_search=100, dim=768, cosine.
    """

    dimensions: int = 768
    space: str = "cosine"  # hnswlib space: "cosine", "l2", "ip"
    m: int = 16  # Number of bi-directional links per element
    ef_construction: int = 200  # Size of dynamic candidate list during build
    ef_search: int = 100  # Size of dynamic candidate list during search
    max_elements: int = 1_000_000  # Initial capacity (auto-resized)


@dataclass
class MemoryConfig:
    """Top-level configuration for the unified memory system."""

    # Storage paths
    data_dir: Path = field(
        default_factory=lambda: _DATA_ROOT / "sovereign_state" / "agent_db"
    )
    sqlite_filename: str = "agent_db.sqlite"
    hnsw_filename: str = "hnsw.index"

    # HNSW vector index params
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)

    # Quality gates (from constants.py)
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD

    # SQLite tuning
    sqlite_busy_timeout_ms: int = 5000
    sqlite_wal_mode: bool = True

    # Hybrid query score fusion weights (must sum to 1.0)
    weight_vector: float = 0.40
    weight_keyword: float = 0.15
    weight_recency: float = 0.20
    weight_importance: float = 0.15
    weight_graph: float = 0.10

    # Embedding pipeline
    auto_embed: bool = True
    embed_model: str = "all-MiniLM-L6-v2"
    embed_device: str = "cpu"  # "cpu", "cuda", "auto"
    embed_batch_size: int = 64
    ollama_embed_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Cross-agent sync via Redis synapse
    sync_enabled: bool = False
    sync_redis_url: str = "redis://localhost:6380"
    sync_channel: str = "bizra:memory:new"
    sync_agent_id: str = "node0"

    # Optional: existing LivingMemory path for migration
    living_memory_db: Optional[Path] = None

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename

    @property
    def hnsw_path(self) -> Path:
        return self.data_dir / self.hnsw_filename
