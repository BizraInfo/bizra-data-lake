#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Memory Server v1.0 - Port 7999
# ═══════════════════════════════════════════════════════════════════════════════
"""
Sovereign Memory API Server for BIZRA PAT+SAT ecosystem.

Provides:
- Multi-tier memory access (working, short-term, long-term, episodic, semantic)
- Knowledge graph operations via HyperGraphRAG connector
- Agent memory isolation with session management
- Expertise file access and evolution tracking

Endpoints:
- GET  /                 → Server info
- GET  /health           → Health check
- GET  /memory           → List memories (with filters)
- POST /memory           → Store a memory
- GET  /memory/{id}      → Get specific memory
- GET  /knowledge        → Query knowledge graph
- POST /knowledge        → Add knowledge node
- GET  /expertise        → Get expertise YAML
- POST /expertise        → Update expertise
- GET  /stats            → Memory statistics
"""

from __future__ import annotations

import asyncio
import os
import sys
import json
import hashlib
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MEMORY_SERVER_PORT = int(os.getenv("BIZRA_MEMORY_PORT", "7999"))
MEMORY_SERVER_HOST = os.getenv("BIZRA_MEMORY_HOST", "127.0.0.1")
BIZRA_DATA_VAULT = Path(os.getenv("BIZRA_DATA_VAULT", "bizra_data_vault"))
BIZRA_MEMORY_DIR = Path(os.getenv("BIZRA_MEMORY_DIR", "bizra_memory"))

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STORES (with file persistence)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore:
    """Multi-tier in-memory store with file persistence."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._persist_every_writes = max(
            int(os.getenv("BIZRA_MEMORY_PERSIST_EVERY_WRITES", "5")),
            1,
        )
        self._pending_persistence_writes = 0
        self.tiers = {
            "working": {},      # Current task context
            "short_term": {},   # Recent interactions
            "long_term": {},    # Persistent knowledge
            "episodic": {},     # Session-based memories
            "semantic": {},     # Conceptual relationships
        }
        self.knowledge_nodes = {}
        self.edges = []
        self.stats = {
            "total_memories": 0,
            "total_knowledge_nodes": 0,
            "queries_served": 0,
            "writes": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._load_persistence()
    
    def _persistence_path(self) -> Path:
        BIZRA_DATA_VAULT.mkdir(parents=True, exist_ok=True)
        return BIZRA_DATA_VAULT / "memory_store.json"
    
    def _load_persistence(self):
        path = self._persistence_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    self.tiers = data.get("tiers", self.tiers)
                    self.knowledge_nodes = data.get("knowledge_nodes", {})
                    self.edges = data.get("edges", [])
                    # Recount
                    self.stats["total_memories"] = sum(len(t) for t in self.tiers.values())
                    self.stats["total_knowledge_nodes"] = len(self.knowledge_nodes)
                print(f"[MEM] Loaded {self.stats['total_memories']} memories, {self.stats['total_knowledge_nodes']} knowledge nodes")
            except Exception as e:
                print(f"[MEM] Could not load persistence: {e}")
    
    def _save_persistence(self, force: bool = False):
        path = self._persistence_path()
        try:
            with self._lock:
                if not force:
                    self._pending_persistence_writes += 1
                    if self._pending_persistence_writes < self._persist_every_writes:
                        return
                self._pending_persistence_writes = 0

                payload = {
                    "tiers": self.tiers,
                    "knowledge_nodes": self.knowledge_nodes,
                    "edges": self.edges,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, path)
        except Exception as e:
            print(f"[MEM] Could not save persistence: {e}")
    
    def add_memory(self, tier: str, content: str, agent_slug: str = "system", 
                   session_id: str = "default", metadata: dict = None) -> dict:
        if tier not in self.tiers:
            tier = "working"
        
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        
        memory = {
            "id": mem_id,
            "tier": tier,
            "content": content,
            "agent_slug": agent_slug,
            "session_id": session_id,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "metadata": metadata or {},
        }
        
        with self._lock:
            self.tiers[tier][mem_id] = memory
            self.stats["total_memories"] += 1
            self.stats["writes"] += 1
        self._save_persistence()
        
        return memory
    
    def get_memory(self, mem_id: str) -> Optional[dict]:
        with self._lock:
            self.stats["queries_served"] += 1
            for _, tier_data in self.tiers.items():
                if mem_id in tier_data:
                    memory = tier_data[mem_id]
                    memory["last_accessed"] = datetime.now(timezone.utc).isoformat()
                    memory["access_count"] += 1
                    return memory
        return None
    
    def list_memories(self, tier: str = None, agent_slug: str = None, 
                      session_id: str = None, limit: int = 100) -> list:
        with self._lock:
            self.stats["queries_served"] += 1
            results = []
            
            tiers_to_search = [tier] if tier and tier in self.tiers else list(self.tiers.keys())
            
            for t in tiers_to_search:
                for mem in self.tiers[t].values():
                    if agent_slug and mem.get("agent_slug") != agent_slug:
                        continue
                    if session_id and mem.get("session_id") != session_id:
                        continue
                    results.append(mem)
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
        
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
    
    def add_knowledge_node(self, node_type: str, content: str, 
                           created_by: str = "system", metadata: dict = None) -> dict:
        node_id = f"kn_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        
        node = {
            "id": node_id,
            "type": node_type,
            "content": content,
            "created_by": created_by,
            "created_at": now,
            "metadata": metadata or {},
            "snr_score": 0.0,
        }
        
        with self._lock:
            self.knowledge_nodes[node_id] = node
            self.stats["total_knowledge_nodes"] += 1
            self.stats["writes"] += 1
        self._save_persistence()
        
        return node
    
    def query_knowledge(self, query: str, limit: int = 20) -> list:
        with self._lock:
            self.stats["queries_served"] += 1
            query_lower = query.lower()
            results = []
            
            for node in self.knowledge_nodes.values():
                content_lower = node.get("content", "").lower()
                if query_lower in content_lower:
                    results.append(node)
                    if len(results) >= limit:
                        break
        
        return results
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self.stats,
                "tier_counts": {t: len(v) for t, v in self.tiers.items()},
                "uptime_since": self.stats["started_at"],
            }


# Global store instance
memory_store = MemoryStore()


def _read_expertise_sync() -> dict:
    import yaml

    expertise_path = BIZRA_MEMORY_DIR / "expertise.yaml"
    if not expertise_path.exists():
        return {"expertise": {}, "source": str(expertise_path), "exists": False}

    with open(expertise_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {"expertise": data, "source": str(expertise_path), "exists": True}


def _update_expertise_sync(payload: "ExpertiseUpdate") -> dict:
    import yaml

    expertise_path = BIZRA_MEMORY_DIR / "expertise.yaml"
    BIZRA_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if expertise_path.exists():
        with open(expertise_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if payload.domain not in data:
        data[payload.domain] = {}
    data[payload.domain].update(payload.content)
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(expertise_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    return {
        "status": "updated",
        "domain": payload.domain,
        "source": str(expertise_path),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MemoryCreate(BaseModel):
    tier: str = Field("working", description="Memory tier: working, short_term, long_term, episodic, semantic")
    content: str = Field(..., min_length=1, description="Memory content")
    agent_slug: str = Field("system", description="Agent that created this memory")
    session_id: str = Field("default", description="Session identifier")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class KnowledgeCreate(BaseModel):
    node_type: str = Field("concept", description="Node type: concept, agent, domain, claim, evidence, memory, skill, tool, session")
    content: str = Field(..., min_length=1, description="Knowledge content")
    created_by: str = Field("system", description="Creator identifier")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ExpertiseUpdate(BaseModel):
    domain: str = Field(..., description="Expertise domain key")
    content: dict = Field(..., description="Expertise content to merge")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN EVENTS
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[MEM] BIZRA Memory Server starting on {MEMORY_SERVER_HOST}:{MEMORY_SERVER_PORT}")
    print(f"[MEM] Data vault: {BIZRA_DATA_VAULT.absolute()}")
    print(f"[MEM] Memory dir: {BIZRA_MEMORY_DIR.absolute()}")
    yield
    print("[MEM] Saving state before shutdown...")
    memory_store._save_persistence(force=True)
    print("[MEM] Memory Server stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIZRA Memory Server",
    description="Sovereign Memory API for PAT+SAT ecosystem",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Memory Server info."""
    return {
        "name": "BIZRA Memory Server",
        "version": "1.0.0",
        "status": "ONLINE",
        "port": MEMORY_SERVER_PORT,
        "capabilities": [
            "multi-tier-memory",
            "knowledge-graph",
            "expertise-evolution",
            "session-isolation",
        ],
        "tiers": ["working", "short_term", "long_term", "episodic", "semantic"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = await asyncio.to_thread(memory_store.get_stats)
    return {
        "status": "healthy",
        "service": "memory-server",
        "port": MEMORY_SERVER_PORT,
        "stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/memory")
async def list_memories(
    tier: Optional[str] = Query(None, description="Filter by tier"),
    agent_slug: Optional[str] = Query(None, description="Filter by agent"),
    session_id: Optional[str] = Query(None, description="Filter by session"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
):
    """List memories with optional filters."""
    memories = await asyncio.to_thread(
        memory_store.list_memories, tier, agent_slug, session_id, limit
    )
    return {
        "count": len(memories),
        "memories": memories,
    }


@app.post("/memory")
async def create_memory(payload: MemoryCreate):
    """Store a new memory."""
    memory = await asyncio.to_thread(
        memory_store.add_memory,
        payload.tier,
        payload.content,
        payload.agent_slug,
        payload.session_id,
        payload.metadata,
    )
    return {
        "status": "created",
        "memory": memory,
    }


@app.get("/memory/{mem_id}")
async def get_memory(mem_id: str):
    """Get a specific memory by ID."""
    memory = await asyncio.to_thread(memory_store.get_memory, mem_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory {mem_id} not found")
    return memory


@app.get("/knowledge")
async def query_knowledge(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """Query the knowledge graph."""
    results = await asyncio.to_thread(memory_store.query_knowledge, q, limit)
    return {
        "query": q,
        "count": len(results),
        "nodes": results,
    }


@app.post("/knowledge")
async def create_knowledge(payload: KnowledgeCreate):
    """Add a knowledge node."""
    node = await asyncio.to_thread(
        memory_store.add_knowledge_node,
        payload.node_type,
        payload.content,
        payload.created_by,
        payload.metadata,
    )
    return {
        "status": "created",
        "node": node,
    }


@app.get("/expertise")
async def get_expertise():
    """Get expertise YAML contents."""
    try:
        return await asyncio.to_thread(_read_expertise_sync)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read expertise: {e}")


@app.post("/expertise")
async def update_expertise(payload: ExpertiseUpdate):
    """Update expertise for a domain."""
    try:
        return await asyncio.to_thread(_update_expertise_sync, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update expertise: {e}")


@app.get("/stats")
async def get_stats():
    """Get memory server statistics."""
    return await asyncio.to_thread(memory_store.get_stats)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BIZRA MEMORY SERVER v1.0")
    print("  Sovereign Memory for PAT+SAT Ecosystem")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host=MEMORY_SERVER_HOST,
        port=MEMORY_SERVER_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
