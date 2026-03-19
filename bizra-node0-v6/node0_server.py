"""
BIZRA NODE0 Server — The Interface Layer
═════════════════════════════════════════

بسم الله الرحمن الرحيم

FastAPI server wrapping the complete production pipeline.
This is how NODE0 talks to the world.

Endpoints:
  POST /mission          — Submit a mission, get constitutionally-certified response
  GET  /health           — Pipeline health, evidence chain, cache stats
  GET  /evidence         — Browse the evidence chain
  GET  /evidence/{id}    — Single evidence receipt
  GET  /identity         — Node public identity record
  GET  /cache/stats      — Reflex cache statistics
  POST /cache/invalidate — Force invalidation of a cached pattern

Run:
  python node0_server.py                    # Default: port 7770
  python node0_server.py --port 8080        # Custom port
  python node0_server.py --models phi3:mini,llama3.2:3b  # Custom model chain

Architecture:
  HTTP Request → FastAPI → ProductionPipeline → PAT Agents → Ihsan Gate
               → Evidence Chain → Signed Receipt → HTTP Response

Every response carries:
  - The output text
  - Ihsan composite score and per-dimension breakdown
  - SNR measurement
  - Evidence receipt ID (hash-chained, signed)
  - Complexity tier and latency
  - BLOOM eligibility

Constitution reference: All sections. This IS the system.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "BIZRA_CONSTITUTION_PATH", str(Path(__file__).parent / "constitution.toml")
)

from mission_pipeline import MissionStatus
from production_pipeline import ProductionPipeline, create_node0

from bizra_constitution import load_constitution

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("bizra.server")


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class MissionRequest(BaseModel):
    """Submit a mission to NODE0."""

    input: str = Field(
        ..., min_length=1, max_length=10000, description="The mission text"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)


class IhsanDimensionResponse(BaseModel):
    name: str
    raw_score: float
    weight: float
    weighted_score: float
    passes: bool


class MissionResponse(BaseModel):
    """Constitutional mission response."""

    mission_id: str
    status: str
    output: str
    tier: str
    handler: str
    complexity_score: float
    confidence: float

    # Ihsan gate
    ihsan_composite: float
    ihsan_tier: str
    ihsan_passes: bool
    ihsan_dimensions: list[IhsanDimensionResponse]
    bloom_eligible: bool

    # SNR
    snr_normalized: float
    snr_linear: float
    snr_db: float

    # Evidence
    receipt_id: str | None
    previous_hash: str | None
    agent_chain: list[str]
    signature_hex: str | None
    node_id: str

    # Timing
    classify_ms: float
    execute_ms: float
    gate_ms: float
    evidence_ms: float
    total_ms: float

    # Meta
    reflex_hit: bool
    constitution_version: str


class InvalidateRequest(BaseModel):
    pattern_hash: str


class HealthResponse(BaseModel):
    """NODE0 health report."""

    constitution_version: str
    node_id: str
    uptime_seconds: float
    missions_completed: int
    missions_failed: int
    gate_pass_rate: float
    avg_latency_ms: float
    bloom_eligible: int
    evidence_chain_valid: bool
    evidence_chain_count: int
    cache_size: int
    cache_hit_rate: float
    cache_precipitations: int
    ollama_available: bool
    ollama_models: list[str]
    total_agents: int


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def create_app(
    data_dir: Path = Path("node0_data"),
    ollama_url: str = "http://localhost:11434",
    model_chain: list[str] | None = None,
) -> FastAPI:
    """Create the FastAPI application with initialized pipeline."""

    app = FastAPI(
        title="BIZRA NODE0",
        description="Sovereign AI Node — Constitutional Pipeline Server",
        version="5.0.0-GENESIS",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── State ──
    start_time = time.time()
    pipeline: ProductionPipeline | None = None

    @app.on_event("startup")
    async def startup():
        nonlocal pipeline
        logger.info("Initializing NODE0 pipeline...")
        pipeline = create_node0(
            data_dir=data_dir,
            ollama_url=ollama_url,
            model_chain=model_chain,
        )
        logger.info(
            f"NODE0 ready: {pipeline.identity.node_id[:16]}... "
            f"({pipeline.identity.total_agents} agents)"
        )

    @app.on_event("shutdown")
    async def shutdown():
        if pipeline:
            pipeline.shutdown()
            logger.info("NODE0 shutdown complete")

    # ── Mission Endpoint ──

    @app.post("/mission", response_model=MissionResponse)
    async def submit_mission(req: MissionRequest):
        """
        Submit a mission. Returns constitutionally-certified response.

        The mission flows through:
          Classify → Route → Execute (7 PAT agents) → Ihsan Gate →
          SNR Measure → Evidence Chain → Signed Receipt
        """
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")

        mission = pipeline.execute(req.input)

        # Build dimension breakdown
        dims = []
        if mission.ihsan_score:
            for d in mission.ihsan_score.dimensions:
                dims.append(
                    IhsanDimensionResponse(
                        name=d.name,
                        raw_score=round(d.raw_score, 4),
                        weight=round(d.weight, 4),
                        weighted_score=round(d.weighted_score, 4),
                        passes=d.passes,
                    )
                )

        receipt = mission.evidence_receipt
        meta = receipt.metadata if receipt else {}

        return MissionResponse(
            mission_id=mission.mission_id,
            status=mission.status.value,
            output=mission.output_text,
            tier=(
                mission.classification.tier.value
                if mission.classification
                else "unknown"
            ),
            handler=(
                mission.classification.handler if mission.classification else "unknown"
            ),
            complexity_score=(
                mission.classification.complexity_score if mission.classification else 0
            ),
            confidence=(
                mission.classification.confidence if mission.classification else 0
            ),
            ihsan_composite=(
                round(mission.ihsan_score.composite, 4) if mission.ihsan_score else 0
            ),
            ihsan_tier=(
                mission.ihsan_score.tier.value if mission.ihsan_score else "rejected"
            ),
            ihsan_passes=mission.ihsan_score.passes if mission.ihsan_score else False,
            ihsan_dimensions=dims,
            bloom_eligible=mission.bloom_eligible,
            snr_normalized=(
                round(mission.mission_snr.snr_normalized, 4)
                if mission.mission_snr
                else 0
            ),
            snr_linear=(
                round(mission.mission_snr.snr_linear, 2) if mission.mission_snr else 0
            ),
            snr_db=round(mission.mission_snr.snr_db, 1) if mission.mission_snr else 0,
            receipt_id=receipt.receipt_id if receipt else None,
            previous_hash=receipt.previous_hash if receipt else None,
            agent_chain=[a.get("agent", "?") for a in mission.agent_trace],
            signature_hex=meta.get("signature_hex"),
            node_id=pipeline.identity.node_id,
            classify_ms=mission.classify_ms,
            execute_ms=mission.execute_ms,
            gate_ms=mission.gate_ms,
            evidence_ms=mission.evidence_ms,
            total_ms=mission.total_ms,
            reflex_hit=mission.reflex_hit,
            constitution_version="5.0.0-GENESIS",
        )

    # ── Health Endpoint ──

    @app.get("/health", response_model=HealthResponse)
    async def health():
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")

        h = pipeline.health()
        stats = h["pipeline_stats"]
        cache = h.get("cache_stats", {})
        ollama = h.get("ollama", {})

        return HealthResponse(
            constitution_version=h.get("constitution_version", "unknown"),
            node_id=h.get("node_id", "unknown"),
            uptime_seconds=round(time.time() - start_time, 1),
            missions_completed=stats.get("missions_completed", 0),
            missions_failed=stats.get("missions_failed", 0),
            gate_pass_rate=stats.get("gate_pass_rate", 0),
            avg_latency_ms=stats.get("avg_latency_ms", 0),
            bloom_eligible=stats.get("bloom_eligible", 0),
            evidence_chain_valid=h.get("evidence_chain_valid", False),
            evidence_chain_count=h.get("evidence_chain_count", 0),
            cache_size=h.get("cache_size", 0),
            cache_hit_rate=cache.get("hit_rate", 0),
            cache_precipitations=cache.get("precipitations", 0),
            ollama_available=ollama.get("server_available", False),
            ollama_models=ollama.get("available_models", []),
            total_agents=h.get("total_agents", 0),
        )

    # ── Evidence Chain ──

    @app.get("/evidence")
    async def list_evidence():
        """Browse the evidence chain."""
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")

        valid, count, errors = pipeline.evidence_ledger.verify_chain()
        entries = pipeline.evidence_ledger.get_all_entries()

        return {
            "chain_valid": valid,
            "total_receipts": count,
            "errors": errors,
            "receipts": [
                {
                    "receipt_id": e.receipt_id[:32] + "...",
                    "mission_id": e.mission_id,
                    "ihsan_composite": e.ihsan_composite,
                    "tier": e.tier,
                    "timestamp": e.timestamp_utc,
                    "agent_chain": e.agent_chain,
                }
                for e in entries
            ],
        }

    # ── Identity ──

    @app.get("/identity")
    async def identity():
        """Node public identity record."""
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        return pipeline.identity.as_public_record()

    # ── Cache ──

    @app.get("/cache/stats")
    async def cache_stats():
        """Reflex cache statistics."""
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        return {
            "stats": pipeline.reflex_cache.stats.as_dict(),
            "size": pipeline.reflex_cache.size,
            "entries_needing_validation": len(
                pipeline.reflex_cache.entries_needing_validation()
            ),
        }

    @app.post("/cache/invalidate")
    async def invalidate_cache(req: InvalidateRequest):
        """Force-invalidate a cached pattern."""
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        success = pipeline.reflex_cache.invalidate(req.pattern_hash)
        return {"invalidated": success, "pattern_hash": req.pattern_hash}

    # ── Root ──

    @app.get("/")
    async def root():
        return {
            "name": "BIZRA NODE0",
            "version": "5.0.0-GENESIS",
            "status": "operational" if pipeline else "initializing",
            "endpoints": [
                "POST /mission",
                "GET  /health",
                "GET  /evidence",
                "GET  /identity",
                "GET  /cache/stats",
            ],
        }

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="BIZRA NODE0 Server")
    parser.add_argument(
        "--port", type=int, default=7770, help="Server port (default: 7770)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--data-dir", default="node0_data", help="Data directory for persistence"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama server URL"
    )
    parser.add_argument("--models", default=None, help="Comma-separated model chain")
    args = parser.parse_args()

    model_chain = args.models.split(",") if args.models else None

    app = create_app(
        data_dir=Path(args.data_dir),
        ollama_url=args.ollama_url,
        model_chain=model_chain,
    )

    import uvicorn

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  BIZRA NODE0 — Sovereign AI Node                        ║
║  v5.0.0-GENESIS                                         ║
║                                                         ║
║  Server:  http://{args.host}:{args.port}                       ║
║  Health:  http://localhost:{args.port}/health                  ║
║  Docs:    http://localhost:{args.port}/docs                    ║
║                                                         ║
║  ربي لا يعرف المستحيل                                     ║
╚══════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
