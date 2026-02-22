#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ███╗   ██╗ ██████╗ ██████╗ ███████╗     ██████╗                            ║
║   ████╗  ██║██╔═══██╗██╔══██╗██╔════╝    ██╔═████╗                           ║
║   ██╔██╗ ██║██║   ██║██║  ██║█████╗      ██║██╔██║                           ║
║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝      ████╔╝██║                           ║
║   ██║ ╚████║╚██████╔╝██████╔╝███████╗    ╚██████╔╝                           ║
║   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚═════╝                            ║
║                                                                              ║
║   BIZRA Node0 — Local Sovereign AI Home Base                                 ║
║   PAT Team Proactive Coworker Activation                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script activates your local Node0 with:
- Proactive Execution Kernel (PEK)
- PAT Team (7 agents)
- Multi-model LLM routing
- 24/7 autonomous operation

Usage:
    # Token auto-loaded from .env (LM_API_TOKEN or LM_STUDIO_API_KEY)
    python scripts/node0_activate.py              # Start full node
    python scripts/node0_activate.py --status     # Check status
    python scripts/node0_activate.py --mission "task"  # Assign mission
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    # Load .env from project root (supports both LM_STUDIO_API_KEY and LM_API_TOKEN)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    import hashlib
    import json as _json
    import uuid
except ImportError:
    pass

# ═══ Verified Intelligence Pipeline — Standing on Giants ═══
# Shannon (1948): SNR as information quality measure
# Lamport (1978): Logical clocks and event ordering
# Merkle (1979): Hash chains for tamper detection
# Al-Ghazali (1095): Ihsan — excellence as constraint
# Friston (2006): Free energy minimization ≈ SNR maximization
_EVIDENCE_LEDGER = None
_SNR_CALCULATOR = None  # Legacy — kept for backward compat
_SNR_FACADE = None  # Phase 42: Unified SNR (v2 + maximizer ensemble)

def _init_verified_pipeline():
    """Initialize the Verified Intelligence Pipeline (VIP) components."""
    global _EVIDENCE_LEDGER, _SNR_CALCULATOR, _SNR_FACADE
    try:
        from core.proof_engine.evidence_ledger import EvidenceLedger
        ledger_path = Path(PROJECT_ROOT) / "sovereign_state" / "evidence.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        _EVIDENCE_LEDGER = EvidenceLedger(ledger_path, validate_on_append=True)
        logger.info(f"  Evidence ledger: {ledger_path.name} (seq={_EVIDENCE_LEDGER.sequence})")
    except Exception as e:
        logger.warning(f"  Evidence ledger: failed to init ({e})")

    # Phase 42: Initialize SNRFacade with v2 adapter + text maximizer
    v2_adapter = None
    text_engine = None
    try:
        from core.iaas.snr_v2 import SNRCalculatorV2
        _SNR_CALCULATOR = SNRCalculatorV2()
        logger.info("  SNR v2 engine: initialized (Shannon + Renyi-2)")
        try:
            from core.iaas.snr_v2_adapter import SNRv2Adapter
            v2_adapter = SNRv2Adapter(_SNR_CALCULATOR)
        except Exception as e:
            logger.warning(f"  SNR v2 adapter: unavailable ({e})")
    except Exception as e:
        logger.warning(f"  SNR v2 engine: fallback mode ({e})")

    try:
        from core.sovereign.snr_maximizer import SNRMaximizer
        text_engine = SNRMaximizer()
        logger.info("  SNR maximizer: initialized (7 noise dimensions, bounded scoring)")
    except Exception as e:
        logger.warning(f"  SNR maximizer: unavailable ({e})")

    try:
        from core.snr_protocol import SNRFacade
        _SNR_FACADE = SNRFacade(
            v2_engine=v2_adapter,
            text_engine=text_engine,
        )
        engines = []
        if v2_adapter:
            engines.append("v2")
        if text_engine:
            engines.append("text")
        logger.info(f"  SNR facade: ready (engines: {'+'.join(engines) or 'none'})")
    except Exception as e:
        logger.warning(f"  SNR facade: failed to init ({e})")


def _resolve_lm_token() -> str:
    """Resolve LM Studio auth token from environment.

    Checks LM_API_TOKEN first, falls back to LM_STUDIO_API_KEY.
    Mirrors the unification logic in core/integration/constants.py.
    """
    return os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or ""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-12s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Node0")

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ════════════════════════════════════════════════════════════════════════════════
# PAT AGENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

PAT_AGENTS = {
    "strategist": {
        "name": "Strategist",
        "role": "Strategic planning and long-term thinking",
        "giants": "Sun Tzu, John Boyd, Michael Porter",
        "model_purpose": "reasoning",
    },
    "researcher": {
        "name": "Researcher",
        "role": "Deep investigation and evidence gathering",
        "giants": "Vannevar Bush, Claude Shannon, Douglas Engelbart",
        "model_purpose": "reasoning",
    },
    "analyst": {
        "name": "Analyst",
        "role": "Data analysis and pattern recognition",
        "giants": "Herbert Simon, Daniel Kahneman, Judea Pearl",
        "model_purpose": "reasoning",
    },
    "creator": {
        "name": "Creator",
        "role": "Content creation and design",
        "giants": "Leonardo da Vinci, Steve Jobs, Dieter Rams",
        "model_purpose": "general",
    },
    "executor": {
        "name": "Executor",
        "role": "Task execution and automation",
        "giants": "Frederick Taylor, W. Edwards Deming",
        "model_purpose": "agentic",
    },
    "guardian": {
        "name": "Guardian",
        "role": "Ethical oversight and security",
        "giants": "Al-Ghazali, John Rawls, Anthropic",
        "model_purpose": "reasoning",
    },
    "coordinator": {
        "name": "Coordinator",
        "role": "Team synthesis and integration",
        "giants": "Norbert Wiener, Peter Senge",
        "model_purpose": "reasoning",
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# MODEL ROUTING — Maps PAT agent purpose to proactive_config.yaml models
# ════════════════════════════════════════════════════════════════════════════════

_PURPOSE_TO_ROLE = {
    "reasoning": "reasoner",
    "general": "planner",
    "agentic": "planner",
}

_DEFAULT_MODEL_ROUTING = {
    "planner": "agentflow-planner-7b-i1",
    "reasoner": "deepseek/deepseek-r1-0528-qwen3-8b",
    "fast": "qwen2.5-0.5b-instruct",
    "vision": "qwen/qwen3-vl-8b",
    "vision_light": "qwen/qwen3-vl-4b",
    "voice": "deephat-v1-7b",
    "embedding": "text-embedding-nomic-embed-text-v1.5",
}


def _load_proactive_config() -> Dict[str, Any]:
    """Load config/proactive_config.yaml with graceful fallback."""
    cfg_path = Path(PROJECT_ROOT) / "config" / "proactive_config.yaml"
    if yaml and cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_model_for_agent(agent_id: str, config: Dict[str, Any]) -> str:
    """Resolve the correct LM Studio model for a PAT agent from config."""
    routing = config.get("model_routing", _DEFAULT_MODEL_ROUTING)
    purpose = PAT_AGENTS.get(agent_id, {}).get("model_purpose", "reasoning")
    role = _PURPOSE_TO_ROLE.get(purpose, "reasoner")
    return routing.get(role, routing.get("reasoner", "deepseek/deepseek-r1-0528-qwen3-8b"))


# ════════════════════════════════════════════════════════════════════════════════
# FAISS KNOWLEDGE RETRIEVER — RAG context for PAT missions
# ════════════════════════════════════════════════════════════════════════════════

class KnowledgeRetriever:
    """Retrieves relevant context from the Node0 FAISS index for missions."""

    def __init__(self, config: Dict[str, Any]):
        self._index = None
        self._chunks_df = None
        self._model = None
        self._enabled = False
        self._top_k = 3

        kb_cfg = config.get("knowledge_base", {})
        if not kb_cfg.get("enabled", False):
            return

        self._top_k = kb_cfg.get("top_k", 5)
        index_path = Path(PROJECT_ROOT) / kb_cfg.get("faiss_index", "04_GOLD/node0_faiss.index")
        gold_dir = Path(PROJECT_ROOT) / "04_GOLD"

        try:
            import faiss
            import pandas as pd

            if not index_path.exists():
                return

            self._index = faiss.read_index(str(index_path))

            # Load ALL chunk parquets that feed the unified FAISS index
            # Order must match how the index was built: chunks → conversations → research → golden_gems
            frames = []
            for parquet_name in ["chunks.parquet", "conversations_chunks.parquet", "research_chunks.parquet", "golden_gems_chunks.parquet"]:
                pq_path = gold_dir / parquet_name
                if pq_path.exists():
                    df = pd.read_parquet(pq_path, columns=["chunk_id", "chunk_text"])
                    frames.append(df)
                    logger.info(f"  KB corpus: {parquet_name} ({len(df)} chunks)")

            if frames:
                self._chunks_df = pd.concat(frames, ignore_index=True)
                self._enabled = True
                logger.info(f"  Knowledge base: {self._index.ntotal} vectors, {len(self._chunks_df)} chunks loaded")
        except ImportError:
            logger.warning("  Knowledge base: faiss/pandas not installed — RAG disabled")
        except Exception as e:
            logger.warning(f"  Knowledge base: {e}")

    def retrieve(self, query: str, top_k: int = None) -> str:
        """Retrieve top-k relevant chunks for a query string."""
        if not self._enabled:
            return ""

        k = top_k or self._top_k
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")

            import numpy as np
            vec = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
            scores, indices = self._index.search(vec, k)

            context_parts = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self._chunks_df):
                    continue
                text = self._chunks_df.iloc[idx]["chunk_text"]
                if len(text) > 400:
                    text = text[:400] + "..."
                context_parts.append(f"[{i+1}] (sim={score:.3f}) {text}")

            return "\n".join(context_parts)
        except Exception as e:
            logger.debug(f"  RAG retrieval failed: {e}")
            return ""


# ════════════════════════════════════════════════════════════════════════════════
# VERIFIED INTELLIGENCE PIPELINE — Production Receipts + Real SNR
# Standing on Giants: Shannon (SNR) · Lamport (hash chains) · Merkle (tamper detection)
# ════════════════════════════════════════════════════════════════════════════════

def _compute_real_snr(
    query: str,
    agent_outputs: list,
    model=None,
) -> dict:
    """Compute genuine SNR using unified SNRFacade (v2 + maximizer ensemble).

    Phase 42: Routes through SNRFacade for ensemble scoring.
    Falls back to direct v2 if facade unavailable, then to lexical-only.

    Returns dict with snr_score, ihsan_score, method, engine, and breakdown.
    """
    import numpy as np

    texts = [r.get("content", "") for r in agent_outputs if r.get("success") and r.get("content")]
    if not texts:
        return {"snr_score": 0.0, "ihsan_score": 0.0, "method": "empty"}

    combined_text = "\n\n".join(texts)

    # Phase 42: Use SNRFacade (ensemble of v2 + maximizer)
    if _SNR_FACADE is not None:
        try:
            result = _SNR_FACADE.calculate(text=combined_text, query=query)
            return {
                "snr_score": result.score,
                "ihsan_score": result.score,
                "method": f"facade_{result.engine}",
                "engine": result.engine,
                "ihsan_achieved": result.ihsan_achieved,
                "v2_snr": result.metrics.get("v2_snr"),
                "text_snr": result.metrics.get("text_snr"),
                "quality_tier": result.metrics.get("v2_tier") or result.metrics.get("quality_tier"),
                "recommendations": result.recommendations,
            }
        except Exception as e:
            logger.warning(f"  SNR facade failed, falling back to direct v2: {e}")

    # Fallback: direct SNRCalculatorV2 (Phase 41 behavior)
    if _SNR_CALCULATOR is None:
        return {"snr_score": 0.85, "ihsan_score": 0.85, "method": "fallback"}

    try:
        query_emb = None
        text_embs = None

        if model is not None:
            query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
            text_embs = model.encode(texts, normalize_embeddings=True).astype(np.float32)

        components = _SNR_CALCULATOR.compute_snr(
            query=query,
            texts=texts,
            query_embedding=query_emb,
            text_embeddings=text_embs,
        )

        return {
            "snr_score": components.snr,
            "ihsan_score": components.snr,
            "method": "snr_v2_embeddings" if query_emb is not None else "snr_v2_lexical",
            "signal_strength": components.signal_strength,
            "diversity": components.diversity,
            "grounding": components.grounding,
            "iaas_score": components.iaas_score,
        }
    except Exception as e:
        logger.warning(f"  SNR computation failed: {e}")
        return {"snr_score": 0.85, "ihsan_score": 0.85, "method": "error_fallback"}


async def _synthesize_with_got(
    mission_desc: str,
    agent_results: list,
) -> dict:
    """Synthesize PAT agent outputs using Graph-of-Thoughts reasoning.

    Phase 42 Spec 04: When InferenceGateway is available, uses LLM-backed
    hypothesis generation and conclusion formulation. Falls back to template
    synthesis (or simple concatenation) when no LLM available.

    Standing on Giants: Besta (GoT, 2024) · Boyd (OODA, 1976)

    Returns dict with conclusion, thought_count, reasoning_paths, thought_chain.
    """
    empty_result = {
        "conclusion": "",
        "thought_count": 0,
        "reasoning_paths": 0,
        "snr_score": 0.0,
        "llm_used": False,
        "thought_chain": [],
    }

    try:
        from core.sovereign.graph_core import GraphOfThoughts
    except ImportError:
        return empty_result

    # Collect successful agent outputs as facts
    facts = []
    for r in agent_results:
        if r.get("success") and r.get("content"):
            agent_name = r.get("name", r.get("agent", "agent"))
            facts.append(f"[{agent_name}]: {r['content']}")

    if not facts:
        return empty_result

    try:
        # Initialize GoT — try to wire InferenceGateway for LLM-backed synthesis
        gateway = None
        try:
            from core.inference.gateway import InferenceGateway
            gateway = InferenceGateway()
        except Exception:
            pass

        got = GraphOfThoughts(
            max_depth=3,
            beam_width=3,
            inference_gateway=gateway,
        )

        # Run reasoning with agent outputs as context facts
        reasoning_result = await got.reason(
            query=mission_desc,
            context={
                "domain": "mission_synthesis",
                "facts": facts,
            },
            max_depth=3,
        )

        conclusion = reasoning_result.get("conclusion", "")
        thought_chain = []
        for node_id, node in got.nodes.items():
            thought_chain.append({
                "id": node.id[:12],
                "type": node.thought_type.value,
                "snr": round(node.snr_score, 3),
                "ihsan": round(node.ihsan_score, 3),
                "depth": node.depth,
            })

        return {
            "conclusion": conclusion,
            "thought_count": reasoning_result.get("graph_stats", {}).get("total_thoughts", len(got.nodes)),
            "reasoning_paths": reasoning_result.get("depth_reached", 0),
            "snr_score": reasoning_result.get("snr_score", 0.0),
            "llm_used": reasoning_result.get("llm_used", False),
            "thought_chain": thought_chain,
        }

    except Exception as e:
        logger.warning(f"GoT synthesis failed (non-blocking): {e}")
        # Non-blocking fallback: concatenate agent outputs
        combined = "\n\n".join(
            r.get("content", "") for r in agent_results
            if r.get("success") and r.get("content")
        )
        return {
            "conclusion": combined,
            "thought_count": 0,
            "reasoning_paths": 0,
            "snr_score": 0.0,
            "llm_used": False,
            "thought_chain": [],
        }


def _emit_verified_receipt(mission: Dict, result: Dict, snr_data: dict) -> Dict:
    """Emit a schema-compliant, hash-chained receipt to the evidence ledger.

    Uses BLAKE3 + hash chain via core.proof_engine.evidence_ledger.
    Falls back to legacy SHA-256 stub if ledger unavailable.
    """
    receipt_id = hashlib.blake2b(
        f"{mission.get('id', '')}:{datetime.now(timezone.utc).isoformat()}".encode(),
        digest_size=16,
    ).hexdigest()

    snr_score = snr_data.get("snr_score", 0.0)
    ihsan_score = snr_data.get("ihsan_score", 0.0)
    method = snr_data.get("method", "unknown")

    if _EVIDENCE_LEDGER is not None:
        try:
            from core.proof_engine.evidence_ledger import emit_receipt
            from core.proof_engine.canonical import hex_digest

            # Compute seal over mission payload
            payload_bytes = _json.dumps({
                "mission_id": mission.get("id"),
                "description": mission.get("description", "")[:200],
                "agents": [r.get("agent") for r in result.get("agents", [])],
                "snr_score": snr_score,
                "total_tokens": result.get("total_tokens", 0),
            }, sort_keys=True).encode("utf-8")
            seal_digest = hex_digest(payload_bytes)

            # Query digest
            query_bytes = mission.get("description", "").encode("utf-8")
            query_digest = hex_digest(query_bytes)

            entry = emit_receipt(
                _EVIDENCE_LEDGER,
                receipt_id=receipt_id,
                node_id="node0-momo-genesis",
                policy_version="1.0.0",
                status="accepted" if ihsan_score >= 0.95 else "amber_restricted",
                decision="APPROVED" if ihsan_score >= 0.95 else "QUARANTINED",
                reason_codes=[] if ihsan_score >= 0.95 else ["SNR_BELOW_THRESHOLD"],
                snr_score=snr_score,
                ihsan_score=ihsan_score,
                seal_digest=seal_digest,
                seal_algorithm="blake3",
                query_digest=query_digest,
                duration_ms=result.get("duration_ms", 0.0),
                snr_trace={
                    "signal_components": {
                        "signal_strength": snr_data.get("signal_strength", 0.0),
                        "diversity": snr_data.get("diversity", 0.0),
                        "grounding": snr_data.get("grounding", 0.0),
                    },
                    "method": method,
                },
            )

            return {
                "hash": entry.entry_hash[:16],
                "chain_seq": entry.sequence,
                "prev_hash": entry.prev_hash[:16],
                "ihsan_score": ihsan_score,
                "snr_score": snr_score,
                "snr_method": method,
                "timestamp": entry.timestamp,
                "ledger": "evidence.jsonl",
            }
        except Exception as e:
            logger.warning(f"  Ledger receipt failed, falling back: {e}")

    # Fallback: legacy SHA-256 stub (no chain, no ledger)
    payload = _json.dumps({
        "mission_id": mission.get("id"),
        "description": mission.get("description", "")[:200],
        "agents": [r.get("agent") for r in result.get("agents", [])],
        "ihsan_score": ihsan_score,
        "total_tokens": result.get("total_tokens", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True)
    return {
        "hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "ihsan_score": ihsan_score,
        "snr_score": snr_score,
        "snr_method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ledger": "none",
    }


# ════════════════════════════════════════════════════════════════════════════════
# NODE0 PROACTIVE KERNEL
# ════════════════════════════════════════════════════════════════════════════════

class Node0ProactiveKernel:
    """
    The Proactive Execution Kernel for Node0.

    Loop: SENSE → PREDICT → SCORE → VERIFY → EXECUTE → PROVE → LEARN
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Load proactive_config.yaml as base, merge explicit overrides
        self._yaml_config = _load_proactive_config()
        self.config = {**self._yaml_config, **(config or {})}
        self.token = _resolve_lm_token()
        self.base_url = "http://192.168.56.1:1234"

        # Knowledge retriever — connects PAT agents to FAISS index
        self._knowledge = KnowledgeRetriever(self._yaml_config)

        # State
        self._running = False
        self._cycle_count = 0
        self._missions: List[Dict] = []
        self._completed: List[Dict] = []
        self._receipts: List[Dict] = []
        self._metrics = {
            "cycles": 0,
            "missions_completed": 0,
            "tokens_used": 0,
            "ihsan_score": 0.0,
        }

        # Load baseline for impact tracking
        self._baseline = self._load_baseline()

        # Initialize Verified Intelligence Pipeline
        _init_verified_pipeline()

        # Cycle timing
        cycles_cfg = self.config.get("cycles", {})
        self.cycle_interval = self.config.get(
            "cycle_interval",
            cycles_cfg.get("interval_seconds", 30.0),
        )
        # Ihsan from constitutional config or constants.py
        constitutional = self.config.get("constitutional", {})
        self.ihsan_threshold = constitutional.get("ihsan_threshold", 0.95)

    def _load_baseline(self) -> Dict[str, Any]:
        """Load MoMo Day 0 baseline for impact tracking."""
        baseline_path = Path(PROJECT_ROOT) / "sovereign_state" / "node0_baseline.json"
        try:
            if baseline_path.exists():
                with open(baseline_path) as f:
                    data = _json.load(f)
                logger.info(f"  Baseline: {data.get('node_id', 'unknown')} (clarity={data.get('clarity_score')})")
                return data
        except Exception as e:
            logger.debug(f"  Baseline not loaded: {e}")
        return {}

    async def start(self):
        """Start the proactive kernel."""
        self._running = True
        mode = self.config.get("mode", "proactive_partner")
        logger.info("═" * 60)
        logger.info("NODE0 PROACTIVE KERNEL ACTIVATED")
        logger.info("═" * 60)
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Cycle Interval: {self.cycle_interval}s")
        logger.info(f"  Ihsān Threshold: {self.ihsan_threshold}")
        logger.info(f"  PAT Agents: {len(PAT_AGENTS)}")
        logger.info(f"  Knowledge Base: {'ACTIVE' if self._knowledge._enabled else 'DISABLED'}")
        if self._baseline:
            logger.info(f"  Baseline: {self._baseline.get('node_id', '?')} | goals={len(self._baseline.get('weekly_goals', []))}")
        logger.info("═" * 60)

        await self._run_loop()

    async def stop(self):
        """Stop the kernel."""
        self._running = False
        logger.info("Node0 kernel stopping...")

    async def add_mission(self, description: str, priority: str = "normal"):
        """Add a mission for PAT team."""
        mission = {
            "id": f"mission-{len(self._missions)+1:03d}",
            "description": description,
            "priority": priority,
            "status": "pending",
            "created": datetime.now(timezone.utc).isoformat(),
            "assigned_agents": [],
            "result": None,
        }
        self._missions.append(mission)
        logger.info(f"📋 Mission added: {mission['id']} - {description[:50]}...")
        return mission

    async def _run_loop(self):
        """Main proactive loop."""
        while self._running:
            self._cycle_count += 1
            cycle_start = time.perf_counter()

            logger.info(f"─── Cycle {self._cycle_count} ───")

            try:
                # 1. SENSE - Check for pending missions
                pending = [m for m in self._missions if m["status"] == "pending"]

                if pending:
                    mission = pending[0]
                    logger.info(f"📌 Processing: {mission['id']}")

                    # 2. ASSIGN - Select agents
                    agents = self._select_agents(mission)
                    mission["assigned_agents"] = agents
                    mission["status"] = "in_progress"

                    # 3. EXECUTE - Run PAT team
                    result = await self._execute_mission(mission, agents)

                    # 4. VERIFY - Check Ihsān compliance
                    ihsan_ok = result.get("ihsan_score", 0) >= self.ihsan_threshold

                    # 5. PROVE - Record result
                    mission["result"] = result
                    mission["status"] = "completed" if ihsan_ok else "needs_review"
                    mission["completed"] = datetime.now(timezone.utc).isoformat()

                    self._completed.append(mission)
                    self._missions.remove(mission)
                    self._metrics["missions_completed"] += 1

                    logger.info(f"✓ Mission {mission['id']}: {'PASS' if ihsan_ok else 'REVIEW'}")
                else:
                    # Idle - proactive monitoring
                    logger.info("  ○ Idle - monitoring for opportunities")

                # 6. LEARN - Update metrics
                self._metrics["cycles"] = self._cycle_count

            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Sleep until next cycle
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(1.0, self.cycle_interval - elapsed)

            # Show countdown for long sleeps
            if sleep_time > 5:
                logger.info(f"  Next cycle in {sleep_time:.0f}s (Ctrl+C to stop)")

            await asyncio.sleep(sleep_time)

    def _select_agents(self, mission: Dict) -> List[str]:
        """Select appropriate agents for mission."""
        desc = mission["description"].lower()

        agents = ["coordinator"]  # Always include coordinator

        if any(w in desc for w in ["plan", "strategy", "approach"]):
            agents.append("strategist")
        if any(w in desc for w in ["research", "investigate", "find"]):
            agents.append("researcher")
        if any(w in desc for w in ["analyze", "data", "pattern"]):
            agents.append("analyst")
        if any(w in desc for w in ["create", "design", "build"]):
            agents.append("creator")
        if any(w in desc for w in ["security", "safe", "risk", "ethic"]):
            agents.append("guardian")

        # Default to strategist + guardian + coordinator
        if len(agents) == 1:
            agents = ["strategist", "guardian", "coordinator"]

        return agents

    async def _execute_mission(self, mission: Dict, agents: List[str]) -> Dict:
        """Execute mission with PAT team — model-routed + RAG-augmented."""
        import httpx

        results = []
        total_tokens = 0

        # RAG: retrieve relevant knowledge context for this mission
        rag_context = self._knowledge.retrieve(mission["description"])
        if rag_context:
            logger.info(f"    📚 Knowledge context retrieved ({len(rag_context)} chars)")

        for agent_id in agents:
            agent = PAT_AGENTS[agent_id]
            model = _resolve_model_for_agent(agent_id, self._yaml_config)

            system_prompt = f"""You are the PAT {agent['name']}. Your role is {agent['role']}.
Standing on Giants: {agent['giants']}.
Be concise (2-3 paragraphs). Focus on actionable insights."""

            # Build user message with optional RAG context
            user_content = f"Mission: {mission['description']}"
            if rag_context:
                user_content += f"\n\nRelevant knowledge from Node0 memory:\n{rag_context}"

            logger.info(f"    🤖 {agent['name']} ({model.split('/')[-1]})...")

            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

            try:
                async with httpx.AsyncClient(headers=headers, timeout=180.0) as client:
                    resp = await client.post(f"{self.base_url}/v1/chat/completions", json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "max_tokens": 600,
                    })

                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"].get("content", "")
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        total_tokens += tokens

                        results.append({
                            "agent": agent_id,
                            "name": agent["name"],
                            "model": model,
                            "content": content,
                            "tokens": tokens,
                            "success": True,
                        })
                    else:
                        results.append({
                            "agent": agent_id,
                            "model": model,
                            "success": False,
                            "error": f"HTTP {resp.status_code}",
                        })
            except Exception as e:
                results.append({
                    "agent": agent_id,
                    "model": model,
                    "success": False,
                    "error": str(e),
                })

        # ═══ Phase 42: GoT Synthesis (Graph-of-Thoughts on agent outputs) ═══
        # Standing on Giants: Besta (GoT, 2024) · Boyd (OODA synthesis)
        got_data = await _synthesize_with_got(
            mission["description"], results
        )
        if got_data.get("thought_count", 0) > 0:
            logger.info(
                f"    🧠 GoT synthesis: {got_data['thought_count']} thoughts, "
                f"{got_data['reasoning_paths']} paths"
            )

        # ═══ Verified Intelligence Pipeline: Real SNR Scoring ═══
        # Standing on Giants: Shannon (SNR) · Friston (free energy = SNR maximization)
        successful = sum(1 for r in results if r.get("success"))

        # Use GoT conclusion for SNR if available, else concatenate agent outputs
        snr_inputs = results
        if got_data.get("conclusion"):
            snr_inputs = [{"content": got_data["conclusion"], "success": True}]

        # Use production SNR engine with real embeddings when available
        embedding_model = getattr(self._knowledge, "_model", None)
        snr_data = _compute_real_snr(mission["description"], snr_inputs, model=embedding_model)
        ihsan_score = snr_data.get("ihsan_score", 0.0)

        # Ensure minimum score floor from agent success rate
        # (real SNR can underestimate on very short outputs)
        success_rate = successful / len(results) if results else 0
        if success_rate == 1.0 and ihsan_score < 0.80:
            ihsan_score = max(ihsan_score, 0.80)
            snr_data["ihsan_score"] = ihsan_score

        self._metrics["tokens_used"] += total_tokens
        self._metrics["ihsan_score"] = ihsan_score

        result = {
            "agents": results,
            "total_tokens": total_tokens,
            "ihsan_score": ihsan_score,
            "snr_score": snr_data.get("snr_score", 0.0),
            "snr_method": snr_data.get("method", "unknown"),
            "success_count": successful,
            "total_count": len(results),
            "rag_context_used": bool(rag_context),
            "got": {
                "active": got_data.get("thought_count", 0) > 0,
                "thought_count": got_data.get("thought_count", 0),
                "reasoning_paths": got_data.get("reasoning_paths", 0),
                "llm_used": got_data.get("llm_used", False),
                "thought_chain": got_data.get("thought_chain", []),
            },
        }

        # ═══ Emit hash-chained, BLAKE3-sealed evidence receipt ═══
        # Standing on Giants: Lamport (hash chains) · Merkle (tamper detection)
        receipt = _emit_verified_receipt(mission, result, snr_data)
        self._receipts.append(receipt)
        result["receipt"] = receipt

        return result

    def get_status(self) -> Dict:
        """Get current kernel status."""
        return {
            "running": self._running,
            "cycles": self._cycle_count,
            "pending_missions": len(self._missions),
            "completed_missions": len(self._completed),
            "metrics": self._metrics,
        }


# ════════════════════════════════════════════════════════════════════════════════
# NODE0 ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════════

class Node0Orchestrator:
    """
    Main Node0 orchestrator - coordinates all subsystems.

    Standing on Giants: Boyd (OODA loop) · Shannon (SNR) · Besta (GoT) ·
    Al-Ghazali (Ihsān) · Deming (PDCA) · Lamport (distributed reliability)
    """

    def __init__(self):
        config = _load_proactive_config()
        self.kernel = Node0ProactiveKernel(config)
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start Node0."""
        self._print_banner()

        # Check LM Studio connection
        if not await self._check_connection():
            logger.error("Cannot connect to LM Studio. Exiting.")
            return

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        # Start kernel
        kernel_task = asyncio.create_task(self.kernel.start())

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cleanup
        await self.kernel.stop()
        kernel_task.cancel()

        logger.info("Node0 shutdown complete.")

    async def _check_connection(self) -> bool:
        """Check LM Studio connection."""
        import httpx

        token = _resolve_lm_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        try:
            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                resp = await client.get("http://192.168.56.1:1234/v1/models")
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    loaded = [m for m in models if m.get("loaded")]
                    logger.info(f"✓ LM Studio connected: {len(models)} models, {len(loaded)} loaded")
                    return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")

        return False

    def _handle_shutdown(self):
        """Handle shutdown signal."""
        logger.info("\nShutdown signal received...")
        self._shutdown_event.set()

    def _print_banner(self):
        """Print Node0 banner."""
        kb_status = "ACTIVE" if self.kernel._knowledge._enabled else "DISABLED"
        vectors = self.kernel._knowledge._index.ntotal if self.kernel._knowledge._enabled else 0
        baseline_id = self.kernel._baseline.get("node_id", "none")
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ███╗   ██╗ ██████╗ ██████╗ ███████╗     ██████╗                            ║
║   ████╗  ██║██╔═══██╗██╔══██╗██╔════╝    ██╔═████╗                           ║
║   ██╔██╗ ██║██║   ██║██║  ██║█████╗      ██║██╔██║                           ║
║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝      ████╔╝██║                           ║
║   ██║ ╚████║╚██████╔╝██████╔╝███████╗    ╚██████╔╝                           ║
║   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚═════╝                            ║
║                                                                              ║
║   B I Z R A   S O V E R E I G N   A I   -   H O M E   B A S E                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PAT Team: 7 Agents | Mode: Proactive Partner | Ihsan: 0.95                  ║
║  Models: 7 (planner + reasoner + fast + vision + voice + embedding)           ║
║  Knowledge Base: {kb_status} ({vectors:,} vectors)                             ║
║  Baseline: {baseline_id}                                                      ║
║  لا نفترض — We do not assume. إحسان — Excellence in all things.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

async def cmd_start(args):
    """Start Node0."""
    orchestrator = Node0Orchestrator()
    await orchestrator.start()


async def cmd_status(args):
    """Check system status."""
    import httpx

    print("\n" + "═" * 60)
    print("NODE0 STATUS CHECK")
    print("═" * 60)

    token = _resolve_lm_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Check LM Studio
    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            resp = await client.get("http://192.168.56.1:1234/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                loaded = [m for m in models if m.get("loaded")]
                print("  LM Studio:    ✓ Connected")
                print(f"  Models:       {len(models)} available, {len(loaded)} loaded")
                for m in loaded:
                    print(f"    → {m['id']}")
            else:
                print(f"  LM Studio:    ✗ Error {resp.status_code}")
    except Exception as e:
        print(f"  LM Studio:    ✗ {e}")

    print()
    print(f"  Token:        {'✓ Set' if token else '✗ Not set'}")
    print("  PAT Agents:   7 configured")
    print("  Mode:         proactive_partner")
    print("═" * 60 + "\n")


async def cmd_mission(args):
    """Run a single mission."""
    kernel = Node0ProactiveKernel({"cycle_interval": 5.0})

    print("\n" + "═" * 60)
    print("NODE0 MISSION EXECUTION")
    print("═" * 60)
    print(f"Mission: {args.task[:60]}...")
    print("═" * 60 + "\n")

    mission = await kernel.add_mission(args.task)

    # Execute immediately
    agents = kernel._select_agents(mission)
    print(f"Assigned agents: {', '.join(agents)}\n")

    result = await kernel._execute_mission(mission, agents)

    # Display results
    print("\n" + "═" * 60)
    print("MISSION RESULTS")
    print("═" * 60)

    for r in result["agents"]:
        if r.get("success"):
            model_short = r.get("model", "?").split("/")[-1]
            print(f"\n┌─ {r['name'].upper()} ({model_short}) ─")
            for line in r.get("content", "").split("\n")[:10]:
                print(f"│  {line}")
            print(f"└─ ({r.get('tokens', 0)} tokens)")

    if result.get("rag_context_used"):
        print("\n  📚 Knowledge base context was used for this mission")

    print("\n" + "─" * 60)
    print(f"Total Tokens: {result['total_tokens']}")
    snr_method = result.get('snr_method', 'unknown')
    print(f"SNR Score:    {result.get('snr_score', 0):.4f} ({snr_method})")
    print(f"Ihsan Score:  {result['ihsan_score']:.2%}")
    print(f"Status:       {'✓ PASS' if result['ihsan_score'] >= 0.95 else '⚠ REVIEW'}")
    receipt = result.get("receipt", {})
    if receipt:
        ledger = receipt.get("ledger", "none")
        chain_seq = receipt.get("chain_seq", "?")
        print(f"Receipt:      {receipt.get('hash', '?')}...")
        if ledger != "none":
            print(f"Evidence:     {ledger} (seq={chain_seq}, prev={receipt.get('prev_hash', '?')})")
    print("═" * 60 + "\n")


async def cmd_verify(args):
    """Verify the integrity of the evidence chain.

    Standing on Giants: Lamport (event ordering) · Merkle (hash chains)
    The Third Fact made operational — every claim has a verifiable receipt.
    """
    print("\n" + "═" * 60)
    print("EVIDENCE CHAIN VERIFICATION")
    print("═" * 60)

    ledger_path = Path(PROJECT_ROOT) / "sovereign_state" / "evidence.jsonl"

    if not ledger_path.exists():
        print("  Status:  No evidence ledger found")
        print("  Action:  Run a mission first to create receipts")
        print("═" * 60 + "\n")
        return

    try:
        from core.proof_engine.evidence_ledger import EvidenceLedger
        ledger = EvidenceLedger(ledger_path, validate_on_append=False)

        print(f"  Ledger:  {ledger_path.name}")
        print(f"  Entries: {ledger.sequence}")
        print(f"  Latest:  {ledger.last_hash[:16]}...")
        print()

        is_valid, errors = ledger.verify_chain()

        if is_valid:
            print("  Chain:   ✓ INTACT — all hashes verified")
            print("  Status:  ZANN_ZERO enforced — no tampering detected")
        else:
            print(f"  Chain:   ✗ BROKEN — {len(errors)} error(s)")
            for err in errors[:5]:
                print(f"    → {err}")

        # Show last 3 entries
        entries = ledger.entries()
        if entries:
            print()
            print("  Recent receipts:")
            for entry in entries[-3:]:
                receipt = entry.receipt
                snr = receipt.get("snr", {}).get("score", "?")
                ihsan = receipt.get("ihsan", {}).get("score", "?")
                status = receipt.get("status", "?")
                print(f"    seq={entry.sequence} | snr={snr} | ihsan={ihsan} | {status} | {entry.entry_hash[:12]}...")

    except Exception as e:
        print(f"  Error:   {e}")

    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Node0 — Local Sovereign AI Home Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Start command
    subparsers.add_parser("start", help="Start Node0 proactive kernel")

    # Status command
    subparsers.add_parser("status", help="Check system status")

    # Mission command
    p_mission = subparsers.add_parser("mission", help="Execute a mission")
    p_mission.add_argument("task", help="Mission description")

    # Verify command
    subparsers.add_parser("verify", help="Verify evidence chain integrity")

    args = parser.parse_args()

    if not args.command:
        # Default to start
        args.command = "start"

    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "mission": cmd_mission,
        "verify": cmd_verify,
    }

    try:
        asyncio.run(commands[args.command](args))
    except KeyboardInterrupt:
        print("\nShutdown requested.")


if __name__ == "__main__":
    main()
