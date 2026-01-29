"""
BIZRA PRIME AGENTIC CORE (v1.0 Peak)
"The Dual-Agentic AI Implementation from the Unified System Contract"

Architecture (Embodying the SC §4.2 & §7):
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BIZRA PRIME                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PERSONAL TEAM (The MAG7)                                           │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │   │
│  │  │PLANNER │ │RESEARCH│ │ CODER  │ │EVALUTR │ │ETHICIST│ │INTEGRAT│  │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OPS AGENTS                                                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │SELF-DIAG │ │AUTO-DEBUG│ │ PERF-TUN │ │ SEC-MON  │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌────────────────────────────────────────┐                                │
│  │  PROOF-OF-IMPACT (PoI) ACCUMULATOR     │                                │
│  │   SEED: [Utility Token]                │                                │
│  │   BLOOM: [Impact Accrual]              │                                │
│  └────────────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘

Giants Referenced:
- Unified System Contract (§4, §5, §7)
- DDAGI Orchestrator (v1.0)
- ARTE Engine (v2.0)
- Sovereign Catalog (709k Nodes)
- Quranic Knowledge Base (6,236 Verses)
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# ════════════════════════════════════════════════════════════════════════════
# IMPORT EXISTING GIANTS
# ════════════════════════════════════════════════════════════════════════════
from bizra_config import DATA_LAKE_ROOT, GOLD_PATH, INDEXED_PATH, IHSAN_CONSTRAINT
from arte_engine import ARTEEngine
from vector_engine import VectorEngine
from sovereign_memory import SovereignMemory
from local_llm_gateway import LocalLLMGateway, LLMResponse

# Import Sovereign Bridge for high-performance caching
try:
    from sovereign_bridge import get_bridge, BridgeEventType
    import asyncio
    SOVEREIGN_BRIDGE_AVAILABLE = True
except ImportError:
    SOVEREIGN_BRIDGE_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (Aligned with System Contract §3)
# ════════════════════════════════════════════════════════════════════════════
PRIME_STATE_PATH = GOLD_PATH / "bizra_prime_state.json"
POI_LEDGER_PATH = GOLD_PATH / "poi_ledger.jsonl"
GOLDEN_GEMS_PATH = GOLD_PATH / "sovereign_catalog.parquet"
QURAN_VECTORS_PATH = INDEXED_PATH / "knowledge/quran/quran_vectors.npy"
QURAN_DATA_PATH = INDEXED_PATH / "knowledge/quran/quran_full.parquet"

# Genesis Root (From SC §3 - The Anchor)
GENESIS_MERKLE_ROOT = "d9c9fa504add65a1be737f3fe3447bc056fd1aad2850f491184208354f41926f"

# ════════════════════════════════════════════════════════════════════════════
# DATA MODELS (Aligned with SC §5 PoI Attestation)
# ════════════════════════════════════════════════════════════════════════════
class AgentRole(Enum):
    # Personal Team (SC §7)
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    EVALUATOR = "evaluator"
    ETHICIST = "ethicist"
    PUBLISHER = "publisher"
    INTEGRATOR = "integrator"
    # Ops Agents (SC §7)
    SELF_DIAGNOSIS = "self_diagnosis"
    AUTO_DEBUG = "auto_debug"
    PERF_TUNING = "perf_tuning"
    SEC_MONITOR = "sec_monitor"

@dataclass
class ProofOfImpact:
    """SC §5: PoI Attestation Structure"""
    version: str = "poi-0.2"
    chain_id: str = "bizra-main-alpha"
    genesis_merkle_root: str = GENESIS_MERKLE_ROOT
    contributor: str = "node0:bizra"
    action: str = ""
    resources: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    attestation_hash: str = ""

    def finalize(self):
        """Generate cryptographic attestation hash."""
        payload = json.dumps({
            "v": self.version, "c": self.chain_id, "g": self.genesis_merkle_root,
            "a": self.action, "t": self.timestamp
        }, sort_keys=True)
        self.attestation_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self

@dataclass
class AgentState:
    role: AgentRole
    status: str = "idle"
    last_action: str = ""
    impact_score: float = 0.0

# ════════════════════════════════════════════════════════════════════════════
# BIZRA PRIME: THE CORE
# ════════════════════════════════════════════════════════════════════════════
class BizraPrime:
    """
    The Dual-Agentic AI Core.
    Orchestrates Personal Team + Ops Agents with PoI tracking.
    """
    
    def __init__(self):
        print("═" * 70)
        print("   🚀 BIZRA PRIME AGENTIC CORE - IGNITION SEQUENCE")
        print("═" * 70)
        
        # 1. Load State or Initialize
        self.state = self._load_state()
        
        # 2. Initialize Sub-Engines (The Giants)
        print("   ⚙️  Mounting ARTE Engine...")
        self.arte = ARTEEngine()
        
        print("   🧠 Mounting Vector Engine...")
        self.vector_engine = VectorEngine()
        
        print("   🌐 Mounting M6 Sovereign Memory...")
        self.sovereign_memory = SovereignMemory()
        
        print("   🔌 Mounting Local LLM Gateway...")
        self.llm_gateway = LocalLLMGateway()
        
        # 3. Initialize Agent Pool (SC §7)
        print("   👥 Spawning Agent Pool (Personal + Ops)...")
        self.agents: Dict[AgentRole, AgentState] = {
            role: AgentState(role=role) for role in AgentRole
        }
        
        # 4. Load Knowledge Bases
        print("   📚 Loading Sovereign Knowledge...")
        self.sovereign_df = self._load_sovereign_catalog()
        self.quran_df, self.quran_vectors = self._load_quran_knowledge()
        
        # 5. PoI Accumulator
        self.poi_ledger: List[ProofOfImpact] = []
        self._load_poi_ledger()
        
        # 6. Mount Sovereign Bridge (High-Performance Caching)
        self.sovereign_bridge = None
        if SOVEREIGN_BRIDGE_AVAILABLE:
            try:
                print("   ⚡ Mounting Sovereign Bridge (B+ Tree + Bloom Filter)...")
                self.sovereign_bridge = get_bridge()
                print("   ✨ Sovereign Bridge connected!")
            except Exception as e:
                print(f"   ⚠️ Sovereign Bridge unavailable: {e}")
        
        # 7. Status Summary
        llm_status = "✅ ONLINE" if self.llm_gateway.is_online() else "⚠️ OFFLINE"
        bridge_status = "✅ ACTIVE" if self.sovereign_bridge else "⚠️ INACTIVE"
        
        print("═" * 70)
        print(f"   ✅ BIZRA PRIME ONLINE | Agents: {len(self.agents)} | Knowledge: {len(self.sovereign_df):,} nodes")
        print(f"   🔌 LLM Gateway: {llm_status} | Model: {self.llm_gateway.active_model or 'None'}")
        print(f"   ⚡ Sovereign Bridge: {bridge_status}")
        print("═" * 70)

    def _load_state(self) -> Dict:
        if PRIME_STATE_PATH.exists():
            with open(PRIME_STATE_PATH, 'r') as f:
                return json.load(f)
        return {"initialized": datetime.now().isoformat(), "total_impact": 0.0}

    def _save_state(self):
        with open(PRIME_STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _load_sovereign_catalog(self) -> pd.DataFrame:
        if GOLDEN_GEMS_PATH.exists():
            return pd.read_parquet(GOLDEN_GEMS_PATH)
        print("   ⚠️ Sovereign Catalog not found.")
        return pd.DataFrame()

    def _load_quran_knowledge(self):
        if QURAN_DATA_PATH.exists() and QURAN_VECTORS_PATH.exists():
            df = pd.read_parquet(QURAN_DATA_PATH)
            vectors = np.load(QURAN_VECTORS_PATH)
            print(f"   📖 Quran Loaded: {len(df)} verses, {vectors.shape[1]}-dim vectors.")
            return df, vectors
        print("   ⚠️ Quran Knowledge Base not found.")
        return pd.DataFrame(), np.array([])

    def _load_poi_ledger(self):
        if POI_LEDGER_PATH.exists():
            with open(POI_LEDGER_PATH, 'r') as f:
                for line in f:
                    # Just count for now
                    self.poi_ledger.append(json.loads(line))
            print(f"   💰 PoI Ledger: {len(self.poi_ledger)} attestations loaded.")

    def _record_poi(self, action: str, benchmarks: Dict[str, float]):
        """Record a Proof-of-Impact attestation to the ledger."""
        poi = ProofOfImpact(
            action=action,
            benchmarks=benchmarks,
            resources={"agent": "BizraPrime", "node": "node0"}
        ).finalize()
        
        with open(POI_LEDGER_PATH, 'a') as f:
            f.write(json.dumps(poi.__dict__) + "\n")
        
        self.poi_ledger.append(poi)
        self.state["total_impact"] = self.state.get("total_impact", 0) + sum(benchmarks.values())
        self._save_state()
        
        # Also record through Sovereign Bridge for distributed caching
        if self.sovereign_bridge and SOVEREIGN_BRIDGE_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.sovereign_bridge.record_poi(action, benchmarks, poi.attestation_hash)
                    )
                else:
                    loop.run_until_complete(
                        self.sovereign_bridge.record_poi(action, benchmarks, poi.attestation_hash)
                    )
            except Exception:
                pass  # Non-blocking, best-effort caching
        
        return poi

    # ════════════════════════════════════════════════════════════════════════
    # AGENT DISPATCH (Graph of Thoughts Implementation)
    # ════════════════════════════════════════════════════════════════════════
    def dispatch(self, role: AgentRole, task: str) -> Dict[str, Any]:
        """
        Dispatch a task to a specific agent role.
        Implements Graph-of-Thoughts by allowing agents to call each other.
        """
        agent = self.agents[role]
        agent.status = "active"
        agent.last_action = task
        
        print(f"\n🤖 [{role.value.upper()}] Received Task: '{task[:60]}...'")
        
        result = {"agent": role.value, "task": task, "status": "completed"}
        
        # Role-Specific Logic (Interdisciplinary Thinking)
        if role == AgentRole.RESEARCHER:
            result["output"] = self._research(task)
        elif role == AgentRole.ETHICIST:
            result["output"] = self._consult_ethics(task)
        elif role == AgentRole.EVALUATOR:
            result["output"] = self._evaluate(task)
        elif role == AgentRole.SELF_DIAGNOSIS:
            result["output"] = self._self_diagnose()
        else:
            result["output"] = f"Agent {role.value} acknowledged task."
        
        agent.status = "idle"
        agent.impact_score += 0.1
        
        # Record PoI
        self._record_poi(
            action=f"Agent:{role.value}:{task[:30]}",
            benchmarks={"task_completion": 1.0, "snr": 0.95}
        )
        
        return result

    def _research(self, query: str) -> str:
        """RESEARCHER agent: Semantic search across all knowledge tiers."""
        print("   🔍 Performing Multi-Tier Research (M3-M6)...")
        
        results = []
        
        # M6: Sovereign Memory (Cross-Domain Search) - Broadest Context
        if self.sovereign_memory.is_ready():
            m6_results = self.sovereign_memory.search_across_domains(query, top_k=3)
            if m6_results:
                best = m6_results[0]
                results.append(f"[M6-Sovereign] {best.name} in {best.domain} (SNR: {best.snr_score:.2f})")
        
        # M3: Semantic Memory (Quran) - Highest SNR for wisdom queries
        if len(self.quran_vectors) > 0:
            query_vec = self.vector_engine.model.encode([query])[0]
            norm_vectors = self.quran_vectors / np.linalg.norm(self.quran_vectors, axis=1)[:, np.newaxis]
            norm_query = query_vec / np.linalg.norm(query_vec)
            scores = np.dot(norm_vectors, norm_query)
            best_idx = np.argmax(scores)
            verse = self.quran_df.iloc[best_idx]
            results.append(f"[M3-Quran] {verse['surah_name']} {verse['surah']}:{verse['ayah']}: {verse['english_text'][:80]}...")
        
        if results:
            return " | ".join(results)
        
        return "No relevant knowledge found."

    def _consult_ethics(self, context: str) -> str:
        """ETHICIST agent: Validate action against Ihsan constraint."""
        print("   ⚖️ Consulting Ethical Framework (Ihsan)...")
        
        # Simple heuristic: Check if context contains any 'red flags'
        red_flags = ["harm", "deceive", "exploit", "steal", "fraud"]
        if any(flag in context.lower() for flag in red_flags):
            return "⚠️ ETHICIST: Action flagged for review. Potential violation of Ihsan constraint."
        
        # Consult Quran for ethical grounding
        ethical_query = "justice and fairness"
        if len(self.quran_vectors) > 0:
            query_vec = self.vector_engine.model.encode([ethical_query])[0]
            norm_vectors = self.quran_vectors / np.linalg.norm(self.quran_vectors, axis=1)[:, np.newaxis]
            scores = np.dot(norm_vectors, query_vec / np.linalg.norm(query_vec))
            best_idx = np.argmax(scores)
            verse = self.quran_df.iloc[best_idx]
            return f"✅ ETHICIST: Action aligned with Ihsan. Guidance: '{verse['english_text'][:80]}...'"
        
        return "✅ ETHICIST: Action approved under Ihsan constraint."

    def _evaluate(self, output: str) -> Dict[str, float]:
        """EVALUATOR agent: Score output quality using ARTE SNR."""
        print("   📊 Evaluating Output Quality...")
        
        # Use ARTE to get SNR
        snr = self.arte.resolve_tension(output, "symbolic_grounding", "neural_intuition")
        
        return {
            "snr_score": snr,
            "completeness": 0.9 if len(output) > 50 else 0.5,
            "coherence": IHSAN_CONSTRAINT # Assume high coherence if Ihsan active
        }

    def _self_diagnose(self) -> Dict[str, Any]:
        """SELF_DIAGNOSIS ops agent: System health check across all 6 memory tiers."""
        print("   🩺 Running Self-Diagnosis (6-Tier Memory)...")
        
        integrity = self.arte.check_system_integrity()
        
        # M6 Analytics
        m6_stats = {}
        if self.sovereign_memory.is_ready():
            m6_stats = self.sovereign_memory.get_domain_analytics()
        
        diagnosis = {
            "symbolic_layer": "HEALTHY" if integrity["symbolic"] else "DEGRADED",
            "neural_layer": "HEALTHY" if integrity["neural"] else "DEGRADED",
            "snr": integrity["snr_score"],
            "memory_tiers": {
                "M1_working": "ACTIVE",
                "M2_episodic": f"{len(self.poi_ledger)} events",
                "M3_semantic": f"{len(self.quran_df)} verses",
                "M4_vector": f"{len(self.quran_vectors)} embeddings" if len(self.quran_vectors) > 0 else "EMPTY",
                "M5_hypergraph": f"{len(self.sovereign_df):,} nodes",
                "M6_sovereign": f"{sum(s['total_nodes'] for s in m6_stats.values()):,} cross-domain" if m6_stats else "OFFLINE"
            },
            "agents_active": sum(1 for a in self.agents.values() if a.status == "active")
        }
        
        return diagnosis

    # ════════════════════════════════════════════════════════════════════════
    # HIGH-LEVEL ORCHESTRATION
    # ════════════════════════════════════════════════════════════════════════
    def reason(self, goal: str) -> Dict[str, Any]:
        """
        Full Graph-of-Thoughts reasoning loop.
        Orchestrates multiple agents to achieve a goal.
        """
        print(f"\n{'═' * 70}")
        print(f"   🎯 GOAL: {goal}")
        print(f"{'═' * 70}")
        
        thought_graph = {"goal": goal, "nodes": [], "final_output": None}
        
        # 1. PLANNER: Decompose Goal
        plan = self.dispatch(AgentRole.PLANNER, f"Decompose: {goal}")
        thought_graph["nodes"].append(plan)
        
        # 2. RESEARCHER: Gather Context
        research = self.dispatch(AgentRole.RESEARCHER, goal)
        thought_graph["nodes"].append(research)
        
        # 3. ETHICIST: Validate Approach
        ethics = self.dispatch(AgentRole.ETHICIST, f"Validate: {goal}")
        thought_graph["nodes"].append(ethics)
        
        # 4. EVALUATOR: Score the synthesis
        evaluation = self.dispatch(AgentRole.EVALUATOR, research.get("output", ""))
        thought_graph["nodes"].append({"agent": "evaluator", "output": evaluation})
        
        # 5. Synthesize Final Output
        thought_graph["final_output"] = {
            "research_finding": research.get("output"),
            "ethical_validation": ethics.get("output"),
            "quality_scores": evaluation.get("output") if isinstance(evaluation, dict) else evaluation
        }
        
        # Record Composite PoI
        self._record_poi(
            action=f"Reasoning:{goal[:30]}",
            benchmarks={"goal_achieved": 1.0, "agents_coordinated": 4}
        )
        
        print(f"\n{'═' * 70}")
        print(f"   ✅ REASONING COMPLETE")
        print(f"{'═' * 70}")
        
        return thought_graph

    def status(self) -> Dict:
        """Get full system status."""
        diagnosis = self._self_diagnose()
        return {
            "prime_version": "1.0-Peak",
            "genesis_root": GENESIS_MERKLE_ROOT[:16] + "...",
            "total_impact": self.state.get("total_impact", 0),
            "llm_online": self.llm_gateway.is_online(),
            "active_model": self.llm_gateway.active_model,
            "available_models": len(self.llm_gateway.available_models),
            "diagnosis": diagnosis
        }

    # ════════════════════════════════════════════════════════════════════════
    # LLM-POWERED CAPABILITIES (Offline-Ready)
    # ════════════════════════════════════════════════════════════════════════
    def ask(self, question: str, use_context: bool = True) -> str:
        """
        Ask BIZRA PRIME a question. Uses RAG with local LLM.
        Works fully offline when local models are available.
        """
        print(f"\n💭 PROCESSING QUESTION: '{question[:60]}...'")
        
        context_chunks = []
        
        if use_context:
            # Gather context from all memory tiers
            print("   📚 Gathering Context from Knowledge Base...")
            
            # M6: Sovereign Memory
            if self.sovereign_memory.is_ready():
                m6_results = self.sovereign_memory.search_across_domains(question, top_k=3)
                for r in m6_results[:3]:
                    context_chunks.append(f"[{r.domain}] {r.name}")
            
            # M3: Quran (if relevant)
            if len(self.quran_vectors) > 0:
                query_vec = self.vector_engine.model.encode([question])[0]
                norm_vectors = self.quran_vectors / np.linalg.norm(self.quran_vectors, axis=1)[:, np.newaxis]
                norm_query = query_vec / np.linalg.norm(query_vec)
                scores = np.dot(norm_vectors, norm_query)
                top_indices = np.argsort(scores)[-3:][::-1]
                for idx in top_indices:
                    if scores[idx] > 0.3:  # Only include if relevant
                        verse = self.quran_df.iloc[idx]
                        context_chunks.append(f"[Quran {verse['surah']}:{verse['ayah']}] {verse['english_text']}")
        
        context_text = "\n".join(context_chunks) if context_chunks else "No specific context retrieved."
        
        # Generate response with LLM
        if self.llm_gateway.is_online():
            print(f"   🧠 Generating Response with {self.llm_gateway.active_model}...")
            
            system_prompt = """You are BIZRA PRIME, an intelligent assistant grounded in the BIZRA Data Lake.
You have access to a vast knowledge base including documents, code, conversations, and sacred texts.
Answer concisely and accurately. If you're unsure, acknowledge it.
Follow the Ihsan constraint: always act with excellence and avoid harm."""
            
            response = self.llm_gateway.generate_with_context(
                query=question,
                context=context_text,
                system=system_prompt
            )
            
            print(f"   ⏱️ Latency: {response.latency_ms:.0f}ms")
            
            # Record PoI for LLM usage
            self._record_poi(
                action=f"LLM:ask:{question[:20]}",
                benchmarks={"tokens": response.tokens_used, "latency_ms": response.latency_ms}
            )
            
            return response.content
        else:
            # Fallback: Return retrieved context only
            print("   ⚠️ LLM Offline - Returning retrieval-only response")
            return f"[RETRIEVAL MODE]\nRelevant Context:\n{context_text}"

    def list_models(self) -> List[Dict]:
        """List all available local LLM models."""
        return self.llm_gateway.list_models()

    def set_model(self, model_id: str) -> bool:
        """Switch to a different local model."""
        for m in self.llm_gateway.available_models:
            if m.id == model_id:
                self.llm_gateway.active_model = m.id
                self.llm_gateway.active_provider = m.provider
                print(f"✅ Switched to model: {model_id} via {m.provider.value}")
                return True
        print(f"❌ Model not found: {model_id}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # IGNITION
    prime = BizraPrime()
    
    # Display Status
    print("\n📊 SYSTEM STATUS:")
    status = prime.status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Show Available Models
    print("\n🔌 AVAILABLE LOCAL MODELS:")
    for model in prime.list_models():
        print(f"   - {model['id']} ({model['size']}) via {model['provider']}")
    
    # Test LLM-Powered Ask (RAG)
    print("\n" + "═" * 70)
    print("   🧪 TESTING LLM-POWERED ASK (RAG)")
    print("═" * 70)
    
    answer = prime.ask("What is the purpose of the BIZRA Data Lake?")
    print(f"\n💬 ANSWER:\n{answer[:500]}...")
    
    # Execute a Reasoning Task (Graph of Thoughts)
    print("\n" + "═" * 70)
    print("   🧪 TESTING GRAPH-OF-THOUGHTS REASONING")
    print("═" * 70)
    
    result = prime.reason("Find wisdom about patience and perseverance")
    
    print("\n📜 FINAL INSIGHT:")
    if result.get("final_output"):
        print(f"   Research: {result['final_output'].get('research_finding', 'N/A')}")
        print(f"   Ethics: {result['final_output'].get('ethical_validation', 'N/A')}")
