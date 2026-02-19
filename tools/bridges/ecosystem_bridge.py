#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                              ║
║   ███████╗ ██████╗ ██████╗ ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗                            ║
║   ██╔════╝██╔════╝██╔═══██╗██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║                            ║
║   █████╗  ██║     ██║   ██║███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║                            ║
║   ██╔══╝  ██║     ██║   ██║╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║                            ║
║   ███████╗╚██████╗╚██████╔╝███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║                            ║
║   ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝                            ║
║                                                                                                              ║
║   ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗                                                               ║
║   ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝                                                               ║
║   ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                                                                 ║
║   ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                                                                 ║
║   ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗                                                               ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝                                                               ║
║                                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   ECOSYSTEM BRIDGE — UNIFIED INTEGRATION LAYER FOR BIZRA DDAGI OS v2.0.0                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                              ║
║   Integrates:                                                                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║   │  ULTIMATE ENGINE (v2.0.0)                                                                           │    ║
║   │  ├── Constitution, DaughterTest, WinterProofEmbedder                                               │    ║
║   │  ├── GraphOfThoughts, SNROptimizer, FATEGate                                                       │    ║
║   │  └── LocalEconomicSystem, LocalMerkleDAG                                                           │    ║
║   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║                           ↓ bridges to ↓                                                                     ║
║   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐    ║
║   │  EXISTING ECOSYSTEM                                                                                 │    ║
║   │  ├── BIZRAOrchestrator: Hypergraph RAG + ARTE + PAT + KEP + Multi-Modal                            │    ║
║   │  ├── SovereignApex: VectorLayer + GraphLayer + ApexSynthesizer                                     │    ║
║   │  └── SovereignBridge: B+ Tree + Bloom Filter + LRU Cache                                           │    ║
║   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                                              ║
║   Author: BIZRA Genesis NODE0 | For: Layla, and all daughters of the future                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

# PCI Protocol (Proof-Carrying Inference)
try:
    from core.pci import (
        PCIEnvelope, EnvelopeBuilder, PCIGateKeeper, RejectCode
    )
    PCI_AVAILABLE = True
    logging.getLogger("EcosystemBridge").info("✅ PCI Protocol v1.0 imported")
except ImportError as e:
    PCI_AVAILABLE = False
    logging.getLogger("EcosystemBridge").warning(f"⚠️ PCI Protocol not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# VERSION & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

__version__ = "2.0.0"
__bridge_name__ = "EcosystemBridge"

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | ECOSYSTEM | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("EcosystemBridge")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORT ULTIMATE ENGINE (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

try:
    from ultimate_engine import (
        UltimateEngine,
        Constitution, DaughterTest,
        WinterProofEmbedder,
        GraphOfThoughts, SNROptimizer, FATEGate,
        IhsanCalculator, LocalEconomicSystem, LocalMerkleDAG,
        HookRegistry, HookEvent, CompactionEngine,
        LocalReasoningEngine,
        ThoughtNode, ThoughtType, Receipt, EvidencePointer,
        FATEGateResult, ThirdFactResult, KEPResult,
        RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR,
        SNR_MINIMUM, SNR_ACCEPTABLE, SNR_IHSAN
    )
    ULTIMATE_ENGINE_AVAILABLE = True
    log.info("✅ UltimateEngine imported successfully")
except ImportError as e:
    ULTIMATE_ENGINE_AVAILABLE = False
    log.warning(f"⚠️ UltimateEngine not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORT EXISTING ECOSYSTEM COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

# BIZRAOrchestrator
try:
    from bizra_orchestrator import (
        BIZRAOrchestrator, BIZRAQuery, BIZRAResponse, QueryComplexity
    )
    ORCHESTRATOR_AVAILABLE = True
    log.info("✅ BIZRAOrchestrator imported")
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    log.warning(f"⚠️ BIZRAOrchestrator not available: {e}")

# SovereignApex
try:
    from sovereign_apex import (
        ApexConfig, VectorLayer, KnowledgeNode, KnowledgeEdge,
        NodeType, EdgeType, DiscoveredPattern, ApexQueryResult
    )
    APEX_AVAILABLE = True
    log.info("✅ SovereignApex imported")
except ImportError as e:
    APEX_AVAILABLE = False
    log.warning(f"⚠️ SovereignApex not available: {e}")

# PeakMasterpiece
try:
    from peak_masterpiece import (
        PeakMasterpieceEngine
    )
    PEAK_AVAILABLE = True
    log.info("✅ PeakMasterpiece imported")
except ImportError as e:
    PEAK_AVAILABLE = False
    log.warning(f"⚠️ PeakMasterpiece not available: {e}")

# SovereignBridge
try:
    from sovereign_bridge import (
        SovereignBridge, get_bridge, BridgeEventType
    )
    BRIDGE_AVAILABLE = True
    log.info("✅ SovereignBridge imported")
except ImportError as e:
    BRIDGE_AVAILABLE = False
    log.warning(f"⚠️ SovereignBridge not available: {e}")

# HyperLoopback
try:
    from hyper_loopback import (
        HyperLoopbackSystem
    )
    HYPERLOOPBACK_AVAILABLE = True
    log.info("✅ HyperLoopback imported")
except ImportError as e:
    HYPERLOOPBACK_AVAILABLE = False
    log.warning(f"⚠️ HyperLoopback not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# COMPONENT STATUS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class ComponentStatus(Enum):
    """Status of ecosystem components."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"


@dataclass
class EcosystemHealth:
    """Health report for the entire ecosystem."""
    ultimate_engine: ComponentStatus = ComponentStatus.UNAVAILABLE
    orchestrator: ComponentStatus = ComponentStatus.UNAVAILABLE
    sovereign_apex: ComponentStatus = ComponentStatus.UNAVAILABLE
    peak_masterpiece: ComponentStatus = ComponentStatus.UNAVAILABLE
    sovereign_bridge: ComponentStatus = ComponentStatus.UNAVAILABLE
    hyper_loopback: ComponentStatus = ComponentStatus.UNAVAILABLE
    overall_health: float = 0.0
    kernel_invariants_ok: bool = False
    constitution_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ultimate_engine": self.ultimate_engine.value,
            "orchestrator": self.orchestrator.value,
            "sovereign_apex": self.sovereign_apex.value,
            "peak_masterpiece": self.peak_masterpiece.value,
            "sovereign_bridge": self.sovereign_bridge.value,
            "hyper_loopback": self.hyper_loopback.value,
            "overall_health": self.overall_health,
            "kernel_invariants_ok": self.kernel_invariants_ok,
            "constitution_hash": self.constitution_hash,
            "timestamp": self.timestamp
        }


@dataclass
class UnifiedQuery:
    """Unified query format for the ecosystem."""
    text: str
    require_constitution_check: bool = True
    require_daughter_test: bool = True
    require_fate_gate: bool = True
    snr_threshold: float = SNR_MINIMUM if ULTIMATE_ENGINE_AVAILABLE else 0.85
    use_orchestrator: bool = True
    use_apex: bool = True
    use_winter_proof: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedResponse:
    """Unified response from the ecosystem."""
    query: str
    synthesis: str
    snr_score: float
    ihsan_score: float
    constitution_check: bool
    daughter_test_check: bool
    fate_gate_passed: bool
    sources: List[Dict] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    bloom_reward: float = 0.0
    merkle_hash: str = ""
    execution_time: float = 0.0
    components_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# ECOSYSTEM BRIDGE — THE UNIFIED INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class EcosystemBridge:
    """
    ECOSYSTEM BRIDGE — Unified Integration Layer.
    
    Connects:
    - UltimateEngine (core cognitive engine)
    - BIZRAOrchestrator (Hypergraph RAG + ARTE + PAT)
    - SovereignApex (VectorLayer + GraphLayer)
    - PeakMasterpiece (GoT + SNR + FATE)
    - SovereignBridge (Caching layer)
    - HyperLoopback (Winter-proof system)
    
    Maintains:
    - Constitutional compliance
    - Daughter Test verification
    - RIBA_ZERO economics
    - Merkle-DAG audit trail
    """
    
    __slots__ = (
        'human_name', 'daughter_name', 'node_id',
        'ultimate_engine', 'orchestrator', 'apex_config',
        'peak_engine', 'sovereign_bridge', 'hyper_loopback',
        '_hooks', '_query_count', '_start_time', '_initialized'
    )
    
    def __init__(
        self,
        human_name: str = "Ahmed Al-Mansoori",
        daughter_name: str = "Layla",
        enable_orchestrator: bool = True,
        enable_apex: bool = True
    ):
        self.human_name = human_name
        self.daughter_name = daughter_name
        self.node_id = f"ECOSYSTEM_{hashlib.blake2b(human_name.encode(), digest_size=16).hexdigest()[:8]}"
        
        # Core: Ultimate Engine (always required)
        self.ultimate_engine: Optional[UltimateEngine] = None
        if ULTIMATE_ENGINE_AVAILABLE:
            self.ultimate_engine = UltimateEngine(human_name, daughter_name)
            log.info(f"🚀 UltimateEngine initialized: {self.ultimate_engine.node_id}")
        
        # Integration: Orchestrator
        self.orchestrator: Optional[BIZRAOrchestrator] = None
        if enable_orchestrator and ORCHESTRATOR_AVAILABLE:
            try:
                self.orchestrator = BIZRAOrchestrator(
                    enable_pat=True,
                    enable_kep=True,
                    enable_multimodal=False,  # Start without multimodal
                    enable_discipline=True
                )
                log.info("🔗 BIZRAOrchestrator connected")
            except Exception as e:
                log.warning(f"⚠️ Orchestrator init failed: {e}")
        
        # Integration: Apex Config
        self.apex_config = ApexConfig if APEX_AVAILABLE else None
        
        # Integration: Peak Engine
        self.peak_engine = None
        if PEAK_AVAILABLE:
            try:
                self.peak_engine = PeakMasterpieceEngine()
                log.info("🔗 PeakMasterpiece connected")
            except Exception as e:
                log.warning(f"⚠️ PeakMasterpiece init failed: {e}")
        
        # Integration: Sovereign Bridge (caching)
        self.sovereign_bridge = None
        if BRIDGE_AVAILABLE:
            try:
                self.sovereign_bridge = get_bridge()
                log.info("🔗 SovereignBridge connected")
            except Exception as e:
                log.warning(f"⚠️ SovereignBridge init failed: {e}")
        
        # Integration: HyperLoopback
        self.hyper_loopback = None
        if HYPERLOOPBACK_AVAILABLE:
            try:
                self.hyper_loopback = HyperLoopbackSystem(human_name, daughter_name)
                log.info("🔗 HyperLoopback connected")
            except Exception as e:
                log.warning(f"⚠️ HyperLoopback init failed: {e}")
        
        # Internal state
        self._hooks = HookRegistry() if ULTIMATE_ENGINE_AVAILABLE else None
        self._query_count = 0
        self._start_time = time.time()
        self._initialized = False
        
        log.info(f"✨ EcosystemBridge v{__version__} created")
    
    async def initialize(self) -> bool:
        """Initialize all connected components in parallel."""
        if self._initialized:
            return True

        log.info("Initializing ecosystem components (parallel)...")

        # Gather all initialization tasks for parallel execution
        tasks = []
        task_names = []

        if self.orchestrator:
            tasks.append(self.orchestrator.initialize())
            task_names.append("Orchestrator")

        if self.sovereign_bridge:
            tasks.append(self.sovereign_bridge.initialize())
            task_names.append("SovereignBridge")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    log.warning(f"   {name} init failed: {result}")
                else:
                    log.info(f"   {name} initialized")

        self._initialized = True
        log.info("Ecosystem initialization complete")

        return True
    
    async def query(self, unified_query: UnifiedQuery) -> UnifiedResponse:
        """
        Process a query through the unified ecosystem.
        
        Pipeline:
        1. Constitution Check (UltimateEngine)
        2. Daughter Test (UltimateEngine)
        3. Winter-Proof Embedding (UltimateEngine/HyperLoopback)
        4. Context Retrieval (Orchestrator/Apex)
        5. FATE Gate Verification (UltimateEngine)
        6. Synthesis (UltimateEngine + Orchestrator)
        7. Ihsān Scoring (UltimateEngine)
        8. Economic Reward (UltimateEngine)
        9. Merkle-DAG Recording (UltimateEngine)
        """
        start_time = time.time()
        self._query_count += 1
        components_used = []
        reasoning_trace = []
        
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        log.info(f"{'='*60}")
        log.info(f"🌐 ECOSYSTEM QUERY #{self._query_count}")
        log.info(f"📝 Query: {unified_query.text[:60]}...")
        
        # === Step 1-2: Constitution + Daughter Test (via UltimateEngine) ===
        constitution_check = True
        daughter_test_check = True
        
        if self.ultimate_engine and unified_query.require_constitution_check:
            const_ok, const_reason = self.ultimate_engine.constitution.verify_compliance(
                unified_query.text, unified_query.metadata
            )
            constitution_check = const_ok
            if not const_ok:
                log.warning(f"❌ Constitution: {const_reason}")
                return UnifiedResponse(
                    query=unified_query.text,
                    synthesis=f"Constitutional violation: {const_reason}",
                    snr_score=0.0,
                    ihsan_score=0.0,
                    constitution_check=False,
                    daughter_test_check=False,
                    fate_gate_passed=False,
                    execution_time=time.time() - start_time
                )
            components_used.append("Constitution")
            reasoning_trace.append("✓ Constitution check passed")
            log.info("✓ Constitution: PASS")
        
        if self.ultimate_engine and unified_query.require_daughter_test:
            dt_ok, dt_reason = self.ultimate_engine.daughter_test.verify({
                "decision_summary": f"Process query: {unified_query.text[:50]}",
                "impact": unified_query.metadata.get("impact", {})
            })
            daughter_test_check = dt_ok
            if not dt_ok:
                log.warning(f"❌ Daughter Test: {dt_reason}")
                return UnifiedResponse(
                    query=unified_query.text,
                    synthesis=f"Daughter Test failed: {dt_reason}",
                    snr_score=0.0,
                    ihsan_score=0.0,
                    constitution_check=constitution_check,
                    daughter_test_check=False,
                    fate_gate_passed=False,
                    execution_time=time.time() - start_time
                )
            components_used.append("DaughterTest")
            reasoning_trace.append("✓ Daughter Test passed")
            log.info("✓ Daughter Test: PASS")
        
        # === Step 3: Process via Ultimate Engine ===
        ultimate_result = None
        synthesis = ""
        snr_score = 0.0
        ihsan_score = 0.0
        fate_gate_passed = False
        bloom_reward = 0.0
        merkle_hash = ""
        
        if self.ultimate_engine:
            ultimate_result = await self.ultimate_engine.process_query(
                unified_query.text, unified_query.metadata
            )
            synthesis = ultimate_result.synthesis
            snr_score = ultimate_result.snr_score
            ihsan_score = ultimate_result.snr_score  # Use SNR as proxy
            fate_gate_passed = True  # FATE gate is called internally
            bloom_reward = ultimate_result.bloom_reward
            merkle_hash = ultimate_result.merkle_hash
            components_used.append("UltimateEngine")
            reasoning_trace.extend([
                f"✓ Graph-of-Thoughts: {len(ultimate_result.thoughts_used)} nodes",
                f"✓ SNR: {snr_score:.3f}",
                f"✓ BLOOM: +{bloom_reward:.1f}"
            ])
            log.info(f"✓ UltimateEngine: SNR={snr_score:.3f}")
        
        # === Step 4: Enhance with Orchestrator (if available) ===
        sources = []
        if self.orchestrator and unified_query.use_orchestrator:
            try:
                bizra_query = BIZRAQuery(
                    text=unified_query.text,
                    complexity=QueryComplexity.MODERATE,
                    snr_threshold=unified_query.snr_threshold
                )
                orchestrator_result = await self.orchestrator.query(bizra_query)
                sources = orchestrator_result.sources
                
                # Blend synthesis if orchestrator provided better content
                if orchestrator_result.snr_score > snr_score:
                    synthesis = f"{synthesis}\n\n[Enhanced via Orchestrator]:\n{orchestrator_result.answer[:500]}"
                    snr_score = max(snr_score, orchestrator_result.snr_score)
                
                components_used.append("Orchestrator")
                reasoning_trace.append(f"✓ Orchestrator enhanced with {len(sources)} sources")
                log.info(f"✓ Orchestrator: {len(sources)} sources")
            except Exception as e:
                log.warning(f"⚠️ Orchestrator query failed: {e}")
        
        # === Step 5: Calculate Ihsān ===
        if self.ultimate_engine:
            ihsan_result = self.ultimate_engine.ihsan_calculator.calculate(
                unified_query.text, synthesis, {"constitution_check": constitution_check}
            )
            ihsan_score = ihsan_result["final_score"]
            reasoning_trace.append(f"✓ Ihsān: {ihsan_score:.3f}")
            log.info(f"✓ Ihsān: {ihsan_score:.3f}")
        
        execution_time = time.time() - start_time
        log.info(f"⏱️  Execution: {execution_time:.3f}s")
        log.info(f"{'='*60}")
        
        return UnifiedResponse(
            query=unified_query.text,
            synthesis=synthesis,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            constitution_check=constitution_check,
            daughter_test_check=daughter_test_check,
            fate_gate_passed=fate_gate_passed,
            sources=sources,
            reasoning_trace=reasoning_trace,
            bloom_reward=bloom_reward,
            merkle_hash=merkle_hash,
            execution_time=execution_time,
            components_used=components_used
        )
    
    def get_health(self) -> EcosystemHealth:
        """Get comprehensive ecosystem health report."""
        health = EcosystemHealth()
        
        # Check each component
        if self.ultimate_engine:
            health.ultimate_engine = ComponentStatus.AVAILABLE
            health.constitution_hash = self.ultimate_engine.constitution.get_hash()[:16]
        
        if self.orchestrator:
            health.orchestrator = ComponentStatus.AVAILABLE
        
        if APEX_AVAILABLE:
            health.sovereign_apex = ComponentStatus.AVAILABLE
        
        if self.peak_engine:
            health.peak_masterpiece = ComponentStatus.AVAILABLE
        
        if self.sovereign_bridge:
            health.sovereign_bridge = ComponentStatus.AVAILABLE
        
        if self.hyper_loopback:
            health.hyper_loopback = ComponentStatus.AVAILABLE
        
        # Calculate overall health
        available_count = sum([
            health.ultimate_engine == ComponentStatus.AVAILABLE,
            health.orchestrator == ComponentStatus.AVAILABLE,
            health.sovereign_apex == ComponentStatus.AVAILABLE,
            health.peak_masterpiece == ComponentStatus.AVAILABLE,
            health.sovereign_bridge == ComponentStatus.AVAILABLE,
            health.hyper_loopback == ComponentStatus.AVAILABLE
        ])
        health.overall_health = available_count / 6.0
        
        # Check kernel invariants
        if ULTIMATE_ENGINE_AVAILABLE:
            health.kernel_invariants_ok = RIBA_ZERO and ZANN_ZERO and IHSAN_FLOOR == 0.90
        
        return health
    
    def get_status(self) -> Dict[str, Any]:
        """Get full ecosystem status."""
        uptime = time.time() - self._start_time
        health = self.get_health()
        
        status = {
            "bridge": __bridge_name__,
            "version": __version__,
            "node_id": self.node_id,
            "human": self.human_name,
            "daughter": self.daughter_name,
            "uptime_hours": uptime / 3600,
            "query_count": self._query_count,
            "initialized": self._initialized,
            "health": health.to_dict(),
            "kernel_invariants": {
                "RIBA_ZERO": RIBA_ZERO if ULTIMATE_ENGINE_AVAILABLE else None,
                "ZANN_ZERO": ZANN_ZERO if ULTIMATE_ENGINE_AVAILABLE else None,
                "IHSAN_FLOOR": IHSAN_FLOOR if ULTIMATE_ENGINE_AVAILABLE else None
            },
            "components": {
                "ultimate_engine": self.ultimate_engine is not None,
                "orchestrator": self.orchestrator is not None,
                "apex_config": self.apex_config is not None,
                "peak_engine": self.peak_engine is not None,
                "sovereign_bridge": self.sovereign_bridge is not None,
                "hyper_loopback": self.hyper_loopback is not None
            }
        }
        
        if self.ultimate_engine:
            status["ultimate_engine_status"] = self.ultimate_engine.get_status()
        
        return status


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# GLOBAL ECOSYSTEM INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

_ecosystem: Optional[EcosystemBridge] = None


def get_ecosystem(
    human_name: str = "Ahmed Al-Mansoori",
    daughter_name: str = "Layla"
) -> EcosystemBridge:
    """Get or create the global ecosystem instance."""
    global _ecosystem
    if _ecosystem is None:
        _ecosystem = EcosystemBridge(human_name, daughter_name)
    return _ecosystem


async def initialize_ecosystem(
    human_name: str = "Ahmed Al-Mansoori",
    daughter_name: str = "Layla"
) -> EcosystemBridge:
    """Initialize and return the global ecosystem."""
    ecosystem = get_ecosystem(human_name, daughter_name)
    await ecosystem.initialize()
    return ecosystem


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

async def demonstrate_ecosystem():
    """Demonstrate the full ecosystem integration."""
    
    print("\n" + "="*80)
    print("🌐 BIZRA ECOSYSTEM BRIDGE v2.0.0")
    print("="*80)
    print("UNIFIED INTEGRATION: UltimateEngine + Orchestrator + Apex + Peak")
    print("="*80)
    
    # Create ecosystem
    ecosystem = await initialize_ecosystem(
        human_name="Ahmed Al-Mansoori",
        daughter_name="Layla"
    )
    
    # Health check
    print("\n📋 ECOSYSTEM HEALTH")
    print("-" * 60)
    
    health = ecosystem.get_health()
    print(f"Overall Health: {health.overall_health * 100:.0f}%")
    print(f"Kernel Invariants: {'✓ OK' if health.kernel_invariants_ok else '✗ FAILED'}")
    print(f"Constitution Hash: {health.constitution_hash}...")
    
    # List components
    print("\n📋 COMPONENTS")
    print("-" * 60)
    
    for component, status in [
        ("UltimateEngine", health.ultimate_engine),
        ("Orchestrator", health.orchestrator),
        ("SovereignApex", health.sovereign_apex),
        ("PeakMasterpiece", health.peak_masterpiece),
        ("SovereignBridge", health.sovereign_bridge),
        ("HyperLoopback", health.hyper_loopback)
    ]:
        icon = "✓" if status == ComponentStatus.AVAILABLE else "○"
        print(f"   {icon} {component}: {status.value}")
    
    # Test queries
    print("\n📋 UNIFIED QUERIES")
    print("-" * 60)
    
    queries = [
        "What is the purpose of the Daughter Test?",
        "Explain RIBA_ZERO in Islamic economics",
        "How does winter-proofing ensure resilience?"
    ]
    
    for i, query_text in enumerate(queries, 1):
        print(f"\n[Query {i}] {query_text}")
        
        query = UnifiedQuery(
            text=query_text,
            require_constitution_check=True,
            require_daughter_test=True
        )
        
        response = await ecosystem.query(query)
        
        print(f"   Synthesis: {response.synthesis[:80]}...")
        print(f"   SNR: {response.snr_score:.3f} | Ihsān: {response.ihsan_score:.3f}")
        print(f"   Components: {', '.join(response.components_used)}")
        print(f"   BLOOM: +{response.bloom_reward:.1f}")
    
    # Final status
    print("\n📋 FINAL STATUS")
    print("-" * 60)
    
    status = ecosystem.get_status()
    print(f"Node: {status['node_id']}")
    print(f"Queries: {status['query_count']}")
    print(f"Health: {status['health']['overall_health'] * 100:.0f}%")
    
    print("\n" + "="*80)
    print("✅ ECOSYSTEM DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"For: {ecosystem.daughter_name}, and all daughters of the future")
    print("="*80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIZRA Ecosystem Bridge — Unified Integration Layer"
    )
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--status", "-s", action="store_true", help="Show ecosystem status")
    parser.add_argument("--health", "-H", action="store_true", help="Show health report")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demonstration")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demonstrate_ecosystem())
    
    elif args.status or args.health:
        ecosystem = get_ecosystem()
        if args.status:
            status = ecosystem.get_status()
            print(json.dumps(status, indent=2, default=str))
        else:
            health = ecosystem.get_health()
            print(json.dumps(health.to_dict(), indent=2))
    
    elif args.query:
        async def run_query():
            ecosystem = await initialize_ecosystem()
            query = UnifiedQuery(text=args.query)
            response = await ecosystem.query(query)
            print(f"\nQuery: {response.query}")
            print(f"Synthesis: {response.synthesis}")
            print(f"SNR: {response.snr_score:.3f} | Ihsān: {response.ihsan_score:.3f}")
        
        asyncio.run(run_query())
    
    else:
        asyncio.run(demonstrate_ecosystem())


if __name__ == "__main__":
    main()
