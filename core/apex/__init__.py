"""
BIZRA Apex System — Peak Sovereign Intelligence
═══════════════════════════════════════════════════════════════════════════════

The Apex layer represents the pinnacle of BIZRA's autonomous capabilities,
filling the final 8% gap to achieve 100% completeness of the Proactive
Sovereign Entity architecture.

Three Pillars:
1. SocialGraph — Relationship Intelligence (Granovetter, Dunbar, PageRank)
2. OpportunityEngine — Active Market Intelligence (Shannon, Markowitz, Lo)
3. SwarmOrchestrator — Autonomous Scaling (Lamport, Borg, Kubernetes)

Integration Points:
- Connects to existing bizra-omega Rust kernel via rust_lifecycle.py
- Uses A2A protocol for agent-to-agent communication
- Feeds ProactiveSovereignEntity with social/market/scaling signals

Standing on the Shoulders of Giants:
- Shannon (1948): Information theory, SNR optimization
- Lamport (1982): Byzantine fault tolerance, distributed consensus
- Granovetter (1973): Weak ties, social network dynamics
- Markowitz (1952): Portfolio theory, risk-return optimization
- Page & Brin (1998): PageRank trust propagation
- Verma et al. (2015): Borg cluster management
- Burns et al. (2016): Kubernetes design principles
- Besta (2024): Graph-of-Thoughts reasoning

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        BIZRA APEX SYSTEM                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
    │  │  SocialGraph    │  │ OpportunityEngine│  │  SwarmOrchestrator     │ │
    │  │  ─────────────  │  │  ───────────────  │  │  ──────────────────   │ │
    │  │  Relationships  │  │  Market Analyzer │  │  Deployment Manager   │ │
    │  │  Trust (PageRank)│  │  Signal Generator│  │  Health Monitor       │ │
    │  │  Collaboration  │  │  Arbitrage       │  │  Scaling Manager      │ │
    │  │  Negotiation    │  │  Position Mgmt   │  │  Self-Healing Loop    │ │
    │  └────────┬────────┘  └────────┬────────┘  └───────────┬───────────┘ │
    │           │                    │                       │             │
    │           └────────────────────┼───────────────────────┘             │
    │                                ▼                                     │
    │              ┌──────────────────────────────────┐                    │
    │              │  ProactiveSovereignEntity        │                    │
    │              │  (Extended OODA + Muraqabah)     │                    │
    │              └──────────────────────────────────┘                    │
    │                                │                                     │
    │                                ▼                                     │
    │              ┌──────────────────────────────────┐                    │
    │              │  bizra-omega (Rust Kernel)       │                    │
    │              │  IhsanVector • GoT • SNR • PBFT  │                    │
    │              └──────────────────────────────────┘                    │
    │                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Apex System v1.0
"""

from __future__ import annotations

# Version
__version__ = "1.0.0"
__author__ = "BIZRA Node0"

# =============================================================================
# Opportunity Engine — Active Market Intelligence
# =============================================================================
from core.apex.opportunity_engine import (  # Enums; Data classes; Component classes; Main class
    ArbitrageDetector,
    ArbitrageOpportunity,
    MarketAnalysis,
    MarketAnalyzer,
    MarketCondition,
    MarketData,
    OpportunityEngine,
    Position,
    PositionStatus,
    SignalGenerator,
    SignalStrength,
    SignalType,
    TradingSignal,
)

# =============================================================================
# Social Graph — Relationship Intelligence
# =============================================================================
from core.apex.social_graph import (  # Enums; Data classes; Main class
    CollaborationOpportunity,
    CollaborationStatus,
    Interaction,
    InteractionType,
    NegotiationOffer,
    Relationship,
    RelationshipType,
    SocialGraph,
)

# =============================================================================
# Swarm Orchestrator — Autonomous Deployment & Scaling
# =============================================================================
from core.apex.swarm_orchestrator import (  # Enums; Data classes; Component classes; Main class
    AgentConfig,
    AgentInstance,
    AgentStatus,
    HealthMonitor,
    HealthReport,
    HealthStatus,
    ScalingAction,
    ScalingDecision,
    ScalingManager,
    Swarm,
    SwarmConfig,
    SwarmOrchestrator,
    SwarmTopology,
)

# =============================================================================
# SNR Apex Engine — Autonomous Signal Optimization (v2.0)
# =============================================================================
from core.apex.snr_apex_engine import (
    APEX_SNR_FLOOR,
    APEX_SNR_TARGET,
    ApexReasoningEngine,
    CognitiveGenerator,
    CognitiveLayer,
    DisciplineSynthesis,
    Giant,
    GiantsRegistry,
    GraphOfThoughts,
    NoiseComponent,
    SignalComponent,
    SNRAnalysis,
    SNRApexEngine,
    ThoughtNode,
    ThoughtStatus,
    ThoughtType,
)

# =============================================================================
# Convenience Exports
# =============================================================================
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Social Graph
    "RelationshipType",
    "InteractionType",
    "CollaborationStatus",
    "Interaction",
    "Relationship",
    "CollaborationOpportunity",
    "NegotiationOffer",
    "SocialGraph",
    # Opportunity Engine
    "MarketCondition",
    "SignalType",
    "SignalStrength",
    "PositionStatus",
    "MarketData",
    "MarketAnalysis",
    "TradingSignal",
    "ArbitrageOpportunity",
    "Position",
    "MarketAnalyzer",
    "SignalGenerator",
    "ArbitrageDetector",
    "OpportunityEngine",
    # Swarm Orchestrator
    "AgentStatus",
    "ScalingAction",
    "HealthStatus",
    "SwarmTopology",
    "AgentConfig",
    "AgentInstance",
    "SwarmConfig",
    "Swarm",
    "ScalingDecision",
    "HealthReport",
    "HealthMonitor",
    "ScalingManager",
    "SwarmOrchestrator",
    # SNR Apex Engine (v2.0)
    "SNRApexEngine",
    "ApexReasoningEngine",
    "GraphOfThoughts",
    "ThoughtNode",
    "ThoughtType",
    "ThoughtStatus",
    "SNRAnalysis",
    "GiantsRegistry",
    "Giant",
    "SignalComponent",
    "NoiseComponent",
    "CognitiveGenerator",
    "CognitiveLayer",
    "DisciplineSynthesis",
    "APEX_SNR_TARGET",
    "APEX_SNR_FLOOR",
]


# =============================================================================
# Unified Apex Interface
# =============================================================================
class ApexSystem:
    """
    Unified interface to all Apex subsystems.

    Provides a single entry point for:
    - Social intelligence (relationships, collaboration, trust)
    - Market intelligence (signals, arbitrage, positions)
    - Swarm management (deployment, scaling, health)

    Example:
        apex = ApexSystem(node_id="node-0")
        await apex.start()

        # Social
        apex.social.add_relationship("agent-1", "agent-2", RelationshipType.COLLABORATOR)
        trust = apex.social.calculate_trust("agent-1")

        # Market
        signals = await apex.opportunity.scan_markets()
        for signal in signals:
            if signal.snr >= 0.85:
                await apex.opportunity.execute_signal(signal)

        # Swarm
        await apex.swarm.deploy_agent(config)
        await apex.swarm.scale(scaling_decision)
    """

    def __init__(
        self,
        node_id: str,
        ihsan_threshold: float = 0.95,
        snr_floor: float = 0.85,
    ):
        """
        Initialize the Apex system.

        Args:
            node_id: Unique identifier for this node
            ihsan_threshold: Constitutional constraint (default 0.95)
            snr_floor: Minimum signal quality (default 0.85)
        """
        self.node_id = node_id
        self.ihsan_threshold = ihsan_threshold
        self.snr_floor = snr_floor

        # Initialize subsystems (lazy loading)
        self._social: SocialGraph | None = None
        self._opportunity: OpportunityEngine | None = None
        self._swarm: SwarmOrchestrator | None = None

        self._running = False

    @property
    def social(self) -> SocialGraph:
        """Get or create the SocialGraph instance."""
        if self._social is None:
            # SocialGraph takes agent_id, not node_id
            self._social = SocialGraph(agent_id=self.node_id)
        return self._social

    @property
    def opportunity(self) -> OpportunityEngine:
        """Get or create the OpportunityEngine instance."""
        if self._opportunity is None:
            # OpportunityEngine takes snr_threshold
            self._opportunity = OpportunityEngine(snr_threshold=self.snr_floor)
        return self._opportunity

    @property
    def swarm(self) -> SwarmOrchestrator:
        """Get or create the SwarmOrchestrator instance."""
        if self._swarm is None:
            # SwarmOrchestrator takes no arguments
            self._swarm = SwarmOrchestrator()
        return self._swarm

    async def start(self) -> None:
        """Start all Apex subsystems."""
        if self._running:
            return

        # Start in parallel
        import asyncio

        await asyncio.gather(
            self.social.start() if hasattr(self.social, "start") else asyncio.sleep(0),
            (
                self.opportunity.start()
                if hasattr(self.opportunity, "start")
                else asyncio.sleep(0)
            ),
            self.swarm.start() if hasattr(self.swarm, "start") else asyncio.sleep(0),
        )

        self._running = True

    async def stop(self) -> None:
        """Stop all Apex subsystems gracefully."""
        if not self._running:
            return

        import asyncio

        await asyncio.gather(
            (
                self.social.stop()  # type: ignore[attr-defined]
                if self._social and hasattr(self._social, "stop")
                else asyncio.sleep(0)
            ),
            (
                self.opportunity.stop()  # type: ignore[attr-defined]
                if self._opportunity and hasattr(self._opportunity, "stop")
                else asyncio.sleep(0)
            ),
            (
                self.swarm.stop()
                if self._swarm and hasattr(self._swarm, "stop")
                else asyncio.sleep(0)
            ),
        )

        self._running = False

    def status(self) -> dict:
        """Get status of all subsystems."""
        return {
            "node_id": self.node_id,
            "running": self._running,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_floor": self.snr_floor,
            "subsystems": {
                "social": "active" if self._social else "inactive",
                "opportunity": "active" if self._opportunity else "inactive",
                "swarm": "active" if self._swarm else "inactive",
            },
        }


# Add to exports
__all__.append("ApexSystem")
