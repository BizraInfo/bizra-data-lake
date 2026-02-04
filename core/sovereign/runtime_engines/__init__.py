"""
BIZRA Sovereign Runtime — Peak Masterpiece
═══════════════════════════════════════════════════════════════════════════════

"We are like dwarfs sitting on the shoulders of giants. We see more,
and things that are more distant, than they did, not because our sight
is superior or because we are taller than they, but because they raise
us up, and by their great stature add to ours."
    — Bernard of Chartres (c. 1159)

This runtime package provides the foundational engines for BIZRA's
sovereign intelligence:

1. GIANTS REGISTRY — Knowledge attribution protocol
2. SNR MAXIMIZER — Shannon-inspired signal optimization
3. GoT BRIDGE — Graph-of-Thoughts multi-path reasoning

Together, these components form the cognitive infrastructure that
enables proactive, constitutional, and explainable AI decision-making.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SOVEREIGN RUNTIME                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
    │  │  Giants         │  │  SNR            │  │  Graph-of-Thoughts      │ │
    │  │  Registry       │  │  Maximizer      │  │  Bridge                 │ │
    │  │  (Attribution)  │  │  (Filtering)    │  │  (Reasoning)            │ │
    │  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘ │
    │           │                    │                       │               │
    │           └────────────────────┼───────────────────────┘               │
    │                                ▼                                       │
    │              ┌──────────────────────────────────────────┐              │
    │              │           SOVEREIGN RUNTIME              │              │
    │              │   (Unified Cognitive Infrastructure)     │              │
    │              └──────────────────────────────────────────┘              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Standing on Giants:
- Shannon (1948): Information theory
- Besta (2024): Graph-of-Thoughts
- Newton (1675): Attribution philosophy
- Lamport (1982): Distributed consensus
- Al-Ghazali (1095): Muraqabah vigilance
- Anthropic (2022): Constitutional AI

Created: 2026-02-04 | BIZRA Sovereign Runtime v1.0
"""

from __future__ import annotations

# Giants Registry — Knowledge Attribution Protocol
from .giants_registry import (
    Giant,
    GiantApplication,
    GiantCategory,
    GiantsRegistry,
    get_giants_registry,
    attribute,
)

# SNR Maximizer — Shannon-Inspired Signal Optimization
from .snr_maximizer import (
    Signal,
    SignalQuality,
    NoiseType,
    ChannelMetrics,
    NoiseEstimator,
    SignalExtractor,
    SNRCalculator,
    SNRFilter,
    SNRMaximizer,
    get_snr_maximizer,
    snr_filter,
    SNR_FLOOR,
    SNR_EXCELLENT,
    SNR_CHANNEL_CAPACITY,
)

# Graph-of-Thoughts Bridge — Multi-Path Reasoning
from .got_bridge import (
    ThoughtNode,
    ThoughtEdge,
    ThoughtType,
    ThoughtStatus,
    ThoughtGraph,
    GoTResult,
    GoTBridge,
    get_got_bridge,
    think,
    MAX_DEPTH,
    MAX_BRANCHES,
    PRUNE_THRESHOLD,
)

__all__ = [
    # Giants Registry
    "Giant",
    "GiantApplication",
    "GiantCategory",
    "GiantsRegistry",
    "get_giants_registry",
    "attribute",
    # SNR Maximizer
    "Signal",
    "SignalQuality",
    "NoiseType",
    "ChannelMetrics",
    "NoiseEstimator",
    "SignalExtractor",
    "SNRCalculator",
    "SNRFilter",
    "SNRMaximizer",
    "get_snr_maximizer",
    "snr_filter",
    "SNR_FLOOR",
    "SNR_EXCELLENT",
    "SNR_CHANNEL_CAPACITY",
    # Graph-of-Thoughts
    "ThoughtNode",
    "ThoughtEdge",
    "ThoughtType",
    "ThoughtStatus",
    "ThoughtGraph",
    "GoTResult",
    "GoTBridge",
    "get_got_bridge",
    "think",
    "MAX_DEPTH",
    "MAX_BRANCHES",
    "PRUNE_THRESHOLD",
]

__version__ = "1.0.0"
__author__ = "BIZRA Node0"
