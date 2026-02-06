"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗   ██╗███╗   ██╗████████╗██╗███╗   ███╗███████╗                ║
║   ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██║████╗ ████║██╔════╝                ║
║   ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║██╔████╔██║█████╗                  ║
║   ██╔══██╗██║   ██║██║╚██╗██║   ██║   ██║██║╚██╔╝██║██╔══╝                  ║
║   ██║  ██║╚██████╔╝██║ ╚████║   ██║   ██║██║ ╚═╝ ██║███████╗                ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝                ║
║                                                                              ║
║                    SOVEREIGN UNIFIED RUNTIME v1.0                            ║
║         The Apex Integration — All Components, One Interface                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   "The whole is greater than the sum of its parts." — Aristotle              ║
║                                                                              ║
║   This runtime unifies:                                                      ║
║   • SovereignEngine (Core Reasoning)                                         ║
║   • GraphOfThoughts (Multi-Path Exploration)                                 ║
║   • SNRMaximizer (Signal Quality Enforcement)                                ║
║   • GuardianCouncil (Byzantine Validation)                                   ║
║   • AutonomousLoop (OODA Cycle)                                              ║
║   • SovereignOrchestrator (Task Decomposition)                               ║
║                                                                              ║
║   Module Structure (SPARC refinement):                                       ║
║   • runtime_types.py  — Type definitions, protocols, configs                 ║
║   • runtime_stubs.py  — Fallback stub implementations                        ║
║   • runtime_core.py   — Main SovereignRuntime class                          ║
║   • runtime_cli.py    — Command-line interface                               ║
║   • runtime.py        — Public facade (this file)                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# PUBLIC API - Re-export from modular components
# =============================================================================

# CLI interface
from .runtime_cli import cli_main

# Core runtime class
from .runtime_core import SovereignRuntime

# Stub implementations for graceful degradation
from .runtime_stubs import (
    AutonomousLoopStub,
    ComponentStub,
    GraphReasonerStub,
    GuardianStub,
    SNROptimizerStub,
    StubFactory,
)

# Type definitions and protocols
from .runtime_types import (  # TypedDicts; Protocols; Enums; Config; Query/Result
    AutonomousCycleResult,
    AutonomousLoopProtocol,
    GraphReasonerProtocol,
    GuardianProtocol,
    HealthStatus,
    LoopStatus,
    ReasoningResult,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeMode,
    SNROptimizerProtocol,
    SNRResult,
    SovereignQuery,
    SovereignResult,
    ValidationResult,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "SovereignRuntime",
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeMetrics",
    # Query/Response
    "SovereignQuery",
    "SovereignResult",
    # Type definitions
    "ReasoningResult",
    "SNRResult",
    "ValidationResult",
    "AutonomousCycleResult",
    "LoopStatus",
    # Protocols
    "GraphReasonerProtocol",
    "SNROptimizerProtocol",
    "GuardianProtocol",
    "AutonomousLoopProtocol",
    # Health
    "HealthStatus",
    # Stubs
    "ComponentStub",
    "GraphReasonerStub",
    "SNROptimizerStub",
    "GuardianStub",
    "AutonomousLoopStub",
    "StubFactory",
    # CLI
    "cli_main",
]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    asyncio.run(cli_main())
