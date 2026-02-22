"""
BIZRA Genesis Module — One-Command Node Bootstrap
====================================================

The Genesis module orchestrates the entire BIZRA node bootstrap
pipeline: identity minting, hardware scanning, PAT/SAT activation,
token allocation, URP pledge, guild joining, quest acceptance,
and Ihsan targeting — all in a single command.

v1.0.0 — Dream CLI Genesis

Architecture:
    GenesisConfig (CLI flags)
        |
        v
    GenesisOrchestrator.run()
        |
        +-- Identity Genesis (core.pat.minting)
        +-- Hardware Scan (core.genesis.hardware)
        +-- PAT-7 Activation (core.pat.agent)
        +-- SAT-5 Activation (core.pat.agent)
        +-- Token Allocation (core.token.mint)
        +-- URP Pledge (core.genesis.urp)
        +-- HDA Bridge (core.bridges)
        +-- Mobile Pairing (core.genesis.mobile_pairing)
        +-- Guild Join (core.guild)
        +-- Quest Accept (core.quest)
        +-- Ihsan Target (constitutional)
        |
        v
    GenesisResult (auditable receipt)

Standing on Giants:
- Nakamoto (2008): Genesis block as network origin
- Shannon (1948): SNR as quality signal
- Al-Ghazali (1058-1111): Ihsan as ethical constraint
"""

from core.genesis.hardware import HardwareInfo, HardwareScanner
from core.genesis.mobile_pairing import MobilePairResult, pair_mobile
from core.genesis.orchestrator import GenesisOrchestrator
from core.genesis.state_persistence import (
    SovereignState,
    load_sovereign_state,
    save_sovereign_state,
    state_exists,
)
from core.genesis.types import (
    CHECKMARK,
    CROSSMARK,
    OMEGA,
    GenesisConfig,
    GenesisResult,
    GenesisStep,
    GenesisStepStatus,
)
from core.genesis.urp import URPPledge, pledge_resources

__version__ = "1.0.0"

__all__ = [
    # Types
    "GenesisConfig",
    "GenesisResult",
    "GenesisStep",
    "GenesisStepStatus",
    "HardwareInfo",
    "URPPledge",
    "MobilePairResult",
    # Engine
    "GenesisOrchestrator",
    "HardwareScanner",
    # State persistence
    "SovereignState",
    "save_sovereign_state",
    "load_sovereign_state",
    "state_exists",
    # Functions
    "pledge_resources",
    "pair_mobile",
    # Constants
    "CHECKMARK",
    "CROSSMARK",
    "OMEGA",
]
