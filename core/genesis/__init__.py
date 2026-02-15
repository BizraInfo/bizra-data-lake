"""
BIZRA Genesis Module — One-Command Node Bootstrap
===================================================
Orchestrates full sovereign node genesis: identity minting,
hardware fingerprinting, PAT-7/SAT-5 activation, token allocation,
guild membership, quest acceptance, and Ihsan targeting.

Standing on Giants:
- Al-Ghazali (1095): Spiritual genesis as ethical covenant
- Shannon (1948): Information-theoretic identity verification
- Wiener (1948): Cybernetic bootstrap — from zero to sovereign
- Deming (1950): Quality-gated step-by-step activation

v1.0.0
"""

from __future__ import annotations

from .orchestrator import GenesisOrchestrator
from .types import GenesisConfig, GenesisResult, GenesisStep

__version__ = "1.0.0"

__all__ = [
    "GenesisConfig",
    "GenesisOrchestrator",
    "GenesisResult",
    "GenesisStep",
]
