"""
BIZRA Sovereign Nexus - Unified Control Interface for BIZRA Sovereign Intelligence

This module provides the unified control interface that consolidates 11 existing components
into a unified apex orchestrator with:
- 47-Discipline Topology Engine
- Autonomous Dreaming capability  
- SNR self-healing optimization (0.95 Ihsān threshold)

Exports:
    - SovereignNexus: The main orchestrator class
    - DisciplineTopologyEngine: 47-discipline topology engine
    - AutonomousDreamer: Proactive hypothesis generation
"""

from .nexus import SovereignNexus
from .topology_engine import DisciplineTopologyEngine
from .dreamer import AutonomousDreamer

__all__ = [
    'SovereignNexus',
    'DisciplineTopologyEngine', 
    'AutonomousDreamer'
]