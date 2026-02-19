"""
BIZRA Core Package — The Stem Cell Kernel

Sovereignty Infrastructure (Thermodynamic Framing):
- pci: Proof-Carrying Inference Protocol (Genome + Epigenome)
- vault: Encryption at Rest (Membrane)
- federation: P2P Network (Mycorrhizal Connections)
- inference: Embedded LLM Gateway (Metabolism)
- a2a: Agent-to-Agent Protocol (Orchestration)
- integration: Unified Bridge (Cohesion)
- ntu: NeuroTemporal Unit (Pattern Detection)
- protocols: Interface Contracts (Structural Typing)

Decomposed from sovereign module (SAPE v2.3.1):
- governance: Constitutional Gates & Autonomy
- reasoning: Graph-of-Thoughts & Quality Validation
- orchestration: Event Bus & Agent Coordination
- treasury: Resource Management & Justice Enforcement
- bridges: Cross-System Integration

"Entropy reduction as a service."

Created: 2026-01-27
Updated: 2026-01-30 — Added integration bridge for module cohesion
Updated: 2026-02-03 — Added NTU (NeuroTemporal Unit) for pattern detection
Updated: 2026-02-05 — Added protocols + decomposed sovereign (SAPE Elite Analysis)
Updated: 2026-02-17 — Phase 31: HyperGraph + Cognitive Fusion + Memory Coder
Updated: 2026-02-17 — Phase 25-28: Genesis + Guild + Quest + HRM + NorthStar + Memory
"""

# Decomposed sovereign sub-packages (backwards compatible)
from . import (
    a2a,
    bridges,
    cognitive_fusion,
    federation,
    genesis,
    governance,
    guild,
    hashtable,
    hrm,
    hypergraph,
    inference,
    integration,
    memory,
    memory_coder,
    northstar,
    ntu,
    orchestration,
    pci,
    prediction,
    protocols,
    quest,
    reasoning,
    search,
    treasury,
    vault,
)

__all__ = [
    # Core infrastructure
    "pci",
    "vault",
    "federation",
    "inference",
    "a2a",
    "integration",
    "ntu",
    # Protocols
    "protocols",
    # Decomposed sovereign (new structure)
    "governance",
    "reasoning",
    "orchestration",
    "treasury",
    "bridges",
    # Phase 44: Hash Table Infrastructure
    "hashtable",
    # Phase 46: Cognitive Resonance
    "search",
    "prediction",
    # Phase 31: Cognitive Fusion
    "hypergraph",
    "cognitive_fusion",
    "memory_coder",
    # Phase 25-28: Ecosystem subsystems
    "genesis",
    "guild",
    "quest",
    "hrm",
    "northstar",
    "memory",
]
__version__ = "2.5.0"
