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
"""

from . import pci
from . import vault
from . import federation
from . import inference
from . import a2a
from . import integration
from . import ntu
from . import protocols

# Decomposed sovereign sub-packages (backwards compatible)
from . import governance
from . import reasoning
from . import orchestration
from . import treasury
from . import bridges

__all__ = [
    # Core infrastructure
    "pci", "vault", "federation", "inference", "a2a", "integration", "ntu",
    # Protocols
    "protocols",
    # Decomposed sovereign (new structure)
    "governance", "reasoning", "orchestration", "treasury", "bridges",
]
__version__ = "2.3.1"
