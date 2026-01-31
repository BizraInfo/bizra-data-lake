"""
BIZRA Core Package — The Stem Cell Kernel

Sovereignty Infrastructure (Thermodynamic Framing):
- pci: Proof-Carrying Inference Protocol (Genome + Epigenome)
- vault: Encryption at Rest (Membrane)
- federation: P2P Network (Mycorrhizal Connections)
- inference: Embedded LLM Gateway (Metabolism)
- a2a: Agent-to-Agent Protocol (Orchestration)
- integration: Unified Bridge (Cohesion)

"Entropy reduction as a service."

Created: 2026-01-27
Updated: 2026-01-30 — Added integration bridge for module cohesion
"""

from . import pci
from . import vault
from . import federation
from . import inference
from . import a2a
from . import integration

__all__ = ["pci", "vault", "federation", "inference", "a2a", "integration"]
__version__ = "2.2.0"
