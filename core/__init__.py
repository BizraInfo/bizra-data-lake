"""
BIZRA Core Package — The Stem Cell Kernel

Sovereignty Infrastructure (Thermodynamic Framing):
- pci: Proof-Carrying Inference Protocol (Genome + Epigenome)
- vault: Encryption at Rest (Membrane)
- federation: P2P Network (Mycorrhizal Connections)
- inference: Embedded LLM Gateway (Metabolism)

"Entropy reduction as a service."

Created: 2026-01-27
Updated: 2026-01-29 — Added inference gateway + epigenome
"""

from . import pci
from . import vault
from . import federation
from . import inference

__all__ = ["pci", "vault", "federation", "inference"]
__version__ = "2.1.0"
