"""
BIZRA Sovereignty Module - HYPER LOOPBACK Architecture

This module implements complete offline sovereignty for BIZRA nodes:
- WinterProofEmbedder: Deterministic embeddings without external APIs
- Constitution: Compliance verification with Ihsān thresholds
- DaughterTest: Continuous integrity verification
- LocalMerkleDAG: Tamper-proof evidence chains

Domain: bizra-pci-v1:
Threshold: 0.95 Ihsān minimum
Hash: SHA-256 and BLAKE3 for integrity

NO external API dependencies - pure HYPER LOOPBACK operation.
"""

from .winter_proof import WinterProofEmbedder
from .constitution import Constitution
from .daughter_test import DaughterTest
from .merkle_dag import LocalMerkleDAG

__all__ = [
    "WinterProofEmbedder",
    "Constitution",
    "DaughterTest",
    "LocalMerkleDAG",
]

__version__ = "1.0.0"
__domain__ = "bizra-pci-v1:"
