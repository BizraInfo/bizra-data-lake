"""
BIZRA Identity Package — Layer 0 (Ontological)

Formalizes Definition 1.6 (Identity Genesis) and Definition 1.7 (Node Body).

Standing on Giants: Bernstein (Ed25519, 2011) | Merkle (key derivation) | Phase 61
"""

from .genesis import (
    IdentityGenesis,
    NodeBody,
    SovereigntyClass,
    derive_agent_keypairs,
    derive_identity_id,
)

__all__ = [
    "IdentityGenesis",
    "NodeBody",
    "SovereigntyClass",
    "derive_identity_id",
    "derive_agent_keypairs",
]

__version__ = "1.0.0"
