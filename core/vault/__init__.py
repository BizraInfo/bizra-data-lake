"""
BIZRA Sovereign Vault — Encryption at Rest

Sovereignty Pillar: DATA SOVEREIGNTY
"""

from .vault import (
    VAULT_VERSION,
    SovereignVault,
    VaultEntry,
    derive_key,
    generate_salt,
    get_vault,
)

__all__ = [
    "SovereignVault",
    "VaultEntry",
    "get_vault",
    "derive_key",
    "generate_salt",
    "VAULT_VERSION",
]
