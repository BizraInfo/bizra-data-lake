"""
BIZRA Sovereign Vault — Encryption at Rest

Sovereignty Pillar: DATA SOVEREIGNTY
"""

from .vault import (
    SovereignVault,
    VaultEntry,
    get_vault,
    derive_key,
    generate_salt,
    VAULT_VERSION,
)

__all__ = [
    "SovereignVault",
    "VaultEntry", 
    "get_vault",
    "derive_key",
    "generate_salt",
    "VAULT_VERSION",
]
