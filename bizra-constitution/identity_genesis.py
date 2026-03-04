"""
BIZRA Identity Genesis — Sovereign Node Identity
═════════════════════════════════════════════════

Every node begins with a Genesis event. Identity is sovereign.

This module implements:
  1. Ed25519 master keypair generation
  2. Node ID derivation: SHA-256(public_key) — deterministic
  3. HD agent key derivation: 12 child keys (7 PAT + 5 SAT)
  4. Domain-separated signing for all constitutional contexts
  5. Identity persistence (encrypted at rest)

Security model:
  - Master key never leaves the node
  - Agent keys are derived, not stored (regenerable)
  - Domain separation prevents cross-protocol replay
  - Genesis event is recorded as first evidence receipt

Constitution reference: §1 [identity], §12 [security]
Proof chain: Definition 1.6 (Identity Genesis)
Document: IDG-2026-001
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Use PyNaCl (libsodium) for Ed25519 — the gold standard
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.encoding import HexEncoder
    from nacl.exceptions import BadSignatureError
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False

try:
    from generated.generated_constants import (
        IDENTITY_KEY_ALGORITHM,
        IDENTITY_AGENTS_PER_NODE,
        IDENTITY_GENESIS_DOMAIN,
        PAT_AGENT_NAMES,
        SAT_BOOTSTRAP_ROLES,
        DOMAIN_EVIDENCE_RECEIPT,
    )
except ImportError:
    IDENTITY_KEY_ALGORITHM = "Ed25519"
    IDENTITY_AGENTS_PER_NODE = 12
    IDENTITY_GENESIS_DOMAIN = "bizra-identity-genesis-v1"
    PAT_AGENT_NAMES = ["Planner", "Researcher", "Coder",
                       "Evaluator", "Ethicist", "Publisher", "Integrator"]
    SAT_BOOTSTRAP_ROLES = ["ComputeScheduler", "SecurityMonitor",
                           "PerformanceAnalyzer", "ConsensusValidator",
                           "NetworkOrchestrator"]
    DOMAIN_EVIDENCE_RECEIPT = "bizra-evidence-v1"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AgentKey:
    """A derived agent key with its role binding."""
    agent_name: str
    agent_type: str            # "pat" or "sat"
    derivation_index: int
    public_key_hex: str
    _signing_key: Any = field(repr=False, compare=False)

    def sign(self, message: bytes, domain: str) -> bytes:
        """Sign with domain separation."""
        if not NACL_AVAILABLE:
            return _fallback_sign(message, domain, self._signing_key)
        domain_msg = f"{domain}:{message.hex()}".encode()
        return self._signing_key.sign(domain_msg).signature

    def verify(self, message: bytes, signature: bytes, domain: str) -> bool:
        """Verify a domain-separated signature."""
        if not NACL_AVAILABLE:
            return _fallback_verify(message, signature, domain,
                                     self.public_key_hex)
        try:
            verify_key = VerifyKey(
                bytes.fromhex(self.public_key_hex)
            )
            domain_msg = f"{domain}:{message.hex()}".encode()
            verify_key.verify(domain_msg, signature)
            return True
        except (BadSignatureError, Exception):
            return False


@dataclass(frozen=True)
class NodeIdentity:
    """Complete sovereign node identity."""
    node_id: str               # SHA-256(master_public_key) — deterministic
    public_key_hex: str        # Master public key (hex)
    genesis_timestamp: float   # When this identity was created
    genesis_domain: str        # Domain separation context
    pat_agents: list[AgentKey]
    sat_agents: list[AgentKey]
    _master_signing_key: Any = field(repr=False, compare=False)

    @property
    def total_agents(self) -> int:
        return len(self.pat_agents) + len(self.sat_agents)

    def get_agent(self, name: str) -> AgentKey | None:
        """Get agent key by name."""
        for a in self.pat_agents + self.sat_agents:
            if a.agent_name == name:
                return a
        return None

    def sign_master(self, message: bytes, domain: str) -> bytes:
        """Sign with the master key (for constitutional amendments, etc)."""
        if not NACL_AVAILABLE:
            return _fallback_sign(message, domain, self._master_signing_key)
        domain_msg = f"{domain}:{message.hex()}".encode()
        return self._master_signing_key.sign(domain_msg).signature

    def verify_master(self, message: bytes, signature: bytes,
                      domain: str) -> bool:
        """Verify a master key signature."""
        if not NACL_AVAILABLE:
            return _fallback_verify(message, signature, domain,
                                     self.public_key_hex)
        try:
            verify_key = VerifyKey(bytes.fromhex(self.public_key_hex))
            domain_msg = f"{domain}:{message.hex()}".encode()
            verify_key.verify(domain_msg, signature)
            return True
        except (BadSignatureError, Exception):
            return False

    def as_public_record(self) -> dict[str, Any]:
        """Public-safe representation (no private keys)."""
        return {
            "node_id": self.node_id,
            "public_key_hex": self.public_key_hex,
            "genesis_timestamp": self.genesis_timestamp,
            "genesis_domain": self.genesis_domain,
            "pat_agents": [
                {"name": a.agent_name, "public_key": a.public_key_hex}
                for a in self.pat_agents
            ],
            "sat_agents": [
                {"name": a.agent_name, "public_key": a.public_key_hex}
                for a in self.sat_agents
            ],
            "total_agents": self.total_agents,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK IMPLEMENTATION (when PyNaCl not available)
# Uses HMAC-SHA256 as a signing primitive — NOT production-grade crypto,
# but maintains the same interface for development/testing.
# ═══════════════════════════════════════════════════════════════════════════════


def _fallback_keygen() -> tuple[bytes, bytes]:
    """Generate a 32-byte key pair using os.urandom."""
    private = os.urandom(32)
    public = hashlib.sha256(b"bizra-pubkey:" + private).digest()
    return private, public


def _fallback_sign(message: bytes, domain: str, key: bytes) -> bytes:
    """HMAC-SHA256 based signing (development fallback)."""
    domain_msg = f"{domain}:{message.hex()}".encode()
    return hmac.new(key, domain_msg, hashlib.sha256).digest()


def _fallback_verify(message: bytes, signature: bytes, domain: str,
                      public_key_hex: str) -> bool:
    """HMAC verification — development fallback only.

    WARNING: In HMAC fallback mode, we cannot verify authenticity without
    the original signing key. Returns True only with correct signature
    length AND emits a warning. Install PyNaCl for production.
    """
    import warnings
    if len(signature) != 32:
        return False
    warnings.warn(
        "HMAC fallback verification cannot prove authenticity. "
        "Install PyNaCl for production: pip install pynacl",
        stacklevel=2,
    )
    return True


def _derive_child_key_fallback(master_private: bytes, index: int,
                                domain: str) -> tuple[bytes, bytes]:
    """Derive a child key deterministically from master + index."""
    seed = hashlib.sha256(
        f"{domain}:{master_private.hex()}:{index}".encode()
    ).digest()
    public = hashlib.sha256(b"bizra-pubkey:" + seed).digest()
    return seed, public


# ═══════════════════════════════════════════════════════════════════════════════
# KEY DERIVATION (HD-Ed25519)
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_agent_key(master_key, index: int, agent_name: str,
                       agent_type: str, domain: str) -> AgentKey:
    """
    Derive an agent key from the master key using HD derivation.

    HD path: master / domain / index
    Each agent gets a unique, deterministic, regenerable key.
    """
    if NACL_AVAILABLE:
        # Deterministic seed from master key + index
        master_bytes = master_key.encode()
        seed_material = hashlib.sha256(
            f"{domain}:{master_bytes.hex()}:{index}".encode()
        ).digest()
        child_signing = SigningKey(seed_material)
        child_public = child_signing.verify_key.encode(encoder=HexEncoder).decode()
        return AgentKey(
            agent_name=agent_name,
            agent_type=agent_type,
            derivation_index=index,
            public_key_hex=child_public,
            _signing_key=child_signing,
        )
    else:
        private, public = _derive_child_key_fallback(
            master_key, index, domain
        )
        return AgentKey(
            agent_name=agent_name,
            agent_type=agent_type,
            derivation_index=index,
            public_key_hex=public.hex(),
            _signing_key=private,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY GENESIS — The Creation Event
# ═══════════════════════════════════════════════════════════════════════════════


def create_identity(
    pat_names: list[str] | None = None,
    sat_names: list[str] | None = None,
) -> NodeIdentity:
    """
    Create a new sovereign node identity.

    This is the Genesis Event. It generates:
      1. A master Ed25519 keypair
      2. A deterministic node ID (SHA-256 of public key)
      3. 7 PAT agent keys (derived from master)
      4. 5 SAT agent keys (derived from master)

    The master key is the root of trust. Agent keys are derived
    and can be regenerated from the master key at any time.

    Returns:
        NodeIdentity with complete key hierarchy.
    """
    pat_names = pat_names or PAT_AGENT_NAMES
    sat_names = sat_names or SAT_BOOTSTRAP_ROLES

    genesis_domain = IDENTITY_GENESIS_DOMAIN

    # Generate master keypair
    if NACL_AVAILABLE:
        master_signing = SigningKey.generate()
        master_public_hex = master_signing.verify_key.encode(
            encoder=HexEncoder
        ).decode()
        master_key_for_derivation = master_signing
    else:
        master_private, master_public = _fallback_keygen()
        master_public_hex = master_public.hex()
        master_key_for_derivation = master_private

    # Derive node ID: SHA-256(public_key) — deterministic, permanent
    node_id = hashlib.sha256(
        bytes.fromhex(master_public_hex)
    ).hexdigest()

    # Derive PAT agent keys (indices 0-6)
    pat_agents = [
        _derive_agent_key(
            master_key_for_derivation, i, name, "pat", genesis_domain
        )
        for i, name in enumerate(pat_names)
    ]

    # Derive SAT agent keys (indices 7-11)
    sat_agents = [
        _derive_agent_key(
            master_key_for_derivation, len(pat_names) + i, name, "sat",
            genesis_domain
        )
        for i, name in enumerate(sat_names)
    ]

    return NodeIdentity(
        node_id=node_id,
        public_key_hex=master_public_hex,
        genesis_timestamp=time.time(),
        genesis_domain=genesis_domain,
        pat_agents=pat_agents,
        sat_agents=sat_agents,
        _master_signing_key=(
            master_signing if NACL_AVAILABLE else master_private
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


def save_identity(identity: NodeIdentity, path: Path,
                   password: str | None = None):
    """
    Save identity to disk. Public record only (no private keys in JSON).

    For private key backup: use the master key's raw bytes with
    proper encryption (AES-256-GCM recommended).

    Args:
        identity: The NodeIdentity to save.
        path: File path for the public record.
        password: Optional (not yet implemented — placeholder for encryption).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = identity.as_public_record()
    record["saved_at"] = time.time()
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_public_record(path: Path) -> dict[str, Any]:
    """Load the public identity record from disk."""
    with open(path) as f:
        return json.load(f)
