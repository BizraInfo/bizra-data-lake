"""Identity Genesis -- Layer 0 Formalization.

Definitions 1.6 (Identity Genesis) and 1.7 (Node Body) from the proof chain.

Standing on Giants: Bernstein (Ed25519, 2011) | BIP-32 (HD derivation, 2012) | Al-Ghazali (1095)
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class SovereigntyClass(IntEnum):
    """Monotonic sovereignty progression.

    P4: S(t+1) >= S(t) under normal operation.

    Ranges:
        SEED   [0.00, 0.25)
        SPROUT [0.25, 0.50)
        TREE   [0.50, 0.75)
        FOREST [0.75, 1.00]
    """

    SEED = 0
    SPROUT = 1
    TREE = 2
    FOREST = 3


def derive_identity_id(public_key: bytes) -> str:
    """Deterministic identity derivation: id = SHA-256(pk).

    Property P1 (Uniqueness): id_i = id_j implies pk_i = pk_j
    with negligible collision probability (2^-128).
    """
    return hashlib.sha256(public_key).hexdigest()


def derive_agent_keypairs(seed: bytes, count: int = 12) -> list[tuple[bytes, bytes]]:
    """HD-derived agent keypairs via HMAC-SHA256.

    Property P3: For k in [0, count-1]:
        (pk_k, sk_k) = HD-Derive(seed, path="m/bizra/0'/k'")

    Agent roles:
        k=0..6  -> PAT agents (personal)
        k=7..11 -> SAT agents (system contribution)

    Returns list of (public_key_bytes, secret_key_bytes) tuples where each
    key is a 32-byte deterministic derivation from the seed.
    """
    keypairs: list[tuple[bytes, bytes]] = []
    for k in range(count):
        path = f"m/bizra/0'/{k}'".encode()
        secret = hmac.new(seed, path, hashlib.sha256).digest()
        public = hashlib.sha256(secret).digest()
        keypairs.append((public, secret))
    return keypairs


@dataclass(frozen=True)
class IdentityGenesis:
    """Definition 1.6: BIZRA Identity.

    A BIZRA identity is a tuple: (pk, sk, id, W, S) where:
        pk  -- Ed25519 public verification key
        id  -- SHA-256(pk) deterministic identity
        S   -- sovereignty class in {SEED, SPROUT, TREE, FOREST}
        W   -- wallet state (seed_balance, bloom_balance)

    Properties:
        P1 (Uniqueness):       id_i = id_j implies pk_i = pk_j
        P2 (Self-Sovereignty): sk never transmitted over any channel
        P3 (Agent Derivation): 12 HD-derived keypairs from seed
        P4 (Monotonic Class):  S(t+1) >= S(t) under normal operation
    """

    public_key: bytes
    identity_id: str
    sovereignty_class: SovereigntyClass
    wallet_seed_balance: float = 0.0
    wallet_bloom_balance: float = 0.0
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def create(public_key: bytes) -> IdentityGenesis:
        """Genesis creation from public key.

        The secret key (sk) stays on-device and is never passed
        to this constructor, enforcing P2 (Self-Sovereignty).
        """
        identity_id = derive_identity_id(public_key)
        return IdentityGenesis(
            public_key=public_key,
            identity_id=identity_id,
            sovereignty_class=SovereigntyClass.SEED,
        )

    def assert_uniqueness(self, other: IdentityGenesis) -> None:
        """P1: id_i = id_j implies pk_i = pk_j."""
        if self.identity_id == other.identity_id:
            assert (
                self.public_key == other.public_key
            ), "Identity collision: same ID, different keys"


@dataclass
class NodeBody:
    """Definition 1.7: Node Body (physical resource inventory).

    B(t) = (CPU, GPU, RAM, Disk, Models(t), KG(t))

    Dynamic state that changes as models are loaded/unloaded
    and knowledge graph grows.
    """

    cpu_cores: int
    gpu_vram_mb: int
    gpu_compute_cap: Optional[float] = None
    ram_bytes: int = 0
    disk_bytes: int = 0
    loaded_models: set[str] = field(default_factory=set)
    knowledge_graph_size: int = 0

    def surplus(self, current_util: Optional[dict[str, Any]] = None) -> dict[str, int]:
        """Compute surplus available for SAT Resource Pool contribution.

        Surplus(t) = B(t) - Util(t), clamped to zero.
        """
        if current_util is None:
            return {
                "cpu_free": self.cpu_cores,
                "gpu_vram_free_mb": self.gpu_vram_mb,
                "ram_free_bytes": self.ram_bytes,
                "disk_free_bytes": self.disk_bytes,
            }
        return {
            "cpu_free": max(0, self.cpu_cores - current_util.get("cpu_used", 0)),
            "gpu_vram_free_mb": max(
                0, self.gpu_vram_mb - current_util.get("gpu_used_mb", 0)
            ),
            "ram_free_bytes": max(0, self.ram_bytes - current_util.get("ram_used", 0)),
            "disk_free_bytes": max(
                0, self.disk_bytes - current_util.get("disk_used", 0)
            ),
        }

    def can_execute_local(self, mission: dict[str, Any]) -> bool:
        """Hardware capability predicate.

        CanExecuteLocal(mission, B(t)) =
            mission.required_models is a subset of Models(t) AND
            mission.min_gpu <= GPU.vram AND
            mission.min_ram <= RAM
        """
        required_models = mission.get("required_models", set())
        if not required_models.issubset(self.loaded_models):
            return False

        min_gpu = mission.get("min_gpu_vram_mb", 0)
        if min_gpu > self.gpu_vram_mb:
            return False

        min_ram = mission.get("min_ram_bytes", 0)
        if min_ram > self.ram_bytes:
            return False

        return True
