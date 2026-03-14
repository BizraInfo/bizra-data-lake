"""Identity Genesis -- Layer 0 Formalization.

Definitions 1.6 (Identity Genesis) and 1.7 (Node Body) from the proof chain.

Standing on Giants: Bernstein (Ed25519, 2011) | BIP-32 (HD derivation, 2012) | Al-Ghazali (1095)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from core.integration.constants import DOMAIN_IDENTITY_GENESIS

GENESIS_SIGNATURE_DOMAIN = DOMAIN_IDENTITY_GENESIS


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


class HumanAttestation(StrEnum):
    """Lightweight human attestation posture at genesis time."""

    SELF_ASSERTED = "self_asserted"
    DEVICE_WITNESSED = "device_witnessed"
    PEER_ATTESTED = "peer_attested"
    VERIFIED_HUMAN = "verified_human"


class SovereigntyScope(StrEnum):
    """How far the genesis identity is intended to operate."""

    DEVICE_LOCAL = "device_local"
    USER_LOCAL = "user_local"
    NODE_LOCAL = "node_local"
    FEDERATION_ELIGIBLE = "federation_eligible"


@dataclass(frozen=True)
class PersonaSeed:
    """Initial persona material carried into genesis formalization."""

    display_name: str = ""
    mission_statement: str = ""
    locale: str = "en"


@dataclass(frozen=True)
class GenesisWalletState:
    """Lightweight wallet stub for SEED/BLOOM bootstrapping."""

    seed_balance: float = 0.0
    bloom_balance: float = 0.0
    seed_retention_ratio: float = 1.0
    zakat_due_ratio: float = 0.025
    bloom_transferable: bool = False


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


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signable_genesis_message(genesis_hash: str) -> bytes:
    return f"{GENESIS_SIGNATURE_DOMAIN}:{genesis_hash}".encode("utf-8")


def _public_key_from_private_key(private_key: bytes) -> bytes:
    signer = Ed25519PrivateKey.from_private_bytes(private_key)
    return signer.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


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
    human_attestation: HumanAttestation = HumanAttestation.SELF_ASSERTED
    sovereignty_scope: SovereigntyScope = SovereigntyScope.DEVICE_LOCAL
    persona_seed: PersonaSeed = field(default_factory=PersonaSeed)
    genesis_wallet_state: GenesisWalletState = field(default_factory=GenesisWalletState)
    genesis_signature: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if (
            self.genesis_wallet_state.seed_balance != self.wallet_seed_balance
            or self.genesis_wallet_state.bloom_balance != self.wallet_bloom_balance
        ):
            object.__setattr__(
                self,
                "genesis_wallet_state",
                GenesisWalletState(
                    seed_balance=self.wallet_seed_balance,
                    bloom_balance=self.wallet_bloom_balance,
                    seed_retention_ratio=self.genesis_wallet_state.seed_retention_ratio,
                    zakat_due_ratio=self.genesis_wallet_state.zakat_due_ratio,
                    bloom_transferable=self.genesis_wallet_state.bloom_transferable,
                ),
            )

    @staticmethod
    def create(
        public_key: bytes,
        *,
        persona_seed: Optional[PersonaSeed] = None,
        human_attestation: HumanAttestation = HumanAttestation.SELF_ASSERTED,
        sovereignty_scope: SovereigntyScope = SovereigntyScope.DEVICE_LOCAL,
        wallet_state: Optional[GenesisWalletState] = None,
        created_at: Optional[float] = None,
        genesis_signing_key: Optional[bytes] = None,
    ) -> IdentityGenesis:
        """Genesis creation from public key.

        The secret key (sk) stays on-device and is never passed
        to this constructor unless a local genesis signature is explicitly
        requested, enforcing P2 (Self-Sovereignty).
        """
        identity_id = derive_identity_id(public_key)
        genesis_wallet_state = wallet_state or GenesisWalletState()
        genesis = IdentityGenesis(
            public_key=public_key,
            identity_id=identity_id,
            sovereignty_class=SovereigntyClass.SEED,
            wallet_seed_balance=genesis_wallet_state.seed_balance,
            wallet_bloom_balance=genesis_wallet_state.bloom_balance,
            human_attestation=human_attestation,
            sovereignty_scope=sovereignty_scope,
            persona_seed=persona_seed or PersonaSeed(),
            genesis_wallet_state=genesis_wallet_state,
            created_at=time.time() if created_at is None else created_at,
        )
        if genesis_signing_key is None:
            return genesis
        return genesis.with_genesis_signature(genesis_signing_key)

    def genesis_payload(self) -> dict[str, Any]:
        """Canonical payload used for genesis hashing and signature derivation."""
        return {
            "public_key_hex": self.public_key.hex(),
            "identity_id": self.identity_id,
            "sovereignty_class": self.sovereignty_class.name,
            "human_attestation": self.human_attestation.value,
            "sovereignty_scope": self.sovereignty_scope.value,
            "persona_seed": {
                "display_name": self.persona_seed.display_name,
                "mission_statement": self.persona_seed.mission_statement,
                "locale": self.persona_seed.locale,
            },
            "wallet_state": {
                "seed_balance": self.genesis_wallet_state.seed_balance,
                "bloom_balance": self.genesis_wallet_state.bloom_balance,
                "seed_retention_ratio": self.genesis_wallet_state.seed_retention_ratio,
                "zakat_due_ratio": self.genesis_wallet_state.zakat_due_ratio,
                "bloom_transferable": self.genesis_wallet_state.bloom_transferable,
            },
            "created_at": round(float(self.created_at), 6),
        }

    @property
    def genesis_hash(self) -> str:
        """Canonical BLAKE2b hash of the genesis payload."""
        return hashlib.blake2b(
            _canonical_json(self.genesis_payload()),
            digest_size=32,
        ).hexdigest()

    @property
    def genesis_signature_domain(self) -> str:
        """Explicit domain separation tag for genesis signatures."""
        return GENESIS_SIGNATURE_DOMAIN

    def signable_payload(self) -> bytes:
        """Domain-separated payload for local genesis signing."""
        return _signable_genesis_message(self.genesis_hash)

    def with_genesis_signature(self, genesis_signing_key: bytes) -> IdentityGenesis:
        """Return a copy with an Ed25519 genesis signature attached."""
        expected_public_key = _public_key_from_private_key(genesis_signing_key)
        if expected_public_key != self.public_key:
            raise ValueError("Genesis signing key does not match public key")
        signer = Ed25519PrivateKey.from_private_bytes(genesis_signing_key)
        signature = signer.sign(self.signable_payload()).hex()
        return replace(self, genesis_signature=signature)

    def verify_genesis_signature(self) -> bool:
        """Verify the current genesis signature against the public key."""
        if not self.genesis_signature:
            return False
        verifier = Ed25519PublicKey.from_public_bytes(self.public_key)
        try:
            verifier.verify(bytes.fromhex(self.genesis_signature), self.signable_payload())
        except (InvalidSignature, ValueError):
            return False
        return True

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
