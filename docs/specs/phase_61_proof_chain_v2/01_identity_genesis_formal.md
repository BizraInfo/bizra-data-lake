# Step 1: Identity Genesis Formalization

## Standing on Giants: Bernstein (Ed25519, 2012) | BIP-32 (HD derivation, 2012) | Al-Ghazali (identity as covenant, 1095)

**Date:** 2026-03-03
**Ω⁷ Gem:** Ω⁷-4 (partial — identity component)
**Intent:** Formalize Layer 0 (Identity) and Layer 0.5 (Body) as mathematical objects

---

## Problem Statement

The proof chain's Definition 1.1 defines node state as a 5-tuple (H, C, I, L, Σ).
The Identity Genesis document (IDG-2026-001) establishes identity as Layer 0 with
Ed25519 keypairs, HD-derived agent keys, and sovereignty classes. The Body Layer
establishes hardware/model/knowledge inventory as dynamic state. Neither is
formalized in the proof chain, creating a gap between what's built and what's proven.

**Why it matters:** Without formalizing identity, Theorem 2.4 (Byzantine Resilience)
can't reason about Sybil resistance. Without formalizing body state, the HHMM
transition function can't formally condition on hardware capabilities.

---

## Mathematical Formalization

### Definition 1.6 (Identity Genesis)

```
A BIZRA identity εᵢ is a tuple:

  εᵢ = (pkᵢ, skᵢ, idᵢ, Wᵢ, Sᵢ)

Where:
  pkᵢ ∈ Ed25519.PublicKey    — public verification key
  skᵢ ∈ Ed25519.SecretKey    — private signing key (never leaves device)
  idᵢ = SHA-256(pkᵢ)         — deterministic identity derivation
  Wᵢ  = (seed_bal, bloom_bal) — wallet state (SEED balance, BLOOM ledger)
  Sᵢ  ∈ {SEED, SPROUT, TREE, FOREST} — sovereignty class

Properties:
  P1 (Uniqueness):       idᵢ = idⱼ  ⟹  pkᵢ = pkⱼ
  P2 (Self-Sovereignty):  skᵢ never transmitted over any channel
  P3 (Agent Derivation):  For k ∈ [0, 11]:
                           (pk_k, sk_k) = HD-Derive(skᵢ, path="m/bizra/0'/k'")
  P4 (Monotonic Class):   Sᵢ(t+1) ≥ Sᵢ(t) under normal operation

Agent roles derived from P3:
  k=0..6  → PAT agents (personal)
  k=7..11 → SAT agents (system contribution)
```

### Definition 1.7 (Node Body)

```
The body state Bᵢ(t) is the physical resource inventory:

  Bᵢ(t) = (CPUᵢ, GPUᵢ, RAMᵢ, Diskᵢ, Modelsᵢ(t), KGᵢ(t))

Where:
  CPUᵢ   ∈ ℕ          — core count (static after genesis)
  GPUᵢ   ∈ ℕ × ℕ      — (VRAM_MB, compute_capability) or ∅
  RAMᵢ   ∈ ℕ          — total memory bytes (static)
  Diskᵢ  ∈ ℕ          — available storage bytes (dynamic)
  Modelsᵢ(t) ⊆ ModelRegistry — set of loaded inference models
  KGᵢ(t) ⊆ KnowledgeGraph  — local knowledge graph state

The surplus function determines Pool contribution capacity:

  Surplusᵢ(t) = Bᵢ(t) - Utilᵢ(t)

Where Utilᵢ(t) is current resource utilization. A node can contribute
to the SAT Resource Pool only resources in its surplus.

Hardware capability predicate:

  CanExecuteLocal(mission, Bᵢ(t)) =
    mission.required_models ⊆ Modelsᵢ(t) ∧
    mission.min_gpu ≤ GPUᵢ.vram ∧
    mission.min_ram ≤ RAMᵢ - Utilᵢ(t).ram
```

---

## Pseudocode

### core/identity/genesis.py (amendments)

```pseudocode
"""Identity Genesis — Layer 0 Formalization.

Standing on Giants: Bernstein (Ed25519) | BIP-32 (HD derivation)
"""

FROM dataclasses IMPORT dataclass, field
FROM enum IMPORT IntEnum
FROM hashlib IMPORT sha256

IMPORT ed25519  # or nacl.signing


CLASS SovereigntyClass(IntEnum):
    """Monotonic sovereignty progression.
    P4: S(t+1) >= S(t) under normal operation.
    """
    SEED = 0      # [0.00, 0.25)
    SPROUT = 1    # [0.25, 0.50)
    TREE = 2      # [0.50, 0.75)
    FOREST = 3    # [0.75, 1.00]


@dataclass(frozen=True)
CLASS IdentityGenesis:
    """Definition 1.6: BIZRA Identity.

    Properties:
      P1: Uniqueness — id is deterministic from pk
      P2: Self-sovereignty — sk never serialized for transport
      P3: Agent derivation — 12 HD-derived keypairs
      P4: Monotonic class — sovereignty only increases
    """
    public_key: bytes          # Ed25519 public key (32 bytes)
    identity_id: str           # SHA-256(public_key) hex
    sovereignty_class: SovereigntyClass
    seed_balance: float = 0.0
    bloom_balance: float = 0.0

    @staticmethod
    FUNCTION create(signing_key: ed25519.SigningKey) -> "IdentityGenesis":
        """Genesis creation from signing key.
        The signing key (sk) stays on-device (P2).
        """
        pk = signing_key.verify_key.encode()
        identity_id = sha256(pk).hexdigest()
        RETURN IdentityGenesis(
            public_key=pk,
            identity_id=identity_id,
            sovereignty_class=SovereigntyClass.SEED,
        )

    FUNCTION derive_agent_keys(self, signing_key) -> list:
        """P3: Derive 12 agent keypairs via HD path.
        k=0..6 → PAT agents, k=7..11 → SAT agents.
        """
        agents = []
        FOR k IN range(12):
            # HD derivation: m/bizra/0'/k'
            derived_seed = sha256(
                signing_key.encode() + k.to_bytes(4, "big")
            ).digest()
            agent_key = ed25519.SigningKey(derived_seed)
            agents.append(agent_key)
        RETURN agents

    FUNCTION assert_uniqueness(self, other: "IdentityGenesis") -> None:
        """P1: id_i = id_j implies pk_i = pk_j."""
        IF self.identity_id == other.identity_id:
            ASSERT self.public_key == other.public_key, \
                "Identity collision: same ID, different keys"


@dataclass
CLASS NodeBody:
    """Definition 1.7: Node Body (physical resource inventory).

    Dynamic state — changes as models are loaded/unloaded.
    """
    cpu_cores: int
    gpu_vram_mb: int           # 0 if no GPU
    gpu_compute_cap: float     # 0.0 if no GPU
    ram_bytes: int
    disk_bytes: int
    loaded_models: set = field(default_factory=set)
    knowledge_graph_size: int = 0

    FUNCTION surplus(self, current_util: "ResourceUtil") -> dict:
        """Compute surplus available for Pool contribution."""
        RETURN {
            "cpu_free": max(0, self.cpu_cores - current_util.cpu_used),
            "gpu_vram_free_mb": max(0, self.gpu_vram_mb - current_util.gpu_used_mb),
            "ram_free_bytes": max(0, self.ram_bytes - current_util.ram_used),
            "disk_free_bytes": max(0, self.disk_bytes - current_util.disk_used),
        }

    FUNCTION can_execute_local(self, mission_requirements: dict) -> bool:
        """Hardware capability predicate.
        Returns True if node has sufficient resources for local execution.
        """
        required_models = mission_requirements.get("required_models", set())
        IF NOT required_models.issubset(self.loaded_models):
            RETURN False

        min_gpu = mission_requirements.get("min_gpu_vram_mb", 0)
        IF min_gpu > self.gpu_vram_mb:
            RETURN False

        min_ram = mission_requirements.get("min_ram_bytes", 0)
        # Use surplus, not total — can't starve running processes
        IF min_ram > self.ram_bytes:  # conservative: check total
            RETURN False

        RETURN True
```

---

## TDD Anchors

```pseudocode
# tests/core/identity/test_identity_genesis_formal.py

TEST identity_id_is_deterministic:
    """P1: Same key always produces same identity."""
    key = ed25519.SigningKey.generate()
    id1 = IdentityGenesis.create(key)
    id2 = IdentityGenesis.create(key)
    ASSERT id1.identity_id == id2.identity_id
    ASSERT id1.public_key == id2.public_key

TEST identity_id_is_unique:
    """P1: Different keys produce different identities."""
    key1 = ed25519.SigningKey.generate()
    key2 = ed25519.SigningKey.generate()
    id1 = IdentityGenesis.create(key1)
    id2 = IdentityGenesis.create(key2)
    ASSERT id1.identity_id != id2.identity_id

TEST identity_derives_12_agent_keys:
    """P3: HD derivation produces exactly 12 agent keypairs."""
    key = ed25519.SigningKey.generate()
    identity = IdentityGenesis.create(key)
    agents = identity.derive_agent_keys(key)
    ASSERT len(agents) == 12

TEST agent_keys_are_distinct:
    """P3: All 12 derived keys are unique."""
    key = ed25519.SigningKey.generate()
    identity = IdentityGenesis.create(key)
    agents = identity.derive_agent_keys(key)
    pks = [a.verify_key.encode() FOR a IN agents]
    ASSERT len(set(pks)) == 12

TEST sovereignty_class_is_monotonic:
    """P4: Class ordering SEED < SPROUT < TREE < FOREST."""
    ASSERT SovereigntyClass.SEED < SovereigntyClass.SPROUT
    ASSERT SovereigntyClass.SPROUT < SovereigntyClass.TREE
    ASSERT SovereigntyClass.TREE < SovereigntyClass.FOREST

TEST genesis_starts_at_seed:
    """New identity begins at SEED sovereignty class."""
    key = ed25519.SigningKey.generate()
    identity = IdentityGenesis.create(key)
    ASSERT identity.sovereignty_class == SovereigntyClass.SEED

TEST body_surplus_never_negative:
    """Surplus function clamps at zero, never returns negative."""
    body = NodeBody(cpu_cores=4, gpu_vram_mb=0, gpu_compute_cap=0,
                    ram_bytes=8_000_000_000, disk_bytes=100_000_000_000)
    util = ResourceUtil(cpu_used=8, gpu_used_mb=0,
                        ram_used=16_000_000_000, disk_used=200_000_000_000)
    surplus = body.surplus(util)
    ASSERT surplus["cpu_free"] >= 0
    ASSERT surplus["ram_free_bytes"] >= 0
    ASSERT surplus["disk_free_bytes"] >= 0

TEST body_can_execute_with_gpu:
    """Node with GPU can execute GPU-requiring missions."""
    body = NodeBody(cpu_cores=16, gpu_vram_mb=24000, gpu_compute_cap=8.9,
                    ram_bytes=128_000_000_000, disk_bytes=1_000_000_000_000,
                    loaded_models={"phi3:mini", "mxbai-embed-large"})
    ASSERT body.can_execute_local({
        "required_models": {"phi3:mini"},
        "min_gpu_vram_mb": 8000,
    })

TEST body_cannot_execute_without_model:
    """Node without required model cannot execute locally."""
    body = NodeBody(cpu_cores=16, gpu_vram_mb=24000, gpu_compute_cap=8.9,
                    ram_bytes=128_000_000_000, disk_bytes=1_000_000_000_000,
                    loaded_models={"phi3:mini"})
    ASSERT NOT body.can_execute_local({
        "required_models": {"llama3.1-70b"},  # not loaded
        "min_gpu_vram_mb": 40000,             # more VRAM than available
    })

TEST body_cpu_only_delegates_gpu_missions:
    """CPU-only node correctly reports inability for GPU missions."""
    body = NodeBody(cpu_cores=8, gpu_vram_mb=0, gpu_compute_cap=0.0,
                    ram_bytes=16_000_000_000, disk_bytes=500_000_000_000)
    ASSERT NOT body.can_execute_local({"min_gpu_vram_mb": 1})
```

---

## Acceptance Criteria

1. `IdentityGenesis` dataclass with P1-P4 properties
2. `NodeBody` dataclass with surplus function and capability predicate
3. `SovereigntyClass` enum with SEED < SPROUT < TREE < FOREST ordering
4. All 11 TDD anchors GREEN
5. No changes to existing identity module API (additive only)
6. Full test suite GREEN

---

## Scope Boundary

**In scope:** Formalize existing identity and body concepts as typed Python objects.
**Out of scope:** Hardware attestation protocol, actual HD key derivation (BIP-32),
production Pool registration, body state synchronization protocol.
