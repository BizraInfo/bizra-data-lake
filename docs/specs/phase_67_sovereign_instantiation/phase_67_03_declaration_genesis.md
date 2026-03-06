# Phase 67.03 — Declaration as Genesis Block
# ═══════════════════════════════════════════

## Standing on Giants
- Nakamoto (2008): Genesis block as immutable origin
- Merkle (1979): Hash chains for integrity
- Jefferson (1776): Declaration of Independence as founding document
- Al-Ghazali (1095): The covenant precedes all action

## Source

`last update/BIZRA_DECLARATION.md` (80 lines)
Hash: `BLAKE2b-256: 859649ea1a44f1bf4c183105a42e66b0a9d34505c53786d639d965b3afa46474`

## Purpose

The Declaration of Digital Sovereignty is not a document — it is the **genesis
block** of the BIZRA event log. Every ActionReceipt chains back to this hash.
If a node's local Declaration hash doesn't match, its receipts are invalid.

This creates **Covenant Locking**: the constitutional invariants (I-1 through I-7)
are not configuration — they are cryptographically anchored to the genesis state.

## Target

```
core/constitutional/declaration.py
00_CONSTITUTION/DECLARATION.md        # Canonical copy
```

## Pseudocode

```
MODULE declaration

IMPORT blake2b FROM hashlib
IMPORT blake3 FROM core.proof_engine.canonical
IMPORT Path

# The canonical Declaration hash — computed from the original text.
# This is a CONSTANT, never recomputed at runtime.
DECLARATION_BLAKE2B_256 = "17d672371bb8eff01676fdec010fef17f20f624bf1b3557d39057a0a16b65fe8"
# NOTE: Hash computed from canonical 00_CONSTITUTION/DECLARATION.md (UTF-8, Unix LF).
# Original document hash (859649ea...) was from initial proclamation text.

# Path to the canonical Declaration file
DECLARATION_PATH = Path("00_CONSTITUTION/DECLARATION.md")

# Seven Constitutional Invariants
@dataclass
CLASS ConstitutionalInvariant:
    code: str       # "I-1" through "I-7"
    guarantee: str  # Human-readable guarantee
    algorithm: str  # Which algorithm enforces it
    threshold: Optional[int]  # Fixed-point threshold (if applicable)

INVARIANTS = [
    ConstitutionalInvariant(
        code="I-1",
        guarantee="No interest shall exist at any layer",
        algorithm="A2_SEED_MINTER",
        threshold=None  # Boolean: SEED is minted from work, not interest
    ),
    ConstitutionalInvariant(
        code="I-2",
        guarantee="No transaction shall contain hidden uncertainty",
        algorithm="A14_EVENT_SOURCER",
        threshold=None  # Boolean: all events are deterministic
    ),
    ConstitutionalInvariant(
        code="I-3",
        guarantee="Wealth concentration shall not exceed Gini 0.35",
        algorithm="A4_GINI_ENFORCER",
        threshold=fp(0.35)  # ADL_GINI_THRESHOLD from constants.py
    ),
    ConstitutionalInvariant(
        code="I-4",
        guarantee="Only work of verified excellence produces value",
        algorithm="A1_IHSAN_SCORER",
        threshold=fp(0.95)  # UNIFIED_IHSAN_THRESHOLD from constants.py
    ),
    ConstitutionalInvariant(
        code="I-5",
        guarantee="Governance belongs to those who participate",
        algorithm="A8_SHURA_GOVERNANCE",
        threshold=None  # BLOOM is soulbound, cannot be bought
    ),
    ConstitutionalInvariant(
        code="I-6",
        guarantee="Every node shall be sovereign over its data",
        algorithm="SOVEREIGNTY_KERNEL",
        threshold=None  # Ed25519 keypair = local control
    ),
    ConstitutionalInvariant(
        code="I-7",
        guarantee="Wealth above threshold shall be purified annually",
        algorithm="A5_ZAKAT_ENGINE",
        threshold=fp(0.025)  # ZAKAT_RATE = 2.5%
    ),
]


FUNCTION load_declaration(path: Path = DECLARATION_PATH) -> str:
    """Load the Declaration text from the canonical file."""
    IF NOT path.exists():
        RAISE FileNotFoundError(f"Declaration not found at {path}")
    RETURN path.read_text(encoding="utf-8")


FUNCTION verify_declaration_hash(text: str) -> bool:
    """Verify the Declaration hash matches the canonical value.

    Any modification to the Declaration — even a single byte — will
    cause this check to fail, preventing covenant-broken nodes from
    participating in the network.
    """
    computed = blake2b(text.encode("utf-8"), digest_size=32).hexdigest()
    RETURN computed == DECLARATION_BLAKE2B_256


FUNCTION create_genesis_event(declaration_text: str) -> Event:
    """Create the genesis event (Event #0) from the Declaration.

    This is the root of the Merkle chain. All subsequent events
    chain back to this hash.
    """
    IF NOT verify_declaration_hash(declaration_text):
        RAISE ConstitutionalViolation("Declaration hash mismatch — covenant broken")

    genesis_hash = blake3(declaration_text.encode("utf-8"))

    RETURN Event(
        event_id=0,
        event_type="genesis",
        actor=b'\x00' * 32,  # System actor (no individual)
        data={
            "declaration_hash_blake2b": DECLARATION_BLAKE2B_256,
            "declaration_hash_blake3": genesis_hash.hex(),
            "invariants": [inv.code FOR inv IN INVARIANTS],
            "version": "1.0.0",
            "proclaimed": "2026-03-05T00:00:00Z",
            "location": "Dubai",
        },
        timestamp=0,  # Genesis timestamp = epoch
        prev_hash=b'\x00' * 32,  # No previous event
        hash=genesis_hash
    )


FUNCTION verify_covenant_chain(event_log: List[Event]) -> (bool, List[str]):
    """Verify that the event log chains back to the Declaration genesis.

    Returns (valid, list_of_errors).
    """
    errors = []

    IF len(event_log) == 0:
        RETURN (False, ["Empty event log — no genesis"])

    genesis = event_log[0]
    IF genesis.event_type != "genesis":
        errors.append("First event is not genesis")

    IF genesis.data.get("declaration_hash_blake2b") != DECLARATION_BLAKE2B_256:
        errors.append("Genesis declaration hash does not match canonical value")

    # Verify chain integrity (A14)
    FOR i IN range(1, len(event_log)):
        IF event_log[i].prev_hash != event_log[i-1].hash:
            errors.append(f"Chain break at event {i}")

    RETURN (len(errors) == 0, errors)


CLASS ConstitutionalViolation(Exception):
    """Raised when a constitutional invariant is violated.

    This exception CANNOT be caught by application code — it is a
    hard stop. The node must halt and report the violation.
    """
    pass
```

## Integration with Existing Genesis Ceremony

The existing `core/proof_engine/genesis_ceremony.py` creates PAT/SAT rosters
and identity keypairs. This spec **extends** (not replaces) that ceremony:

```
EXISTING FLOW:
  GenesisOrchestrator → identity → hardware → PAT → SAT → genesis_hash

EXTENDED FLOW:
  GenesisOrchestrator → Declaration verification → identity → hardware →
  PAT → SAT → Constitutional invariants binding → genesis_hash
```

The Declaration hash becomes the **first link** in the genesis hash chain.
`sovereign_state/node0_genesis.json` gains a new field:
```json
{
  "declaration_hash": "859649ea...",
  "covenant_verified": true,
  "invariants_bound": ["I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-7"]
}
```

## File Placement

```
00_CONSTITUTION/
├── DECLARATION.md           # Canonical copy (hash-verified)
├── INVARIANTS.json          # Machine-readable I-1 through I-7
└── GENESIS_HASH.txt         # BLAKE3 of Declaration (for quick verification)
```

## TDD Anchors

```python
# tests/constitutional/test_declaration.py

def test_declaration_hash_matches():
    """Canonical Declaration produces expected BLAKE2b hash."""
    text = load_declaration()
    assert verify_declaration_hash(text)

def test_modified_declaration_fails():
    """Any modification to Declaration fails verification."""
    text = load_declaration()
    modified = text.replace("BIZRA", "MODIFIED")
    assert not verify_declaration_hash(modified)

def test_genesis_event_creation():
    """Genesis event has correct structure."""
    text = load_declaration()
    event = create_genesis_event(text)
    assert event.event_id == 0
    assert event.event_type == "genesis"
    assert event.prev_hash == b'\x00' * 32

def test_covenant_chain_valid():
    """Valid chain passes verification."""
    text = load_declaration()
    genesis = create_genesis_event(text)
    valid, errors = verify_covenant_chain([genesis])
    assert valid
    assert len(errors) == 0

def test_covenant_chain_broken():
    """Tampered chain fails verification."""
    text = load_declaration()
    genesis = create_genesis_event(text)
    tampered = Event(event_id=1, event_type="mint",
                     prev_hash=b'\xff' * 32, ...)  # Wrong prev_hash
    valid, errors = verify_covenant_chain([genesis, tampered])
    assert not valid
    assert "Chain break" in errors[0]

def test_all_seven_invariants_present():
    """All 7 invariants are defined and have algorithms."""
    assert len(INVARIANTS) == 7
    codes = {inv.code for inv in INVARIANTS}
    assert codes == {"I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-7"}
    for inv in INVARIANTS:
        assert inv.algorithm  # Every invariant has an enforcement algorithm
```
