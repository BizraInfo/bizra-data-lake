# Phase 19: Integration Completion — Ed25519Signer + Token CLI

## Context

Swarm validation sweep (Task #47) achieved **6,345 passed** across the full test suite.
26 remaining failures are all in `tests/integration/` — the last barrier to a fully
green suite. Root cause analysis identifies **2 missing modules** and **1 fixture gap**.

| Failure Group | Count | Root Cause |
|---------------|-------|-----------|
| Ed25519Signer class missing | 14 | `core/proof_engine/receipt.py` has no production signer |
| Token CLI handlers missing | 4 | `core/sovereign/__main__.py` lacks wallet/tokens commands |
| Token API test isolation | 1 | `/v1/token/verify` uses corrupted global ledger |
| Ed25519Signer transient (subset of #1) | 7 | Cascade from missing signer class |

**Total: 26 failures, 2 modules to implement, 1 test fixture to isolate.**

---

## Module 1: Ed25519Signer (14 failures)

### File: `core/proof_engine/receipt.py`

**Insert after** `SimpleSigner` class (line 81), **before** `Metrics` dataclass.

### Existing Infrastructure (zero new dependencies)

```
core/pci/crypto.py:
  generate_keypair() -> Tuple[str, str]           # Ed25519 key generation
  sign_message(digest_hex, private_key_hex) -> str # Ed25519 signing
  verify_signature(digest_hex, sig_hex, pub_hex) -> bool

core/proof_engine/canonical.py:
  hex_digest(data: bytes | str) -> str             # BLAKE3 bare hash (64 hex chars)
```

### Pseudocode

```
CLASS Ed25519Signer
    """
    Production Ed25519 signer implementing SovereignSigner protocol.
    Uses BLAKE3 for message hashing (SEC-001 compliant).

    Standing on: Bernstein et al. (2012) — Ed25519 high-speed signatures
    """

    CONSTRUCTOR(private_key_hex: str, public_key_hex: str)
        STORE _private_key_hex = private_key_hex
        STORE _public_key_hex = public_key_hex
        VALIDATE len(public_key_hex) == 64  # 32 bytes hex-encoded

    CLASS METHOD generate() -> Ed25519Signer
        """Generate fresh keypair and return initialized signer."""
        private, public = crypto.generate_keypair()
        RETURN Ed25519Signer(private_key_hex=private, public_key_hex=public)

    PROPERTY public_key_hex -> str
        RETURN self._public_key_hex

    METHOD sign(msg: bytes) -> bytes
        """Sign msg using BLAKE3 + Ed25519. Returns 64 raw signature bytes."""
        digest_hex = canonical.hex_digest(msg)              # BLAKE3 bare hash
        signature_hex = crypto.sign_message(digest_hex, self._private_key_hex)
        RETURN bytes.fromhex(signature_hex)                 # 64 bytes

    METHOD verify(msg: bytes, signature: bytes) -> bool
        """Verify signature against msg using BLAKE3 + Ed25519."""
        digest_hex = canonical.hex_digest(msg)
        RETURN crypto.verify_signature(
            digest_hex, signature.hex(), self._public_key_hex
        )

    METHOD public_key_bytes() -> bytes
        """Return raw 32-byte public key."""
        RETURN bytes.fromhex(self._public_key_hex)
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| BLAKE3 bare (not domain-separated) | Tests explicitly verify `hex_digest(msg)` path; PCI envelopes use domain-separated. Both are SEC-001 compliant. |
| Wraps `core.pci.crypto` | Zero duplication — all Ed25519 math in one place |
| Returns raw `bytes` from `sign()` | Tests assert `len(sig) == 64` and `isinstance(sig, bytes)` |
| `generate()` classmethod | Tests call `Ed25519Signer.generate()` directly |
| No key validation beyond length | `core.pci.crypto` handles actual Ed25519 validation |

### TDD Anchors (14 tests satisfied)

```
test_ed25519_signer_creation
    signer = Ed25519Signer.generate()
    ASSERT signer.public_key_hex is not None
    ASSERT len(signer.public_key_hex) == 64

test_ed25519_signer_sign_verify
    signer = Ed25519Signer.generate()
    sig = signer.sign(b"sovereignty integrity check")
    ASSERT len(sig) == 64                        # Ed25519 = 64 bytes
    ASSERT signer.verify(b"sovereignty integrity check", sig) is True

test_ed25519_signer_tamper_detection
    signer = Ed25519Signer.generate()
    sig = signer.sign(b"original")
    ASSERT signer.verify(b"tampered message", sig) is False

test_ed25519_signer_public_key_bytes
    signer = Ed25519Signer.generate()
    pk_bytes = signer.public_key_bytes()
    ASSERT pk_bytes == bytes.fromhex(signer.public_key_hex)

test_ed25519_signer_deterministic_with_key
    priv, pub = generate_keypair()
    s1 = Ed25519Signer(private_key_hex=priv, public_key_hex=pub)
    s2 = Ed25519Signer(private_key_hex=priv, public_key_hex=pub)
    ASSERT s1.public_key_hex == s2.public_key_hex

test_receipt_sign_with_ed25519
    signer = Ed25519Signer.generate()
    builder = ReceiptBuilder(signer)
    receipt = builder.accepted(query, policy, payload, snr=0.92, ihsan_score=0.96)
    ASSERT receipt.verify_signature(signer) is True

test_receipt_cross_signer_fails
    s1 = Ed25519Signer.generate()
    s2 = Ed25519Signer.generate()
    builder = ReceiptBuilder(s1)
    receipt = builder.accepted(...)
    ASSERT receipt.verify_signature(s2) is False

test_ed25519_signer_uses_blake3
    signer = Ed25519Signer(private_key_hex=priv, public_key_hex=pub)
    sig = signer.sign(b"signer blake3 check")
    ASSERT isinstance(sig, bytes)
    ASSERT len(sig) == 64
    # Cross-verify: hex_digest(msg) + verify_signature should match
    digest_hex = hex_digest(b"signer blake3 check")
    ASSERT verify_signature(digest_hex, sig.hex(), pub) is True

test_ed25519_signer_verify_matches_crypto_module
    # Ed25519Signer.verify() produces same result as raw crypto.verify_signature()
    VERIFIED by cross-module check above

# FullStackSmoke tests (3) and CrossLayerL2L3 tests (2) depend on Ed25519Signer
# existing — they will pass once the class is implemented.
```

---

## Module 2: Token CLI Handlers (4 failures)

### File: `core/sovereign/__main__.py`

### Functions to Add

```
FUNCTION _handle_wallet_command() -> None
    """Display token balances for all known accounts."""
    FROM core.token.ledger IMPORT TokenLedger
    FROM core.token.types IMPORT TokenType

    ledger = TokenLedger()
    accounts = ledger.list_accounts()          # Returns list of account IDs

    IF NOT accounts:
        PRINT "No accounts found. Run 'sovereign onboard' to create identity."
        RETURN

    PRINT "=== BIZRA Token Wallet ==="
    PRINT ""

    FOR account IN accounts:
        PRINT f"  {account}:"
        FOR token_type IN TokenType:
            balance = ledger.get_balance(account, token_type)
            IF balance > 0:
                PRINT f"    {token_type.value}: {balance:,.2f}"
        PRINT ""

    PRINT f"  Total accounts: {len(accounts)}"


FUNCTION _handle_tokens_command() -> None
    """Display token supply statistics and chain validity."""
    FROM core.token.ledger IMPORT TokenLedger
    FROM core.token.types IMPORT TokenType, SEED_SUPPLY_CAP_PER_YEAR

    ledger = TokenLedger()

    PRINT "=== BIZRA Token Supply ==="
    PRINT ""

    FOR token_type IN TokenType:
        supply = ledger.total_supply(token_type)
        PRINT f"  {token_type.value}: {supply:,.2f}"

    PRINT ""
    PRINT f"  Yearly SEED cap: {SEED_SUPPLY_CAP_PER_YEAR:,.0f}"

    # Chain integrity
    valid, tx_count, error = ledger.verify_chain()
    status = "VALID" IF valid ELSE f"INVALID ({error})"
    PRINT f"  Chain status: {status}"
    PRINT f"  Transactions: {tx_count}"
```

### CLI Wiring

```
# In main() — after existing subparsers (after line 770, before "version"):

ADD subparser "wallet" with help="View token wallet balances"
ADD subparser "tokens" with help="View token supply and stats"

# In command routing (after bridge block, before version):

ELIF args.command == "wallet":
    _handle_wallet_command()
ELIF args.command == "tokens":
    _handle_tokens_command()
```

### TDD Anchors (4 tests satisfied)

```
test_wallet_command_shows_balances
    # Setup: Create ledger in tmp_path, mint some tokens
    FROM core.sovereign.__main__ IMPORT _handle_wallet_command
    _handle_wallet_command()
    captured = capsys.readouterr()
    ASSERT "BIZRA Token Wallet" IN captured.out
    ASSERT "SEED" IN captured.out OR "No accounts" IN captured.out

test_tokens_command_shows_supply
    FROM core.sovereign.__main__ IMPORT _handle_tokens_command
    _handle_tokens_command()
    captured = capsys.readouterr()
    ASSERT "BIZRA Token Supply" IN captured.out

test_wallet_command_no_genesis
    # Empty ledger scenario
    FROM core.sovereign.__main__ IMPORT _handle_wallet_command
    _handle_wallet_command()
    captured = capsys.readouterr()
    ASSERT "No accounts" IN captured.out

test_cli_token_functions_exist
    FROM core.sovereign.__main__ IMPORT (
        _handle_wallet_command,
        _handle_tokens_command,
    )
    ASSERT callable(_handle_wallet_command)
    ASSERT callable(_handle_tokens_command)
```

---

## Module 3: Token API Test Isolation (1 failure)

### File: `tests/integration/test_token_api.py`

### Problem

`test_token_verify_endpoint` asserts `data["valid"] is True`, but the `/v1/token/verify`
endpoint reads a global ledger file (`04_GOLD/token_ledger.jsonl`) that has hash chain
corruption from schema migration.

### Fix Strategy

The test already uses a `client` fixture with `TestClient`. The fixture should
override the `TokenLedger` path to use `tmp_path` with a freshly-minted genesis:

```
FIXTURE client(tmp_path):
    FROM core.token.ledger IMPORT TokenLedger
    FROM core.token.mint IMPORT TokenMinter

    # Create isolated ledger
    ledger = TokenLedger(storage_dir=tmp_path)
    minter = TokenMinter(ledger=ledger)
    minter.genesis()  # Clean genesis mint

    # Patch the app's ledger creation to use our isolated instance
    WITH patch("core.sovereign.api._get_token_ledger", return_value=ledger):
        FROM core.sovereign.api IMPORT create_app
        app = create_app()
        YIELD TestClient(app)
```

### TDD Anchor

```
test_token_verify_endpoint
    resp = client.get("/v1/token/verify")
    ASSERT resp.status_code == 200
    data = resp.json()
    ASSERT data["valid"] is True            # Fresh ledger = valid chain
    ASSERT data["transaction_count"] >= 1   # Genesis tx exists
```

---

## Implementation Order

| Step | Module | Est. Lines | Dependencies |
|------|--------|-----------|--------------|
| 1 | Ed25519Signer class | ~45 | None (wraps existing crypto) |
| 2 | Token CLI handlers | ~50 | `core/token/` (exists) |
| 3 | Token API test fixture | ~15 | `core/token/` (exists) |
| 4 | Regression test sweep | — | Steps 1-3 complete |

**Total: ~110 lines of implementation across 3 files.**

---

## Verification Plan

```bash
# Step 1: Ed25519Signer (14 failures → 0)
PYTHONPATH=. pytest tests/integration/test_seven_layer_stack.py \
                    tests/integration/test_cross_language_crypto.py -v

# Step 2: Token CLI (4 failures → 0)
PYTHONPATH=. pytest tests/integration/test_token_integration.py -v

# Step 3: Token API fixture (1 failure → 0)
PYTHONPATH=. pytest tests/integration/test_token_api.py -v

# Step 4: Full sweep (26 failures → 0)
PYTHONPATH=. pytest tests/ -q --timeout=60

# Expected: 6,371 passed, 0 failed, ~47 skipped
```

---

## Cross-Language Interop Note

The Ed25519Signer must produce signatures verifiable by the Rust implementation in
`bizra-omega/bizra-core/src/identity.rs`. Both use:
- **BLAKE3** bare hash for receipts (not domain-separated)
- **Ed25519** (RFC 8032) for signatures
- **64-byte** raw signature format

The existing `core.pci.crypto` module already implements this correctly using
`ed25519-dalek` compatible operations — wrapping it preserves cross-language parity.
