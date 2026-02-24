# Genesis State Persistence

> Sovereign identity, PAT-7 agents, recovery phrase, and encrypted keys that survive process restarts.

## Quick Start

```python
from core.genesis.state_persistence import (
    SovereignState,
    save_sovereign_state,
    load_sovereign_state,
    state_exists,
    encrypt_private_key,
    decrypt_private_key,
    private_key_to_phrase,
)
from pathlib import Path

# Check if genesis already happened
if state_exists(Path("sovereign_state")):
    state = load_sovereign_state(Path("sovereign_state"))
    print(f"Node: {state.node_id}")
else:
    # First-time genesis
    state = SovereignState(
        node_id="NODE0-GENESIS",
        public_key="<ed25519-public-hex>",
        identity_card={...},
        pat_agents=[...],
        sat_agents=[...],
    )
    save_sovereign_state(state, Path("sovereign_state"))
```

## Directory Layout

```
sovereign_state/
  identity.json           # Signed identity card
  pat_manifest.json       # PAT-7 agent roster
  sat_manifest.json       # SAT-5 agent roster
  urp_pledge.json         # Universal Resource Pledge
  hardware.json           # Hardware fingerprint
  genesis_receipt.json    # Full ceremony record
  recovery_phrase.txt     # BIP39-style words (DELETE AFTER COPYING)
  .keystore/
    sovereign.enc         # Encrypted private key (PBKDF2 + XOR)
```

All files set to `0o700`/`0o600` permissions on Unix. `.keystore/sovereign.enc` restricted to owner read/write only.

## Recovery Phrase

24-word BIP39-compatible phrase derived from Ed25519 private key entropy:

```python
# Generate from private key
phrase = private_key_to_phrase(private_key_hex)
# ['abandon', 'ability', 'able', ...]  (24 words)

# Recover from phrase
recovered_key = phrase_to_private_key(phrase)
```

The recovery phrase file (`recovery_phrase.txt`) includes instructions to write words on paper and delete the file.

## Key Encryption

Private keys are encrypted at rest with a user passphrase:

```python
# Encrypt
encrypted = encrypt_private_key(private_key_hex, "my-secure-passphrase")
# Returns: {"salt": "...", "ciphertext": "...", "checksum": "...", "kdf": "pbkdf2-sha256"}

# Decrypt
key_hex = decrypt_private_key(encrypted, "my-secure-passphrase")
# Raises ValueError on wrong passphrase
```

**Security model**: PBKDF2-SHA256 (100,000 iterations) for key derivation, XOR for local encryption. Production deployments should use AES-256-GCM or a hardware security module.

## SovereignState Fields

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique node identifier |
| `public_key` | `str` | Ed25519 public key (hex) |
| `identity_card` | `dict` | Signed identity with creation timestamp |
| `pat_agents` | `List[dict]` | PAT-7 agent roster |
| `sat_agents` | `List[dict]` | SAT-5 agent roster |
| `urp_pledge` | `dict` | Universal Resource Pledge record |
| `genesis_receipt` | `dict` | Full ceremony receipt |
| `hardware_info` | `dict` | Hardware fingerprint (CPU, RAM, GPU) |
| `recovery_phrase` | `List[str]` | 24-word BIP39 recovery phrase |
| `encrypted_key` | `dict` | Encrypted private key bundle |
| `created_at` | `str` | ISO 8601 UTC timestamp |
| `version` | `str` | State format version |

## API Reference

| Function | Description |
|----------|-------------|
| `save_sovereign_state(state, dir)` | Write all state files to directory |
| `load_sovereign_state(dir) -> Optional[SovereignState]` | Load state; returns `None` if not found |
| `state_exists(dir) -> bool` | Check if `identity.json` exists |
| `encrypt_private_key(key, passphrase) -> dict` | Encrypt key with passphrase |
| `decrypt_private_key(encrypted, passphrase) -> str` | Decrypt key; raises `ValueError` on wrong passphrase |
| `private_key_to_phrase(key_hex) -> List[str]` | Derive 24-word phrase from key |
| `phrase_to_private_key(words) -> str` | Recover key from phrase (BLAKE3 KDF expansion) |
| `entropy_to_phrase(bytes, count) -> List[str]` | Raw entropy to word list |
| `phrase_to_entropy(words) -> bytes` | Word list back to raw entropy |

## Tests

```bash
pytest tests/core/genesis/ -v
# test_state_persistence.py — 27 tests (save/load/encryption/phrase roundtrips)
# test_smoke_genesis.py — 16 tests (ceremony smoke tests)
```

## Security Notes

- Private keys are never stored in plaintext on disk
- Recovery phrase file should be deleted after the user copies the words
- Checksum verification prevents returning garbage on wrong passphrase
- File permissions are set to owner-only (Unix); Windows relies on NTFS ACLs
- The 256-word BIP39 subset provides 192 bits of entropy (exceeds Ed25519's 128-bit security level)
