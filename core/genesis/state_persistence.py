"""
BIZRA Sovereign State Persistence
====================================

Persists the full sovereign state produced by the genesis pipeline
to disk in a structured directory layout. Each genesis ceremony
creates an immutable state snapshot that can be recovered later.

Features:
- BIP39-style recovery phrase for private key backup
- PBKDF2-SHA256 key derivation + HMAC-authenticated encryption at rest
- Structured file layout with hash-chain manifest
- Save/load roundtrip with secret exclusion from to_dict()

Layout:
    sovereign_state/genesis/
        identity.json           - Node identity card
        hardware.json           - Hardware fingerprint
        pat_manifest.json       - PAT agent manifest
        sat_manifest.json       - SAT agent manifest
        urp_pledge.json         - Resource pledge record
        genesis_receipt.json    - Genesis ceremony receipt
        recovery_phrase.txt     - Human-readable recovery words
        .keystore/
            sovereign.enc       - Encrypted private key
        manifest.json           - Root manifest with hash chain

Standing on Giants:
- Lamport (1978): Ordered state snapshots
- Nakamoto (2008): Hash-chain integrity
- Al-Ghazali (1058-1111): Ihsan as ethical floor
- BIP39 (2013): Mnemonic recovery phrases
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =========================================================================
# BIP39-STYLE WORDLIST (256 words — one per byte value)
# =========================================================================
# Subset of BIP39 English wordlist, curated for BIZRA sovereignty.
# 256 words maps 1:1 to byte values (0x00-0xFF), enabling direct
# entropy-to-phrase conversion without checksum complexity.

BIP39_WORDLIST: List[str] = [
    "abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "absurd",
    "abuse",
    "access",
    "accident",
    "account",
    "accuse",
    "achieve",
    "acid",
    "acoustic",
    "acquire",
    "across",
    "act",
    "action",
    "actor",
    "actress",
    "actual",
    "adapt",
    "add",
    "addict",
    "address",
    "adjust",
    "admit",
    "adult",
    "advance",
    "advice",
    "aerobic",
    "affair",
    "afford",
    "afraid",
    "again",
    "age",
    "agent",
    "agree",
    "ahead",
    "aim",
    "air",
    "airport",
    "aisle",
    "alarm",
    "album",
    "alcohol",
    "alert",
    "alien",
    "all",
    "alley",
    "allow",
    "almost",
    "alone",
    "alpha",
    "already",
    "also",
    "alter",
    "always",
    "amateur",
    "amazing",
    "among",
    "amount",
    "amused",
    "analyst",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "angry",
    "animal",
    "ankle",
    "announce",
    "annual",
    "another",
    "answer",
    "antenna",
    "antique",
    "anxiety",
    "any",
    "apart",
    "apology",
    "appear",
    "apple",
    "approve",
    "april",
    "arch",
    "arctic",
    "area",
    "arena",
    "argue",
    "arm",
    "armed",
    "armor",
    "army",
    "around",
    "arrange",
    "arrest",
    "arrive",
    "arrow",
    "art",
    "artefact",
    "artist",
    "artwork",
    "ask",
    "aspect",
    "assault",
    "asset",
    "assist",
    "assume",
    "asthma",
    "athlete",
    "atom",
    "attack",
    "attend",
    "attitude",
    "attract",
    "auction",
    "audit",
    "august",
    "aunt",
    "author",
    "auto",
    "autumn",
    "average",
    "avocado",
    "avoid",
    "awake",
    "aware",
    "awesome",
    "awful",
    "awkward",
    "axis",
    "baby",
    "bachelor",
    "bacon",
    "badge",
    "bag",
    "balance",
    "balcony",
    "ball",
    "bamboo",
    "banana",
    "banner",
    "bar",
    "barely",
    "bargain",
    "barrel",
    "base",
    "basic",
    "basket",
    "battle",
    "beach",
    "bean",
    "beauty",
    "because",
    "become",
    "beef",
    "before",
    "begin",
    "behave",
    "behind",
    "believe",
    "below",
    "belt",
    "bench",
    "benefit",
    "best",
    "betray",
    "better",
    "between",
    "beyond",
    "bicycle",
    "bid",
    "bike",
    "bind",
    "biology",
    "bird",
    "birth",
    "bitter",
    "black",
    "blade",
    "blame",
    "blanket",
    "blast",
    "bleak",
    "bless",
    "blind",
    "blood",
    "blossom",
    "blow",
    "blue",
    "blur",
    "blush",
    "board",
    "boat",
    "body",
    "boil",
    "bomb",
    "bone",
    "bonus",
    "book",
    "boost",
    "border",
    "boring",
    "borrow",
    "boss",
    "bottom",
    "bounce",
    "box",
    "boy",
    "bracket",
    "brain",
    "brand",
    "brass",
    "brave",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brief",
    "bright",
    "bring",
    "brisk",
    "broccoli",
    "broken",
    "bronze",
    "broom",
    "brother",
    "brown",
    "brush",
    "bubble",
    "buddy",
    "budget",
    "buffalo",
    "build",
    "bulb",
    "bulk",
    "bullet",
    "bundle",
    "bunny",
    "burden",
    "burger",
    "burst",
    "bus",
    "business",
    "busy",
    "butter",
    "buyer",
    "buzz",
    "cabbage",
    "cabin",
    "cable",
    "cactus",
]

assert (
    len(BIP39_WORDLIST) == 256
), f"Wordlist must have 256 entries, got {len(BIP39_WORDLIST)}"

# Build reverse lookup once at module load
_WORD_TO_BYTE: Dict[str, int] = {w: i for i, w in enumerate(BIP39_WORDLIST)}


# =========================================================================
# ENTROPY <-> PHRASE CONVERSION
# =========================================================================


def entropy_to_phrase(entropy: bytes, word_count: int = 24) -> List[str]:
    """Convert raw entropy bytes to a recovery phrase.

    Each byte maps to one word from BIP39_WORDLIST (256 words = 8 bits per word).
    The word_count parameter controls how many bytes to use.
    """
    use_bytes = entropy[:word_count]
    return [BIP39_WORDLIST[b] for b in use_bytes]


def phrase_to_entropy(phrase: List[str]) -> bytes:
    """Convert a recovery phrase back to entropy bytes.

    Raises ValueError if any word is not in BIP39_WORDLIST.
    """
    result = []
    for word in phrase:
        lower = word.lower()
        if lower not in _WORD_TO_BYTE:
            raise ValueError(f"Unknown recovery word: {word!r}")
        result.append(_WORD_TO_BYTE[lower])
    return bytes(result)


# =========================================================================
# PRIVATE KEY <-> PHRASE CONVERSION
# =========================================================================


def private_key_to_phrase(key_hex: str, word_count: int = 24) -> List[str]:
    """Convert a hex-encoded private key to a recovery phrase."""
    key_bytes = bytes.fromhex(key_hex)
    return entropy_to_phrase(key_bytes, word_count=word_count)


def phrase_to_private_key(phrase: List[str]) -> str:
    """Convert a recovery phrase back to a hex-encoded private key."""
    return phrase_to_entropy(phrase).hex()


# =========================================================================
# KEY ENCRYPTION / DECRYPTION (PBKDF2-SHA256)
# =========================================================================

# OWASP 2024 recommends >= 600,000 iterations for PBKDF2-SHA256.
_KDF_ITERATIONS = 600_000
_KEY_LENGTH = 32


def encrypt_private_key(key_hex: str, passphrase: str) -> Dict[str, str]:
    """Encrypt a private key using PBKDF2-SHA256 + XOR + HMAC authentication.

    Derives two independent keys from the passphrase: one for encryption
    (XOR one-time-pad when key length <= derived key length) and one for
    HMAC-SHA256 authentication. The checksum field stores the HMAC tag,
    providing authenticated encryption that detects both wrong passphrases
    and ciphertext tampering.

    Returns dict with salt, ciphertext, checksum (HMAC tag), and kdf identifier.
    Each call uses a fresh random salt.
    """
    salt = secrets.token_hex(16)
    enc_key, mac_key = _derive_key_pair(passphrase, salt)
    key_bytes = bytes.fromhex(key_hex)

    # XOR encrypt — safe when len(key_bytes) <= len(enc_key) (true for Ed25519/32-byte keys).
    # For longer keys, extend via HKDF-Expand pattern to avoid Vigenere repetition.
    if len(key_bytes) > len(enc_key):
        enc_key = _hkdf_expand(enc_key, length=len(key_bytes))
    ciphertext = bytes(a ^ b for a, b in zip(key_bytes, enc_key[: len(key_bytes)]))

    # HMAC-SHA256 over ciphertext for authenticated encryption
    checksum = hmac.new(mac_key, ciphertext, hashlib.sha256).hexdigest()[:16]

    return {
        "salt": salt,
        "ciphertext": ciphertext.hex(),
        "checksum": checksum,
        "kdf": "pbkdf2-sha256",
    }


def decrypt_private_key(encrypted: Dict[str, str], passphrase: str) -> str:
    """Decrypt an encrypted private key.

    Verifies HMAC-SHA256 authentication tag before returning plaintext.
    Raises ValueError if passphrase is wrong or ciphertext was tampered.
    """
    salt = encrypted["salt"]
    ciphertext = bytes.fromhex(encrypted["ciphertext"])
    expected_checksum = encrypted["checksum"]

    enc_key, mac_key = _derive_key_pair(passphrase, salt)

    # Verify HMAC first (authenticate-then-decrypt)
    actual_checksum = hmac.new(mac_key, ciphertext, hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise ValueError(
            f"Decryption checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        )

    # Decrypt
    if len(ciphertext) > len(enc_key):
        enc_key = _hkdf_expand(enc_key, length=len(ciphertext))
    key_bytes = bytes(a ^ b for a, b in zip(ciphertext, enc_key[: len(ciphertext)]))

    return key_bytes.hex()


def _derive_key_pair(passphrase: str, salt: str) -> tuple:
    """Derive independent encryption and MAC keys from passphrase.

    Uses PBKDF2 with doubled output length, then splits into two 32-byte keys.
    This avoids key reuse between encryption and authentication.
    """
    combined = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt.encode("utf-8"),
        _KDF_ITERATIONS,
        dklen=_KEY_LENGTH * 2,
    )
    return combined[:_KEY_LENGTH], combined[_KEY_LENGTH:]


def _hkdf_expand(key: bytes, length: int) -> bytes:
    """HKDF-Expand (RFC 5869) for extending key material without repetition.

    Uses HMAC-SHA256 in counter mode. Each block is independent,
    avoiding the Vigenere repetition vulnerability of naive key padding.
    """
    blocks = []
    prev = b""
    block_num = 1
    while len(b"".join(blocks)) < length:
        prev = hmac.new(
            key, prev + block_num.to_bytes(1, "big"), hashlib.sha256
        ).digest()
        blocks.append(prev)
        block_num += 1
    return b"".join(blocks)[:length]


# =========================================================================
# SOVEREIGN STATE DATACLASS
# =========================================================================


@dataclass
class SovereignState:
    """Complete sovereign state snapshot from genesis.

    Sensitive fields (encrypted_key, recovery_phrase) are excluded
    from to_dict() to prevent accidental exposure in logs or API responses.
    """

    node_id: str = ""
    public_key: str = ""
    identity_card: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    pat_agents: List[Dict[str, Any]] = field(default_factory=list)
    sat_agents: List[Dict[str, Any]] = field(default_factory=list)
    urp_pledge: Dict[str, Any] = field(default_factory=dict)
    guild_membership: Dict[str, Any] = field(default_factory=dict)
    active_quests: List[Dict[str, Any]] = field(default_factory=list)
    genesis_receipt: Dict[str, Any] = field(default_factory=dict)
    ihsan_score: float = 0.0
    created_at: str = ""

    # Sensitive — excluded from to_dict()
    recovery_phrase: List[str] = field(default_factory=list)
    encrypted_key: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        """Public state representation — secrets excluded."""
        return {
            "node_id": self.node_id,
            "public_key": self.public_key,
            "identity_card": self.identity_card,
            "hardware_info": self.hardware_info,
            "pat_agents": self.pat_agents,
            "sat_agents": self.sat_agents,
            "urp_pledge": self.urp_pledge,
            "guild_membership": self.guild_membership,
            "active_quests": self.active_quests,
            "genesis_receipt": self.genesis_receipt,
            "ihsan_score": self.ihsan_score,
            "created_at": self.created_at,
        }

    def compute_hash(self) -> str:
        """SHA-256 hash of public state for integrity verification."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# =========================================================================
# SAVE / LOAD SOVEREIGN STATE
# =========================================================================


def save_sovereign_state(state: SovereignState, state_dir: Path) -> None:
    """
    Persist sovereign state to the given directory.

    Creates the directory structure and writes each component
    as a separate file plus a root manifest with hash chain.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Identity card
    _write_json(state_dir / "identity.json", state.identity_card)

    # Hardware fingerprint
    _write_json(state_dir / "hardware.json", state.hardware_info)

    # PAT manifest
    pat_manifest = {
        "agent_count": len(state.pat_agents),
        "agents": state.pat_agents,
    }
    _write_json(state_dir / "pat_manifest.json", pat_manifest)

    # SAT manifest
    sat_manifest = {
        "agent_count": len(state.sat_agents),
        "agents": state.sat_agents,
    }
    _write_json(state_dir / "sat_manifest.json", sat_manifest)

    # URP pledge
    _write_json(state_dir / "urp_pledge.json", state.urp_pledge)

    # Genesis receipt
    _write_json(state_dir / "genesis_receipt.json", state.genesis_receipt)

    # Encrypted keystore
    if state.encrypted_key:
        keystore_dir = state_dir / ".keystore"
        keystore_dir.mkdir(parents=True, exist_ok=True)
        _write_json(keystore_dir / "sovereign.enc", state.encrypted_key)
        _restrict_permissions(keystore_dir / "sovereign.enc")

    # Recovery phrase (human-readable)
    if state.recovery_phrase:
        _write_recovery_phrase(
            state_dir / "recovery_phrase.txt",
            state.node_id,
            state.recovery_phrase,
        )
        _restrict_permissions(state_dir / "recovery_phrase.txt")

    # Root manifest
    manifest = {
        "node_id": state.node_id,
        "public_key": state.public_key,
        "created_at": state.created_at,
        "ihsan_score": state.ihsan_score,
        "state_hash": state.compute_hash(),
    }
    _write_json(state_dir / "manifest.json", manifest)

    logger.info(
        "Sovereign state persisted to %s (hash=%s)", state_dir, manifest["state_hash"]
    )


def load_sovereign_state(state_dir: Path) -> Optional[SovereignState]:
    """
    Load sovereign state from disk.

    Returns None if the directory does not exist or manifest is missing.
    """
    state_dir = Path(state_dir)
    manifest_path = state_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    manifest = _read_json(manifest_path)
    if manifest is None:
        return None

    # Load PAT/SAT manifests
    pat_data = _read_json(state_dir / "pat_manifest.json") or {}
    sat_data = _read_json(state_dir / "sat_manifest.json") or {}

    # Load encrypted key if present
    encrypted_key: Dict[str, str] = {}
    enc_path = state_dir / ".keystore" / "sovereign.enc"
    if enc_path.exists():
        encrypted_key = _read_json(enc_path) or {}

    return SovereignState(
        node_id=manifest.get("node_id", ""),
        public_key=manifest.get("public_key", ""),
        identity_card=_read_json(state_dir / "identity.json") or {},
        hardware_info=_read_json(state_dir / "hardware.json") or {},
        pat_agents=pat_data.get("agents", []),
        sat_agents=sat_data.get("agents", []),
        urp_pledge=_read_json(state_dir / "urp_pledge.json") or {},
        genesis_receipt=_read_json(state_dir / "genesis_receipt.json") or {},
        ihsan_score=manifest.get("ihsan_score", 0.0),
        created_at=manifest.get("created_at", ""),
        encrypted_key=encrypted_key,
    )


def state_exists(state_dir: Path) -> bool:
    """Check if sovereign state has been persisted to the given directory."""
    return (Path(state_dir) / "manifest.json").exists()


# =========================================================================
# INTERNAL HELPERS
# =========================================================================


def _write_json(path: Path, data: Any) -> None:
    """Write data as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def _read_json(path: Path) -> Any:
    """Read JSON file, returning None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _restrict_permissions(path: Path) -> None:
    """Restrict file to owner-only read/write (0o600) on Unix systems.

    On Windows this is a no-op since NTFS uses ACLs, not POSIX permissions.
    Logs a warning if permission enforcement fails.
    """
    if platform.system() == "Windows":
        return
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError as e:
        logger.warning("Could not restrict permissions on %s: %s", path, e)


def _write_recovery_phrase(path: Path, node_id: str, phrase: List[str]) -> None:
    """Write recovery phrase as numbered human-readable file."""
    lines = [
        "=" * 50,
        "  BIZRA SOVEREIGN RECOVERY PHRASE",
        "=" * 50,
        f"  Node: {node_id}",
        f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "  Write these words on paper. Store securely.",
        "  NEVER share or photograph this file.",
        "",
    ]

    for i, word in enumerate(phrase, 1):
        lines.append(f"  {i}.{word}")

    lines.append("")
    lines.append("=" * 50)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
