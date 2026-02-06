"""
Tamper-Evident Audit Log — Cryptographic Integrity for Audit Trails
=====================================================================

Provides tamper-evident logging with HMAC signatures and hash chain linking.
Each entry is cryptographically bound to its predecessor, enabling detection
of any tampering, insertion, or deletion of log entries.

Features:
- HMAC-SHA256 signature per entry (authenticity)
- Hash chain linking (integrity)
- Nanosecond precision timestamps
- Sequence number monotonicity
- Key rotation support
- Full chain verification

Standing on Giants:
- Merkle (1979): Hash chains for data integrity
- RFC 2104 (1997): HMAC keyed-hashing for message authentication
- Haber & Stornetta (1991): Timestamping digital documents
- Lamport (1979): Time, clocks, and ordering of events

Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple

from core.integration.constants import (
    UNIFIED_CLOCK_SKEW_SECONDS,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Domain separator for HMAC to prevent cross-protocol attacks
HMAC_DOMAIN_PREFIX: Final[str] = "bizra-audit-v1:"

# Hash algorithm used for chain linking
CHAIN_HASH_ALGO: Final[str] = "sha256"

# Genesis block hash (used for first entry in chain)
GENESIS_HASH: Final[str] = "0" * 64

# Key derivation iteration count (PBKDF2)
KEY_DERIVATION_ITERATIONS: Final[int] = 100_000

# Minimum key length in bytes
MIN_KEY_LENGTH: Final[int] = 32

# Maximum allowed timestamp drift (nanoseconds)
MAX_TIMESTAMP_DRIFT_NS: Final[int] = UNIFIED_CLOCK_SKEW_SECONDS * 1_000_000_000


# =============================================================================
# ENUMS
# =============================================================================


class VerificationStatus(Enum):
    """Status of entry or chain verification."""

    VALID = "valid"
    INVALID_HMAC = "invalid_hmac"
    INVALID_CHAIN = "invalid_chain"
    INVALID_CONTENT_HASH = "invalid_content_hash"
    INVALID_SEQUENCE = "invalid_sequence"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MISSING_PREVIOUS = "missing_previous"
    CORRUPTED = "corrupted"


class TamperType(Enum):
    """Type of tampering detected."""

    CONTENT_MODIFIED = "content_modified"
    ENTRY_DELETED = "entry_deleted"
    ENTRY_INSERTED = "entry_inserted"
    CHAIN_BROKEN = "chain_broken"
    TIMESTAMP_ANOMALY = "timestamp_anomaly"
    SEQUENCE_GAP = "sequence_gap"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TamperEvidentEntry:
    """
    Tamper-evident audit log entry with cryptographic integrity.

    Each entry contains:
    - sequence: Monotonically increasing sequence number
    - timestamp_ns: Nanosecond-precision UNIX timestamp
    - content: The actual log data (JSON-serializable dict)
    - content_hash: SHA-256 hash of canonicalized content
    - prev_hash: Hash of the previous entry (chain link)
    - hmac_signature: HMAC-SHA256 signature binding all fields

    Standing on Giants:
    - Merkle (1979): Hash chains for tamper evidence
    - RFC 2104 (1997): HMAC specification
    """

    sequence: int
    timestamp_ns: int
    content: Dict[str, Any]
    content_hash: str  # SHA-256 of canonicalized content
    prev_hash: str  # Hash of previous entry (chain link)
    hmac_signature: str  # HMAC-SHA256 with secret key

    def verify(
        self,
        secret_key: bytes,
        prev_entry: Optional["TamperEvidentEntry"] = None,
    ) -> VerificationStatus:
        """
        Verify this entry's integrity.

        Args:
            secret_key: The HMAC key (or derived session key)
            prev_entry: The previous entry in the chain (None for genesis)

        Returns:
            VerificationStatus indicating validity
        """
        # 1. Verify content hash
        computed_content_hash = _compute_content_hash(self.content)
        if not hmac.compare_digest(computed_content_hash, self.content_hash):
            return VerificationStatus.INVALID_CONTENT_HASH

        # 2. Verify HMAC signature
        expected_hmac = _compute_entry_hmac(
            sequence=self.sequence,
            timestamp_ns=self.timestamp_ns,
            content_hash=self.content_hash,
            prev_hash=self.prev_hash,
            secret_key=secret_key,
        )
        if not hmac.compare_digest(expected_hmac, self.hmac_signature):
            return VerificationStatus.INVALID_HMAC

        # 3. Verify chain link
        if prev_entry is None:
            # Genesis entry must link to GENESIS_HASH
            if self.prev_hash != GENESIS_HASH:
                return VerificationStatus.INVALID_CHAIN
            if self.sequence != 0:
                return VerificationStatus.INVALID_SEQUENCE
        else:
            # Verify link to previous entry
            expected_prev_hash = _compute_entry_hash(prev_entry)
            if not hmac.compare_digest(self.prev_hash, expected_prev_hash):
                return VerificationStatus.INVALID_CHAIN

            # Verify sequence monotonicity
            if self.sequence != prev_entry.sequence + 1:
                return VerificationStatus.INVALID_SEQUENCE

            # Verify timestamp monotonicity (with tolerance for clock skew)
            if self.timestamp_ns < prev_entry.timestamp_ns:
                # Allow small backwards drift due to clock adjustments
                drift = prev_entry.timestamp_ns - self.timestamp_ns
                if drift > MAX_TIMESTAMP_DRIFT_NS:
                    return VerificationStatus.INVALID_TIMESTAMP

        return VerificationStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "content": self.content,
            "content_hash": self.content_hash,
            "prev_hash": self.prev_hash,
            "hmac_signature": self.hmac_signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TamperEvidentEntry":
        """Deserialize entry from dictionary."""
        return cls(
            sequence=data["sequence"],
            timestamp_ns=data["timestamp_ns"],
            content=data["content"],
            content_hash=data["content_hash"],
            prev_hash=data["prev_hash"],
            hmac_signature=data["hmac_signature"],
        )

    @property
    def timestamp_datetime(self) -> datetime:
        """Convert timestamp to datetime object."""
        return datetime.fromtimestamp(
            self.timestamp_ns / 1_000_000_000,
            tz=timezone.utc,
        )


@dataclass
class TamperingReport:
    """Report of detected tampering in audit log."""

    is_tampered: bool
    tamper_type: Optional[TamperType]
    affected_sequences: List[int]
    first_invalid_sequence: Optional[int]
    details: str
    verified_count: int
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "is_tampered": self.is_tampered,
            "tamper_type": self.tamper_type.value if self.tamper_type else None,
            "affected_sequences": self.affected_sequences,
            "first_invalid_sequence": self.first_invalid_sequence,
            "details": self.details,
            "verified_count": self.verified_count,
            "total_count": self.total_count,
            "integrity_ratio": (
                self.verified_count / self.total_count if self.total_count > 0 else 0.0
            ),
        }


@dataclass
class KeyRotationEvent:
    """Record of a key rotation event."""

    old_key_id: str
    new_key_id: str
    rotation_timestamp_ns: int
    sequence_at_rotation: int
    reason: str = "scheduled"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "old_key_id": self.old_key_id,
            "new_key_id": self.new_key_id,
            "rotation_timestamp_ns": self.rotation_timestamp_ns,
            "sequence_at_rotation": self.sequence_at_rotation,
            "reason": self.reason,
        }


# =============================================================================
# KEY MANAGEMENT
# =============================================================================


@dataclass
class AuditKeyManager:
    """
    Secure key management for audit log HMAC operations.

    Features:
    - Key derivation from master secret
    - Per-session key derivation
    - Key rotation with audit trail
    - Secure key storage interface

    Standing on Giants:
    - NIST SP 800-132: Key Derivation Functions
    - RFC 5869: HKDF specification
    """

    _master_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    _key_id: str = field(default_factory=lambda: secrets.token_hex(8))
    _rotation_history: List[KeyRotationEvent] = field(default_factory=list)
    _derived_keys: Dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self):
        """Validate key length."""
        if len(self._master_key) < MIN_KEY_LENGTH:
            raise ValueError(f"Master key must be at least {MIN_KEY_LENGTH} bytes")

    @property
    def key_id(self) -> str:
        """Current key identifier."""
        return self._key_id

    def get_signing_key(self) -> bytes:
        """Get the current signing key."""
        return self._master_key

    def derive_session_key(self, session_id: str) -> bytes:
        """
        Derive a session-specific key from master key.

        Uses HKDF-like derivation for session isolation.

        Args:
            session_id: Unique session identifier

        Returns:
            32-byte derived key
        """
        if session_id in self._derived_keys:
            return self._derived_keys[session_id]

        # Domain separation for session keys
        info = f"bizra-audit-session:{session_id}".encode("utf-8")

        # Simple HKDF-Extract-Expand using HMAC-SHA256
        prk = hmac.new(
            self._master_key,
            info,
            hashlib.sha256,
        ).digest()

        # Store for reuse
        self._derived_keys[session_id] = prk
        return prk

    def rotate_key(
        self,
        new_key: Optional[bytes] = None,
        reason: str = "scheduled",
        current_sequence: int = 0,
    ) -> KeyRotationEvent:
        """
        Rotate the master key.

        Args:
            new_key: New key (generated if None)
            reason: Reason for rotation
            current_sequence: Current log sequence number

        Returns:
            KeyRotationEvent recording the rotation
        """
        old_key_id = self._key_id

        # Generate new key if not provided
        if new_key is None:
            new_key = secrets.token_bytes(32)

        if len(new_key) < MIN_KEY_LENGTH:
            raise ValueError(f"New key must be at least {MIN_KEY_LENGTH} bytes")

        # Create rotation event
        event = KeyRotationEvent(
            old_key_id=old_key_id,
            new_key_id=secrets.token_hex(8),
            rotation_timestamp_ns=time.time_ns(),
            sequence_at_rotation=current_sequence,
            reason=reason,
        )

        # Update key state
        self._master_key = new_key
        self._key_id = event.new_key_id
        self._derived_keys.clear()  # Invalidate derived keys
        self._rotation_history.append(event)

        return event

    def get_rotation_history(self) -> List[KeyRotationEvent]:
        """Get key rotation history."""
        return list(self._rotation_history)

    @classmethod
    def from_hex(cls, key_hex: str, key_id: Optional[str] = None) -> "AuditKeyManager":
        """Create manager from hex-encoded key."""
        key_bytes = bytes.fromhex(key_hex)
        return cls(
            _master_key=key_bytes,
            _key_id=key_id or secrets.token_hex(8),
        )

    def export_key_hex(self) -> str:
        """Export current key as hex (for secure backup)."""
        return self._master_key.hex()


# =============================================================================
# TAMPER-EVIDENT LOG
# =============================================================================


class TamperEvidentLog:
    """
    Tamper-evident audit log with cryptographic integrity.

    Provides append-only logging with:
    - HMAC authentication per entry
    - Hash chain for integrity verification
    - Full chain verification
    - Tampering detection

    Usage:
        key_manager = AuditKeyManager()
        log = TamperEvidentLog(key_manager)

        # Append entries
        entry = log.append({"event": "login", "user": "alice"})

        # Verify single entry
        status = log.verify_entry(entry)

        # Verify entire chain
        report = log.detect_tampering()

    Standing on Giants:
    - Haber & Stornetta (1991): Timestamping digital documents
    - Lamport (1979): Logical clocks and causality
    """

    def __init__(
        self,
        key_manager: AuditKeyManager,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize tamper-evident log.

        Args:
            key_manager: Key manager for HMAC operations
            persist_path: Optional path for log persistence
        """
        self._key_manager = key_manager
        self._persist_path = persist_path
        self._entries: List[TamperEvidentEntry] = []
        self._last_hash: str = GENESIS_HASH
        self._next_sequence: int = 0

        # Load existing entries if persistence path provided
        if persist_path and persist_path.exists():
            self._load_from_disk()

    def append(
        self,
        content: Dict[str, Any],
        timestamp_ns: Optional[int] = None,
    ) -> TamperEvidentEntry:
        """
        Append a new entry to the log.

        Args:
            content: Log entry content (must be JSON-serializable)
            timestamp_ns: Optional timestamp (current time if None)

        Returns:
            The created TamperEvidentEntry
        """
        # Generate timestamp if not provided
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        # Compute content hash
        content_hash = _compute_content_hash(content)

        # Get signing key
        secret_key = self._key_manager.get_signing_key()

        # Compute HMAC signature
        hmac_sig = _compute_entry_hmac(
            sequence=self._next_sequence,
            timestamp_ns=timestamp_ns,
            content_hash=content_hash,
            prev_hash=self._last_hash,
            secret_key=secret_key,
        )

        # Create entry
        entry = TamperEvidentEntry(
            sequence=self._next_sequence,
            timestamp_ns=timestamp_ns,
            content=content,
            content_hash=content_hash,
            prev_hash=self._last_hash,
            hmac_signature=hmac_sig,
        )

        # Update state
        self._entries.append(entry)
        self._last_hash = _compute_entry_hash(entry)
        self._next_sequence += 1

        # Persist if enabled
        if self._persist_path:
            self._persist_entry(entry)

        return entry

    def verify_entry(
        self,
        entry: TamperEvidentEntry,
        secret_key: Optional[bytes] = None,
    ) -> VerificationStatus:
        """
        Verify a single entry's integrity.

        Args:
            entry: The entry to verify
            secret_key: Optional key override (uses manager's key if None)

        Returns:
            VerificationStatus
        """
        if secret_key is None:
            secret_key = self._key_manager.get_signing_key()

        # Find previous entry
        prev_entry = None
        if entry.sequence > 0:
            for e in self._entries:
                if e.sequence == entry.sequence - 1:
                    prev_entry = e
                    break

            if prev_entry is None:
                return VerificationStatus.MISSING_PREVIOUS

        return entry.verify(secret_key, prev_entry)

    def verify_chain(
        self,
        entries: Optional[List[TamperEvidentEntry]] = None,
        secret_key: Optional[bytes] = None,
    ) -> Tuple[bool, List[Tuple[int, VerificationStatus]]]:
        """
        Verify integrity of entire chain or subset.

        Args:
            entries: Optional specific entries to verify (uses log entries if None)
            secret_key: Optional key override

        Returns:
            Tuple of (all_valid, list of (sequence, status) for invalid entries)
        """
        if entries is None:
            entries = self._entries

        if secret_key is None:
            secret_key = self._key_manager.get_signing_key()

        if not entries:
            return True, []

        invalid_entries: List[Tuple[int, VerificationStatus]] = []
        prev_entry: Optional[TamperEvidentEntry] = None

        for entry in sorted(entries, key=lambda e: e.sequence):
            status = entry.verify(secret_key, prev_entry)
            if status != VerificationStatus.VALID:
                invalid_entries.append((entry.sequence, status))
            prev_entry = entry

        return len(invalid_entries) == 0, invalid_entries

    def detect_tampering(
        self,
        entries: Optional[List[TamperEvidentEntry]] = None,
        secret_key: Optional[bytes] = None,
    ) -> TamperingReport:
        """
        Detect tampering in the log and classify the type.

        Args:
            entries: Optional specific entries to check
            secret_key: Optional key override

        Returns:
            TamperingReport with detailed findings
        """
        if entries is None:
            entries = self._entries

        if secret_key is None:
            secret_key = self._key_manager.get_signing_key()

        total_count = len(entries)
        if total_count == 0:
            return TamperingReport(
                is_tampered=False,
                tamper_type=None,
                affected_sequences=[],
                first_invalid_sequence=None,
                details="Empty log - no entries to verify",
                verified_count=0,
                total_count=0,
            )

        # Sort by sequence
        sorted_entries = sorted(entries, key=lambda e: e.sequence)

        affected_sequences: List[int] = []
        first_invalid: Optional[int] = None
        tamper_type: Optional[TamperType] = None
        prev_entry: Optional[TamperEvidentEntry] = None
        verified_count = 0

        for entry in sorted_entries:
            status = entry.verify(secret_key, prev_entry)

            if status == VerificationStatus.VALID:
                verified_count += 1
            else:
                affected_sequences.append(entry.sequence)
                if first_invalid is None:
                    first_invalid = entry.sequence

                    # Classify tampering type
                    if status == VerificationStatus.INVALID_CONTENT_HASH:
                        tamper_type = TamperType.CONTENT_MODIFIED
                    elif status == VerificationStatus.INVALID_CHAIN:
                        tamper_type = TamperType.CHAIN_BROKEN
                    elif status == VerificationStatus.INVALID_SEQUENCE:
                        tamper_type = TamperType.SEQUENCE_GAP
                    elif status == VerificationStatus.INVALID_TIMESTAMP:
                        tamper_type = TamperType.TIMESTAMP_ANOMALY
                    else:
                        tamper_type = TamperType.CONTENT_MODIFIED

            # Check for sequence gaps (entry deletion)
            if prev_entry is not None:
                expected_seq = prev_entry.sequence + 1
                if entry.sequence > expected_seq:
                    if first_invalid is None:
                        first_invalid = expected_seq
                    tamper_type = TamperType.ENTRY_DELETED
                    for missing_seq in range(expected_seq, entry.sequence):
                        if missing_seq not in affected_sequences:
                            affected_sequences.append(missing_seq)

            prev_entry = entry

        is_tampered = len(affected_sequences) > 0

        return TamperingReport(
            is_tampered=is_tampered,
            tamper_type=tamper_type,
            affected_sequences=sorted(affected_sequences),
            first_invalid_sequence=first_invalid,
            details=self._generate_tamper_details(
                is_tampered, tamper_type, affected_sequences, first_invalid
            ),
            verified_count=verified_count,
            total_count=total_count,
        )

    def _generate_tamper_details(
        self,
        is_tampered: bool,
        tamper_type: Optional[TamperType],
        affected: List[int],
        first_invalid: Optional[int],
    ) -> str:
        """Generate human-readable tampering details."""
        if not is_tampered:
            return "Chain integrity verified - no tampering detected"

        details = [f"TAMPERING DETECTED at sequence {first_invalid}"]

        if tamper_type:
            type_descriptions = {
                TamperType.CONTENT_MODIFIED: "Entry content was modified after signing",
                TamperType.ENTRY_DELETED: "One or more entries were deleted from the chain",
                TamperType.ENTRY_INSERTED: "Entries were inserted into the chain",
                TamperType.CHAIN_BROKEN: "Hash chain link is broken",
                TamperType.TIMESTAMP_ANOMALY: "Timestamp ordering violation detected",
                TamperType.SEQUENCE_GAP: "Sequence number gap indicates missing entries",
            }
            details.append(f"Type: {type_descriptions.get(tamper_type, 'Unknown')}")

        details.append(f"Affected entries: {len(affected)}")
        if len(affected) <= 10:
            details.append(f"Sequences: {affected}")
        else:
            details.append(f"First 10 sequences: {affected[:10]}...")

        return "; ".join(details)

    def get_entry(self, sequence: int) -> Optional[TamperEvidentEntry]:
        """Get entry by sequence number."""
        for entry in self._entries:
            if entry.sequence == sequence:
                return entry
        return None

    def get_entries(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> List[TamperEvidentEntry]:
        """Get entries in sequence range."""
        result = []
        for entry in self._entries:
            if entry.sequence >= start_sequence:
                if end_sequence is None or entry.sequence <= end_sequence:
                    result.append(entry)
        return sorted(result, key=lambda e: e.sequence)

    def __len__(self) -> int:
        """Number of entries in log."""
        return len(self._entries)

    def __iter__(self) -> Iterator[TamperEvidentEntry]:
        """Iterate over entries in sequence order."""
        return iter(sorted(self._entries, key=lambda e: e.sequence))

    @property
    def last_entry(self) -> Optional[TamperEvidentEntry]:
        """Get the most recent entry."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.sequence)

    @property
    def last_hash(self) -> str:
        """Get hash of the last entry (or GENESIS_HASH if empty)."""
        return self._last_hash

    def _persist_entry(self, entry: TamperEvidentEntry) -> None:
        """Persist a single entry to disk."""
        if not self._persist_path:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with open(self._persist_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _load_from_disk(self) -> None:
        """Load entries from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        with open(self._persist_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entry = TamperEvidentEntry.from_dict(data)
                    self._entries.append(entry)

        # Rebuild state
        if self._entries:
            self._entries.sort(key=lambda e: e.sequence)
            last_entry = self._entries[-1]
            self._last_hash = _compute_entry_hash(last_entry)
            self._next_sequence = last_entry.sequence + 1

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export entire chain as list of dictionaries."""
        return [
            entry.to_dict() for entry in sorted(self._entries, key=lambda e: e.sequence)
        ]

    def import_chain(
        self,
        chain_data: List[Dict[str, Any]],
        verify: bool = True,
    ) -> Tuple[int, List[Tuple[int, VerificationStatus]]]:
        """
        Import entries from exported chain data.

        Args:
            chain_data: List of entry dictionaries
            verify: Whether to verify entries during import

        Returns:
            Tuple of (imported_count, invalid_entries)
        """
        entries = [TamperEvidentEntry.from_dict(d) for d in chain_data]
        invalid: List[Tuple[int, VerificationStatus]] = []

        if verify:
            secret_key = self._key_manager.get_signing_key()
            is_valid, invalid = self.verify_chain(entries, secret_key)
            if not is_valid:
                return 0, invalid

        # Import entries
        self._entries.extend(entries)
        self._entries.sort(key=lambda e: e.sequence)

        # Rebuild state
        if self._entries:
            last_entry = self._entries[-1]
            self._last_hash = _compute_entry_hash(last_entry)
            self._next_sequence = last_entry.sequence + 1

        return len(entries), invalid


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_content_hash(content: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of canonicalized content.

    Uses RFC 8785 JSON Canonicalization for deterministic serialization.
    """
    canonical = json.dumps(
        content,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def _compute_entry_hash(entry: TamperEvidentEntry) -> str:
    """
    Compute hash of entire entry for chain linking.

    Includes all fields to ensure full entry integrity.
    """
    # Create deterministic representation
    hash_input = (
        f"{entry.sequence}:"
        f"{entry.timestamp_ns}:"
        f"{entry.content_hash}:"
        f"{entry.prev_hash}:"
        f"{entry.hmac_signature}"
    ).encode("utf-8")

    return hashlib.sha256(hash_input).hexdigest()


def _compute_entry_hmac(
    sequence: int,
    timestamp_ns: int,
    content_hash: str,
    prev_hash: str,
    secret_key: bytes,
) -> str:
    """
    Compute HMAC-SHA256 signature for entry.

    Domain-separated to prevent cross-protocol attacks.
    """
    # Create message with domain separation
    message = (
        f"{HMAC_DOMAIN_PREFIX}"
        f"seq={sequence};"
        f"ts={timestamp_ns};"
        f"content={content_hash};"
        f"prev={prev_hash}"
    ).encode("utf-8")

    return hmac.new(secret_key, message, hashlib.sha256).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_audit_log(
    persist_path: Optional[Path] = None,
    key_hex: Optional[str] = None,
) -> Tuple[TamperEvidentLog, AuditKeyManager]:
    """
    Create a new tamper-evident audit log with key manager.

    Args:
        persist_path: Optional path for log persistence
        key_hex: Optional hex-encoded key (generates new if None)

    Returns:
        Tuple of (TamperEvidentLog, AuditKeyManager)
    """
    if key_hex:
        key_manager = AuditKeyManager.from_hex(key_hex)
    else:
        key_manager = AuditKeyManager()

    log = TamperEvidentLog(key_manager, persist_path)
    return log, key_manager


def verify_entry(
    entry: TamperEvidentEntry,
    secret_key: bytes,
    prev_entry: Optional[TamperEvidentEntry] = None,
) -> bool:
    """
    Convenience function to verify a single entry.

    Args:
        entry: Entry to verify
        secret_key: HMAC key
        prev_entry: Previous entry in chain (None for genesis)

    Returns:
        True if valid
    """
    return entry.verify(secret_key, prev_entry) == VerificationStatus.VALID


def verify_chain(
    entries: List[TamperEvidentEntry],
    secret_key: bytes,
) -> bool:
    """
    Convenience function to verify a chain of entries.

    Args:
        entries: List of entries to verify
        secret_key: HMAC key

    Returns:
        True if entire chain is valid
    """
    if not entries:
        return True

    sorted_entries = sorted(entries, key=lambda e: e.sequence)
    prev_entry: Optional[TamperEvidentEntry] = None

    for entry in sorted_entries:
        status = entry.verify(secret_key, prev_entry)
        if status != VerificationStatus.VALID:
            return False
        prev_entry = entry

    return True


def detect_tampering(
    entries: List[TamperEvidentEntry],
    secret_key: bytes,
) -> TamperingReport:
    """
    Convenience function to detect tampering in entries.

    Args:
        entries: List of entries to check
        secret_key: HMAC key

    Returns:
        TamperingReport with findings
    """
    key_manager = AuditKeyManager(_master_key=secret_key)
    log = TamperEvidentLog(key_manager)
    return log.detect_tampering(entries, secret_key)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "TamperEvidentEntry",
    "TamperEvidentLog",
    "AuditKeyManager",
    # Reports
    "TamperingReport",
    "KeyRotationEvent",
    # Enums
    "VerificationStatus",
    "TamperType",
    # Functions
    "create_audit_log",
    "verify_entry",
    "verify_chain",
    "detect_tampering",
    # Constants
    "GENESIS_HASH",
    "HMAC_DOMAIN_PREFIX",
]
