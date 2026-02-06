"""
Deterministic Canonicalization — Lock Replayability

Same input → Same bytes → Same receipt hash

Key principles:
- Stable JSON key ordering
- Whitespace normalization
- Unicode normalization
- BLAKE3 hashing for determinism
"""

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def canonical_json(obj: Any) -> Any:
    """
    Recursively canonicalize a JSON-compatible object.

    - Object keys are sorted alphabetically
    - Arrays maintain order
    - Strings are unicode-normalized (NFC)
    - Numbers are preserved as-is
    """
    if isinstance(obj, dict):
        # Sort keys and recursively canonicalize values
        return {k: canonical_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Preserve order, canonicalize elements
        return [canonical_json(item) for item in obj]
    elif isinstance(obj, str):
        # Unicode normalization + whitespace trim
        return unicodedata.normalize("NFC", obj.strip())
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj
    else:
        # Convert to string for unknown types
        return str(obj)


def canonical_bytes(obj: Any) -> bytes:
    """Convert object to deterministic bytes."""
    canon = canonical_json(obj)
    # Use separators to ensure no whitespace variance
    return json.dumps(
        canon,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def blake3_digest(data: bytes) -> bytes:
    """
    Compute BLAKE3 hash of data.

    Falls back to SHA-256 if BLAKE3 not available.
    """
    try:
        import blake3

        return blake3.blake3(data).digest()
    except ImportError:
        # Fallback to SHA-256
        return hashlib.sha256(data).digest()


def hex_digest(data: bytes) -> str:
    """Compute hex-encoded digest."""
    return blake3_digest(data).hex()


@dataclass
class CanonQuery:
    """
    Canonicalized query structure.

    All fields are normalized for deterministic hashing.
    """

    user_id: str
    user_state: str
    intent: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    nonce: Optional[str] = None

    def __post_init__(self):
        # Normalize string fields
        self.user_id = unicodedata.normalize("NFC", self.user_id.strip())
        self.user_state = unicodedata.normalize("NFC", self.user_state.strip())
        self.intent = unicodedata.normalize("NFC", self.intent.strip())

        # Canonicalize payload
        self.payload = canonical_json(self.payload)

        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        data = {
            "user_id": self.user_id,
            "user_state": self.user_state,
            "intent": self.intent,
            "payload": self.payload,
        }

        # Include timestamp only if reproducibility not required
        # For deterministic replay, omit timestamp
        if self.nonce:
            data["nonce"] = self.nonce

        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Compute BLAKE3 digest of canonical form."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Compute hex-encoded digest."""
        return self.digest().hex()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "user_state": self.user_state,
            "intent": self.intent,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "nonce": self.nonce,
            "digest": self.hex_digest(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonQuery":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            if isinstance(data["timestamp"], str):
                timestamp = datetime.fromisoformat(data["timestamp"])
            else:
                timestamp = data["timestamp"]

        return cls(
            user_id=data.get("user_id", ""),
            user_state=data.get("user_state", ""),
            intent=data.get("intent", ""),
            payload=data.get("payload", {}),
            timestamp=timestamp,
            nonce=data.get("nonce"),
        )


@dataclass
class CanonPolicy:
    """
    Canonicalized policy structure.

    Used for policy hashing in receipts.
    """

    policy_id: str
    version: str
    rules: Dict[str, Any]
    thresholds: Dict[str, float]
    constraints: List[str] = field(default_factory=list)

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        data = {
            "policy_id": self.policy_id,
            "version": self.version,
            "rules": canonical_json(self.rules),
            "thresholds": canonical_json(self.thresholds),
            "constraints": sorted(self.constraints),
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Compute BLAKE3 digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Compute hex-encoded digest."""
        return self.digest().hex()


@dataclass
class CanonEnvironment:
    """
    Environment fingerprint for reproducibility.

    Captures machine/runtime state for audit trail.
    """

    platform: str
    python_version: str
    hostname: str
    cpu_count: int
    memory_gb: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        data = {
            "platform": self.platform,
            "python_version": self.python_version,
            "hostname": self.hostname,
            "cpu_count": self.cpu_count,
            "memory_gb": round(self.memory_gb, 2),
            "extra": canonical_json(self.extra),
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Compute BLAKE3 digest."""
        return blake3_digest(self.canonical_bytes())

    @classmethod
    def capture(cls) -> "CanonEnvironment":
        """Capture current environment."""
        import os
        import platform
        import sys

        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            memory_gb = 0.0

        return cls(
            platform=platform.system(),
            python_version=sys.version.split()[0],
            hostname=platform.node(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=memory_gb,
        )


def verify_determinism(
    query: CanonQuery,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Verify that canonicalization is deterministic.

    Runs N iterations and checks hash equality.
    """
    digests = []

    for _ in range(iterations):
        # Re-create query to test serialization
        data = query.to_dict()
        reconstructed = CanonQuery.from_dict(data)
        digests.append(reconstructed.hex_digest())

    unique = set(digests)

    return {
        "deterministic": len(unique) == 1,
        "iterations": iterations,
        "unique_hashes": len(unique),
        "canonical_digest": digests[0] if digests else None,
    }
