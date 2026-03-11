"""
Shamir Secret Sharing — Social Recovery for Sovereign Identity

Implements k-of-n threshold secret sharing over a prime field for
identity recovery. When a user loses their Ed25519 private key,
k out of n designated guardians can reconstruct it.

Algorithm:
    Split: Generate random polynomial P(x) of degree k-1 where P(0) = secret.
           Evaluate at n distinct points to produce n shares.
    Reconstruct: Given k shares (x_i, y_i), use Lagrange interpolation
                 to recover P(0) = secret.

Security:
    - Fewer than k shares reveal NO information about the secret
      (information-theoretic security)
    - Shares are individually meaningless without threshold cooperation
    - Guardian identities are hashed (privacy-preserving)
    - Recovery ceremony requires multi-party coordination

Standing on Giants:
- Adi Shamir (1979): "How to Share a Secret" — threshold cryptography
- Lagrange (1795): Polynomial interpolation
- Shannon (1949): Information-theoretic secrecy
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Prime for the finite field — a 256-bit safe prime
# This is large enough for Ed25519 private keys (32 bytes = 256 bits)
SHAMIR_PRIME = (1 << 256) - 189  # 2^256 - 189 is prime

# Default threshold parameters
DEFAULT_THRESHOLD = 3  # k: minimum shares to reconstruct
DEFAULT_TOTAL_SHARES = 5  # n: total shares distributed

# Maximum supported values
MAX_SHARES = 255
MAX_THRESHOLD = 255

# Recovery ceremony timeout (seconds)
CEREMONY_TIMEOUT_SECONDS = 3600  # 1 hour


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _extended_gcd(b % a, a)
    return gcd, y1 - (b // a) * x1, x1


def _mod_inverse(a: int, prime: int) -> int:
    """Compute modular multiplicative inverse using extended GCD."""
    _, inv, _ = _extended_gcd(a % prime, prime)
    return inv % prime


def _eval_poly(coeffs: List[int], x: int, prime: int) -> int:
    """Evaluate polynomial with given coefficients at point x over prime field."""
    result = 0
    power = 1
    for coeff in coeffs:
        result = (result + coeff * power) % prime
        power = (power * x) % prime
    return result


def _lagrange_interpolate(x: int, x_s: List[int], y_s: List[int], prime: int) -> int:
    """
    Lagrange interpolation over a prime field.

    Given k points (x_s[i], y_s[i]), recover the polynomial value at x.

    Standing on Giants: Lagrange (1795) — polynomial interpolation
    """
    k = len(x_s)
    if k != len(y_s):
        raise ValueError("x_s and y_s must have the same length")

    result = 0
    for j in range(k):
        num = 1
        den = 1
        for m in range(k):
            if m != j:
                num = (num * (x - x_s[m])) % prime
                den = (den * (x_s[j] - x_s[m])) % prime
        inv = _mod_inverse(den, prime)
        result = (result + y_s[j] * num * inv) % prime
    return result


@dataclass
class Share:
    """A single share of a split secret."""

    index: int  # x-coordinate (1-based)
    value: int  # y-coordinate (P(index))
    threshold: int  # k: minimum shares needed
    total_shares: int  # n: total shares created
    share_hash: str = ""  # SHA-256 of the share for verification

    def __post_init__(self):
        if not self.share_hash:
            h = hashlib.sha256()
            h.update(f"{self.index}:{self.value}".encode())
            self.share_hash = h.hexdigest()

    def to_hex(self) -> str:
        """Encode share value as hex string."""
        byte_len = (self.value.bit_length() + 7) // 8
        return self.value.to_bytes(max(byte_len, 1), "big").hex()

    @classmethod
    def from_hex(
        cls, index: int, hex_value: str, threshold: int, total_shares: int
    ) -> "Share":
        """Create share from hex-encoded value."""
        value = int.from_bytes(bytes.fromhex(hex_value), "big")
        return cls(
            index=index, value=value, threshold=threshold, total_shares=total_shares
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "value_hex": self.to_hex(),
            "threshold": self.threshold,
            "total_shares": self.total_shares,
            "share_hash": self.share_hash,
        }


@dataclass
class Guardian:
    """A designated recovery guardian."""

    guardian_id: str  # Hashed identifier (privacy-preserving)
    display_name: str
    share_index: int  # Which share this guardian holds
    registered_at: str = ""
    contact_hash: str = ""  # SHA-256 of contact info

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guardian_id": self.guardian_id,
            "display_name": self.display_name,
            "share_index": self.share_index,
            "registered_at": self.registered_at,
            "contact_hash": self.contact_hash,
        }


class ShamirSplitter:
    """
    Split a secret into n shares with k-of-n threshold.

    The secret is treated as an integer in GF(p) where p is SHAMIR_PRIME.
    A random polynomial of degree k-1 is generated with the secret as
    the constant term. Shares are evaluations at x = 1, 2, ..., n.
    """

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        total_shares: int = DEFAULT_TOTAL_SHARES,
    ):
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        if total_shares < threshold:
            raise ValueError("Total shares must be >= threshold")
        if total_shares > MAX_SHARES:
            raise ValueError(f"Total shares must be <= {MAX_SHARES}")

        self.threshold = threshold
        self.total_shares = total_shares

    def split(self, secret_bytes: bytes) -> List[Share]:
        """
        Split a secret into shares.

        Args:
            secret_bytes: The secret to split (e.g., Ed25519 private key, 32 bytes)

        Returns:
            List of n Share objects
        """
        # Convert secret to integer
        secret_int = int.from_bytes(secret_bytes, "big")

        if secret_int >= SHAMIR_PRIME:
            raise ValueError("Secret too large for the prime field")

        # Generate random polynomial coefficients
        # P(x) = secret + a_1*x + a_2*x^2 + ... + a_{k-1}*x^{k-1}
        coeffs = [secret_int]
        for _ in range(self.threshold - 1):
            coeffs.append(secrets.randbelow(SHAMIR_PRIME))

        # Evaluate at x = 1, 2, ..., n
        shares = []
        for i in range(1, self.total_shares + 1):
            y = _eval_poly(coeffs, i, SHAMIR_PRIME)
            shares.append(
                Share(
                    index=i,
                    value=y,
                    threshold=self.threshold,
                    total_shares=self.total_shares,
                )
            )

        logger.info(
            "Secret split into %d shares (threshold=%d)",
            self.total_shares,
            self.threshold,
        )
        return shares


class ShamirReconstructor:
    """Reconstruct a secret from k-of-n shares using Lagrange interpolation."""

    @staticmethod
    def reconstruct(shares: List[Share], secret_length: int = 32) -> bytes:
        """
        Reconstruct the secret from shares.

        Args:
            shares: At least k shares (threshold)
            secret_length: Expected length of the secret in bytes

        Returns:
            The reconstructed secret as bytes

        Raises:
            ValueError: If insufficient shares or inconsistent metadata
        """
        if not shares:
            raise ValueError("No shares provided")

        threshold = shares[0].threshold
        if len(shares) < threshold:
            raise ValueError(f"Need at least {threshold} shares, got {len(shares)}")

        # Verify all shares have consistent metadata
        for share in shares:
            if share.threshold != threshold:
                raise ValueError("Inconsistent threshold across shares")

        # Use exactly threshold shares (first k)
        selected = shares[:threshold]
        x_s = [s.index for s in selected]
        y_s = [s.value for s in selected]

        # Check for duplicate indices
        if len(set(x_s)) != len(x_s):
            raise ValueError("Duplicate share indices detected")

        # Lagrange interpolation at x=0 to recover the secret
        secret_int = _lagrange_interpolate(0, x_s, y_s, SHAMIR_PRIME)

        return secret_int.to_bytes(secret_length, "big")


@dataclass
class GuardianRegistry:
    """Registry of designated recovery guardians for a node."""

    node_id: str
    guardians: List[Guardian] = field(default_factory=list)
    threshold: int = DEFAULT_THRESHOLD
    total_guardians: int = DEFAULT_TOTAL_SHARES

    def register_guardian(
        self,
        display_name: str,
        share_index: int,
        contact_info: str = "",
    ) -> Guardian:
        """
        Register a new guardian.

        Args:
            display_name: Human-readable name
            share_index: Which share this guardian holds
            contact_info: Contact information (hashed for privacy)

        Returns:
            The registered Guardian
        """
        # Generate privacy-preserving guardian ID
        guardian_id = hashlib.sha256(
            f"{self.node_id}:guardian:{display_name}:{share_index}".encode()
        ).hexdigest()[:16]

        # Hash contact info
        contact_hash = ""
        if contact_info:
            contact_hash = hashlib.sha256(contact_info.encode()).hexdigest()

        guardian = Guardian(
            guardian_id=guardian_id,
            display_name=display_name,
            share_index=share_index,
            contact_hash=contact_hash,
        )

        self.guardians.append(guardian)
        logger.info(
            "Guardian registered: %s (share %d) for node %s",
            display_name,
            share_index,
            self.node_id,
        )
        return guardian

    def get_guardian_by_index(self, share_index: int) -> Optional[Guardian]:
        """Find guardian by share index."""
        for g in self.guardians:
            if g.share_index == share_index:
                return g
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "guardians": [g.to_dict() for g in self.guardians],
            "threshold": self.threshold,
            "total_guardians": self.total_guardians,
        }


@dataclass
class RecoveryCeremonyResult:
    """Result of a recovery ceremony."""

    success: bool
    recovered_key: Optional[bytes] = None
    error: Optional[str] = None
    shares_used: int = 0
    ceremony_duration_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )


class RecoveryCeremony:
    """
    Multi-party recovery ceremony using Shamir shares.

    Guardians submit their shares. Once the threshold is met,
    the secret is reconstructed and verified.
    """

    def __init__(self, node_id: str, threshold: int = DEFAULT_THRESHOLD):
        self.node_id = node_id
        self.threshold = threshold
        self._submitted_shares: List[Share] = []
        self._start_time = time.monotonic()
        self._completed = False

    @property
    def shares_collected(self) -> int:
        return len(self._submitted_shares)

    @property
    def shares_needed(self) -> int:
        return max(0, self.threshold - len(self._submitted_shares))

    @property
    def is_ready(self) -> bool:
        return len(self._submitted_shares) >= self.threshold

    def submit_share(self, share: Share) -> bool:
        """
        Submit a share to the ceremony.

        Args:
            share: A valid Share object

        Returns:
            True if the share was accepted
        """
        if self._completed:
            return False

        # Check timeout
        elapsed = time.monotonic() - self._start_time
        if elapsed > CEREMONY_TIMEOUT_SECONDS:
            logger.warning("Recovery ceremony timed out for %s", self.node_id)
            return False

        # Check for duplicate
        existing_indices = {s.index for s in self._submitted_shares}
        if share.index in existing_indices:
            logger.warning("Duplicate share index %d submitted", share.index)
            return False

        self._submitted_shares.append(share)
        logger.info(
            "Share %d submitted for %s (%d/%d)",
            share.index,
            self.node_id,
            self.shares_collected,
            self.threshold,
        )
        return True

    def reconstruct(
        self,
        expected_public_key: Optional[str] = None,
        secret_length: int = 32,
    ) -> RecoveryCeremonyResult:
        """
        Attempt to reconstruct the secret from submitted shares.

        Args:
            expected_public_key: If provided, verify the reconstructed key
                                 derives this public key
            secret_length: Expected byte length of the secret

        Returns:
            RecoveryCeremonyResult with the recovered key or error
        """
        elapsed_ms = (time.monotonic() - self._start_time) * 1000

        if not self.is_ready:
            return RecoveryCeremonyResult(
                success=False,
                error=f"Not enough shares: {self.shares_collected}/{self.threshold}",
                shares_used=self.shares_collected,
                ceremony_duration_ms=elapsed_ms,
            )

        try:
            reconstructor = ShamirReconstructor()
            recovered = reconstructor.reconstruct(
                self._submitted_shares,
                secret_length=secret_length,
            )

            # Optionally verify against expected public key
            if expected_public_key:
                try:
                    from core.pci.crypto import derive_public_key

                    derived_pub = derive_public_key(recovered.hex())
                    if derived_pub != expected_public_key:
                        return RecoveryCeremonyResult(
                            success=False,
                            error="Reconstructed key does not match expected public key",
                            shares_used=self.shares_collected,
                            ceremony_duration_ms=elapsed_ms,
                        )
                except ImportError:
                    logger.warning(
                        "crypto module unavailable — skipping public key verification"
                    )

            self._completed = True
            logger.info(
                "Recovery ceremony succeeded for %s (%d shares, %.1fms)",
                self.node_id,
                self.shares_collected,
                elapsed_ms,
            )

            return RecoveryCeremonyResult(
                success=True,
                recovered_key=recovered,
                shares_used=self.shares_collected,
                ceremony_duration_ms=elapsed_ms,
            )

        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.error("Recovery reconstruction failed: %s", e)
            return RecoveryCeremonyResult(
                success=False,
                error=str(e),
                shares_used=self.shares_collected,
                ceremony_duration_ms=elapsed_ms,
            )


def generate_recovery_shares(
    private_key_hex: str,
    threshold: int = DEFAULT_THRESHOLD,
    total_shares: int = DEFAULT_TOTAL_SHARES,
) -> List[Share]:
    """
    Convenience function to split an Ed25519 private key into recovery shares.

    Args:
        private_key_hex: Hex-encoded Ed25519 private key
        threshold: Minimum shares needed for recovery (k)
        total_shares: Total shares to generate (n)

    Returns:
        List of Share objects to distribute to guardians
    """
    secret_bytes = bytes.fromhex(private_key_hex)
    splitter = ShamirSplitter(threshold=threshold, total_shares=total_shares)
    return splitter.split(secret_bytes)


__all__ = [
    "ShamirSplitter",
    "ShamirReconstructor",
    "Share",
    "Guardian",
    "GuardianRegistry",
    "RecoveryCeremony",
    "RecoveryCeremonyResult",
    "generate_recovery_shares",
    "SHAMIR_PRIME",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOTAL_SHARES",
]
