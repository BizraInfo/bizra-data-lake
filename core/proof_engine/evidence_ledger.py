"""
Evidence Ledger — Append-only, hash-chained, signed JSONL log.

Every verification call, query, and tool execution emits a receipt
into the ledger. Entries are hash-chained (each entry includes the
hash of the previous entry) so tampering is detectable.

Standing on Giants:
- Lamport (1978): Logical clocks and event ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR as information quality
- BIZRA Spearpoint PRD SP-002: "every verification call emits a receipt"
"""

import hashlib
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# G6 FIX: Inter-process file locking (Lamport — strict chronological ordering).
# threading.Lock() only guards within a single process. When node0_activate.py
# runs the proactive loop AND a CLI `mission` command simultaneously, two separate
# processes each hold their own Lock — the seq counter races between them.
# fcntl.flock (POSIX) blocks at the OS level across all processes on the same file.
if sys.platform != "win32":
    import fcntl  # POSIX only — available on WSL/Linux where Node0 runs

    @contextmanager
    def _posix_file_lock(lock_path: Path) -> Generator[None, None, None]:
        """Acquire an exclusive POSIX file lock for the duration of the block."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)  # blocks until no other holder
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

else:
    # Windows fallback — threading.Lock only (no fcntl); concurrent processes
    # on Windows should use a named mutex, but Node0 targets WSL/Linux.
    @contextmanager
    def _posix_file_lock(lock_path: Path) -> Generator[None, None, None]:  # type: ignore[misc]
        yield


from core.proof_engine.schema_validator import validate_receipt
from core.snr_protocol import assert_snr_normalized
from core.sovereign.origin_guard import (
    ROLE_NODE0,
    normalize_node_role,
    resolve_origin_snapshot,
)

logger = logging.getLogger(__name__)

# Sentinel hash for the first entry in a ledger
GENESIS_HASH = "0" * 64


@dataclass
class LedgerEntry:
    """A single entry in the evidence ledger."""

    sequence: int
    receipt: Dict[str, Any]
    prev_hash: str
    entry_hash: str
    timestamp: str

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(
            {
                "seq": self.sequence,
                "receipt": self.receipt,
                "prev_hash": self.prev_hash,
                "entry_hash": self.entry_hash,
                "ts": self.timestamp,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_jsonl(cls, line: str) -> "LedgerEntry":
        """Deserialize from a JSONL line."""
        data = json.loads(line)
        return cls(
            sequence=data["seq"],
            receipt=data["receipt"],
            prev_hash=data["prev_hash"],
            entry_hash=data["entry_hash"],
            timestamp=data["ts"],
        )


def _compute_entry_hash(sequence: int, receipt: Dict[str, Any], prev_hash: str) -> str:
    """Compute BLAKE3 hash of an entry (deterministic, cross-language compatible).

    SEC-001 remediation: migrated from SHA-256 to BLAKE3 for Rust interop.
    Standing on Giants: O'Connor et al. (BLAKE3, 2020)
    """
    canonical = json.dumps(
        {"seq": sequence, "receipt": receipt, "prev_hash": prev_hash},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    from core.proof_engine.canonical import hex_digest

    return hex_digest(canonical)


def _compute_entry_hash_legacy_sha256(
    sequence: int, receipt: Dict[str, Any], prev_hash: str
) -> str:
    """Legacy SHA-256 entry hash used before SEC-001 BLAKE3 migration.

    Keep verification backward-compatible with historical ledgers while
    preserving fail-closed tamper detection semantics.
    """
    canonical = json.dumps(
        {"seq": sequence, "receipt": receipt, "prev_hash": prev_hash},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()  # noqa: S324  # noqa: SEC-001
    return digest


class EvidenceLedger:
    """
    Append-only, hash-chained evidence ledger.

    Thread-safe. Each entry includes:
    - sequence number (monotonic)
    - receipt (schema-validated JSON)
    - prev_hash (chain link to prior entry)
    - entry_hash (BLAKE3 of canonical entry)
    - timestamp (ISO 8601 UTC)
    """

    def __init__(self, path: Path, *, validate_on_append: bool = True):
        self._path = path
        self._validate = validate_on_append
        self._lock = threading.Lock()
        # G6 FIX: .lock file sits next to the ledger — used by _posix_file_lock
        # for cross-process mutual exclusion (fcntl.LOCK_EX on POSIX).
        self._lock_file = path.with_suffix(".lock")
        self._sequence = 0
        self._last_hash = GENESIS_HASH

        # Resume from existing ledger if it exists
        if path.exists() and path.stat().st_size > 0:
            self._resume()

    def _resume(self) -> None:
        """Resume sequence and chain state from existing ledger file."""
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = LedgerEntry.from_jsonl(line)
                self._sequence = entry.sequence
                self._last_hash = entry.entry_hash

    def _resume_tail(self) -> None:
        """Resume seq+hash from the last line only — O(1) disk seek.

        G6 FIX: Called inside the cross-process lock before each append to
        ensure freshness without re-scanning the entire file.
        Uses seek(-N, 2) to read backwards from EOF.
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            return
        with open(self._path, "rb") as f:
            # Walk back from EOF to find the last non-empty JSONL line
            chunk_size = 4096
            f.seek(0, 2)  # seek to EOF
            pos = f.tell()
            buf = b""
            while pos > 0:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                buf = chunk + buf
                # Look for a complete JSONL line from the right
                lines = buf.split(b"\n")
                for raw in reversed(lines):
                    raw = raw.strip()
                    if raw:
                        try:
                            entry = LedgerEntry.from_jsonl(raw.decode("utf-8"))
                            self._sequence = entry.sequence
                            self._last_hash = entry.entry_hash
                            return
                        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                            continue
                # If no complete line yet, keep reading backwards
                if len(lines) > 2:
                    # Keep only the partial first line to prepend next iteration
                    buf = lines[0]

    @property
    def sequence(self) -> int:
        """Current sequence number (0 = empty)."""
        return self._sequence

    @property
    def last_hash(self) -> str:
        """Hash of the most recent entry (or GENESIS_HASH if empty)."""
        return self._last_hash

    def append(self, receipt: Dict[str, Any]) -> LedgerEntry:
        """
        Append a receipt to the ledger.

        G6 FIX (chain fork prevention):
        Uses a two-level lock:
          1. threading.Lock  — fast guard for concurrent threads in the same process
          2. fcntl.flock     — OS-level guard for concurrent processes (proactive loop
                               vs. CLI mission command running simultaneously)
        After acquiring both locks we re-read the last line of the ledger from disk
        to ensure sequence freshness — preventing the "phantom seq=13" scenario where
        two processes each resume at seq=13 and both increment to 14.

        Standing on Giants: Lamport (1978) — strict causal ordering.
        Artifact: core/proof_engine/evidence_ledger.py

        Args:
            receipt: Schema-compliant receipt dict.

        Returns:
            The LedgerEntry that was appended.

        Raises:
            ValueError: If receipt fails schema validation (when validate_on_append=True).
        """
        if self._validate:
            is_valid, errors = validate_receipt(receipt)
            if not is_valid:
                raise ValueError(f"Receipt fails schema validation: {errors}")

            # Phase 60 Step 4: SNR normalization contract enforcement.
            # Any receipt SNR representation MUST be in [0.0, 1.0].
            # Standing on Giants: Meyer (Design by Contract, 1986)
            snr_candidates: list[tuple[str, Any]] = []
            if "snr_score" in receipt:
                snr_candidates.append(("snr_score", receipt.get("snr_score")))
            snr_obj = receipt.get("snr")
            if isinstance(snr_obj, dict) and "score" in snr_obj:
                snr_candidates.append(("snr.score", snr_obj.get("score")))
            for label, snr_val in snr_candidates:
                if snr_val is None:
                    continue
                try:
                    assert_snr_normalized(snr_val, label=label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Receipt {label} invalid: {exc}. "
                        f"Use normalize_snr_linear() before appending."
                    ) from exc

        with self._lock:  # in-process thread safety
            with _posix_file_lock(self._lock_file):  # cross-process safety (G6 fix)
                # Re-read seq+hash from disk (tail-only, O(1)) after acquiring
                # the cross-process lock — so concurrent processes see each
                # other's writes before incrementing the sequence counter.
                self._resume_tail()

                self._sequence += 1
                ts = datetime.now(timezone.utc).isoformat()
                entry_hash = _compute_entry_hash(
                    self._sequence, receipt, self._last_hash
                )

                entry = LedgerEntry(
                    sequence=self._sequence,
                    receipt=receipt,
                    prev_hash=self._last_hash,
                    entry_hash=entry_hash,
                    timestamp=ts,
                )

                # Append to file
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(entry.to_jsonl() + "\n")

                self._last_hash = entry_hash
                return entry

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire ledger chain.

        Returns:
            (is_valid, errors) — errors is empty on success.
        """
        errors: List[str] = []
        prev_hash = GENESIS_HASH
        expected_seq = 0

        if not self._path.exists():
            return True, []

        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = LedgerEntry.from_jsonl(line)
                except (json.JSONDecodeError, KeyError) as e:
                    errors.append(f"Line {line_num}: parse error: {e}")
                    continue

                expected_seq += 1

                # Check sequence monotonicity
                if entry.sequence != expected_seq:
                    errors.append(
                        f"Line {line_num}: expected seq {expected_seq}, got {entry.sequence}"
                    )

                # Check chain link
                if entry.prev_hash != prev_hash:
                    errors.append(
                        f"Line {line_num}: prev_hash mismatch "
                        f"(expected {prev_hash[:16]}..., got {entry.prev_hash[:16]}...)"
                    )

                # Recompute entry hash. Accept either current BLAKE3 hash or
                # historical SHA-256 hash to support pre-migration ledgers.
                recomputed_blake3 = _compute_entry_hash(
                    entry.sequence, entry.receipt, entry.prev_hash
                )
                if entry.entry_hash != recomputed_blake3:
                    recomputed_sha256 = _compute_entry_hash_legacy_sha256(
                        entry.sequence, entry.receipt, entry.prev_hash
                    )
                    if entry.entry_hash == recomputed_sha256:
                        logger.debug(
                            "Evidence ledger line %d verified with legacy SHA-256 hash",
                            line_num,
                        )
                    else:
                        errors.append(
                            f"Line {line_num}: entry_hash mismatch "
                            f"(expected blake3 {recomputed_blake3[:16]}... or "
                            f"legacy sha256 {recomputed_sha256[:16]}..., got "
                            f"{entry.entry_hash[:16]}...)"
                        )

                prev_hash = entry.entry_hash

        return len(errors) == 0, errors

    def entries(self) -> List[LedgerEntry]:
        """Read all entries from the ledger."""
        if not self._path.exists():
            return []

        result = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result.append(LedgerEntry.from_jsonl(line))
        return result

    def count(self) -> int:
        """Number of entries in the ledger."""
        return self._sequence


# =============================================================================
# RECEIPT EMITTER — Bridges proof_engine.Receipt → Schema Receipt → Ledger
# =============================================================================


def emit_receipt(
    ledger: EvidenceLedger,
    *,
    receipt_id: str,
    node_id: str,
    policy_version: str = "1.0.0",
    status: str = "accepted",
    decision: str = "APPROVED",
    reason_codes: Optional[List[str]] = None,
    snr_score: float = 0.0,
    ihsan_score: float = 0.0,
    ihsan_threshold: float = 0.95,
    seal_digest: str = "",
    seal_algorithm: str = "blake3",
    query_digest: Optional[str] = None,
    policy_digest: Optional[str] = None,
    payload_digest: Optional[str] = None,
    graph_hash: Optional[str] = None,
    gate_passed: Optional[str] = None,
    duration_ms: float = 0.0,
    claim_tags: Optional[Dict[str, int]] = None,
    snr_trace: Optional[Dict[str, Any]] = None,
    signer_private_key_hex: Optional[str] = None,
    signer_public_key_hex: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    critical_decision: bool = False,
    node_role: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> LedgerEntry:
    """
    Emit a schema-compliant receipt into the evidence ledger.

    This is the bridge between the proof engine's internal Receipt
    objects and the schema-validated JSONL ledger.

    Returns the LedgerEntry that was appended.
    """
    # Build ihsan decision from score vs threshold
    ihsan_decision = "APPROVED" if ihsan_score >= ihsan_threshold else "REJECTED"

    # Build SNR section — include authoritative trace components when available
    snr_section: Dict[str, Any] = {"score": snr_score}
    if snr_trace:
        snr_section["signal_mass"] = snr_trace.get("signal_mass", 0.0)
        snr_section["noise_mass"] = snr_trace.get("noise_mass", 0.0)
        if "signal_components" in snr_trace:
            snr_section["signal_components"] = snr_trace["signal_components"]
        if "noise_components" in snr_trace:
            snr_section["noise_components"] = snr_trace["noise_components"]
        if "policy_digest" in snr_trace:
            snr_section["policy_digest"] = snr_trace["policy_digest"]
        if "trace_id" in snr_trace:
            snr_section["trace_id"] = snr_trace["trace_id"]
        if "method" in snr_trace:
            snr_section["method"] = snr_trace["method"]
        if "engine" in snr_trace:
            snr_section["engine"] = snr_trace["engine"]
        if "quality_tier" in snr_trace:
            snr_section["quality_tier"] = snr_trace["quality_tier"]
        if "recommendations" in snr_trace:
            recs = snr_trace["recommendations"]
            if isinstance(recs, (list, tuple)):
                snr_section["recommendations"] = list(recs)
        if "diffusion" in snr_trace and isinstance(snr_trace["diffusion"], dict):
            snr_section["diffusion"] = snr_trace["diffusion"]

    receipt: Dict[str, Any] = {
        "receipt_id": receipt_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_id": node_id,
        "policy_version": policy_version,
        "status": status,
        "decision": decision,
        "reason_codes": reason_codes or [],
        "snr": snr_section,
        "ihsan": {
            "score": ihsan_score,
            "threshold": ihsan_threshold,
            "decision": ihsan_decision,
        },
        "seal": {
            "algorithm": seal_algorithm,
            "digest": seal_digest or ("0" * 64),
        },
    }

    # Optional fields
    if query_digest or policy_digest or payload_digest or graph_hash:
        outputs: Dict[str, str] = {}
        inputs: Dict[str, str] = {}
        if query_digest:
            inputs["query_digest"] = query_digest
        if policy_digest:
            inputs["policy_digest"] = policy_digest
        if payload_digest:
            outputs["payload_digest"] = payload_digest
        if graph_hash:
            outputs["graph_hash"] = graph_hash
        if inputs:
            receipt["inputs"] = inputs
        if outputs:
            receipt["outputs"] = outputs

    if gate_passed:
        receipt["gate_passed"] = gate_passed

    # SNR claim tags from authoritative trace
    if snr_trace and "claim_tags" in snr_trace:
        receipt["snr"]["claim_tags"] = snr_trace["claim_tags"]

    if duration_ms > 0:
        receipt["metrics"] = {"duration_ms": duration_ms}

    if claim_tags:
        receipt["claim_tags_summary"] = claim_tags

    role = normalize_node_role(node_role or os.getenv("BIZRA_NODE_ROLE", "node"))
    if critical_decision or origin is not None:
        from core.proof_engine.canonical import blake3_digest, canonical_bytes

        origin_payload = origin or resolve_origin_snapshot(
            state_dir or Path("sovereign_state"),
            role,
        )
        receipt["origin"] = origin_payload
        receipt["origin_digest"] = blake3_digest(canonical_bytes(origin_payload)).hex()

    _attach_optional_signature(
        receipt,
        signer_private_key_hex=signer_private_key_hex,
        signer_public_key_hex=signer_public_key_hex,
        strict_required=critical_decision and role == ROLE_NODE0,
    )

    return ledger.append(receipt)


def _attach_optional_signature(
    receipt: Dict[str, Any],
    *,
    signer_private_key_hex: Optional[str] = None,
    signer_public_key_hex: Optional[str] = None,
    strict_required: bool = False,
) -> None:
    """Attach Ed25519 signature to receipt when a signer key is configured.

    Signature is computed over `seal.digest` to align with the receipt schema:
    "Ed25519 signature over the seal digest".

    Signer can be provided explicitly or via environment:
    - BIZRA_RECEIPT_PRIVATE_KEY_HEX (required to sign)
    - BIZRA_RECEIPT_PUBLIC_KEY_HEX (optional sanity check)
    """
    private_key_hex = (
        signer_private_key_hex or os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip()
    )
    if not private_key_hex:
        if strict_required:
            raise RuntimeError(
                "Unsigned critical receipt forbidden in Node0 mode: "
                "BIZRA_RECEIPT_PRIVATE_KEY_HEX is required"
            )
        # CRITICAL-6 FIX: Warn instead of silently producing unsigned receipts.
        # Constitutional requirement: "Every operation produces a signed receipt."
        import warnings

        warnings.warn(
            "BIZRA_RECEIPT_PRIVATE_KEY_HEX not set — receipts will be UNSIGNED. "
            "This violates the constitutional requirement for proof-carrying inference. "
            "Set the key via environment variable or sovereign_state/key_registry.json.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    from core.pci.crypto import PrivateKeyWrapper, verify_signature

    signer = PrivateKeyWrapper(private_key_hex)
    derived_public_key_hex = signer.public_key_hex
    expected_public_key_hex = (
        signer_public_key_hex or os.getenv("BIZRA_RECEIPT_PUBLIC_KEY_HEX", "").strip()
    )

    if expected_public_key_hex:
        expected_normalized = expected_public_key_hex.lower()
        if expected_normalized != derived_public_key_hex:
            raise ValueError(
                "Configured BIZRA_RECEIPT_PUBLIC_KEY_HEX does not match "
                "BIZRA_RECEIPT_PRIVATE_KEY_HEX"
            )

    seal = receipt.get("seal", {})
    seal_digest = str(seal.get("digest", "")).lower()
    if len(seal_digest) != 64 or any(c not in "0123456789abcdef" for c in seal_digest):
        raise ValueError("Receipt seal.digest must be a 64-char hex digest")

    signature_hex = signer.sign(seal_digest)
    if not verify_signature(
        seal_digest,
        signature_hex,
        derived_public_key_hex,
    ):
        raise ValueError("Generated receipt signature failed self-verification")

    receipt["signature"] = {
        "algorithm": "ed25519",
        "value": signature_hex,
        "public_key": derived_public_key_hex,
    }


# =============================================================================
# UNIFORM VERIFIER RESPONSE
# =============================================================================


@dataclass
class VerifierResponse:
    """
    Uniform response shape for all verification endpoints.

    Every /v1/verify/* endpoint returns this structure.
    """

    decision: str  # "APPROVED" | "REJECTED" | "QUARANTINED"
    reason_codes: List[str] = field(default_factory=list)
    receipt_id: str = ""
    receipt_signature: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the canonical response shape."""
        return {
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "receipt_id": self.receipt_id,
            "receipt_signature": self.receipt_signature,
            "artifacts": self.artifacts,
        }

    @classmethod
    def approved(
        cls,
        receipt_id: str,
        receipt_signature: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> "VerifierResponse":
        """Create an APPROVED response."""
        return cls(
            decision="APPROVED",
            reason_codes=[],
            receipt_id=receipt_id,
            receipt_signature=receipt_signature,
            artifacts=artifacts or {},
        )

    @classmethod
    def rejected(
        cls,
        reason_codes: List[str],
        receipt_id: str = "",
        receipt_signature: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> "VerifierResponse":
        """Create a REJECTED response with reason codes."""
        if not reason_codes:
            raise ValueError("REJECTED response must include at least one reason code")
        return cls(
            decision="REJECTED",
            reason_codes=reason_codes,
            receipt_id=receipt_id,
            receipt_signature=receipt_signature,
            artifacts=artifacts or {},
        )

    @classmethod
    def quarantined(
        cls,
        reason_codes: List[str],
        receipt_id: str = "",
        receipt_signature: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> "VerifierResponse":
        """Create a QUARANTINED response with reason codes."""
        if not reason_codes:
            raise ValueError(
                "QUARANTINED response must include at least one reason code"
            )
        return cls(
            decision="QUARANTINED",
            reason_codes=reason_codes,
            receipt_id=receipt_id,
            receipt_signature=receipt_signature,
            artifacts=artifacts or {},
        )


__all__ = [
    "EvidenceLedger",
    "LedgerEntry",
    "VerifierResponse",
    "emit_receipt",
    "GENESIS_HASH",
]
