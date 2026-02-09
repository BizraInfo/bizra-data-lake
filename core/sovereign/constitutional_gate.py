"""
Constitutional Gate — Strict Synthesis Admission Controller
===========================================================
Z3 proof-based admission controller with Four Pillars Architecture:
- PILLAR 1 (Runtime): Z3-proven agents ONLY (Ihsan = 1.0)
- PILLAR 2 (Museum): SNR-v2 scored, awaiting Z3 synthesis
- PILLAR 3 (Sandbox): Isolated simulation, no PCI signing
- PILLAR 4 (Cutoff): T+72h absolute deadline for proofs

Standing on Giants: Z3 SMT Solver + Shannon (SNR) + Lamport (BFT)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import blake3

try:
    from core.integration.constants import (
        PILLAR_3_SANDBOX_SNR_FLOOR,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    UNIFIED_SNR_THRESHOLD = 0.85  # type: ignore[misc]
    PILLAR_3_SANDBOX_SNR_FLOOR = 0.70  # type: ignore[misc]

from core.pci.crypto import canonical_json, verify_signature

from .integration_types import (
    AdmissionResult,
    AdmissionStatus,
    Z3Certificate,
)

logger = logging.getLogger(__name__)

Z3_CERT_DOMAIN_PREFIX = "bizra-z3-cert-v1:"


class ConstitutionalGate:
    """
    The Strict Synthesis Admission Controller.

    "Mathematical certainty in execution; archival mercy in development."
    """

    def __init__(
        self,
        z3_certificates_path: Optional[Path] = None,
        museum_path: Optional[Path] = None,
    ):
        """
        Initialize the Constitutional Gate.

        Args:
            z3_certificates_path: Path to Z3 proof certificates
            museum_path: Path to Museum archive for unproven code
        """
        self.z3_certificates_path = (
            z3_certificates_path or Path.home() / ".bizra" / "proofs"
        )
        self.museum_path = museum_path or Path.home() / ".bizra" / "museum"
        self.museum_queue: List[Tuple[str, float]] = []
        self.runtime_agents: List[str] = []
        self._z3_cache: Dict[str, Z3Certificate] = {}
        self._trusted_z3_pubkey = self._load_trusted_z3_pubkey()
        self._allow_unsigned_z3 = os.getenv("BIZRA_Z3_CERT_ALLOW_UNSIGNED", "0") == "1"
        self._allow_self_signed_z3 = (
            os.getenv("BIZRA_Z3_CERT_ALLOW_SELF_SIGNED", "0") == "1"
        )

        # Production security enforcement — fail-closed by default
        # Standing on Giants: Saltzer & Schroeder (1975) — "Fail-safe defaults"
        is_production = os.getenv("BIZRA_ENV", "").lower() == "production"
        allow_override = os.getenv("BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE", "0") == "1"

        if is_production and (self._allow_unsigned_z3 or self._allow_self_signed_z3):
            if not allow_override:
                raise RuntimeError(
                    "SECURITY HALT: BIZRA_Z3_CERT_ALLOW_UNSIGNED and BIZRA_Z3_CERT_ALLOW_SELF_SIGNED "
                    "are forbidden in production. This is a fail-closed default. "
                    "Set BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE=1 to explicitly accept this risk."
                )
            else:
                logger.warning(
                    "SECURITY WARNING: Unsigned/self-signed Z3 certificates explicitly "
                    "overridden in production via BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE=1. "
                    "This bypasses cryptographic verification."
                )

        # Try to import SNR v2 calculator
        try:
            from core.iaas.snr_v2 import SNRCalculatorV2

            self.calculator = SNRCalculatorV2()  # type: ignore[assignment]
        except ImportError:
            self.calculator = None  # type: ignore[assignment]
            logger.warning("SNR v2 calculator not available, using fallback")

    async def admit(
        self,
        candidate: str,
        query: str,
        candidate_id: Optional[str] = None,
    ) -> AdmissionResult:
        """
        Strict Synthesis admission: Z3 proof required for Runtime.

        Args:
            candidate: The candidate content/code to evaluate
            query: Context query for SNR calculation
            candidate_id: Optional identifier for Z3 certificate lookup

        Returns:
            AdmissionResult with status, score, and evidence
        """
        candidate_hash = candidate_id or self._compute_hash(candidate)

        # Check for existing Z3 certificate
        z3_cert = self._get_z3_certificate(candidate_hash)

        if z3_cert and z3_cert.valid:
            # PILLAR 1: RUNTIME ADMISSION (Z3-proven)
            self.runtime_agents.append(candidate_hash)
            return AdmissionResult(
                status=AdmissionStatus.RUNTIME,
                score=1.0,
                evidence={
                    "verification": "Z3_PROVEN",
                    "certificate_hash": z3_cert.hash,
                    "proof_type": z3_cert.proof_type,
                    "ihsan": 1.0,
                },
            )

        # No Z3 proof — calculate SNR v2 for Museum consideration
        snr_score, snr_components = self._calculate_snr(query, candidate)

        if snr_score >= UNIFIED_SNR_THRESHOLD:
            # PILLAR 2: MUSEUM ARCHIVAL (awaiting Z3 proof)
            self.museum_queue.append((candidate_hash, snr_score))
            self._schedule_background_proofing(candidate_hash, candidate)

            return AdmissionResult(
                status=AdmissionStatus.MUSEUM,
                score=snr_score,
                evidence={
                    "verification": "SNR_V2_PENDING",
                    "snr_components": snr_components,
                    "ihsan": snr_score,
                    "museum_position": len(self.museum_queue),
                },
                promotion_path="background_z3_synthesis",
            )

        # Below SNR threshold — SANDBOX or REJECT
        if snr_score >= PILLAR_3_SANDBOX_SNR_FLOOR:
            # PILLAR 3: SANDBOX (simulation only)
            return AdmissionResult(
                status=AdmissionStatus.SANDBOX,
                score=snr_score,
                evidence={
                    "verification": "SANDBOX_ONLY",
                    "reason": f"SNR {snr_score:.3f} qualifies for sandbox simulation",
                    "snr_components": snr_components,
                    "restrictions": [
                        "read_only_data_lake",
                        "no_pci_signing",
                        "no_consensus_votes",
                        "recommendations_unverified",
                    ],
                },
            )

        # REJECTION
        return AdmissionResult(
            status=AdmissionStatus.REJECTED,
            score=snr_score,
            evidence={
                "verification": "SUBSTANDARD",
                "reason": f"SNR {snr_score:.3f} < 0.70 minimum threshold",
                "snr_components": snr_components,
            },
        )

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for identification."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_trusted_z3_pubkey(self) -> str:
        """Load trusted public key for Z3 certificate verification."""
        env_pubkey = os.getenv("BIZRA_Z3_CERT_PUBKEY", "").strip()
        if env_pubkey:
            return env_pubkey

        keypair_path = Path(
            os.path.expanduser(os.getenv("BIZRA_KEYPAIR_PATH", "~/.bizra/keypair.json"))
        )
        if keypair_path.exists():
            try:
                with open(keypair_path) as f:
                    data = json.load(f)
                return data.get("public_key", "").strip()
            except Exception as e:
                logger.warning(f"Failed to load trusted Z3 pubkey: {e}")

        return ""

    def _z3_signable_dict(
        self, data: Dict[str, Any], candidate_id: str
    ) -> Dict[str, Any]:
        """Build canonical signable fields for Z3 certificate."""
        return {
            "hash": data.get("hash", candidate_id),
            "valid": bool(data.get("valid", False)),
            "proof_type": data.get("proof_type", "z3-smt2"),
            "verified_at": data.get("verified_at", ""),
        }

    def _z3_digest(self, signable: Dict[str, Any]) -> str:
        """Domain-separated digest for Z3 certificate signatures."""
        canonical = canonical_json(signable)
        hasher = blake3.blake3()
        hasher.update(Z3_CERT_DOMAIN_PREFIX.encode("utf-8"))
        hasher.update(canonical)
        return hasher.hexdigest()

    def _verify_z3_certificate_signature(
        self, data: Dict[str, Any], candidate_id: str
    ) -> bool:
        """Verify signed Z3 certificate metadata."""
        signature = data.get("signature", "").strip()
        cert_pubkey = data.get("public_key", "").strip()

        if not signature:
            if self._allow_unsigned_z3:
                logger.warning(
                    "Unsigned Z3 certificate accepted (BIZRA_Z3_CERT_ALLOW_UNSIGNED=1)"
                )
                return True
            logger.warning("Unsigned Z3 certificate rejected")
            return False

        if self._trusted_z3_pubkey:
            if cert_pubkey and cert_pubkey != self._trusted_z3_pubkey:
                logger.warning("Z3 certificate public key mismatch against trusted key")
                return False
            pubkey = self._trusted_z3_pubkey
        else:
            if not self._allow_self_signed_z3:
                logger.warning(
                    "Z3 certificate rejected: no trusted pubkey. "
                    "Set BIZRA_Z3_CERT_PUBKEY or allow self-signed with BIZRA_Z3_CERT_ALLOW_SELF_SIGNED=1"
                )
                return False
            if not cert_pubkey:
                logger.warning(
                    "Z3 certificate rejected: missing public_key for self-signed verification"
                )
                return False
            pubkey = cert_pubkey

        signable = self._z3_signable_dict(data, candidate_id)
        digest = self._z3_digest(signable)
        if not verify_signature(digest, signature, pubkey):
            logger.warning("Z3 certificate signature verification failed")
            return False

        return True

    def _get_z3_certificate(self, candidate_id: str) -> Optional[Z3Certificate]:
        """Load Z3 proof certificate if exists and signature verifies."""
        # Check cache first
        if candidate_id in self._z3_cache:
            return self._z3_cache[candidate_id]

        # Try file-based certificate
        cert_path = self.z3_certificates_path / f"{candidate_id}_z3.cert"
        if cert_path.exists():
            try:
                with open(cert_path) as f:
                    data = json.load(f)
                if not self._verify_z3_certificate_signature(data, candidate_id):
                    return None
                cert = Z3Certificate(
                    hash=data.get("hash", candidate_id),
                    valid=data.get("valid", False),
                    proof_type=data.get("proof_type", "z3-smt2"),
                    verified_at=data.get("verified_at", ""),
                )
                self._z3_cache[candidate_id] = cert
                return cert
            except Exception as e:
                logger.warning(f"Failed to load Z3 certificate: {e}")

        # Try Rust FFI for Z3 verification (when available)
        try:
            from bizra_omega import z3_verify

            result = z3_verify.get_certificate(candidate_id)
            if result:
                cert = Z3Certificate(
                    hash=result.hash,
                    valid=result.valid,
                    proof_type="z3-smt2",
                )
                self._z3_cache[candidate_id] = cert
                return cert
        except ImportError:
            pass

        return None

    def _calculate_snr(
        self,
        query: str,
        candidate: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate SNR v2 score for candidate."""
        if self.calculator:
            components = self.calculator.calculate_simple(
                query=query,
                texts=[candidate],
                iaas_score=0.9,
            )
            return components.snr, components.to_dict()
        else:
            return self._fallback_snr(candidate), {"method": "fallback"}

    def _fallback_snr(self, text: str) -> float:
        """Fallback SNR calculation when calculator unavailable."""
        words = text.split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        length_factor = min(1.0, len(words) / 100)
        return (unique_ratio * 0.6 + length_factor * 0.4) * 0.9

    def _schedule_background_proofing(self, candidate_id: str, candidate: str) -> None:
        """Queue candidate for background Z3 synthesis."""
        if self.museum_path:
            self.museum_path.mkdir(parents=True, exist_ok=True)
            entry_path = self.museum_path / f"{candidate_id}.json"
            try:
                with open(entry_path, "w") as f:
                    json.dump(
                        {
                            "id": candidate_id,
                            "content_hash": self._compute_hash(candidate),
                            "status": "pending_proof",
                            "queued_at": (
                                asyncio.get_event_loop().time()
                                if asyncio.get_event_loop().is_running()
                                else 0
                            ),
                        },
                        f,
                    )
                logger.info(f"Queued {candidate_id} for background Z3 proofing")
            except Exception as e:
                logger.warning(f"Failed to queue for proofing: {e}")

    def get_runtime_count(self) -> int:
        """Get count of runtime-admitted agents."""
        return len(self.runtime_agents)

    def get_museum_count(self) -> int:
        """Get count of museum-archived candidates."""
        return len(self.museum_queue)

    def promote_to_runtime(self, candidate_id: str) -> bool:
        """Promote a museum candidate to runtime after Z3 proof."""
        for i, (cid, _) in enumerate(self.museum_queue):
            if cid == candidate_id:
                cert = self._get_z3_certificate(candidate_id)
                if cert and cert.valid:
                    self.museum_queue.pop(i)
                    self.runtime_agents.append(candidate_id)
                    logger.info(f"Promoted {candidate_id} from Museum to Runtime")
                    return True
                else:
                    logger.warning(
                        f"Cannot promote {candidate_id}: no valid Z3 certificate"
                    )
                    return False
        return False


__all__ = [
    "ConstitutionalGate",
    "Z3_CERT_DOMAIN_PREFIX",
]
