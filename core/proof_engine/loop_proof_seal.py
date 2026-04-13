"""
Loop Proof Seal — Signature verification gate for canonical loop proofs.

Implements the Ed25519 signature sidecar and verification workflow
per Unified Node Contract §9.3:

  1. Claude Code computes the manifest hash (done by loop_proof.py)
  2. Human signs the hash offline
  3. Human provides the signature to Claude Code
  4. Claude Code writes the .sig sidecar and verifies
  5. Loop proof status moves from canonical:false to canonical:true

This module does NOT perform signing. Signing is a Human sovereign act.
This module ONLY verifies, writes sidecars, and reports canonical status.

Standing on Giants:
- Bernstein (2012): Ed25519 high-speed signatures
- BIZRA UNC §9: Key custody and signing workflow
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loop_proof.seal")


@dataclass
class SealStatus:
    """Status of a loop proof's canonical seal."""

    state: str  # "unsigned" | "signed_valid" | "signed_invalid" | "missing_proof" | "missing_sidecar"
    manifest_hash: str = ""
    signature_hex: str = ""
    pubkey_hex: str = ""
    proof_path: str = ""
    sidecar_path: str = ""
    error: str = ""

    @property
    def is_canonical(self) -> bool:
        return self.state == "signed_valid"

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "is_canonical": self.is_canonical,
            "manifest_hash": self.manifest_hash,
            "signature_hex": (
                self.signature_hex[:32] + "..."
                if len(self.signature_hex) > 32
                else self.signature_hex
            ),
            "pubkey_hex": (
                self.pubkey_hex[:32] + "..."
                if len(self.pubkey_hex) > 32
                else self.pubkey_hex
            ),
            "proof_path": self.proof_path,
            "sidecar_path": self.sidecar_path,
            "error": self.error,
        }


def sidecar_path_for(proof_path: Path) -> Path:
    """Get the .sig sidecar path for a proof artifact."""
    return proof_path.with_suffix(".sig.json")


def write_sidecar(
    proof_path: Path,
    signature_hex: str,
    pubkey_hex: str,
) -> Path:
    """Write a signature sidecar file for a loop proof artifact.

    This does NOT sign. The Human provides the signature after
    signing the manifest_hash offline.

    Args:
        proof_path: Path to the loop proof JSON artifact.
        signature_hex: Hex-encoded Ed25519 signature from the Human.
        pubkey_hex: Hex-encoded Ed25519 public key.

    Returns:
        Path to the written sidecar file.
    """
    proof = json.loads(proof_path.read_text())
    manifest_hash = proof.get("manifest_hash", "")

    sidecar = {
        "proof_path": str(proof_path),
        "manifest_hash": manifest_hash,
        "signature_hex": signature_hex,
        "pubkey_hex": pubkey_hex,
        "algorithm": "Ed25519",
        "signed_field": "manifest_hash",
    }

    out = sidecar_path_for(proof_path)
    out.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    logger.info("Sidecar written: %s", out)
    return out


def verify_seal(proof_path: Path) -> SealStatus:
    """Verify the canonical seal status of a loop proof artifact.

    Checks:
      1. Proof artifact exists and is valid JSON
      2. Sidecar exists
      3. Signature verifies against manifest_hash using Ed25519

    Returns:
        SealStatus with the verification result.
    """
    if not proof_path.exists():
        return SealStatus(state="missing_proof", proof_path=str(proof_path))

    try:
        proof = json.loads(proof_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return SealStatus(
            state="missing_proof", proof_path=str(proof_path), error=str(e)
        )

    manifest_hash = proof.get("manifest_hash", "")
    if not manifest_hash:
        return SealStatus(
            state="unsigned",
            proof_path=str(proof_path),
            error="No manifest_hash in proof",
        )

    sidecar = sidecar_path_for(proof_path)
    if not sidecar.exists():
        return SealStatus(
            state="unsigned",
            manifest_hash=manifest_hash,
            proof_path=str(proof_path),
            sidecar_path=str(sidecar),
        )

    try:
        sig_data = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return SealStatus(
            state="signed_invalid",
            manifest_hash=manifest_hash,
            proof_path=str(proof_path),
            sidecar_path=str(sidecar),
            error=f"Sidecar unreadable: {e}",
        )

    sig_hex = sig_data.get("signature_hex", "")
    pubkey_hex = sig_data.get("pubkey_hex", "")
    sidecar_manifest = sig_data.get("manifest_hash", "")

    if sidecar_manifest != manifest_hash:
        return SealStatus(
            state="signed_invalid",
            manifest_hash=manifest_hash,
            signature_hex=sig_hex,
            pubkey_hex=pubkey_hex,
            proof_path=str(proof_path),
            sidecar_path=str(sidecar),
            error=f"Manifest hash mismatch: proof={manifest_hash[:16]}... sidecar={sidecar_manifest[:16]}...",
        )

    # Ed25519 verification
    try:
        from core.pci.crypto import verify_signature

        valid = verify_signature(manifest_hash, sig_hex, pubkey_hex)
    except ImportError:
        return SealStatus(
            state="signed_invalid",
            manifest_hash=manifest_hash,
            signature_hex=sig_hex,
            pubkey_hex=pubkey_hex,
            proof_path=str(proof_path),
            sidecar_path=str(sidecar),
            error="Ed25519 crypto unavailable",
        )
    except Exception as e:
        return SealStatus(
            state="signed_invalid",
            manifest_hash=manifest_hash,
            signature_hex=sig_hex,
            pubkey_hex=pubkey_hex,
            proof_path=str(proof_path),
            sidecar_path=str(sidecar),
            error=f"Verification error: {e}",
        )

    state = "signed_valid" if valid else "signed_invalid"
    return SealStatus(
        state=state,
        manifest_hash=manifest_hash,
        signature_hex=sig_hex,
        pubkey_hex=pubkey_hex,
        proof_path=str(proof_path),
        sidecar_path=str(sidecar),
        error="" if valid else "Signature does not match manifest_hash",
    )


def canonicalize_proof(proof_path: Path) -> bool:
    """Update a proof artifact's canonical flag after successful verification.

    Only succeeds if verify_seal returns signed_valid.

    Returns:
        True if proof was marked canonical, False otherwise.
    """
    status = verify_seal(proof_path)
    if not status.is_canonical:
        logger.warning("Cannot canonicalize: %s (%s)", status.state, status.error)
        return False

    proof = json.loads(proof_path.read_text())
    proof["canonical"] = True
    proof["signature"] = status.signature_hex
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True))
    logger.info("Proof canonicalized: %s", proof_path)
    return True
