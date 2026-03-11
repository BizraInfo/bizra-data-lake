"""
BIZRA URP Pledge — Universal Resource Pool Contribution
=========================================================

When a node joins the network, it pledges a portion of its
hardware resources to the Universal Resource Pool (URP).
This stub creates a signed pledge record; actual enforcement
and scheduling lives in Rust (bizra-resourcepool crate).

Standing on Giants:
- Baran (1964): Distributed resource sharing
- Ostrom (1990): Commons governance with commitment
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.pci.crypto import (
    PrivateKeyWrapper,
    canonicalize_and_validate,
    domain_separated_digest,
    sign_message,
    verify_signature,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class URPPledge:
    """A node's resource pledge to the Universal Resource Pool."""

    node_id: str
    ram_gb: int
    vram_gb: int
    storage_gb: int = 0
    pledge_hash: str = ""
    pledged_at: str = ""
    signed: bool = False
    enforced: bool = False
    enforcement_mode: str = "stub"
    status: str = "deferred"
    reason_code: str = "GENESIS_URP_UNSIGNED_STUB"
    payload_digest: str = ""
    signature: str = ""
    signer_public_key: str = ""
    resource_budget: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "node_local"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "pledge_hash": self.pledge_hash,
            "pledged_at": self.pledged_at,
            "signed": self.signed,
            "enforced": self.enforced,
            "enforcement_mode": self.enforcement_mode,
            "status": self.status,
            "reason_code": self.reason_code,
            "payload_digest": self.payload_digest,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
            "resource_budget": self.resource_budget,
            "provenance": self.provenance,
        }


def _canonical_pledge_payload(
    node_id: str,
    ram_gb: int,
    vram_gb: int,
    storage_gb: int,
    pledged_at: str,
) -> Dict[str, Any]:
    """Canonical payload shared by signing and verification."""
    return {
        "node_id": node_id,
        "resource_budget": {
            "ram_gb": ram_gb,
            "vram_gb": vram_gb,
            "storage_gb": storage_gb,
        },
        "pledged_at": pledged_at,
        "enforcement_mode": "single_node_signed_v1",
    }


def _derive_hashes(payload: Dict[str, Any]) -> tuple[str, str]:
    """Return payload digest and short pledge hash."""
    canonical = canonicalize_and_validate(payload)
    payload_digest = domain_separated_digest(canonical)
    pledge_hash = hashlib.sha256(canonical).hexdigest()[:16]
    return payload_digest, pledge_hash


def pledge_resources(
    node_id: str,
    hardware_info: Dict[str, Any],
    signing_private_key_hex: Optional[str] = None,
) -> URPPledge:
    """
    Create a URP resource pledge from hardware info.

    The pledge commits a portion of the node's resources:
    - RAM: full amount detected
    - VRAM: full GPU memory detected
    - Storage: 0 (placeholder for future)

    In production, this pledge is signed with the node's Ed25519
    key and submitted to the federation for validation.

    Args:
        node_id: Pledging node's identity
        hardware_info: Dict with ram_gb, vram_gb keys

    Returns:
        URPPledge with deterministic hash
    """
    ram_gb = hardware_info.get("ram_gb", 0)
    vram_gb = hardware_info.get("vram_gb", 0)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    storage_gb = int(hardware_info.get("storage_gb", 0))
    payload = _canonical_pledge_payload(node_id, ram_gb, vram_gb, storage_gb, now)
    payload_digest, pledge_hash = _derive_hashes(payload)

    if signing_private_key_hex is None:
        signing_private_key_hex = os.getenv("BIZRA_URP_PRIVATE_KEY_HEX")

    if signing_private_key_hex:
        try:
            signer = PrivateKeyWrapper(signing_private_key_hex)
            signature = sign_message(payload_digest, signing_private_key_hex)
            pledge = URPPledge(
                node_id=node_id,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                storage_gb=storage_gb,
                pledge_hash=pledge_hash,
                pledged_at=now,
                signed=True,
                enforced=True,
                enforcement_mode="single_node_signed",
                status="enforced",
                reason_code="GENESIS_URP_SIGNED",
                payload_digest=payload_digest,
                signature=signature,
                signer_public_key=signer.public_key_hex,
                resource_budget=payload["resource_budget"],
                provenance="node_local",
            )
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            pledge = URPPledge(
                node_id=node_id,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                storage_gb=storage_gb,
                pledge_hash=pledge_hash,
                pledged_at=now,
                signed=False,
                enforced=False,
                enforcement_mode="stub",
                status="deferred",
                reason_code="GENESIS_URP_SIGNING_FAILED",
                payload_digest=payload_digest,
                signature="",
                signer_public_key="",
                resource_budget=payload["resource_budget"],
                provenance="node_local",
            )
            logger.warning("URP signing failed for %s: %s", node_id, exc)
    else:
        pledge = URPPledge(
            node_id=node_id,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            storage_gb=storage_gb,
            pledge_hash=pledge_hash,
            pledged_at=now,
            signed=False,
            enforced=False,
            enforcement_mode="stub",
            status="deferred",
            reason_code="GENESIS_URP_UNSIGNED_STUB",
            payload_digest=payload_digest,
            signature="",
            signer_public_key="",
            resource_budget=payload["resource_budget"],
            provenance="node_local",
        )

    logger.info(
        "URP pledge created: %s — %dGB RAM + %dGB VRAM (hash: %s)",
        node_id,
        ram_gb,
        vram_gb,
        pledge_hash,
    )
    return pledge


def verify_pledge_signature(pledge: URPPledge) -> bool:
    """Verify URP pledge digest and signature in fail-closed manner."""
    if not pledge.signed or not pledge.signature or not pledge.signer_public_key:
        return False
    payload = _canonical_pledge_payload(
        pledge.node_id,
        pledge.ram_gb,
        pledge.vram_gb,
        pledge.storage_gb,
        pledge.pledged_at,
    )
    payload_digest, pledge_hash = _derive_hashes(payload)
    if payload_digest != pledge.payload_digest:
        return False
    if pledge_hash != pledge.pledge_hash:
        return False
    return verify_signature(payload_digest, pledge.signature, pledge.signer_public_key)
