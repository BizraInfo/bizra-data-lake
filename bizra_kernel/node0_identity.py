"""
bizra_kernel/node0_identity.py
==============================
Node0 identity anchor: Ed25519 signing, restricted mode, tier-2 attestation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    serialization = None  # type: ignore


DEFAULT_KEY_DIR = Path.home() / ".bizra" / "node0" / "keys"
LEGACY_KEY_DIRS = [
    Path.home() / ".bizra" / "keys",
    Path.home() / ".bizra" / "node0" / "legacy_keys",
    Path("/mnt/c/BIZRA-VAULT/keys"),
]

RESTRICTED_MODE_FLAG = DEFAULT_KEY_DIR / "restricted_mode.flag"


@dataclass
class RestrictedState:
    reason: str
    disabled_capabilities: List[str]
    activated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Node0Identity:
    key_dir: Path
    attestation_dir: Path
    private_key: Optional[Ed25519PrivateKey] = None
    public_key: Optional[Ed25519PublicKey] = None
    public_key_bytes: Optional[bytes] = None
    is_restricted: bool = False
    restricted_state: Optional[RestrictedState] = None

    @property
    def has_private_key(self) -> bool:
        return self.private_key is not None

    @property
    def public_key_fingerprint(self) -> str:
        if self.public_key_bytes is None:
            return ""
        return hashlib.sha256(self.public_key_bytes).hexdigest()

    @classmethod
    def load_or_create(
        cls,
        force_create: bool = False,
        key_dir: Optional[Path] = None,
        attestation_dir: Optional[Path] = None,
    ) -> "Node0Identity":
        key_dir = key_dir or _get_vault_dir()
        attestation_dir = attestation_dir or (key_dir / "attestations")
        key_dir.mkdir(parents=True, exist_ok=True)
        attestation_dir.mkdir(parents=True, exist_ok=True)

        if force_create or not _dir_has_keys(key_dir):
            if not CRYPTO_AVAILABLE:
                raise RuntimeError("cryptography is required for Node0Identity")
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            _write_keypair(key_dir, private_key, public_key)
        else:
            private_key, public_key = _load_keypair(key_dir)

        public_key_bytes = _get_public_key_bytes(public_key, key_dir)

        return cls(
            key_dir=key_dir,
            attestation_dir=attestation_dir,
            private_key=private_key,
            public_key=public_key,
            public_key_bytes=public_key_bytes,
            is_restricted=_restricted_flag_exists(),
        )

    def enter_restricted_mode(self, reason: str) -> None:
        self.is_restricted = True
        self.restricted_state = RestrictedState(
            reason=reason,
            disabled_capabilities=["signing", "genesis"],
        )
        try:
            RESTRICTED_MODE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            RESTRICTED_MODE_FLAG.write_text(reason, encoding="utf-8")
        except Exception:
            pass

    def sign_genesis(self, genesis: Dict[str, Any]) -> str:
        if self.is_restricted:
            raise RuntimeError("restricted mode: signing disabled")
        if not CRYPTO_AVAILABLE or self.private_key is None:
            raise RuntimeError("cryptography private key not available")

        payload = dict(genesis)
        payload.pop("signature", None)
        data = _canonical_json(payload)
        signature = self.private_key.sign(data)
        return "signature:" + base64.b64encode(signature).decode("utf-8")

    def verify_genesis(self, genesis: Dict[str, Any]) -> Dict[str, Any]:
        signature = genesis.get("signature")
        if not signature:
            return {"verified": False, "error": "No signature"}

        if not CRYPTO_AVAILABLE or self.public_key is None:
            return {"verified": False, "error": "No public key"}

        sig_bytes = _decode_signature(signature)
        payload = dict(genesis)
        payload.pop("signature", None)
        data = _canonical_json(payload)
        try:
            self.public_key.verify(sig_bytes, data)
            return {
                "verified": True,
                "pubkey_fingerprint": self.public_key_fingerprint,
            }
        except Exception as exc:
            return {
                "verified": False,
                "error": f"signature mismatch: {exc}",
            }

    def create_tier2_attestation(
        self,
        *,
        previous_hash: str,
        new_hash: str,
        reason: str,
        changed_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not CRYPTO_AVAILABLE or self.private_key is None:
            raise RuntimeError("cryptography private key not available")

        payload = {
            "previous_hash": previous_hash,
            "new_hash": new_hash,
            "reason": reason,
            "changed_components": changed_components,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        attestation_id = hashlib.sha256(_canonical_json(payload)).hexdigest()
        payload["attestation_id"] = attestation_id
        signature = self.private_key.sign(_canonical_json(payload))
        payload["signature"] = "signature:" + base64.b64encode(signature).decode("utf-8")

        self.attestation_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.attestation_dir / f"tier2_attestation_{attestation_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def verify_tier2_attestation(
        self,
        *,
        current_tier2_hash: str,
        expected_tier2_hash: str,
    ) -> Dict[str, Any]:
        if current_tier2_hash == expected_tier2_hash:
            return {"verified": True, "attestation_required": False}

        # Look for matching attestation
        for path in self.attestation_dir.glob("tier2_attestation_*.json"):
            try:
                attestation = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                attestation.get("previous_hash") == expected_tier2_hash
                and attestation.get("new_hash") == current_tier2_hash
            ):
                valid = _verify_attestation(attestation, self.public_key)
                return {
                    "verified": bool(valid),
                    "attestation_valid": bool(valid),
                }

        return {"verified": False, "attestation_required": True}


# =============================================================================
# Helpers
# =============================================================================

def _restricted_flag_exists() -> bool:
    try:
        return RESTRICTED_MODE_FLAG.exists()
    except Exception:
        return False


def _canonical_json(data: Dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _decode_signature(signature: str) -> bytes:
    if signature.startswith("signature:"):
        signature = signature.split(":", 1)[1]
    return base64.b64decode(signature)


def _write_keypair(key_dir: Path, private_key: Ed25519PrivateKey, public_key: Ed25519PublicKey) -> None:
    key_dir.mkdir(parents=True, exist_ok=True)
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    (key_dir / "node0_signing.key").write_bytes(private_bytes)
    (key_dir / "node0_signing.pub").write_bytes(public_bytes)


def _load_keypair(key_dir: Path) -> tuple[Optional[Ed25519PrivateKey], Optional[Ed25519PublicKey]]:
    priv_path = key_dir / "node0_signing.key"
    pub_path = key_dir / "node0_signing.pub"

    private_key = None
    public_key = None

    if CRYPTO_AVAILABLE:
        if priv_path.exists():
            private_bytes = priv_path.read_bytes()
            private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        if pub_path.exists():
            public_bytes = pub_path.read_bytes()
            public_key = Ed25519PublicKey.from_public_bytes(public_bytes)

    return private_key, public_key


def _get_public_key_bytes(public_key: Optional[Ed25519PublicKey], key_dir: Path) -> Optional[bytes]:
    pub_path = key_dir / "node0_signing.pub"
    if pub_path.exists():
        return pub_path.read_bytes()
    if public_key is None:
        return None
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _verify_attestation(attestation: Dict[str, Any], public_key: Optional[Ed25519PublicKey]) -> bool:
    if not CRYPTO_AVAILABLE or public_key is None:
        return False
    signature = attestation.get("signature")
    if not signature:
        return False
    sig_bytes = _decode_signature(signature)
    payload = {k: v for k, v in attestation.items() if k != "signature"}
    try:
        public_key.verify(sig_bytes, _canonical_json(payload))
        return True
    except Exception:
        return False


def _dir_has_keys(path: Path) -> bool:
    return (path / "node0_signing.key").exists() and (path / "node0_signing.pub").exists()


def _get_vault_dir() -> Path:
    env_override = os.getenv("BIZRA_VAULT_DIR")
    if env_override:
        return Path(env_override)
    for legacy in LEGACY_KEY_DIRS:
        if _dir_has_keys(legacy):
            return legacy
    return DEFAULT_KEY_DIR


def _is_windows_mount(path: Path) -> bool:
    parts = path.resolve().parts
    return len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha()
