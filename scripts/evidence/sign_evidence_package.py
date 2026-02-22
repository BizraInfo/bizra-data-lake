#!/usr/bin/env python3
"""Sign evidence manifest chain with BLAKE3 + Ed25519 and SHA256 compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from nacl.signing import SigningKey

try:
    from common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_PACKAGE_ROOT,
        REPO_ROOT,
        canonical_json_bytes,
        hash_bytes_blake3,
        hash_bytes_sha256,
        hash_file_sha256,
        load_yaml,
        resolve_path,
        utc_now_iso,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_PACKAGE_ROOT,
        REPO_ROOT,
        canonical_json_bytes,
        hash_bytes_blake3,
        hash_bytes_sha256,
        hash_file_sha256,
        load_yaml,
        resolve_path,
        utc_now_iso,
        write_json,
    )


def _load_or_create_key(key_path: Path) -> tuple[SigningKey, str]:
    if key_path.exists():
        key_obj = json.loads(key_path.read_text(encoding="utf-8"))
        sk = SigningKey(bytes.fromhex(key_obj["private_key_hex"]))
        return sk, str(key_obj["public_key_hex"])

    sk = SigningKey.generate()
    pk = sk.verify_key.encode().hex()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        key_path,
        {
            "private_key_hex": sk.encode().hex(),
            "public_key_hex": pk,
            "generated_at": utc_now_iso(),
        },
    )
    return sk, pk


def _chain_payload(
    *, manifest_blake3: str, previous_chain_blake3: str, timestamp_utc: str, tier: str, stage: str, policy_version: str
) -> dict[str, str]:
    return {
        "manifest_blake3": manifest_blake3,
        "previous_chain_blake3": previous_chain_blake3,
        "timestamp_utc": timestamp_utc,
        "tier": tier,
        "stage": stage,
        "policy_version": policy_version,
    }


def run(package_root: Path, tier: str, config_path: Path) -> int:
    cfg = load_yaml(config_path)
    policy_version = str(cfg.get("policy_version", "evidence-v1.0"))

    tier_root = package_root / tier
    manifest_path = tier_root / "manifest" / "evidence_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: missing manifest {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_blake3 = hash_bytes_blake3(manifest_bytes)
    manifest_sha256 = hash_bytes_sha256(manifest_bytes)

    integrity_root = package_root / "integrity"
    chain_path = integrity_root / "evidence_chain.json"
    key_path = integrity_root / "operator_signing_key.json"

    signing_key, public_key_hex = _load_or_create_key(key_path)
    if chain_path.exists():
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        if not isinstance(chain, list):
            raise ValueError(f"Invalid chain format at {chain_path}")
    else:
        chain = []

    prev = str(chain[-1]["chain_blake3"]) if chain else "0" * 64
    stage = str(manifest.get("stage", "scaffold"))
    timestamp = utc_now_iso()

    payload = _chain_payload(
        manifest_blake3=manifest_blake3,
        previous_chain_blake3=prev,
        timestamp_utc=timestamp,
        tier=tier,
        stage=stage,
        policy_version=policy_version,
    )
    chain_blake3 = hash_bytes_blake3(canonical_json_bytes(payload))
    signature = signing_key.sign(chain_blake3.encode("utf-8")).signature.hex()

    rec = {
        "seq": len(chain) + 1,
        "timestamp_utc": timestamp,
        "manifest_blake3": manifest_blake3,
        "manifest_sha256": manifest_sha256,
        "previous_chain_blake3": prev,
        "chain_blake3": chain_blake3,
        "sha256_compat": hashlib.sha256(chain_blake3.encode("utf-8")).hexdigest(),
        "signer_pubkey": public_key_hex,
        "ed25519_signature": signature,
        "policy_version": policy_version,
        "tier": tier,
        "stage": stage,
    }
    chain.append(rec)
    write_json(chain_path, chain)

    checksums_dir = tier_root / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    (checksums_dir / "evidence_manifest.blake3").write_text(manifest_blake3 + "\n", encoding="utf-8")
    (checksums_dir / "evidence_manifest.sha256").write_text(manifest_sha256 + "\n", encoding="utf-8")
    (checksums_dir / "evidence_manifest.sig").write_text(signature + "\n", encoding="utf-8")

    constitution_source = cfg.get("constitution_policy_source")
    policy_hash = None
    if constitution_source:
        constitution_path = resolve_path(REPO_ROOT, str(constitution_source))
        if constitution_path.exists() and constitution_path.is_file():
            policy_hash = hash_file_sha256(constitution_path)

    anchor = {
        "generated_at": timestamp,
        "genesis_blake3": cfg.get("genesis_anchor", "Ω"),
        "merkle_root": manifest_blake3,
        "rfc3161_token": None,
        "ots_proof": None,
        "policy_hash": policy_hash,
        "chain_blake3": chain_blake3,
        "signer_pubkey": public_key_hex,
        "ed25519_signature": signature,
    }
    write_json(tier_root / "integrity" / "CHAIN_ANCHOR.json", anchor)

    print(f"Signed manifest: {manifest_path}")
    print(f"Chain record seq: {rec['seq']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign evidence package tier")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--tier", choices=["private_full", "public_redacted"], default="private_full")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    raise SystemExit(run(package_root=args.package_root, tier=args.tier, config_path=args.config))


if __name__ == "__main__":
    main()
