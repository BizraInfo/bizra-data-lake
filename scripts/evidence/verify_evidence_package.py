#!/usr/bin/env python3
"""Verify evidence package integrity, signatures, and gate outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

try:
    from common import (
        DEFAULT_PACKAGE_ROOT,
        canonical_json_bytes,
        hash_bytes_blake3,
        hash_file_blake3,
        hash_file_sha256,
        manifest_content_hash,
    )
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_PACKAGE_ROOT,
        canonical_json_bytes,
        hash_bytes_blake3,
        hash_file_blake3,
        hash_file_sha256,
        manifest_content_hash,
    )


def _verify_chain(chain: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not chain:
        return False, "empty_chain"

    prev = "0" * 64
    for rec in chain:
        payload = {
            "manifest_blake3": rec["manifest_blake3"],
            "previous_chain_blake3": rec["previous_chain_blake3"],
            "timestamp_utc": rec["timestamp_utc"],
            "tier": rec["tier"],
            "stage": rec["stage"],
            "policy_version": rec["policy_version"],
        }
        expected_chain = hash_bytes_blake3(canonical_json_bytes(payload))
        if rec["chain_blake3"] != expected_chain:
            return False, f"chain_hash_mismatch:seq:{rec.get('seq')}"
        if rec["previous_chain_blake3"] != prev:
            return False, f"chain_link_mismatch:seq:{rec.get('seq')}"

        verify_key = VerifyKey(bytes.fromhex(rec["signer_pubkey"]))
        try:
            verify_key.verify(rec["chain_blake3"].encode("utf-8"), bytes.fromhex(rec["ed25519_signature"]))
        except BadSignatureError:
            return False, f"signature_invalid:seq:{rec.get('seq')}"

        prev = rec["chain_blake3"]

    return True, None


def run(package_root: Path, tier: str, stage: str) -> int:
    tier_root = package_root / tier
    manifest_path = tier_root / "manifest" / "evidence_manifest.json"
    gate_path = tier_root / "gate_reports" / f"{stage}_{tier}_gate_report.json"
    chain_path = package_root / "integrity" / "evidence_chain.json"

    errors: list[str] = []

    if not manifest_path.exists():
        errors.append(f"missing_manifest:{manifest_path}")
    if not gate_path.exists():
        errors.append(f"missing_gate_report:{gate_path}")
    if not chain_path.exists():
        errors.append(f"missing_chain:{chain_path}")

    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])

    content_hash = manifest_content_hash(
        str(manifest.get("stage", stage)),
        str(manifest.get("tier", tier)),
        str(manifest.get("policy_version", "evidence-v1.0")),
        entries,
    )
    if content_hash != manifest.get("manifest_content_hash"):
        errors.append("manifest_content_hash_mismatch")

    # Validate entry file hashes where copied bytes exist.
    for entry in entries:
        copied_path = entry.get("copied_path")
        if copied_path:
            fp = tier_root / str(copied_path)
            if not fp.exists() or not fp.is_file():
                errors.append(f"missing_copied_file:{copied_path}")
                continue
            sha256 = hash_file_sha256(fp)
            blake3 = hash_file_blake3(fp)
            if entry.get("sha256") and sha256 != entry.get("sha256"):
                errors.append(f"sha256_mismatch:{copied_path}")
            if entry.get("blake3") and blake3 != entry.get("blake3"):
                errors.append(f"blake3_mismatch:{copied_path}")

        if tier == "public_redacted" and entry.get("public_mode") == "hash_metadata":
            redacted_fp = tier_root / str(entry.get("logical_path"))
            if redacted_fp.exists():
                errors.append(f"redaction_violation:{entry.get('logical_path')}")

    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    ok_chain, chain_error = _verify_chain(chain)
    if not ok_chain:
        errors.append(str(chain_error))
    else:
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_blake3 = hash_bytes_blake3(manifest_bytes)
        stage_in_manifest = str(manifest.get("stage", stage))
        matching = [
            rec
            for rec in chain
            if rec.get("tier") == tier and rec.get("stage") == stage_in_manifest
        ]
        if not matching:
            errors.append("missing_chain_record_for_tier_stage")
        else:
            latest_for_tier = matching[-1]
            if latest_for_tier.get("manifest_blake3") != manifest_blake3:
                errors.append("tier_stage_chain_manifest_mismatch")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required_gate_fields = [
        "research_discovered_count",
        "research_indexed_count",
        "research_unindexed_count",
        "founding_docs_present",
        "theological_bridge_present",
        "passed",
    ]
    for field in required_gate_fields:
        if field not in gate:
            errors.append(f"missing_gate_field:{field}")

    if not bool(gate.get("passed", False)):
        errors.append("gate_not_passed")

    output = {
        "passed": len(errors) == 0,
        "manifest_path": manifest_path.as_posix(),
        "gate_report_path": gate_path.as_posix(),
        "chain_path": chain_path.as_posix(),
        "errors": errors,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

    return 0 if len(errors) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evidence package integrity and gate status")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--tier", choices=["private_full", "public_redacted"], default="private_full")
    parser.add_argument("--stage", choices=["scaffold", "final"], default="final")
    args = parser.parse_args()
    raise SystemExit(run(package_root=args.package_root, tier=args.tier, stage=args.stage))


if __name__ == "__main__":
    main()
