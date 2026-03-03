#!/usr/bin/env python3
"""
Proof Forge — Evidence Receipt Generator & Chain Manager

Generates SHA-256 hash-chained evidence receipts from development artifacts
and verification results. Maintains an append-only evidence index.

Usage:
    # Generate a new evidence receipt
    forge_evidence.py --project-dir ./my-project \
                      --description "Implemented JWT auth module" \
                      --verification-report ./verification.json

    # Generate receipt with explicit artifact list
    forge_evidence.py --project-dir ./my-project \
                      --description "Fixed memory leak in parser" \
                      --artifacts src/parser.rs src/lib.rs \
                      --verification-report ./verification.json

    # Verify existing chain integrity
    forge_evidence.py --verify --project-dir ./my-project

    # Generate genesis receipt (first in chain)
    forge_evidence.py --project-dir ./my-project \
                      --description "Project initialization" \
                      --genesis
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# ─── Ed25519 Signing (True Spearpoint) ────────────────────────────────────────
# Standing on: Bernstein (2011) — Ed25519 for tamper-evident evidence.

OPERATOR_KEY_FILE = ".proof-forge/operator_key.json"


def _load_or_create_operator_key(project_dir: Path) -> tuple:
    """Load existing operator Ed25519 keypair, or generate one.

    Returns (private_key_hex, public_key_hex).
    """
    key_path = project_dir / OPERATOR_KEY_FILE
    if key_path.exists():
        with open(key_path) as f:
            data = json.load(f)
        return data["private_key"], data["public_key"]

    try:
        from core.pci.crypto import generate_keypair

        private_key, public_key = generate_keypair()
    except ImportError:
        # Fallback if core.pci not available
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = Ed25519PrivateKey.generate()
        private_key = sk.private_bytes_raw().hex()
        public_key = sk.public_key().public_bytes_raw().hex()

    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "w") as f:
        json.dump({"private_key": private_key, "public_key": public_key}, f, indent=2)
    print(f"  🔑 Generated new operator key: {key_path}")
    return private_key, public_key


def _sign_receipt(receipt_body: str, private_key_hex: str) -> str:
    """Sign receipt body (canonical JSON) with Ed25519.

    Returns hex-encoded signature.
    """
    try:
        from core.pci.crypto import sign_message, domain_separated_digest

        digest = domain_separated_digest(receipt_body.encode("utf-8"))
        return sign_message(digest, private_key_hex)
    except ImportError:
        # Fallback
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        sig = sk.sign(receipt_body.encode("utf-8"))
        return sig.hex()


def _canonical_receipt_body(receipt: dict) -> str:
    """Return deterministic receipt body used for Ed25519 signing/verification."""
    body = {k: v for k, v in receipt.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _verify_receipt_signature(receipt: dict) -> Tuple[bool, str]:
    """Verify receipt signature over canonical body.

    Returns (is_valid, reason_code).
    """
    signature = receipt.get("signature")
    if not isinstance(signature, dict):
        return False, "signature_missing"

    if signature.get("algorithm") != "Ed25519":
        return False, "signature_algorithm_invalid"

    sig_hex = signature.get("value")
    pub_hex = signature.get("signer_pubkey")
    if not isinstance(sig_hex, str) or not isinstance(pub_hex, str):
        return False, "signature_fields_invalid"
    if not sig_hex or not pub_hex:
        return False, "signature_fields_invalid"

    receipt_body = _canonical_receipt_body(receipt)
    body_bytes = receipt_body.encode("utf-8")

    # Preferred path: domain-separated digest verification via core.pci.crypto
    try:
        from core.pci.crypto import domain_separated_digest, verify_signature

        digest = domain_separated_digest(body_bytes)
        if verify_signature(digest, sig_hex, pub_hex):
            return True, "ok"
    except Exception:
        # Fall through to raw-message verification for legacy fallback signatures.
        pass

    # Compatibility path: raw message signing fallback used when core.pci.crypto
    # is unavailable in older environments.
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pk.verify(bytes.fromhex(sig_hex), body_bytes)
        return True, "ok_raw"
    except Exception:
        return False, "signature_invalid"


# ─── Constants ───────────────────────────────────────────────────────────────

PROOF_DIR = ".proof-forge"
RECEIPTS_DIR = "receipts"
SUMMARIES_DIR = "summaries"
INDEX_FILE = "EVIDENCE_INDEX.json"
GENESIS_HASH = "0" * 64  # SHA-256 zero hash for genesis block

# File extensions to auto-discover (when no explicit list given)
CODE_EXTENSIONS = {
    ".rs",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".zig",
    ".wasm",
}
TEST_PATTERNS = {"test", "spec", "_test", "test_", "tests", "specs"}
CONFIG_EXTENSIONS = {".toml", ".yaml", ".yml", ".json", ".xml", ".ini", ".cfg"}
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".adoc", ".org"}
DATA_EXTENSIONS = {".csv", ".tsv", ".sql", ".parquet", ".arrow"}

# Max file size to hash (skip large binaries)
MAX_HASH_SIZE = 50 * 1024 * 1024  # 50 MB


# ─── Hashing ─────────────────────────────────────────────────────────────────


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError) as e:
        return f"ERROR:{e}"


def sha256_string(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_evidence_hash(artifacts: list, verification: dict, description: str) -> str:
    """
    Compute a composite evidence hash over all inputs.

    Uses canonical JSON serialization to ensure deterministic hashing.
    The evidence hash represents the content of THIS receipt specifically.
    """
    canonical = json.dumps(
        {
            "artifacts": sorted(artifacts, key=lambda a: a["path"]),
            "verification": verification,
            "description": description,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_string(canonical)


def compute_chain_hash(evidence_hash: str, previous_hash: str, timestamp: str) -> str:
    """
    Compute the chain hash that links this receipt to the previous one.

    chain_hash = SHA-256(evidence_hash + previous_hash + timestamp)

    This is the value that the NEXT receipt will reference as previous_hash.
    """
    combined = f"{evidence_hash}{previous_hash}{timestamp}"
    return sha256_string(combined)


# ─── Artifact Discovery ─────────────────────────────────────────────────────


def classify_file(filepath: Path) -> str:
    """Classify a file by type."""
    name = filepath.name.lower()
    suffix = filepath.suffix.lower()

    # Test files
    if any(pat in name for pat in TEST_PATTERNS):
        return "test"

    # Code
    if suffix in CODE_EXTENSIONS:
        return "code"

    # Config
    if suffix in CONFIG_EXTENSIONS or name in {
        "makefile",
        "dockerfile",
        "rakefile",
        "gemfile",
        "cargo.toml",
        "package.json",
        "pyproject.toml",
    }:
        return "config"

    # Documentation
    if suffix in DOC_EXTENSIONS:
        return "doc"

    # Data
    if suffix in DATA_EXTENSIONS:
        return "data"

    return "binary"


def discover_artifacts(project_dir: Path, explicit_paths: list = None) -> list:
    """
    Discover and hash all relevant artifacts in the project.

    If explicit_paths is provided, only those files are included.
    Otherwise, walks the directory tree, skipping common noise.
    """
    artifacts = []
    skip_dirs = {
        ".git",
        ".proof-forge",
        "node_modules",
        "target",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "vendor",
        ".venv",
        "venv",
        "env",
        "_unzipped_spearpoint",
        "_unzipped_files",
        "external_links",
        # Data lake intake/raw/processed dirs (too large for full hashing)
        "00_INTAKE",
        "01_RAW",
        "02_PROCESSED",
        "03_INDEXED",
    }

    if explicit_paths:
        for p in explicit_paths:
            fp = project_dir / p
            if fp.is_file():
                artifacts.append(_make_artifact(fp, project_dir))
    else:
        for root, dirs, files in os.walk(project_dir):
            # Prune skip dirs
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

            for fname in files:
                fp = Path(root) / fname
                if fname.startswith("."):
                    continue
                try:
                    if fp.stat().st_size > MAX_HASH_SIZE:
                        continue
                    artifacts.append(_make_artifact(fp, project_dir))
                except (OSError, PermissionError, ValueError):
                    # Skip inaccessible files (symlinks, junctions, locked)
                    continue

    return artifacts


def _make_artifact(filepath: Path, project_dir: Path) -> dict:
    """Create an artifact metadata record for a single file."""
    stat = filepath.stat()
    return {
        "path": str(filepath.relative_to(project_dir)),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(filepath),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "type": classify_file(filepath),
    }


# ─── Chain Management ────────────────────────────────────────────────────────


def ensure_proof_dir(project_dir: Path) -> Path:
    """Create .proof-forge directory structure if it doesn't exist."""
    proof_dir = project_dir / PROOF_DIR
    (proof_dir / RECEIPTS_DIR).mkdir(parents=True, exist_ok=True)
    (proof_dir / SUMMARIES_DIR).mkdir(parents=True, exist_ok=True)
    return proof_dir


def load_index(proof_dir: Path) -> dict:
    """Load or initialize the evidence index."""
    index_path = proof_dir / INDEX_FILE
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {
        "schema_version": "1.0.0",
        "project": str(proof_dir.parent.name),
        "genesis_timestamp": None,
        "chain_length": 0,
        "latest_hash": None,
        "latest_receipt": None,
        "receipts": [],
    }


def save_index(proof_dir: Path, index: dict):
    """Write the evidence index to disk."""
    index_path = proof_dir / INDEX_FILE
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)
    print(f"  ✅ Updated {INDEX_FILE}")


def get_previous_hash(index: dict) -> str:
    """Get the chain hash of the most recent receipt, or genesis hash."""
    if index["chain_length"] == 0:
        return GENESIS_HASH
    return index["latest_hash"]


def get_confidence_level(verification: dict) -> dict:
    """
    Compute confidence level from verification results.

    Returns: {"level": int, "label": str, "criteria": str}
    """
    if not verification or not verification.get("checks"):
        return {
            "level": 1,
            "label": "Logged",
            "criteria": "Evidence collected, no verification run",
        }

    checks = verification.get("checks", [])
    types_passed = set()
    any_passed = False

    for check in checks:
        if check.get("passed"):
            any_passed = True
            types_passed.add(check.get("type", "unknown"))

    if not any_passed:
        if verification.get("checks_run", 0) == 0:
            # Manual attestation
            if verification.get("manual_attestation"):
                return {
                    "level": 2,
                    "label": "Attested",
                    "criteria": "Manual attestation provided",
                }
            return {
                "level": 1,
                "label": "Logged",
                "criteria": "Evidence collected, no verification run",
            }
        return {"level": 1, "label": "Logged", "criteria": "Checks ran but none passed"}

    type_count = len(types_passed)
    has_tests = "test_suite" in types_passed
    has_bench = "benchmark" in types_passed
    has_static = "static_analysis" in types_passed
    has_schema = "schema_validation" in types_passed

    if has_tests and type_count >= 4:
        return {
            "level": 5,
            "label": "Ironclad",
            "criteria": f"Tests + {type_count - 1} additional verification types",
        }
    if has_tests and type_count >= 2:
        return {
            "level": 4,
            "label": "Strong",
            "criteria": f"Tests + {type_count - 1} additional check(s)",
        }
    if has_tests or (has_static and type_count >= 2):
        return {"level": 3, "label": "Solid", "criteria": "Core verification passed"}
    if any_passed:
        return {"level": 2, "label": "Attested", "criteria": "Partial verification"}

    return {"level": 1, "label": "Logged", "criteria": "Minimal verification"}


# ─── Receipt Generation ──────────────────────────────────────────────────────


def forge_receipt(
    project_dir: Path,
    description: str,
    verification: dict = None,
    explicit_artifacts: list = None,
) -> dict:
    """
    Generate a complete evidence receipt.

    Returns the receipt dict and writes it to disk.
    """
    proof_dir = ensure_proof_dir(project_dir)
    index = load_index(proof_dir)

    # Timestamp
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    file_timestamp = now.strftime("%Y-%m-%d_%H%M%S")

    print(f"\n🔨 PROOF FORGE — Generating Evidence Receipt")
    print(f"  📂 Project: {project_dir}")
    print(f"  📝 Description: {description}")
    print(f"  ⏰ Timestamp: {timestamp}")

    # Phase 1: Discover artifacts
    print(f"\n  📦 Phase 1: Discovering artifacts...")
    artifacts = discover_artifacts(project_dir, explicit_artifacts)
    print(f"     Found {len(artifacts)} artifacts")

    artifact_summary = {}
    for a in artifacts:
        t = a["type"]
        artifact_summary[t] = artifact_summary.get(t, 0) + 1
    for t, count in sorted(artifact_summary.items()):
        print(f"     • {t}: {count} files")

    # Phase 2: Load verification
    print(f"\n  🔍 Phase 2: Processing verification report...")
    if verification is None:
        verification = {
            "checks": [],
            "checks_run": 0,
            "checks_passed": 0,
            "overall_pass": None,
        }
        print(
            f"     No verification report provided — receipt will be attestation-only"
        )
    else:
        cr = verification.get("checks_run", 0)
        cp = verification.get("checks_passed", 0)
        print(f"     {cp}/{cr} checks passed")

    # Confidence level
    confidence = get_confidence_level(verification)
    print(f"     Confidence: Level {confidence['level']} — {confidence['label']}")

    # Phase 3: Compute hashes
    print(f"\n  🔐 Phase 3: Computing evidence chain...")
    previous_hash = get_previous_hash(index)
    evidence_hash = compute_evidence_hash(artifacts, verification, description)
    chain_hash = compute_chain_hash(evidence_hash, previous_hash, timestamp)
    chain_position = index["chain_length"] + 1

    print(f"     Evidence hash: {evidence_hash[:16]}...")
    print(f"     Previous hash: {previous_hash[:16]}...")
    print(f"     Chain hash:    {chain_hash[:16]}...")
    print(f"     Chain position: #{chain_position}")

    # Build receipt
    receipt = {
        "schema_version": "1.0.0",
        "receipt_id": f"receipt-{file_timestamp}",
        "timestamp": timestamp,
        "description": description,
        "chain_position": chain_position,
        "hashes": {
            "evidence_hash": evidence_hash,
            "previous_hash": previous_hash,
            "chain_hash": chain_hash,
        },
        "confidence": confidence,
        "artifacts": {
            "count": len(artifacts),
            "summary": artifact_summary,
            "details": artifacts,
        },
        "verification": verification,
    }

    # Phase 4: Sign receipt (Ed25519 — True Spearpoint)
    print(f"\n  ✍️  Phase 4: Signing receipt with Ed25519...")
    try:
        private_key, public_key = _load_or_create_operator_key(project_dir)
        receipt_body = _canonical_receipt_body(receipt)
        signature = _sign_receipt(receipt_body, private_key)
        receipt["signature"] = {
            "algorithm": "Ed25519",
            "value": signature,
            "signer_pubkey": public_key,
        }
        print(f"     Signature: {signature[:32]}...")
        print(f"     Public key: {public_key[:32]}...")
    except Exception as e:
        raise RuntimeError(
            f"Receipt signing failed (fail-closed): {e}. "
            "Set up Ed25519 signing dependencies before forging evidence."
        ) from e

    # Write receipt
    receipt_filename = f"{file_timestamp}.json"
    receipt_path = proof_dir / RECEIPTS_DIR / receipt_filename
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  ✅ Receipt written: {receipt_path}")

    # Update index
    index["chain_length"] = chain_position
    index["latest_hash"] = chain_hash
    index["latest_receipt"] = receipt_filename
    if chain_position == 1:
        index["genesis_timestamp"] = timestamp
    index["receipts"].append(
        {
            "position": chain_position,
            "filename": receipt_filename,
            "timestamp": timestamp,
            "chain_hash": chain_hash,
            "confidence_level": confidence["level"],
            "confidence_label": confidence["label"],
            "description": description[:200],
        }
    )
    save_index(proof_dir, index)

    print(f"\n  🏁 EVIDENCE FORGED — Receipt #{chain_position}")
    print(f"     Chain hash: {chain_hash}")
    print(f"     Confidence: {confidence['label']} (Level {confidence['level']})")

    return receipt, str(receipt_path)


# ─── Chain Verification ──────────────────────────────────────────────────────


def verify_chain(project_dir: Path, allow_legacy_unsigned: bool = False) -> dict:
    """
    Verify the integrity of the entire evidence chain.

    Walks every receipt from genesis to latest, recomputing hashes.
    Reports any breaks or inconsistencies.
    """
    proof_dir = project_dir / PROOF_DIR
    index_path = proof_dir / INDEX_FILE

    if not index_path.exists():
        print("❌ No evidence chain found. Run forge_evidence.py first.")
        return {"valid": False, "error": "No chain found"}

    with open(index_path) as f:
        index = json.load(f)

    print(f"\n🔍 CHAIN VERIFICATION — {index['project']}")
    print(f"   Chain length: {index['chain_length']} receipts")
    print(f"   Genesis: {index.get('genesis_timestamp', 'unknown')}")
    print()

    results = []
    expected_previous = GENESIS_HASH

    for entry in index["receipts"]:
        receipt_path = proof_dir / RECEIPTS_DIR / entry["filename"]

        if not receipt_path.exists():
            print(
                f"  ❌ #{entry['position']}: Receipt file missing: {entry['filename']}"
            )
            results.append(
                {"position": entry["position"], "valid": False, "error": "file_missing"}
            )
            continue

        with open(receipt_path) as f:
            receipt = json.load(f)

        hashes = receipt.get("hashes", {})
        if not isinstance(hashes, dict):
            print(f"  ❌ #{entry['position']}: Missing or invalid hashes block")
            results.append(
                {
                    "position": entry["position"],
                    "valid": False,
                    "error": "invalid_hashes_block",
                }
            )
            continue

        receipt_errors = []

        previous_hash = hashes.get("previous_hash")
        chain_hash = hashes.get("chain_hash")
        evidence_hash = hashes.get("evidence_hash")
        if not isinstance(previous_hash, str) or not isinstance(chain_hash, str):
            receipt_errors.append("hash_fields_invalid")

        # Verify chain link
        if previous_hash != expected_previous:
            receipt_errors.append("chain_break")

        # Recompute evidence hash from canonical receipt payload
        recomputed_evidence = None
        try:
            recomputed_evidence = compute_evidence_hash(
                receipt.get("artifacts", {}).get("details", []),
                receipt.get("verification", {}),
                receipt.get("description", ""),
            )
        except Exception:
            receipt_errors.append("evidence_payload_invalid")
        if recomputed_evidence is not None and recomputed_evidence != evidence_hash:
            receipt_errors.append("evidence_hash_mismatch")

        # Recompute chain hash from recomputed evidence hash and linked previous hash
        recomputed_chain = compute_chain_hash(
            recomputed_evidence or "",
            previous_hash or "",
            receipt.get("timestamp", ""),
        )
        if recomputed_chain != chain_hash:
            receipt_errors.append("chain_hash_mismatch")

        sig_ok, sig_reason = _verify_receipt_signature(receipt)
        if not sig_ok:
            if allow_legacy_unsigned and sig_reason == "signature_missing":
                print(
                    f"  ⚠️  #{entry['position']}: Legacy unsigned receipt accepted by override"
                )
            else:
                receipt_errors.append(sig_reason)

        if receipt_errors:
            print(
                f"  ❌ #{entry['position']}: Verification failed ({', '.join(receipt_errors)})"
            )
            results.append(
                {
                    "position": entry["position"],
                    "valid": False,
                    "error": "|".join(receipt_errors),
                }
            )
        else:
            conf = receipt.get("confidence", {})
            label = conf.get("label", "?")
            print(
                f"  ✅ #{entry['position']}: Valid — {label} — {receipt['description'][:60]}"
            )
            results.append({"position": entry["position"], "valid": True})

        if isinstance(chain_hash, str):
            expected_previous = chain_hash

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    total = len(results)
    all_valid = valid_count == total

    print()
    if all_valid:
        print(f"  🏁 CHAIN INTACT — {valid_count}/{total} receipts verified")
    else:
        print(
            f"  ⚠️  CHAIN ISSUES — {valid_count}/{total} receipts valid, {total - valid_count} problems found"
        )

    return {
        "valid": all_valid,
        "total_receipts": total,
        "valid_receipts": valid_count,
        "results": results,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Proof Forge — Evidence Receipt Generator"
    )
    parser.add_argument(
        "--project-dir", required=True, help="Path to project directory"
    )
    parser.add_argument("--description", help="Description of what was built/changed")
    parser.add_argument(
        "--verification-report", help="Path to verification report JSON"
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        help="Explicit list of artifact paths (relative to project-dir)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify existing chain integrity"
    )
    parser.add_argument(
        "--allow-legacy-unsigned",
        action="store_true",
        help="Allow legacy unsigned receipts during chain verification",
    )
    parser.add_argument("--genesis", action="store_true", help="Create genesis receipt")

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    if not project_dir.is_dir():
        print(f"❌ Project directory not found: {project_dir}")
        sys.exit(1)

    if args.verify:
        result = verify_chain(
            project_dir,
            allow_legacy_unsigned=args.allow_legacy_unsigned,
        )
        sys.exit(0 if result["valid"] else 1)

    if not args.description:
        print("❌ --description is required when generating a receipt")
        sys.exit(1)

    # Load verification report if provided
    verification = None
    if args.verification_report:
        vr_path = Path(args.verification_report)
        if vr_path.exists():
            with open(vr_path) as f:
                verification = json.load(f)
        else:
            print(f"⚠️  Verification report not found: {vr_path}, proceeding without")

    receipt, receipt_path = forge_receipt(
        project_dir=project_dir,
        description=args.description,
        verification=verification,
        explicit_artifacts=args.artifacts,
    )

    # Output receipt path for downstream scripts
    print(f"\nRECEIPT_PATH={receipt_path}")


if __name__ == "__main__":
    main()
