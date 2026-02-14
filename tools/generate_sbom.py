#!/usr/bin/env python3
"""
SBOM Generator — Software Bill of Materials (CycloneDX)

Generates a CycloneDX 1.5 SBOM from Python (pip freeze) and Rust (Cargo.lock)
dependencies. Produces a signed manifest suitable for release attestation.

Standing on Giants: OWASP CycloneDX + NTIA Minimum Elements + SPDX

Usage:
    python tools/generate_sbom.py                 # Generate sbom/
    python tools/generate_sbom.py --sign           # Generate + sign with Ed25519
    python tools/generate_sbom.py --output /path   # Custom output directory
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_python_packages() -> List[Dict[str, str]]:
    """Get installed Python packages via pip freeze."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        packages = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if "==" in line:
                name, version = line.split("==", 1)
                packages.append({"name": name.strip(), "version": version.strip()})
            elif line and not line.startswith("#"):
                packages.append({"name": line, "version": "unknown"})
        return packages
    except Exception as e:
        print(f"Warning: pip freeze failed: {e}", file=sys.stderr)
        return []


def _get_rust_packages(cargo_lock: Path) -> List[Dict[str, str]]:
    """Parse Cargo.lock for Rust dependencies."""
    if not cargo_lock.exists():
        return []

    packages = []
    try:
        content = cargo_lock.read_text()
        current_pkg: Optional[Dict[str, str]] = None
        for line in content.splitlines():
            line = line.strip()
            if line == "[[package]]":
                if current_pkg and current_pkg.get("name"):
                    packages.append(current_pkg)
                current_pkg = {}
            elif current_pkg is not None:
                if line.startswith('name = "'):
                    current_pkg["name"] = line.split('"')[1]
                elif line.startswith('version = "'):
                    current_pkg["version"] = line.split('"')[1]
        if current_pkg and current_pkg.get("name"):
            packages.append(current_pkg)
    except Exception as e:
        print(f"Warning: Cargo.lock parse failed: {e}", file=sys.stderr)

    return packages


def generate_cyclonedx(
    python_packages: List[Dict[str, str]],
    rust_packages: List[Dict[str, str]],
    project_name: str = "bizra-data-lake",
    project_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Generate a CycloneDX 1.5 SBOM JSON document."""
    components = []

    for pkg in python_packages:
        components.append({
            "type": "library",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "scope": "required",
        })

    for pkg in rust_packages:
        components.append({
            "type": "library",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:cargo/{pkg['name']}@{pkg['version']}",
            "scope": "required",
        })

    sbom: Dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:32]}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {
                "type": "application",
                "name": project_name,
                "version": project_version,
            },
            "tools": [
                {"name": "bizra-sbom-generator", "version": "1.0.0"},
            ],
        },
        "components": components,
    }

    return sbom


def generate_slsa_provenance(
    sbom: Dict[str, Any],
    sbom_hash: str,
    git_commit: str = "",
    git_repo: str = "",
    builder_id: str = "bizra-node0-builder",
) -> Dict[str, Any]:
    """Generate a SLSA v1.0 provenance attestation for the SBOM.

    Standing on Giants:
    - SLSA (supply-chain Levels for Software Artifacts, Google/OpenSSF)
    - in-toto (NYU, 2019): Attestation framework
    - BIZRA Spearpoint PRD SP-010: "SLSA provenance + SBOM signing"

    Produces an in-toto Statement wrapping a SLSA Provenance predicate.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Resolve git info if not provided
    if not git_commit:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_commit = result.stdout.strip() or "unknown"
        except Exception:
            git_commit = "unknown"

    if not git_repo:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_repo = result.stdout.strip() or "unknown"
        except Exception:
            git_repo = "unknown"

    # Count components for materials
    component_count = len(sbom.get("components", []))

    provenance: Dict[str, Any] = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": "bizra-sbom.cdx.json",
                "digest": {"sha256": sbom_hash},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://bizra.ai/build/v1",
                "externalParameters": {
                    "source": {
                        "uri": git_repo,
                        "digest": {"gitCommit": git_commit},
                    },
                },
                "internalParameters": {
                    "python_version": sys.version.split()[0],
                    "builder_id": builder_id,
                },
                "resolvedDependencies": [
                    {
                        "name": "sbom-components",
                        "count": component_count,
                    }
                ],
            },
            "runDetails": {
                "builder": {
                    "id": builder_id,
                    "version": "1.0.0",
                },
                "metadata": {
                    "invocationId": hashlib.sha256(
                        f"{timestamp}:{git_commit}".encode()
                    ).hexdigest()[:16],
                    "startedOn": timestamp,
                    "finishedOn": timestamp,
                },
            },
        },
    }

    return provenance


def sign_provenance(
    provenance: Dict[str, Any],
    private_key_hex: str,
    public_key_hex: str,
) -> Dict[str, Any]:
    """Sign a SLSA provenance attestation with Ed25519.

    Returns a DSSE (Dead Simple Signing Envelope) containing the
    signed provenance.

    Standing on: Bernstein (2011) — Ed25519,
    DSSE (in-toto/securesystemslib) — envelope format.
    """
    import base64

    from core.pci.crypto import sign_message

    # Canonical JSON payload
    payload = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    # Hash the payload (sign_message expects hex digest, not raw text)
    payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Sign the digest
    signature = sign_message(payload_digest, private_key_hex)

    envelope: Dict[str, Any] = {
        "payloadType": "application/vnd.in-toto+json",
        "payload": payload_b64,
        "signatures": [
            {
                "keyid": hashlib.sha256(public_key_hex.encode()).hexdigest()[:16],
                "sig": signature,
            }
        ],
    }

    return envelope


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate CycloneDX SBOM + SLSA Provenance")
    parser.add_argument("--output", "-o", default="sbom", help="Output directory")
    parser.add_argument("--sign", action="store_true", help="Sign SBOM + provenance with Ed25519")
    parser.add_argument("--slsa", action="store_true", help="Generate SLSA provenance attestation")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting Python packages...")
    python_pkgs = _get_python_packages()
    print(f"  Found {len(python_pkgs)} Python packages")

    cargo_lock = project_root / "bizra-omega" / "Cargo.lock"
    print(f"Parsing Cargo.lock ({cargo_lock})...")
    rust_pkgs = _get_rust_packages(cargo_lock)
    print(f"  Found {len(rust_pkgs)} Rust packages")

    print("Generating CycloneDX SBOM...")
    sbom = generate_cyclonedx(python_pkgs, rust_pkgs)

    sbom_path = output_dir / "bizra-sbom.cdx.json"
    sbom_path.write_text(json.dumps(sbom, indent=2))
    print(f"  Written: {sbom_path}")

    # Checksum
    sbom_hash = hashlib.sha256(sbom_path.read_bytes()).hexdigest()
    checksum_path = output_dir / "sbom-checksums.txt"
    checksum_path.write_text(f"{sbom_hash}  bizra-sbom.cdx.json\n")
    print(f"  Checksum: {checksum_path}")

    # SLSA Provenance Attestation (SP-010)
    if args.slsa or args.sign:
        print("Generating SLSA provenance attestation...")
        provenance = generate_slsa_provenance(sbom, sbom_hash)
        provenance_path = output_dir / "provenance.slsa.json"
        provenance_path.write_text(json.dumps(provenance, indent=2))
        print(f"  Written: {provenance_path}")

    # Optional signing (SBOM + Provenance)
    if args.sign:
        try:
            from core.pci.crypto import generate_keypair, sign_message

            priv_hex, pub_hex = generate_keypair()

            # Sign the SBOM
            sbom_signature = sign_message(sbom_hash, priv_hex)
            sig_bundle = {
                "sbom_hash": sbom_hash,
                "signature": sbom_signature,
                "public_key": pub_hex,
                "algorithm": "Ed25519",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            sig_path = output_dir / "sbom-signature.json"
            sig_path.write_text(json.dumps(sig_bundle, indent=2))
            print(f"  SBOM Signature: {sig_path}")

            # Sign the provenance in DSSE envelope
            envelope = sign_provenance(provenance, priv_hex, pub_hex)
            envelope_path = output_dir / "provenance.dsse.json"
            envelope_path.write_text(json.dumps(envelope, indent=2))
            print(f"  Provenance DSSE: {envelope_path}")
        except ImportError:
            print("  Warning: core.pci.crypto not available, skipping signatures")

    total = len(python_pkgs) + len(rust_pkgs)
    print(f"\nSBOM complete: {total} components across Python + Rust")
    if args.slsa or args.sign:
        print("SLSA provenance attestation generated")


if __name__ == "__main__":
    main()
