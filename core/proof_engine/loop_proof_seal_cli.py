"""CLI for loop proof seal operations: verify, write-sidecar, canonicalize."""

import json
import sys
from pathlib import Path

from core.proof_engine.loop_proof_seal import (
    SealStatus,
    canonicalize_proof,
    sidecar_path_for,
    verify_seal,
    write_sidecar,
)


def cmd_verify(proof_path: Path):
    """Verify seal status of a proof artifact."""
    status = verify_seal(proof_path)
    print(f"Proof:    {status.proof_path}")
    print(f"State:    {status.state}")
    print(f"Canonical: {status.is_canonical}")
    if status.manifest_hash:
        print(f"Manifest: {status.manifest_hash}")
    if status.sidecar_path:
        print(f"Sidecar:  {status.sidecar_path}")
    if status.error:
        print(f"Error:    {status.error}")


def cmd_write_sidecar(proof_path: Path, signature_hex: str, pubkey_hex: str):
    """Write a signature sidecar after Human signs the manifest hash."""
    out = write_sidecar(proof_path, signature_hex, pubkey_hex)
    print(f"Sidecar written: {out}")
    status = verify_seal(proof_path)
    print(f"Verification:    {status.state}")
    if status.error:
        print(f"Error:           {status.error}")


def cmd_canonicalize(proof_path: Path):
    """Mark proof as canonical after successful verification."""
    ok = canonicalize_proof(proof_path)
    if ok:
        print(f"Canonicalized: {proof_path}")
    else:
        print(f"Failed to canonicalize. Run 'verify' first.")
        sys.exit(1)


def cmd_show_hash(proof_path: Path):
    """Show the manifest hash for Human to sign offline."""
    proof = json.loads(proof_path.read_text())
    manifest = proof.get("manifest_hash", "")
    print(f"Proof:         {proof_path}")
    print(f"Manifest hash: {manifest}")
    print()
    print("To sign (example with openssl):")
    print(
        f"  echo -n '{manifest}' | openssl pkeyutl -sign -inkey key.pem | xxd -p -c 64"
    )
    print()
    print("Then provide signature:")
    print(
        f"  python -m core.proof_engine.loop_proof_seal_cli write-sidecar {proof_path} <signature_hex> <pubkey_hex>"
    )


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m core.proof_engine.loop_proof_seal_cli verify <proof.json>")
        print(
            "  python -m core.proof_engine.loop_proof_seal_cli show-hash <proof.json>"
        )
        print(
            "  python -m core.proof_engine.loop_proof_seal_cli write-sidecar <proof.json> <sig_hex> <pubkey_hex>"
        )
        print(
            "  python -m core.proof_engine.loop_proof_seal_cli canonicalize <proof.json>"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    proof_path = Path(sys.argv[2])

    if cmd == "verify":
        cmd_verify(proof_path)
    elif cmd == "show-hash":
        cmd_show_hash(proof_path)
    elif cmd == "write-sidecar":
        if len(sys.argv) < 5:
            print("Need: <proof.json> <signature_hex> <pubkey_hex>")
            sys.exit(1)
        cmd_write_sidecar(proof_path, sys.argv[3], sys.argv[4])
    elif cmd == "canonicalize":
        cmd_canonicalize(proof_path)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
