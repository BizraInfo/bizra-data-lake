#!/usr/bin/env python3
"""Create signed release archives for evidence package tiers.

Emits for each tier:
- .tar.gz
- .blake3
- .sha256
- .sig (Ed25519 over canonical archive descriptor)
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

from nacl.signing import SigningKey

try:
    from common import (
        DEFAULT_PACKAGE_ROOT,
        canonical_json_bytes,
        hash_file_blake3,
        hash_file_sha256,
        utc_now_iso,
        write_json,
    )
    from verify_evidence_package import run as verify_run
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_PACKAGE_ROOT,
        canonical_json_bytes,
        hash_file_blake3,
        hash_file_sha256,
        utc_now_iso,
        write_json,
    )
    from scripts.evidence.verify_evidence_package import run as verify_run

TIERS = ("private_full", "public_redacted")


def _load_or_create_key(key_path: Path) -> tuple[SigningKey, str]:
    if key_path.exists():
        key_obj = json.loads(key_path.read_text(encoding="utf-8"))
        sk = SigningKey(bytes.fromhex(key_obj["private_key_hex"]))
        return sk, str(key_obj["public_key_hex"])

    sk = SigningKey.generate()
    pk = sk.verify_key.encode().hex()
    write_json(
        key_path,
        {
            "private_key_hex": sk.encode().hex(),
            "public_key_hex": pk,
            "generated_at": utc_now_iso(),
        },
    )
    # Keep signing material private on POSIX systems.
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return sk, pk


def _iter_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.as_posix())
    return files


def _build_deterministic_tar_gz(src_root: Path, archive_path: Path, tier: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with archive_path.open("wb") as out_f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out_f, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for fp in _iter_files(src_root):
                    rel = fp.relative_to(src_root).as_posix()
                    arcname = f"{tier}/{rel}"
                    data = fp.read_bytes()

                    info = tarfile.TarInfo(name=arcname)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    tar.addfile(info, io.BytesIO(data))


def _signature_payload(*, archive_name: str, tier: str, stage: str, blake3: str, sha256: str) -> dict[str, str]:
    return {
        "archive_name": archive_name,
        "tier": tier,
        "stage": stage,
        "blake3": blake3,
        "sha256": sha256,
    }


def _sign_archive(
    signing_key: SigningKey,
    *,
    archive_name: str,
    tier: str,
    stage: str,
    blake3: str,
    sha256: str,
) -> str:
    payload = _signature_payload(
        archive_name=archive_name,
        tier=tier,
        stage=stage,
        blake3=blake3,
        sha256=sha256,
    )
    sig = signing_key.sign(canonical_json_bytes(payload)).signature.hex()
    return sig


def _verify_tier(package_root: Path, tier: str, stage: str) -> tuple[bool, str | None]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = verify_run(package_root=package_root, tier=tier, stage=stage)
    if rc == 0:
        return True, None

    payload = buf.getvalue().strip()
    if not payload:
        return False, "verify_failed"

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return False, f"verify_failed:{payload}"

    errors = parsed.get("errors")
    if isinstance(errors, list):
        return False, ",".join(str(x) for x in errors)
    return False, "verify_failed"


def run(
    *,
    package_root: Path,
    stage: str,
    tiers: Iterable[str],
    outdir: Path,
    skip_verify: bool,
    json_stdout: bool,
) -> int:
    tier_list = [str(t) for t in tiers]
    for tier in tier_list:
        if tier not in TIERS:
            print(f"ERROR: unsupported tier '{tier}'")
            return 1

    outdir.mkdir(parents=True, exist_ok=True)
    integrity_root = package_root / "integrity"
    integrity_root.mkdir(parents=True, exist_ok=True)
    key_path = integrity_root / "operator_signing_key.json"

    signing_key, public_key_hex = _load_or_create_key(key_path)

    archive_records: list[dict[str, Any]] = []

    for tier in tier_list:
        tier_root = package_root / tier
        if not tier_root.exists() or not tier_root.is_dir():
            print(f"ERROR: missing tier directory: {tier_root}")
            return 1

        if not skip_verify:
            ok, reason = _verify_tier(package_root, tier, stage)
            if not ok:
                print(f"ERROR: verification failed for {tier} ({stage}): {reason}")
                return 1

        archive_name = f"{package_root.name}.{tier}.tar.gz"
        archive_path = outdir / archive_name
        _build_deterministic_tar_gz(tier_root, archive_path, tier)

        blake3 = hash_file_blake3(archive_path)
        sha256 = hash_file_sha256(archive_path)
        sig = _sign_archive(
            signing_key,
            archive_name=archive_name,
            tier=tier,
            stage=stage,
            blake3=blake3,
            sha256=sha256,
        )

        (archive_path.with_suffix(archive_path.suffix + ".blake3")).write_text(
            f"{blake3}  {archive_name}\n", encoding="utf-8"
        )
        (archive_path.with_suffix(archive_path.suffix + ".sha256")).write_text(
            f"{sha256}  {archive_name}\n", encoding="utf-8"
        )
        (archive_path.with_suffix(archive_path.suffix + ".sig")).write_text(sig + "\n", encoding="utf-8")

        archive_records.append(
            {
                "tier": tier,
                "stage": stage,
                "archive": archive_name,
                "archive_path": archive_path.as_posix(),
                "blake3": blake3,
                "sha256": sha256,
                "sig": sig,
            }
        )

    release_manifest = {
        "generated_at": utc_now_iso(),
        "package_root": package_root.as_posix(),
        "stage": stage,
        "signer_pubkey": public_key_hex,
        "archives": archive_records,
    }
    manifest_path = outdir / f"{package_root.name}.release_manifest.json"
    write_json(manifest_path, release_manifest)

    output = {
        "passed": True,
        "outdir": outdir.as_posix(),
        "release_manifest": manifest_path.as_posix(),
        "archives": archive_records,
    }

    if json_stdout:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Release manifest: {manifest_path}")
        for rec in archive_records:
            print(f"- {rec['archive']} ({rec['tier']})")
            print(f"  blake3={rec['blake3']}")
            print(f"  sha256={rec['sha256']}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack evidence tiers into signed release archives")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--stage", choices=["scaffold", "final"], default="final")
    parser.add_argument(
        "--tiers",
        default="private_full,public_redacted",
        help="Comma-separated tiers (default: private_full,public_redacted)",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    outdir = args.outdir or (args.package_root / "release")

    raise SystemExit(
        run(
            package_root=args.package_root,
            stage=args.stage,
            tiers=tiers,
            outdir=outdir,
            skip_verify=args.skip_verify,
            json_stdout=args.json,
        )
    )


if __name__ == "__main__":
    main()
