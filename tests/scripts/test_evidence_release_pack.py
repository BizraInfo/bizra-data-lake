from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence.build_evidence_package import run as build_run
from scripts.evidence.import_external_assets import run as import_run
from scripts.evidence.release_pack import run as release_pack_run
from scripts.evidence.sign_evidence_package import run as sign_run
from tests.scripts.test_evidence_package import _bootstrap_repo


def _prep_verified_final(ctx: dict[str, Path]) -> None:
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="final",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    assert (
        sign_run(
            package_root=ctx["package_root"],
            tier="private_full",
            config_path=ctx["config"],
        )
        == 0
    )

    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="final",
            tier="public_redacted",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    assert (
        sign_run(
            package_root=ctx["package_root"],
            tier="public_redacted",
            config_path=ctx["config"],
        )
        == 0
    )


def test_release_pack_emits_archives_and_sidecars(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    _prep_verified_final(ctx)

    outdir = ctx["package_root"] / "release"
    rc = release_pack_run(
        package_root=ctx["package_root"],
        stage="final",
        tiers=["private_full", "public_redacted"],
        outdir=outdir,
        skip_verify=False,
        json_stdout=False,
    )
    assert rc == 0

    for tier in ("private_full", "public_redacted"):
        base = outdir / f"{ctx['package_root'].name}.{tier}.tar.gz"
        assert base.exists()
        assert base.with_suffix(base.suffix + ".blake3").exists()
        assert base.with_suffix(base.suffix + ".sha256").exists()
        assert base.with_suffix(base.suffix + ".sig").exists()

    manifest = outdir / f"{ctx['package_root'].name}.release_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload.get("archives", [])) == 2


def test_release_pack_is_deterministic_for_hash_outputs(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    _prep_verified_final(ctx)

    outdir = ctx["package_root"] / "release"
    assert (
        release_pack_run(
            package_root=ctx["package_root"],
            stage="final",
            tiers=["private_full", "public_redacted"],
            outdir=outdir,
            skip_verify=False,
            json_stdout=False,
        )
        == 0
    )

    first = {}
    for tier in ("private_full", "public_redacted"):
        base = outdir / f"{ctx['package_root'].name}.{tier}.tar.gz"
        first[tier] = {
            "blake3": base.with_suffix(base.suffix + ".blake3").read_text(
                encoding="utf-8"
            ),
            "sha256": base.with_suffix(base.suffix + ".sha256").read_text(
                encoding="utf-8"
            ),
            "sig": base.with_suffix(base.suffix + ".sig").read_text(encoding="utf-8"),
        }

    assert (
        release_pack_run(
            package_root=ctx["package_root"],
            stage="final",
            tiers=["private_full", "public_redacted"],
            outdir=outdir,
            skip_verify=False,
            json_stdout=False,
        )
        == 0
    )

    for tier in ("private_full", "public_redacted"):
        base = outdir / f"{ctx['package_root'].name}.{tier}.tar.gz"
        assert (
            base.with_suffix(base.suffix + ".blake3").read_text(encoding="utf-8")
            == first[tier]["blake3"]
        )
        assert (
            base.with_suffix(base.suffix + ".sha256").read_text(encoding="utf-8")
            == first[tier]["sha256"]
        )
        assert (
            base.with_suffix(base.suffix + ".sig").read_text(encoding="utf-8")
            == first[tier]["sig"]
        )


def test_release_pack_fails_when_tier_verification_fails(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    _prep_verified_final(ctx)

    # Tamper private tier after signing/build; release pack should fail on verify.
    tamper_file = (
        ctx["package_root"]
        / "private_full"
        / "00_GENESIS/03_SPIRITUAL_TECHNICAL/ihsan_as_architecture.md"
    )
    tamper_file.write_text("tampered-after-sign", encoding="utf-8")

    outdir = ctx["package_root"] / "release"
    rc = release_pack_run(
        package_root=ctx["package_root"],
        stage="final",
        tiers=["private_full", "public_redacted"],
        outdir=outdir,
        skip_verify=False,
        json_stdout=False,
    )
    assert rc == 1
