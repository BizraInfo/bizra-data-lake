from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from scripts.evidence.build_evidence_package import run as build_run
from scripts.evidence.export_public_redacted import run as export_run
from scripts.evidence.import_external_assets import run as import_run
from scripts.evidence.sign_evidence_package import run as sign_run
from scripts.evidence.verify_evidence_package import run as verify_run


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _bootstrap_repo(tmp_path: Path, *, include_ihsan: bool = True) -> dict[str, Path]:
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    package_root = repo / "artifacts" / "evidence" / "BIZRA-EVIDENCE-PACKAGE-v1.0-GENESIS"

    (external).mkdir(parents=True, exist_ok=True)
    risalah_src = external / "the_massage.pdf"
    bazrah_src = external / "al_bazrah.pdf"
    risalah_src.write_bytes(b"risalah-source-pdf")
    bazrah_src.write_bytes(b"bazrah-source-pdf")

    (repo / "research_archive").mkdir(parents=True, exist_ok=True)
    (repo / "research_archive" / "r1.md").write_text("research-1", encoding="utf-8")

    if include_ihsan:
        ihsan = repo / "00_GENESIS" / "03_SPIRITUAL_TECHNICAL" / "ihsan_as_architecture.md"
        ihsan.parent.mkdir(parents=True, exist_ok=True)
        ihsan.write_text("ihsan-bridge", encoding="utf-8")

    config_path = repo / "scripts" / "evidence" / "config" / "evidence_package.yaml"
    gate_path = repo / "scripts" / "evidence" / "config" / "final_gate.yaml"

    cfg = {
        "policy_version": "evidence-v1.0",
        "package": {
            "root": str(package_root),
            "tiers": ["private_full", "public_redacted"],
            "default_stage": "scaffold",
        },
        "genesis_anchor": "Ω",
        "founding_import": {
            "receipt_path": str(package_root / "manifest" / "import_receipt.json"),
            "assets": [
                {
                    "logical_path": "00_GENESIS/01_ARABIC_FOUNDING/al_risalah_original.pdf",
                    "source": str(risalah_src),
                    "expected_sha256": _sha256(risalah_src),
                    "source_root": str(external),
                    "discovery_rule": "source_lock",
                    "visibility": "private_only",
                    "public_mode": "hash_metadata",
                    "classification": "founding_document",
                    "required_scaffold": True,
                    "required_final": True,
                },
                {
                    "logical_path": "00_GENESIS/01_ARABIC_FOUNDING/al_bazrah_original.pdf",
                    "source": str(bazrah_src),
                    "expected_sha256": _sha256(bazrah_src),
                    "source_root": str(external),
                    "discovery_rule": "source_lock",
                    "visibility": "private_only",
                    "public_mode": "hash_metadata",
                    "classification": "founding_document",
                    "required_scaffold": True,
                    "required_final": True,
                },
            ],
        },
        "artifacts": [
            {
                "logical_path": "00_GENESIS/03_SPIRITUAL_TECHNICAL/ihsan_as_architecture.md",
                "source": "00_GENESIS/03_SPIRITUAL_TECHNICAL/ihsan_as_architecture.md",
                "source_root": "00_GENESIS/03_SPIRITUAL_TECHNICAL",
                "discovery_rule": "canonical_repo_path",
                "visibility": "both",
                "public_mode": "full",
                "classification": "theological_bridge",
                "required_scaffold": False,
                "required_final": True,
            }
        ],
        "research_policy": {
            "gate_mode": "manifest_completeness",
            "fixed_minimum": None,
            "approved_roots": ["research_archive"],
            "exclude_patterns": [],
        },
    }
    gate_cfg = {
        "policy_version": "evidence-v1.0",
        "fail_closed": True,
        "stages": {
            "scaffold": {
                "required_gates": ["FOUNDING_DOCS_PRESENT", "REQUIRED_ARTIFACTS_PRESENT"],
            },
            "final": {
                "required_gates": [
                    "FOUNDING_DOCS_PRESENT",
                    "REQUIRED_ARTIFACTS_PRESENT",
                    "THEOLOGICAL_BRIDGE_PRESENT",
                    "RESEARCH_MANIFEST_COMPLETE",
                ]
            },
        },
    }

    _write_yaml(config_path, cfg)
    _write_yaml(gate_path, gate_cfg)

    return {
        "repo": repo,
        "external": external,
        "package_root": package_root,
        "config": config_path,
        "gate": gate_path,
    }


def _read_gate(package_root: Path, tier: str, stage: str) -> dict:
    return json.loads(
        (package_root / tier / "gate_reports" / f"{stage}_{tier}_gate_report.json").read_text(
            encoding="utf-8"
        )
    )


def _read_manifest(package_root: Path, tier: str) -> dict:
    return json.loads(
        (package_root / tier / "manifest" / "evidence_manifest.json").read_text(encoding="utf-8")
    )


def test_scaffold_fails_without_risalah_or_bazrah(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)

    rc = build_run(
        config_path=ctx["config"],
        gate_config_path=ctx["gate"],
        stage="scaffold",
        tier="private_full",
        repo_root=ctx["repo"],
        package_root=ctx["package_root"],
        allow_fail=False,
        json_stdout=True,
    )

    gate = _read_gate(ctx["package_root"], "private_full", "scaffold")
    assert rc == 1
    assert gate["founding_docs_present"] is False
    assert "GATE_FAILED:FOUNDING_DOCS_PRESENT" in gate["reasons"]


def test_scaffold_passes_with_founding_docs_and_hash_metadata_export(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="public_redacted",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )

    manifest = _read_manifest(ctx["package_root"], "public_redacted")
    risalah = next(
        e for e in manifest["entries"] if e["logical_path"].endswith("al_risalah_original.pdf")
    )
    assert risalah["public_mode"] == "hash_metadata"
    assert risalah["copied_path"] is None
    assert not (
        ctx["package_root"]
        / "public_redacted"
        / "00_GENESIS/01_ARABIC_FOUNDING/al_risalah_original.pdf"
    ).exists()


def test_final_fails_without_ihsan_as_architecture(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path, include_ihsan=False)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    rc = build_run(
        config_path=ctx["config"],
        gate_config_path=ctx["gate"],
        stage="final",
        tier="private_full",
        repo_root=ctx["repo"],
        package_root=ctx["package_root"],
        allow_fail=False,
        json_stdout=False,
    )

    gate = _read_gate(ctx["package_root"], "private_full", "final")
    assert rc == 1
    assert gate["theological_bridge_present"] is False
    assert "GATE_FAILED:THEOLOGICAL_BRIDGE_PRESENT" in gate["reasons"]


def test_research_manifest_completeness_fails_when_discovered_unindexed(
    tmp_path: Path, monkeypatch
) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    import scripts.evidence.build_evidence_package as beb

    orig_hash = beb.hash_file_blake3

    def _broken(path: Path) -> str:
        if path.name == "r1.md":
            raise OSError("simulated read failure")
        return orig_hash(path)

    monkeypatch.setattr(beb, "hash_file_blake3", _broken)

    rc = build_run(
        config_path=ctx["config"],
        gate_config_path=ctx["gate"],
        stage="final",
        tier="private_full",
        repo_root=ctx["repo"],
        package_root=ctx["package_root"],
        allow_fail=False,
        json_stdout=False,
    )

    gate = _read_gate(ctx["package_root"], "private_full", "final")
    assert rc == 1
    assert gate["research_unindexed_count"] > 0
    assert "GATE_FAILED:RESEARCH_MANIFEST_COMPLETE" in gate["reasons"]


def test_research_manifest_completeness_passes_after_index_sync(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    rc = build_run(
        config_path=ctx["config"],
        gate_config_path=ctx["gate"],
        stage="final",
        tier="private_full",
        repo_root=ctx["repo"],
        package_root=ctx["package_root"],
        allow_fail=False,
        json_stdout=False,
    )

    gate = _read_gate(ctx["package_root"], "private_full", "final")
    assert rc == 0
    assert gate["research_discovered_count"] == gate["research_indexed_count"]
    assert gate["research_unindexed_count"] == 0


def test_public_redacted_excludes_private_founding_pdf_bytes_but_keeps_metadata_hash(
    tmp_path: Path,
) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0
    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )

    assert export_run(ctx["package_root"], "private_full", "public_redacted") == 0
    manifest = _read_manifest(ctx["package_root"], "public_redacted")

    bazrah = next(
        e for e in manifest["entries"] if e["logical_path"].endswith("al_bazrah_original.pdf")
    )
    assert bazrah["sha256"] is not None
    assert bazrah["copied_path"] is None
    assert not (
        ctx["package_root"]
        / "public_redacted"
        / "00_GENESIS/01_ARABIC_FOUNDING/al_bazrah_original.pdf"
    ).exists()


def test_chain_verification_detects_tamper(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0
    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    assert sign_run(package_root=ctx["package_root"], tier="private_full", config_path=ctx["config"]) == 0
    assert verify_run(package_root=ctx["package_root"], tier="private_full", stage="scaffold") == 0

    tamper_path = (
        ctx["package_root"]
        / "private_full"
        / "00_GENESIS/03_SPIRITUAL_TECHNICAL/ihsan_as_architecture.md"
    )
    tamper_path.write_text("tampered", encoding="utf-8")

    assert verify_run(package_root=ctx["package_root"], tier="private_full", stage="scaffold") == 1


def test_signing_key_file_permissions_are_restricted(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0
    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    assert sign_run(package_root=ctx["package_root"], tier="private_full", config_path=ctx["config"]) == 0

    key_path = ctx["package_root"] / "integrity" / "operator_signing_key.json"
    assert key_path.exists()

    # POSIX world/group bits must be clear for private signing material when
    # filesystem metadata supports chmod semantics (e.g., not WSL /mnt/c).
    if os.name != "nt":
        if key_path.as_posix().startswith("/mnt/"):
            pytest.skip("POSIX mode bits are not reliably enforced on mounted filesystems")
        mode = stat.S_IMODE(key_path.stat().st_mode)
        assert (mode & 0o077) == 0


def test_deterministic_manifest_ordering_and_stable_hash(tmp_path: Path) -> None:
    ctx = _bootstrap_repo(tmp_path)
    assert import_run(config_path=ctx["config"], repo_root=ctx["repo"]) == 0

    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    m1 = _read_manifest(ctx["package_root"], "private_full")["manifest_content_hash"]

    assert (
        build_run(
            config_path=ctx["config"],
            gate_config_path=ctx["gate"],
            stage="scaffold",
            tier="private_full",
            repo_root=ctx["repo"],
            package_root=ctx["package_root"],
            allow_fail=False,
            json_stdout=False,
        )
        == 0
    )
    m2 = _read_manifest(ctx["package_root"], "private_full")["manifest_content_hash"]

    assert m1 == m2
