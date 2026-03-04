from __future__ import annotations

from pathlib import Path

import yaml

from scripts.ops.elite_fullstack_blueprint_audit import audit_repo


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_phase65_config(repo_root: Path) -> None:
    _write(
        repo_root / "config/phase65_masterpiece_roadmap.yaml",
        yaml.safe_dump(
            {
                "quality_gates": {
                    "required": {
                        "min_avg_ihsan": 0.75,
                        "min_speedup_system1_vs_system2": 8.0,
                        "max_avg_latency_ms": 2200.0,
                        "signed_receipts_required": True,
                    },
                    "scoring": {"min_snr_score": 0.90},
                }
            },
            sort_keys=False,
        ),
    )


def _minimal_cfg() -> dict:
    return {
        "program": {"id": "elite", "version": "1.0.0"},
        "pmbok_domains": ["initiating", "planning"],
        "checks": {
            "files": {
                "workflows": [
                    ".github/workflows/ci.yml",
                    ".github/workflows/phase65-masterpiece.yml",
                ],
                "scripts": [
                    "scripts/ops/phase65_masterpiece_runner.py",
                ],
                "docs": [
                    "ROADMAP.md",
                    "COMMUNITY.md",
                ],
            },
            "jobs": {
                ".github/workflows/ci.yml": ["phase65-masterpiece-gate"],
                ".github/workflows/phase65-masterpiece.yml": ["phase65-gate"],
            },
            "readme": {
                "required_patterns": [
                    "[![CI Status]",
                    "[![Roadmap]",
                    "## Community",
                ]
            },
            "phase65_thresholds": {
                "min_snr_score": 0.90,
                "min_avg_ihsan": 0.75,
                "min_speedup_system1_vs_system2": 8.0,
                "max_avg_latency_ms": 2200.0,
                "signed_receipts_required": True,
            },
        },
        "scoring": {
            "min_score": 0.95,
            "weights": {
                "files": 0.30,
                "jobs": 0.20,
                "readme": 0.10,
                "thresholds": 0.40,
            },
        },
    }


def test_elite_blueprint_audit_passes_with_complete_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump({"jobs": {"phase65-masterpiece-gate": {}}}, sort_keys=False),
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _seed_phase65_config(repo)

    report = audit_repo(repo, _minimal_cfg())
    assert report["gate_passed"] is True
    assert report["hard_fail"] is False
    assert report["weighted_score"] >= 0.95


def test_elite_blueprint_audit_fails_when_requirements_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml", yaml.safe_dump({"jobs": {}}, sort_keys=False)
    )
    _write(repo / "README.md", "# partial\n")
    _seed_phase65_config(repo)

    report = audit_repo(repo, _minimal_cfg())
    assert report["gate_passed"] is False
    assert report["hard_fail"] is True
    assert len(report["failed_checks"]) > 0


def test_elite_blueprint_audit_checks_ethical_invariants(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump(
            {"jobs": {"phase65-masterpiece-gate": {}, "deploy": {"needs": []}}},
            sort_keys=False,
        ),
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _seed_phase65_config(repo)
    _write(
        repo / "core/integration/constants.py",
        "KERNEL_INVARIANTS = ('RIBA_ZERO', 'CLAIM_MUST_BIND')\n",
    )

    cfg = _minimal_cfg()
    cfg["checks"]["ethical_integrity"] = {
        "source_file": "core/integration/constants.py",
        "required_invariants": ["RIBA_ZERO", "CLAIM_MUST_BIND", "IHSAN_FLOOR"],
    }
    cfg["scoring"]["weights"]["ethics"] = 0.10

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is False
    assert any(
        check["name"] == "ethics:invariant:IHSAN_FLOOR"
        for check in report["failed_checks"]
    )


def test_elite_blueprint_audit_emits_priority_roadmap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump({"jobs": {"phase65-masterpiece-gate": {}}}, sort_keys=False),
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _seed_phase65_config(repo)

    cfg = _minimal_cfg()
    cfg["checks"]["phase65_thresholds"]["min_avg_ihsan"] = 0.95

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is False
    assert report["optimization_roadmap"]
    assert report["optimization_roadmap"][0]["priority"] == "P0"
