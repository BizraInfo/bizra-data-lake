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


def test_elite_blueprint_audit_emits_snr_and_graph_of_thought(tmp_path: Path) -> None:
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
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _seed_phase65_config(repo)

    report = audit_repo(repo, _minimal_cfg())
    assert "snr" in report
    assert report["snr"]["signal"] >= 1
    assert report["snr"]["noise"] == 0
    assert report["snr"]["normalized"] > 0.5
    assert "graph_of_thought" in report
    node_ids = {n["id"] for n in report["graph_of_thought"]["nodes"]}
    assert "files" in node_ids
    assert "release_readiness" in node_ids
    assert report["autonomous_next_step"]["priority"] == "P0"


def test_elite_blueprint_audit_fails_on_invalid_weight_sum(tmp_path: Path) -> None:
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
    cfg["scoring"]["weights"]["thresholds"] = 0.41  # force sum > 1.0

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is False
    assert any(
        check["name"] == "config:weights_sum" for check in report["failed_checks"]
    )


def test_elite_blueprint_audit_fails_on_non_numeric_weight(tmp_path: Path) -> None:
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
    cfg["scoring"]["weights"]["files"] = "high"

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is False
    assert any(
        check["name"] == "config:weights_numeric" for check in report["failed_checks"]
    )


def test_elite_blueprint_audit_checks_docs_truth_and_terminal_contract(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump({"jobs": {"phase65-masterpiece-gate": {}}}, sort_keys=False)
        + "\n# python scripts/ci_docs_truth_gate.py\n"
        + "# needs.performance-gate.result\n"
        + "# Gate 12: Performance Gate\n",
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(
        repo / ".github/workflows/docs-quality.yml",
        "# python scripts/ci_docs_truth_gate.py\n",
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(repo / "scripts/ci_docs_truth_gate.py", "# docs truth gate\n")
    _write(repo / "tests/scripts/test_ci_docs_truth_gate.py", "# tests\n")
    _write(repo / "docs/internal/UNIFIED_ACTIONABLE_FRAMEWORK.md", "# framework\n")
    _write(repo / "docs/internal/DOCS_INDEX.md", "UNIFIED_ACTIONABLE_FRAMEWORK.md\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _write(
        repo / "frontend/src/components/terminal/terminal-manifest.ts",
        "export const PAT_AGENT_MANIFEST = [];\nexport const SAT_AGENT_MANIFEST = [];\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-shell.tsx",
        'import "./terminal-manifest";\nconst motto = "One mission, one proof, remembered forever";\n',
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-mission.tsx",
        "export default 1;\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-timeline.tsx",
        "export default 1;\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-memory.tsx",
        "export default 1;\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-skills.tsx",
        "const x = 'PAT_AGENT_MANIFEST SAT_AGENT_MANIFEST';\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-network.tsx",
        "export default 1;\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-settings.tsx",
        "export default 1;\n",
    )
    _write(
        repo / "frontend/tests/terminal-panels.test.tsx",
        "describe('terminal', () => {});\n",
    )
    _write(
        repo
        / "docs/specs/phase_78_terminal_v1/BIZRA-Terminal-v1-Locked-Build-Contract.md",
        "# locked contract\n",
    )
    _seed_phase65_config(repo)

    cfg = _minimal_cfg()
    cfg["checks"]["docs_truth"] = {
        "required_files": [
            "scripts/ci_docs_truth_gate.py",
            "tests/scripts/test_ci_docs_truth_gate.py",
            "docs/internal/UNIFIED_ACTIONABLE_FRAMEWORK.md",
            "docs/internal/DOCS_INDEX.md",
        ],
        "required_patterns": {
            ".github/workflows/ci.yml": ["python scripts/ci_docs_truth_gate.py"],
            ".github/workflows/docs-quality.yml": [
                "python scripts/ci_docs_truth_gate.py"
            ],
            "docs/internal/DOCS_INDEX.md": ["UNIFIED_ACTIONABLE_FRAMEWORK.md"],
        },
    }
    cfg["checks"]["terminal_contract"] = {
        "required_files": [
            "docs/specs/phase_78_terminal_v1/BIZRA-Terminal-v1-Locked-Build-Contract.md",
            "frontend/src/components/terminal/terminal-manifest.ts",
            "frontend/src/components/terminal/terminal-shell.tsx",
            "frontend/src/components/terminal/terminal-mission.tsx",
            "frontend/src/components/terminal/terminal-timeline.tsx",
            "frontend/src/components/terminal/terminal-memory.tsx",
            "frontend/src/components/terminal/terminal-skills.tsx",
            "frontend/src/components/terminal/terminal-network.tsx",
            "frontend/src/components/terminal/terminal-settings.tsx",
            "frontend/tests/terminal-panels.test.tsx",
        ],
        "required_patterns": {
            "frontend/src/components/terminal/terminal-shell.tsx": [
                "terminal-manifest",
                "One mission, one proof, remembered forever",
            ],
            "frontend/src/components/terminal/terminal-skills.tsx": [
                "PAT_AGENT_MANIFEST",
                "SAT_AGENT_MANIFEST",
            ],
        },
    }

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is True
    assert report["sections"]["docs_truth"]["score"] == 1.0
    assert report["sections"]["terminal"]["score"] == 1.0
    assert "operator_experience" in report["interdisciplinary_lenses"]


def test_elite_blueprint_audit_flags_missing_performance_pattern(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump(
            {"jobs": {"phase65-masterpiece-gate": {}, "performance-gate": {}}},
            sort_keys=False,
        )
        + "\n# needs.performance-gate.result\n",
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(
        repo / ".github/workflows/performance.yml",
        yaml.safe_dump({"jobs": {"performance-gate": {}}}, sort_keys=False),
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(repo / "scripts/ci_release_readiness.py", "# release readiness\n")
    _write(repo / "scripts/ci_quality_gate.py", "# quality gate\n")
    _write(repo / "scripts/quality_spine.py", "# quality spine\n")
    _write(repo / "tests/scripts/test_quality_spine.py", "# tests\n")
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _seed_phase65_config(repo)

    cfg = _minimal_cfg()
    cfg["checks"]["performance_controls"] = {
        "required_files": [
            ".github/workflows/performance.yml",
            "scripts/ci_release_readiness.py",
            "scripts/ci_quality_gate.py",
            "scripts/quality_spine.py",
            "tests/scripts/test_quality_spine.py",
        ],
        "required_jobs": {
            ".github/workflows/ci.yml": ["performance-gate"],
            ".github/workflows/performance.yml": ["performance-gate"],
        },
        "required_patterns": {
            ".github/workflows/ci.yml": [
                "needs.performance-gate.result",
                "Gate 12: Performance Gate",
            ]
        },
    }

    report = audit_repo(repo, cfg)
    assert report["gate_passed"] is False
    assert any(
        check["name"]
        == "performance:pattern:.github/workflows/ci.yml:Gate 12: Performance Gate"
        for check in report["failed_checks"]
    )


def test_elite_blueprint_audit_emits_architecture_security_risks_and_strategy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump({"jobs": {"phase65-masterpiece-gate": {}}}, sort_keys=False)
        + "\n# scripts/ci_secret_scan.py\n",
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False),
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(repo / "scripts/ci_secret_scan.py", "# secret scan\n")
    _write(repo / "core/auth/middleware.py", "class AuthMiddleware: pass\n")
    _write(
        repo / "core/sovereign/api.py",
        'ROUTES = ["/v1/plan", "/v1/stream", "/v1/settings/model-routing"]\n'
        'EVENTS = ["auth.boundary.crossed", "receipt.verified", "tick.completed"]\n',
    )
    _write(
        repo / "core/sovereign/terminal.py",
        'STATES = ["BLOCKED_CONSTITUTIONALLY", "SYSTEM_1_CACHE_HIT", "AWAITING_ESCALATION"]\n',
    )
    _write(repo / "core/proof_engine/receipt.py", "hash_chain_ref = 'root'\n")
    _write(
        repo / "core/reasoning/verified_graph.py",
        "class VerifiedReasoningGraph:\n    pass\n",
    )
    _write(
        repo / "docs/internal/UNIFIED_ACTIONABLE_FRAMEWORK.md",
        "# doc\nStrategic Workstreams\nPrioritized Roadmap\nSAPE Execution Method\n",
    )
    _write(
        repo / "frontend/src/components/terminal/terminal-manifest.ts",
        "export const PAT_AGENT_MANIFEST = [];\n",
    )
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _seed_phase65_config(repo)

    cfg = _minimal_cfg()
    cfg["checks"]["architecture_coherence"] = {
        "required_files": [
            "core/sovereign/api.py",
            "core/sovereign/terminal.py",
            "core/proof_engine/receipt.py",
            "core/reasoning/verified_graph.py",
            "docs/internal/UNIFIED_ACTIONABLE_FRAMEWORK.md",
            "frontend/src/components/terminal/terminal-manifest.ts",
        ],
        "required_patterns": {
            "core/sovereign/api.py": [
                "/v1/plan",
                "/v1/stream",
                "/v1/settings/model-routing",
            ],
            "core/sovereign/terminal.py": [
                "BLOCKED_CONSTITUTIONALLY",
                "SYSTEM_1_CACHE_HIT",
                "AWAITING_ESCALATION",
            ],
        },
    }
    cfg["checks"]["security_coherence"] = {
        "required_files": [
            "docs/security/hardening-checklist.md",
            "core/auth/middleware.py",
            "scripts/ci_secret_scan.py",
            ".github/workflows/ci.yml",
            "core/sovereign/api.py",
        ],
        "required_patterns": {
            ".github/workflows/ci.yml": ["scripts/ci_secret_scan.py"],
            "core/sovereign/api.py": [
                "auth.boundary.crossed",
                "receipt.verified",
                "tick.completed",
            ],
        },
    }

    report = audit_repo(repo, cfg)

    assert report["gate_passed"] is False
    assert report["sections"]["architecture"]["score"] == 1.0
    assert report["sections"]["security"]["score"] < 1.0
    assert "security" in report["interdisciplinary_lenses"]
    plane_ids = {plane["id"] for plane in report["control_planes"]}
    assert "architecture" in plane_ids
    assert "security" in plane_ids
    assert any(risk["dimension"] == "security" for risk in report["risk_register"])
    assert (
        report["implementation_strategy"]["current_phase"]
        == "stabilize_truth_and_trust"
    )
    assert "ihsan" in report["ethical_integrity_posture"]


def test_elite_blueprint_audit_checks_runtime_canon_lock_plane(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".github/workflows/ci.yml",
        yaml.safe_dump({"jobs": {"phase65-masterpiece-gate": {}}}, sort_keys=False)
        + "\n# scripts/ops/runtime_canon_lock_gate.py\n# config/runtime_canon_lock_gate.json\n",
    )
    _write(
        repo / ".github/workflows/phase65-masterpiece.yml",
        yaml.safe_dump({"jobs": {"phase65-gate": {}}}, sort_keys=False)
        + "\n# scripts/ops/runtime_canon_lock_gate.py\n# config/runtime_canon_lock_gate.json\n",
    )
    _write(repo / "scripts/ops/phase65_masterpiece_runner.py", "# runner\n")
    _write(repo / "scripts/ops/runtime_canon_lock_gate.py", "# canon lock gate\n")
    _write(repo / "config/runtime_canon_lock_gate.json", "{}\n")
    _write(repo / "tests/scripts/test_runtime_canon_lock_gate.py", "# tests\n")
    _write(repo / "tests/core/sovereign/test_main_cli.py", "# cli tests\n")
    _write(
        repo / "core/sovereign/api.py",
        "\n".join(
            [
                "if not runtime_has_canonical_authority:",
                "    pass",
                "runtime_receipt = await runtime_mission(description)",
            ]
        ),
    )
    _write(
        repo / "core/sovereign/__main__.py",
        "receipt = await runtime.mission(description, source=source, context={})\n",
    )
    _write(
        repo / "README.md",
        "\n".join(["[![CI Status]", "[![Roadmap]", "## Community"]),
    )
    _write(repo / "ROADMAP.md", "# roadmap\n")
    _write(repo / "COMMUNITY.md", "# community\n")
    _seed_phase65_config(repo)

    cfg = _minimal_cfg()
    cfg["checks"]["runtime_canon_lock"] = {
        "required_files": [
            "scripts/ops/runtime_canon_lock_gate.py",
            "config/runtime_canon_lock_gate.json",
            "tests/scripts/test_runtime_canon_lock_gate.py",
            "tests/core/sovereign/test_main_cli.py",
        ],
        "required_patterns": {
            ".github/workflows/ci.yml": [
                "scripts/ops/runtime_canon_lock_gate.py",
                "config/runtime_canon_lock_gate.json",
            ],
            ".github/workflows/phase65-masterpiece.yml": [
                "scripts/ops/runtime_canon_lock_gate.py",
                "config/runtime_canon_lock_gate.json",
            ],
            "core/sovereign/api.py": [
                "if not runtime_has_canonical_authority:",
                "runtime_receipt = await runtime_mission(",
            ],
            "core/sovereign/__main__.py": [
                "receipt = await runtime.mission(description, source=source, context={})"
            ],
        },
    }
    cfg["scoring"]["weights"]["files"] = 0.25
    cfg["scoring"]["weights"]["runtime_canon_lock"] = 0.05

    report = audit_repo(repo, cfg)

    assert report["gate_passed"] is True
    assert report["sections"]["runtime_canon_lock"]["score"] == 1.0
    plane_ids = {plane["id"] for plane in report["control_planes"]}
    assert "runtime_canon_lock" in plane_ids
    assert report["interdisciplinary_lenses"]["architecture"] == 1.0
