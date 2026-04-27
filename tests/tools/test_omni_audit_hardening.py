from pathlib import Path

from tools.audit.omni_audit import secret_pattern_scanner, urp_canonicality
from tools.audit.omni_audit.run_audit import _derive_findings, _load_yaml_config


def test_yaml_loader_strips_inline_comments(tmp_path: Path) -> None:
    config = tmp_path / "audit_config.yaml"
    config.write_text(
        """
output:
  evidence_index_limit: 2000        # cap for deterministic output
  default_strict: false             # keep as advisory until strict mode lands
scope:
  secret_scan_roots:
    - "core"
    - "runtime"  # include historical runtime configs
""".strip(),
        encoding="utf-8",
    )

    loaded = _load_yaml_config(config)

    assert loaded["output"]["evidence_index_limit"] == 2000
    assert loaded["output"]["default_strict"] is False
    assert loaded["scope"]["secret_scan_roots"] == ["core", "runtime"]


def test_secret_scanner_dedupes_roots_and_suppresses_known_noise(
    tmp_path: Path,
) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / ".claude" / "logs").mkdir(parents=True)
    (tmp_path / "tools" / "audit" / "omni_audit").mkdir(parents=True)

    (tmp_path / "core" / "settings.py").write_text(
        "\n".join(
            [
                'API_KEY_HASH_PREFIX = "sha256:"',
                'DATABASE_URL = "postgresql://app:real_password@localhost:5432/app"',
                'SAFE_URL = "postgresql://app:${POSTGRES_PASSWORD}@localhost:5432/app"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".claude" / "logs" / "audit.jsonl").write_text(
        "prompt_preview=-----BEGIN PRIVATE KEY-----",
        encoding="utf-8",
    )
    (tmp_path / "tools" / "audit" / "omni_audit" / "secret_pattern_scanner.py").write_text(
        'PATTERN = "-----BEGIN PGP PRIVATE KEY BLOCK-----"',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-" + ("A" * 30),
        encoding="utf-8",
    )

    findings = secret_pattern_scanner.scan(
        repo_root=tmp_path,
        roots=["core", "."],
        top_level_globs=["*.env"],
        max_bytes=1024,
        limit=20,
    )

    by_path_line = {(f["path"], f["line"], f["pattern_class"]) for f in findings}

    assert len(findings) == len(by_path_line)
    assert ("core/settings.py", 2, "POSTGRES_URL_WITH_PASSWORD") in by_path_line
    assert (".env", 1, "OPENAI_API_KEY") in by_path_line
    assert all("API_KEY_HASH_PREFIX" not in f["redacted_preview"] for f in findings)
    assert all("${POSTGRES_PASSWORD}" not in f["redacted_preview"] for f in findings)
    assert all(not f["path"].startswith(".claude/logs/") for f in findings)
    assert all(f["path"] != "tools/audit/omni_audit/secret_pattern_scanner.py" for f in findings)


def test_urp_canonicality_detects_alternate_expansions(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "architecture.md").write_text(
        "\n".join(
            [
                "URP (Universal Resource Pool) is the shared resource layer.",
                "Legacy wording: Universal Receipt Plane (URP).",
                "Another old deck says URP: Universal Reasoning Protocol.",
            ]
        ),
        encoding="utf-8",
    )

    observations = urp_canonicality.scan(
        repo_root=tmp_path,
        roots=["docs"],
        exclude_dirs=[],
        max_bytes=4096,
        limit=20,
    )

    by_expansion = {item["expansion"]: item["classification"] for item in observations}

    assert by_expansion["Universal Resource Pool"] == "CANONICAL"
    assert by_expansion["Universal Receipt Plane"] == "ALTERNATE"
    assert by_expansion["Universal Reasoning Protocol"] == "ALTERNATE"


def test_urp_canonicality_derives_signal_finding() -> None:
    findings = _derive_findings(
        evidence=[],
        claims=[],
        secrets=[],
        code_risks=[],
        deps={"gaps": []},
        captures=[],
        urp_observations=[
            {
                "path": "docs/example.md",
                "line": 1,
                "expansion": "Universal Receipt Plane",
                "classification": "ALTERNATE",
            }
        ],
    )

    urp_findings = [f for f in findings if f.subsystem == "urp_canonicality"]

    assert len(urp_findings) == 1
    assert urp_findings[0].domain == "DOCUMENTATION"
    assert urp_findings[0].signal_score >= 0.65
    assert "docs/example.md" in urp_findings[0].evidence_paths


def test_urp_canonicality_allows_alias_table_in_canonical_doc(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "architecture"
    docs.mkdir(parents=True)
    (docs / "URP_CANONICAL_DEFINITION.md").write_text(
        "URP = Universal Resource Pool.\n"
        "| Universal Receipt Plane | Historical alias |\n",
        encoding="utf-8",
    )

    observations = urp_canonicality.scan(
        repo_root=tmp_path,
        roots=["docs"],
        exclude_dirs=[],
        max_bytes=4096,
        limit=20,
    )

    by_expansion = {item["expansion"]: item["classification"] for item in observations}

    assert by_expansion["Universal Resource Pool"] == "CANONICAL"
    assert by_expansion["Universal Receipt Plane"] == "DOCUMENTED_ALIAS"
