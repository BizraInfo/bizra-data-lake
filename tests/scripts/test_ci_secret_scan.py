from __future__ import annotations

from pathlib import Path

from scripts.ci_secret_scan import scan_file, tracked_files


def test_tracked_files_include_json_and_root_configs(tmp_path: Path) -> None:
    relpaths = [
        ".claude/settings.local.json",
        "scripts/ci_secret_scan.py",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "docs/readme.md",
        "node_modules/library/config.json",
    ]

    files = tracked_files(root=tmp_path, relpaths=relpaths)
    rels = {str(path.relative_to(tmp_path)).replace("\\", "/") for path in files}

    assert ".claude/settings.local.json" in rels
    # scripts/ci_secret_scan.py is intentionally self-excluded by the scanner
    assert "scripts/ci_secret_scan.py" not in rels
    assert "pyproject.toml" in rels
    assert ".github/workflows/ci.yml" in rels
    assert "docs/readme.md" not in rels
    assert "node_modules/library/config.json" not in rels


def test_scan_file_flags_real_secret_in_json(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"allow":["Bash(export LM_STUDIO_API_KEY=\\"sk-lm-a1b2c3d4e5f6g7h8\\")"]}',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert any("high-entropy token pattern" in item for item in findings)


def test_scan_file_ignores_placeholder_values(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"OPENAI_API_KEY":"${OPENAI_API_KEY}","api_token":"example-token"}',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert findings == []


def test_scan_file_ignores_documented_placeholder_secret(tmp_path: Path) -> None:
    path = tmp_path / "jwt_auth.py"
    path.write_text(
        'auth = JWTAuth(secret="your-256-bit-secret")',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert findings == []


def test_scan_file_catches_bearer_token_pattern(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text(
        'Authorization="Bearer ghs_1234567890abcdefghijklmnop"',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert any("bearer token pattern" in item for item in findings)


def test_scan_file_ignores_bizra_hex_env_name_references(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yml"
    path.write_text(
        "\n".join(
            [
                "env:",
                "  BRANCH_SIGNER_KEY: ${{ secrets.BIZRA_RECEIPT_PRIVATE_KEY_HEX }}",
                "run: echo BIZRA_RECEIPT_PRIVATE_KEY_HEX is required",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert findings == []


def test_scan_file_flags_bizra_private_hex_secret_literal(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yml"
    path.write_text(
        'BIZRA_RECEIPT_PRIVATE_KEY_HEX="' + "a" * 64 + '"',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert any("BIZRA hex secret literal" in item for item in findings)


def test_scan_file_allows_bizra_public_hex_literal(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yml"
    path.write_text(
        'BIZRA_RECEIPT_PUBLIC_KEY_HEX="' + "b" * 64 + '"',
        encoding="utf-8",
    )

    findings = scan_file(path, root=tmp_path)

    assert findings == []
