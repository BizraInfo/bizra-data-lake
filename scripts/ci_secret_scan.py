#!/usr/bin/env python3
"""
Lightweight secret scan for tracked config and code without external dependencies.

Fail criteria:
- Hardcoded token/secret/password-like assignments in tracked config/code files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

YAML_EXT = {".yml", ".yaml"}
SCRIPT_EXT = {".py", ".sh", ".bash", ".zsh", ".ahk", ".ps1"}
CONFIG_EXT = {".json", ".toml", ".ini"}
PATH_SKIP_PARTS = {
    ".git/",
    "tests/",
    "docs/",
    "data/",
    "node_modules/",
    ".venv/",
    ".venv-linux/",
    "venv/",
    ".mypy_cache/",
    ".pytest_cache/",
    "__pycache__/",
    ".swarm/",
}

ASSIGNMENT_RE = re.compile(r"""(?ix)
    \b(
        api[_-]?token|
        api[_-]?key|
        access[_-]?token|
        auth[_-]?token|
        bearer|
        client[_-]?secret|
        jwt[_-]?secret|
        secret|
        password
    )\b
    [^:=\n]{0,32}
    \s*[:=]\s*
    ["']([^"'\n]{8,})["']
    """)

HIGH_ENTROPY_RE = re.compile(r"""(?ix)
    (
        sk-(?=[a-z0-9:_-]{16,})(?=[a-z0-9:_-]*\d)[a-z0-9:_-]+|
        gh[pousr]_[a-z0-9]{20,}|
        xox[baprs]-[a-z0-9-]{20,}
    )
    """)

BEARER_RE = re.compile(r"""(?ix)
    \bbearer\b
    [\s:='""]+
    ["']?([a-z0-9._:-]{16,})["']?
    """)


def _is_env_file(rel_posix: str) -> bool:
    name = Path(rel_posix).name.lower()
    return name == ".env" or name.startswith(".env.")


def _should_scan_relpath(rel_posix: str) -> bool:
    if any(skip in rel_posix for skip in PATH_SKIP_PARTS):
        return False
    if rel_posix == "scripts/ci_secret_scan.py":
        return False

    suffix = Path(rel_posix).suffix.lower()
    return (
        suffix in YAML_EXT
        or suffix in SCRIPT_EXT
        or suffix in CONFIG_EXT
        or _is_env_file(rel_posix)
    )


def _tracked_relpaths(root: Path = ROOT) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=root,
            text=True,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def tracked_files(
    root: Path = ROOT,
    relpaths: list[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    for rel in relpaths if relpaths is not None else _tracked_relpaths(root):
        rel_posix = rel.replace("\\", "/")
        if not _should_scan_relpath(rel_posix):
            continue
        files.append(root / rel_posix)
    return files


def is_placeholder(value: str) -> bool:
    v = value.strip()
    normalized = re.sub(r"[^a-z0-9]+", "", v.lower())
    return (
        v.startswith("${")
        or v.startswith("$(")
        or v.lower().startswith("your-")
        or v.lower().startswith("env:")
        or v.startswith("<")
        or v.endswith(">")
        or v.lower() in {"lm-studio", "local", "localhost", "none", "null"}
        or normalized in {"password", "apikey", "secret", "token"}
        or "changeme" in v.lower()
        or "example" in v.lower()
        or "dummy" in v.lower()
        or "sample" in v.lower()
        or "test" in v.lower()
        or "placeholder" in v.lower()
        or "redacted" in v.lower()
        or "synthetic" in v.lower()
    )


def scan_file(path: Path, root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    for i, line in enumerate(text.splitlines(), start=1):
        assignment = ASSIGNMENT_RE.search(line)
        if assignment:
            value = assignment.group(2)
            if not is_placeholder(value):
                findings.append(
                    f"{path.relative_to(root)}:{i}: hardcoded {assignment.group(1)}"
                )
                continue

        bearer = BEARER_RE.search(line)
        if bearer and not is_placeholder(bearer.group(1)):
            findings.append(f"{path.relative_to(root)}:{i}: bearer token pattern")
            continue

        entropy = HIGH_ENTROPY_RE.search(line)
        if entropy and not is_placeholder(entropy.group(1)):
            findings.append(f"{path.relative_to(root)}:{i}: high-entropy token pattern")

    return findings


def main() -> int:
    files = tracked_files()
    findings: list[str] = []
    for path in files:
        findings.extend(scan_file(path, root=ROOT))

    if findings:
        print("Secret scan failed. Findings:")
        for item in findings:
            print(f"  - {item}")
        return 1

    print(f"Secret scan passed ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
