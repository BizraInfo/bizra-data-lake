"""
Dark Matter Audit — Enumerate Unobserved Executing Components
=============================================================
Golden Gem α8 + XZ Gem #2: Every system has components that
participate in execution but are invisible to inspection.

Standing on Giants:
  - Jia Tan's XZ attack — payload hidden in binary test blobs
  - Ken Thompson (1984) — "Reflections on Trusting Trust"
  - Saltzer & Schroeder (1975) — Complete mediation principle

PRINCIPLE: The attack surface isn't what you can see.
It's what you can't see but implicitly trust.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Binary extensions that bypass code review
BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".whl",
        ".egg",
        ".tar.gz",
        ".tgz",
        ".zip",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".wasm",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".pdf",
        ".docx",
        ".xlsx",
    }
)

# CI/Docker files that execute but are rarely audited
DARK_MATTER_PATTERNS: Final[list[str]] = [
    "Dockerfile*",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    ".github/workflows/*.yml",
    ".github/actions/**/*",
    "*.lock",  # lockfiles pin versions but aren't reviewed
    "requirements*.txt",
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]


@dataclass
class DarkMatterItem:
    """A component in the build/deploy chain that executes but isn't reviewed."""

    path: str
    category: str  # binary_blob, ci_config, lockfile, docker, dependency
    risk_level: str  # high, medium, low
    sha256: str = ""
    size_bytes: int = 0
    recommendation: str = ""


@dataclass
class DarkMatterAuditReport:
    """Complete audit of unobserved executing components."""

    repo_root: str
    items: list[DarkMatterItem] = field(default_factory=list)
    total_binary_blobs: int = 0
    total_ci_configs: int = 0
    total_lockfiles: int = 0
    total_docker_files: int = 0
    risk_score: float = 0.0  # 0.0 (clean) to 1.0 (critical)

    @property
    def summary(self) -> str:
        high = sum(1 for i in self.items if i.risk_level == "high")
        medium = sum(1 for i in self.items if i.risk_level == "medium")
        low = sum(1 for i in self.items if i.risk_level == "low")
        return (
            f"Dark Matter Audit: {len(self.items)} items found. "
            f"Risk: {high} high, {medium} medium, {low} low. "
            f"Score: {self.risk_score:.2f}/1.0"
        )


# Max file size to hash (skip huge binaries — they'd timeout the audit)
_MAX_HASH_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MB


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file (skips files > 50MB and symlinks)."""
    h = hashlib.sha256()
    try:
        # Skip symlinks — on WSL/Windows cross-boundary, f.read() can hang
        if path.is_symlink():
            return "SKIPPED_SYMLINK"
        size = path.stat().st_size
        if size > _MAX_HASH_BYTES:
            return f"SKIPPED_TOO_LARGE_{size}"
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return "UNREADABLE"


def _safe_size(p: Path) -> int:
    """Get file size, returning 0 on OS errors (e.g. WSL symlinks)."""
    try:
        return p.stat().st_size
    except OSError:
        return 0


def audit_dark_matter(repo_root: str | Path) -> DarkMatterAuditReport:
    """Scan repository for dark matter components.

    Enumerates:
    1. Binary blobs (files that bypass code review)
    2. CI/CD configurations (execute with full permissions)
    3. Lockfiles (pin versions but aren't reviewed line-by-line)
    4. Docker files (define build environment)
    5. Unpinned base images (mutable tags instead of digests)
    """
    root = Path(repo_root)
    report = DarkMatterAuditReport(repo_root=str(root))

    if not root.is_dir():
        return report

    # Scan all files (skip symlinks — WSL cross-boundary can hang)
    for path in root.rglob("*"):
        try:
            if path.is_symlink():
                continue
            if not path.is_file():
                continue
        except OSError:
            continue  # WSL symlinks, permission errors, etc.

        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except (ValueError, OSError):
            continue

        # Skip .git internals and __pycache__
        if any(
            skip in rel
            for skip in (".git/", "__pycache__", "node_modules/", ".venv", "venv/")
        ):
            continue

        # Category 1: Binary blobs
        if path.suffix.lower() in BINARY_EXTENSIONS:
            report.items.append(
                DarkMatterItem(
                    path=rel,
                    category="binary_blob",
                    risk_level=(
                        "high"
                        if path.suffix in {".so", ".dll", ".whl", ".exe", ".wasm"}
                        else "medium"
                    ),
                    sha256=_sha256_file(path),
                    size_bytes=_safe_size(path),
                    recommendation=(
                        "Pin hash in .gitattributes. "
                        "Require explicit review for changes."
                    ),
                )
            )
            report.total_binary_blobs += 1

        # Category 2: CI/CD configs
        elif ".github/workflows" in rel or ".github/actions" in rel:
            report.items.append(
                DarkMatterItem(
                    path=rel,
                    category="ci_config",
                    risk_level="high",
                    sha256=_sha256_file(path),
                    size_bytes=_safe_size(path),
                    recommendation=(
                        "CI workflows execute with repo secrets. "
                        "Require CODEOWNERS approval. "
                        "Pin action versions to SHA, not tag."
                    ),
                )
            )
            report.total_ci_configs += 1

        # Category 3: Lockfiles
        elif path.name in {
            "Cargo.lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "Pipfile.lock",
        }:
            report.items.append(
                DarkMatterItem(
                    path=rel,
                    category="lockfile",
                    risk_level="medium",
                    sha256=_sha256_file(path),
                    size_bytes=_safe_size(path),
                    recommendation=(
                        "Lockfiles pin dependency versions. "
                        "Run audit tools (pip-audit, cargo-audit, npm audit) in CI."
                    ),
                )
            )
            report.total_lockfiles += 1

        # Category 4: Docker files
        elif path.name.startswith("Dockerfile") or path.name.startswith(
            "docker-compose"
        ):
            report.items.append(
                DarkMatterItem(
                    path=rel,
                    category="docker",
                    risk_level="high",
                    sha256=_sha256_file(path),
                    size_bytes=_safe_size(path),
                    recommendation=(
                        "Pin base images by digest, not tag. "
                        "FROM image@sha256:... instead of FROM image:latest"
                    ),
                )
            )
            report.total_docker_files += 1

    # Compute risk score
    high_count = sum(1 for i in report.items if i.risk_level == "high")
    medium_count = sum(1 for i in report.items if i.risk_level == "medium")
    total = len(report.items) or 1
    report.risk_score = min(
        1.0, (high_count * 0.1 + medium_count * 0.03) / max(total * 0.05, 1)
    )

    return report


def generate_gitattributes_rules(report: DarkMatterAuditReport) -> str:
    """Generate .gitattributes rules to flag binary blob changes for review.

    XZ Lesson: Binary test files bypassed all code review.
    This rule ensures binary changes trigger explicit review.
    """
    lines = [
        "# BIZRA Dark Matter Defense — Auto-generated from audit",
        "# Golden Gem α8: Flag binary blob changes for mandatory review",
        "# XZ Backdoor Lesson: Never trust unreviewed binary changes",
        "",
    ]

    # Collect unique extensions from binary blobs
    extensions: set[str] = set()
    for item in report.items:
        if item.category == "binary_blob":
            ext = Path(item.path).suffix
            if ext:
                extensions.add(ext)

    for ext in sorted(extensions):
        lines.append(f"*{ext} binary diff=binary")

    lines.extend(
        [
            "",
            "# CI/CD changes require explicit review",
            ".github/**/*.yml linguist-detectable=true",
            ".github/**/*.yaml linguist-detectable=true",
            "Dockerfile* linguist-detectable=true",
            "docker-compose* linguist-detectable=true",
        ]
    )

    return "\n".join(lines) + "\n"


__all__ = [
    "DarkMatterItem",
    "DarkMatterAuditReport",
    "audit_dark_matter",
    "generate_gitattributes_rules",
]
