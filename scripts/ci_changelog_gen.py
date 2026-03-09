#!/usr/bin/env python3
"""
BIZRA Automated Changelog Generator
=====================================

Parses conventional commits and generates structured CHANGELOG sections.
Integrates with the release pipeline to produce auditable release notes.

Conventional Commit Format:
    type(scope): description

    Where type ∈ {feat, fix, perf, refactor, docs, test, ci, chore, security, breaking}

Standing on Giants:
- Angular Team (Conventional Commits, 2016)
- Keep a Changelog (olivierlacan, 2014)
- Semantic Versioning (Tom Preston-Werner, 2011)

Usage:
    # Generate changelog from last tag to HEAD
    python scripts/ci_changelog_gen.py --from-tag v2.0.0

    # Generate for specific range
    python scripts/ci_changelog_gen.py --from-sha abc123 --to-sha def456

    # Append to CHANGELOG.md
    python scripts/ci_changelog_gen.py --from-tag v2.0.0 --append CHANGELOG.md

Exit Codes:
    0 - Success
    1 - No commits found
    2 - Git error
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Conventional Commit Parser
# ─────────────────────────────────────────────────────────────

COMMIT_PATTERN = re.compile(
    r'^(?P<type>feat|fix|perf|refactor|docs|test|ci|chore|security|breaking)'
    r'(?:\((?P<scope>[^)]+)\))?'
    r'(?P<breaking_mark>!)?:\s*'
    r'(?P<description>.+)$'
)

# Display order and section titles
TYPE_SECTIONS: Dict[str, str] = {
    "breaking": "⚠️ Breaking Changes",
    "security": "🔒 Security",
    "feat": "✨ Features",
    "fix": "🐛 Bug Fixes",
    "perf": "⚡ Performance",
    "refactor": "♻️ Refactoring",
    "docs": "📝 Documentation",
    "test": "✅ Tests",
    "ci": "🏗️ CI/CD",
    "chore": "🔧 Chores",
}


@dataclass
class ParsedCommit:
    """A parsed conventional commit."""
    sha: str
    type: str
    scope: Optional[str]
    description: str
    body: str = ""
    is_breaking: bool = False
    raw_message: str = ""
    author: str = ""
    date: str = ""


@dataclass
class ChangelogSection:
    """A grouped section of the changelog."""
    title: str
    type_key: str
    commits: List[ParsedCommit] = field(default_factory=list)


@dataclass
class ChangelogRelease:
    """A complete changelog for a release."""
    version: str
    date: str
    sections: List[ChangelogSection] = field(default_factory=list)
    total_commits: int = 0
    contributors: List[str] = field(default_factory=list)
    evidence_hash: str = ""


# ─────────────────────────────────────────────────────────────
# Git Interface
# ─────────────────────────────────────────────────────────────

def get_commits(
    from_ref: str,
    to_ref: str = "HEAD",
    workspace: Path = Path("."),
) -> List[Tuple[str, str, str, str]]:
    """Get git commits in range as (sha, author, date, message) tuples."""
    separator = "---BIZRA-COMMIT-SEP---"
    fmt = f"%H{separator}%an{separator}%ai{separator}%B{separator}"

    result = subprocess.run(
        [
            "git", "log",
            f"{from_ref}..{to_ref}",
            f"--format={fmt}",
            "--no-merges",
        ],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr}")

    commits = []
    raw = result.stdout.strip()
    if not raw:
        return commits

    entries = raw.split(f"{separator}\n")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(separator, 3)
        if len(parts) >= 4:
            sha, author, date, message = parts[0], parts[1], parts[2], parts[3]
            commits.append((sha.strip(), author.strip(), date.strip(), message.strip()))
        elif len(parts) == 3:
            sha, author, date = parts
            commits.append((sha.strip(), author.strip(), date.strip(), ""))

    return commits


def get_latest_tag(workspace: Path = Path(".")) -> Optional[str]:
    """Get the most recent git tag."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=10,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


# ─────────────────────────────────────────────────────────────
# Commit Parser
# ─────────────────────────────────────────────────────────────

def parse_commit(sha: str, author: str, date: str, message: str) -> ParsedCommit:
    """Parse a commit message into a structured ParsedCommit."""
    lines = message.strip().split("\n")
    first_line = lines[0] if lines else ""
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    match = COMMIT_PATTERN.match(first_line)
    if not match:
        return ParsedCommit(
            sha=sha[:8],
            type="chore",  # Default for non-conventional commits
            scope=None,
            description=first_line,
            body=body,
            is_breaking=False,
            raw_message=message,
            author=author,
            date=date,
        )

    commit_type = match.group("type")
    is_breaking = match.group("breaking_mark") == "!" or commit_type == "breaking"

    # Check for BREAKING CHANGE footer
    if "BREAKING CHANGE:" in body or "BREAKING-CHANGE:" in body:
        is_breaking = True

    return ParsedCommit(
        sha=sha[:8],
        type=commit_type,
        scope=match.group("scope"),
        description=match.group("description"),
        body=body,
        is_breaking=is_breaking,
        raw_message=message,
        author=author,
        date=date,
    )


# ─────────────────────────────────────────────────────────────
# Changelog Generator
# ─────────────────────────────────────────────────────────────

def generate_changelog(
    commits: List[ParsedCommit],
    version: str = "Unreleased",
) -> ChangelogRelease:
    """Generate a structured changelog from parsed commits."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Group commits by type
    grouped: Dict[str, List[ParsedCommit]] = {}
    for commit in commits:
        key = "breaking" if commit.is_breaking else commit.type
        grouped.setdefault(key, []).append(commit)

    # Build sections in display order
    sections = []
    for type_key, title in TYPE_SECTIONS.items():
        if type_key in grouped:
            sections.append(ChangelogSection(
                title=title,
                type_key=type_key,
                commits=grouped[type_key],
            ))

    # Unique contributors
    contributors = sorted(set(c.author for c in commits if c.author))

    # Evidence hash
    content = json.dumps(
        [{"sha": c.sha, "type": c.type, "desc": c.description} for c in commits],
        sort_keys=True,
    )
    evidence_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    return ChangelogRelease(
        version=version,
        date=now,
        sections=sections,
        total_commits=len(commits),
        contributors=contributors,
        evidence_hash=evidence_hash,
    )


def render_markdown(release: ChangelogRelease) -> str:
    """Render a ChangelogRelease as Markdown."""
    lines = []
    lines.append(f"## [{release.version}] - {release.date}")
    lines.append("")
    lines.append(
        f"> {release.total_commits} commits | "
        f"{len(release.contributors)} contributors | "
        f"Evidence: `{release.evidence_hash}`"
    )
    lines.append("")

    for section in release.sections:
        lines.append(f"### {section.title}")
        lines.append("")
        for commit in section.commits:
            scope_str = f"**{commit.scope}**: " if commit.scope else ""
            lines.append(f"- {scope_str}{commit.description} ({commit.sha})")
        lines.append("")

    if release.contributors:
        lines.append("### 👥 Contributors")
        lines.append("")
        lines.append(", ".join(release.contributors))
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIZRA Automated Changelog Generator",
    )
    parser.add_argument("--from-tag", help="Generate from this tag")
    parser.add_argument("--from-sha", help="Generate from this commit SHA")
    parser.add_argument("--to-sha", default="HEAD", help="Generate to this ref (default: HEAD)")
    parser.add_argument("--version", default="Unreleased", help="Version label")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Workspace root")
    parser.add_argument("--append", type=Path, default=None, help="Append to this file")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("04_GOLD/changelog_evidence.jsonl"),
        help="Evidence log path",
    )

    args = parser.parse_args()

    # Determine from-ref
    from_ref = args.from_sha or args.from_tag
    if not from_ref:
        from_ref = get_latest_tag(args.workspace)
        if not from_ref:
            print("[ERROR] No --from-tag/--from-sha and no tags found", file=sys.stderr)
            return 2

    # Get and parse commits
    try:
        raw_commits = get_commits(from_ref, args.to_sha, args.workspace)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not raw_commits:
        print("[INFO] No commits found in range")
        return 1

    parsed = [parse_commit(sha, author, date, msg) for sha, author, date, msg in raw_commits]
    release = generate_changelog(parsed, version=args.version)

    # Output
    if args.json:
        print(json.dumps(asdict(release), indent=2, default=str))
    else:
        md = render_markdown(release)
        print(md)

        if args.append:
            existing = args.append.read_text(encoding="utf-8") if args.append.exists() else ""
            # Insert after the first heading
            if existing:
                insert_marker = existing.find("\n## ")
                if insert_marker >= 0:
                    new_content = existing[:insert_marker] + "\n" + md + existing[insert_marker:]
                else:
                    new_content = existing + "\n" + md
            else:
                new_content = "# Changelog\n\n" + md

            args.append.write_text(new_content, encoding="utf-8")
            print(f"\n[APPENDED] {args.append}")

    # Evidence
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evidence, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": release.version,
            "total_commits": release.total_commits,
            "from_ref": from_ref,
            "to_ref": args.to_sha,
            "evidence_hash": release.evidence_hash,
        }, default=str) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
