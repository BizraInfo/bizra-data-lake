"""Read-only secret-pattern scanner.

Never prints the matched value. Emits (path, line, pattern_class, redacted_preview).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List


# Each pattern has a human-readable class name and a regex.
PATTERNS: List[tuple[str, str]] = [
    ("PRIVATE_KEY_BLOCK", r"-----BEGIN (RSA |DSA |EC |OPENSSH |)PRIVATE KEY-----"),
    ("PGP_PRIVATE_KEY_BLOCK", r"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
    ("SSH_PRIVATE_KEY", r"ssh-(rsa|ed25519|dss)\s+[A-Za-z0-9+/=]{200,}"),
    ("AWS_ACCESS_KEY", r"AKIA[0-9A-Z]{16}"),
    ("AWS_SECRET_KEY", r"aws(.{0,20})?(secret|access)(_key|key)?[\"':=\s]+[A-Za-z0-9/+=]{40}"),
    ("SLACK_TOKEN", r"xox[baprs]-[A-Za-z0-9\-]{10,48}"),
    ("GITHUB_TOKEN", r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}"),
    ("OPENAI_API_KEY", r"sk-[A-Za-z0-9]{20,48}"),
    ("ANTHROPIC_API_KEY", r"sk-ant-api\d{2}-[A-Za-z0-9\-_]{40,}"),
    ("GENERIC_API_KEY", r"api[_-]?key[\"':=\s]+[A-Za-z0-9_\-]{24,}"),
    ("JWT_TOKEN", r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    ("REDIS_URL_WITH_PASSWORD", r"redis(s)?://[^:\s]+:[^@\s]+@[A-Za-z0-9\.\-]+:\d+"),
    ("POSTGRES_URL_WITH_PASSWORD", r"postgres(ql)?://[^:\s]+:[^@\s]+@[A-Za-z0-9\.\-]+"),
    ("MONGODB_URL_WITH_PASSWORD", r"mongodb(\+srv)?://[^:\s]+:[^@\s]+@[A-Za-z0-9\.\-]+"),
    ("CERT_BLOCK", r"-----BEGIN CERTIFICATE-----"),
    ("DOTENV_LIKE", r"(?m)^\s*(export\s+)?(SECRET|PASSWORD|TOKEN|API_KEY|PRIVATE_KEY)\w*\s*=\s*[^\s#\n]{8,}"),
]

# Paths to always skip even if inside scan roots.
ALWAYS_SKIP_PARTS = {".git", "target", "node_modules", "__pycache__", ".venv",
                     ".venv-linux", "venv", "dist", "build",
                     ".claude/logs", ".tmp_prod_artifacts_v2",
                     "docs/audits",
                     "docs/brand/public_launch_media_kit_v0_1/extracted",
                     "tools/cognitive_foundry/claude_lane/output",
                     "tools/cognitive_foundry/claude_lane/canon_packs",
                     "tools/audit/omni_audit/secret_pattern_scanner.py",
                     "frontend/node_modules"}


def _should_skip(rel: Path) -> bool:
    s = rel.as_posix()
    for skip in ALWAYS_SKIP_PARTS:
        if skip in s:
            return True
    return False


def _iter_text_files(repo_root: Path, roots: List[str], max_bytes: int) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        base = repo_root if root == "." else (repo_root / root)
        if not base.exists():
            continue
        if base.is_file():
            try:
                rel = base.relative_to(repo_root)
            except ValueError:
                continue
            if rel not in seen:
                seen.add(rel)
                yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ALWAYS_SKIP_PARTS]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    if p.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                # skip obvious binaries by suffix
                suf = p.suffix.lower()
                if suf in {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".tgz", ".gz",
                           ".webp", ".ico", ".woff", ".woff2", ".ttf", ".otf",
                           ".svg", ".mp4", ".mov", ".wav", ".mp3", ".parquet"}:
                    continue
                try:
                    rel = p.relative_to(repo_root)
                except ValueError:
                    continue
                if rel in seen:
                    continue
                seen.add(rel)
                yield p


def _iter_top_level_globs(repo_root: Path, globs: List[str],
                          max_bytes: int) -> Iterable[Path]:
    seen: set[Path] = set()
    for name in os.listdir(repo_root):
        for pattern in globs:
            if not fnmatch.fnmatch(name, pattern):
                continue
            p = repo_root / name
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(repo_root)
                if rel in seen or p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            seen.add(rel)
            yield p


def _redact(line: str, match_start: int, match_end: int) -> str:
    left = line[:match_start]
    right = line[match_end:]
    red = "[REDACTED:{}]".format(match_end - match_start)
    return (left + red + right)[:240]


def _looks_like_safe_substitution(matched: str) -> bool:
    return "${" in matched and "}" in matched


def _looks_like_placeholder(matched: str) -> bool:
    low = matched.lower()
    return (
        ("<" in matched and ">" in matched)
        or "://postgres:test@" in low
        or "://user:pass@" in low
        or "://user:password@" in low
    )


def _looks_like_non_secret_assignment(line: str, pattern_class: str) -> bool:
    if pattern_class != "DOTENV_LIKE" or "=" not in line:
        return False

    rhs = line.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'")
    if not rhs:
        return True

    safe_literals = {
        "sha256:",
        "api_key",
        "apiKey",
        "password",
        "bizra-token-v1:",
    }
    if rhs in safe_literals:
        return True
    if rhs.startswith(("/", "./", "../")):
        return True
    if rhs.startswith(("os.environ.get(", "os.getenv(")):
        return True
    if rhs.startswith("$") or _looks_like_safe_substitution(rhs):
        return True
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", rhs):
        return True
    return False


def _should_report(pattern_class: str, line: str, matched: str) -> bool:
    if _looks_like_safe_substitution(matched):
        return False
    if _looks_like_placeholder(matched):
        return False
    if _looks_like_non_secret_assignment(line, pattern_class):
        return False
    return True


def scan(repo_root: Path, roots: List[str], top_level_globs: List[str],
         max_bytes: int, limit: int) -> List[dict]:
    findings: List[dict] = []
    compiled = [(name, re.compile(pat)) for name, pat in PATTERNS]

    # Walk declared roots.
    seen: set[Path] = set()
    candidates = list(_iter_text_files(repo_root, roots, max_bytes))
    candidates.extend(_iter_top_level_globs(repo_root, top_level_globs, max_bytes))
    for f in candidates:
        try:
            rel = f.relative_to(repo_root)
        except ValueError:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if _should_skip(rel):
            continue
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    for name, pat in compiled:
                        m = pat.search(line)
                        if m:
                            matched = line[m.start():m.end()]
                            if not _should_report(name, line, matched):
                                continue
                            findings.append({
                                "finding_id": f"S{len(findings) + 1:04d}",
                                "pattern_class": name,
                                "path": rel.as_posix(),
                                "line": lineno,
                                "redacted_preview": _redact(line.rstrip("\n"),
                                                             m.start(), m.end()),
                            })
                            if len(findings) >= limit:
                                return findings
        except OSError:
            continue
    return findings


def write_outputs(findings: List[dict], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "secret_findings.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    return {"secret_findings_json": str(path)}
