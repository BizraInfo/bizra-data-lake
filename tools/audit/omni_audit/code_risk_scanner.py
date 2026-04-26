"""Scan Python + Rust source for risky patterns. Read-only."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List


PYTHON_RULES: List[tuple[str, str, str]] = [
    ("PY_BROAD_EXCEPT_PASS", r"^\s*except\s*:\s*\n\s*pass\b", "LOW"),
    ("PY_BARE_EXCEPT", r"^\s*except\s*:\s*$", "LOW"),
    ("PY_BROAD_EXCEPT", r"^\s*except\s+Exception\s*(as\s+\w+)?\s*:\s*$", "LOW"),
    ("PY_EVAL_EXEC", r"\b(eval|exec)\s*\(", "MEDIUM"),
    ("PY_SHELL_TRUE", r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", "HIGH"),
    ("PY_OS_SYSTEM", r"\bos\.system\s*\(", "MEDIUM"),
    ("PY_PICKLE_LOAD", r"pickle\.(load|loads)\s*\(", "MEDIUM"),
    ("PY_TODO", r"#\s*(TODO|FIXME|HACK|XXX)\b", "LOW"),
    ("PY_URL_FETCH", r"(requests|httpx|urllib\.request)\.(get|post|put|delete|request)\s*\(", "LOW"),
    ("PY_YAML_LOAD", r"yaml\.load\s*\(", "MEDIUM"),
]

RUST_RULES: List[tuple[str, str, str]] = [
    ("RS_UNWRAP", r"\.unwrap\s*\(\)", "LOW"),
    ("RS_EXPECT", r"\.expect\s*\(\s*\"", "LOW"),
    ("RS_PANIC", r"\bpanic!\s*\(", "MEDIUM"),
    ("RS_UNSAFE_BLOCK", r"\bunsafe\s*\{", "MEDIUM"),
    ("RS_TODO", r"//\s*(TODO|FIXME|HACK|XXX)\b", "LOW"),
    ("RS_TODO_MACRO", r"\btodo!\s*\(", "LOW"),
    ("RS_UNIMPLEMENTED", r"\bunimplemented!\s*\(", "MEDIUM"),
    ("RS_PROCESS_COMMAND", r"std::process::Command::new", "LOW"),
    ("RS_HTTP_CLIENT_NEW", r"(reqwest|ureq|isahc)::(Client|get|post)\s*\(", "LOW"),
]

SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv",
             ".venv-linux", "venv", "dist", "build",
             "docs/brand/public_launch_media_kit_v0_1/extracted",
             "tools/cognitive_foundry/claude_lane/output"}


def _iter_files(repo_root: Path, roots: List[str], suffix: str, max_bytes: int) -> Iterable[Path]:
    for r in roots:
        base = repo_root / r
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix == suffix:
                yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(suffix):
                    p = Path(dirpath) / fn
                    try:
                        if p.stat().st_size > max_bytes:
                            continue
                    except OSError:
                        continue
                    yield p


def _scan(paths: Iterable[Path], rules: List[tuple[str, str, str]],
          repo_root: Path, limit: int) -> List[dict]:
    findings: List[dict] = []
    compiled = [(name, re.compile(pat, flags=re.MULTILINE), sev) for name, pat, sev in rules]
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(repo_root).as_posix()
        for name, pat, sev in compiled:
            for m in pat.finditer(text):
                # line number
                line = text[:m.start()].count("\n") + 1
                # skip finding in our own scanner (self-reference noise)
                if "tools/audit/omni_audit" in rel and name.startswith(("PY_", "RS_")):
                    continue
                findings.append({
                    "finding_id": f"R{len(findings) + 1:05d}",
                    "rule": name,
                    "severity": sev,
                    "path": rel,
                    "line": line,
                    "excerpt": text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")[:200],
                })
                if len(findings) >= limit:
                    return findings
    return findings


def scan(repo_root: Path, python_roots: List[str], rust_roots: List[str],
         max_bytes: int, limit: int) -> List[dict]:
    py_findings = _scan(_iter_files(repo_root, python_roots, ".py", max_bytes),
                        PYTHON_RULES, repo_root, limit)
    remaining = max(0, limit - len(py_findings))
    rs_findings = _scan(_iter_files(repo_root, rust_roots, ".rs", max_bytes),
                        RUST_RULES, repo_root, remaining)
    return py_findings + rs_findings


def write_outputs(findings: List[dict], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "code_risks.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    return {"code_risks_json": str(path)}
