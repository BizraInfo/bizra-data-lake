import argparse
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
    "__pycache__",
    # Local datasets / generated artifacts
    "chat data sample",
    "receipts",
    "bizra_data_vault",
}

TEXT_EXTS = {
    ".rs",
    ".toml",
    ".lock",
    ".js",
    ".cjs",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".py",
    ".ps1",
    ".sh",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".scss",
}

DEFAULT_MAX_BYTES_TEXT_SCAN = 5 * 1024 * 1024  # 5MB


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(repo_root: Path, args: Sequence[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (p.stdout or "").strip()
        if not out and p.stderr:
            out = (p.stderr or "").strip()
        return p.returncode, out
    except Exception as e:
        return 1, f"git_error: {e}"


def iter_files(repo_root: Path) -> Iterable[Path]:
    stack = [repo_root]
    while stack:
        cur = stack.pop()
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    if child.name in SKIP_DIR_NAMES:
                        continue
                    stack.append(child)
                elif child.is_file():
                    yield child
        except Exception:
            continue


def safe_read_text(path: Path, max_bytes: int) -> str:
    try:
        st = path.stat()
        if st.st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def sanitize_rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("/", "\\")
    except Exception:
        return str(path)


def try_parse_toml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import tomllib  # type: ignore
    except Exception:
        tomllib = None  # type: ignore
    if tomllib is None:
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_cargo_dependencies(repo_root: Path) -> Dict[str, Any]:
    cargo = repo_root / "Cargo.toml"
    if not cargo.exists():
        return {"present": False, "truth_label": "MEASURED"}

    parsed = try_parse_toml(cargo)
    deps: Dict[str, str] = {}
    if parsed and isinstance(parsed, dict):
        raw = parsed.get("dependencies") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, str):
                    deps[str(k)] = v
                elif isinstance(v, dict):
                    ver = v.get("version")
                    deps[str(k)] = str(ver) if ver is not None else "(complex)"
                else:
                    deps[str(k)] = "(complex)"
    else:
        # Fallback: best-effort regex parse
        txt = cargo.read_text(encoding="utf-8", errors="ignore")
        in_deps = False
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("[dependencies]"):
                in_deps = True
                continue
            if in_deps and s.startswith("[") and s.endswith("]"):
                break
            if not in_deps or not s or s.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\"([^\"]+)\"", s)
            if m:
                deps[m.group(1)] = m.group(2)
                continue
            m = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\\{.*version\\s*=\\s*\"([^\"]+)\".*\\}", s)
            if m:
                deps[m.group(1)] = m.group(2)

    pkg = {}
    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("package"), dict):
        p = parsed["package"]
        pkg = {
            "name": p.get("name"),
            "version": p.get("version"),
            "edition": p.get("edition"),
        }

    return {
        "present": True,
        "package": pkg,
        "dependencies": [{"name": k, "version": v} for k, v in sorted(deps.items())],
        "counts": {"dependencies": len(deps)},
        "truth_label": "MEASURED",
        "inputs": [{"path": str(cargo), "note": "Cargo.toml parsed"}],
    }


def parse_package_jsons(repo_root: Path) -> Dict[str, Any]:
    packages: List[Dict[str, Any]] = []
    for p in iter_files(repo_root):
        if p.name != "package.json":
            continue
        if p.parent.name in SKIP_DIR_NAMES:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            deps = obj.get("dependencies") or {}
            dev = obj.get("devDependencies") or {}
            packages.append(
                {
                    "path": sanitize_rel_path(p, repo_root),
                    "name": obj.get("name"),
                    "version": obj.get("version"),
                    "dependencies_count": len(deps) if isinstance(deps, dict) else 0,
                    "dev_dependencies_count": len(dev) if isinstance(dev, dict) else 0,
                }
            )
        except Exception:
            packages.append({"path": sanitize_rel_path(p, repo_root), "error": "parse_failed"})

    return {"packages": packages, "counts": {"packages": len(packages)}, "truth_label": "MEASURED"}


def compute_security_signals(repo_root: Path) -> Dict[str, Any]:
    patterns = {
        "rust_unsafe": (re.compile(r"\bunsafe\b"), {".rs"}),
        "rust_unwrap": (re.compile(r"\.unwrap\s*\("), {".rs"}),
        "rust_expect": (re.compile(r"\.expect\s*\("), {".rs"}),
        "node_eval": (re.compile(r"\beval\s*\("), {".js", ".cjs", ".mjs", ".ts"}),
        "node_child_exec": (
            re.compile(r"\bchild_process\s*\.\s*exec(?:Sync)?\s*\("),
            {".js", ".cjs", ".mjs", ".ts"},
        ),
        "node_spawn": (
            re.compile(r"\bchild_process\s*\.\s*spawn(?:Sync)?\s*\("),
            {".js", ".cjs", ".mjs", ".ts"},
        ),
        "ps_invoke_expression": (re.compile(r"\bInvoke-Expression\b|\bIEX\b", re.IGNORECASE), {".ps1"}),
    }

    counts = Counter()
    scanned_files = 0
    for f in iter_files(repo_root):
        ext = f.suffix.lower()
        if ext not in TEXT_EXTS:
            continue
        if f.stat().st_size > DEFAULT_MAX_BYTES_TEXT_SCAN:
            continue
        text = safe_read_text(f, DEFAULT_MAX_BYTES_TEXT_SCAN)
        if not text:
            continue
        scanned_files += 1
        for name, (rx, exts) in patterns.items():
            if ext not in exts:
                continue
            counts[name] += len(rx.findall(text))

    return {
        "counts": dict(counts),
        "scanned_files": scanned_files,
        "truth_label": "DERIVED",
        "notes": [
            "Heuristic scan for high-risk constructs; use as a prioritization signal, not a proof of vulnerability.",
            f"Skipped files > {DEFAULT_MAX_BYTES_TEXT_SCAN} bytes.",
        ],
    }


def build_summary_text(inv: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("CODEBASE_CONTEXT (MEASURED/DERIVED, high-SNR)")
    lines.append(f"generated_at: {inv.get('generated_at')}")
    lines.append(f"repo_root: {inv.get('repo_root')}")
    lines.append("")

    git = inv.get("git") or {}
    lines.append("Git:")
    lines.append(f"- head: {git.get('head')}")
    lines.append(f"- branch: {git.get('branch')}")
    lines.append(f"- dirty: {git.get('dirty')}")
    lines.append("")

    files = inv.get("files") or {}
    lines.append("Files:")
    lines.append(f"- total_files: {files.get('total_files')}")
    lines.append(f"- total_bytes: {files.get('total_bytes')}")
    lines.append("- top_extensions:")
    for r in (files.get("by_extension") or [])[:10]:
        lines.append(f"  - {r.get('ext')}: files={r.get('files')} bytes={r.get('bytes')}")
    lines.append("")

    loc = inv.get("loc") or {}
    lines.append("LOC (heuristic):")
    lines.append(f"- scanned_files: {loc.get('scanned_files')}")
    for r in (loc.get("by_extension") or [])[:10]:
        lines.append(f"  - {r.get('ext')}: loc={r.get('loc')} files={r.get('files')}")
    lines.append("")

    cargo = (inv.get("dependencies") or {}).get("cargo") or {}
    if cargo.get("present"):
        lines.append("Rust/Cargo:")
        pkg = cargo.get("package") or {}
        lines.append(f"- crate: {pkg.get('name')} {pkg.get('version')} (edition {pkg.get('edition')})")
        lines.append(f"- deps: {((cargo.get('counts') or {}).get('dependencies'))}")
        lines.append("")

    pkgs = (inv.get("dependencies") or {}).get("npm") or {}
    if (pkgs.get("counts") or {}).get("packages"):
        lines.append("Node packages:")
        for p in (pkgs.get("packages") or [])[:5]:
            lines.append(f"- {p.get('name')} {p.get('version')} @ {p.get('path')}")
        lines.append("")

    sec = inv.get("security_signals") or {}
    lines.append("Security signals (heuristic):")
    for k, v in sorted((sec.get("counts") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    docs = inv.get("docs") or {}
    lines.append("Docs:")
    lines.append(f"- markdown_files: {docs.get('markdown_files')}")
    for d in (docs.get("top_docs") or [])[:8]:
        lines.append(f"- {d}")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="High-SNR codebase inventory (evidence-first).")
    ap.add_argument("--repo-root", default="", help="Repo root (defaults to script parent)")
    ap.add_argument("--out", default="", help="Write JSON to this path")
    ap.add_argument("--summary-out", default="", help="Write a compact context text file")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    rc, head = run_git(repo_root, ["rev-parse", "HEAD"])
    _, branch = run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    _, status = run_git(repo_root, ["status", "--porcelain=v1"])
    dirty = bool(status.strip()) if rc == 0 else None

    ext_files = Counter()
    ext_bytes = Counter()
    total_files = 0
    total_bytes = 0

    loc_files = 0
    loc_by_ext = Counter()
    loc_files_by_ext = Counter()

    largest: List[Tuple[int, str]] = []

    for f in iter_files(repo_root):
        total_files += 1
        try:
            st = f.stat()
            size = int(st.st_size)
        except Exception:
            size = 0

        total_bytes += size
        ext = f.suffix.lower() if f.suffix else "(none)"
        ext_files[ext] += 1
        ext_bytes[ext] += size

        if size and len(largest) < 30:
            largest.append((size, sanitize_rel_path(f, repo_root)))
            largest.sort(reverse=True)
        elif size and largest and size > largest[-1][0]:
            largest.append((size, sanitize_rel_path(f, repo_root)))
            largest.sort(reverse=True)
            largest = largest[:30]

        if ext in TEXT_EXTS:
            txt = safe_read_text(f, DEFAULT_MAX_BYTES_TEXT_SCAN)
            if txt:
                loc_files += 1
                loc = count_lines(txt)
                loc_by_ext[ext] += loc
                loc_files_by_ext[ext] += 1

    by_ext_rows = [
        {"ext": k, "files": int(ext_files[k]), "bytes": int(ext_bytes[k]), "truth_label": "MEASURED"}
        for k in ext_files.keys()
    ]
    by_ext_rows.sort(key=lambda r: (-r["bytes"], r["ext"]))

    loc_rows = [
        {"ext": k, "loc": int(loc_by_ext[k]), "files": int(loc_files_by_ext[k]), "truth_label": "MEASURED"}
        for k in loc_by_ext.keys()
    ]
    loc_rows.sort(key=lambda r: (-r["loc"], r["ext"]))

    docs = []
    for f in iter_files(repo_root):
        if f.suffix.lower() == ".md":
            docs.append(sanitize_rel_path(f, repo_root))
    docs.sort()

    inv: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "repo_root": str(repo_root),
        "truth_label": "MEASURED",
        "git": {
            "head": head if rc == 0 else None,
            "branch": branch if rc == 0 else None,
            "dirty": dirty,
            "status_lines": len(status.splitlines()) if rc == 0 else None,
            "truth_label": "MEASURED",
        },
        "files": {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "by_extension": by_ext_rows,
            "largest_files": [{"bytes": int(sz), "path": p, "truth_label": "MEASURED"} for sz, p in largest],
            "truth_label": "MEASURED",
        },
        "loc": {
            "scanned_files": loc_files,
            "by_extension": loc_rows,
            "truth_label": "MEASURED",
            "notes": [f"Counts only files <= {DEFAULT_MAX_BYTES_TEXT_SCAN} bytes; binary/large files skipped."],
        },
        "dependencies": {
            "cargo": parse_cargo_dependencies(repo_root),
            "npm": parse_package_jsons(repo_root),
            "truth_label": "MEASURED",
        },
        "security_signals": compute_security_signals(repo_root),
        "docs": {
            "markdown_files": len(docs),
            "top_docs": docs[:25],
            "truth_label": "MEASURED",
        },
    }

    out_json = json.dumps(inv, indent=2, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
        print(str(out_path))
    else:
        print(out_json)

    if args.summary_out:
        summary_path = Path(args.summary_out).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(build_summary_text(inv), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
