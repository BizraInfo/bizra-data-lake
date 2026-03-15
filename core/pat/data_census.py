#!/usr/bin/env python3
"""PAT Data Source Census — scan each root separately with progress."""
import os, time, json
from collections import defaultdict
from pathlib import Path

SKIP = {".git","node_modules",".venv",".venv-linux",".venv-apex","__pycache__",
        ".mypy_cache",".pytest_cache",".ruff_cache",".cache",".hypothesis",
        ".benchmarks","dist",".next",".fastembed_cache","coverage",".codex",
        ".swarm",".claude-flow",".tmp_prod_artifacts_v2"}
SKIP_EXT = {".exe",".dll",".so",".dylib",".whl",".pyc",".pyo",".lock",".idx",".pack"}

ROOTS = [
    "/mnt/c/BIZRA-DATA-LAKE",
    "/mnt/c/BIZRA-Dual-Agentic-system--main",
    "/mnt/c/BIZRA-NODE0",
    "/mnt/c/Users/BIZRA-OS/Downloads",
    "/mnt/b/BIZRA-SOVEREIGN",
    "/mnt/c/BIZRA-PROJECTS",
    "/mnt/c/BIZRA-TaskMaster",
]

def scan(root):
    rp = Path(root)
    if not rp.exists():
        return None
    t0 = time.time()
    by_ext = defaultdict(lambda: {"n": 0, "b": 0})
    by_kind = defaultdict(lambda: {"n": 0, "b": 0})
    total = 0
    KIND_MAP = {
        ".py":"code",".rs":"code",".ts":"code",".tsx":"code",".js":"code",
        ".jsx":"code",".sh":"code",".ps1":"code",".bat":"code",".sql":"code",
        ".md":"doc",".txt":"doc",".pdf":"doc",".docx":"doc",".pptx":"doc",".xlsx":"doc",
        ".json":"data",".jsonl":"data",".yaml":"data",".yml":"data",".toml":"data",
        ".csv":"data",".parquet":"data",".db":"data",
        ".html":"web",".css":"web",".svg":"web",
        ".png":"media",".jpg":"media",".jpeg":"media",".gif":"media",
        ".webp":"media",".mp4":"media",".m4a":"media",".wav":"media",
        ".zip":"archive",".tar":"archive",".gz":"archive",".tgz":"archive",
    }
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP]
        for fn in fns:
            ext = Path(fn).suffix.lower()
            if ext in SKIP_EXT:
                continue
            try:
                sz = (Path(dp) / fn).stat().st_size
                by_ext[ext]["n"] += 1
                by_ext[ext]["b"] += sz
                kind = KIND_MAP.get(ext, "other")
                by_kind[kind]["n"] += 1
                by_kind[kind]["b"] += sz
                total += 1
            except (PermissionError, OSError):
                pass
    elapsed = time.time() - t0
    return {
        "root": root, "files": total, "time": round(elapsed, 1),
        "total_gb": round(sum(v["b"] for v in by_ext.values()) / 1e9, 3),
        "by_ext": dict(sorted(by_ext.items(), key=lambda x: -x[1]["n"])[:12]),
        "by_kind": dict(sorted(by_kind.items(), key=lambda x: -x[1]["b"])),
    }

print("=" * 60, flush=True)
print("  PAT DATA SOURCE CENSUS", flush=True)
print("=" * 60, flush=True)
print(flush=True)

grand_files = 0
grand_gb = 0
grand_kinds = defaultdict(lambda: {"n": 0, "b": 0})
results = []

for root in ROOTS:
    print(f"  [{root}]", flush=True)
    r = scan(root)
    if r is None:
        print(f"    NOT FOUND", flush=True)
    else:
        results.append(r)
        grand_files += r["files"]
        grand_gb += r["total_gb"]
        print(f"    {r['files']:>8,} files | {r['total_gb']:>8.3f} GB | {r['time']}s", flush=True)
        for kind, info in r["by_kind"].items():
            grand_kinds[kind]["n"] += info["n"]
            grand_kinds[kind]["b"] += info["b"]
            print(f"      {kind:8s}: {info['n']:>7,} files  {info['b']/1e6:>8.1f} MB", flush=True)
    print(flush=True)

print(f"  {'='*50}", flush=True)
print(f"  GRAND TOTAL: {grand_files:,} files | {grand_gb:.3f} GB", flush=True)
print(flush=True)
print(f"  BY KIND (all sources):", flush=True)
for kind in sorted(grand_kinds, key=lambda k: -grand_kinds[k]["b"]):
    info = grand_kinds[kind]
    print(f"    {kind:8s}: {info['n']:>8,} files  {info['b']/1e9:>8.3f} GB", flush=True)

# Write manifest
manifest = {
    "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "grand_total_files": grand_files,
    "grand_total_gb": round(grand_gb, 3),
    "sources": results,
    "by_kind": {k: dict(v) for k, v in grand_kinds.items()},
    "uncovered_sources": [
        {"name": "Chat history (Claude/GPT/Gemini/Qwen/DeepSeek)", "est_conversations": "4000-5000", "status": "NOT_SCANNED", "action": "Export from each platform"},
        {"name": "Google Drive", "status": "NOT_SCANNED", "action": "Mount or export"},
        {"name": "Mobile device", "status": "NOT_SCANNED", "action": "Transfer to B:\\\\"},
        {"name": "Email archives", "status": "NOT_SCANNED", "action": "Export .mbox or .eml"},
    ],
}
with open("04_GOLD/data_source_census.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\n  Census written: 04_GOLD/data_source_census.json", flush=True)
