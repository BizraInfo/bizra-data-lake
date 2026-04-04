import os
import collections

exts = collections.Counter()
ext_lines = collections.Counter()
ext_dirs = collections.defaultdict(set)
cargo_tomls = []
skip = {
    "node_modules",
    ".git",
    "target",
    ".venv",
    ".venv-apex",
    ".venv-linux",
    ".mypy_cache",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "out",
    ".hypothesis",
    "~",
    ".swarm",
    ".agentdb",
    ".claude",
    ".claude-flow",
    ".codex",
    ".proof-forge",
    ".tmp_prod_artifacts_v2",
    "99_ARCHIVE",
    "99_QUARANTINE",
    "voices",
    "checkpoints",
    "models",
    "logs",
}
code_exts = {
    ".rs",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".sol",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".html",
    ".css",
    ".json",
    ".sh",
    ".ps1",
    ".bat",
    ".coq",
    ".v",
    ".wasm",
    ".wat",
}

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in skip]
    for f in files:
        fp = os.path.join(root, f)
        ext = os.path.splitext(f)[1].lower()
        if ext in code_exts:
            exts[ext] += 1
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = sum(1 for _ in fh)
                ext_lines[ext] += lines
            except (OSError, UnicodeDecodeError):
                pass
            ext_dirs[ext].add(root)
        if f == "Cargo.toml":
            cargo_tomls.append(fp)

print("=" * 65)
print("  BIZRA-DATA-LAKE — COMPLETE LANGUAGE CENSUS")
print("=" * 65)
print(f'  {"EXT":8} {"FILES":>6} {"LINES":>9}  {"DIRS":>5}')
print("  " + "-" * 35)
for ext, count in sorted(ext_lines.items(), key=lambda x: -x[1]):
    dirs_count = len(ext_dirs[ext])
    print(f"  {ext:8} {exts[ext]:6} {ext_lines[ext]:9}  {dirs_count:5}")

total_files = sum(exts.values())
total_lines = sum(ext_lines.values())
print("  " + "-" * 35)
print(f'  {"TOTAL":8} {total_files:6} {total_lines:9}')

# Rust vs Python specifically
rs_lines = ext_lines.get(".rs", 0)
py_lines = ext_lines.get(".py", 0)
ts_lines = ext_lines.get(".ts", 0) + ext_lines.get(".tsx", 0)
print(f'\n  RUST:       {rs_lines:>8} lines ({exts.get(".rs",0)} files)')
print(f'  PYTHON:     {py_lines:>8} lines ({exts.get(".py",0)} files)')
print(
    f'  TYPESCRIPT: {ts_lines:>8} lines ({exts.get(".ts",0)+exts.get(".tsx",0)} files)'
)

if rs_lines > 0 and py_lines > 0:
    print(f"\n  Rust/Python ratio: {rs_lines/py_lines:.2f}x")

print(f"\n  Cargo.toml locations ({len(cargo_tomls)}):")
for c in sorted(cargo_tomls):
    print(f"    {c}")

# Find Rust workspace roots
print("\n  Rust source directories:")
rs_dirs = sorted(ext_dirs.get(".rs", set()))
for d in rs_dirs[:30]:
    rs_count = sum(1 for f in os.listdir(d) if f.endswith(".rs"))
    print(f"    {d} ({rs_count} .rs files)")
if len(rs_dirs) > 30:
    print(f"    ... and {len(rs_dirs)-30} more")

print("=" * 65)
