import os
from collections import Counter

root = r"C:\BIZRA-DATA-LAKE"
SKIP = {
    "node_modules",
    ".git",
    "target",
    ".venv",
    ".venv-apex",
    ".venv-linux",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    ".hypothesis",
}

ext_count = Counter()
ext_lines = Counter()
cargo_tomls = []
rs_files = []

for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        if ext:
            ext_count[ext] += 1
            if ext in {
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
            }:
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = sum(1 for _ in fh)
                    ext_lines[ext] += lines
                except (OSError, UnicodeDecodeError):
                    pass
            if ext == ".rs":
                rs_files.append(os.path.join(dirpath, f))
            if f == "Cargo.toml":
                cargo_tomls.append(os.path.join(dirpath, f))

print("=" * 70)
print("  BIZRA-DATA-LAKE — FULL LANGUAGE CENSUS")
print("=" * 70)

# Top extensions by file count
print("\n  BY FILE COUNT (top 20):")
for ext, count in ext_count.most_common(20):
    print(f"    {ext:10} {count:6} files")

# Top extensions by line count (code only)
print("\n  BY LINE COUNT (source files):")
for ext, lines in ext_lines.most_common(20):
    print(f"    {ext:10} {lines:8} lines")

# Rust specifics
print(f"\n  RUST FILES: {len(rs_files)}")
print(f"  CARGO.TOML FILES: {len(cargo_tomls)}")
for ct in cargo_tomls:
    rel = ct.replace(root, "")
    print(f"    {rel}")

# Show first 30 .rs file paths
print(f"\n  RUST SOURCE FILES (first 30 of {len(rs_files)}):")
for rf in rs_files[:30]:
    rel = rf.replace(root, "")
    print(f"    {rel}")
if len(rs_files) > 30:
    print(f"    ... and {len(rs_files) - 30} more")

# Python vs Rust comparison
py_files = ext_count.get(".py", 0)
py_lines = ext_lines.get(".py", 0)
rs_count = ext_count.get(".rs", 0)
rs_lines = ext_lines.get(".rs", 0)
ts_files = ext_count.get(".ts", 0) + ext_count.get(".tsx", 0)
ts_lines = ext_lines.get(".ts", 0) + ext_lines.get(".tsx", 0)

print("\n  LANGUAGE COMPARISON:")
print(f"    Rust:       {rs_count:5} files / {rs_lines:8} lines")
print(f"    Python:     {py_files:5} files / {py_lines:8} lines")
print(f"    TypeScript: {ts_files:5} files / {ts_lines:8} lines")
print(
    f"    Solidity:   {ext_count.get('.sol', 0):5} files / {ext_lines.get('.sol', 0):8} lines"
)

if rs_lines > py_lines:
    ratio = rs_lines / py_lines if py_lines > 0 else float("inf")
    print(f"\n  VERDICT: Rust dominates — {ratio:.1f}x more lines than Python")
elif py_lines > rs_lines:
    ratio = py_lines / rs_lines if rs_lines > 0 else float("inf")
    print(f"\n  VERDICT: Python dominates — {ratio:.1f}x more lines than Rust")
else:
    print("\n  VERDICT: Equal")

print("=" * 70)
