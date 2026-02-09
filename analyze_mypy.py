#!/usr/bin/env python3
"""Analyze mypy output to find highest-impact fixes.

Handles multi-line error messages by joining continuation lines.
Usage:
  python analyze_mypy.py             # Full analysis
  python analyze_mypy.py FILE_PATH   # Errors for specific file
"""
import re
import sys
from collections import Counter

# Read previously saved output
with open("/mnt/c/BIZRA-DATA-LAKE/mypy_full.txt") as f:
    raw = f.read()

# Join continuation lines (lines not starting with "core/")
joined_lines = []
for line in raw.split("\n"):
    if line.startswith("core/") and ": error:" in line:
        joined_lines.append(line)
    elif line.startswith("core/") and ": note:" in line:
        continue  # skip notes
    elif joined_lines and not line.startswith("core/") and not line.strip().startswith("^"):
        # continuation of previous error line
        joined_lines[-1] += " " + line.strip()

# If a file argument is given, filter to just that file's errors
if len(sys.argv) > 1:
    target = sys.argv[1]
    for line in joined_lines:
        if line.startswith(target):
            print(line)
    sys.exit(0)

print(f"TOTAL ERRORS: {len(joined_lines)}\n")

# By file
file_counts = Counter()
for line in joined_lines:
    fp = line.split(":")[0]
    file_counts[fp] += 1

print("=== TOP 40 FILES ===")
for fp, c in file_counts.most_common(40):
    print(f"  {c:4d}  {fp}")

# By error code (the bracketed code at end of line)
code_counts = Counter()
for line in joined_lines:
    # Find the last bracketed code in the line
    codes = re.findall(r"\[([a-z][a-z0-9_-]+)\]", line)
    if codes:
        code_counts[codes[-1]] += 1
    else:
        code_counts["unknown"] += 1

print("\n=== TOP 20 ERROR CODES ===")
for code, c in code_counts.most_common(20):
    print(f"  {c:4d}  [{code}]")

# By directory
dir_counts = Counter()
for fp, c in file_counts.items():
    parts = fp.split("/")
    if len(parts) >= 2:
        d = "/".join(parts[:2])
    else:
        d = fp
    dir_counts[d] += c

print("\n=== TOP 20 DIRECTORIES ===")
for d, c in dir_counts.most_common(20):
    print(f"  {c:4d}  {d}")

# Show sample errors for each code
print("\n=== SAMPLE ERRORS (first 2 per code) ===")
code_samples: dict[str, list[str]] = {}
for line in joined_lines:
    codes = re.findall(r"\[([a-z][a-z0-9_-]+)\]", line)
    if codes:
        code = codes[-1]
        if code not in code_samples:
            code_samples[code] = []
        if len(code_samples[code]) < 2:
            code_samples[code].append(line.strip()[:120])

for code in sorted(code_samples, key=lambda c: -code_counts[c]):
    print(f"\n[{code}] ({code_counts[code]} errors):")
    for sample in code_samples[code]:
        print(f"  {sample}")
