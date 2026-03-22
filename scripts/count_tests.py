import os, subprocess, sys

root = r"C:\BIZRA-DATA-LAKE\tests"
dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

print(f"Test directories found: {len(dirs)}")
print()

for d in dirs:
    path = os.path.join(root, d)
    count = 0
    for r, _, files in os.walk(path):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                fp = os.path.join(r, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                        count += sum(1 for line in fh if line.strip().startswith("def test_"))
                except:
                    pass
    if count > 0:
        print(f"  {d}: {count} test functions")

# Also count root-level test files
root_count = 0
for f in os.listdir(root):
    if f.startswith("test_") and f.endswith(".py"):
        fp = os.path.join(root, f)
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                root_count += sum(1 for line in fh if line.strip().startswith("def test_"))
        except:
            pass
if root_count > 0:
    print(f"  (root): {root_count} test functions")
