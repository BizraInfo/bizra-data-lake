commit acafe7aa473ed9179cd3ce975b3593ef80ac8134
Author: bizrainfo <m.beshr@bizra.info>
Date:   Tue Feb 24 20:51:58 2026 +0400

    chore: normalize CRLF→LF line endings, fix MCP config, add test collection guard
    
    - Normalize 148 Python files from CRLF to LF line endings across core/ and tests/
    - Fix .mcp.json: correct npm scope (@anthropic-ai → @modelcontextprotocol), bump to v1.1.0
    - Add torch/pandas collection guard in tests/integration/conftest.py to prevent
      pandas.__spec__ failures during full-suite collection
    - Add sape_metrics.py utility script
    - Update .gitignore: exclude BIZRA-DATA-LAKE/ worktree and .codeviz/ tool output
    
    Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

diff --git a/sape_metrics.py b/sape_metrics.py
new file mode 100644
index 0000000..bf51532
--- /dev/null
+++ b/sape_metrics.py
@@ -0,0 +1,66 @@
+import subprocess, os, glob
+os.chdir(r"C:\BIZRA-DATA-LAKE")
+
+# Python test count
+r = subprocess.run(["python", "-m", "pytest", "tests/", "--co", "-q"], capture_output=True, text=True, timeout=120)
+lines = r.stdout.strip().split("\n")
+py_tests = lines[-1] if lines else "unknown"
+print(f"PYTHON_TESTS: {py_tests}")
+
+# Rust test count
+r2 = subprocess.run(["cargo", "test", "--workspace", "--", "--list"], capture_output=True, text=True, timeout=120)
+rust_tests = sum(1 for l in r2.stdout.split("\n") if ": test" in l)
+print(f"RUST_TESTS: {rust_tests}")
+
+# Python LOC
+py_files = glob.glob("core/**/*.py", recursive=True)
+py_loc = 0
+for f in py_files:
+    try:
+        py_loc += sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
+    except:
+        pass
+print(f"PYTHON_LOC: {py_loc}")
+print(f"PYTHON_FILES: {len(py_files)}")
+
+# Rust LOC
+rs_files = glob.glob("bizra-omega/**/*.rs", recursive=True)
+rs_loc = 0
+for f in rs_files:
+    try:
+        rs_loc += sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
+    except:
+        pass
+print(f"RUST_LOC: {rs_loc}")
+print(f"RUST_FILES: {len(rs_files)}")
+
+# TypeScript LOC
+ts_files = glob.glob("**/*.ts", recursive=True) + glob.glob("**/*.tsx", recursive=True)
+ts_files = [f for f in ts_files if "node_modules" not in f]
+ts_loc = 0
+for f in ts_files:
+    try:
+        ts_loc += sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
+    except:
+        pass
+print(f"TS_LOC: {ts_loc}")
+print(f"TS_FILES: {len(ts_files)}")
+
+# Total
+total_loc = py_loc + rs_loc + ts_loc
+print(f"TOTAL_LOC: {total_loc}")
+
+# Key module sizes
+modules = [
+    "core/integration/constants.py",
+    "core/sovereign/omega_engine.py",
+    "core/constitutional/omega_engine.py",
+    "core/token/ledger.py",
+    "core/sovereign/adl_invariant.py",
+]
+for m in modules:
+    try:
+        loc = sum(1 for _ in open(m, encoding="utf-8", errors="ignore"))
+        print(f"MODULE {m}: {loc} lines")
+    except:
+        print(f"MODULE {m}: NOT FOUND")
