#!/usr/bin/env python3
"""Simple script to run cargo check and capture output."""
import subprocess
import os
import sys

os.chdir("/mnt/c/BIZRA-Dual-Agentic-system--main")

try:
    result = subprocess.run(
        ["/root/.cargo/bin/cargo", "check"],
        capture_output=True,
        text=True,
        timeout=300
    )
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print("\nReturn code:", result.returncode)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
