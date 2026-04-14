#!/usr/bin/env python3
"""Compute canonical artifact hash for Cycle 1."""
import hashlib

files = sorted([
    "core/inference/_connection_pool.py",
    "core/sovereign/runtime_core.py",
    "core/pat/runtime.py",
    "core/sat/runtime.py",
    "core/sovereign/dema_router.py",
    "core/sovereign/fate_boundary.py",
    "deploy/node0/activation_smoke_test.py",
    "deploy/node0/integration_test.py",
])

h = hashlib.blake2b(digest_size=32)
for f in files:
    h.update(open(f, "rb").read())
print(f"BLAKE2B-256: {h.hexdigest()}")

try:
    import blake3
    h3 = blake3.blake3()
    for f in files:
        h3.update(open(f, "rb").read())
    print(f"BLAKE3: {h3.hexdigest()}")
except ImportError:
    print("blake3 not installed, using BLAKE2B-256 as canonical hash")
