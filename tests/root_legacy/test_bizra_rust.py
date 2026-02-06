#!/usr/bin/env python3
"""Quick test to verify Rust-Python integration"""
import sys

try:
    import bizra
    print(f"✅ bizra module imported successfully")
    print(f"Version: {bizra.__version__}")
    print(f"IHSAN_THRESHOLD: {bizra.IHSAN_THRESHOLD}")
    print(f"SNR_THRESHOLD: {bizra.SNR_THRESHOLD}")
    
    # Test if it's the compiled Rust version or fallback
    identity = bizra.NodeIdentity()
    print(f"\n✅ NodeIdentity created: {identity.node_id}")
    
    # Test signing (Rust should be much faster)
    import time
    message = b"BIZRA performance test"
    start = time.perf_counter()
    for _ in range(1000):
        sig = identity.sign(message)
    elapsed = time.perf_counter() - start
    
    print(f"\n⚡ Performance: 1000 signatures in {elapsed:.4f}s")
    print(f"   Rate: {1000/elapsed:.0f} ops/sec")
    
    if elapsed < 0.1:
        print("   🚀 RUST NATIVE (Expected: 57K ops/sec)")
    else:
        print("   🐌 PYTHON FALLBACK (Much slower)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
