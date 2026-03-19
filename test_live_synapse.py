import json, sys, os
sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")

# Check if PyO3 module is available
try:
    import bizra
    print(f"RUST MODULE LIVE: bizra v{getattr(bizra, '__version__', '?')}")
    print(f"  IHSAN_THRESHOLD = {getattr(bizra, 'IHSAN_THRESHOLD', 'N/A')}")
    print(f"  SNR_THRESHOLD   = {getattr(bizra, 'SNR_THRESHOLD', 'N/A')}")

    # Test PyEventBridge
    bridge = bizra.PyEventBridge(False)
    wired = bridge.wire_subscribers()
    print(f"  PyEventBridge: {wired} subscribers wired")
    health = bridge.health()
    print(f"  Health: {health}")
    print("\nSYNAPSE READY: Python -> Rust constitutional pipeline ACTIVE")

except ImportError as e:
    print(f"RUST MODULE NOT COMPILED: {e}")
    # Check if .pyd exists
    import glob
    pyds = glob.glob(r"C:\BIZRA-DATA-LAKE\bizra-omega\target\release\*.pyd")
    pyds += glob.glob(r"C:\BIZRA-DATA-LAKE\bizra-omega\target\release\bizra*.dll")
    pyds += glob.glob(r"C:\BIZRA-DATA-LAKE\.venv\Lib\site-packages\bizra*.pyd")
    if pyds:
        print(f"  Found compiled artifacts: {pyds}")
    else:
        print("  No .pyd found — maturin develop hasn't completed")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
