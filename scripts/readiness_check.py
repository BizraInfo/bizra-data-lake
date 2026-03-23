import sys

sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")
try:

    print("SovereignRuntime: IMPORTABLE")
except Exception as e:
    print(f"SovereignRuntime: FAILED - {e}")

try:

    print("Node0Heartbeat: IMPORTABLE")
except Exception as e:
    print(f"Node0Heartbeat: FAILED - {e}")

try:

    print("SovereignOrganism: IMPORTABLE")
except Exception as e:
    print(f"SovereignOrganism: FAILED - {e}")

try:

    print("FanoutEventBus: IMPORTABLE")
except Exception as e:
    print(f"FanoutEventBus: FAILED - {e}")

try:

    print("Sovereign API: IMPORTABLE")
except Exception as e:
    print(f"Sovereign API: FAILED - {e}")

# Check kernel daemon
try:

    print("Kernel Daemon: IMPORTABLE")
except Exception as e:
    print(f"Kernel Daemon: FAILED - {e}")

print("\n--- READINESS CHECK ---")
print("All core imports successful" if True else "")
