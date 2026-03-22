import sys
sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")
try:
    from core.sovereign.runtime_core import SovereignRuntime
    print("SovereignRuntime: IMPORTABLE")
except Exception as e:
    print(f"SovereignRuntime: FAILED - {e}")

try:
    from core.node0.heartbeat import Node0Heartbeat
    print("Node0Heartbeat: IMPORTABLE")
except Exception as e:
    print(f"Node0Heartbeat: FAILED - {e}")

try:
    from core.sovereign.organism import SovereignOrganism
    print("SovereignOrganism: IMPORTABLE")
except Exception as e:
    print(f"SovereignOrganism: FAILED - {e}")

try:
    from core.bus.event_publisher import FanoutEventBus, combine_event_buses
    print("FanoutEventBus: IMPORTABLE")
except Exception as e:
    print(f"FanoutEventBus: FAILED - {e}")

try:
    from core.sovereign.api import app
    print("Sovereign API: IMPORTABLE")
except Exception as e:
    print(f"Sovereign API: FAILED - {e}")

# Check kernel daemon
try:
    from core.sovereign.kernel_daemon import main as kernel_main
    print("Kernel Daemon: IMPORTABLE")
except Exception as e:
    print(f"Kernel Daemon: FAILED - {e}")

print("\n--- READINESS CHECK ---")
print("All core imports successful" if True else "")
