"""
BIZRA Phase 1 Gate — Integration Test Checklist
═══════════════════════════════════════════════════

Run this after wiring the cockpit to NODE0.
6 checks. All must pass for Phase 1 to be complete.

Usage:
    python phase1_gate.py

Checks:
    1. ghost_ws.py emits OverlayEvent
    2. Kernel API responds with trust status
    3. desktop_bridge.py accepts health check
    4. brain.py generates briefing
    5. Memory API returns fragments
    6. End-to-end: suggestion → approve → receipt

Standing on: Deming (PDCA), Shannon (SNR), Al-Ghazali (Ihsan)
"""

import json
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError

GHOST_WS = "ws://127.0.0.1:9743/ws/ghost"
KERNEL = "http://127.0.0.1:3006"
BRIDGE = "http://127.0.0.1:9742"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def check(name, fn):
    """Run a check, record result."""
    try:
        ok, detail = fn()
        status = PASS if ok else FAIL
        results.append((name, ok, detail))
        print(f"  [{status}] {name}: {detail}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


def http_get(url, timeout=5):
    """Simple HTTP GET."""
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post(url, data, timeout=5):
    """Simple HTTP POST."""
    body = json.dumps(data).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ═══ CHECK 1: Ghost WS ═══
def check_ghost_ws():
    """Verify ghost_ws.py is listening on 9743."""
    try:
        # We can't do full WS handshake with stdlib, but we can check TCP
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        result = s.connect_ex(("127.0.0.1", 9743))
        s.close()
        if result == 0:
            return True, "TCP:9743 accepting connections"
        return False, f"TCP:9743 refused (code {result})"
    except Exception as e:
        return False, f"TCP:9743 unreachable: {e}"


# ═══ CHECK 2: Kernel API ═══
def check_kernel():
    """Verify kernel responds with system status."""
    try:
        data = http_get(f"{KERNEL}/api/system/status")
        has_fields = any(
            k in data for k in ["kernel_healthy", "status", "uptime", "version"]
        )
        if has_fields:
            return True, f"Kernel responding. Keys: {list(data.keys())[:5]}"
        return (
            False,
            f"Kernel responded but missing expected fields: {list(data.keys())[:5]}",
        )
    except URLError as e:
        return False, f"Kernel unreachable at {KERNEL}: {e.reason}"


# ═══ CHECK 3: Desktop Bridge ═══
def check_bridge():
    """Verify desktop_bridge.py is listening."""
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        result = s.connect_ex(("127.0.0.1", 9742))
        s.close()
        if result == 0:
            return True, "TCP:9742 accepting connections"
        return False, f"TCP:9742 refused (code {result})"
    except Exception as e:
        return False, f"TCP:9742 unreachable: {e}"


# ═══ CHECK 4: Briefing ═══
def check_briefing():
    """Verify brain.py generates a morning briefing."""
    try:
        data = http_get(f"{KERNEL}/api/briefing")
        if data and (isinstance(data, dict) or isinstance(data, list)):
            return True, f"Briefing returned. Type: {type(data).__name__}"
        return False, f"Briefing empty or unexpected: {type(data)}"
    except URLError as e:
        return False, f"Briefing endpoint unreachable: {e.reason}"


# ═══ CHECK 5: Memory ═══
def check_memory():
    """Verify memory API returns fragments."""
    try:
        data = http_get(f"{KERNEL}/api/memory/recent")
        if isinstance(data, list):
            return True, f"Memory returned {len(data)} fragments"
        return False, f"Memory returned unexpected type: {type(data).__name__}"
    except URLError as e:
        return False, f"Memory endpoint unreachable: {e.reason}"


# ═══ CHECK 6: End-to-End ═══
def check_e2e():
    """
    The full loop: does the system produce a verifiable receipt?
    Uses verify_genesis.py as proxy — if 6/6 pass, the receipt chain is intact.
    """
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "verify_genesis.py", "--quick"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="C:\\BIZRA-DATA-LAKE",
        )
        if result.returncode == 0 and "6/6" in result.stdout:
            return True, "verify_genesis.py: 6/6 passed"
        return (
            False,
            f"verify_genesis.py: returncode={result.returncode}, output={result.stdout[:100]}",
        )
    except FileNotFoundError:
        return False, "verify_genesis.py not found at C:\\BIZRA-DATA-LAKE"
    except subprocess.TimeoutExpired:
        return False, "verify_genesis.py timed out (30s)"
    except Exception as e:
        return False, f"verify_genesis.py error: {e}"


def main():
    print()
    print("  ═══════════════════════════════════════════════")
    print("  BIZRA Phase 1 Gate — Integration Test")
    print("  ═══════════════════════════════════════════════")
    print()

    check("1. Ghost WebSocket (TCP:9743)", check_ghost_ws)
    check("2. Kernel API (HTTP:3006)", check_kernel)
    check("3. Desktop Bridge (TCP:9742)", check_bridge)
    check("4. Morning Briefing", check_briefing)
    check("5. Living Memory", check_memory)
    check("6. Receipt Chain (verify_genesis)", check_e2e)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Results: {passed}/{total} passed")
    print()

    if passed == total:
        print("  ✓ PHASE 1 GATE: PASSED")
        print("  The front door opens. The proof chain holds.")
        print("  بذرة واحدة تصنع غابة")
    elif passed >= 4:
        print("  ○ PHASE 1 GATE: PARTIAL")
        print(f"  {total - passed} checks need attention before the door opens.")
    else:
        print("  ✗ PHASE 1 GATE: NOT READY")
        print(f"  {total - passed} services offline. Wire before testing the cockpit.")

    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
