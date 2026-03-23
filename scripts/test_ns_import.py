import os
import sys

# Add project root to sys.path
project_root = "/mnt/c/BIZRA-DATA-LAKE"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

print(f"sys.path[0] = {sys.path[0]}")
print(f"cwd = {os.getcwd()}")

try:
    from core.sovereign.mission_nervous_system import SovereignNervousSystem

    print("SovereignNervousSystem: OK")
except Exception as e:
    print(f"SovereignNervousSystem FAIL: {e}")

try:
    from core.sovereign.moe_bridge import MOEBridge

    print("MOEBridge: OK")
    bridge = MOEBridge.create()
    print(f"Bridge created: {type(bridge)}")
except Exception as e:
    print(f"MOEBridge FAIL: {e}")

try:
    ns = SovereignNervousSystem(inference=bridge)
    print(f"NervousSystem created: {type(ns)}")
except Exception as e:
    print(f"NervousSystem creation FAIL: {e}")
