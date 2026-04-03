
try:
    print("Importing SAPEEngine...")
    from bizra_kernel.sape_engine import SAPEEngine
    print("OK")
except Exception as e:
    print(f"FAIL SAPEEngine: {e}")

try:
    print("Importing SymbolicHarness...")
    from bizra_kernel.symbolic_harness import SymbolicHarness
    print("OK")
except Exception as e:
    print(f"FAIL SymbolicHarness: {e}")

try:
    print("Importing AbstractionElevator...")
    from bizra_kernel.abstraction_elevator import AbstractionElevator
    print("OK")
except Exception as e:
    print(f"FAIL AbstractionElevator: {e}")

try:
    print("Importing TensionStudio...")
    from bizra_kernel.tension_studio import TensionStudio
    print("OK")
except Exception as e:
    print(f"FAIL TensionStudio: {e}")

try:
    print("Importing IhsanVector...")
    from bizra_kernel.ihsan_vector import IhsanVector
    print("OK")
except Exception as e:
    print(f"FAIL IhsanVector: {e}")

try:
    print("Importing OmniAwareness...")
    from bizra_kernel.omni_awareness import OmniAwareness
    print("OK")
except Exception as e:
    print(f"FAIL OmniAwareness: {e}")

try:
    print("Importing CognitivePermanence...")
    from bizra_kernel.memory_system import CognitivePermanence
    print("OK")
except Exception as e:
    print(f"FAIL CognitivePermanence: {e}")

try:
    print("Importing ConsensusEngine...")
    from bizra_kernel.consensus_engine import ConsensusEngine
    print("OK")
except Exception as e:
    print(f"FAIL ConsensusEngine: {e}")

try:
    print("Importing verify all from init...")
    from bizra_kernel import *
    print("OK ALL")
except Exception as e:
    print(f"FAIL ALL: {e}")
