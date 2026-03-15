#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 << 'EOF'
import sys; sys.path.insert(0, ".")
print("=== FINDING REAL API SIGNATURES ===")
print()

# 1. EventBus — check subscribe returns
from core.sovereign.event_bus import EventBus, Event
bus = EventBus()
print("EventBus._subscribers type:", type(bus._subscribers))
print("EventBus methods:", [m for m in dir(bus) if not m.startswith("_")])
print()

# 2. Reflex cache — find real location
import importlib, pkgutil
for mod in pkgutil.walk_packages(["core"], prefix="core."):
    if "reflex" in mod.name.lower():
        print(f"Reflex module found: {mod.name}")
print()

# 3. PCI gates — real constructor
from core.pci.gates import PCIGateKeeper
import inspect
print("PCIGateKeeper.__init__ sig:", inspect.signature(PCIGateKeeper.__init__))
print("PCIGateKeeper methods:", [m for m in dir(PCIGateKeeper) if not m.startswith("_")])
print()

# 4. BLOOM — real exports
from core.token import bloom
print("bloom exports:", [x for x in dir(bloom) if not x.startswith("_")])
print()

# 5. EntropyRouter — real methods
from core.reasoning.entropy_router import EntropyRouter
print("EntropyRouter methods:", [m for m in dir(EntropyRouter) if not m.startswith("_")])
er = EntropyRouter()
print("EntropyRouter instance attrs:", [a for a in dir(er) if not a.startswith("_")])
print()

# 6. Living memory — real exports
from core.living_memory import core as lmc
print("living_memory.core exports:", [x for x in dir(lmc) if not x.startswith("_")])
print()

# 7. Constitutional simulation
from core.constitutional import simulation as csim
print("constitutional.simulation exports:", [x for x in dir(csim) if not x.startswith("_")])
print()
EOF
