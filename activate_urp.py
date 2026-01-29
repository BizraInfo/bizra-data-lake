# BIZRA Unified Resource Protocol (URP) v1.0
# Standardizes resource discovery and access across the BIZRA ecosystem
# Part of Phase 1: Foundation Deploy

import os
import psutil
import torch
import json
from pathlib import Path

class BIZRA_URP:
    def __init__(self):
        self.config_path = Path("C:/BIZRA-DATA-LAKE/urp_registry.json")
        self.registry = {
            "compute": self._scan_compute(),
            "storage": self._scan_storage(),
            "network": self._scan_network()
        }
        self.save_registry()

    def _scan_compute(self):
        resources = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "load": psutil.cpu_percent()
            },
            "ram": {
                "total_gb": psutil.virtual_memory().total >> 30,
                "available_gb": psutil.virtual_memory().available >> 30
            }
        }
        
        if torch.cuda.is_available():
            resources["gpu"] = {
                "count": torch.cuda.device_count(),
                "devices": []
            }
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                resources["gpu"]["devices"].append({
                    "name": props.name,
                    "total_vram_gb": props.total_memory >> 30,
                    "multi_processor_count": props.multi_processor_count
                })
        return resources

    def _scan_storage(self):
        base = Path("C:/BIZRA-DATA-LAKE")
        return {
            "lake_root": str(base),
            "layers": {
                "00_INTAKE": str(base / "00_INTAKE"),
                "01_RAW": str(base / "01_RAW"),
                "02_PROCESSED": str(base / "02_PROCESSED"),
                "03_INDEXED": str(base / "03_INDEXED"),
                "04_GOLD": str(base / "04_GOLD")
            },
            "disk_usage": psutil.disk_usage('C:')._asdict()
        }

    def _scan_network(self):
        # Placeholder for p2p/distributed node discovery
        return {
            "node_id": os.environ.get("COMPUTERNAME", "unknown_node"),
            "ip_local": "127.0.0.1", # Basic placeholder
            "status": "online"
        }

    def save_registry(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
        print(f"📡 URP Registry Updated: {self.config_path}")

    def get_resource(self, resource_type):
        return self.registry.get(resource_type)

if __name__ == "__main__":
    print("📡 Activating BIZRA Unified Resource Protocol (URP)...")
    urp = BIZRA_URP()
    print("✅ Resource Pool Linked.")
    print(f"🖥️  CPU Cores: {urp.registry['compute']['cpu']['cores']}")
    if 'gpu' in urp.registry['compute']:
        print(f"🚀 GPU Active: {urp.registry['compute']['gpu']['devices'][0]['name']}")
