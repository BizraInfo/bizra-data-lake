#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NODE0 ASSET INDEXER                                        ║
║                                                                              ║
║  Index EVERYTHING on this machine — hardware, software, and data.            ║
║  This is the full inventory of BIZRA Node0 assets.                           ║
║                                                                              ║
║  The entire machine is Node0. Not just the Data Lake folder.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import subprocess
import platform
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "gold": Path(r"C:\BIZRA-DATA-LAKE\04_GOLD"),
    "inventory": Path(r"C:\BIZRA-DATA-LAKE\04_GOLD\node0_full_inventory.json"),
    "folder_index": Path(r"C:\BIZRA-DATA-LAKE\03_INDEXED\knowledge\node0_folders.jsonl"),
    "software_index": Path(r"C:\BIZRA-DATA-LAKE\03_INDEXED\knowledge\node0_software.jsonl"),
}

# BIZRA-specific folder patterns to identify
BIZRA_PATTERNS = ["BIZRA", "bizra", "genesis", "node0", "sovereign", "agentic"]

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("Node0Indexer")


# ═══════════════════════════════════════════════════════════════════════════════
# NODE0 ASSET INDEXER
# ═══════════════════════════════════════════════════════════════════════════════

class Node0AssetIndexer:
    """
    Indexes the entire Node0 machine — hardware, software, and data.
    """
    
    def __init__(self):
        self.inventory = {
            "node_id": "NODE0",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "hardware": {},
            "software": {},
            "data": {},
        }
        log.info("🔍 Node0 Asset Indexer initialized")
    
    def run_powershell(self, command: str) -> str:
        """Execute a PowerShell command and return output."""
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except Exception as e:
            log.warning(f"PowerShell command failed: {e}")
            return ""
    
    def index_hardware(self) -> Dict[str, Any]:
        """Index all hardware components."""
        log.info("📟 Indexing hardware...")
        
        hardware = {
            "platform": self._get_platform_info(),
            "cpu": self._get_cpu_info(),
            "gpu": self._get_gpu_info(),
            "memory": self._get_memory_info(),
            "storage": self._get_storage_info(),
        }
        
        self.inventory["hardware"] = hardware
        return hardware
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get system platform info."""
        output = self.run_powershell(
            "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model | ConvertTo-Json"
        )
        try:
            data = json.loads(output)
            return {
                "manufacturer": data.get("Manufacturer", "Unknown"),
                "model": data.get("Model", "Unknown"),
                "os": platform.platform(),
                "python": platform.python_version(),
            }
        except:
            return {"os": platform.platform(), "python": platform.python_version()}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU info."""
        output = self.run_powershell(
            "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json"
        )
        try:
            data = json.loads(output)
            return {
                "name": data.get("Name", "Unknown"),
                "cores": data.get("NumberOfCores", 0),
                "threads": data.get("NumberOfLogicalProcessors", 0),
            }
        except:
            return {"cores": os.cpu_count()}
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU info."""
        output = self.run_powershell(
            "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json"
        )
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return [
                {
                    "name": gpu.get("Name", "Unknown"),
                    "vram_bytes": gpu.get("AdapterRAM", 0),
                    "driver": gpu.get("DriverVersion", "Unknown"),
                }
                for gpu in data
            ]
        except:
            return []
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get RAM info."""
        output = self.run_powershell(
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
        )
        try:
            total_bytes = int(output)
            return {
                "total_gb": round(total_bytes / (1024**3), 2),
                "total_bytes": total_bytes,
            }
        except:
            return {}
    
    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage info."""
        output = self.run_powershell(
            "Get-Volume | Where-Object {$_.DriveLetter} | Select-Object DriveLetter, FileSystemLabel, Size, SizeRemaining | ConvertTo-Json"
        )
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return {
                "volumes": [
                    {
                        "letter": v.get("DriveLetter", "?"),
                        "label": v.get("FileSystemLabel", ""),
                        "size_gb": round(v.get("Size", 0) / (1024**3), 2),
                        "free_gb": round(v.get("SizeRemaining", 0) / (1024**3), 2),
                    }
                    for v in data
                ]
            }
        except:
            return {}
    
    def index_software(self) -> Dict[str, Any]:
        """Index installed software."""
        log.info("📦 Indexing software...")
        
        # Get installed programs
        output = self.run_powershell(
            "Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Where-Object {$_.DisplayName} | Select-Object DisplayName | ConvertTo-Json"
        )
        
        programs = []
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            programs = [p.get("DisplayName", "") for p in data if p.get("DisplayName")]
        except:
            pass
        
        # Get WSL distros
        wsl_output = self.run_powershell("wsl --list --quiet")
        wsl_distros = [d.strip() for d in wsl_output.split('\n') if d.strip()]
        
        # Get Python versions
        py_output = self.run_powershell("py --list")
        
        software = {
            "installed_count": len(programs),
            "programs": programs[:100],  # First 100 for brevity
            "wsl_distros": wsl_distros,
            "python_versions": py_output,
        }
        
        self.inventory["software"] = software
        return software
    
    def index_data_folders(self) -> Dict[str, Any]:
        """Index top-level folders and identify BIZRA assets."""
        log.info("📁 Indexing data folders...")
        
        root = Path("C:\\")
        folders = []
        bizra_folders = []
        
        try:
            for item in root.iterdir():
                if item.is_dir() and not item.name.startswith('$'):
                    folder_info = {
                        "name": item.name,
                        "path": str(item),
                        "is_bizra": any(p.lower() in item.name.lower() for p in BIZRA_PATTERNS),
                    }
                    folders.append(folder_info)
                    
                    if folder_info["is_bizra"]:
                        bizra_folders.append(folder_info)
        except PermissionError:
            pass
        
        data = {
            "total_top_level_folders": len(folders),
            "bizra_folder_count": len(bizra_folders),
            "bizra_folders": bizra_folders,
            "all_folders": folders,
        }
        
        self.inventory["data"] = data
        return data
    
    def index_user_folders(self) -> Dict[str, Any]:
        """Index user profile folders (Downloads, Desktop, Documents)."""
        log.info("👤 Indexing user folders...")
        
        user_profile = os.environ.get("USERPROFILE", "")
        user_data = {"user_profile": user_profile, "folders": {}}
        
        # Downloads
        downloads_path = Path(user_profile) / "Downloads"
        if downloads_path.exists():
            try:
                dl_folders = list(downloads_path.iterdir())
                dl_dirs = [f for f in dl_folders if f.is_dir()]
                dl_files = [f for f in dl_folders if f.is_file()]
                
                # Identify BIZRA-related folders in Downloads
                bizra_in_dl = [d.name for d in dl_dirs if any(p.lower() in d.name.lower() for p in BIZRA_PATTERNS)]
                
                user_data["folders"]["downloads"] = {
                    "path": str(downloads_path),
                    "top_level_folders": len(dl_dirs),
                    "top_level_files": len(dl_files),
                    "bizra_related_folders": bizra_in_dl,
                    "folder_names": [d.name for d in dl_dirs[:50]],  # First 50
                }
            except PermissionError:
                user_data["folders"]["downloads"] = {"error": "Permission denied"}
        
        # Desktop
        desktop_path = Path(user_profile) / "Desktop"
        if desktop_path.exists():
            try:
                dt_items = list(desktop_path.iterdir())
                user_data["folders"]["desktop"] = {
                    "path": str(desktop_path),
                    "items": [{"name": i.name, "is_dir": i.is_dir()} for i in dt_items],
                }
            except PermissionError:
                user_data["folders"]["desktop"] = {"error": "Permission denied"}
        
        # Documents
        documents_path = Path(user_profile) / "Documents"
        if documents_path.exists():
            try:
                doc_dirs = [d for d in documents_path.iterdir() if d.is_dir()]
                bizra_in_docs = [d.name for d in doc_dirs if any(p.lower() in d.name.lower() for p in BIZRA_PATTERNS)]
                
                user_data["folders"]["documents"] = {
                    "path": str(documents_path),
                    "top_level_folders": len(doc_dirs),
                    "bizra_related_folders": bizra_in_docs,
                }
            except PermissionError:
                user_data["folders"]["documents"] = {"error": "Permission denied"}
        
        self.inventory["user_data"] = user_data
        return user_data
    
    def save_inventory(self) -> Path:
        """Save the full inventory to JSON."""
        self.inventory["indexed_at"] = datetime.now(timezone.utc).isoformat()
        
        with open(PATHS["inventory"], 'w', encoding='utf-8') as f:
            json.dump(self.inventory, f, indent=2, ensure_ascii=False)
        
        log.info(f"💾 Inventory saved to {PATHS['inventory']}")
        return PATHS["inventory"]
    
    def export_to_knowledge_base(self):
        """Export inventory as JSONL entries for the knowledge base."""
        log.info("📤 Exporting to knowledge base...")
        
        # Export folders
        PATHS["folder_index"].parent.mkdir(parents=True, exist_ok=True)
        with open(PATHS["folder_index"], 'w', encoding='utf-8') as f:
            for folder in self.inventory.get("data", {}).get("all_folders", []):
                entry = {
                    "id": f"folder_{hashlib.blake2b(folder['path'].encode(), digest_size=16).hexdigest()[:8]}",
                    "type": "folder",
                    "name": folder["name"],
                    "path": folder["path"],
                    "is_bizra_asset": folder.get("is_bizra", False),
                    "indexed_at": self.inventory["indexed_at"],
                }
                f.write(json.dumps(entry) + '\n')
        
        log.info(f"  → {PATHS['folder_index']}")
        
        # Export software
        with open(PATHS["software_index"], 'w', encoding='utf-8') as f:
            for program in self.inventory.get("software", {}).get("programs", []):
                entry = {
                    "id": f"sw_{hashlib.blake2b(program.encode(), digest_size=16).hexdigest()[:8]}",
                    "type": "software",
                    "name": program,
                    "indexed_at": self.inventory["indexed_at"],
                }
                f.write(json.dumps(entry) + '\n')
        
        log.info(f"  → {PATHS['software_index']}")
    
    def full_scan(self) -> Dict[str, Any]:
        """Run a complete Node0 inventory scan."""
        log.info("═" * 60)
        log.info("🖥️  NODE0 FULL ASSET SCAN")
        log.info("═" * 60)
        
        self.index_hardware()
        self.index_software()
        self.index_data_folders()
        self.index_user_folders()
        self.save_inventory()
        self.export_to_knowledge_base()
        
        # Summary
        log.info("═" * 60)
        log.info("📊 SCAN COMPLETE")
        log.info("═" * 60)
        log.info(f"  Platform: {self.inventory['hardware'].get('platform', {}).get('model', 'Unknown')}")
        log.info(f"  CPU: {self.inventory['hardware'].get('cpu', {}).get('name', 'Unknown')}")
        log.info(f"  RAM: {self.inventory['hardware'].get('memory', {}).get('total_gb', 0)} GB")
        log.info(f"  Software: {self.inventory['software'].get('installed_count', 0)} programs")
        log.info(f"  WSL Distros: {len(self.inventory['software'].get('wsl_distros', []))}")
        log.info(f"  BIZRA Folders: {self.inventory['data'].get('bizra_folder_count', 0)}")
        
        # User folders
        user_data = self.inventory.get('user_data', {}).get('folders', {})
        downloads_info = user_data.get('downloads', {})
        log.info(f"  Downloads Folders: {downloads_info.get('top_level_folders', 0)}")
        log.info(f"  BIZRA in Downloads: {len(downloads_info.get('bizra_related_folders', []))}")
        log.info("═" * 60)
        
        return self.inventory


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    
    indexer = Node0AssetIndexer()
    
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║          NODE0 ASSET INDEXER — Full Machine Inventory          ║
╠════════════════════════════════════════════════════════════════╣
║  Usage:                                                         ║
║    python node0_indexer.py scan       — Full inventory scan     ║
║    python node0_indexer.py hardware   — Hardware only           ║
║    python node0_indexer.py software   — Software only           ║
║    python node0_indexer.py folders    — Folders only            ║
║    python node0_indexer.py bizra      — BIZRA folders only      ║
╚════════════════════════════════════════════════════════════════╝
        """)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "scan":
        indexer.full_scan()
        
    elif cmd == "hardware":
        hw = indexer.index_hardware()
        print(json.dumps(hw, indent=2))
        
    elif cmd == "software":
        sw = indexer.index_software()
        print(f"Installed: {sw.get('installed_count', 0)} programs")
        print(f"WSL: {sw.get('wsl_distros', [])}")
        
    elif cmd == "folders":
        data = indexer.index_data_folders()
        print(f"Total folders: {data.get('total_top_level_folders', 0)}")
        print(f"BIZRA folders: {data.get('bizra_folder_count', 0)}")
        
    elif cmd == "bizra":
        data = indexer.index_data_folders()
        print("\n🌱 BIZRA ASSET FOLDERS:")
        print("═" * 40)
        for f in data.get("bizra_folders", []):
            print(f"  {f['path']}")
        print("═" * 40)
        
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
