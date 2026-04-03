import os
import platform
import psutil
import datetime
from typing import Dict, Any, Optional
from .memory_system import CognitivePermanence
from .identity import get_identity

class OmniAwareness:
    """
    BIZRA Omni-Awareness Module - Proprioception for the Sovereign Organism.
    Maps the 'Space' (Hardware + Software + Data).
    """
    
    def __init__(self, memory: CognitivePermanence):
        self.memory = memory
        self.identity = get_identity()
        self.home_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.latest_budget = None
        # Ecosystem Territory Map (Unified Node0 View)
        self.ecosystem_nodes = {
            "UI_LAYER": r"C:\award-winner-design",
            "EVIDENCE_SCAFFOLD": r"C:\bizra_scaffold",
            "CORE_TASKMASTER": r"C:\BIZRA-TaskMaster",
            "CORE_NODE0": r"C:\BIZRA-NODE0",
            "CORE_OS": r"C:\BIZRA-OS",
            "DATA_LAKE": r"C:\BIZRA-DATA-LAKE",
            "SECURE_VAULT": r"C:\BIZRA-SECURE-VAULT",
            "GENESIS_REPAIRED": r"C:\bizra-genesis-node-repaired",
            "QUARANTINE_SCAFFOLD": r"C:\BIZRA_QUARANTINE_2025-12-30"
        }
        
    def _scan_models(self) -> Dict[str, Any]:
        """Detects the 'Dormant Giants' (Local LLMs/Embeddings)."""
        model_home = r"C:\Users\BIZRA-OS\.ollama\models\manifests\registry.ollama.ai\library"
        models = []
        if os.path.exists(model_home):
            try:
                models = os.listdir(model_home)
            except (IOError, OSError) as e:
                print(f"Failed to scan models directory: {e}")
        return {
            "count": len(models),
            "manifest": models
        }

    def perceive_hardware(self) -> Dict[str, Any]:
        """Scans the physical body (Hardware)."""
        metrics = {
            "os": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        return metrics

    def _gpu_telemetry(self) -> Optional[Dict[str, float]]:
        """Best-effort GPU/VRAM telemetry; fails closed if drivers are absent."""
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = mem.total or 1
            vram_used = mem.used / total
            gpu_util = util.gpu / 100.0
            pynvml.nvmlShutdown()
            return {
                "vram_used_ratio": round(vram_used, 4),
                "gpu_utilization_ratio": round(gpu_util, 4)
            }
        except Exception:
            return None

    def compute_cognitive_budget(self) -> Dict[str, Any]:
        """
        Real-time Cognitive Budget (0.0-1.0) derived from CPU/RAM/VRAM pressure.
        Budget = 1 - max(utilization); lower means less headroom for spawning/recursion.
        """
        cpu_util = psutil.cpu_percent(interval=0.05) / 100.0
        vm = psutil.virtual_memory()
        ram_util = vm.percent / 100.0
        gpu_metrics = self._gpu_telemetry()

        pressures = [cpu_util, ram_util]
        if gpu_metrics and "vram_used_ratio" in gpu_metrics:
            pressures.append(gpu_metrics["vram_used_ratio"])

        max_pressure = max(pressures) if pressures else 1.0
        budget = max(0.0, 1.0 - max_pressure)

        # Stability heuristics
        status = "optimal"
        if budget < 0.25:
            status = "critical"
        elif budget < 0.5:
            status = "constrained"

        snapshot = {
            "cpu_utilization": round(cpu_util, 4),
            "ram_utilization": round(ram_util, 4),
            "gpu": gpu_metrics,
            "max_pressure": round(max_pressure, 4),
            "budget_score": round(budget, 4),
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.latest_budget = snapshot
        return snapshot

    def perceive_territory(self) -> Dict[str, Any]:
        """ Maps the digital space (Codebase + Ecosystem). Hardened with Sensitivity Masking."""
        territory = []
        
        # Sensitivity Mask: Redact environment-specific absolute paths
        def redact(path):
            return path.replace("C:\\Users\\BIZRA-OS", "[REDACTED_SPACE]").replace("c:\\", "[ROOT]\\")
            
        # 1. Map Home Base (Kernel)
        territory.append({
            "node": "KERNEL_HOME",
            "path": redact(self.home_base),
            "status": "ACTIVE" if os.path.exists(self.home_base) else "MISSING"
        })
        
        # 2. Map Distributed Ecosystem
        for node_name, node_path in self.ecosystem_nodes.items():
            status = "DETECTED" if os.path.exists(node_path) else "REMOTE"
            territory.append({
                "node": node_name,
                "path": redact(node_path),
                "status": status
            })
            
        # 3. Map Capabilities (The Giants)
        models = self._scan_models()
        
        return {
            "total_nodes": len(territory),
            "total_models": models["count"],
            "model_manifest": models["manifest"],
            "map": territory,
            "ownership_certified": self.identity.is_architect("momo"),
            "owner": self.identity.architect.name
        }

    def probe_dependencies(self) -> Dict[str, bool]:
        """Lightweight dependency probe. Extend with real RPCs when available."""
        return {
            "mcp": True,  # Placeholder: replace with real MCP ping
            "ollama": os.getenv("OLLAMA_DISABLED", "0").strip() not in {"1", "true"},
            "ledger": os.path.exists("bizra_memory/ledger.json"),
        }

    def synchronize_self_model(self):
        """Integrates awareness into Semantic Memory (L4)."""
        print("[*] Synchronizing Omni-Awareness Self-Model...")
        hw = self.perceive_hardware()
        space = self.perceive_territory()
        budget = self.compute_cognitive_budget()
        
        # Dependency Health Probe
        self.dependency_health = self.probe_dependencies()
        
        # Inject into L4: Semantic Knowledge
        self.memory.add_semantic_fact("Self", "HardwareProfile", hw)
        self.memory.add_semantic_fact("Self", "DigitalTerritory", space)
        self.memory.add_semantic_fact("Self", "CognitiveBudget", budget)
        
        print(f"[+] Self-Model Synced. Unified Territory: {space['total_nodes']} nodes recognized.")
        print(f"[+] Ownership: {space['owner']} (Certified: {space['ownership_certified']})")
        print(f"[+] Cognitive Budget: {budget['budget_score']:.3f} ({budget['status']})")
        return {"hardware": hw, "territory": space, "budget": budget, "dependencies": self.dependency_health, "sovereignty": self.identity.to_dict()}

if __name__ == "__main__":
    mem = CognitivePermanence()
    awareness = OmniAwareness(mem)
    report = awareness.synchronize_self_model()
    print(f"[+] CPU Cores: {report['hardware']['cpu_count']} | RAM: {report['hardware']['ram_total_gb']}GB")
    print(f"[+] Territory Map: {len(report['territory']['map'])} nodes identified.")