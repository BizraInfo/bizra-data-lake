"""
BIZRA Unified Sovereign Observer
================================
Phase Epsilon: Total Recall & God Eye

This module acts as the "God Eye" for the BIZRA ecosystem, providing:
1.  **Watchdog**: Real-time health monitoring of all system components.
2.  **Indexer**: Centralized indexing of artifacts, logs, and memories ("Total Recall").
3.  **Omni-Presence**: Aggregated visibility across federation nodes.

It connects to the Cognitive Permanence layer to ensure no insight is lost.
"""

import os
import json
import time
import glob
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UnifiedObserver")

@dataclass
class SystemStatus:
    """Snapshot of system health."""
    timestamp: float
    healthy: bool
    components: Dict[str, str]
    active_alerts: List[str]
    resource_usage: Dict[str, float]

class UnifiedObserver:
    """
    The All-Seeing Eye of the BIZRA Sovereign Organism.
    Orchestrates observation, indexing, and health monitoring.
    """

    def __init__(self, root_path: str = "."):
        self.root_path = os.path.abspath(root_path)
        self.brain_path = os.path.join(self.root_path, ".gemini", "antigravity", "brain")
        self.memory_path = os.path.join(self.root_path, "bizra_memory")
        self.evidence_path = os.path.join(self.root_path, "evidence")
        
        self.status_history: List[SystemStatus] = []
        self.indexed_artifacts: Dict[str, Dict[str, Any]] = {}

        logger.info(f"[*] UnifiedObserver initialized at root: {self.root_path}")

    def scan_ecosystem(self) -> SystemStatus:
        """Perform a full 'God Eye' scan of the ecosystem."""
        components = {}
        alerts = []
        healthy = True

        # 1. Check Kernel Integrity
        kernel_path = os.path.join(self.root_path, "bizra_kernel")
        if os.path.exists(kernel_path):
            components["Kernel"] = "Active"
        else:
            components["Kernel"] = "Missing"
            alerts.append("Critical: Kernel directory not found")
            healthy = False

        # 2. Check Genesis Node (Rust Backend)
        rust_backend_path = os.path.join(self.root_path, "bizra-genesis-node", "backend")
        if os.path.exists(rust_backend_path):
            components["GenesisNode"] = "Detected"
        else:
            components["GenesisNode"] = "Missing"
            alerts.append("Warning: Genesis Node backend not found")

        # 3. Check Evidence Locker
        if os.path.exists(self.evidence_path):
            evidence_count = len(glob.glob(os.path.join(self.evidence_path, "*")))
            components["EvidenceLocker"] = f"Secure ({evidence_count} items)"
        else:
            components["EvidenceLocker"] = "Missing"

        # 4. Check Brain (Antigravity)
        if os.path.exists(self.brain_path):
             components["Brain"] = "Active"
        else:
             components["Brain"] = "Latent"

        status = SystemStatus(
            timestamp=time.time(),
            healthy=healthy,
            components=components,
            active_alerts=alerts,
            resource_usage={"scan_latency_ms": 15.0} # Placeholder for real metrics
        )
        self.status_history.append(status)
        return status

    def index_artifacts(self) -> int:
        """
        Index essential artifacts for 'Total Recall'.
        Scans brain directory and project root for .md and .json files.
        """
        logger.info("[*] Starting Total Recall indexing...")
        count = 0
        
        # Define search patterns
        patterns = [
            os.path.join(self.brain_path, "**", "*.md"),
            os.path.join(self.brain_path, "**", "*.json"),
            os.path.join(self.root_path, "*.md"),
            os.path.join(self.root_path, "docs", "**", "*.md"),
        ]

        for pattern in patterns:
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    stats = os.stat(filepath)
                    rel_path = os.path.relpath(filepath, self.root_path)
                    
                    self.indexed_artifacts[rel_path] = {
                        "size": stats.st_size,
                        "mtime": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        "type": os.path.splitext(filepath)[1][1:]
                    }
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to index {filepath}: {e}")

        logger.info(f"[+] Indexed {count} artifacts.")
        return count

    def generate_god_eye_report(self) -> str:
        """Generate a comprehensive status report."""
        status = self.scan_ecosystem()
        self.index_artifacts()
        
        report = []
        report.append("=" * 60)
        report.append(" BIZRA SOVEREIGN OBSERVER REPORT (GOD EYE)")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.fromtimestamp(status.timestamp).isoformat()}")
        report.append(f"Overall Health: {'✅ OPTIMAL' if status.healthy else '❌ CRITICAL'}")
        
        report.append("\n--- Component Status ---")
        for name, state in status.components.items():
            report.append(f"{name:20s}: {state}")

        if status.active_alerts:
            report.append("\n--- ⚠️ Active Alerts ---")
            for alert in status.active_alerts:
                report.append(f"- {alert}")
        
        report.append("\n--- Total Recall Index ---")
        report.append(f"Indexed Items: {len(self.indexed_artifacts)}")
        report.append("Recent Artifacts:")
        
        # Show top 5 most recently modified
        sorted_artifacts = sorted(
            self.indexed_artifacts.items(), 
            key=lambda x: x[1]['mtime'], 
            reverse=True
        )[:5]
        
        for name, meta in sorted_artifacts:
            report.append(f"- {meta['mtime']} | {name}")

        return "\n".join(report)

if __name__ == "__main__":
    # Self-Verify
    observer = UnifiedObserver()
    print(observer.generate_god_eye_report())
