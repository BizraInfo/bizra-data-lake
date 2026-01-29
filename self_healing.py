# BIZRA Self-Healing Protocol v1.0
# Monitors system health and triggers recovery for missing artifacts
# Part of Phase 1: Foundation Deploy

import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    filename='C:/BIZRA-DATA-LAKE/RECOVERY-REPORT.md',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HealingAgent:
    def __init__(self):
        self.root = Path("C:/BIZRA-DATA-LAKE")
        self.critical_files = [
            self.root / "04_GOLD/documents.parquet",
            self.root / "03_INDEXED/chat_history/graph.json", # Usually point to latest
            self.root / "bizra_config.py"
        ]

    def check_health(self):
        print("🩺 Running BIZRA Foundation Health Check...")
        issues = []
        for cf in self.critical_files:
            if not cf.exists():
                message = f"🚨 MISSING CRITICAL FILE: {cf.name} at {cf}"
                print(message)
                logging.error(message)
                issues.append(cf.name)
            else:
                print(f"✅ {cf.name} is intact.")
        
        if not issues:
            print("✨ System Health: 100%. No healing required.")
            return True
        else:
            self.attempt_repair(issues)
            return False

    def attempt_repair(self, issues):
        print(f"🛠️  Attempting repair for {len(issues)} issues...")
        
        if "documents.parquet" in issues:
            print("🔄 Triggering Corpus Canonization...")
            try:
                subprocess.run(["python", str(self.root / "corpus_manager.py")], check=True)
                logging.info("✅ Repaired: documents.parquet")
            except Exception as e:
                logging.error(f"❌ Repair Failed: documents.parquet - {e}")

        if "graph.json" in issues:
            print("🔄 Triggering Hypergraph Synthesis...")
            try:
                subprocess.run(["python", str(self.root / "build-hypergraph.py")], check=True)
                logging.info("✅ Repaired: graph.json")
            except Exception as e:
                logging.error(f"❌ Repair Failed: graph.json - {e}")

if __name__ == "__main__":
    agent = HealingAgent()
    agent.check_health()
