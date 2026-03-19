"""
BIZRA SOVEREIGN ACTUATOR (v1.0)
"The Autonomous Watchman"

Mission:
Continuously monitors the Sovereign Domain for new high-value artifacts.
When a new "Golden Gem" is detected, it triggers the PRIME reasoning loop
and records the impact.

Aligned with:
- SC §4.2: Ops Agents (Self-Diagnosis, Auto-Debug)
- SC §7: Agentic Systems

This runs as a background daemon (or scheduled task).
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from bizra_prime import AgentRole, BizraPrime
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import BIZRA systems
from bizra_config import DATA_LAKE_ROOT, GOLD_PATH

# Configuration
WATCH_PATHS = [
    DATA_LAKE_ROOT / "00_INTAKE",
    DATA_LAKE_ROOT / "01_RAW" / "external_links",
]
ACTUATOR_LOG = GOLD_PATH / "actuator_events.jsonl"
HIGH_VALUE_EXTENSIONS = {".md", ".txt", ".json", ".py", ".rs", ".ts"}


class SovereignEventHandler(FileSystemEventHandler):
    """Handles file system events for high-value artifacts."""

    def __init__(self, prime: BizraPrime):
        super().__init__()
        self.prime = prime
        self.processed_hashes = set()
        self._load_processed()

    def _load_processed(self):
        if ACTUATOR_LOG.exists():
            with open(ACTUATOR_LOG, "r") as f:
                for line in f:
                    event = json.loads(line)
                    self.processed_hashes.add(event.get("file_hash", ""))

    def _log_event(self, event_type, path, result=None):
        file_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "path": str(path),
            "file_hash": file_hash,
            "result_summary": str(result)[:100] if result else None,
        }
        with open(ACTUATOR_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.processed_hashes.add(file_hash)
        return file_hash

    def _is_high_value(self, path):
        p = Path(path)
        return p.suffix.lower() in HIGH_VALUE_EXTENSIONS

    def _file_hash(self, path):
        return hashlib.sha256(str(path).encode()).hexdigest()[:16]

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        file_hash = self._file_hash(path)

        if file_hash in self.processed_hashes:
            return

        if not self._is_high_value(path):
            return

        print(f"\n🔔 [ACTUATOR] New Artifact Detected: {path.name}")

        # Dispatch to RESEARCHER for initial analysis
        research_result = self.prime.dispatch(
            AgentRole.RESEARCHER, f"Analyze new artifact: {path.name}"
        )

        # Log the event
        self._log_event("NEW_ARTIFACT", path, research_result)

        print(f"   ✅ Artifact indexed. Hash: {file_hash}")

    def on_modified(self, event):
        # For now, we only react to creations
        pass


def run_actuator(watch_once=False):
    """
    Main Actuator Loop.
    watch_once=True for testing (single scan then exit).
    """
    print("═" * 70)
    print("   🛡️  BIZRA SOVEREIGN ACTUATOR - ONLINE")
    print("═" * 70)

    # Initialize PRIME (this loads all knowledge)
    # Suppressing TensorFlow warnings for cleaner output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    prime = BizraPrime()

    handler = SovereignEventHandler(prime)
    observer = Observer()

    for watch_path in WATCH_PATHS:
        if watch_path.exists():
            observer.schedule(handler, str(watch_path), recursive=True)
            print(f"   👁️  Watching: {watch_path}")
        else:
            print(f"   ⚠️  Path not found: {watch_path}")

    if watch_once:
        # For testing: just do a quick scan
        print("\n   🔍 Performing initial scan...")
        for watch_path in WATCH_PATHS:
            if watch_path.exists():
                for root, dirs, files in os.walk(watch_path):
                    # Limit depth and count for demo
                    if len(files) > 0:
                        for f in files[:3]:  # Sample first 3
                            p = Path(root) / f
                            if (
                                handler._is_high_value(p)
                                and handler._file_hash(p)
                                not in handler.processed_hashes
                            ):
                                print(f"   📄 Found: {p.name}")
                    break  # Only top level
        print("\n   ✅ Initial scan complete.")
        return

    observer.start()
    print("\n   ⏳ Actuator running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(10)
            # Periodic health check
            diagnosis = prime._self_diagnose()
            if diagnosis.get("snr", 0) < 0.5:
                print("   ⚠️ [ACTUATOR] Low SNR detected. Triggering self-healing...")
                prime.dispatch(AgentRole.AUTO_DEBUG, "Investigate low SNR")
    except KeyboardInterrupt:
        observer.stop()
        print("\n   🛑 Actuator stopped.")

    observer.join()


if __name__ == "__main__":
    import sys

    # Run once for demo, or continuous with --daemon
    if "--daemon" in sys.argv:
        run_actuator(watch_once=False)
    else:
        run_actuator(watch_once=True)
