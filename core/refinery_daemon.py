#!/usr/bin/env python3
"""
BIZRA Refinery Daemon - Continuous Ingestion Service
======================================================
Fixes F-PERF-002: Ingestion Latency

Converts the batch-only refinery into a continuous background
service with:
- File system watching for real-time ingestion
- Configurable throughput limiting (10MB/sec target)
- Queue-based processing with prioritization
- Health monitoring and metrics
- Docker-ready architecture

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    REFINERY DAEMON                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
    │   │   WATCHER    │───▶│    QUEUE     │───▶│   WORKER     │  │
    │   │  (inotify)   │    │  (Priority)  │    │  (Refinery)  │  │
    │   └──────────────┘    └──────────────┘    └──────────────┘  │
    │          │                   │                   │          │
    │          │                   │                   │          │
    │          ▼                   ▼                   ▼          │
    │   ┌──────────────────────────────────────────────────────┐  │
    │   │              METRICS / HEALTH                         │  │
    │   │   Files/sec: 12.5   Bytes/sec: 8.2MB   Queue: 42     │  │
    │   └──────────────────────────────────────────────────────┘  │
    │                              │                               │
    │                              ▼                               │
    │   ┌──────────────────────────────────────────────────────┐  │
    │   │                    LEDGER OUTPUT                      │  │
    │   │              BIZRA_KNOWLEDGE_LEDGER.jsonl             │  │
    │   └──────────────────────────────────────────────────────┘  │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

Endpoint: /health, /metrics, /queue
"""

import hashlib
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("refinery.daemon")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DaemonConfig:
    """Refinery Daemon configuration."""
    watch_paths: List[Path] = field(default_factory=lambda: [Path("bizra_data_vault/roots")])
    ledger_path: Path = Path("BIZRA_KNOWLEDGE_LEDGER.jsonl")
    manifest_path: Path = Path("BIZRA_KNOWLEDGE_MANIFEST.json")
    genesis_path: Path = Path("BIZRA_GENESIS_BLOCK_0.json")
    
    # Throughput control
    target_bytes_per_sec: int = 10 * 1024 * 1024  # 10MB/sec
    batch_size: int = 50  # Files per batch
    batch_interval_sec: float = 1.0  # Seconds between batches
    
    # Queue limits
    max_queue_size: int = 10000
    priority_extensions: Set[str] = field(default_factory=lambda: {".py", ".rs", ".md", ".json"})
    
    # Health server
    health_port: int = 8081
    
    # Directories to skip
    skip_dirs: Set[str] = field(default_factory=lambda: {
        ".git", "__pycache__", "node_modules", "target", "dist", "build", ".next"
    })


# Impact multipliers (from bizra_refinery.py)
IMPACT_MULTIPLIERS: Dict[str, float] = {
    ".rs": 50.0, ".py": 40.0, ".js": 35.0, ".ts": 35.0, ".go": 40.0,
    ".pdf": 25.0, ".md": 15.0, ".json": 10.0, ".xml": 10.0,
    ".csv": 5.0, ".txt": 2.0, ".html": 2.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DaemonMetrics:
    """Real-time daemon metrics."""
    files_processed: int = 0
    bytes_processed: int = 0
    total_value: float = 0.0
    errors: int = 0
    queue_size: int = 0
    started_at: str = ""
    last_file_at: str = ""
    
    # Rolling window for rate calculation
    _window_files: int = 0
    _window_bytes: int = 0
    _window_start: float = 0.0
    
    def record_file(self, size_bytes: int, value: float) -> None:
        """Record a processed file."""
        now = time.monotonic()
        
        # Reset window every 60 seconds
        if now - self._window_start > 60:
            self._window_files = 0
            self._window_bytes = 0
            self._window_start = now
        
        self.files_processed += 1
        self.bytes_processed += size_bytes
        self.total_value += value
        self._window_files += 1
        self._window_bytes += size_bytes
        self.last_file_at = datetime.now(timezone.utc).isoformat()
    
    def record_error(self) -> None:
        """Record a processing error."""
        self.errors += 1
    
    @property
    def files_per_sec(self) -> float:
        """Calculate files per second (rolling window)."""
        elapsed = time.monotonic() - self._window_start
        if elapsed < 1:
            return 0.0
        return self._window_files / elapsed
    
    @property
    def bytes_per_sec(self) -> float:
        """Calculate bytes per second (rolling window)."""
        elapsed = time.monotonic() - self._window_start
        if elapsed < 1:
            return 0.0
        return self._window_bytes / elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics."""
        return {
            "files_processed": self.files_processed,
            "bytes_processed": self.bytes_processed,
            "bytes_processed_mb": round(self.bytes_processed / (1024 * 1024), 2),
            "total_value": round(self.total_value, 4),
            "errors": self.errors,
            "queue_size": self.queue_size,
            "files_per_sec": round(self.files_per_sec, 2),
            "bytes_per_sec_mb": round(self.bytes_per_sec / (1024 * 1024), 2),
            "started_at": self.started_at,
            "last_file_at": self.last_file_at,
            "uptime_sec": round(time.monotonic() - self._window_start, 0)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FILE QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(order=True)
class FileItem:
    """Queued file with priority."""
    priority: int  # Lower = higher priority
    path: Path = field(compare=False)
    event_type: str = field(compare=False)  # "created", "modified", "initial"
    timestamp: float = field(compare=False)


class FileQueue:
    """Priority queue for files to process."""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self._queue: queue.PriorityQueue[FileItem] = queue.PriorityQueue(
            maxsize=config.max_queue_size
        )
        self._seen: Set[str] = set()  # Dedup
        self._lock = threading.Lock()
    
    def _priority_for(self, path: Path) -> int:
        """Calculate priority (lower = process first)."""
        ext = path.suffix.lower()
        if ext in self.config.priority_extensions:
            return 0  # High priority
        if ext in IMPACT_MULTIPLIERS:
            return 1  # Medium priority
        return 2  # Low priority
    
    def enqueue(self, path: Path, event_type: str = "modified") -> bool:
        """Add file to queue (returns False if duplicate or full)."""
        path_str = str(path.absolute())
        
        with self._lock:
            if path_str in self._seen:
                return False  # Already queued
            
            try:
                item = FileItem(
                    priority=self._priority_for(path),
                    path=path,
                    event_type=event_type,
                    timestamp=time.monotonic()
                )
                self._queue.put_nowait(item)
                self._seen.add(path_str)
                return True
            except queue.Full:
                logger.warning("Queue full, dropping file")
                return False
    
    def dequeue_batch(self, max_size: int) -> List[FileItem]:
        """Get batch of files to process."""
        batch = []
        while len(batch) < max_size:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                
                with self._lock:
                    self._seen.discard(str(item.path.absolute()))
            except queue.Empty:
                break
        return batch
    
    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()


# ═══════════════════════════════════════════════════════════════════════════════
# FILE WATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class FileWatcher:
    """Watch directories for file changes."""
    
    def __init__(self, config: DaemonConfig, file_queue: FileQueue):
        self.config = config
        self.queue = file_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_interval = 2.0  # Polling fallback
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        parts = path.parts
        return any(part in self.config.skip_dirs for part in parts)
    
    def _is_valid_file(self, path: Path) -> bool:
        """Check if file should be processed."""
        if not path.is_file():
            return False
        if path.name.startswith(".") or path.name.startswith("__"):
            return False
        if self._should_skip(path):
            return False
        return True
    
    def _scan_directory(self, root: Path, event_type: str = "initial") -> int:
        """Recursively scan directory and queue files."""
        count = 0
        try:
            for item in root.iterdir():
                if item.is_dir():
                    if item.name not in self.config.skip_dirs:
                        count += self._scan_directory(item, event_type)
                elif self._is_valid_file(item):
                    if self.queue.enqueue(item, event_type):
                        count += 1
        except PermissionError:
            pass
        except Exception as e:
            logger.warning(f"Scan error in {root}: {e}")
        return count
    
    def _poll_loop(self) -> None:
        """Polling-based file watching (fallback)."""
        last_mtimes: Dict[str, float] = {}
        
        while self._running:
            for watch_path in self.config.watch_paths:
                if not watch_path.exists():
                    continue
                
                try:
                    for root, dirs, files in os.walk(watch_path):
                        # Skip unwanted directories
                        dirs[:] = [d for d in dirs if d not in self.config.skip_dirs]
                        
                        for name in files:
                            if name.startswith(".") or name.startswith("__"):
                                continue
                            
                            path = Path(root) / name
                            path_str = str(path)
                            
                            try:
                                mtime = path.stat().st_mtime
                                if path_str not in last_mtimes:
                                    last_mtimes[path_str] = mtime
                                elif mtime > last_mtimes[path_str]:
                                    last_mtimes[path_str] = mtime
                                    self.queue.enqueue(path, "modified")
                            except (OSError, IOError):
                                pass
                except Exception as e:
                    logger.warning(f"Poll error: {e}")
            
            time.sleep(self._poll_interval)
    
    def start(self) -> None:
        """Start the file watcher."""
        if self._running:
            return
        
        self._running = True
        
        # Initial scan
        total = 0
        for watch_path in self.config.watch_paths:
            if watch_path.exists():
                count = self._scan_directory(watch_path, "initial")
                total += count
                logger.info(f"Initial scan: {count} files from {watch_path}")
        logger.info(f"Total files queued: {total}")
        
        # Start polling thread
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("File watcher started (polling mode)")
    
    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)


# ═══════════════════════════════════════════════════════════════════════════════
# REFINERY WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class RefineryWorker:
    """Process files from queue."""
    
    def __init__(self, config: DaemonConfig, file_queue: FileQueue, metrics: DaemonMetrics):
        self.config = config
        self.queue = file_queue
        self.metrics = metrics
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ledger_lock = threading.Lock()
        self._chain = hashlib.sha256()
        
        # Load genesis hash
        self._load_genesis()
    
    def _load_genesis(self) -> None:
        """Load genesis hash for chain continuity."""
        try:
            with open(self.config.genesis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            genesis_hash = data.get("genesis_hash", "")
            if genesis_hash:
                self._chain.update(genesis_hash.encode("utf-8"))
                self._chain.update(b"\0")
                logger.info(f"Genesis hash loaded: {genesis_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Genesis load error: {e}")
    
    def _hash_file(self, path: Path) -> str:
        """Calculate file content hash."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
        except Exception:
            h.update(str(path).encode())
        return h.hexdigest()
    
    def _calculate_impact(self, path: Path, size_mb: float) -> float:
        """Calculate impact value."""
        ext = path.suffix.lower()
        multiplier = IMPACT_MULTIPLIERS.get(ext, 1.0)
        return round(size_mb * multiplier, 4)
    
    def _process_file(self, item: FileItem) -> Optional[Dict[str, Any]]:
        """Process a single file."""
        path = item.path
        
        try:
            if not path.exists():
                return None
            
            st = path.stat()
            size_bytes = st.st_size
            size_mb = size_bytes / (1024 * 1024)
            
            file_hash = self._hash_file(path)
            impact = self._calculate_impact(path, size_mb)
            
            # Update chain
            with self._ledger_lock:
                self._chain.update(file_hash.encode("utf-8"))
                self._chain.update(b"\0")
            
            record = {
                "filename": path.name,
                "path": str(path),
                "hash": file_hash,
                "hash_kind": "content_sha256",
                "size_mb": round(size_mb, 4),
                "impact_value": impact,
                "type": "SOVEREIGN_ASSET",
                "event": item.event_type,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.metrics.record_file(size_bytes, impact)
            return record
            
        except Exception as e:
            logger.warning(f"Process error {path}: {e}")
            self.metrics.record_error()
            return None
    
    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        """Append records to ledger."""
        if not records:
            return
        
        with self._ledger_lock:
            try:
                with open(self.config.ledger_path, "a", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Ledger write error: {e}")
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            # Get batch
            batch = self.queue.dequeue_batch(self.config.batch_size)
            
            if not batch:
                time.sleep(0.1)
                continue
            
            # Process batch with throughput limiting
            records = []
            batch_bytes = 0
            batch_start = time.monotonic()
            
            for item in batch:
                record = self._process_file(item)
                if record:
                    records.append(record)
                    batch_bytes += int(record.get("size_mb", 0) * 1024 * 1024)
                
                # Throughput limiting
                elapsed = time.monotonic() - batch_start
                target_time = batch_bytes / self.config.target_bytes_per_sec
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            # Write batch
            self._write_records(records)
            
            # Update queue size metric
            self.metrics.queue_size = self.queue.size
            
            # Batch interval
            time.sleep(self.config.batch_interval_sec)
    
    def start(self) -> None:
        """Start the worker."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("Refinery worker started")
    
    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
    
    @property
    def chain_hash(self) -> str:
        """Current chain hash."""
        return self._chain.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health/metrics endpoints."""
    
    daemon: 'RefineryDaemon' = None  # Set by server
    
    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queue_size": HealthHandler.daemon.metrics.queue_size,
                "chain_hash": HealthHandler.daemon.worker.chain_hash[:16] + "..."
            })
        elif self.path == "/metrics":
            self._send_json(HealthHandler.daemon.metrics.to_dict())
        elif self.path == "/queue":
            self._send_json({
                "size": HealthHandler.daemon.queue.size,
                "max_size": HealthHandler.daemon.config.max_queue_size
            })
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


# ═══════════════════════════════════════════════════════════════════════════════
# DAEMON
# ═══════════════════════════════════════════════════════════════════════════════

class RefineryDaemon:
    """Main daemon coordinator."""
    
    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()
        self.metrics = DaemonMetrics()
        self.queue = FileQueue(self.config)
        self.watcher = FileWatcher(self.config, self.queue)
        self.worker = RefineryWorker(self.config, self.queue, self.metrics)
        self._http_server: Optional[HTTPServer] = None
        self._running = False
    
    def start(self) -> None:
        """Start all daemon components."""
        if self._running:
            return
        
        self._running = True
        self.metrics.started_at = datetime.now(timezone.utc).isoformat()
        self.metrics._window_start = time.monotonic()
        
        logger.info("=" * 60)
        logger.info("  BIZRA REFINERY DAEMON")
        logger.info("=" * 60)
        logger.info(f"  Watch paths: {[str(p) for p in self.config.watch_paths]}")
        logger.info(f"  Target throughput: {self.config.target_bytes_per_sec / (1024*1024):.1f} MB/sec")
        logger.info(f"  Health port: {self.config.health_port}")
        logger.info("=" * 60)
        
        # Start components
        self.watcher.start()
        self.worker.start()
        
        # Start health server
        try:
            HealthHandler.daemon = self
            self._http_server = HTTPServer(("0.0.0.0", self.config.health_port), HealthHandler)
            threading.Thread(target=self._http_server.serve_forever, daemon=True).start()
            logger.info(f"Health server on http://0.0.0.0:{self.config.health_port}")
        except Exception as e:
            logger.warning(f"Health server failed: {e}")
        
        logger.info("Daemon started successfully")
    
    def stop(self) -> None:
        """Stop all daemon components."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping daemon...")
        
        self.watcher.stop()
        self.worker.stop()
        
        if self._http_server:
            self._http_server.shutdown()
        
        logger.info("Daemon stopped")
        logger.info(f"Final metrics: {json.dumps(self.metrics.to_dict(), indent=2)}")
    
    def run_forever(self) -> None:
        """Run daemon until interrupted."""
        self.start()
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep main thread alive
        while self._running:
            time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the Refinery Daemon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Refinery Daemon")
    parser.add_argument("--watch", type=str, nargs="+", default=["bizra_data_vault/roots"],
                        help="Directories to watch")
    parser.add_argument("--ledger", type=str, default="BIZRA_KNOWLEDGE_LEDGER.jsonl",
                        help="Ledger output file")
    parser.add_argument("--throughput", type=float, default=10.0,
                        help="Target throughput in MB/sec")
    parser.add_argument("--port", type=int, default=8081,
                        help="Health server port")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Files per batch")
    
    args = parser.parse_args()
    
    config = DaemonConfig(
        watch_paths=[Path(p) for p in args.watch],
        ledger_path=Path(args.ledger),
        target_bytes_per_sec=int(args.throughput * 1024 * 1024),
        health_port=args.port,
        batch_size=args.batch_size
    )
    
    daemon = RefineryDaemon(config)
    daemon.run_forever()


if __name__ == "__main__":
    main()
