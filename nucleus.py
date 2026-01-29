#!/usr/bin/env python3
"""
BIZRA NUCLEUS
═══════════════════════════════════════════════════════════════════════════════

The Unified Entry Point — The Center That Holds

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIZRA NUCLEUS                                      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  CLI Layer                                                           │   │
│   │  nucleus start | stop | status | health | infer | accumulator       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  Orchestration Layer            ▼                                   │   │
│   │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐      │   │
│   │  │ Boot Sequence  │  │ Health Monitor   │  │ Graceful       │      │   │
│   │  │ Manager        │  │ (Continuous)     │  │ Shutdown       │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  Component Layer                ▼                                   │   │
│   │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐      │   │
│   │  │ FLYWHEEL       │  │ ACCUMULATOR      │  │ PRIME          │      │   │
│   │  │ (Inference)    │  │ (Impact)         │  │ (Agentic)      │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  Infrastructure Layer           ▼                                   │   │
│   │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐      │   │
│   │  │ LLM Backend    │  │ State Persist    │  │ Event Bus      │      │   │
│   │  │ (Ollama/LMS)   │  │ (JSON/SQLite)    │  │ (Async Queue)  │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Giants Protocol:
- Al-Khwarizmi: Algorithmic boot sequence
- Ibn Sina: Diagnostic health monitoring
- Al-Jazari: Engineering precision in orchestration
- Ibn Khaldun: Civilizational center that holds

Usage:
    python nucleus.py start       # Boot entire stack
    python nucleus.py stop        # Graceful shutdown
    python nucleus.py status      # Component status
    python nucleus.py health      # Deep health check
    python nucleus.py infer       # Quick inference test
    python nucleus.py shell       # Interactive REPL
"""

import os
import sys
import json
import time
import signal
import asyncio
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import subprocess
import socket

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUCLEUS_VERSION = "1.0.0"
NUCLEUS_STATE_PATH = Path(os.getenv("BIZRA_NUCLEUS_STATE", "/var/lib/bizra/nucleus_state.json"))
NUCLEUS_LOG_PATH = Path(os.getenv("BIZRA_NUCLEUS_LOG", "/var/log/bizra/nucleus.log"))

# Component URLs (configurable)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://192.168.56.1:1234")
FLYWHEEL_URL = os.getenv("FLYWHEEL_URL", "http://localhost:8100")

# Boot timeouts
BOOT_TIMEOUT_SECONDS = 120
COMPONENT_CHECK_INTERVAL = 2

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class NucleusState(str, Enum):
    """Nucleus lifecycle states."""
    OFFLINE = "offline"
    BOOTING = "booting"
    ONLINE = "online"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"


class ComponentState(str, Enum):
    """Individual component states."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ComponentStatus:
    """Status of a single component."""
    name: str
    state: ComponentState
    url: Optional[str] = None
    version: Optional[str] = None
    latency_ms: Optional[float] = None
    last_check: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NucleusStatus:
    """Overall nucleus status."""
    state: NucleusState
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentStatus]
    total_bloom: float = 0.0
    inference_available: bool = False
    last_health_check: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def log(message: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    
    # Also write to log file
    try:
        NUCLEUS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NUCLEUS_LOG_PATH, 'a') as f:
            f.write(line + "\n")
    except:
        pass


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def check_url(url: str, timeout: float = 5.0) -> tuple[bool, float, Optional[str]]:
    """Check if URL is reachable. Returns (success, latency_ms, error)."""
    import urllib.request
    import urllib.error
    
    start = time.time()
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = (time.time() - start) * 1000
            return True, latency, None
    except urllib.error.URLError as e:
        return False, 0, str(e)
    except Exception as e:
        return False, 0, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT CHECKERS
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentChecker:
    """Check health of individual components."""
    
    @staticmethod
    def check_ollama() -> ComponentStatus:
        """Check Ollama LLM backend."""
        url = OLLAMA_URL
        success, latency, error = check_url(f"{url}/api/tags")
        
        if success:
            # Try to get models list
            try:
                import urllib.request
                req = urllib.request.Request(f"{url}/api/tags")
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read())
                    models = [m.get("name", "unknown") for m in data.get("models", [])]
                    return ComponentStatus(
                        name="ollama",
                        state=ComponentState.HEALTHY,
                        url=url,
                        latency_ms=latency,
                        last_check=datetime.now(timezone.utc).isoformat(),
                        metadata={"models": models, "model_count": len(models)},
                    )
            except Exception as e:
                return ComponentStatus(
                    name="ollama",
                    state=ComponentState.DEGRADED,
                    url=url,
                    latency_ms=latency,
                    last_check=datetime.now(timezone.utc).isoformat(),
                    error=str(e),
                )
        else:
            return ComponentStatus(
                name="ollama",
                state=ComponentState.FAILED,
                url=url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=error,
            )
    
    @staticmethod
    def check_lmstudio() -> ComponentStatus:
        """Check LM Studio backend."""
        url = LMSTUDIO_URL
        success, latency, error = check_url(f"{url}/v1/models")
        
        if success:
            try:
                import urllib.request
                req = urllib.request.Request(f"{url}/v1/models")
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read())
                    models = [m.get("id", "unknown") for m in data.get("data", [])]
                    return ComponentStatus(
                        name="lmstudio",
                        state=ComponentState.HEALTHY,
                        url=url,
                        latency_ms=latency,
                        last_check=datetime.now(timezone.utc).isoformat(),
                        metadata={"models": models, "model_count": len(models)},
                    )
            except Exception as e:
                return ComponentStatus(
                    name="lmstudio",
                    state=ComponentState.DEGRADED,
                    url=url,
                    latency_ms=latency,
                    last_check=datetime.now(timezone.utc).isoformat(),
                    error=str(e),
                )
        else:
            return ComponentStatus(
                name="lmstudio",
                state=ComponentState.STOPPED,  # Optional component
                url=url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=error,
            )
    
    @staticmethod
    def check_flywheel() -> ComponentStatus:
        """Check Flywheel API."""
        url = FLYWHEEL_URL
        success, latency, error = check_url(f"{url}/health")
        
        if success:
            return ComponentStatus(
                name="flywheel",
                state=ComponentState.HEALTHY,
                url=url,
                latency_ms=latency,
                last_check=datetime.now(timezone.utc).isoformat(),
            )
        else:
            return ComponentStatus(
                name="flywheel",
                state=ComponentState.FAILED,
                url=url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=error,
            )
    
    @staticmethod
    def check_accumulator() -> ComponentStatus:
        """Check Accumulator (in-process)."""
        try:
            from accumulator import get_accumulator
            acc = get_accumulator()
            status = acc.status()
            return ComponentStatus(
                name="accumulator",
                state=ComponentState.HEALTHY,
                last_check=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "state": status["state"],
                    "total_bloom": status["total_bloom"],
                    "contributors": status["total_contributors"],
                    "poi_count": status["poi_attestations"],
                },
            )
        except ImportError:
            return ComponentStatus(
                name="accumulator",
                state=ComponentState.STOPPED,
                last_check=datetime.now(timezone.utc).isoformat(),
                error="Module not available",
            )
        except Exception as e:
            return ComponentStatus(
                name="accumulator",
                state=ComponentState.FAILED,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )
    
    @staticmethod
    def check_prime() -> ComponentStatus:
        """Check BIZRA Prime (agentic core)."""
        try:
            from bizra_prime import BizraPrime
            # Don't instantiate, just check module exists
            return ComponentStatus(
                name="prime",
                state=ComponentState.HEALTHY,
                last_check=datetime.now(timezone.utc).isoformat(),
                metadata={"module": "bizra_prime", "status": "available"},
            )
        except ImportError:
            return ComponentStatus(
                name="prime",
                state=ComponentState.STOPPED,
                last_check=datetime.now(timezone.utc).isoformat(),
                error="Module not available",
            )
        except Exception as e:
            return ComponentStatus(
                name="prime",
                state=ComponentState.DEGRADED,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# BOOT SEQUENCE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class BootSequence:
    """
    Al-Khwarizmi's Algorithmic Boot Sequence.
    
    Order matters. Dependencies respected.
    """
    
    # Boot order (dependency-aware)
    BOOT_ORDER = [
        ("infrastructure", ["ollama", "lmstudio"]),
        ("core", ["accumulator", "flywheel"]),
        ("agentic", ["prime"]),
    ]
    
    @classmethod
    def execute(cls, skip_docker: bool = False) -> Dict[str, ComponentStatus]:
        """Execute full boot sequence."""
        results = {}
        
        log("═" * 70)
        log("   BIZRA NUCLEUS — Boot Sequence Initiated")
        log("═" * 70)
        
        for phase, components in cls.BOOT_ORDER:
            log(f"\n📦 Phase: {phase.upper()}")
            log("─" * 40)
            
            for component in components:
                status = cls._boot_component(component, skip_docker)
                results[component] = status
                
                icon = "✅" if status.state == ComponentState.HEALTHY else \
                       "⚠️" if status.state == ComponentState.DEGRADED else \
                       "⏸️" if status.state == ComponentState.STOPPED else "❌"
                
                latency_str = f" ({status.latency_ms:.0f}ms)" if status.latency_ms else ""
                log(f"   {icon} {component}: {status.state.value}{latency_str}")
                
                if status.error and status.state == ComponentState.FAILED:
                    log(f"      └─ {status.error}", "ERROR")
        
        log("\n" + "═" * 70)
        
        # Summary
        healthy = sum(1 for s in results.values() if s.state == ComponentState.HEALTHY)
        total = len(results)
        log(f"   Boot Complete: {healthy}/{total} components healthy")
        log("═" * 70)
        
        return results
    
    @classmethod
    def _boot_component(cls, name: str, skip_docker: bool) -> ComponentStatus:
        """Boot individual component."""
        checkers = {
            "ollama": ComponentChecker.check_ollama,
            "lmstudio": ComponentChecker.check_lmstudio,
            "flywheel": ComponentChecker.check_flywheel,
            "accumulator": ComponentChecker.check_accumulator,
            "prime": ComponentChecker.check_prime,
        }
        
        if name not in checkers:
            return ComponentStatus(
                name=name,
                state=ComponentState.UNKNOWN,
                error=f"No checker for {name}",
            )
        
        # For Docker components, optionally start them
        if not skip_docker and name == "ollama":
            cls._ensure_ollama_running()
        
        return checkers[name]()
    
    @staticmethod
    def _ensure_ollama_running():
        """Ensure Ollama is running (Docker or native)."""
        # Check if already running
        if check_port("localhost", 11434):
            return
        
        # Try to start via Docker
        try:
            log("   Starting Ollama container...")
            subprocess.run(
                ["docker", "start", "ollama"],
                capture_output=True,
                timeout=30,
            )
            # Wait for it to be ready
            for _ in range(15):
                if check_port("localhost", 11434):
                    return
                time.sleep(2)
        except Exception as e:
            log(f"   Could not start Ollama: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class HealthMonitor:
    """
    Ibn Sina's Diagnostic Engine — Continuous Health Monitoring.
    """
    
    def __init__(self, interval_seconds: float = 30):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_status: Dict[str, ComponentStatus] = {}
    
    def start(self):
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log("Health monitor started (background)")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._last_status = self.check_all()
                
                # Log any state changes
                unhealthy = [
                    name for name, status in self._last_status.items()
                    if status.state not in (ComponentState.HEALTHY, ComponentState.STOPPED)
                ]
                
                if unhealthy:
                    log(f"Health alert: {', '.join(unhealthy)} not healthy", "WARN")
                
            except Exception as e:
                log(f"Health check error: {e}", "ERROR")
            
            time.sleep(self.interval)
    
    def check_all(self) -> Dict[str, ComponentStatus]:
        """Run all health checks."""
        return {
            "ollama": ComponentChecker.check_ollama(),
            "lmstudio": ComponentChecker.check_lmstudio(),
            "flywheel": ComponentChecker.check_flywheel(),
            "accumulator": ComponentChecker.check_accumulator(),
            "prime": ComponentChecker.check_prime(),
        }
    
    def get_status(self) -> Dict[str, ComponentStatus]:
        """Get last known status."""
        return self._last_status if self._last_status else self.check_all()


# ═══════════════════════════════════════════════════════════════════════════════
# THE NUCLEUS
# ═══════════════════════════════════════════════════════════════════════════════

class Nucleus:
    """
    The BIZRA Nucleus — The Center That Holds.
    
    Unified entry point for the entire BIZRA stack.
    """
    
    _instance: Optional["Nucleus"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.state = NucleusState.OFFLINE
        self.started_at: Optional[datetime] = None
        self.health_monitor = HealthMonitor()
        self.components: Dict[str, ComponentStatus] = {}
        
        # Lazy-loaded components
        self._flywheel = None
        self._accumulator = None
        self._prime = None
        
        self._initialized = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────
    
    def start(self, skip_docker: bool = False) -> bool:
        """Start the Nucleus and all components."""
        if self.state == NucleusState.ONLINE:
            log("Nucleus already online")
            return True
        
        self.state = NucleusState.BOOTING
        self.started_at = datetime.now(timezone.utc)
        
        # Execute boot sequence
        self.components = BootSequence.execute(skip_docker=skip_docker)
        
        # Determine overall state
        healthy = sum(1 for s in self.components.values() if s.state == ComponentState.HEALTHY)
        critical_healthy = all(
            self.components.get(c, ComponentStatus(name=c, state=ComponentState.FAILED)).state 
            in (ComponentState.HEALTHY, ComponentState.STOPPED)
            for c in ["accumulator"]  # Critical components
        )
        
        if healthy >= 2 and critical_healthy:
            self.state = NucleusState.ONLINE
            self.health_monitor.start()
            log("\n🚀 BIZRA NUCLEUS ONLINE")
            return True
        elif healthy >= 1:
            self.state = NucleusState.DEGRADED
            self.health_monitor.start()
            log("\n⚠️ BIZRA NUCLEUS DEGRADED (some components failed)")
            return True
        else:
            self.state = NucleusState.OFFLINE
            log("\n❌ BIZRA NUCLEUS FAILED TO START")
            return False
    
    def stop(self) -> bool:
        """Gracefully stop the Nucleus."""
        if self.state == NucleusState.OFFLINE:
            return True
        
        log("Nucleus shutting down...")
        self.state = NucleusState.SHUTTING_DOWN
        
        # Stop health monitor
        self.health_monitor.stop()
        
        # Save state
        self._save_state()
        
        self.state = NucleusState.OFFLINE
        log("Nucleus offline")
        return True
    
    def status(self) -> NucleusStatus:
        """Get current nucleus status."""
        components = self.health_monitor.get_status() if self.state != NucleusState.OFFLINE else {}
        
        # Get accumulator bloom
        total_bloom = 0.0
        if "accumulator" in components and components["accumulator"].metadata:
            total_bloom = components["accumulator"].metadata.get("total_bloom", 0.0)
        
        # Check inference availability
        inference_available = any(
            components.get(c, ComponentStatus(name=c, state=ComponentState.FAILED)).state == ComponentState.HEALTHY
            for c in ["ollama", "lmstudio"]
        )
        
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        
        return NucleusStatus(
            state=self.state,
            version=NUCLEUS_VERSION,
            uptime_seconds=uptime,
            components=components,
            total_bloom=total_bloom,
            inference_available=inference_available,
            last_health_check=datetime.now(timezone.utc).isoformat(),
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    async def infer(
        self,
        prompt: str,
        model: Optional[str] = None,
        contributor: str = "nucleus:user",
    ) -> str:
        """
        Run inference through the Nucleus.
        
        Auto-selects best available backend.
        Records impact to Accumulator.
        """
        # Check which backend is available
        status = self.health_monitor.get_status()
        
        backend = None
        backend_url = None
        
        if status.get("lmstudio", ComponentStatus(name="lmstudio", state=ComponentState.FAILED)).state == ComponentState.HEALTHY:
            backend = "lmstudio"
            backend_url = LMSTUDIO_URL
        elif status.get("ollama", ComponentStatus(name="ollama", state=ComponentState.FAILED)).state == ComponentState.HEALTHY:
            backend = "ollama"
            backend_url = OLLAMA_URL
        else:
            raise RuntimeError("No inference backend available")
        
        # Run inference
        start_time = time.time()
        
        if backend == "lmstudio":
            response = await self._infer_lmstudio(prompt, model, backend_url)
        else:
            response = await self._infer_ollama(prompt, model, backend_url)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Record to accumulator
        try:
            from accumulator import get_accumulator
            acc = get_accumulator()
            tokens_est = len(prompt.split()) + len(response.split())
            acc.record_computation(
                contributor=contributor,
                tokens_processed=tokens_est,
                latency_ms=latency_ms,
                model=model or backend,
            )
        except Exception as e:
            log(f"Accumulator record failed: {e}", "WARN")
        
        return response
    
    async def _infer_lmstudio(self, prompt: str, model: Optional[str], url: str) -> str:
        """LM Studio inference."""
        import urllib.request
        
        payload = json.dumps({
            "model": model or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode()
        
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read())
            return data["choices"][0]["message"]["content"]
    
    async def _infer_ollama(self, prompt: str, model: Optional[str], url: str) -> str:
        """Ollama inference."""
        import urllib.request
        
        payload = json.dumps({
            "model": model or "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
        }).encode()
        
        req = urllib.request.Request(
            f"{url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read())
            return data.get("response", "")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_state(self):
        """Save nucleus state for resurrection."""
        try:
            NUCLEUS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "state": self.state.value,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "version": NUCLEUS_VERSION,
            }
            
            with open(NUCLEUS_STATE_PATH, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log(f"Failed to save state: {e}", "ERROR")
    
    def _load_state(self):
        """Load persisted state."""
        if NUCLEUS_STATE_PATH.exists():
            try:
                with open(NUCLEUS_STATE_PATH, 'r') as f:
                    data = json.load(f)
                    # State is informational only on load
                    log(f"Found previous state from {data.get('saved_at', 'unknown')}")
            except Exception as e:
                log(f"Failed to load state: {e}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SHELL
# ═══════════════════════════════════════════════════════════════════════════════

class NucleusShell:
    """Interactive REPL for the Nucleus."""
    
    def __init__(self, nucleus: Nucleus):
        self.nucleus = nucleus
        self.commands = {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "health": self.cmd_health,
            "infer": self.cmd_infer,
            "bloom": self.cmd_bloom,
            "leaderboard": self.cmd_leaderboard,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit,
        }
    
    def run(self):
        """Run interactive shell."""
        print("\n" + "═" * 60)
        print("   BIZRA NUCLEUS — Interactive Shell")
        print("   Type 'help' for commands, 'quit' to exit")
        print("═" * 60 + "\n")
        
        while True:
            try:
                line = input("nucleus> ").strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n(Ctrl+C) Use 'quit' to exit")
            except EOFError:
                break
    
    def cmd_help(self, args: str):
        """Show help."""
        print("""
Available Commands:
  status      - Show nucleus status
  health      - Deep health check
  infer TEXT  - Run inference
  bloom       - Show accumulator bloom
  leaderboard - Show top contributors
  help        - This help
  quit        - Exit shell
        """)
    
    def cmd_status(self, args: str):
        """Show status."""
        status = self.nucleus.status()
        print(f"\n📊 Nucleus Status: {status.state.value}")
        print(f"   Version: {status.version}")
        print(f"   Uptime: {status.uptime_seconds:.0f}s")
        print(f"   Inference: {'✅' if status.inference_available else '❌'}")
        print(f"   Total Bloom: {status.total_bloom:.2f}")
        print("\n   Components:")
        for name, comp in status.components.items():
            icon = "✅" if comp.state == ComponentState.HEALTHY else \
                   "⚠️" if comp.state == ComponentState.DEGRADED else \
                   "⏸️" if comp.state == ComponentState.STOPPED else "❌"
            print(f"     {icon} {name}: {comp.state.value}")
        print()
    
    def cmd_health(self, args: str):
        """Deep health check."""
        print("\n🔍 Running health checks...")
        components = self.nucleus.health_monitor.check_all()
        for name, comp in components.items():
            icon = "✅" if comp.state == ComponentState.HEALTHY else \
                   "⚠️" if comp.state == ComponentState.DEGRADED else \
                   "⏸️" if comp.state == ComponentState.STOPPED else "❌"
            latency = f" ({comp.latency_ms:.0f}ms)" if comp.latency_ms else ""
            print(f"  {icon} {name}: {comp.state.value}{latency}")
            if comp.error:
                print(f"     └─ {comp.error}")
        print()
    
    def cmd_infer(self, args: str):
        """Run inference."""
        if not args:
            print("Usage: infer <prompt>")
            return
        
        print("🧠 Running inference...")
        try:
            response = asyncio.run(self.nucleus.infer(args))
            print(f"\n{response}\n")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
    
    def cmd_bloom(self, args: str):
        """Show bloom status."""
        try:
            from accumulator import get_accumulator
            acc = get_accumulator()
            status = acc.status()
            print(f"\n🌸 Accumulator Status: {status['state']}")
            print(f"   Total Bloom: {status['total_bloom']:.2f}")
            print(f"   Contributors: {status['total_contributors']}")
            print(f"   Zakat Pool: {status['zakat_pool']:.2f}")
            print(f"   PoI Attestations: {status['poi_attestations']}")
            print()
        except Exception as e:
            print(f"❌ Accumulator not available: {e}")
    
    def cmd_leaderboard(self, args: str):
        """Show leaderboard."""
        try:
            from accumulator import get_accumulator
            acc = get_accumulator()
            leaders = acc.leaderboard(limit=10)
            print("\n🏆 Leaderboard:")
            for entry in leaders:
                harv = "🍎" if entry["harvestable"] else "🌱"
                print(f"   {entry['rank']}. {entry['contributor']}: {entry['total_bloom']:.2f} {harv}")
            print()
        except Exception as e:
            print(f"❌ Accumulator not available: {e}")
    
    def cmd_quit(self, args: str):
        """Quit shell."""
        print("Goodbye!")
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA NUCLEUS — The Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nucleus.py start        # Boot the stack
  python nucleus.py status       # Check status
  python nucleus.py health       # Deep health check
  python nucleus.py infer "Hi"   # Quick inference
  python nucleus.py shell        # Interactive mode
        """
    )
    
    parser.add_argument("command", choices=[
        "start", "stop", "status", "health", "infer", "shell", "version"
    ], help="Command to execute")
    
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker component startup")
    parser.add_argument("--model", help="Model for inference")
    
    args = parser.parse_args()
    
    nucleus = Nucleus()
    
    if args.command == "version":
        print(f"BIZRA Nucleus v{NUCLEUS_VERSION}")
        return
    
    elif args.command == "start":
        success = nucleus.start(skip_docker=args.skip_docker)
        sys.exit(0 if success else 1)
    
    elif args.command == "stop":
        nucleus.stop()
    
    elif args.command == "status":
        status = nucleus.status()
        print(f"\n📊 Nucleus: {status.state.value}")
        print(f"   Uptime: {status.uptime_seconds:.0f}s")
        print(f"   Inference: {'✅' if status.inference_available else '❌'}")
        print(f"   Bloom: {status.total_bloom:.2f}")
        print()
    
    elif args.command == "health":
        # Quick boot check
        nucleus.start(skip_docker=True)
        health = nucleus.health_monitor.check_all()
        
        print("\n🔍 Component Health:")
        for name, comp in health.items():
            icon = "✅" if comp.state == ComponentState.HEALTHY else \
                   "⚠️" if comp.state == ComponentState.DEGRADED else \
                   "⏸️" if comp.state == ComponentState.STOPPED else "❌"
            print(f"  {icon} {name}: {comp.state.value}")
        print()
    
    elif args.command == "infer":
        if not args.args:
            print("Usage: nucleus.py infer <prompt>")
            sys.exit(1)
        
        # Quick boot
        nucleus.start(skip_docker=True)
        
        prompt = " ".join(args.args)
        try:
            response = asyncio.run(nucleus.infer(prompt, model=args.model))
            print(response)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "shell":
        nucleus.start(skip_docker=True)
        shell = NucleusShell(nucleus)
        shell.run()


if __name__ == "__main__":
    main()
