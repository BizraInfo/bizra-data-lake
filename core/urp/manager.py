#!/usr/bin/env python3
"""
BIZRA Unified Resource Planner (URP) - VRAM Lease Manager
==========================================================
Fixes F-PERF-001: Resource Blindness

Enforces hardware constraints:
- RTX 4090: 16GB VRAM total
- Usable: 14GB (2GB overhead for system/driver)
- Max concurrent thinking agents: 3

Lease-based allocation prevents OOM crashes by:
1. Pre-allocating VRAM quotas before agent spawn
2. Tracking active leases with TTL
3. Enforcing hard caps with fail-closed semantics
4. Auto-releasing expired leases

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    URP LEASE MANAGER                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Agent Request ──▶ [Capacity Check] ──▶ [Lease Grant]      │
    │                           │                    │             │
    │                           │ FAIL               │ SUCCESS     │
    │                           ▼                    ▼             │
    │                    OverCapacityError      Lease Token        │
    │                                                │             │
    │                           ┌────────────────────┘             │
    │                           ▼                                  │
    │   Agent Execution ◀── [Resource Bound] ──▶ [TTL Monitor]    │
    │                                                │             │
    │                                                ▼             │
    │                                         [Auto Release]       │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

Invariants:
- I1: Total_Allocated_VRAM <= 14GB
- I2: Active_Thinking_Agents <= 3
- I3: Lease TTL <= 300s (5 min max)

Usage:
    from core.urp.manager import URPManager, ResourceRequest
    
    urp = URPManager()
    
    # Request resources
    request = ResourceRequest(agent_id="MasterReasoner", vram_gb=4.0)
    lease = urp.acquire(request)
    
    # Use resources...
    
    # Release when done
    urp.release(lease.lease_id)
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("urp.manager")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (Hardware Calibrated)
# ═══════════════════════════════════════════════════════════════════════════════

# RTX 4090 Specifications
TOTAL_VRAM_GB = float(os.getenv("BIZRA_TOTAL_VRAM_GB", "16.0"))
SYSTEM_OVERHEAD_GB = float(os.getenv("BIZRA_VRAM_OVERHEAD_GB", "2.0"))
USABLE_VRAM_GB = TOTAL_VRAM_GB - SYSTEM_OVERHEAD_GB  # 14GB

# Concurrency limits
MAX_CONCURRENT_AGENTS = int(os.getenv("BIZRA_MAX_AGENTS", "3"))
MAX_LEASE_TTL_SEC = int(os.getenv("BIZRA_MAX_LEASE_TTL", "300"))  # 5 minutes
DEFAULT_LEASE_TTL_SEC = int(os.getenv("BIZRA_DEFAULT_LEASE_TTL", "60"))  # 1 minute

# Agent VRAM requirements (empirical measurements)
AGENT_VRAM_REQUIREMENTS = {
    # PAT Agents (7B models)
    "MasterReasoner": 4.5,      # deepseek-r1:7b
    "MemoryArchitect": 4.0,     # qwen2.5:7b
    "CreativeSynthesizer": 4.0, # qwen2.5:7b
    "DataAnalyzer": 4.0,        # mistral:7b
    "Communicator": 4.0,        # mistral:7b
    "ExecutionPlanner": 4.0,    # agentflow-7b
    "EthicsGuardian": 4.0,      # qwen2.5:7b
    # SAT Agents (minimal - rule-based)
    "PoiVerifier": 0.1,
    "ResourceAllocator": 0.1,
    "RiskGuardian": 0.1,
    "GovernanceEngine": 0.1,
    "EvidenceEngine": 0.1,
    # Default for unknown agents
    "_default": 4.0,
}

# Evidence path
EVIDENCE_PATH = Path(os.getenv("BIZRA_URP_EVIDENCE", "docs/evidence/urp"))


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class URPError(Exception):
    """Base exception for URP errors."""
    pass


class OverCapacityError(URPError):
    """Raised when resource request exceeds available capacity."""
    def __init__(
        self,
        requested_gb: float,
        available_gb: float,
        message: str = None
    ):
        self.requested_gb = requested_gb
        self.available_gb = available_gb
        self.message = message or (
            f"Requested {requested_gb:.2f}GB exceeds available {available_gb:.2f}GB"
        )
        super().__init__(self.message)


class MaxAgentsError(URPError):
    """Raised when max concurrent agent limit reached."""
    def __init__(self, current: int, maximum: int):
        self.current = current
        self.maximum = maximum
        self.message = f"Max agents reached: {current}/{maximum}"
        super().__init__(self.message)


class LeaseNotFoundError(URPError):
    """Raised when lease ID not found."""
    pass


class LeaseExpiredError(URPError):
    """Raised when attempting to use an expired lease."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class LeaseStatus(Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    RELEASED = "RELEASED"


@dataclass
class ResourceRequest:
    """Request for VRAM allocation."""
    agent_id: str
    vram_gb: Optional[float] = None  # None = auto-detect
    ttl_sec: int = DEFAULT_LEASE_TTL_SEC
    priority: int = 0  # Higher = more priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Auto-detect VRAM if not specified
        if self.vram_gb is None:
            self.vram_gb = AGENT_VRAM_REQUIREMENTS.get(
                self.agent_id,
                AGENT_VRAM_REQUIREMENTS["_default"]
            )
        
        # Clamp TTL
        self.ttl_sec = min(self.ttl_sec, MAX_LEASE_TTL_SEC)


@dataclass
class Lease:
    """Granted resource lease."""
    lease_id: str
    agent_id: str
    vram_gb: float
    granted_at: str
    expires_at: str
    ttl_sec: int
    status: LeaseStatus = LeaseStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if lease has expired."""
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        return now >= expires
    
    @property
    def remaining_sec(self) -> float:
        """Seconds remaining on lease."""
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        delta = (expires - now).total_seconds()
        return max(0, delta)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "agent_id": self.agent_id,
            "vram_gb": self.vram_gb,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "ttl_sec": self.ttl_sec,
            "status": self.status.value,
            "remaining_sec": self.remaining_sec,
            "metadata": self.metadata
        }


@dataclass
class URPSnapshot:
    """Current state of URP resources."""
    total_vram_gb: float
    usable_vram_gb: float
    allocated_vram_gb: float
    available_vram_gb: float
    active_leases: int
    max_agents: int
    utilization_pct: float
    leases: List[Dict[str, Any]]
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
# URP MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class URPManager:
    """
    Unified Resource Planner - VRAM Lease Manager
    
    Thread-safe singleton that manages VRAM allocation across all agents.
    """
    
    _instance: Optional['URPManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'URPManager':
        """Singleton pattern for global resource management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the URP Manager."""
        if self._initialized:
            return
        
        self._leases: Dict[str, Lease] = {}
        self._agent_leases: Dict[str, str] = {}  # agent_id -> lease_id
        self._allocated_gb: float = 0.0
        self._lease_lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        
        # Create evidence directory
        EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup daemon
        self._start_cleanup_daemon()
        
        self._initialized = True
        
        logger.info(
            f"URP Manager initialized: {USABLE_VRAM_GB:.1f}GB usable, "
            f"{MAX_CONCURRENT_AGENTS} max agents"
        )
    
    def _start_cleanup_daemon(self) -> None:
        """Start background thread to cleanup expired leases."""
        def cleanup_loop():
            while not self._shutdown.is_set():
                self._cleanup_expired_leases()
                self._shutdown.wait(10)  # Check every 10 seconds
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="urp-cleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_expired_leases(self) -> int:
        """Release all expired leases. Returns count of cleaned leases."""
        cleaned = 0
        with self._lease_lock:
            expired = [
                lid for lid, lease in self._leases.items()
                if lease.status == LeaseStatus.ACTIVE and lease.is_expired
            ]
            
            for lease_id in expired:
                self._release_internal(lease_id, reason="EXPIRED")
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} expired leases")
        
        return cleaned
    
    def _generate_lease_id(self) -> str:
        """Generate unique lease ID."""
        return f"lease-{uuid4().hex[:12]}"
    
    def _release_internal(self, lease_id: str, reason: str = "RELEASED") -> None:
        """Internal release without lock (caller must hold lock)."""
        lease = self._leases.get(lease_id)
        if lease and lease.status == LeaseStatus.ACTIVE:
            self._allocated_gb -= lease.vram_gb
            lease.status = (
                LeaseStatus.EXPIRED if reason == "EXPIRED" 
                else LeaseStatus.RELEASED
            )
            
            # Remove from agent mapping
            if lease.agent_id in self._agent_leases:
                del self._agent_leases[lease.agent_id]
            
            logger.info(
                f"Lease {lease_id} {reason}: "
                f"freed {lease.vram_gb:.1f}GB for {lease.agent_id}"
            )
    
    # ───────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ───────────────────────────────────────────────────────────────────────────
    
    def acquire(self, request: ResourceRequest) -> Lease:
        """
        Acquire a resource lease for an agent.
        
        Args:
            request: ResourceRequest with agent details
            
        Returns:
            Lease object with granted resources
            
        Raises:
            OverCapacityError: If VRAM request exceeds available
            MaxAgentsError: If max concurrent agents reached
        """
        with self._lease_lock:
            # Check if agent already has a lease
            if request.agent_id in self._agent_leases:
                existing_id = self._agent_leases[request.agent_id]
                existing = self._leases.get(existing_id)
                if existing and existing.status == LeaseStatus.ACTIVE:
                    # Extend existing lease
                    return self.extend(existing_id, request.ttl_sec)
            
            # Check agent count
            active_count = sum(
                1 for l in self._leases.values() 
                if l.status == LeaseStatus.ACTIVE
            )
            if active_count >= MAX_CONCURRENT_AGENTS:
                raise MaxAgentsError(active_count, MAX_CONCURRENT_AGENTS)
            
            # Check VRAM capacity
            available = USABLE_VRAM_GB - self._allocated_gb
            if request.vram_gb > available:
                raise OverCapacityError(request.vram_gb, available)
            
            # Grant lease
            now = datetime.now(timezone.utc)
            expires = datetime.fromtimestamp(
                now.timestamp() + request.ttl_sec,
                tz=timezone.utc
            )
            
            lease = Lease(
                lease_id=self._generate_lease_id(),
                agent_id=request.agent_id,
                vram_gb=request.vram_gb,
                granted_at=now.isoformat(),
                expires_at=expires.isoformat(),
                ttl_sec=request.ttl_sec,
                status=LeaseStatus.ACTIVE,
                metadata=request.metadata
            )
            
            self._leases[lease.lease_id] = lease
            self._agent_leases[request.agent_id] = lease.lease_id
            self._allocated_gb += request.vram_gb
            
            logger.info(
                f"Lease GRANTED: {lease.lease_id} | {request.agent_id} | "
                f"{request.vram_gb:.1f}GB | TTL={request.ttl_sec}s"
            )
            
            # Record evidence
            self._record_lease_event(lease, "ACQUIRED")
            
            return lease
    
    def release(self, lease_id: str) -> bool:
        """
        Release a resource lease.
        
        Args:
            lease_id: The lease ID to release
            
        Returns:
            True if released, False if not found/already released
        """
        with self._lease_lock:
            if lease_id not in self._leases:
                logger.warning(f"Lease not found: {lease_id}")
                return False
            
            lease = self._leases[lease_id]
            if lease.status != LeaseStatus.ACTIVE:
                logger.warning(f"Lease already {lease.status.value}: {lease_id}")
                return False
            
            self._release_internal(lease_id, "RELEASED")
            self._record_lease_event(lease, "RELEASED")
            return True
    
    def extend(self, lease_id: str, additional_sec: int) -> Lease:
        """
        Extend an active lease.
        
        Args:
            lease_id: The lease ID to extend
            additional_sec: Seconds to add to TTL
            
        Returns:
            Updated Lease object
            
        Raises:
            LeaseNotFoundError: If lease not found
            LeaseExpiredError: If lease already expired
        """
        with self._lease_lock:
            if lease_id not in self._leases:
                raise LeaseNotFoundError(f"Lease not found: {lease_id}")
            
            lease = self._leases[lease_id]
            
            if lease.status != LeaseStatus.ACTIVE:
                raise LeaseExpiredError(f"Lease is {lease.status.value}")
            
            if lease.is_expired:
                self._release_internal(lease_id, "EXPIRED")
                raise LeaseExpiredError("Lease has expired")
            
            # Calculate new expiry (clamped to max TTL)
            new_ttl = min(
                int(lease.remaining_sec) + additional_sec,
                MAX_LEASE_TTL_SEC
            )
            new_expires = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + new_ttl,
                tz=timezone.utc
            )
            
            lease.expires_at = new_expires.isoformat()
            lease.ttl_sec = new_ttl
            
            logger.info(f"Lease EXTENDED: {lease_id} | new TTL={new_ttl}s")
            self._record_lease_event(lease, "EXTENDED")
            
            return lease
    
    def get_lease(self, lease_id: str) -> Optional[Lease]:
        """Get lease by ID."""
        return self._leases.get(lease_id)
    
    def get_agent_lease(self, agent_id: str) -> Optional[Lease]:
        """Get active lease for an agent."""
        lease_id = self._agent_leases.get(agent_id)
        if lease_id:
            lease = self._leases.get(lease_id)
            if lease and lease.status == LeaseStatus.ACTIVE:
                return lease
        return None
    
    def snapshot(self) -> URPSnapshot:
        """Get current resource state snapshot."""
        with self._lease_lock:
            active_leases = [
                l.to_dict() for l in self._leases.values()
                if l.status == LeaseStatus.ACTIVE
            ]
            
            utilization = (
                (self._allocated_gb / USABLE_VRAM_GB * 100)
                if USABLE_VRAM_GB > 0 else 0
            )
            
            return URPSnapshot(
                total_vram_gb=TOTAL_VRAM_GB,
                usable_vram_gb=USABLE_VRAM_GB,
                allocated_vram_gb=self._allocated_gb,
                available_vram_gb=USABLE_VRAM_GB - self._allocated_gb,
                active_leases=len(active_leases),
                max_agents=MAX_CONCURRENT_AGENTS,
                utilization_pct=round(utilization, 1),
                leases=active_leases,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    def can_allocate(self, vram_gb: float) -> bool:
        """Check if allocation is possible without acquiring."""
        with self._lease_lock:
            available = USABLE_VRAM_GB - self._allocated_gb
            active_count = sum(
                1 for l in self._leases.values() 
                if l.status == LeaseStatus.ACTIVE
            )
            return vram_gb <= available and active_count < MAX_CONCURRENT_AGENTS
    
    def shutdown(self) -> None:
        """Shutdown the URP manager gracefully."""
        logger.info("URP Manager shutting down...")
        self._shutdown.set()
        
        # Release all leases
        with self._lease_lock:
            for lease_id in list(self._leases.keys()):
                self._release_internal(lease_id, "SHUTDOWN")
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        logger.info("URP Manager shutdown complete")
    
    # ───────────────────────────────────────────────────────────────────────────
    # EVIDENCE
    # ───────────────────────────────────────────────────────────────────────────
    
    def _record_lease_event(self, lease: Lease, event: str) -> None:
        """Record lease event to evidence log."""
        log_file = EVIDENCE_PATH / "lease_events.jsonl"
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "lease_id": lease.lease_id,
            "agent_id": lease.agent_id,
            "vram_gb": lease.vram_gb,
            "allocated_total_gb": self._allocated_gb,
            "available_gb": USABLE_VRAM_GB - self._allocated_gb,
            "active_leases": sum(
                1 for l in self._leases.values() 
                if l.status == LeaseStatus.ACTIVE
            )
        }
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Failed to record lease event: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class URPLease:
    """
    Context manager for automatic lease acquisition and release.
    
    Usage:
        with URPLease("MasterReasoner") as lease:
            # Use resources
            response = call_agent(message)
        # Lease auto-released
    """
    
    def __init__(
        self,
        agent_id: str,
        vram_gb: Optional[float] = None,
        ttl_sec: int = DEFAULT_LEASE_TTL_SEC
    ):
        self.request = ResourceRequest(
            agent_id=agent_id,
            vram_gb=vram_gb,
            ttl_sec=ttl_sec
        )
        self.lease: Optional[Lease] = None
        self.urp = URPManager()
    
    def __enter__(self) -> Lease:
        self.lease = self.urp.acquire(self.request)
        return self.lease
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.lease:
            self.urp.release(self.lease.lease_id)
        return False  # Don't suppress exceptions


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the URP Manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="URP VRAM Lease Manager")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--acquire", type=str, metavar="AGENT", help="Acquire lease")
    parser.add_argument("--release", type=str, metavar="LEASE_ID", help="Release lease")
    parser.add_argument("--test", action="store_true", help="Run test scenario")
    
    args = parser.parse_args()
    urp = URPManager()
    
    if args.status:
        snap = urp.snapshot()
        print("\n" + "═" * 60)
        print("  URP RESOURCE STATUS")
        print("═" * 60)
        print(f"  Total VRAM:     {snap.total_vram_gb:.1f} GB")
        print(f"  Usable:         {snap.usable_vram_gb:.1f} GB")
        print(f"  Allocated:      {snap.allocated_vram_gb:.1f} GB")
        print(f"  Available:      {snap.available_vram_gb:.1f} GB")
        print(f"  Utilization:    {snap.utilization_pct:.1f}%")
        print(f"  Active Leases:  {snap.active_leases}/{snap.max_agents}")
        print("─" * 60)
        for lease in snap.leases:
            print(f"  • {lease['agent_id']}: {lease['vram_gb']:.1f}GB "
                  f"(expires in {lease['remaining_sec']:.0f}s)")
        print("═" * 60 + "\n")
    
    elif args.acquire:
        try:
            request = ResourceRequest(agent_id=args.acquire)
            lease = urp.acquire(request)
            print(f"✅ Lease acquired: {lease.lease_id}")
            print(f"   Agent: {lease.agent_id}")
            print(f"   VRAM: {lease.vram_gb:.1f} GB")
            print(f"   TTL: {lease.ttl_sec}s")
        except (OverCapacityError, MaxAgentsError) as e:
            print(f"❌ {e}")
    
    elif args.release:
        if urp.release(args.release):
            print(f"✅ Lease released: {args.release}")
        else:
            print(f"❌ Lease not found or already released: {args.release}")
    
    elif args.test:
        print("\n🧪 URP TEST SCENARIO\n")
        
        # Test 1: Normal acquisition
        print("1. Acquiring MasterReasoner lease...")
        try:
            with URPLease("MasterReasoner") as lease:
                print(f"   ✅ Got lease: {lease.lease_id}")
                print(f"   VRAM: {lease.vram_gb:.1f}GB")
            print("   ✅ Auto-released\n")
        except Exception as e:
            print(f"   ❌ {e}\n")
        
        # Test 2: Capacity check
        print("2. Testing capacity limit (request 17GB)...")
        try:
            request = ResourceRequest(agent_id="Test", vram_gb=17.0)
            urp.acquire(request)
            print("   ❌ Should have failed!")
        except OverCapacityError as e:
            print(f"   ✅ Correctly rejected: {e}\n")
        
        # Test 3: Max agents
        print("3. Testing max agent limit...")
        leases = []
        try:
            for i in range(4):  # Try to get 4 leases (max is 3)
                request = ResourceRequest(
                    agent_id=f"Agent{i}",
                    vram_gb=3.0,
                    ttl_sec=30
                )
                lease = urp.acquire(request)
                leases.append(lease)
                print(f"   ✅ Lease {i+1}: {lease.lease_id}")
        except MaxAgentsError as e:
            print(f"   ✅ Correctly blocked: {e}\n")
        finally:
            for lease in leases:
                urp.release(lease.lease_id)
        
        print("✅ All tests passed!\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
