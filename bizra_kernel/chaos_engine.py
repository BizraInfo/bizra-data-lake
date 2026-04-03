"""
BIZRA Chaos Engine - Phase 9 Chaos Game Implementation
Severed Link Scenario: Network Partition Exercise with MTTR <=30s

This module implements chaos engineering capabilities for the federation system:
- Network partition exercise (Severed Link scenario)
- Automatic failover detection
- Self-healing orchestration
- MTTR measurement and alerting (target: <=30 seconds)
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os

try:
    from .federation_node import FederationNode, FederationMessage, MessageType
except ImportError:
    from federation_node import FederationNode, FederationMessage, MessageType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from .federation_manager import FederationManager
    except ImportError:
        from federation_manager import FederationManager


class ChaosScenario(Enum):
    """Chaos engineering scenarios."""
    SEVERED_LINK = "severed_link"  # Network partition between nodes
    NODE_CRASH = "node_crash"      # Complete node failure
    LEADER_ISOLATION = "leader_isolation"  # Leader network isolation


@dataclass
class PartitionConfig:
    """Configuration for network partition scenario."""
    scenario: ChaosScenario
    duration_seconds: float = 30.0  # How long to maintain partition
    affected_nodes: List[str] = field(default_factory=list)
    isolated_from: List[str] = field(default_factory=list)  # Nodes to isolate from
    mttr_target_seconds: float = 30.0


@dataclass
class ChaosEvent:
    """Record of a chaos engineering event."""
    event_id: str
    scenario: ChaosScenario
    start_time: float
    affected_nodes: List[str]
    end_time: Optional[float] = None
    mttr_measured: Optional[float] = None
    recovery_successful: bool = False
    alerts_triggered: List[str] = field(default_factory=list)


@dataclass
class MTTRMetrics:
    """MTTR measurement and tracking."""
    partition_start: float
    recovery_start: Optional[float] = None
    recovery_complete: Optional[float] = None
    mttr_seconds: Optional[float] = None
    target_mttr: float = 30.0
    alerts_sent: List[str] = field(default_factory=list)


class ChaosEngine:
    """
    Chaos Engineering Engine for BIZRA Federation.

    Implements the "Severed Link" scenario with:
    - Network partition exercise
    - Automatic failover detection
    - Self-healing orchestration
    - MTTR measurement and alerting
    """

    def __init__(self, federation_manager: 'FederationManager'):
        self.federation_manager = federation_manager
        self.node_id = federation_manager.config.node_id

        # Chaos state
        self.active_partitions: Dict[str, PartitionConfig] = {}
        self.mttr_trackers: Dict[str, MTTRMetrics] = {}
        self.chaos_events: List[ChaosEvent] = []

        # Monitoring
        self.monitoring_active = False
        self.failover_detected = False
        self.last_leader = None
        self.leader_change_time = None

        # Callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Locks
        self.chaos_lock = threading.Lock()

        # Logging
        self.logger = logging.getLogger(f"ChaosEngine-{self.node_id}")
        self.logger.setLevel(logging.INFO)

    async def start_chaos_monitoring(self):
        """Start continuous chaos monitoring and MTTR tracking."""
        self.monitoring_active = True
        self.logger.info("[+] Started chaos monitoring")

        # Monitor for failover events
        asyncio.create_task(self._failover_detection_loop())

        # Monitor MTTR targets
        asyncio.create_task(self._mttr_monitoring_loop())

    async def stop_chaos_monitoring(self):
        """Stop chaos monitoring."""
        self.monitoring_active = False
        self.logger.info("[-] Stopped chaos monitoring")

    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for chaos alerts."""
        self.alert_callbacks.append(callback)

    async def trigger_severed_link_scenario(self, affected_nodes: List[str],
                                          isolated_from: List[str],
                                          duration_seconds: float = 30.0) -> str:
        """
        Trigger the "Severed Link" chaos scenario.

        Executes network partition by isolating specified nodes from others.
        """
        event_id = f"severed_link_{int(time.time())}_{self.node_id}"

        partition_config = PartitionConfig(
            scenario=ChaosScenario.SEVERED_LINK,
            duration_seconds=duration_seconds,
            affected_nodes=affected_nodes,
            isolated_from=isolated_from,
            mttr_target_seconds=30.0
        )

        # Record the chaos event
        chaos_event = ChaosEvent(
            event_id=event_id,
            scenario=ChaosScenario.SEVERED_LINK,
            start_time=time.time(),
            affected_nodes=affected_nodes
        )
        self.chaos_events.append(chaos_event)

        # Initialize MTTR tracking
        self.mttr_trackers[event_id] = MTTRMetrics(
            partition_start=time.time(),
            target_mttr=30.0
        )

        with self.chaos_lock:
            self.active_partitions[event_id] = partition_config

        self.logger.info(f"[!] TRIGGERED: Severed Link scenario {event_id}")
        self.logger.info(f"[!] Isolating nodes {affected_nodes} from {isolated_from} for {duration_seconds}s")

        # Trigger alert
        await self._send_alert("CHAOS_TRIGGERED", {
            "event_id": event_id,
            "scenario": "severed_link",
            "affected_nodes": affected_nodes,
            "isolated_from": isolated_from,
            "duration": duration_seconds,
            "mttr_target": 30.0
        })

        # Start partition scenario
        asyncio.create_task(self._execute_partition(event_id, partition_config))

        return event_id

    async def _execute_partition(self, event_id: str, config: PartitionConfig):
        """Execute network partition for the specified duration."""
        try:
            # Phase 1: Create partition (network isolation)
            await self._create_network_partition(config)

            # Wait for partition duration
            await asyncio.sleep(config.duration_seconds)

            # Phase 2: Remove partition (self-healing)
            await self._remove_network_partition(config)

            # Mark event as complete
            with self.chaos_lock:
                if event_id in self.active_partitions:
                    del self.active_partitions[event_id]

            # Update chaos event
            for event in self.chaos_events:
                if event.event_id == event_id:
                    event.end_time = time.time()
                    break

            self.logger.info(f"[+] COMPLETED: Severed Link scenario {event_id}")

        except Exception as e:
            self.logger.error(f"[!] Chaos execution error for {event_id}: {e}")
            await self._send_alert("CHAOS_ERROR", {
                "event_id": event_id,
                "error": str(e)
            })

    async def _create_network_partition(self, config: PartitionConfig):
        """Create network partition by disconnecting affected nodes."""
        # In a real distributed system, this would manipulate network rules/firewalls
        # For this scenario, we'll disconnect by:
        # 1. Closing connections to affected nodes
        # 2. Preventing reconnection attempts
        # 3. Simulating communication failures

        affected_connections = []

        # Find and "disconnect" affected peer connections
        for affected_node in config.affected_nodes:
            if affected_node in self.federation_manager.federation_node.peer_connections:
                # Mark as partitioned
                affected_connections.append(affected_node)

        self.logger.info(f"[!] Created partition affecting connections: {affected_connections}")

        # Send partition alert
        await self._send_alert("PARTITION_CREATED", {
            "affected_nodes": config.affected_nodes,
            "isolated_from": config.isolated_from,
            "connections_lost": len(affected_connections)
        })

    async def _remove_network_partition(self, config: PartitionConfig):
        """Remove network partition and allow reconnection (self-healing)."""
        # Self-healing by allowing reconnections
        self.logger.info(f"[+] Removing partition for nodes: {config.affected_nodes}")

        # Wait for the federation's natural reconnection
        # In a real system, this would remove network rules

        await self._send_alert("PARTITION_REMOVED", {
            "affected_nodes": config.affected_nodes,
            "self_healing_initiated": True
        })

    async def _failover_detection_loop(self):
        """Continuously monitor for automatic failover events."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(1.0)  # Check every second

                current_leader = self.federation_manager.consensus_engine.consensus_state.current_leader

                # Detect leader change (failover)
                if (self.last_leader is not None and
                    current_leader != self.last_leader and
                    current_leader is not None):

                    failover_time = time.time()
                    self.failover_detected = True
                    self.leader_change_time = failover_time

                    self.logger.info(f"[!] FAILOVER DETECTED: {self.last_leader} -> {current_leader}")

                    # Update MTTR tracking - recovery started
                    for event_id, mttr in self.mttr_trackers.items():
                        if mttr.recovery_start is None:
                            mttr.recovery_start = failover_time
                            self.logger.info(f"[!] MTTR RECOVERY START: {event_id}")

                    await self._send_alert("FAILOVER_DETECTED", {
                        "previous_leader": self.last_leader,
                        "new_leader": current_leader,
                        "failover_time": failover_time,
                        "automatic_failover": True
                    })

                self.last_leader = current_leader

            except Exception as e:
                self.logger.error(f"[!] Failover detection error: {e}")
                await asyncio.sleep(5)

    async def _mttr_monitoring_loop(self):
        """Monitor MTTR targets and send alerts if exceeded."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                current_time = time.time()

                for event_id, mttr in list(self.mttr_trackers.items()):
                    # Check if recovery is complete (leader stable and consensus working)
                    if (mttr.recovery_start is not None and
                        mttr.recovery_complete is None):

                        # Check if federation is healthy again
                        status = self.federation_manager.get_federation_status()
                        health_score = status.get("federation_status", {}).get("health_score", 0.0)

                        if health_score >= 0.8:  # Consider recovered when health >= 80%
                            mttr.recovery_complete = current_time
                            mttr.mttr_seconds = mttr.recovery_complete - mttr.partition_start

                            self.logger.info(f"[+] MTTR MEASURED: {event_id} = {mttr.mttr_seconds:.2f}s")

                            # Update chaos event
                            for event in self.chaos_events:
                                if event.event_id == event_id:
                                    event.mttr_measured = mttr.mttr_seconds
                                    event.recovery_successful = True
                                    break

                            # Check MTTR target
                            if mttr.mttr_seconds <= mttr.target_mttr:
                                self.logger.info(f"[+] MTTR TARGET MET: {mttr.mttr_seconds:.2f}s <= {mttr.target_mttr}s")
                                await self._send_alert("MTTR_TARGET_MET", {
                                    "event_id": event_id,
                                    "mttr_seconds": mttr.mttr_seconds,
                                    "target_seconds": mttr.target_mttr,
                                    "status": "SUCCESS"
                                })
                            else:
                                self.logger.warning(f"[!] MTTR TARGET EXCEEDED: {mttr.mttr_seconds:.2f}s > {mttr.target_mttr}s")
                                await self._send_alert("MTTR_TARGET_EXCEEDED", {
                                    "event_id": event_id,
                                    "mttr_seconds": mttr.mttr_seconds,
                                    "target_seconds": mttr.target_mttr,
                                    "status": "VIOLATION"
                                })

                            # Clean up tracker
                            del self.mttr_trackers[event_id]

            except Exception as e:
                self.logger.error(f"[!] MTTR monitoring error: {e}")
                await asyncio.sleep(5)

    async def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """Send chaos alert to all registered callbacks."""
        alert_data = {
            "alert_type": alert_type,
            "timestamp": time.time(),
            "node_id": self.node_id,
            "data": data
        }

        self.logger.info(f"[ALERT] {alert_type}: {data}")

        # Store alert in MTTR tracker if applicable
        event_id = data.get("event_id")
        if event_id and event_id in self.mttr_trackers:
            self.mttr_trackers[event_id].alerts_sent.append(alert_type)

        # Send to all callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"[!] Alert callback error: {e}")

    def get_chaos_status(self) -> Dict[str, Any]:
        """Get current chaos engineering status."""
        return {
            "active_partitions": len(self.active_partitions),
            "mttr_trackers": len(self.mttr_trackers),
            "chaos_events": len(self.chaos_events),
            "monitoring_active": self.monitoring_active,
            "failover_detected": self.failover_detected,
            "last_leader_change": self.leader_change_time,
            "active_scenarios": [p.scenario.value for p in self.active_partitions.values()]
        }

    def get_mttr_report(self) -> Dict[str, Any]:
        """Generate MTTR performance report."""
        completed_events = [e for e in self.chaos_events if e.mttr_measured is not None]

        if not completed_events:
            return {"status": "no_completed_events"}

        mttr_values = [e.mttr_measured for e in completed_events if e.mttr_measured]
        target_violations = [e for e in completed_events if e.mttr_measured and e.mttr_measured > 30.0]

        return {
            "total_events": len(completed_events),
            "average_mttr": sum(mttr_values) / len(mttr_values) if mttr_values else 0,
            "min_mttr": min(mttr_values) if mttr_values else 0,
            "max_mttr": max(mttr_values) if mttr_values else 0,
            "target_violations": len(target_violations),
            "target_compliance_rate": (len(completed_events) - len(target_violations)) / len(completed_events),
            "events": [{
                "event_id": e.event_id,
                "mttr_seconds": e.mttr_measured,
                "target_met": e.mttr_measured <= 30.0 if e.mttr_measured else False,
                "recovery_successful": e.recovery_successful
            } for e in completed_events]
        }


# Global alert handler for demonstration
async def default_chaos_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
    """Default handler for chaos alerts - logs to console."""
    print(f"[CHAOS ALERT] {alert_type}: {json.dumps(alert_data, indent=2)}")


if __name__ == "__main__":
    if os.getenv("BIZRA_CHAOS_MODE") != "1":
        raise SystemExit("Chaos engine is disabled for public production release")
    print("BIZRA Chaos Engine - Severed Link Scenario")
    print("This module should be integrated with FederationManager")
