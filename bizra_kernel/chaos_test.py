"""
BIZRA Chaos Game - Severed Link Scenario Test
Phase 9: Chaos Engineering with MTTR <=30s

This test implements the "Severed Link" scenario:
- Simulates network partition between federation nodes
- Tests automatic failover detection
- Measures MTTR (Mean Time To Recovery)
- Validates MTTR <=30s target
- Demonstrates self-healing orchestration
"""

import asyncio
import time
import json
from typing import List, Dict, Any

try:
    # Try relative imports first (for package usage)
    from .federation_manager import FederationManager, FederationConfig, create_federation_node
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from federation_manager import FederationManager, FederationConfig, create_federation_node


class ChaosGameTest:
    """
    Chaos Game Test Suite for Severed Link Scenario.

    Tests the federation's resilience to network partitions and measures
    recovery time against the MTTR <=30s target.
    """

    def __init__(self):
        self.federation_nodes: Dict[str, FederationManager] = {}
        self.test_results: List[Dict[str, Any]] = []
        self.chaos_events: List[Dict[str, Any]] = []

    async def setup_federation_cluster(self, num_nodes: int = 3) -> List[str]:
        """
        Set up a test federation cluster with the specified number of nodes.

        Returns list of node IDs.
        """
        print(f"[+] Setting up {num_nodes}-node federation cluster for chaos testing")

        node_ids = [f"node_{i}" for i in range(num_nodes)]
        peer_lists = []

        # Create peer lists for each node
        for i, node_id in enumerate(node_ids):
            peers = [n for n in node_ids if n != node_id]
            peer_lists.append(peers)

        # Create and start federation nodes
        for i, node_id in enumerate(node_ids):
            peers = peer_lists[i]
            port = 8888 + i

            config = FederationConfig(
                node_id=node_id,
                peer_nodes=peers,
                port=port
            )

            manager = FederationManager(config)
            self.federation_nodes[node_id] = manager

            # Start node in background
            asyncio.create_task(self._start_node_with_delay(manager, i * 2))

        # Wait for cluster to stabilize
        await asyncio.sleep(10)

        print(f"[+] Federation cluster ready with nodes: {node_ids}")
        return node_ids

    async def _start_node_with_delay(self, manager: FederationManager, delay: float):
        """Start a federation node with delay to avoid connection conflicts."""
        await asyncio.sleep(delay)
        try:
            await manager.start_federation()
        except Exception as e:
            print(f"[!] Failed to start node {manager.config.node_id}: {e}")

    async def run_severed_link_test(self, test_duration: float = 60.0):
        """
        Run the Severed Link chaos test.

        Simulates network partition and measures recovery time.
        """
        print("\n" + "="*60)
        print(">>> STARTING CHAOS GAME: Severed Link Scenario")
        print("="*60)
        print("Target: MTTR <=30s for network partition recovery")
        print(f"Test Duration: {test_duration}s")
        print("="*60)

        # Setup cluster
        node_ids = await self.setup_federation_cluster(3)

        # Wait for initial stabilization
        await asyncio.sleep(15)

        # Get initial status
        initial_status = await self._get_cluster_status()
        print(f"[+] Initial cluster status: {initial_status}")

        # Trigger Severed Link scenario
        print("\n[!] TRIGGERING: Severed Link Scenario")
        print("Isolating node_1 from node_0 and node_2 for 30 seconds")

        # Trigger chaos on node_0 (coordinator)
        coordinator = self.federation_nodes["node_0"]
        event_id = await coordinator.trigger_severed_link_scenario(
            affected_nodes=["node_1"],
            isolated_from=["node_0", "node_2"],
            duration_seconds=30.0
        )

        print(f"[+] Chaos event started: {event_id}")

        # Monitor chaos progress
        start_time = time.time()
        chaos_complete = False

        while time.time() - start_time < test_duration and not chaos_complete:
            await asyncio.sleep(5)

            # Get current status
            current_status = await self._get_cluster_status()
            chaos_status = coordinator.get_chaos_status()

            print(f"[STATUS] t={time.time()-start_time:.1f}s | Chaos: {chaos_status}")

            # Check if chaos event completed
            mttr_report = coordinator.get_mttr_report()
            if mttr_report.get("total_events", 0) > 0:
                chaos_complete = True
                break

        # Get final results
        final_status = await self._get_cluster_status()
        mttr_report = coordinator.get_mttr_report()

        # Record test results
        test_result = {
            "test_name": "severed_link_scenario",
            "timestamp": time.time(),
            "duration_seconds": time.time() - start_time,
            "initial_status": initial_status,
            "final_status": final_status,
            "mttr_report": mttr_report,
            "chaos_events": len(chaos_status.get("chaos_events", [])),
            "target_mttr_seconds": 30.0
        }

        self.test_results.append(test_result)

        # Print results
        await self._print_test_results(test_result)

        # Cleanup
        await self._cleanup_cluster()

        return test_result

    async def _get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all nodes in the cluster."""
        cluster_status = {}

        for node_id, manager in self.federation_nodes.items():
            try:
                status = manager.get_federation_status()
                cluster_status[node_id] = {
                    "leader": status.get("federation_status", {}).get("leader_node"),
                    "health": status.get("federation_status", {}).get("health_score", 0.0),
                    "active_nodes": len(status.get("federation_status", {}).get("active_nodes", [])),
                    "chaos_active": status.get("chaos_status", {}).get("active_partitions", 0) > 0
                }
            except Exception as e:
                cluster_status[node_id] = {"error": str(e)}

        return cluster_status

    async def _print_test_results(self, result: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "="*60)
        print("[RESULTS] CHAOS GAME RESULTS: Severed Link Scenario")
        print("="*60)

        mttr_report = result.get("mttr_report", {})

        if mttr_report.get("total_events", 0) == 0:
            print("❌ No chaos events completed - test may have failed")
            return

        avg_mttr = mttr_report.get("average_mttr", 0)
        target_mttr = result.get("target_mttr_seconds", 30.0)
        compliance_rate = mttr_report.get("target_compliance_rate", 0)

        print(f"MTTR Average: {avg_mttr:.2f}s")
        print(f"MTTR Target: <={target_mttr}s")
        print(f"Compliance Rate: {compliance_rate:.1%}")
        print(f"Total Events: {mttr_report.get('total_events', 0)}")
        print(f"Target Violations: {mttr_report.get('target_violations', 0)}")

        if avg_mttr <= target_mttr:
            print("✅ TARGET MET: MTTR requirement satisfied!")
        else:
            print(f"❌ TARGET MISSED: MTTR {avg_mttr:.2f}s exceeds limit of {target_mttr}s")

        # Print individual events
        events = mttr_report.get("events", [])
        if events:
            print("\nEvent Details:")
            for event in events:
                status = "✅" if event.get("target_met", False) else "❌"
                print(f"  {status} {event['event_id']}: {event['mttr_seconds']:.2f}s")

        print("="*60)

    async def _cleanup_cluster(self):
        """Clean up the test federation cluster."""
        print("\n[-] Cleaning up federation cluster")

        cleanup_tasks = []
        for manager in self.federation_nodes.values():
            cleanup_tasks.append(manager.stop_federation())

        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.federation_nodes.clear()

        print("[-] Cluster cleanup complete")

    def save_test_results(self, filename: str = "chaos_game_results.json"):
        """Save test results to file."""
        with open(filename, 'w') as f:
            json.dump({
                "test_suite": "chaos_game_severed_link",
                "timestamp": time.time(),
                "results": self.test_results
            }, f, indent=2)

        print(f"[+] Test results saved to {filename}")


async def run_chaos_game():
    """Run the complete chaos game test suite."""
    print("[GAME] BIZRA CHAOS GAME - Phase 9")
    print("Testing federation resilience with MTTR <=30s target")

    test_suite = ChaosGameTest()

    try:
        # Run Severed Link test
        result = await test_suite.run_severed_link_test(test_duration=120.0)

        # Save results
        test_suite.save_test_results()

        # Summary
        mttr_report = result.get("mttr_report", {})
        avg_mttr = mttr_report.get("average_mttr", 0)
        target_met = avg_mttr <= 30.0

        print("\n🎯 FINAL RESULT:")
        if target_met:
            print("   ✅ CHAOS GAME PASSED - Federation resilient to network partitions")
        else:
            print("   ❌ CHAOS GAME FAILED - MTTR target not met")

        return target_met

    except Exception as e:
        print(f"[!] Chaos game failed: {e}")
        return False


if __name__ == "__main__":
    # Run chaos game
    success = asyncio.run(run_chaos_game())

    if success:
        print("\n[SUCCESS] Chaos engineering validation complete!")
        exit(0)
    else:
        print("\n[FAILURE] Chaos engineering validation failed!")
        exit(1)