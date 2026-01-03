"""
BIZRA Federation Integration Test
Phase 9: 3-Node Local Federation Validation

Comprehensive test suite for the federation implementation including:
- Node initialization and connectivity
- Byzantine fault tolerant consensus
- Knowledge graph sharding and distribution
- Cross-node Graph-of-Thoughts reasoning
- Failure recovery and rebalancing
"""

import asyncio
import time
import threading
import sys
from typing import List, Dict, Any
import json

try:
    from .federation_manager import FederationManager, FederationConfig
    from .memory_system import CognitivePermanence
except ImportError:
    from federation_manager import FederationManager, FederationConfig
    from memory_system import CognitivePermanence


class FederationTestSuite:
    """Comprehensive test suite for the BIZRA federation."""

    def __init__(self):
        self.nodes: Dict[str, FederationManager] = {}
        self.test_results: Dict[str, Any] = {}
        self.test_start_time = time.time()

    async def setup_test_federation(self, node_ids: List[str] = None):
        """Set up a test federation with multiple nodes."""
        if node_ids is None:
            node_ids = ["node_0", "node_1", "node_2"]

        print(f"[+] Setting up test federation with nodes: {node_ids}")

        # Create federation managers
        for i, node_id in enumerate(node_ids):
            peer_nodes = [n for n in node_ids if n != node_id]

            config = FederationConfig(
                node_id=node_id,
                peer_nodes=peer_nodes,
                port=8888 + i  # Different ports
            )

            manager = FederationManager(config)
            self.nodes[node_id] = manager

        # Pre-populate with test knowledge
        await self._populate_test_knowledge()

        print(f"[+] Test federation setup complete")

    async def _populate_test_knowledge(self):
        """Populate nodes with test knowledge for federation testing."""
        test_entities = [
            {
                "entity": "BIZRA",
                "fact": "The primary sovereign node for post-labor economics and decentralized sovereignty",
                "rels": {"Consensus": "uses", "Security": "provides", "Sovereignty": "achieves"}
            },
            {
                "entity": "Consensus",
                "fact": "Distributed agreement protocol ensuring Byzantine fault tolerance",
                "rels": {"BIZRA": "powers", "Security": "requires", "Trust": "establishes"}
            },
            {
                "entity": "Security",
                "fact": "Cryptographic protection mechanisms against adversarial attacks",
                "rels": {"Consensus": "enables", "Privacy": "protects", "Integrity": "maintains"}
            },
            {
                "entity": "Sovereignty",
                "fact": "Autonomous control over digital assets and decision making",
                "rels": {"BIZRA": "achieves", "Decentralization": "requires", "Trust": "builds"}
            },
            {
                "entity": "Decentralization",
                "fact": "Distribution of power and control across multiple independent nodes",
                "rels": {"Sovereignty": "enables", "Consensus": "depends_on", "Resilience": "provides"}
            }
        ]

        # Add entities to each node's memory system
        for node_id, manager in self.nodes.items():
            for i, entity_data in enumerate(test_entities):
                entity_id = f"{node_id}_entity_{i}"
                manager.memory_system.add_semantic_fact(
                    entity_data["entity"],
                    entity_data["fact"],
                    entity_data["rels"]
                )

        print("[+] Test knowledge populated across all nodes")

    async def run_connectivity_test(self) -> Dict[str, Any]:
        """Test node connectivity and basic federation formation."""
        print("[*] Running connectivity test...")

        start_time = time.time()

        # Start all nodes
        node_tasks = []
        for node_id, manager in self.nodes.items():
            task = asyncio.create_task(manager.start_federation())
            node_tasks.append((node_id, task))

        # Let nodes start up
        await asyncio.sleep(5)

        # Check connectivity
        connectivity_results = {}
        for node_id, manager in self.nodes.items():
            status = manager.get_federation_status()
            network_status = status.get("network_status", {})
            connectivity_results[node_id] = {
                "connected_peers": network_status.get("connected_peers", 0),
                "expected_peers": network_status.get("expected_peers", 0),
                "health_score": status.get("federation_status", {}).get("health_score", 0)
            }

        # Stop nodes
        stop_tasks = [manager.stop_federation() for manager in self.nodes.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        test_duration = time.time() - start_time

        result = {
            "test": "connectivity",
            "duration": test_duration,
            "connectivity_results": connectivity_results,
            "overall_success": all(r["connected_peers"] >= len(self.nodes) - 1 for r in connectivity_results.values())
        }

        print(f"[+] Connectivity test completed in {test_duration:.2f}s")
        return result

    async def run_consensus_test(self) -> Dict[str, Any]:
        """Test Byzantine fault tolerant consensus."""
        print("[*] Running consensus test...")

        start_time = time.time()

        # Start federation
        await self._start_test_federation()

        # Submit consensus requests
        consensus_results = []
        test_requests = [
            {
                "action_name": "test_consensus_1",
                "action_data": {"test": "data_1"},
                "metrics": {"im_score": 0.95, "status": "APPROVED", "timestamp": time.time(), "signature": "test"}
            },
            {
                "action_name": "test_consensus_2",
                "action_data": {"test": "data_2"},
                "metrics": {"im_score": 0.92, "status": "APPROVED", "timestamp": time.time(), "signature": "test"}
            }
        ]

        # Find leader
        leader_node_id = None
        for node_id, manager in self.nodes.items():
            status = manager.get_federation_status()
            if status.get("federation_status", {}).get("leader_node") == node_id:
                leader_node_id = node_id
                break
        
        if not leader_node_id and self.nodes:
             # Fallback if leader election lagging
             leader_node_id = list(self.nodes.keys())[0]
             print(f"[!] No leader detected, defaulting to {leader_node_id}")

        print(f"[*] Submitting requests via Leader: {leader_node_id}")
        leader_manager = self.nodes[leader_node_id]

        # Submit requests via Leader
        for i, request_data in enumerate(test_requests):
            print(f"[*] Submitting request {i+1}/{len(test_requests)}: {request_data['action_name']}")
            result = await leader_manager.submit_federated_request("validate_and_commit", request_data)
            
            consensus_results.append({
                "node": leader_node_id,
                "request": request_data["action_name"],
                "result": result
            })

        # Check consensus success
        successful_consensus = sum(1 for r in consensus_results if r["result"].get("status") == "completed")

        await self._stop_test_federation()

        test_duration = time.time() - start_time

        result = {
            "test": "consensus",
            "duration": test_duration,
            "consensus_results": consensus_results,
            "successful_consensus": successful_consensus,
            "total_requests": len(test_requests),
            "success_rate": successful_consensus / len(test_requests) if test_requests else 0
        }

        print(f"[+] Consensus test completed in {test_duration:.2f}s - {successful_consensus}/{len(test_requests)} successful")
        return result

    async def run_sharding_test(self) -> Dict[str, Any]:
        """Test knowledge graph sharding and distribution."""
        print("[*] Running sharding test...")

        start_time = time.time()

        # Start federation
        await self._start_test_federation()

        # Add entities through federation
        sharding_results = []
        test_entities = [
            {
                "entity": "FederationTest",
                "fact": "Testing distributed knowledge graph sharding",
                "rels": {"BIZRA": "tests", "Consensus": "validates"}
            },
            {
                "entity": "ShardDistribution",
                "fact": "Entities distributed across federation nodes",
                "rels": {"Decentralization": "demonstrates", "Resilience": "improves"}
            }
        ]

        for entity_data in test_entities:
            # Add through random node
            node_id = list(self.nodes.keys())[0]  # Use first node
            manager = self.nodes[node_id]

            result = await manager.add_knowledge_entity(f"test_{int(time.time())}_{hash(str(entity_data)) % 1000}", entity_data)
            sharding_results.append(result)

        # Check sharding distribution
        sharding_stats = {}
        for node_id, manager in self.nodes.items():
            status = manager.get_federation_status()
            sharding_stats[node_id] = status.get("sharding_stats", {})

        await self._stop_test_federation()

        test_duration = time.time() - start_time

        result = {
            "test": "sharding",
            "duration": test_duration,
            "sharding_results": sharding_results,
            "sharding_stats": sharding_stats,
            "successful_sharding": sum(1 for r in sharding_results if r.get("consensus_result", {}).get("status") == "completed")
        }

        print(f"[+] Sharding test completed in {test_duration:.2f}s")
        return result

    async def run_reasoning_test(self) -> Dict[str, Any]:
        """Test cross-node Graph-of-Thoughts reasoning."""
        print("[*] Running reasoning test...")

        start_time = time.time()

        # Start federation
        await self._start_test_federation()

        # Initiate distributed reasoning
        reasoning_results = []

        test_queries = [
            "How does BIZRA achieve sovereignty?",
            "What role does consensus play in security?",
            "How does decentralization improve resilience?"
        ]

        for query in test_queries:
            # Start reasoning on first node
            manager = self.nodes[list(self.nodes.keys())[0]]
            session_id = await manager.initiate_distributed_reasoning(query)

            # Wait for completion
            await asyncio.sleep(5)  # Give time for reasoning

            # Check status
            status = manager.reasoning_federation.get_session_status(session_id)
            reasoning_results.append({
                "query": query,
                "session_id": session_id,
                "status": status
            })

        await self._stop_test_federation()

        test_duration = time.time() - start_time

        completed_sessions = sum(1 for r in reasoning_results if r.get("status", {}).get("completed", False))

        result = {
            "test": "reasoning",
            "duration": test_duration,
            "reasoning_results": reasoning_results,
            "completed_sessions": completed_sessions,
            "total_sessions": len(test_queries)
        }

        print(f"[+] Reasoning test completed in {test_duration:.2f}s - {completed_sessions}/{len(test_queries)} completed")
        return result

    async def run_failure_recovery_test(self) -> Dict[str, Any]:
        """Test failure recovery and rebalancing."""
        print("[*] Running failure recovery test...")

        start_time = time.time()

        # Start federation
        await self._start_test_federation()

        # Simulate node failure by stopping one node
        failed_node_id = list(self.nodes.keys())[1]
        failed_manager = self.nodes[failed_node_id]

        print(f"[*] Simulating failure of node {failed_node_id}")
        await failed_manager.stop_federation()

        # Wait for recovery
        await asyncio.sleep(15)

        # Check recovery status
        recovery_results = {}
        for node_id, manager in self.nodes.items():
            if node_id != failed_node_id:  # Skip failed node
                status = manager.get_federation_status()
                recovery_results[node_id] = {
                    "health_score": status.get("federation_status", {}).get("health_score", 0),
                    "active_nodes": len(status.get("federation_status", {}).get("active_nodes", [])),
                    "sharding_stats": status.get("sharding_stats", {})
                }

        # Check if system recovered
        avg_health = sum(r["health_score"] for r in recovery_results.values()) / len(recovery_results) if recovery_results else 0

        await self._stop_test_federation()

        test_duration = time.time() - start_time

        result = {
            "test": "failure_recovery",
            "duration": test_duration,
            "failed_node": failed_node_id,
            "recovery_results": recovery_results,
            "average_health_after_failure": avg_health,
            "recovery_success": avg_health > 0.6  # Consider recovered if health > 60%
        }

        print(f"[+] Failure recovery test completed in {test_duration:.2f}s - Recovery success: {result['recovery_success']}")
        return result

    async def _start_test_federation(self):
        """Start all test federation nodes."""
        start_tasks = [manager.start_federation() for manager in self.nodes.values()]

        # Start nodes concurrently
        node_tasks = []
        for manager in self.nodes.values():
            task = asyncio.create_task(manager.start_federation())
            node_tasks.append(task)

        # Let them start
        await asyncio.sleep(3)

    async def _stop_test_federation(self):
        """Stop all test federation nodes."""
        stop_tasks = [manager.stop_federation() for manager in self.nodes.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete federation test suite."""
        print("[+] Starting BIZRA Federation Test Suite")
        print("=" * 50)

        self.test_start_time = time.time()

        # Set up federation
        await self.setup_test_federation()

        # Run all tests
        test_results = {}

        try:
            test_results["connectivity"] = await self.run_connectivity_test()
            test_results["consensus"] = await self.run_consensus_test()
            test_results["sharding"] = await self.run_sharding_test()
            test_results["reasoning"] = await self.run_reasoning_test()
            test_results["failure_recovery"] = await self.run_failure_recovery_test()

        except Exception as e:
            print(f"[!] Test suite failed: {e}")
            test_results["error"] = str(e)

        # Calculate overall results
        total_duration = time.time() - self.test_start_time

        successful_tests = sum(1 for result in test_results.values()
                             if isinstance(result, dict) and not result.get("error") and
                             result.get("overall_success", result.get("success_rate", 0) > 0.8))

        total_tests = len([r for r in test_results.values() if isinstance(r, dict) and "test" in r])

        overall_result = {
            "test_suite": "federation_integration",
            "total_duration": total_duration,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "test_results": test_results,
            "federation_ready": successful_tests >= 4  # Require 4/5 tests to pass
        }

        print("=" * 50)
        print(f"[+] Test Suite Complete: {successful_tests}/{total_tests} tests passed ({overall_result['success_rate']:.1%})")
        print(f"[+] Total duration: {total_duration:.2f}s")
        print(f"[+] Federation Ready: {overall_result['federation_ready']}")

        return overall_result


async def main():
    """Run the federation test suite."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if "--full" in args:
        # Run full test suite
        test_suite = FederationTestSuite()
        results = await test_suite.run_full_test_suite()

        # Save results
        with open("federation_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[+] Test results saved to federation_test_results.json")

    elif "consensus" in args:
        # Run consensus test only
        print("Running federation consensus test...")
        test_suite = FederationTestSuite()
        await test_suite.setup_test_federation()
        result = await test_suite.run_consensus_test()
        print(f"Consensus test result: {result}")

    elif "chaos" in args:
        # Run chaos test only via this runner if needed, though chaos_test.py exists
        print("Running federation chaos resilience test...")
        test_suite = FederationTestSuite()
        await test_suite.setup_test_federation()
        result = await test_suite.run_chaos_resilience_test()
        print(f"Chaos test result: {result}")

    else:
        # Quick connectivity test
        print("Running quick federation connectivity test...")

        test_suite = FederationTestSuite()
        await test_suite.setup_test_federation()
        result = await test_suite.run_connectivity_test()

        print(f"Connectivity test result: {result}")


if __name__ == "__main__":
    asyncio.run(main())