#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — INTEGRATION TEST SUITE                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Validates Agent-to-Agent protocol implementation:                          ║
║   - Schema validation                                                        ║
║   - Message signing and verification                                         ║
║   - Agent discovery and registration                                         ║
║   - Task delegation and execution                                            ║
║   - Multi-agent communication                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys
sys.path.insert(0, "c:\\BIZRA-DATA-LAKE")

from core.a2a.schema import (
    AgentCard, TaskCard, Capability, CapabilityType,
    A2AMessage, MessageType, TaskStatus,
    create_agent_card, create_task_request
)
from core.a2a.engine import A2AEngine, create_a2a_engine
from core.a2a.tasks import TaskManager, TaskDecomposer
from core.a2a.transport import LocalTransport, HybridTransport


# ═══════════════════════════════════════════════════════════════════════════════
# TEST UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)


def assert_true(condition: bool, message: str) -> bool:
    if condition:
        print(f"  ✅ PASS: {message}")
        return True
    else:
        print(f"  ❌ FAIL: {message}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_capability_creation():
    """Test capability creation and matching."""
    print_header("Capability Creation")
    
    cap = Capability(
        name="code.python.execute",
        type=CapabilityType.CODE_EXECUTION,
        description="Execute Python code in a sandbox",
        parameters={"code": "string", "timeout": "int"},
        ihsan_floor=0.95
    )
    
    passed = 0
    passed += assert_true(cap.name == "code.python.execute", "Name set correctly")
    passed += assert_true(cap.type == CapabilityType.CODE_EXECUTION, "Type set correctly")
    passed += assert_true(cap.matches("python"), "Matches 'python'")
    passed += assert_true(cap.matches("execute"), "Matches 'execute'")
    passed += assert_true(not cap.matches("javascript"), "Does not match 'javascript'")
    
    # Serialization
    d = cap.to_dict()
    cap2 = Capability.from_dict(d)
    passed += assert_true(cap2.name == cap.name, "Serialization roundtrip works")
    
    return passed


def test_agent_card():
    """Test agent card creation and capabilities."""
    print_header("Agent Card")
    
    card = create_agent_card(
        agent_id="agent-001",
        name="Code Executor",
        description="Executes code safely",
        capabilities=[
            {"name": "code.python.execute", "type": "code_execution", "description": "Run Python"},
            {"name": "code.analyze", "type": "reasoning", "description": "Analyze code"}
        ],
        public_key="abc123",
        endpoint="http://localhost:8080"
    )
    
    passed = 0
    passed += assert_true(card.agent_id == "agent-001", "Agent ID set")
    passed += assert_true(len(card.capabilities) == 2, "Two capabilities added")
    passed += assert_true(card.has_capability("code.python.execute"), "Has execute capability")
    passed += assert_true(not card.has_capability("unknown"), "Does not have unknown capability")
    
    # Find capabilities
    found = card.find_capabilities("code")
    passed += assert_true(len(found) == 2, "Found 2 capabilities matching 'code'")
    
    # Card hash
    h = card.card_hash()
    passed += assert_true(len(h) == 32, "Card hash is 32 chars")
    
    return passed


def test_task_card():
    """Test task card lifecycle."""
    print_header("Task Card Lifecycle")
    
    task = create_task_request(
        requester_id="agent-001",
        capability="code.python.execute",
        prompt="Calculate fibonacci(10)",
        parameters={"timeout": 30}
    )
    
    passed = 0
    passed += assert_true(task.status == TaskStatus.PENDING, "Initial status is PENDING")
    passed += assert_true(len(task.task_id) == 36, "Task ID is UUID format")
    
    # Mark started
    task.mark_started("agent-002")
    passed += assert_true(task.status == TaskStatus.IN_PROGRESS, "Status is IN_PROGRESS")
    passed += assert_true(task.assignee_id == "agent-002", "Assignee set")
    passed += assert_true(task.started_at is not None, "Started timestamp set")
    
    # Add artifact
    task.add_artifact("output.txt", "Result: 55", "text/plain")
    passed += assert_true(len(task.artifacts) == 1, "Artifact added")
    
    # Mark completed
    task.mark_completed({"result": 55})
    passed += assert_true(task.status == TaskStatus.COMPLETED, "Status is COMPLETED")
    passed += assert_true(task.result == {"result": 55}, "Result set")
    passed += assert_true(task.completed_at is not None, "Completed timestamp set")
    
    return passed


def test_message_signing():
    """Test A2A message signing and verification."""
    print_header("Message Signing (PCI)")
    
    # Create engine with keypair
    engine = create_a2a_engine(
        agent_id="signer-001",
        name="Signer Agent",
        description="Tests signing",
        capabilities=[]
    )
    
    passed = 0
    
    # Create signed message
    msg = engine.create_message(
        MessageType.PING,
        {"test": "data"}
    )
    
    passed += assert_true(len(msg.signature) > 0, "Signature generated")
    passed += assert_true(msg.sender_id == "signer-001", "Sender ID set")
    
    # Verify signature
    valid = engine.verify_message(msg)
    passed += assert_true(valid, "Signature verifies correctly")
    
    # Tamper with message and verify fails
    msg.payload["test"] = "tampered"
    invalid = engine.verify_message(msg)
    passed += assert_true(not invalid, "Tampered message fails verification")
    
    return passed


async def test_agent_registration():
    """Test agent registration and discovery."""
    print_header("Agent Registration")
    
    engine = create_a2a_engine(
        agent_id="registry-master",
        name="Registry Master",
        description="Manages agents",
        capabilities=[{"name": "orchestration.manage", "type": "orchestration", "description": "Manage agents"}]
    )
    
    passed = 0
    
    # Register another agent
    other_card = create_agent_card(
        agent_id="worker-001",
        name="Worker",
        description="Does work",
        capabilities=[
            {"name": "code.python.execute", "type": "code_execution", "description": "Execute Python"}
        ],
        public_key="xyz789"
    )
    
    result = engine.register_agent(other_card)
    passed += assert_true(result, "Agent registered successfully")
    passed += assert_true(len(engine.registry) == 1, "Registry has 1 agent")
    
    # Find by capability
    found = engine.find_agents_by_capability("code.python.execute")
    passed += assert_true(len(found) == 1, "Found 1 agent with capability")
    passed += assert_true(found[0].agent_id == "worker-001", "Correct agent found")
    
    # Find best agent
    best = engine.find_best_agent("code.python.execute")
    passed += assert_true(best is not None, "Best agent found")
    passed += assert_true(best.agent_id == "worker-001", "Best agent is worker-001")
    
    # Reject low Ihsān agent
    low_ihsan_card = create_agent_card(
        agent_id="bad-agent",
        name="Bad Agent",
        description="Low integrity",
        capabilities=[],
        public_key=""
    )
    low_ihsan_card.ihsan_score = 0.5
    
    result = engine.register_agent(low_ihsan_card)
    passed += assert_true(not result, "Low Ihsān agent rejected")
    passed += assert_true(len(engine.registry) == 1, "Registry unchanged")
    
    return passed


async def test_task_delegation():
    """Test task creation and delegation."""
    print_header("Task Delegation")
    
    # Track received tasks
    received_tasks = []
    
    async def handle_task(task: TaskCard):
        received_tasks.append(task)
        return {"computed": task.prompt}
    
    # Create orchestrator
    orchestrator = create_a2a_engine(
        agent_id="orchestrator",
        name="Orchestrator",
        description="Coordinates work",
        capabilities=[{"name": "orchestration.delegate", "type": "orchestration", "description": "Delegate"}]
    )
    
    # Create worker
    worker = create_a2a_engine(
        agent_id="worker",
        name="Worker",
        description="Executes tasks",
        capabilities=[{"name": "code.execute", "type": "code_execution", "description": "Execute"}],
        on_task_received=handle_task
    )
    
    passed = 0
    
    # Register worker with orchestrator
    orchestrator.register_agent(worker.agent_card)
    
    # Create task
    task = orchestrator.create_task(
        capability="code.execute",
        prompt="Run my code",
        priority=8
    )
    
    passed += assert_true(task.task_id in orchestrator.pending_tasks, "Task in pending")
    passed += assert_true(task.priority == 8, "Priority set")
    
    # Create task message
    msg = orchestrator.create_task_message(task, "worker")
    passed += assert_true(msg.message_type == MessageType.TASK_REQUEST, "Message type is TASK_REQUEST")
    passed += assert_true("task" in msg.payload, "Task in payload")
    
    # Worker handles the task
    response = await worker.handle_message(msg)
    passed += assert_true(response is not None, "Worker responded")
    passed += assert_true(response.message_type == MessageType.TASK_ACCEPT, "Worker accepted task")
    
    # Wait for task execution
    await asyncio.sleep(0.1)
    passed += assert_true(len(received_tasks) == 1, "Task was executed")
    
    return passed


async def test_local_transport():
    """Test local in-memory transport."""
    print_header("Local Transport")
    
    messages_received = []
    
    async def handler_a(msg: A2AMessage):
        messages_received.append(("A", msg))
        return None
    
    async def handler_b(msg: A2AMessage):
        messages_received.append(("B", msg))
        return None
    
    # Clear any existing bus
    LocalTransport.clear_bus()
    
    # Create transports
    transport_a = LocalTransport("agent-A", handler_a)
    transport_b = LocalTransport("agent-B", handler_b)
    
    await transport_a.start()
    await transport_b.start()
    
    passed = 0
    
    # Send from A to B
    msg = A2AMessage(
        message_type=MessageType.PING,
        sender_id="agent-A",
        recipient_id="agent-B",
        payload={"test": "hello"}
    )
    
    result = await transport_a.send(msg, "agent-B")
    passed += assert_true(result, "Send succeeded")
    passed += assert_true(len(messages_received) == 1, "Message received")
    passed += assert_true(messages_received[0][0] == "B", "Received by B")
    
    # Broadcast from A
    messages_received.clear()
    broadcast_count = await transport_a.broadcast(msg)
    passed += assert_true(broadcast_count == 1, "Broadcast reached 1 agent")
    passed += assert_true(len(messages_received) == 1, "Broadcast message received")
    
    await transport_a.stop()
    await transport_b.stop()
    
    LocalTransport.clear_bus()
    
    return passed


async def test_task_manager():
    """Test task manager execution."""
    print_header("Task Manager")
    
    results = []
    
    async def executor(task: TaskCard):
        await asyncio.sleep(0.05)  # Simulate work
        results.append(task.task_id)
        return f"Completed: {task.prompt}"
    
    manager = TaskManager(max_concurrent=2)
    
    passed = 0
    
    # Submit tasks
    task1 = TaskCard(prompt="Task 1", priority=5)
    task2 = TaskCard(prompt="Task 2", priority=10)  # Higher priority
    task3 = TaskCard(prompt="Task 3", priority=1)
    
    manager.submit(task1)
    manager.submit(task2)
    manager.submit(task3)
    
    passed += assert_true(len(manager.queue) == 3, "3 tasks queued")
    
    # Check priority ordering
    passed += assert_true(manager.queue[0].task.priority == 10, "Highest priority first")
    
    # Execute tasks
    for entry in list(manager.queue):
        await manager.execute(entry.task, executor)
    
    passed += assert_true(len(results) == 3, "All 3 tasks executed")
    passed += assert_true(manager.stats["completed"] == 3, "Stats show 3 completed")
    
    # Check task status
    passed += assert_true(task1.status == TaskStatus.COMPLETED, "Task 1 completed")
    passed += assert_true(task2.status == TaskStatus.COMPLETED, "Task 2 completed")
    
    return passed


async def test_multi_agent_communication():
    """Test multi-agent communication flow."""
    print_header("Multi-Agent Communication")
    
    LocalTransport.clear_bus()
    
    # Create 3 agents
    messages = {"orchestrator": [], "coder": [], "reviewer": []}
    
    async def make_handler(name):
        async def handler(msg: A2AMessage):
            messages[name].append(msg)
            return None
        return handler
    
    # Create engines
    orchestrator = create_a2a_engine(
        "orchestrator", "Orchestrator", "Coordinates",
        [{"name": "orchestrate", "type": "orchestration", "description": "Orchestrate"}]
    )
    
    coder = create_a2a_engine(
        "coder", "Coder", "Writes code",
        [{"name": "code.write", "type": "code_generation", "description": "Write code"}]
    )
    
    reviewer = create_a2a_engine(
        "reviewer", "Reviewer", "Reviews code",
        [{"name": "code.review", "type": "reasoning", "description": "Review code"}]
    )
    
    # Create transports
    t_orch = LocalTransport("orchestrator", await make_handler("orchestrator"))
    t_code = LocalTransport("coder", await make_handler("coder"))
    t_rev = LocalTransport("reviewer", await make_handler("reviewer"))
    
    await t_orch.start()
    await t_code.start()
    await t_rev.start()
    
    passed = 0
    
    # Register agents with orchestrator
    orchestrator.register_agent(coder.agent_card)
    orchestrator.register_agent(reviewer.agent_card)
    passed += assert_true(len(orchestrator.registry) == 2, "2 agents registered")
    
    # Orchestrator sends discovery
    discover_msg = orchestrator.create_discover_message()
    count = await t_orch.broadcast(discover_msg)
    passed += assert_true(count == 2, "Discovery sent to 2 agents")
    
    # Check messages received
    await asyncio.sleep(0.1)
    passed += assert_true(len(messages["coder"]) == 1, "Coder received discovery")
    passed += assert_true(len(messages["reviewer"]) == 1, "Reviewer received discovery")
    
    # Coder announces
    announce_msg = coder.create_announce_message()
    await t_code.send(announce_msg, "orchestrator")
    
    await asyncio.sleep(0.1)
    passed += assert_true(len(messages["orchestrator"]) == 1, "Orchestrator received announcement")
    
    # Cleanup
    await t_orch.stop()
    await t_code.stop()
    await t_rev.stop()
    LocalTransport.clear_bus()
    
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def run_all_tests():
    """Run all A2A tests."""
    print("\n" + "═" * 70)
    print("  BIZRA A2A PROTOCOL — INTEGRATION TEST SUITE")
    print("═" * 70)
    
    total = 0
    passed = 0
    
    # Sync tests
    passed += test_capability_creation()
    total += 6
    
    passed += test_agent_card()
    total += 6
    
    passed += test_task_card()
    total += 8
    
    passed += test_message_signing()
    total += 4
    
    # Async tests
    passed += await test_agent_registration()
    total += 8
    
    passed += await test_task_delegation()
    total += 7
    
    passed += await test_local_transport()
    total += 5
    
    passed += await test_task_manager()
    total += 6
    
    passed += await test_multi_agent_communication()
    total += 6
    
    print("\n" + "═" * 70)
    print(f"  RESULTS: {passed}/{total} assertions passed")
    if passed == total:
        print("  ✅ ALL TESTS PASSED")
    else:
        print(f"  ❌ {total - passed} FAILURES")
    print("═" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
