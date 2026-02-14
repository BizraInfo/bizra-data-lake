#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — MULTI-AGENT SWARM DEMO                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Demonstrates elite multi-agent collaboration:                              ║
║   - Orchestrator agent coordinates work                                      ║
║   - Specialist agents handle domain tasks                                    ║
║   - PCI-signed messages for trust                                            ║
║   - Task delegation and result aggregation                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys
sys.path.insert(0, "c:\\BIZRA-DATA-LAKE")

from core.a2a.schema import (
    TaskCard, TaskStatus, MessageType, A2AMessage
)
from core.a2a.engine import A2AEngine, create_a2a_engine
from core.a2a.transport import LocalTransport


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Specialized agents with distinct capabilities
AGENTS = {
    "prime": {
        "name": "PRIME",
        "description": "Strategic Orchestrator — Coordinates multi-agent workflows",
        "capabilities": [
            {"name": "orchestrate.coordinate", "type": "orchestration", "description": "Coordinate agents"},
            {"name": "orchestrate.delegate", "type": "orchestration", "description": "Delegate tasks"},
            {"name": "reasoning.strategic", "type": "reasoning", "description": "Strategic planning"}
        ]
    },
    "tekne": {
        "name": "TEKNE",
        "description": "Technical Implementation — Code and system operations",
        "capabilities": [
            {"name": "code.python.generate", "type": "code_generation", "description": "Generate Python code"},
            {"name": "code.python.execute", "type": "code_execution", "description": "Execute Python code"},
            {"name": "code.review", "type": "reasoning", "description": "Review code quality"}
        ]
    },
    "logos": {
        "name": "LOGOS",
        "description": "Critical Analysis — Verification and validation",
        "capabilities": [
            {"name": "verify.logic", "type": "reasoning", "description": "Verify logical correctness"},
            {"name": "verify.security", "type": "security", "description": "Security analysis"},
            {"name": "verify.ihsan", "type": "reasoning", "description": "Ihsān compliance check"}
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentSwarm:
    """
    Orchestrates a swarm of specialized agents.
    
    Features:
    - Agent lifecycle management
    - Task routing based on capabilities
    - Result aggregation
    - Collaborative workflows
    """
    
    def __init__(self):
        self.engines: dict[str, A2AEngine] = {}
        self.transports: dict[str, LocalTransport] = {}
        self.results: dict[str, list] = {}
        
    async def spawn_agents(self):
        """Spawn all agents in the swarm."""
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│  SPAWNING AGENT SWARM                                          │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        for agent_id, config in AGENTS.items():
            # Create handler for this agent
            self.results[agent_id] = []
            
            # Use closures properly to capture agent_id
            def make_handlers(aid):
                async def task_handler(task: TaskCard):
                    print(f"  📋 {AGENTS[aid]['name']} processing: {task.prompt[:50]}...")
                    await asyncio.sleep(0.1)  # Simulate work
                    result = f"{AGENTS[aid]['name']} completed: {task.capability_required}"
                    self.results[aid].append(result)
                    return result
                
                async def message_handler(msg: A2AMessage):
                    if msg.message_type == MessageType.TASK_REQUEST:
                        task_data = msg.payload.get("task", {})
                        task = TaskCard.from_dict(task_data)
                        print(f"  📥 {AGENTS[aid]['name']} received task: {task.capability_required}")
                        # Actually execute the task
                        result = await task_handler(task)
                        return None
                    return None
                
                return task_handler, message_handler
            
            task_handler, message_handler = make_handlers(agent_id)
            
            # Create engine
            engine = create_a2a_engine(
                agent_id=agent_id,
                name=config["name"],
                description=config["description"],
                capabilities=config["capabilities"],
                on_task_received=task_handler
            )
            
            # Create transport with message handler
            transport = LocalTransport(agent_id, message_handler)
            await transport.start()
            
            self.engines[agent_id] = engine
            self.transports[agent_id] = transport
            
            print(f"  🤖 Spawned {config['name']}: {len(config['capabilities'])} capabilities")
        
        # Register all agents with PRIME (orchestrator)
        prime = self.engines["prime"]
        for agent_id, engine in self.engines.items():
            if agent_id != "prime":
                prime.register_agent(engine.agent_card)
        
        print(f"\n  ✅ Swarm active: {len(self.engines)} agents")
        print()
    
    async def shutdown(self):
        """Shutdown all agents."""
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│  SHUTTING DOWN SWARM                                           │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        for agent_id, transport in self.transports.items():
            await transport.stop()
        
        LocalTransport.clear_bus()
        print("  ✅ All agents stopped")
    
    async def execute_workflow(self, workflow: dict):
        """
        Execute a multi-agent workflow.
        
        Workflow format:
        {
            "name": "workflow name",
            "steps": [
                {"capability": "code.python.generate", "prompt": "..."},
                {"capability": "verify.logic", "prompt": "..."},
            ]
        }
        """
        print("┌─────────────────────────────────────────────────────────────────┐")
        print(f"│  EXECUTING WORKFLOW: {workflow['name']:<37} │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        prime = self.engines["prime"]
        results = []
        
        for i, step in enumerate(workflow["steps"], 1):
            print(f"\n  ═══ Step {i}/{len(workflow['steps'])}: {step['capability']} ═══")
            
            # Find best agent for this capability
            best_agent = prime.find_best_agent(step["capability"])
            
            if not best_agent:
                print(f"  ⚠️ No agent found for: {step['capability']}")
                continue
            
            print(f"  🎯 Routing to: {best_agent.name}")
            
            # Create task
            task = prime.create_task(
                capability=step["capability"],
                prompt=step["prompt"],
                priority=10 - i  # Earlier steps higher priority
            )
            
            # Create and send message
            msg = prime.create_task_message(task, best_agent.agent_id)
            transport = self.transports["prime"]
            await transport.send(msg, best_agent.agent_id)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Get result from target agent
            target_results = self.results.get(best_agent.agent_id, [])
            if target_results:
                result = target_results[-1]
                results.append({"step": i, "agent": best_agent.name, "result": result})
                print(f"  ✅ Result: {result}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo():
    """Run the multi-agent swarm demo."""
    
    print("\n" + "═" * 70)
    print("  BIZRA A2A — MULTI-AGENT SWARM DEMO")
    print("═" * 70)
    print()
    
    # Create swarm
    swarm = AgentSwarm()
    await swarm.spawn_agents()
    
    # Define a complex workflow
    workflow = {
        "name": "Secure Code Generation",
        "steps": [
            {
                "capability": "code.python.generate",
                "prompt": "Generate a secure REST API endpoint with input validation"
            },
            {
                "capability": "code.review",
                "prompt": "Review the generated code for quality and best practices"
            },
            {
                "capability": "verify.security",
                "prompt": "Perform security analysis on the generated code"
            },
            {
                "capability": "verify.ihsan",
                "prompt": "Verify the code meets Ihsān compliance requirements"
            }
        ]
    }
    
    # Execute workflow
    results = await swarm.execute_workflow(workflow)
    
    # Summary
    print("\n" + "═" * 70)
    print("  WORKFLOW COMPLETE")
    print("═" * 70)
    print()
    print("  📊 Results Summary:")
    for r in results:
        print(f"     Step {r['step']}: {r['agent']} → {r['result']}")
    
    print()
    print("  📈 Swarm Statistics:")
    for agent_id, engine in swarm.engines.items():
        stats = engine.get_stats()
        print(f"     {AGENTS[agent_id]['name']}: {stats['my_capabilities']} caps, {len(swarm.results.get(agent_id, []))} tasks completed")
    
    # Shutdown
    print()
    await swarm.shutdown()
    
    print("\n" + "═" * 70)
    print("  ✅ DEMO COMPLETE — Multi-Agent Collaboration Verified")
    print("═" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
