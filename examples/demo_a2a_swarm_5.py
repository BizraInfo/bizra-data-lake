#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — 5-AGENT SWARM DEMO (COMPLEX WORKFLOWS)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Full MAG-5 Team for enterprise-grade multi-agent collaboration:           ║
║   - PRIME: Strategic Orchestrator                                            ║
║   - TEKNE: Technical Implementation                                          ║
║   - LOGOS: Critical Verification                                             ║
║   - GNOSTIC: Knowledge Retrieval                                             ║
║   - AESTHETE: UX & Design                                                    ║
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
# AGENT DEFINITIONS — MAG-5 TEAM
# ═══════════════════════════════════════════════════════════════════════════════

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
    },
    "gnostic": {
        "name": "GNOSTIC",
        "description": "Knowledge Retrieval — Memory and semantic context",
        "capabilities": [
            {"name": "knowledge.retrieve", "type": "knowledge_retrieval", "description": "Retrieve knowledge"},
            {"name": "knowledge.synthesize", "type": "reasoning", "description": "Synthesize information"},
            {"name": "memory.recall", "type": "knowledge_retrieval", "description": "Recall past interactions"}
        ]
    },
    "aesthete": {
        "name": "AESTHETE",
        "description": "UX & Design — User experience and interface design",
        "capabilities": [
            {"name": "design.ui", "type": "design", "description": "Design user interfaces"},
            {"name": "design.ux", "type": "design", "description": "UX optimization"},
            {"name": "format.output", "type": "formatting", "description": "Format and polish outputs"}
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentSwarm:
    """
    Orchestrates a 5-agent MAG team for complex workflows.
    
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
        print("│  SPAWNING MAG-5 AGENT SWARM                                     │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        for agent_id, config in AGENTS.items():
            self.results[agent_id] = []
            
            def make_handlers(aid):
                async def task_handler(task: TaskCard):
                    print(f"  📋 {AGENTS[aid]['name']} processing: {task.prompt[:50]}...")
                    await asyncio.sleep(0.1)
                    result = f"{AGENTS[aid]['name']} completed: {task.capability_required}"
                    self.results[aid].append(result)
                    return result
                
                async def message_handler(msg: A2AMessage):
                    if msg.message_type == MessageType.TASK_REQUEST:
                        task_data = msg.payload.get("task", {})
                        task = TaskCard.from_dict(task_data)
                        print(f"  📥 {AGENTS[aid]['name']} received task: {task.capability_required}")
                        result = await task_handler(task)
                        return None
                    return None
                
                return task_handler, message_handler
            
            task_handler, message_handler = make_handlers(agent_id)
            
            engine = create_a2a_engine(
                agent_id=agent_id,
                name=config["name"],
                description=config["description"],
                capabilities=config["capabilities"],
                on_task_received=task_handler
            )
            
            transport = LocalTransport(agent_id, message_handler)
            await transport.start()
            
            self.engines[agent_id] = engine
            self.transports[agent_id] = transport
            
            print(f"  🤖 Spawned {config['name']}: {len(config['capabilities'])} capabilities")
        
        # Register all agents with PRIME
        prime = self.engines["prime"]
        for agent_id, engine in self.engines.items():
            if agent_id != "prime":
                prime.register_agent(engine.agent_card)
        
        print(f"\n  ✅ MAG-5 Swarm active: {len(self.engines)} agents")
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
        """Execute a multi-agent workflow."""
        print("┌─────────────────────────────────────────────────────────────────┐")
        print(f"│  EXECUTING: {workflow['name']:<49} │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        prime = self.engines["prime"]
        results = []
        
        for i, step in enumerate(workflow["steps"], 1):
            print(f"\n  ═══ Step {i}/{len(workflow['steps'])}: {step['capability']} ═══")
            
            best_agent = prime.find_best_agent(step["capability"])
            
            if not best_agent:
                print(f"  ⚠️ No agent found for: {step['capability']}")
                continue
            
            print(f"  🎯 Routing to: {best_agent.name}")
            
            task = prime.create_task(
                capability=step["capability"],
                prompt=step["prompt"],
                priority=10 - i
            )
            
            msg = prime.create_task_message(task, best_agent.agent_id)
            transport = self.transports["prime"]
            await transport.send(msg, best_agent.agent_id)
            
            await asyncio.sleep(0.2)
            
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
    """Run the 5-agent MAG swarm demo with complex workflow."""
    
    print("\n" + "═" * 70)
    print("  BIZRA A2A — MAG-5 AGENT SWARM DEMO (Complex Workflow)")
    print("═" * 70)
    print()
    
    swarm = AgentSwarm()
    await swarm.spawn_agents()
    
    # Complex 7-step workflow utilizing all 5 agents
    workflow = {
        "name": "Enterprise API Development Pipeline",
        "steps": [
            {
                "capability": "knowledge.retrieve",
                "prompt": "Retrieve best practices for secure Python API development and industry standards"
            },
            {
                "capability": "knowledge.synthesize",
                "prompt": "Synthesize requirements into technical specification"
            },
            {
                "capability": "code.python.generate",
                "prompt": "Generate a secure REST API endpoint with input validation and rate limiting"
            },
            {
                "capability": "code.review",
                "prompt": "Review the generated code for quality, maintainability, and best practices"
            },
            {
                "capability": "verify.security",
                "prompt": "Perform comprehensive security analysis on the generated code"
            },
            {
                "capability": "design.ux",
                "prompt": "Design API response format for optimal developer experience"
            },
            {
                "capability": "verify.ihsan",
                "prompt": "Verify the complete solution meets Ihsān compliance requirements"
            }
        ]
    }
    
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
    print("  📈 MAG-5 Swarm Statistics:")
    for agent_id, engine in swarm.engines.items():
        stats = engine.get_stats()
        tasks_done = len(swarm.results.get(agent_id, []))
        print(f"     {AGENTS[agent_id]['name']}: {stats['my_capabilities']} caps, {tasks_done} tasks completed")
    
    print()
    await swarm.shutdown()
    
    print("\n" + "═" * 70)
    print("  ✅ DEMO COMPLETE — MAG-5 Multi-Agent Collaboration Verified")
    print("═" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
