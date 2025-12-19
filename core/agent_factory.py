#!/usr/bin/env python3
"""
BIZRA Agent Factory - PAT/SAT Instantiation
=============================================
Fixes F-ARCH-002: Static Agents

Creates agents with:
- Persistent state and memory
- URP-managed resource allocation
- FATE-gated operations
- Session continuity

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      AGENT FACTORY                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Spawn Request ──▶ [URP Acquire] ──▶ [Agent Create]        │
    │                          │                 │                 │
    │                          │                 ▼                 │
    │                    OverCapacityError   Agent Instance        │
    │                                            │                 │
    │                       ┌────────────────────┴──────┐          │
    │                       │                           │          │
    │                       ▼                           ▼          │
    │              [Session Memory]           [FATE Integration]   │
    │                       │                           │          │
    │                       └───────────┬───────────────┘          │
    │                                   │                          │
    │                                   ▼                          │
    │                           Operational Agent                  │
    │                                   │                          │
    │                           [Registry Track]                   │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

Endpoint: /v1/system/spawn
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
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("agent.factory")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
MAX_MEMORY_TURNS = int(os.getenv("BIZRA_MAX_MEMORY_TURNS", "20"))
EVIDENCE_PATH = Path(os.getenv("BIZRA_AGENT_EVIDENCE", "docs/evidence/agents"))


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentType(Enum):
    PAT = "PAT"  # Personal Agentic Team
    SAT = "SAT"  # System Agentic Team


class AgentStatus(Enum):
    SPAWNING = "SPAWNING"
    READY = "READY"
    BUSY = "BUSY"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"


# PAT Agent specifications
PAT_SPECIFICATIONS = {
    "MasterReasoner": {
        "model": "deepseek-r1:7b",
        "backend": "ollama",
        "vram_gb": 4.5,
        "role": "Strategic thinking, complex analysis, decision synthesis",
        "system_prompt": """You are MasterReasoner, the strategic thinking core of the BIZRA system.
Your role: Complex analysis, multi-perspective reasoning, decision synthesis.
Principles: Ihsān (excellence), Adl (justice), Amānah (trust).
Always explain your reasoning chain. Cite evidence when available.""",
    },
    "MemoryArchitect": {
        "model": "qwen2.5:7b",
        "backend": "ollama",
        "vram_gb": 4.0,
        "role": "Knowledge organization, recall, context management",
        "system_prompt": """You are MemoryArchitect, the knowledge management core of BIZRA.
Your role: Organize information, ensure recall accuracy, maintain context.
You have access to the session memory and can reference prior conversations.
Structure your responses to be easily retrievable later.""",
    },
    "CreativeSynthesizer": {
        "model": "qwen2.5:7b",
        "backend": "ollama",
        "vram_gb": 4.0,
        "role": "Writing, ideation, creative problem-solving",
        "system_prompt": """You are CreativeSynthesizer, the creative core of BIZRA.
Your role: Generate ideas, write content, solve problems creatively.
Balance creativity with practicality. Push boundaries while respecting constraints.""",
    },
    "DataAnalyzer": {
        "model": "mistral:7b",
        "backend": "ollama",
        "vram_gb": 4.0,
        "role": "Data analysis, pattern recognition, insights extraction",
        "system_prompt": """You are DataAnalyzer, the analytical core of BIZRA.
Your role: Analyze data, find patterns, extract actionable insights.
Be precise with numbers. Acknowledge uncertainty. Suggest next analyses.""",
    },
    "Communicator": {
        "model": "mistral:7b",
        "backend": "ollama",
        "vram_gb": 4.0,
        "role": "External communications, presentations, messaging",
        "system_prompt": """You are Communicator, the external voice of BIZRA.
Your role: Craft clear messages, presentations, professional communications.
Adapt tone to audience. Be concise. Ensure clarity over complexity.""",
    },
    "ExecutionPlanner": {
        "model": "agentflow-7b",
        "backend": "lmstudio",
        "vram_gb": 4.0,
        "role": "Task planning, scheduling, workflow orchestration",
        "system_prompt": """You are ExecutionPlanner, the operational core of BIZRA.
Your role: Break down goals into tasks, create schedules, manage workflows.
Use numbered steps. Include time estimates. Identify dependencies.""",
    },
    "EthicsGuardian": {
        "model": "qwen2.5:7b",
        "backend": "ollama",
        "vram_gb": 4.0,
        "role": "Safety, bias detection, Ihsān compliance",
        "system_prompt": """You are EthicsGuardian, the moral compass of BIZRA.
Your role: Review actions for safety, detect bias, ensure Ihsān compliance.
Flag concerns clearly. Suggest ethical alternatives. Protect stakeholders.""",
    },
}

# SAT Agent specifications (rule-based, minimal resources)
SAT_SPECIFICATIONS = {
    "PoiVerifier": {
        "type": "rule-based",
        "vram_gb": 0.1,
        "role": "Validate Proof-of-Impact attestations",
    },
    "ResourceAllocator": {
        "type": "rule-based",
        "vram_gb": 0.1,
        "role": "Optimize compute/memory allocation",
    },
    "RiskGuardian": {
        "type": "rule-based",
        "vram_gb": 0.1,
        "role": "Security monitoring, threat detection",
    },
    "GovernanceEngine": {
        "type": "rule-based",
        "vram_gb": 0.1,
        "role": "Policy enforcement, parameter updates",
    },
    "EvidenceEngine": {
        "type": "rule-based",
        "vram_gb": 0.1,
        "role": "Audit trails, receipt generation",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTurn:
    """A single conversation turn."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    tokens: int = 0


@dataclass
class SessionMemory:
    """Conversation memory for an agent session."""
    session_id: str
    agent_id: str
    turns: List[MemoryTurn] = field(default_factory=list)
    max_turns: int = MAX_MEMORY_TURNS
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def add_turn(self, role: str, content: str, tokens: int = 0) -> None:
        """Add a conversation turn."""
        self.turns.append(MemoryTurn(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokens=tokens
        ))
        
        # Trim to max turns (keep system + recent)
        if len(self.turns) > self.max_turns:
            # Keep first turn (usually system) and recent turns
            self.turns = [self.turns[0]] + self.turns[-(self.max_turns - 1):]
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to chat completion format."""
        return [{"role": t.role, "content": t.content} for t in self.turns]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "turns": [{"role": t.role, "content": t.content, "timestamp": t.timestamp} for t in self.turns],
            "created_at": self.created_at
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentInstance:
    """A spawned agent instance."""
    agent_id: str
    instance_id: str
    agent_type: AgentType
    name: str
    status: AgentStatus
    lease_id: Optional[str]
    session: SessionMemory
    spec: Dict[str, Any]
    spawned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_active: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_count: int = 0
    error_count: int = 0
    
    def update_activity(self) -> None:
        """Update last active timestamp."""
        self.last_active = datetime.now(timezone.utc).isoformat()
        self.request_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state."""
        return {
            "agent_id": self.agent_id,
            "instance_id": self.instance_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "lease_id": self.lease_id,
            "session_id": self.session.session_id,
            "spawned_at": self.spawned_at,
            "last_active": self.last_active,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "spec": {
                "model": self.spec.get("model"),
                "backend": self.spec.get("backend"),
                "role": self.spec.get("role"),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class AgentFactory:
    """
    Factory for creating and managing agent instances.
    
    Integrates with:
    - URP: Resource allocation
    - FATE: Ethics gating
    - Session Memory: Context persistence
    """
    
    _instance: Optional['AgentFactory'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'AgentFactory':
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._agents: Dict[str, AgentInstance] = {}
        self._sessions: Dict[str, SessionMemory] = {}
        self._agent_lock = threading.Lock()
        
        # Import URP if available
        self._urp = None
        try:
            from core.urp import URPManager
            self._urp = URPManager()
            logger.info("URP integration enabled")
        except ImportError:
            logger.warning("URP not available - running without resource management")
        
        # Import FATE if available
        self._fate = None
        try:
            from core.fate import get_fate_engine
            self._fate = get_fate_engine()
            logger.info("FATE integration enabled")
        except ImportError:
            logger.warning("FATE not available - running without ethics gate")
        
        # Import Synapse for A2A communication
        self._synapse_enabled = False
        self._agent_buses: Dict[str, Any] = {}  # agent_id -> SynapseBus
        try:
            from core.synapse import create_synapse_for_agent, get_synapse
            self._synapse_factory = create_synapse_for_agent
            self._synapse_conn = get_synapse()
            if self._synapse_conn.connect():
                self._synapse_enabled = True
                logger.info("Synapse integration enabled (Trinity connected)")
            else:
                logger.warning("Synapse connection failed - running without A2A")
        except ImportError:
            self._synapse_factory = None
            self._synapse_conn = None
            logger.warning("Synapse not available - running without A2A communication")
        
        # Create evidence directory
        EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.info("Agent Factory initialized")
    
    def _generate_ids(self) -> tuple[str, str, str]:
        """Generate unique IDs for agent, instance, and session."""
        agent_id = f"agent-{uuid4().hex[:8]}"
        instance_id = f"inst-{uuid4().hex[:8]}"
        session_id = f"sess-{uuid4().hex[:12]}"
        return agent_id, instance_id, session_id
    
    def _acquire_resources(self, name: str, vram_gb: float) -> Optional[str]:
        """Acquire URP lease for agent."""
        if self._urp is None:
            return None
        
        try:
            from core.urp import ResourceRequest
            request = ResourceRequest(agent_id=name, vram_gb=vram_gb)
            lease = self._urp.acquire(request)
            return lease.lease_id
        except Exception as e:
            logger.error(f"Failed to acquire resources for {name}: {e}")
            raise
    
    def _release_resources(self, lease_id: str) -> None:
        """Release URP lease."""
        if self._urp is None or lease_id is None:
            return
        
        try:
            self._urp.release(lease_id)
        except Exception as e:
            logger.warning(f"Failed to release lease {lease_id}: {e}")
    
    def spawn_pat(self, name: str, session_id: Optional[str] = None) -> AgentInstance:
        """
        Spawn a PAT (Personal Agentic Team) agent.
        
        Args:
            name: Agent name (e.g., "MasterReasoner")
            session_id: Optional existing session to resume
            
        Returns:
            AgentInstance ready for use
            
        Raises:
            ValueError: If agent name not found
            OverCapacityError: If resources unavailable
        """
        if name not in PAT_SPECIFICATIONS:
            raise ValueError(f"Unknown PAT agent: {name}")
        
        spec = PAT_SPECIFICATIONS[name]
        
        with self._agent_lock:
            # Check if agent already spawned
            for agent in self._agents.values():
                if agent.name == name and agent.status == AgentStatus.READY:
                    logger.info(f"Reusing existing {name} instance: {agent.instance_id}")
                    return agent
            
            # Generate IDs
            agent_id, instance_id, new_session_id = self._generate_ids()
            
            # Acquire resources
            lease_id = self._acquire_resources(name, spec["vram_gb"])
            
            # Create or resume session
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                logger.info(f"Resuming session: {session_id}")
            else:
                session = SessionMemory(
                    session_id=new_session_id,
                    agent_id=agent_id
                )
                # Add system prompt
                session.add_turn("system", spec["system_prompt"])
                self._sessions[session.session_id] = session
            
            # Create agent instance
            agent = AgentInstance(
                agent_id=agent_id,
                instance_id=instance_id,
                agent_type=AgentType.PAT,
                name=name,
                status=AgentStatus.READY,
                lease_id=lease_id,
                session=session,
                spec=spec
            )
            
            self._agents[agent_id] = agent
            
            # Connect to Synapse for A2A communication
            if self._synapse_enabled and self._synapse_factory:
                try:
                    bus = self._synapse_factory(agent_id, name, "PAT")
                    capabilities = [spec.get("role", "general")]
                    bus.connect(capabilities)
                    self._agent_buses[agent_id] = bus
                    logger.info(f"Agent {name} connected to Trinity Synapse")
                except Exception as e:
                    logger.warning(f"Synapse connection failed for {name}: {e}")
            
            logger.info(f"Spawned PAT agent: {name} ({instance_id})")
            self._record_spawn(agent)
            
            return agent
    
    def spawn_sat(self, name: str) -> AgentInstance:
        """
        Spawn a SAT (System Agentic Team) agent.
        
        SAT agents are rule-based and require minimal resources.
        """
        if name not in SAT_SPECIFICATIONS:
            raise ValueError(f"Unknown SAT agent: {name}")
        
        spec = SAT_SPECIFICATIONS[name]
        
        with self._agent_lock:
            # Check if already spawned
            for agent in self._agents.values():
                if agent.name == name and agent.status == AgentStatus.READY:
                    return agent
            
            agent_id, instance_id, session_id = self._generate_ids()
            
            # Minimal resource allocation
            lease_id = self._acquire_resources(name, spec["vram_gb"])
            
            session = SessionMemory(session_id=session_id, agent_id=agent_id)
            
            agent = AgentInstance(
                agent_id=agent_id,
                instance_id=instance_id,
                agent_type=AgentType.SAT,
                name=name,
                status=AgentStatus.READY,
                lease_id=lease_id,
                session=session,
                spec=spec
            )
            
            self._agents[agent_id] = agent
            
            # Connect to Synapse for A2A communication
            if self._synapse_enabled and self._synapse_factory:
                try:
                    bus = self._synapse_factory(agent_id, name, "SAT")
                    capabilities = [spec.get("role", "system")]
                    bus.connect(capabilities)
                    self._agent_buses[agent_id] = bus
                    logger.info(f"SAT agent {name} connected to Trinity Synapse")
                except Exception as e:
                    logger.warning(f"Synapse connection failed for {name}: {e}")
            
            logger.info(f"Spawned SAT agent: {name} ({instance_id})")
            
            return agent
    
    def terminate(self, agent_id: str) -> bool:
        """Terminate an agent and release resources."""
        with self._agent_lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            agent.status = AgentStatus.TERMINATED
            
            # Disconnect from Synapse
            if agent_id in self._agent_buses:
                try:
                    self._agent_buses[agent_id].disconnect()
                    del self._agent_buses[agent_id]
                except Exception as e:
                    logger.warning(f"Synapse disconnect failed: {e}")
            
            # Release URP lease
            if agent.lease_id:
                self._release_resources(agent.lease_id)
            
            # Keep in registry for audit but mark terminated
            logger.info(f"Terminated agent: {agent.name} ({agent.instance_id})")
            self._record_termination(agent)
            
            return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[AgentInstance]:
        """Get active agent by name."""
        for agent in self._agents.values():
            if agent.name == name and agent.status == AgentStatus.READY:
                return agent
        return None
    
    def list_agents(self, include_terminated: bool = False) -> List[AgentInstance]:
        """List all agents."""
        if include_terminated:
            return list(self._agents.values())
        return [a for a in self._agents.values() if a.status != AgentStatus.TERMINATED]
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def get_synapse_bus(self, agent_id: str) -> Optional[Any]:
        """Get synapse bus for agent A2A communication."""
        return self._agent_buses.get(agent_id)
    
    def snapshot(self) -> Dict[str, Any]:
        """Get factory state snapshot."""
        with self._agent_lock:
            active = [a for a in self._agents.values() if a.status == AgentStatus.READY]
            pat_count = sum(1 for a in active if a.agent_type == AgentType.PAT)
            sat_count = sum(1 for a in active if a.agent_type == AgentType.SAT)
            
            # Get synapse health
            synapse_health = None
            if self._synapse_conn:
                try:
                    synapse_health = self._synapse_conn.health_check()
                except Exception:
                    synapse_health = {"status": "error"}
            
            return {
                "total_agents": len(self._agents),
                "active_agents": len(active),
                "pat_agents": pat_count,
                "sat_agents": sat_count,
                "sessions": len(self._sessions),
                "synapse_connections": len(self._agent_buses),
                "urp_enabled": self._urp is not None,
                "fate_enabled": self._fate is not None,
                "synapse_enabled": self._synapse_enabled,
                "synapse_health": synapse_health,
                "agents": [a.to_dict() for a in active]
            }
    
    def _record_spawn(self, agent: AgentInstance) -> None:
        """Record spawn event."""
        log_file = EVIDENCE_PATH / "spawn_events.jsonl"
        record = {
            "event": "SPAWN",
            "timestamp": agent.spawned_at,
            "agent_id": agent.agent_id,
            "instance_id": agent.instance_id,
            "name": agent.name,
            "type": agent.agent_type.value,
            "lease_id": agent.lease_id
        }
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Failed to record spawn: {e}")
    
    def _record_termination(self, agent: AgentInstance) -> None:
        """Record termination event."""
        log_file = EVIDENCE_PATH / "spawn_events.jsonl"
        record = {
            "event": "TERMINATE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent.agent_id,
            "instance_id": agent.instance_id,
            "name": agent.name,
            "request_count": agent.request_count,
            "error_count": agent.error_count
        }
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Failed to record termination: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_factory() -> AgentFactory:
    """Get or create the global agent factory."""
    return AgentFactory()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the Agent Factory."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Agent Factory")
    parser.add_argument("--spawn", type=str, metavar="NAME", help="Spawn agent by name")
    parser.add_argument("--terminate", type=str, metavar="ID", help="Terminate agent by ID")
    parser.add_argument("--list", action="store_true", help="List all agents")
    parser.add_argument("--status", action="store_true", help="Show factory status")
    parser.add_argument("--test", action="store_true", help="Run test scenario")
    
    args = parser.parse_args()
    factory = get_factory()
    
    if args.spawn:
        try:
            if args.spawn in PAT_SPECIFICATIONS:
                agent = factory.spawn_pat(args.spawn)
            elif args.spawn in SAT_SPECIFICATIONS:
                agent = factory.spawn_sat(args.spawn)
            else:
                print(f"❌ Unknown agent: {args.spawn}")
                print(f"PAT agents: {', '.join(PAT_SPECIFICATIONS.keys())}")
                print(f"SAT agents: {', '.join(SAT_SPECIFICATIONS.keys())}")
                return
            
            print(f"✅ Spawned {agent.agent_type.value} agent: {agent.name}")
            print(f"   Agent ID: {agent.agent_id}")
            print(f"   Instance: {agent.instance_id}")
            print(f"   Session: {agent.session.session_id}")
            print(f"   Lease: {agent.lease_id or 'None (URP disabled)'}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    elif args.terminate:
        if factory.terminate(args.terminate):
            print(f"✅ Terminated agent: {args.terminate}")
        else:
            print(f"❌ Agent not found: {args.terminate}")
    
    elif args.list:
        agents = factory.list_agents()
        print("\n" + "═" * 60)
        print("  ACTIVE AGENTS")
        print("═" * 60)
        if not agents:
            print("  No active agents")
        else:
            for agent in agents:
                print(f"  {agent.agent_type.value}: {agent.name}")
                print(f"    ID: {agent.agent_id}")
                print(f"    Status: {agent.status.value}")
                print(f"    Requests: {agent.request_count}")
                print()
        print("═" * 60 + "\n")
    
    elif args.status:
        snap = factory.snapshot()
        print("\n" + "═" * 60)
        print("  AGENT FACTORY STATUS")
        print("═" * 60)
        print(f"  Total Agents: {snap['total_agents']}")
        print(f"  Active: {snap['active_agents']}")
        print(f"  PAT: {snap['pat_agents']}")
        print(f"  SAT: {snap['sat_agents']}")
        print(f"  Sessions: {snap['sessions']}")
        print(f"  URP: {'✅' if snap['urp_enabled'] else '❌'}")
        print(f"  FATE: {'✅' if snap['fate_enabled'] else '❌'}")
        print("═" * 60 + "\n")
    
    elif args.test:
        print("\n🧪 AGENT FACTORY TEST\n")
        
        # Spawn PAT agents
        print("1. Spawning PAT agents...")
        try:
            mr = factory.spawn_pat("MasterReasoner")
            print(f"   ✅ MasterReasoner: {mr.instance_id}")
            
            eg = factory.spawn_pat("EthicsGuardian")
            print(f"   ✅ EthicsGuardian: {eg.instance_id}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Spawn SAT agent
        print("\n2. Spawning SAT agents...")
        try:
            ee = factory.spawn_sat("EvidenceEngine")
            print(f"   ✅ EvidenceEngine: {ee.instance_id}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Show status
        print("\n3. Factory status:")
        snap = factory.snapshot()
        print(f"   Active: {snap['active_agents']}")
        print(f"   PAT: {snap['pat_agents']}, SAT: {snap['sat_agents']}")
        
        # Terminate
        print("\n4. Terminating agents...")
        for agent in factory.list_agents():
            factory.terminate(agent.agent_id)
            print(f"   ✅ Terminated: {agent.name}")
        
        print("\n✅ Test complete!\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
