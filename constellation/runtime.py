# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Unified Runtime v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Unified runtime that integrates all constellation subsystems:
- HyperGraphRAG knowledge connector
- Multi-tier memory system
- Hook and trigger systems
- Slash commands
- Skills and sub-agents
- MCP server
- A2A protocol
- Auto-audit
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

# Memory & Knowledge
from .memory import (
    HyperGraphRAGConnector,
    AgentKnowledgeInterface,
    MemoryManager,
    AgentMemoryInterface,
    MemoryPriority,
)

# Hooks & Triggers
from .hooks import (
    HookRegistry,
    HookPoint,
    HookMiddleware,
    get_hook_registry,
    on,
)

from .triggers import (
    TriggerEngine,
    TriggerType,
    PatternCondition,
    TriggerAction,
    get_trigger_engine,
    trigger,
)

# Commands
from .commands import (
    CommandRegistry,
    CommandContext,
    CommandResult,
    get_command_registry,
    command,
)

# Skills
from .skills import (
    SkillRegistry,
    SubAgentManager,
    get_skill_registry,
    get_sub_agent_manager,
)

# Protocols
from .protocols import (
    MCPServer,
    A2ARouter,
    A2AProtocol,
    get_mcp_server,
    get_a2a_router,
    get_a2a_protocol,
)

# Audit
from .audit import (
    AutoAuditEngine,
    AuditHooks,
    get_audit_engine,
    get_audit_hooks,
)

# Orchestrator
from .orchestrator import ConstellationOrchestrator


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    """Configuration for the constellation runtime."""
    # Storage
    data_vault_path: Path = field(default_factory=lambda: Path("bizra_data_vault"))
    
    # Memory
    working_memory_capacity: int = 50
    short_term_ttl_hours: int = 24
    
    # MCP
    mcp_enabled: bool = True
    mcp_server_name: str = "bizra-constellation"
    
    # A2A
    a2a_enabled: bool = True
    
    # Audit
    auto_audit_enabled: bool = True
    auto_apply_threshold: float = 0.95
    
    # Hooks
    hooks_enabled: bool = True
    
    # Triggers
    triggers_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTELLATION RUNTIME
# ─────────────────────────────────────────────────────────────────────────────

class ConstellationRuntime:
    """
    Unified runtime for the BIZRA Islamic Masterminds Constellation.
    
    Integrates and manages all subsystems:
    - Knowledge graph (HyperGraphRAG)
    - Memory system (multi-tier)
    - Hooks and triggers
    - Slash commands
    - Skills and sub-agents
    - MCP server
    - A2A protocol
    - Auto-audit
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self._initialized = False
        self._running = False
        
        # Subsystems (lazily initialized)
        self._orchestrator: Optional[ConstellationOrchestrator] = None
        self._hook_registry: Optional[HookRegistry] = None
        self._trigger_engine: Optional[TriggerEngine] = None
        self._command_registry: Optional[CommandRegistry] = None
        self._skill_registry: Optional[SkillRegistry] = None
        
        self._knowledge_connector: Optional[HyperGraphRAGConnector] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._sub_agent_manager: Optional[SubAgentManager] = None
        self._mcp_server: Optional[MCPServer] = None
        self._a2a_router: Optional[A2ARouter] = None
        self._audit_engine: Optional[AutoAuditEngine] = None
        
        # Agent interfaces
        self._agent_knowledge: dict[str, AgentKnowledgeInterface] = {}
        self._agent_memory: dict[str, AgentMemoryInterface] = {}
        self._agent_a2a: dict[str, A2AProtocol] = {}
        
    # ─────────────────────────────────────────────────────────────────────────
    # PROPERTY ACCESSORS (Synchronous)
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def orchestrator(self) -> ConstellationOrchestrator:
        """Get the constellation orchestrator."""
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        return self._orchestrator
        
    @property
    def hook_registry(self) -> HookRegistry:
        """Get the hook registry."""
        if not self.config.hooks_enabled:
            raise RuntimeError("Hooks subsystem is disabled via configuration.")
        if self._hook_registry is None:
            raise RuntimeError("Hook registry not initialized. Call initialize() first.")
        return self._hook_registry
        
    @property
    def trigger_engine(self) -> TriggerEngine:
        """Get the trigger engine."""
        if not self.config.triggers_enabled:
            raise RuntimeError("Triggers subsystem is disabled via configuration.")
        if self._trigger_engine is None:
            raise RuntimeError("Trigger engine not initialized. Call initialize() first.")
        return self._trigger_engine
        
    @property
    def command_registry(self) -> CommandRegistry:
        """Get the command registry."""
        if self._command_registry is None:
            raise RuntimeError("Command registry not initialized. Call initialize() first.")
        return self._command_registry
        
    @property
    def skill_registry(self) -> SkillRegistry:
        """Get the skill registry."""
        if self._skill_registry is None:
            raise RuntimeError("Skill registry not initialized. Call initialize() first.")
        return self._skill_registry
        
    @property
    def memory_manager(self) -> Optional[MemoryManager]:
        """Get the memory manager (may be None until initialized)."""
        return self._memory_manager
        
    @property
    def knowledge_connector(self) -> Optional[HyperGraphRAGConnector]:
        """Get the knowledge connector (may be None until initialized)."""
        return self._knowledge_connector
        
    @property
    def mcp_server(self) -> Optional[MCPServer]:
        """Get the MCP server (may be None if not enabled)."""
        return self._mcp_server
        
    @property
    def a2a_router(self) -> Optional[A2ARouter]:
        """Get the A2A router (may be None if not enabled)."""
        return self._a2a_router
        
    @property
    def audit_engine(self) -> Optional[AutoAuditEngine]:
        """Get the audit engine (may be None if not enabled)."""
        return self._audit_engine
        
    def run_command(self, command_text: str, **kwargs) -> CommandResult:
        """Execute a slash command synchronously."""
        if not self._initialized:
            raise RuntimeError("Runtime not initialized. Call initialize() first.")
        if self._command_registry is None:
            raise RuntimeError("Command registry not initialized. Call initialize() first.")
        return self._command_registry.execute_sync(command_text, **kwargs)
        
    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    async def initialize(self) -> None:
        """Initialize all subsystems."""
        if self._initialized:
            return
            
        logger.info("Initializing BIZRA Constellation Runtime...")
        
        # Configure logging
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Initialize orchestrator
        if self._orchestrator is None:
            self._orchestrator = ConstellationOrchestrator()
            
        # Initialize knowledge connector
        if self._knowledge_connector is None:
            self._knowledge_connector = HyperGraphRAGConnector(
                graph_path=self.config.data_vault_path / "knowledge_graph",
            )
            self._knowledge_connector.initialize()
        
        # Initialize memory manager
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(
                storage_base=self.config.data_vault_path / "memory",
                working_capacity=self.config.working_memory_capacity,
                short_term_ttl=self.config.short_term_ttl_hours,
            )
        
        # Initialize hooks
        if self.config.hooks_enabled:
            if self._hook_registry is None:
                self._hook_registry = get_hook_registry()
            self._register_system_hooks()
            
        # Initialize triggers
        if self.config.triggers_enabled:
            if self._trigger_engine is None:
                self._trigger_engine = get_trigger_engine()
            self._register_default_triggers()
            
        # Initialize commands
        if self._command_registry is None:
            self._command_registry = get_command_registry()
        self._register_runtime_commands()
        
        # Initialize skills
        if self._skill_registry is None:
            self._skill_registry = get_skill_registry()
        if self._sub_agent_manager is None:
            self._sub_agent_manager = get_sub_agent_manager()
        
        # Initialize MCP server
        if self.config.mcp_enabled and self._mcp_server is None:
            self._mcp_server = get_mcp_server()
            
        # Initialize A2A router
        if self.config.a2a_enabled and self._a2a_router is None:
            self._a2a_router = get_a2a_router()
            
        # Initialize audit engine
        if self.config.auto_audit_enabled and self._audit_engine is None:
            self._audit_engine = get_audit_engine()
            self._register_audit_hooks()
            
        self._initialized = True
        logger.info("BIZRA Constellation Runtime initialized successfully")
        
    def _register_system_hooks(self) -> None:
        """Register system-level hooks."""
        
        @on(HookPoint.AGENT_COMPLETE)
        async def on_agent_complete(event):
            """Store agent output in memory and knowledge graph."""
            agent_slug = event.agent_slug
            result = event.data.get("result", {})
            
            # Store in memory
            if self._memory_manager and agent_slug:
                interface = self.get_memory_interface(agent_slug)
                interface.remember(
                    content=str(result.get("content", "")),
                    priority=MemoryPriority.MEDIUM,
                    metadata={"session_id": event.session_id},
                )
                
            # Store in knowledge graph
            if self._knowledge_connector and agent_slug:
                kg_interface = self.get_knowledge_interface(agent_slug)
                kg_interface.remember(
                    content=str(result.get("content", "")),
                    claims=result.get("claims", []),
                    confidence=result.get("snr_score", 0.0),
                    session_id=event.session_id,
                )
                
        @on(HookPoint.TASK_RECEIVED)
        async def on_task_received(event):
            """Trigger evaluation when task is received."""
            if self._trigger_engine:
                await self._trigger_engine.evaluate({
                    "event_type": "task_received",
                    "content": event.data.get("task", ""),
                    "agent_slug": event.agent_slug,
                })
                
    def _register_default_triggers(self) -> None:
        """Register default triggers."""
        
        # Trigger medical expert for health-related queries
        trigger("medical-query-trigger") \
            .of_type(TriggerType.PATTERN) \
            .when_pattern(r"(health|medical|symptom|diagnos|treat)", is_regex=True) \
            .invoke_agent("ibn-sina") \
            .with_cooldown(60) \
            .described_as("Invoke Ibn Sina for medical queries") \
            .build()
            
        # Trigger mathematician for calculation queries
        trigger("math-query-trigger") \
            .of_type(TriggerType.PATTERN) \
            .when_pattern(r"(calculat|equation|algorithm|formula)", is_regex=True) \
            .invoke_agent("al-khwarizmi") \
            .with_cooldown(60) \
            .described_as("Invoke Al-Khwarizmi for mathematical queries") \
            .build()
            
    def _register_runtime_commands(self) -> None:
        """Register runtime-specific commands."""
        
        @command(
            name="runtime",
            description="Get runtime status and statistics",
        )
        async def runtime_status(ctx: CommandContext) -> CommandResult:
            status = self.get_status()
            return CommandResult(
                success=True,
                message="Runtime Status",
                data=status,
            )
            
        @command(
            name="memory",
            description="Memory operations",
        )
        async def memory_ops(ctx: CommandContext) -> CommandResult:
            return CommandResult(
                success=True,
                message="Memory system operational",
                data={
                    "working_capacity": self.config.working_memory_capacity,
                    "short_term_ttl": self.config.short_term_ttl_hours,
                },
            )
            
    def _register_audit_hooks(self) -> None:
        """Register audit hooks."""
        if not self._audit_engine:
            return
            
        audit_hooks = get_audit_hooks()
        
        @on(HookPoint.AGENT_COMPLETE)
        async def audit_on_complete(event):
            await audit_hooks.on_agent_complete(event.data)
            
    # ─────────────────────────────────────────────────────────────────────────
    # AGENT INTERFACES
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_knowledge_interface(self, agent_slug: str) -> AgentKnowledgeInterface:
        """Get knowledge graph interface for an agent."""
        if agent_slug not in self._agent_knowledge:
            self._agent_knowledge[agent_slug] = AgentKnowledgeInterface(
                agent_slug=agent_slug,
                connector=self._knowledge_connector,
            )
        return self._agent_knowledge[agent_slug]
        
    def get_memory_interface(self, agent_slug: str) -> AgentMemoryInterface:
        """Get memory interface for an agent."""
        if agent_slug not in self._agent_memory:
            self._agent_memory[agent_slug] = self._memory_manager.get_interface(agent_slug)
        return self._agent_memory[agent_slug]
        
    def get_a2a_protocol(self, agent_slug: str) -> A2AProtocol:
        """Get A2A protocol interface for an agent."""
        if agent_slug not in self._agent_a2a:
            self._agent_a2a[agent_slug] = get_a2a_protocol(agent_slug)
        return self._agent_a2a[agent_slug]
        
    # ─────────────────────────────────────────────────────────────────────────
    # RUNTIME OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    async def process_input(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        agent_slug: Optional[str] = None,
    ) -> dict:
        """
        Process input through the constellation.
        
        Handles:
        - Slash commands (if starts with /)
        - Trigger evaluation
        - Agent invocation
        """
        if not self._initialized:
            await self.initialize()
            
        # Check for slash command
        if input_text.startswith("/"):
            result = await self._command_registry.execute(
                input_text,
                session_id=session_id,
                agent_slug=agent_slug,
            )
            return {
                "type": "command",
                "result": result.message,
                "data": result.data,
                "success": result.success,
            }
            
        # Evaluate triggers
        if self._trigger_engine:
            fired = await self._trigger_engine.evaluate({
                "content": input_text,
                "session_id": session_id,
                "agent_slug": agent_slug,
            })
            if fired:
                return {
                    "type": "trigger",
                    "triggers_fired": len(fired),
                    "details": fired,
                }
                
        # Default: return for agent processing
        return {
            "type": "input",
            "text": input_text,
            "session_id": session_id,
        }
        
    async def execute_command(
        self,
        command_text: str,
        **kwargs,
    ) -> CommandResult:
        """Execute a slash command."""
        return await self._command_registry.execute(command_text, **kwargs)
        
    async def invoke_agent(
        self,
        agent_slug: str,
        task: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Invoke a specific agent with a task."""
        # Trigger hooks
        if self._hook_registry:
            await self._hook_registry.trigger(
                HookPoint.AGENT_START,
                agent_slug=agent_slug,
                session_id=session_id,
                data={"task": task},
            )
            
        # Recall relevant knowledge
        knowledge = self.get_knowledge_interface(agent_slug)
        context = knowledge.recall(task, min_snr=0.85)
        
        # Get memory context
        memory = self.get_memory_interface(agent_slug)
        memories = memory.get_context(limit=5)
        
        result = {
            "agent": agent_slug,
            "task": task,
            "knowledge_context": len(context.nodes),
            "memory_context": len(memories),
            "status": "ready_for_execution",
        }
        
        return result
        
    # ─────────────────────────────────────────────────────────────────────────
    # STATUS & LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_status(self) -> dict:
        """Get runtime status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "subsystems": {
                "knowledge_graph": self._knowledge_connector is not None,
                "memory_manager": self._memory_manager is not None,
                "hooks": self._hook_registry is not None,
                "triggers": self._trigger_engine is not None,
                "commands": self._command_registry is not None,
                "skills": self._skill_registry is not None,
                "mcp_server": self._mcp_server is not None,
                "a2a_router": self._a2a_router is not None,
                "audit_engine": self._audit_engine is not None,
            },
            "agents_connected": {
                "knowledge": len(self._agent_knowledge),
                "memory": len(self._agent_memory),
                "a2a": len(self._agent_a2a),
            },
            "config": {
                "data_vault": str(self.config.data_vault_path),
                "mcp_enabled": self.config.mcp_enabled,
                "a2a_enabled": self.config.a2a_enabled,
                "auto_audit": self.config.auto_audit_enabled,
            },
        }
        
    async def start(self) -> None:
        """Start the runtime."""
        if not self._initialized:
            await self.initialize()
            
        self._running = True
        
        # Trigger system start hook
        if self._hook_registry:
            await self._hook_registry.trigger(HookPoint.SYSTEM_START)
            
        logger.info("BIZRA Constellation Runtime started")
        
    async def shutdown(self) -> None:
        """Shutdown the runtime."""
        logger.info("Shutting down BIZRA Constellation Runtime...")
        
        # Trigger shutdown hook
        if self._hook_registry:
            await self._hook_registry.trigger(HookPoint.SYSTEM_SHUTDOWN)
            
        # Stop MCP server
        if self._mcp_server:
            self._mcp_server.stop()
            
        self._running = False
        logger.info("BIZRA Constellation Runtime shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_runtime: Optional[ConstellationRuntime] = None


def get_runtime(config: Optional[RuntimeConfig] = None) -> ConstellationRuntime:
    """Get the global constellation runtime."""
    global _runtime
    if _runtime is None:
        _runtime = ConstellationRuntime(config)
    return _runtime


async def initialize_constellation(config: Optional[RuntimeConfig] = None) -> ConstellationRuntime:
    """Initialize and return the constellation runtime."""
    runtime = get_runtime(config)
    await runtime.initialize()
    return runtime
