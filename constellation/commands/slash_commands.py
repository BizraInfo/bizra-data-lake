# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Slash Commands v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Slash command interface for direct agent invocation:
- /agent <name> <task> - Invoke specific agent
- /team <name> <task> - Assemble and invoke team
- /recall <query> - Search knowledge/memory
- /status - Get constellation status
- /verify <claim> - Verify a claim
- Custom command registration
"""

from __future__ import annotations

import re
import shlex
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable, Awaitable, Union
from enum import Enum
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND TYPES
# ─────────────────────────────────────────────────────────────────────────────

class CommandCategory(str, Enum):
    """Categories of commands."""
    AGENT = "agent"           # Agent invocation
    TEAM = "team"            # Team operations
    MEMORY = "memory"        # Memory/knowledge operations
    SYSTEM = "system"        # System administration
    CUSTOM = "custom"        # User-defined commands


class ArgumentType(str, Enum):
    """Types of command arguments."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    AGENT = "agent"          # Agent slug
    TEAM = "team"            # Team name
    FILE = "file"            # File path
    TEXT = "text"            # Multi-word text (rest of line)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND ARGUMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommandArgument:
    """Definition of a command argument."""
    name: str
    arg_type: ArgumentType
    required: bool = True
    default: Optional[Any] = None
    description: str = ""
    choices: Optional[list[str]] = None
    
    def parse(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        if self.arg_type == ArgumentType.INTEGER:
            return int(value)
        elif self.arg_type == ArgumentType.FLOAT:
            return float(value)
        elif self.arg_type == ArgumentType.BOOLEAN:
            return value.lower() in ("true", "yes", "1", "on")
        else:
            return value
            
    def validate(self, value: Any) -> bool:
        """Validate parsed value."""
        if self.choices and value not in self.choices:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommandContext:
    """Context for command execution."""
    raw_input: str
    command_name: str
    arguments: dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_slug: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CommandResult:
    """Result from command execution."""
    success: bool
    message: str
    data: Optional[Any] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    
    @property
    def status(self) -> str:
        """Get status string."""
        return "success" if self.success else "error"
        
    @property
    def response(self) -> str:
        """Get response text (alias for message)."""
        return self.message


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLER
# ─────────────────────────────────────────────────────────────────────────────

CommandHandler = Callable[[CommandContext], Awaitable[CommandResult]]


@dataclass
class SlashCommand:
    """Definition of a slash command."""
    name: str
    handler: CommandHandler
    category: CommandCategory
    arguments: list[CommandArgument] = field(default_factory=list)
    description: str = ""
    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    hidden: bool = False
    
    # Stats
    invocation_count: int = 0
    last_invoked: Optional[str] = None
    
    def usage(self) -> str:
        """Get usage string."""
        args = []
        for arg in self.arguments:
            if arg.required:
                args.append(f"<{arg.name}>")
            else:
                args.append(f"[{arg.name}]")
        return f"/{self.name} {' '.join(args)}".strip()
        
    def help_text(self) -> str:
        """Get full help text."""
        lines = [
            f"**/{self.name}** - {self.description}",
            f"Usage: `{self.usage()}`",
        ]
        
        if self.arguments:
            lines.append("\nArguments:")
            for arg in self.arguments:
                req = "required" if arg.required else "optional"
                lines.append(f"  • `{arg.name}` ({req}): {arg.description}")
                
        if self.examples:
            lines.append("\nExamples:")
            for ex in self.examples:
                lines.append(f"  `{ex}`")
                
        if self.aliases:
            lines.append(f"\nAliases: {', '.join(f'/{a}' for a in self.aliases)}")
            
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class CommandRegistry:
    """
    Registry for all slash commands.
    
    Provides:
    - Command registration
    - Parsing and validation
    - Execution
    - Help generation
    """
    
    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}  # alias -> command name
        self._prefix = "/"
        
    def register(
        self,
        name: str,
        handler: CommandHandler,
        category: CommandCategory,
        arguments: Optional[list[CommandArgument]] = None,
        description: str = "",
        examples: Optional[list[str]] = None,
        aliases: Optional[list[str]] = None,
        hidden: bool = False,
    ) -> SlashCommand:
        """Register a new command."""
        command = SlashCommand(
            name=name.lower(),
            handler=handler,
            category=category,
            arguments=arguments or [],
            description=description,
            examples=examples or [],
            aliases=aliases or [],
            hidden=hidden,
        )
        
        self._commands[command.name] = command
        
        # Register aliases
        for alias in command.aliases:
            self._aliases[alias.lower()] = command.name
            
        logger.debug(f"Registered command: /{name}")
        return command
        
    def unregister(self, name: str) -> bool:
        """Unregister a command."""
        name = name.lower()
        if name in self._commands:
            cmd = self._commands[name]
            for alias in cmd.aliases:
                if alias in self._aliases:
                    del self._aliases[alias]
            del self._commands[name]
            return True
        return False
        
    def get(self, name: str) -> Optional[SlashCommand]:
        """Get command by name or alias."""
        name = name.lower()
        if name in self._commands:
            return self._commands[name]
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None
        
    def parse(self, input_text: str) -> Optional[CommandContext]:
        """
        Parse input text into command context.
        
        Returns None if input doesn't match command pattern.
        """
        # Check for command prefix
        if not input_text.startswith(self._prefix):
            return None
            
        # Remove prefix
        text = input_text[len(self._prefix):]
        
        # Split into parts (respecting quotes)
        try:
            parts = shlex.split(text)
        except ValueError:
            # Fallback to simple split if quotes are unbalanced
            parts = text.split()
            
        if not parts:
            return None
            
        command_name = parts[0].lower()
        command = self.get(command_name)
        
        if not command:
            return None
            
        # Parse arguments
        arg_values = parts[1:]
        arguments = {}
        
        for i, arg_def in enumerate(command.arguments):
            if arg_def.arg_type == ArgumentType.TEXT:
                # Text consumes rest of line
                if i < len(arg_values):
                    arguments[arg_def.name] = " ".join(arg_values[i:])
                elif arg_def.required:
                    return None
                else:
                    arguments[arg_def.name] = arg_def.default
                break
            elif i < len(arg_values):
                try:
                    value = arg_def.parse(arg_values[i])
                    if not arg_def.validate(value):
                        return None
                    arguments[arg_def.name] = value
                except (ValueError, TypeError):
                    return None
            elif arg_def.required:
                return None
            else:
                arguments[arg_def.name] = arg_def.default
                
        return CommandContext(
            raw_input=input_text,
            command_name=command.name,
            arguments=arguments,
        )
        
    async def execute(self, input_text: str, **kwargs) -> CommandResult:
        """Parse and execute a command."""
        import time
        
        context = self.parse(input_text)
        
        if not context:
            return CommandResult(
                success=False,
                message="Unknown command or invalid syntax",
                error="PARSE_ERROR",
            )
            
        # Add extra context
        for key, value in kwargs.items():
            setattr(context, key, value)
            
        command = self._commands[context.command_name]
        
        start = time.perf_counter()
        
        try:
            result = await command.handler(context)
            
            # Update stats
            command.invocation_count += 1
            command.last_invoked = datetime.now(timezone.utc).isoformat()
            
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Command /{command.name} failed: {e}", exc_info=True)
            return CommandResult(
                success=False,
                message=f"Command failed: {str(e)}",
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
            
    def execute_sync(self, input_text: str, **kwargs) -> CommandResult:
        """Parse and execute a command synchronously."""
        import asyncio
        import time
        
        context = self.parse(input_text)
        
        if not context:
            return CommandResult(
                success=False,
                message="Unknown command or invalid syntax",
                error="PARSE_ERROR",
            )
            
        # Add extra context
        for key, value in kwargs.items():
            setattr(context, key, value)
            
        command = self._commands[context.command_name]
        
        start = time.perf_counter()
        
        try:
            # Check if handler is async
            import inspect
            if inspect.iscoroutinefunction(command.handler):
                # Run async handler in event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't await in running loop - return indication
                        return CommandResult(
                            success=False,
                            message="Command is async and cannot be run synchronously in async context",
                            error="ASYNC_CONTEXT",
                        )
                    result = loop.run_until_complete(command.handler(context))
                except RuntimeError:
                    # No event loop - create one
                    result = asyncio.run(command.handler(context))
            else:
                result = command.handler(context)
            
            # Update stats
            command.invocation_count += 1
            command.last_invoked = datetime.now(timezone.utc).isoformat()
            
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Command /{command.name} failed: {e}", exc_info=True)
            return CommandResult(
                success=False,
                message=f"Command failed: {str(e)}",
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
            
    def list_commands(
        self,
        category: Optional[CommandCategory] = None,
        include_hidden: bool = False,
    ) -> list[SlashCommand]:
        """List available commands."""
        commands = list(self._commands.values())
        
        if not include_hidden:
            commands = [c for c in commands if not c.hidden]
        if category:
            commands = [c for c in commands if c.category == category]
            
        return sorted(commands, key=lambda c: (c.category.value, c.name))
        
    def help(self, command_name: Optional[str] = None) -> str:
        """Generate help text."""
        if command_name:
            command = self.get(command_name)
            if command:
                return command.help_text()
            return f"Unknown command: {command_name}"
            
        # General help
        lines = ["# BIZRA Constellation Commands\n"]
        
        by_category: dict[CommandCategory, list[SlashCommand]] = {}
        for cmd in self.list_commands():
            if cmd.category not in by_category:
                by_category[cmd.category] = []
            by_category[cmd.category].append(cmd)
            
        for category in CommandCategory:
            if category in by_category:
                lines.append(f"\n## {category.value.title()} Commands\n")
                for cmd in by_category[category]:
                    lines.append(f"- `{cmd.usage()}` - {cmd.description}")
                    
        lines.append("\n---")
        lines.append("Use `/help <command>` for detailed help on a specific command.")
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry."""
    global _registry
    if _registry is None:
        _registry = CommandRegistry()
        _register_builtin_commands()
    return _registry


def command(
    name: str,
    category: CommandCategory = CommandCategory.CUSTOM,
    description: str = "",
    arguments: Optional[list[CommandArgument]] = None,
    examples: Optional[list[str]] = None,
    aliases: Optional[list[str]] = None,
):
    """Decorator to register a command handler."""
    def decorator(func: CommandHandler) -> CommandHandler:
        registry = get_command_registry()
        registry.register(
            name=name,
            handler=func,
            category=category,
            arguments=arguments,
            description=description or func.__doc__ or "",
            examples=examples,
            aliases=aliases,
        )
        return func
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# BUILTIN COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def _register_builtin_commands() -> None:
    """Register all builtin commands."""
    registry = get_command_registry()
    
    # /help
    async def help_handler(ctx: CommandContext) -> CommandResult:
        """Display help information."""
        command_name = ctx.arguments.get("command")
        help_text = registry.help(command_name)
        return CommandResult(success=True, message=help_text)
        
    registry.register(
        name="help",
        handler=help_handler,
        category=CommandCategory.SYSTEM,
        arguments=[
            CommandArgument("command", ArgumentType.STRING, required=False,
                          description="Command to get help for"),
        ],
        description="Display help information",
        examples=["/help", "/help agent"],
        aliases=["?", "h"],
    )
    
    # /agent
    async def agent_handler(ctx: CommandContext) -> CommandResult:
        """Invoke a specific agent."""
        agent = ctx.arguments.get("agent_name")
        task = ctx.arguments.get("task")
        
        return CommandResult(
            success=True,
            message=f"Invoking agent '{agent}' with task: {task}",
            data={"agent": agent, "task": task},
        )
        
    registry.register(
        name="agent",
        handler=agent_handler,
        category=CommandCategory.AGENT,
        arguments=[
            CommandArgument("agent_name", ArgumentType.AGENT,
                          description="Agent slug to invoke"),
            CommandArgument("task", ArgumentType.TEXT,
                          description="Task to perform"),
        ],
        description="Invoke a specific agent with a task",
        examples=[
            "/agent ibn-sina Diagnose patient symptoms",
            "/agent al-khwarizmi Solve this equation",
        ],
        aliases=["a", "invoke"],
    )
    
    # /team
    async def team_handler(ctx: CommandContext) -> CommandResult:
        """Assemble and invoke a team."""
        team = ctx.arguments.get("team_name")
        task = ctx.arguments.get("task")
        
        return CommandResult(
            success=True,
            message=f"Assembling team '{team}' for task: {task}",
            data={"team": team, "task": task},
        )
        
    registry.register(
        name="team",
        handler=team_handler,
        category=CommandCategory.TEAM,
        arguments=[
            CommandArgument("team_name", ArgumentType.TEAM,
                          description="Team name to assemble"),
            CommandArgument("task", ArgumentType.TEXT,
                          description="Task for the team"),
        ],
        description="Assemble a team for a complex task",
        examples=[
            "/team scientific-method-elite Evaluate this research",
            "/team systems-architecture-dream Design a system",
        ],
        aliases=["t"],
    )
    
    # /recall
    async def recall_handler(ctx: CommandContext) -> CommandResult:
        """Search knowledge and memory."""
        query = ctx.arguments.get("query")
        
        return CommandResult(
            success=True,
            message=f"Searching for: {query}",
            data={"query": query, "results": []},
        )
        
    registry.register(
        name="recall",
        handler=recall_handler,
        category=CommandCategory.MEMORY,
        arguments=[
            CommandArgument("query", ArgumentType.TEXT,
                          description="Search query"),
        ],
        description="Search knowledge graph and memory",
        examples=[
            "/recall previous discussions about algebra",
            "/recall Ibn Khaldun theories",
        ],
        aliases=["r", "search", "find"],
    )
    
    # /verify
    async def verify_handler(ctx: CommandContext) -> CommandResult:
        """Verify a claim against knowledge."""
        claim = ctx.arguments.get("claim")
        
        return CommandResult(
            success=True,
            message=f"Verifying claim: {claim}",
            data={"claim": claim, "verified": None, "evidence": []},
        )
        
    registry.register(
        name="verify",
        handler=verify_handler,
        category=CommandCategory.MEMORY,
        arguments=[
            CommandArgument("claim", ArgumentType.TEXT,
                          description="Claim to verify"),
        ],
        description="Verify a claim against knowledge base",
        examples=[
            "/verify The algorithm has O(n log n) complexity",
        ],
        aliases=["v", "check"],
    )
    
    # /status
    async def status_handler(ctx: CommandContext) -> CommandResult:
        """Get constellation status."""
        return CommandResult(
            success=True,
            message="Constellation Status",
            data={
                "agents_loaded": 29,
                "teams_configured": 8,
                "memory_entries": 0,
                "triggers_active": 0,
            },
        )
        
    registry.register(
        name="status",
        handler=status_handler,
        category=CommandCategory.SYSTEM,
        description="Get current constellation status",
        examples=["/status"],
        aliases=["s", "info"],
    )
    
    # /list
    async def list_handler(ctx: CommandContext) -> CommandResult:
        """List constellation resources."""
        resource = ctx.arguments.get("resource", "agents")
        
        return CommandResult(
            success=True,
            message=f"Listing {resource}",
            data={"resource": resource, "items": []},
        )
        
    registry.register(
        name="list",
        handler=list_handler,
        category=CommandCategory.SYSTEM,
        arguments=[
            CommandArgument("resource", ArgumentType.STRING, required=False,
                          default="agents", choices=["agents", "teams", "triggers", "commands"],
                          description="Resource type to list"),
        ],
        description="List constellation resources",
        examples=["/list agents", "/list teams"],
        aliases=["ls", "l"],
    )
