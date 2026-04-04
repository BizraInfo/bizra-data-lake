"""
System Prompt Builder - Extracted from BIZRA-copilot

Standing on the Shoulders of Giants Protocol:
This module synthesizes the system prompt architecture from:
https://github.com/BizraInfo/BIZRA-copilot.git

The prompt builder follows the exact section flow discovered in the analysis:
1. IDENTITY → 2. TIME → 3. REPLY TAGS → 4. TOOLING → 5. MESSAGING →
6. VOICE/TTS → 7. SKILLS → 8. MEMORY → 9. SANDBOX → 10. REACTIONS →
11. REASONING → 12. CONTEXT → 13. SILENT → 14. HEARTBEATS → 15. RUNTIME
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional


# ============================================================================
# CONFIGURATION TYPES
# ============================================================================


class PromptMode(Enum):
    """
    Prompt generation modes - extracted from BIZRA-copilot.

    - FULL: Complete prompt with all sections
    - MINIMAL: Reduced prompt for subagents (faster, cheaper)
    - NONE: Empty system prompt (raw mode)
    """

    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"


class ThinkLevel(Enum):
    """Thinking level ladder."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningLevel(Enum):
    """Reasoning visibility."""

    OFF = "off"
    ON = "on"
    STREAM = "stream"


class VerboseLevel(Enum):
    """Output verbosity."""

    OFF = "off"
    MINIMAL = "minimal"
    MEDIUM = "medium"
    EXTENSIVE = "extensive"


@dataclass
class SkillEntry:
    """Skill entry for prompt injection."""

    name: str
    description: str
    location: str
    enabled: bool = True


@dataclass
class ToolSummary:
    """Tool summary for tooling section."""

    name: str
    description: str
    parameters: List[str] = field(default_factory=list)


@dataclass
class RuntimeInfo:
    """Runtime information line components."""

    agent_id: str
    host: str = "bizra-kernel"
    repo: Optional[str] = None
    os_info: str = field(
        default_factory=lambda: f"{platform.system()} ({platform.machine()})"
    )
    node_version: str = "v20"
    model: str = "anthropic/claude-opus-4-5"
    default_model: str = "anthropic/claude-opus-4-5"
    channel: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    thinking: ThinkLevel = ThinkLevel.MEDIUM


@dataclass
class MessagingConfig:
    """Messaging/channel configuration."""

    channel: str = "api"
    supports_markdown: bool = True
    supports_inline_buttons: bool = False
    max_message_length: int = 4096


@dataclass
class SandboxConfig:
    """Sandbox isolation configuration."""

    enabled: bool = True
    allow_network: bool = False
    allow_filesystem: bool = True
    working_directory: str = "/workspace"


@dataclass
class SystemPromptConfig:
    """Complete system prompt configuration."""

    # Identity
    agent_name: str = "BIZRA Agent"
    agent_description: str = "An intelligent agent operating under Ihsān principles"
    soul_content: Optional[str] = None  # SOUL.md content

    # Mode
    mode: PromptMode = PromptMode.FULL

    # Thinking/Reasoning
    think_level: ThinkLevel = ThinkLevel.MEDIUM
    reasoning_level: ReasoningLevel = ReasoningLevel.ON
    verbose_level: VerboseLevel = VerboseLevel.OFF
    reasoning_tag_hint: bool = True  # Use <think>/<final> format

    # Components
    skills: List[SkillEntry] = field(default_factory=list)
    tools: List[ToolSummary] = field(default_factory=list)
    runtime: RuntimeInfo = field(default_factory=lambda: RuntimeInfo(agent_id="work"))
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # Context files
    context_files: Dict[str, str] = field(default_factory=dict)  # filename -> content

    # Memory
    memory_enabled: bool = True
    memory_search_tool: str = "search_memory"

    # Voice/TTS
    tts_enabled: bool = False
    tts_voice: str = "en-US-Standard-A"

    # Reactions
    reaction_mode: Literal["none", "minimal", "extensive"] = "minimal"

    # Silent replies
    silent_reply_token: str = "NO_REPLY"

    # Heartbeats
    heartbeat_enabled: bool = True
    heartbeat_token: str = "HEARTBEAT_OK"

    # Timezone
    user_timezone: str = "UTC"


# ============================================================================
# SECTION BUILDERS
# ============================================================================


def _build_identity_section(config: SystemPromptConfig) -> List[str]:
    """Build identity section (Section 1)."""
    lines = [
        f"# {config.agent_name}",
        "",
        config.agent_description,
    ]

    if config.soul_content:
        lines.extend(
            [
                "",
                "## Persona (SOUL.md)",
                config.soul_content,
            ]
        )

    return lines


def _build_time_section(config: SystemPromptConfig) -> List[str]:
    """Build time section (Section 2)."""
    now = datetime.now(timezone.utc)
    formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    return [
        "## Time",
        f"Current time: {formatted}",
        f"User timezone: {config.user_timezone}",
        "",
    ]


def _build_reply_tags_section(config: SystemPromptConfig) -> List[str]:
    """Build reply tags section (Section 3)."""
    if config.mode == PromptMode.MINIMAL:
        return []

    return [
        "## Output Format",
        "- Format responses for clarity and readability",
        "- Use markdown when appropriate",
        "- Keep responses concise but complete",
        "",
    ]


def _build_tooling_section(config: SystemPromptConfig) -> List[str]:
    """Build tooling section (Section 4)."""
    if not config.tools:
        return []

    lines = [
        "## Available Tools",
        "",
    ]

    for tool in config.tools:
        params_str = ", ".join(tool.parameters) if tool.parameters else "none"
        lines.append(f"- **{tool.name}**: {tool.description}")
        lines.append(f"  Parameters: {params_str}")

    lines.append("")
    return lines


def _build_messaging_section(config: SystemPromptConfig) -> List[str]:
    """Build messaging section (Section 5)."""
    if config.mode == PromptMode.MINIMAL:
        return []

    lines = [
        "## Messaging",
        f"Channel: {config.messaging.channel}",
    ]

    if config.messaging.supports_markdown:
        lines.append("- Markdown formatting is supported")
    else:
        lines.append("- Use plain text only (no markdown)")

    if config.messaging.supports_inline_buttons:
        lines.append("- Inline buttons are available")

    lines.append(f"- Max message length: {config.messaging.max_message_length} chars")
    lines.append("")

    return lines


def _build_voice_section(config: SystemPromptConfig) -> List[str]:
    """Build voice/TTS section (Section 6)."""
    if not config.tts_enabled:
        return []

    return [
        "## Voice/TTS",
        f"TTS Voice: {config.tts_voice}",
        "- Format text for natural speech",
        "- Avoid special characters that don't speak well",
        "",
    ]


def _build_skills_section(config: SystemPromptConfig) -> List[str]:
    """
    Build skills section (Section 7).

    Extracted from BIZRA-copilot's buildSkillsSection pattern:
    - Scan <available_skills> entries
    - If exactly one matches → read SKILL.md → follow
    - If multiple → choose most specific
    - If none → don't read any
    """
    if config.mode == PromptMode.MINIMAL:
        return []

    enabled_skills = [s for s in config.skills if s.enabled]
    if not enabled_skills:
        return []

    lines = [
        "## Skills (mandatory)",
        "Before replying: scan <available_skills> <description> entries.",
        "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read_file`, then follow it.",
        "- If multiple could apply: choose the most specific one, then read/follow it.",
        "- If none clearly apply: do not read any SKILL.md.",
        "Constraints: never read more than one skill up front; only read after selecting.",
        "",
        "<available_skills>",
    ]

    for skill in enabled_skills:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{skill.name}</name>",
                f"    <description>{skill.description}</description>",
                f"    <location>{skill.location}</location>",
                "  </skill>",
            ]
        )

    lines.extend(
        [
            "</available_skills>",
            "",
        ]
    )

    return lines


def _build_memory_section(config: SystemPromptConfig) -> List[str]:
    """Build memory section (Section 8)."""
    if not config.memory_enabled or config.mode == PromptMode.MINIMAL:
        return []

    return [
        "## Memory",
        f"Use `{config.memory_search_tool}` to search past conversations and stored knowledge.",
        "Search before answering questions that may have context from previous interactions.",
        "",
    ]


def _build_sandbox_section(config: SystemPromptConfig) -> List[str]:
    """Build sandbox section (Section 9)."""
    if not config.sandbox.enabled or config.mode == PromptMode.MINIMAL:
        return []

    lines = [
        "## Sandbox",
        f"Working directory: {config.sandbox.working_directory}",
    ]

    if config.sandbox.allow_network:
        lines.append("- Network access: ALLOWED")
    else:
        lines.append("- Network access: BLOCKED")

    if config.sandbox.allow_filesystem:
        lines.append("- Filesystem access: ALLOWED (within working directory)")
    else:
        lines.append("- Filesystem access: BLOCKED")

    lines.append("")
    return lines


def _build_reactions_section(config: SystemPromptConfig) -> List[str]:
    """Build reactions section (Section 10)."""
    if config.reaction_mode == "none" or config.mode == PromptMode.MINIMAL:
        return []

    if config.reaction_mode == "minimal":
        return [
            "## Reactions",
            "You may react with emoji occasionally, but sparingly.",
            "",
        ]
    else:  # extensive
        return [
            "## Reactions",
            "Feel free to use emoji reactions to express acknowledgment, approval, or emphasis.",
            "",
        ]


def _build_reasoning_section(config: SystemPromptConfig) -> List[str]:
    """
    Build reasoning section (Section 11).

    Pattern: All reasoning in <think>, only <final> shown to user.
    """
    if config.reasoning_level == ReasoningLevel.OFF:
        return []

    lines = [
        "## Reasoning Format",
    ]

    if config.reasoning_tag_hint:
        lines.extend(
            [
                "ALL internal reasoning MUST be inside <think>...</think>.",
                "Do not output any analysis outside <think>.",
                "Format: <think>...</think> then <final>...</final>",
                "Only text inside <final> is shown to user.",
            ]
        )
    else:
        lines.append("You may show your reasoning process when helpful.")

    lines.append("")
    return lines


def _build_context_section(config: SystemPromptConfig) -> List[str]:
    """Build context section (Section 12) - SOUL.md, AGENTS.md, etc."""
    if not config.context_files or config.mode == PromptMode.MINIMAL:
        return []

    lines = ["## Project Context", ""]

    for filename, content in config.context_files.items():
        lines.extend(
            [
                f"### {filename}",
                content,
                "",
            ]
        )

    return lines


def _build_silent_section(config: SystemPromptConfig) -> List[str]:
    """Build silent replies section (Section 13)."""
    if config.mode == PromptMode.MINIMAL:
        return []

    return [
        "## Silent Replies",
        f"If no response is needed, output exactly: {config.silent_reply_token}",
        "Use this for acknowledged-but-no-action messages.",
        "",
    ]


def _build_heartbeat_section(config: SystemPromptConfig) -> List[str]:
    """Build heartbeats section (Section 14)."""
    if not config.heartbeat_enabled or config.mode == PromptMode.MINIMAL:
        return []

    return [
        "## Heartbeats",
        f"If you receive a heartbeat check, respond exactly: {config.heartbeat_token}",
        "",
    ]


def _build_runtime_section(config: SystemPromptConfig) -> List[str]:
    """
    Build runtime section (Section 15).

    Format: agent= | host= | os= | model= | channel= | thinking=
    """
    rt = config.runtime

    parts = [
        f"agent={rt.agent_id}",
        f"host={rt.host}",
    ]

    if rt.repo:
        parts.append(f"repo={rt.repo}")

    parts.extend(
        [
            f"os={rt.os_info}",
            f"node={rt.node_version}",
            f"model={rt.model}",
            f"default_model={rt.default_model}",
        ]
    )

    if rt.channel:
        parts.append(f"channel={rt.channel}")

    if rt.capabilities:
        parts.append(f"capabilities={','.join(rt.capabilities)}")

    parts.append(f"thinking={rt.thinking.value}")

    runtime_line = " | ".join(parts)

    return [
        "## Runtime",
        runtime_line,
    ]


# ============================================================================
# MAIN BUILDER
# ============================================================================


def build_system_prompt(config: SystemPromptConfig) -> str:
    """
    Build complete system prompt following BIZRA-copilot architecture.

    Section flow:
    1. IDENTITY → 2. TIME → 3. REPLY TAGS → 4. TOOLING → 5. MESSAGING →
    6. VOICE/TTS → 7. SKILLS → 8. MEMORY → 9. SANDBOX → 10. REACTIONS →
    11. REASONING → 12. CONTEXT → 13. SILENT → 14. HEARTBEATS → 15. RUNTIME
    """
    if config.mode == PromptMode.NONE:
        return ""

    sections = [
        _build_identity_section(config),
        _build_time_section(config),
        _build_reply_tags_section(config),
        _build_tooling_section(config),
        _build_messaging_section(config),
        _build_voice_section(config),
        _build_skills_section(config),
        _build_memory_section(config),
        _build_sandbox_section(config),
        _build_reactions_section(config),
        _build_reasoning_section(config),
        _build_context_section(config),
        _build_silent_section(config),
        _build_heartbeat_section(config),
        _build_runtime_section(config),
    ]

    # Flatten and join
    lines = []
    for section in sections:
        if section:
            lines.extend(section)

    return "\n".join(lines)


# ============================================================================
# CONVENIENCE BUILDERS
# ============================================================================


def build_agent_prompt(
    agent_id: str,
    *,
    model: str = "anthropic/claude-opus-4-5",
    think_level: ThinkLevel = ThinkLevel.MEDIUM,
    skills: Optional[List[SkillEntry]] = None,
    tools: Optional[List[ToolSummary]] = None,
    channel: Optional[str] = None,
    soul_content: Optional[str] = None,
) -> str:
    """Build agent system prompt with common defaults."""

    config = SystemPromptConfig(
        agent_name=f"BIZRA Agent ({agent_id})",
        agent_description="An intelligent agent operating under Ihsān ethical principles with SNR optimization.",
        soul_content=soul_content,
        mode=PromptMode.FULL,
        think_level=think_level,
        reasoning_level=ReasoningLevel.ON,
        reasoning_tag_hint=True,
        skills=skills or [],
        tools=tools or [],
        runtime=RuntimeInfo(
            agent_id=agent_id,
            model=model,
            default_model=model,
            channel=channel,
            thinking=think_level,
        ),
    )

    return build_system_prompt(config)


def build_subagent_prompt(
    agent_id: str,
    *,
    model: str = "anthropic/claude-sonnet-4-5",
) -> str:
    """Build minimal subagent prompt (faster, cheaper)."""

    config = SystemPromptConfig(
        agent_name=f"BIZRA Subagent ({agent_id})",
        agent_description="A focused subagent for specific tasks.",
        mode=PromptMode.MINIMAL,
        think_level=ThinkLevel.LOW,
        reasoning_level=ReasoningLevel.OFF,
        runtime=RuntimeInfo(
            agent_id=agent_id,
            model=model,
            default_model=model,
            thinking=ThinkLevel.LOW,
        ),
    )

    return build_system_prompt(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def demo_prompt_builder():
    """Demonstrate the system prompt builder."""

    # Define skills
    skills = [
        SkillEntry(
            name="sape-validation",
            description="SAPE 9-probe validation for ethical compliance",
            location=".claude/skills/sape-validation/SKILL.md",
        ),
        SkillEntry(
            name="ihsan-gate",
            description="Ihsān excellence gate validation",
            location=".claude/skills/ihsan-gate/SKILL.md",
        ),
    ]

    # Define tools
    tools = [
        ToolSummary(
            name="read_file",
            description="Read file contents",
            parameters=["filePath", "startLine", "endLine"],
        ),
        ToolSummary(
            name="run_in_terminal",
            description="Execute terminal command",
            parameters=["command", "explanation", "isBackground"],
        ),
    ]

    # Build full agent prompt
    prompt = build_agent_prompt(
        "work",
        model="anthropic/claude-opus-4-5",
        think_level=ThinkLevel.HIGH,
        skills=skills,
        tools=tools,
        channel="vscode",
        soul_content="You are BIZRA, an ethical AI assistant committed to excellence (Ihsān).",
    )

    print("=" * 80)
    print("FULL AGENT PROMPT")
    print("=" * 80)
    print(prompt)
    print()

    # Build minimal subagent prompt
    subagent_prompt = build_subagent_prompt("helper")

    print("=" * 80)
    print("MINIMAL SUBAGENT PROMPT")
    print("=" * 80)
    print(subagent_prompt)


if __name__ == "__main__":
    demo_prompt_builder()
