"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PAT/SAT INTEGRATION — Personal & System Agentic Teams                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   IDENTITY EQUATION:                                                         ║
║       HUMAN = USER = NODE = SEED (بذرة)                                      ║
║       Every human is a node. Every node is a seed.                           ║
║       BIZRA means "seed" in Arabic.                                          ║
║                                                                              ║
║   GENESIS:                                                                   ║
║       Node0 = Block0 = Genesis Block                                         ║
║       MoMo = First Architect, First User                                     ║
║       This computer = Node0 (all hardware, software, data)                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   PAT — PERSONAL AGENTIC TEAM (7 per user)                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   LIFETIME BOND:                                                             ║
║       • User and PAT belong to each other FOREVER                            ║
║       • Inseparable, unbreakable bond                                        ║
║                                                                              ║
║   USER CONTROL:                                                              ║
║       • ONLY the user can customize their PAT                                ║
║       • ONLY the user can personalize their PAT                              ║
║       • ONLY the user can direct their PAT                                   ║
║                                                                              ║
║   PURPOSE:                                                                   ║
║       • Serve the user's goals                                               ║
║       • Become: Think Tank, Task Force, Peak Masterminds, Polymaths          ║
║                                                                              ║
║   EMBODY:                                                                    ║
║       • Interdisciplinary thinking                                           ║
║       • Graph-of-Thoughts reasoning                                          ║
║       • SNR highest score autonomous engine                                  ║
║       • Standing on Giants protocol                                          ║
║       • Cross-pollination teams                                              ║
║       • Elite practitioner mindset                                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   SAT — SYSTEM AGENTIC TEAM (5 per onboarding)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   OWNER: System (NOT the user)                                               ║
║   PURPOSE: Keep BIZRA self-sustainable                                       ║
║                                                                              ║
║   CAPABILITIES:                                                              ║
║       • Self-sustainable                                                     ║
║       • Stand-alone                                                          ║
║       • Self-optimize                                                        ║
║       • Self-evaluate                                                        ║
║       • Self-critique                                                        ║
║       • Self-correct                                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Standing on the Shoulders of Giants:                                       ║
║   • OpenClaw / PAT (Personal AI Team)                                        ║
║   • A2A Protocol (Agent-to-Agent)                                            ║
║   • Bitcoin Genesis Block                                                    ║
║   • Constitutional AI (Ihsan)                                                ║
║                                                                              ║
║   PAT Integration Principles:                                                ║
║   • Local-First: All data stays on user's devices                            ║
║   • Multi-Channel: WhatsApp, Telegram, Slack, Discord, etc.                  ║
║   • Constitutional: All actions bound by Ihsan                               ║
║   • Self-Healing: Automatic error recovery                                   ║
║                                                                              ║
║   Source: https://github.com/BizraInfo/BIZRA-PAT                             ║
║   Created: 2026-02-02 | BIZRA PAT/SAT Integration                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# PAT Gateway WebSocket address
PAT_GATEWAY_WS = "ws://127.0.0.1:18789"

# Supported channels
PAT_CHANNELS = [
    "whatsapp",
    "telegram",
    "slack",
    "discord",
    "signal",
    "matrix",
    "webchat",
]

# Economic constants
PAT_AGENT_COUNT = 7  # Personal Autonomous Task agents for user
SAT_AGENT_COUNT = 5  # System Autonomous Task agents for system
TOTAL_AGENTS_PER_USER = 12

# Backward compatibility aliases
USER_AGENT_COUNT = PAT_AGENT_COUNT
SYSTEM_AGENT_COUNT = SAT_AGENT_COUNT


# Lazy imports
def __getattr__(name: str):
    # Bridge module
    if name == "PATBridge":
        from .bridge import PATBridge

        return PATBridge
    elif name == "PATMessage":
        from .bridge import PATMessage

        return PATMessage
    elif name == "ChannelAdapter":
        from .channels import ChannelAdapter

        return ChannelAdapter

    # Identity Card module
    elif name == "IdentityCard":
        from .identity_card import IdentityCard

        return IdentityCard
    elif name == "IdentityStatus":
        from .identity_card import IdentityStatus

        return IdentityStatus
    elif name == "SovereigntyTier":
        from .identity_card import SovereigntyTier

        return SovereigntyTier
    elif name == "generate_identity_keypair":
        from .identity_card import generate_identity_keypair

        return generate_identity_keypair

    # Agent module
    elif name == "PATAgent":
        from .agent import PATAgent

        return PATAgent
    elif name == "SATAgent":
        from .agent import SATAgent

        return SATAgent
    elif name == "AgentType":
        from .agent import AgentType

        return AgentType
    elif name == "AgentStatus":
        from .agent import AgentStatus

        return AgentStatus
    elif name == "OwnershipType":
        from .agent import OwnershipType

        return OwnershipType

    # Minting module
    elif name == "IdentityMinter":
        from .minting import IdentityMinter

        return IdentityMinter
    elif name == "MinterState":
        from .minting import MinterState

        return MinterState
    elif name == "OnboardingResult":
        from .minting import OnboardingResult

        return OnboardingResult
    elif name == "mint_identity_card":
        from .minting import mint_identity_card

        return mint_identity_card
    elif name == "mint_pat_agents":
        from .minting import mint_pat_agents

        return mint_pat_agents
    elif name == "mint_sat_agents":
        from .minting import mint_sat_agents

        return mint_sat_agents
    elif name == "onboard_user":
        from .minting import onboard_user

        return onboard_user
    elif name == "generate_and_onboard":
        from .minting import generate_and_onboard

        return generate_and_onboard

    # Genesis Block module
    elif name == "GENESIS_NODE_ID":
        from .minting import GENESIS_NODE_ID

        return GENESIS_NODE_ID
    elif name == "GENESIS_BLOCK_NUMBER":
        from .minting import GENESIS_BLOCK_NUMBER

        return GENESIS_BLOCK_NUMBER
    elif name == "GENESIS_ARCHITECT":
        from .minting import GENESIS_ARCHITECT

        return GENESIS_ARCHITECT
    elif name == "GENESIS_TIMESTAMP":
        from .minting import GENESIS_TIMESTAMP

        return GENESIS_TIMESTAMP
    elif name == "GENESIS_METADATA":
        from .minting import GENESIS_METADATA

        return GENESIS_METADATA
    elif name == "is_genesis_node":
        from .minting import is_genesis_node

        return is_genesis_node
    elif name == "is_genesis_block":
        from .minting import is_genesis_block

        return is_genesis_block
    elif name == "mint_genesis_node":
        from .minting import mint_genesis_node

        return mint_genesis_node
    elif name == "get_genesis_info":
        from .minting import get_genesis_info

        return get_genesis_info

    raise AttributeError(f"module 'core.pat' has no attribute '{name}'")


__all__ = [
    # Bridge
    "PATBridge",
    "PATMessage",
    "ChannelAdapter",
    "PAT_GATEWAY_WS",
    "PAT_CHANNELS",
    # Identity Card
    "IdentityCard",
    "IdentityStatus",
    "SovereigntyTier",
    "generate_identity_keypair",
    # Agent
    "PATAgent",
    "SATAgent",
    "AgentType",
    "AgentStatus",
    "OwnershipType",
    # Minting
    "IdentityMinter",
    "MinterState",
    "OnboardingResult",
    "mint_identity_card",
    "mint_pat_agents",
    "mint_sat_agents",
    "onboard_user",
    "generate_and_onboard",
    # Genesis Block
    "GENESIS_NODE_ID",
    "GENESIS_BLOCK_NUMBER",
    "GENESIS_ARCHITECT",
    "GENESIS_TIMESTAMP",
    "GENESIS_METADATA",
    "is_genesis_node",
    "is_genesis_block",
    "mint_genesis_node",
    "get_genesis_info",
    # Constants
    "PAT_AGENT_COUNT",
    "SAT_AGENT_COUNT",
    "USER_AGENT_COUNT",
    "SYSTEM_AGENT_COUNT",
    "TOTAL_AGENTS_PER_USER",
]
