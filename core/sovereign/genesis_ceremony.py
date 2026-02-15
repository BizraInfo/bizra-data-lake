"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BIZRA GENESIS CEREMONY — The Birth of a Sovereign Node                    ║
║                                                                              ║
║  This is the moment of creation. When a human runs `bizra genesis`, they     ║
║  cross the threshold from user to sovereign. An Ed25519 keypair is born,     ║
║  seven PAT agents awaken, five SAT agents stand guard, and Block₀ is         ║
║  committed to the chain of trust.                                            ║
║                                                                              ║
║  Standing on Giants:                                                         ║
║    • Satoshi Nakamoto — Genesis block as founding act (2009)                 ║
║    • Daniel J. Bernstein — Ed25519 curve25519 (2012)                         ║
║    • Al-Ghazali — إحياء علوم الدين: revival through intention (1111)          ║
║    • General Magic — Mobile agents with identity (1990)                      ║
║                                                                              ║
║  Architecture:                                                               ║
║    OnboardingWizard (identity) → AgentActivator (PAT team) → Block₀          ║
║    → GuildRegistry (community) → QuestEngine (purpose) → Receipt             ║
║                                                                              ║
║  Design: Every step is fault-tolerant. A failed guild join does not block     ║
║  identity creation. The ceremony degrades gracefully — the essential act      ║
║  (keypair + PAT mint) always succeeds if the machine can generate entropy.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Genesis Block Constants ──────────────────────────────────────────────────
GENESIS_VERSION = "1.0.0"
GENESIS_CODENAME = "بذرة"  # Al-Badhrah — The Seed


@dataclass
class GenesisBlock:
    """Block₀ — the founding record of a sovereign node.

    Immutable once created. Contains the cryptographic anchor that all
    subsequent attestations chain back to.
    """

    version: str = GENESIS_VERSION
    codename: str = GENESIS_CODENAME
    node_id: str = ""
    public_key: str = ""
    pat_agent_ids: List[str] = field(default_factory=list)
    sat_agent_ids: List[str] = field(default_factory=list)
    timestamp_utc: str = ""
    hardware_fingerprint: str = ""
    guild_id: Optional[str] = None
    active_quests: List[str] = field(default_factory=list)
    sovereignty_tier: str = "SEED"
    sovereignty_score: float = 0.0
    parent_hash: str = "0" * 64  # Genesis has no parent
    nonce: int = 0

    def compute_hash(self) -> str:
        """Deterministic BLAKE3-style hash (SHA-256 fallback) of Block₀."""
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CeremonyResult:
    """Output of the genesis ceremony."""

    success: bool = False
    node_id: str = ""
    public_key: str = ""
    genesis_hash: str = ""
    pat_count: int = 0
    sat_count: int = 0
    guild_joined: Optional[str] = None
    quests_activated: int = 0
    sovereignty_tier: str = "SEED"
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ── ASCII Art ────────────────────────────────────────────────────────────────

CEREMONY_BANNER = """
\033[38;5;214m
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗  ║
    ║      ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝  ║
    ║      ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗  ║
    ║      ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║  ║
    ║      ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║  ║
    ║       ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝  ║
    ║                                                              ║
    ║              بسم الله الرحمن الرحيم                           ║
    ║                                                              ║
    ║          "Every seed has infinite potential."                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
\033[0m"""

SOVEREIGNTY_RECEIPT = """
\033[38;5;82m
    ╔══════════════════════════════════════════════════════════════╗
    ║                  SOVEREIGNTY RECEIPT                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Node ID:       {node_id:<42s}  ║
    ║  Public Key:    {pub_key_short:<42s}  ║
    ║  Genesis Hash:  {genesis_hash_short:<42s}  ║
    ║                                                              ║
    ║  PAT Agents:    {pat_count} awakened                                   ║
    ║  SAT Agents:    {sat_count} standing guard                             ║
    ║  Guild:         {guild:<42s}  ║
    ║  Quests:        {quests} activated                                     ║
    ║  Tier:          {tier} 🌱                                    ║
    ║                                                              ║
    ║  Duration:      {duration:<42s}  ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Your sovereign identity is yours alone.                     ║
    ║  Your private key never leaves this device.                  ║
    ║  Your PAT team serves only you.                              ║
    ║                                                              ║
    ║  الحمد لله الذي هدانا لهذا                                   ║
    ║  وما كنا لنهتدي لولا أن هدانا الله                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
\033[0m"""


def _step_print(step_num: int, total: int, msg: str, status: str = "...") -> None:
    """Print a ceremony step with progress indicator."""
    bar_len = 20
    filled = int(bar_len * step_num / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] Step {step_num}/{total}: {msg} {status}")


class GenesisCeremony:
    """Orchestrates the full genesis ceremony.

    Each step is isolated — a failure in step N does not block step N+1.
    The ceremony degrades gracefully: at minimum, you get identity + PAT.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        node_dir: Optional[Path] = None,
        guild_id: Optional[str] = None,
        interactive: bool = True,
        json_output: bool = False,
    ):
        self.name = name
        self.node_dir = node_dir or Path.home() / ".bizra-node"
        self.guild_id = guild_id
        self.interactive = interactive
        self.json_output = json_output
        self._errors: List[str] = []
        self._block: Optional[GenesisBlock] = None

    def run(self) -> CeremonyResult:
        """Execute the full genesis ceremony."""
        start = time.perf_counter()
        total_steps = 6
        result = CeremonyResult()

        if not self.json_output:
            print(CEREMONY_BANNER)
            print("  Initiating Genesis Ceremony...\n")

        # ── Step 1: Check for existing identity ──
        _step_print(1, total_steps, "Checking existing identity") if not self.json_output else None
        from ..pat.onboarding import OnboardingWizard

        wizard = OnboardingWizard(node_dir=self.node_dir)
        existing = wizard.load_existing_credentials()
        if existing:
            if not self.json_output:
                _step_print(1, total_steps, "Identity exists", "✓ (resuming)")
                print(f"    Node ID: {existing.node_id}")
            result.node_id = existing.node_id
            result.public_key = existing.public_key
            result.pat_count = len(existing.pat_agent_ids)
            result.sat_count = len(existing.sat_agent_ids)
            result.sovereignty_tier = existing.sovereignty_tier
            credentials = existing
        else:
            # ── Step 1b: Mint sovereign identity ──
            _step_print(1, total_steps, "Minting sovereign identity") if not self.json_output else None
            try:
                credentials = wizard.onboard(name=self.name)
                result.node_id = credentials.node_id
                result.public_key = credentials.public_key
                result.pat_count = len(credentials.pat_agent_ids)
                result.sat_count = len(credentials.sat_agent_ids)
                result.sovereignty_tier = credentials.sovereignty_tier
                if not self.json_output:
                    _step_print(1, total_steps, "Sovereign identity minted", "✓")
                    print(f"    Node ID: {credentials.node_id}")
            except Exception as e:
                self._errors.append(f"Identity minting failed: {e}")
                logger.error(f"Genesis step 1 failed: {e}")
                if not self.json_output:
                    _step_print(1, total_steps, "Identity minting", f"✗ ({e})")
                result.errors = self._errors
                result.duration_ms = (time.perf_counter() - start) * 1000
                return result

        # ── Step 2: Hardware fingerprint ──
        _step_print(2, total_steps, "Scanning hardware") if not self.json_output else None
        hw_fingerprint = ""
        try:
            from .hardware_covenant import HardwareFingerprint

            hw = HardwareFingerprint.capture()
            hw_fingerprint = hashlib.sha256(
                json.dumps(hw.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:16]
            if not self.json_output:
                _step_print(2, total_steps, "Hardware scanned", f"✓ ({hw_fingerprint})")
        except Exception as e:
            self._errors.append(f"Hardware scan failed: {e}")
            if not self.json_output:
                _step_print(2, total_steps, "Hardware scan", "⚠ (skipped)")

        # ── Step 3: Create Genesis Block ──
        _step_print(3, total_steps, "Forging Block₀") if not self.json_output else None
        try:
            import datetime

            self._block = GenesisBlock(
                node_id=result.node_id,
                public_key=result.public_key,
                pat_agent_ids=getattr(credentials, "pat_agent_ids", []),
                sat_agent_ids=getattr(credentials, "sat_agent_ids", []),
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hardware_fingerprint=hw_fingerprint,
                sovereignty_tier=result.sovereignty_tier,
                sovereignty_score=getattr(credentials, "sovereignty_score", 0.0),
            )
            result.genesis_hash = self._block.compute_hash()

            # Persist Block₀
            genesis_dir = self.node_dir / "genesis"
            genesis_dir.mkdir(parents=True, exist_ok=True)
            block_file = genesis_dir / "block_0.json"
            block_data = asdict(self._block)
            block_data["hash"] = result.genesis_hash
            block_file.write_text(json.dumps(block_data, indent=2))

            if not self.json_output:
                _step_print(3, total_steps, "Block₀ forged", f"✓ ({result.genesis_hash[:12]}...)")
        except Exception as e:
            self._errors.append(f"Block₀ creation failed: {e}")
            if not self.json_output:
                _step_print(3, total_steps, "Block₀", f"⚠ ({e})")

        # ── Step 4: Activate PAT agents ──
        _step_print(4, total_steps, "Awakening PAT team") if not self.json_output else None
        try:
            from .agent_activator import AgentActivator

            activator = AgentActivator()
            active = activator.activate_from_credentials(
                pat_agent_ids=getattr(credentials, "pat_agent_ids", []),
                sat_agent_ids=getattr(credentials, "sat_agent_ids", []),
                node_id=result.node_id,
            )
            result.pat_count = len([a for a in active if a.team == "PAT"])
            result.sat_count = len([a for a in active if a.team == "SAT"])
            if not self.json_output:
                _step_print(4, total_steps, "PAT team awakened", f"✓ ({result.pat_count} PAT + {result.sat_count} SAT)")
        except Exception as e:
            self._errors.append(f"Agent activation failed: {e}")
            if not self.json_output:
                _step_print(4, total_steps, "Agent activation", f"⚠ ({e})")

        # ── Step 5: Join guild ──
        _step_print(5, total_steps, "Joining guild") if not self.json_output else None
        try:
            from ..guild.registry import GuildRegistry

            registry = GuildRegistry()
            guild_to_join = self.guild_id or "guild_sovereigns"
            joined = registry.join_guild(guild_to_join, result.node_id)
            if joined:
                result.guild_joined = guild_to_join
                if not self.json_output:
                    guild_info = registry.get_guild(guild_to_join)
                    guild_name = guild_info.name if guild_info else guild_to_join
                    _step_print(5, total_steps, f"Joined {guild_name}", "✓")
            else:
                if not self.json_output:
                    _step_print(5, total_steps, "Guild join", "⚠ (already member)")
                result.guild_joined = guild_to_join
        except Exception as e:
            self._errors.append(f"Guild join failed: {e}")
            if not self.json_output:
                _step_print(5, total_steps, "Guild join", f"⚠ ({e})")

        # ── Step 6: Activate quests ──
        _step_print(6, total_steps, "Activating quests") if not self.json_output else None
        try:
            from ..quest.engine import QuestEngine

            engine = QuestEngine()
            available = engine.get_available_quests(result.node_id)
            result.quests_activated = len(available)
            if not self.json_output:
                _step_print(6, total_steps, f"{len(available)} quests available", "✓")
        except Exception as e:
            self._errors.append(f"Quest activation failed: {e}")
            if not self.json_output:
                _step_print(6, total_steps, "Quest activation", f"⚠ ({e})")

        # ── Ceremony Complete ──
        result.success = bool(result.node_id)  # Success if we have an identity
        result.errors = self._errors
        result.duration_ms = (time.perf_counter() - start) * 1000

        if not self.json_output:
            print()
            if result.success:
                self._print_receipt(result)
            else:
                print("\n  ✗ Genesis ceremony failed. Check errors above.\n")
        
        return result

    def _print_receipt(self, result: CeremonyResult) -> None:
        """Print the sovereignty receipt."""
        pub_short = f"{result.public_key[:20]}...{result.public_key[-8:]}" if len(result.public_key) > 30 else result.public_key
        hash_short = f"{result.genesis_hash[:20]}...{result.genesis_hash[-8:]}" if len(result.genesis_hash) > 30 else result.genesis_hash
        duration = f"{result.duration_ms:.0f}ms"

        print(SOVEREIGNTY_RECEIPT.format(
            node_id=result.node_id,
            pub_key_short=pub_short,
            genesis_hash_short=hash_short,
            pat_count=result.pat_count,
            sat_count=result.sat_count,
            guild=result.guild_joined or "none",
            quests=result.quests_activated,
            tier=result.sovereignty_tier,
            duration=duration,
        ))


def run_genesis_ceremony(
    name: Optional[str] = None,
    node_dir: Optional[str] = None,
    guild: Optional[str] = None,
    json_output: bool = False,
) -> None:
    """Entry point for `bizra genesis` CLI command."""
    dir_path = Path(node_dir) if node_dir else None
    ceremony = GenesisCeremony(
        name=name,
        node_dir=dir_path,
        guild_id=guild,
        json_output=json_output,
    )
    result = ceremony.run()

    if json_output:
        print(json.dumps(asdict(result), indent=2))

    if not result.success:
        import sys
        sys.exit(1)
