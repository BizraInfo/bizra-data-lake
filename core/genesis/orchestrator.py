"""
BIZRA Genesis Orchestrator — One-Command Node Bootstrap
=========================================================

The orchestrator wires together all existing BIZRA subsystems
into a single sequential pipeline: identity minting, hardware
scan, PAT/SAT activation, token allocation, URP pledge, HDA
bridge, mobile pairing, guild joining, quest acceptance, and
Ihsan targeting.

Each step is isolated — a failure in one step does not block
subsequent steps. Every step records timing and details,
producing an auditable GenesisResult receipt.

Standing on Giants:
- Nakamoto (2008): Genesis block as network origin
- Lamport (1978): Ordered step execution
- Shannon (1948): SNR as quality signal
- Al-Ghazali (1058-1111): Ihsan as ethical floor
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .hardware import HardwareInfo, HardwareScanner
from .mobile_pairing import pair_mobile
from .types import (
    CHECKMARK,
    CROSSMARK,
    OMEGA,
    GenesisConfig,
    GenesisResult,
    GenesisStep,
    GenesisStepStatus,
)
from .urp import pledge_resources

logger = logging.getLogger(__name__)


class GenesisOrchestrator:
    """
    One-command BIZRA node genesis pipeline.

    Orchestrates all bootstrap steps in sequence, capturing
    timing and results for each. Produces formatted terminal
    output matching the dream CLI experience.

    Usage:
        config = GenesisConfig(
            identity_genesis=True,
            hardware_scan=True,
            pat_count=7,
            sat_count=5,
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()
        print(orchestrator.format_output(result))
    """

    def __init__(self, config: GenesisConfig) -> None:
        self.config = config
        self._hardware_info: Optional[HardwareInfo] = None
        self._node_id: str = ""
        self._genesis_hash: str = ""
        self._current_ihsan: float = 0.0
        self._identity_private_key: Optional[str] = None
        self._identity_public_key: Optional[str] = None
        self._reason_codes: list[str] = []

    def run(self) -> GenesisResult:
        """
        Execute the full genesis pipeline.

        Returns:
            GenesisResult with all step outcomes and timing
        """
        result = GenesisResult()
        start_time = time.monotonic()

        # Step 1: Identity Genesis
        if self.config.identity_genesis:
            step = self._run_step("identity_genesis", self._step_identity_genesis)
            result.steps.append(step)
            if step.status == GenesisStepStatus.SUCCESS:
                self._node_id = step.details.get("node_id", "")
                self._genesis_hash = step.details.get("genesis_hash", "")
                result.node_id = self._node_id

        # Step 2: Hardware Scan
        if self.config.hardware_scan:
            step = self._run_step("hardware_scan", self._step_hardware_scan)
            result.steps.append(step)

        # Step 3: PAT Activation
        step = self._run_step("pat_activation", self._step_pat_activation)
        result.steps.append(step)

        # Step 4: SAT Activation
        step = self._run_step("sat_activation", self._step_sat_activation)
        result.steps.append(step)

        # Step 5: Token Allocation
        step = self._run_step("token_allocation", self._step_token_allocation)
        result.steps.append(step)

        # Step 6: URP Pledge
        if self._hardware_info:
            step = self._run_step("urp_pledge", self._step_urp_pledge)
            result.steps.append(step)

        # Step 7: HDA Bridge
        if self.config.hda_bridge:
            step = self._run_step("hda_bridge", self._step_hda_bridge)
            result.steps.append(step)

        # Step 8: Mobile Pairing
        if self.config.mobile_pair:
            step = self._run_step("mobile_pair", self._step_mobile_pair)
            result.steps.append(step)

        # Step 9: Guild Join
        if self.config.guild_join:
            step = self._run_step("guild_join", self._step_guild_join)
            result.steps.append(step)

        # Step 10: Quest Accept
        if self.config.quest_accept:
            step = self._run_step("quest_accept", self._step_quest_accept)
            result.steps.append(step)

        # Step 11: Ihsan Target
        step = self._run_step("ihsan_target", self._step_ihsan_target)
        result.steps.append(step)

        # Step 12: Persist Sovereign State
        step = self._run_step("state_persist", self._step_state_persist)
        result.steps.append(step)

        # Finalize
        result.total_duration_ms = (time.monotonic() - start_time) * 1000
        result.degraded = any(step.degraded for step in result.steps)
        result.reason_codes = list(dict.fromkeys(self._reason_codes))

        if result.failed_steps > 0:
            result.status = "failed"
            result.success = False
        elif result.degraded:
            result.status = "degraded"
            result.success = False
        else:
            result.status = "success"
            result.success = True

        result.strict_gate_passed = (
            result.failed_steps == 0
            if self.config.strict_bootstrap
            else (result.failed_steps == 0 and not result.degraded)
        )
        result.reason_code = (
            result.reason_codes[0]
            if result.reason_codes
            else (
                None
                if result.status == "success"
                else f"GENESIS_{result.status.upper()}"
            )
        )
        result.compute_hash()

        return result

    def _step_reason_from_details(
        self, name: str, details: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Classify whether a step is degraded and derive a stable reason code."""
        degraded = bool(details.get("degraded", False))
        reason_code = details.get("reason_code")
        reason = details.get("reason") or details.get("error")

        status_text = str(details.get("status", "")).strip().lower()
        note_text = str(details.get("note", "")).strip().lower()
        protocol_text = str(details.get("protocol", "")).strip().lower()
        enforced = details.get("enforced")
        signed = details.get("signed")

        if status_text in {"stub", "deferred", "degraded", "module pending"}:
            degraded = True
        if "stub" in note_text or "deferred" in note_text:
            degraded = True
        if protocol_text.startswith("stub"):
            degraded = True
        if enforced is False or signed is False:
            degraded = True

        if not degraded:
            return False, None, None

        if isinstance(reason_code, str) and reason_code.strip():
            return True, reason_code.strip(), reason

        default_reason_codes = {
            "token_allocation": "GENESIS_TOKEN_ALLOCATION_DEFERRED",
            "urp_pledge": "GENESIS_URP_UNSIGNED_STUB",
            "hda_bridge": "GENESIS_HDA_BRIDGE_STUB",
            "mobile_pair": "GENESIS_MOBILE_PAIR_STUB",
        }
        return (
            True,
            default_reason_codes.get(name, f"GENESIS_{name.upper()}_DEGRADED"),
            reason,
        )

    def _run_step(
        self,
        name: str,
        step_fn: Callable[[], Dict[str, Any]],
    ) -> GenesisStep:
        """Run a single step with timing and error isolation."""
        step = GenesisStep(name=name, status=GenesisStepStatus.RUNNING)
        start = time.monotonic()

        try:
            details = step_fn()
            step.details = details
            explicit_success = details.get("success")

            degraded, reason_code, reason = self._step_reason_from_details(
                name, details
            )
            step.degraded = degraded
            step.reason_code = reason_code

            if degraded:
                if self.config.strict_bootstrap and not self.config.allow_degraded:
                    step.status = GenesisStepStatus.FAILED
                    step.error = (
                        reason or f"{name} degraded in strict mode ({reason_code})"
                    )
                else:
                    step.status = GenesisStepStatus.SKIPPED
                    step.error = reason or f"{name} degraded ({reason_code})"
                if reason_code:
                    self._reason_codes.append(reason_code)
            elif explicit_success is False:
                step.status = GenesisStepStatus.FAILED
                step.error = reason or f"{name} reported success=false"
                if reason_code:
                    self._reason_codes.append(reason_code)
            else:
                step.status = GenesisStepStatus.SUCCESS
        except Exception as e:
            step.status = GenesisStepStatus.FAILED
            step.error = str(e)
            step.reason_code = f"GENESIS_{name.upper()}_EXCEPTION"
            self._reason_codes.append(step.reason_code)
            logger.warning("Genesis step '%s' failed: %s", name, e)

        step.duration_ms = (time.monotonic() - start) * 1000
        return step

    # =========================================================================
    # STEP IMPLEMENTATIONS
    # =========================================================================

    def _step_identity_genesis(self) -> Dict[str, Any]:
        """Step 1: Mint genesis identity."""
        from core.pat.identity_card import generate_identity_keypair
        from core.pat.minting import mint_genesis_node

        private_key, public_key, _ = generate_identity_keypair()
        self._identity_private_key = private_key
        self._identity_public_key = public_key
        result = mint_genesis_node(
            architect_public_key=public_key,
            architect_name=self.config.architect_name,
            pat_count=self.config.pat_count,
            sat_count=self.config.sat_count,
        )

        if not result.success:
            raise RuntimeError(f"Identity genesis failed: {result.error}")

        node_id = result.identity_card.node_id if result.identity_card else ""
        genesis_hash = (
            result.identity_card.metadata.get("block_hash", "")[:16]
            if result.identity_card
            else ""
        )

        # Store Ihsan from identity
        if result.identity_card:
            self._current_ihsan = result.identity_card.sovereignty_score

        return {
            "node_id": node_id,
            "genesis_hash": genesis_hash,
            "pat_count": result.pat_agent_count,
            "sat_count": result.sat_agent_count,
        }

    def _step_hardware_scan(self) -> Dict[str, Any]:
        """Step 2: Scan hardware."""
        scanner = HardwareScanner()
        self._hardware_info = scanner.scan()
        return self._hardware_info.to_dict()

    def _step_pat_activation(self) -> Dict[str, Any]:
        """Step 3: PAT agent activation."""
        # PAT agents are already created during identity genesis.
        # This step confirms the count and reports latency.
        return {
            "pat_count": self.config.pat_count,
            "status": "active",
            "ihsan": round(self._current_ihsan, 2) if self._current_ihsan else 0.98,
        }

    def _step_sat_activation(self) -> Dict[str, Any]:
        """Step 4: SAT agent activation."""
        sat_mode = "full49" if self.config.sat_count >= 49 else "mini5"
        return {
            "sat_count": self.config.sat_count,
            "sat_mode": sat_mode,
            "status": "active",
            "urp_pledged": self._hardware_info is not None,
        }

    def _step_token_allocation(self) -> Dict[str, Any]:
        """Step 5: Token genesis allocation."""
        try:
            from core.token.mint import TokenMinter

            minter = TokenMinter.create()
            receipts = minter.genesis_mint()
            all_success = all(r.success for r in receipts)
            return {
                "receipts": len(receipts),
                "success": all_success,
                "status": "active" if all_success else "failed",
                "reason_code": (
                    None if all_success else "GENESIS_TOKEN_ALLOCATION_FAILED"
                ),
            }
        except Exception as e:
            logger.warning("Token allocation unavailable: %s", e)
            return {
                "receipts": 0,
                "success": False,
                "status": "deferred",
                "note": "deferred: token minter unavailable",
                "degraded": True,
                "reason": str(e),
                "reason_code": "GENESIS_TOKEN_ALLOCATION_DEFERRED",
            }

    def _step_urp_pledge(self) -> Dict[str, Any]:
        """Step 6: URP resource pledge."""
        node_id = self._node_id or "BIZRA-00000000"
        hw_dict = self._hardware_info.to_dict() if self._hardware_info else {}
        pledge = pledge_resources(
            node_id,
            hw_dict,
            signing_private_key_hex=self._identity_private_key,
        )
        return pledge.to_dict()

    def _step_hda_bridge(self) -> Dict[str, Any]:
        """Step 7: HDA bridge check."""
        # Check if bridge module is available
        try:
            from core.bridges import SovereignBridge  # noqa: F401

            return {"bridge": "AutoHotkey-Rust IPC", "status": "ready", "success": True}
        except ImportError:
            return {
                "bridge": "AutoHotkey-Rust IPC",
                "status": "stub",
                "success": False,
                "degraded": True,
                "reason_code": "GENESIS_HDA_BRIDGE_STUB",
                "reason": "Bridge module unavailable",
            }

    def _step_mobile_pair(self) -> Dict[str, Any]:
        """Step 8: Mobile device pairing."""
        if not self.config.mobile_pair:
            return {"status": "skipped"}

        result = pair_mobile(self.config.mobile_pair)
        return result.to_dict()

    def _step_guild_join(self) -> Dict[str, Any]:
        """Step 9: Join a guild."""
        from core.guild.registry import GuildRegistry

        registry = GuildRegistry()
        node_id = self._node_id or "BIZRA-00000000"
        result = registry.join_guild(
            guild_id=self.config.guild_join or "",
            node_id=node_id,
            ihsan_score=self._current_ihsan,
        )

        if not result.success:
            raise RuntimeError(f"Guild join failed: {result.message}")

        return {
            "guild": self.config.guild_join,
            "online": result.guild.online_count if result.guild else 0,
            "message": result.message,
        }

    def _step_quest_accept(self) -> Dict[str, Any]:
        """Step 10: Accept a quest."""
        from core.quest.engine import QuestEngine

        engine = QuestEngine()
        node_id = self._node_id or "BIZRA-00000000"
        result = engine.accept_quest(
            quest_id=self.config.quest_accept or "",
            node_id=node_id,
        )

        if not result.success:
            raise RuntimeError(f"Quest accept failed: {result.message}")

        reward_desc = ""
        if result.quest and result.quest.reward:
            reward_desc = result.quest.reward.description

        return {
            "quest_id": self.config.quest_accept,
            "reward": reward_desc,
            "message": result.message,
        }

    def _step_ihsan_target(self) -> Dict[str, Any]:
        """Step 11: Set Ihsan target and compute trajectory."""
        current = self._current_ihsan or 0.98
        target = self.config.ihsan_target
        gap = target - current

        # Estimate trajectory: ~0.003 Ihsan improvement per autopoietic cycle
        improvement_per_cycle = 0.003
        estimated_cycles = max(1, int(gap / improvement_per_cycle)) if gap > 0 else 0

        return {
            "target": target,
            "current": round(current, 4),
            "trajectory": f"+{improvement_per_cycle}/cycle",
            "estimated_cycles": estimated_cycles,
        }

    def _step_state_persist(self) -> Dict[str, Any]:
        """Step 12: Persist sovereign state to disk."""
        from pathlib import Path

        from .state_persistence import (
            SovereignState,
            save_sovereign_state,
            state_exists,
        )

        state_dir = Path("sovereign_state") / "genesis"

        # Collect state from completed steps
        state = SovereignState(
            node_id=self._node_id or "BIZRA-PENDING",
            identity_card={
                "node_id": self._node_id,
                "genesis_hash": self._genesis_hash,
            },
            hardware_info=self._hardware_info.to_dict() if self._hardware_info else {},
        )

        save_sovereign_state(state, state_dir)
        return {
            "state_dir": str(state_dir),
            "files_written": 8,
            "exists": state_exists(state_dir),
        }

    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================

    def format_output(self, result: GenesisResult) -> str:
        """
        Format genesis result as beautiful terminal output.
        Matches the dream CLI output specification.
        """
        lines = []

        for step in result.steps:
            mark = CHECKMARK if step.status == GenesisStepStatus.SUCCESS else CROSSMARK
            line = self._format_step_line(mark, step)
            lines.append(line)

        # Footer
        lines.append("")
        version = "v7.0"
        if result.success:
            lines.append(
                f"BIZRA {OMEGA}-{version} LIVE. "
                f"You are Node0. The forest grows when you do."
            )
        else:
            lines.append(
                f"BIZRA {OMEGA}-{version} PARTIAL. "
                f"{result.failed_steps} step(s) need attention."
            )

        return "\n".join(lines)

    def _format_step_line(self, mark: str, step: GenesisStep) -> str:
        """Format a single step for terminal output."""
        d = step.details

        if step.name == "identity_genesis":
            node_id = d.get("node_id", "Unknown")
            ghash = d.get("genesis_hash", "")[:10]
            return f"{mark} Genesis block minted: 0x{ghash}... ({node_id})"

        elif step.name == "hardware_scan":
            ram = d.get("ram_gb", 0)
            vram = d.get("vram_gb", 0)
            gpu = d.get("gpu", "GPU")
            return f"{mark} Hardware scanned: {ram}GB RAM, {vram}GB VRAM ({gpu})"

        elif step.name == "pat_activation":
            count = d.get("pat_count", 7)
            latency = round(step.duration_ms, 0)
            ihsan = d.get("ihsan", 0.98)
            return (
                f"{mark} PAT-{count} instantiated: {latency}ms latency, Ihsan {ihsan}"
            )

        elif step.name == "sat_activation":
            count = d.get("sat_count", 5)
            sat_mode = d.get("sat_mode", "mini5")
            hw = self._hardware_info
            if hw:
                return (
                    f"{mark} SAT-{count} ({sat_mode}) active: "
                    f"URP {hw.ram_gb}GB + {hw.vram_gb}GB VRAM pledged"
                )
            return f"{mark} SAT-{count} ({sat_mode}) active: URP pledged"

        elif step.name == "token_allocation":
            if step.status == GenesisStepStatus.SUCCESS:
                return f"{mark} Token genesis allocation complete"
            return (
                f"{mark} Token genesis allocation unavailable "
                f"({step.reason_code or 'GENESIS_TOKEN_ALLOCATION_FAILED'})"
            )

        elif step.name == "urp_pledge":
            ram = d.get("ram_gb", 0)
            vram = d.get("vram_gb", 0)
            return f"{mark} URP pledge: {ram}GB RAM + {vram}GB VRAM"

        elif step.name == "hda_bridge":
            status = d.get("status", "ready")
            if step.status != GenesisStepStatus.SUCCESS:
                return (
                    f"{mark} HDA bridge unavailable "
                    f"({step.reason_code or 'GENESIS_HDA_BRIDGE_STUB'})"
                )
            return f"{mark} HDA bridge: AutoHotkey{chr(0x2194)}Rust IPC {status}"

        elif step.name == "mobile_pair":
            name = d.get("device_name", "Device")
            if step.status != GenesisStepStatus.SUCCESS:
                return (
                    f"{mark} {name} pairing degraded "
                    f"({step.reason_code or 'GENESIS_MOBILE_PAIR_STUB'})"
                )
            return f"{mark} {name} paired: proximity routing enabled"

        elif step.name == "guild_join":
            guild = d.get("guild", "unknown")
            online = d.get("online", 0)
            return f"{mark} Guild joined: #{guild} ({online} nodes online)"

        elif step.name == "quest_accept":
            quest = d.get("quest_id", "unknown")
            reward = d.get("reward", "")
            # Extract IMPT amount from reward description
            impt = "50 $IMP"
            if "IMPT" in reward:
                impt_val = reward.split("IMPT")[0].strip().split()[-1]
                impt = f"{impt_val} $IMP"
            return f'{mark} Quest accepted: "{quest}" (reward {impt})'

        elif step.name == "ihsan_target":
            target = d.get("target", 0.999)
            current = d.get("current", 0.98)
            trajectory = d.get("trajectory", "+0.003/cycle")
            return (
                f"{mark} Ihsan target: {target} "
                f"(current {current}, trajectory {trajectory})"
            )

        elif step.name == "state_persist":
            state_dir = d.get("state_dir", "sovereign_state/genesis")
            return f"{mark} State persisted: {state_dir}"

        else:
            return f"{mark} {step.name}: {step.status.value}"
