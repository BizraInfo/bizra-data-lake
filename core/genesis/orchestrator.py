"""
BIZRA Genesis Orchestrator — One-Command Sovereign Node Bootstrap
==================================================================

Executes the full 8-step homeostatic genesis protocol:

  Step 1: identity_genesis   — Mint or load node identity (PAT-7 + SAT-5)
  Step 2: hardware_scan      — Fingerprint CPU/GPU/RAM covenant
  Step 3: hda_bridge         — Hardware Data Attestation (stub, future)
  Step 4: urp_pledge         — Universal Resource Pool resource pledge
  Step 5: mobile_pair        — Companion device pairing (stub, future)
  Step 6: guild_join         — Join an impact-domain guild
  Step 7: quest_accept       — Accept an impact mission quest
  Step 8: ihsan_target       — Confirm constitutional Ihsan target

Each step produces a GenesisStep record. Failure in any step is non-fatal
unless marked critical — remaining steps proceed or skip gracefully.
The genesis_hash commits the full covenant in a single SHA-256 over all
step outputs, binding identity + hardware + community + intent.

Standing on Giants:
- Wiener (1948, Cybernetics): Bootstrap as homeostatic convergence to
  sovereignty equilibrium
- Al-Ghazali (1095, Ihya Ulum al-Din): Covenant = technical + ethical bond
- Shannon (1948): Each step reduces identity uncertainty entropy
- Deming (1950, PDCA): Step-gated quality — Plan→Do→Check before advancing
- Szabo (1997, Smart Contracts): Deterministic, verifiable genesis steps

v1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .hardware import HardwareScanner
from .mobile_pairing import pair_mobile
from .types import GenesisConfig, GenesisResult, GenesisStep
from .urp import pledge_resources

logger = logging.getLogger(__name__)


class GenesisOrchestrator:
    """
    One-command genesis bootstrap — from zero to sovereign node.

    Executes 8 sequential steps, each yielding a GenesisStep record.
    Steps that are disabled via config are marked 'skipped'.
    Steps that fail are marked 'failed' with error detail.
    The genesis_hash is computed over all non-skipped step outputs.

    Usage:
        config = GenesisConfig(
            identity_genesis=True,
            hardware_scan=True,
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
        )
        orchestrator = GenesisOrchestrator()
        result = orchestrator.run(config)
        if result.success:
            print(f"Genesis complete: {result.node_id}")
    """

    def __init__(self) -> None:
        self._scanner = HardwareScanner()
        self._hardware_info: Optional[Dict[str, Any]] = None

    def run(self, config: GenesisConfig) -> GenesisResult:
        """
        Execute the full genesis bootstrap protocol.

        Args:
            config: GenesisConfig specifying which steps to run

        Returns:
            GenesisResult with step records, node_id, genesis_hash
        """
        result = GenesisResult()
        start_time = time.monotonic()
        hash_inputs: list[str] = []

        logger.info("╔══ BIZRA Genesis Protocol v1.0.0 ══╗")

        # Step 1: Identity Genesis
        step = self._run_identity_genesis(config)
        result.steps.append(step)
        if step.success:
            result.node_id = step.details.get("node_id", "")
            hash_inputs.append(json.dumps(step.details, sort_keys=True))
        elif step.status != "skipped":
            logger.warning("⚠ Identity genesis failed — proceeding with partial genesis")

        # Step 2: Hardware Scan
        step = self._run_hardware_scan(config)
        result.steps.append(step)
        if step.success:
            self._hardware_info = step.details
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 3: HDA Bridge (stub)
        step = self._run_hda_bridge(config)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 4: URP Pledge
        step = self._run_urp_pledge(config, result.node_id)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 5: Mobile Pair (stub)
        step = self._run_mobile_pair(config)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 6: Guild Join
        step = self._run_guild_join(config, result.node_id)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 7: Quest Accept
        step = self._run_quest_accept(config, result.node_id)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Step 8: Ihsan Target
        step = self._run_ihsan_target(config)
        result.steps.append(step)
        if step.success:
            hash_inputs.append(json.dumps(step.details, sort_keys=True))

        # Compute genesis_hash: SHA-256 over all non-skipped step outputs
        if hash_inputs:
            covenant_data = "|".join(hash_inputs)
            result.genesis_hash = hashlib.sha256(
                covenant_data.encode()
            ).hexdigest()[:32]

        result.total_duration_ms = (time.monotonic() - start_time) * 1000
        result.success = result.failed_count == 0 and result.step_count > 0

        logger.info(
            "╚══ Genesis %s: %d/%d steps OK | hash=%s | %.0fms ══╝",
            "COMPLETE" if result.success else "PARTIAL",
            result.success_count,
            result.step_count,
            result.genesis_hash[:12] + "..." if result.genesis_hash else "none",
            result.total_duration_ms,
        )

        return result

    # ─────────────────────────────────────────────────────────────
    # Step implementations
    # ─────────────────────────────────────────────────────────────

    def _run_identity_genesis(self, config: GenesisConfig) -> GenesisStep:
        step = GenesisStep(name="identity_genesis")
        if not config.identity_genesis:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            from core.pat.minting import generate_and_onboard

            private_key, public_key, onboarding = generate_and_onboard()
            if not onboarding.success:
                step.status = "failed"
                step.error = "Onboarding returned success=False"
                return step

            node_id = (
                onboarding.identity_card.node_id
                if onboarding.identity_card
                else "unknown"
            )
            step.details = {
                "node_id": node_id,
                "public_key": public_key[:16] + "...",
                "pat_count": len(onboarding.pat_agents),
                "sat_count": len(onboarding.sat_agents),
            }
            step.status = "success"
            logger.info(
                "✓ identity_genesis: %s (PAT=%d, SAT=%d)",
                node_id,
                len(onboarding.pat_agents),
                len(onboarding.sat_agents),
            )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ identity_genesis: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_hardware_scan(self, config: GenesisConfig) -> GenesisStep:
        step = GenesisStep(name="hardware_scan")
        if not config.hardware_scan:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            info = self._scanner.scan()
            step.details = info
            step.status = "success"
            logger.info(
                "✓ hardware_scan: %s | GPU=%s | RAM=%.1fGB | fp=%s",
                info.get("cpu", "?")[:30],
                info.get("gpu", "?")[:20],
                info.get("ram_gb", 0.0),
                info.get("fingerprint", "?")[:12],
            )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ hardware_scan: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_hda_bridge(self, config: GenesisConfig) -> GenesisStep:
        """HDA Bridge — stub. Future: Hardware Data Attestation via TPM/ZKP."""
        step = GenesisStep(name="hda_bridge")
        if not config.hda_bridge:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        # Stub: returns placeholder attestation. Real implementation in bizra-omega.
        step.details = {
            "attestation": "stub",
            "tpm_available": False,
            "zpk_ready": False,
        }
        step.status = "success"
        step.duration_ms = (time.monotonic() - t0) * 1000
        logger.info("✓ hda_bridge: stub attestation (TPM/ZKP future)")
        return step

    def _run_urp_pledge(self, config: GenesisConfig, node_id: str) -> GenesisStep:
        """Universal Resource Pool — pledge compute resources."""
        step = GenesisStep(name="urp_pledge")
        if not config.hardware_scan and not self._hardware_info:
            # Can't pledge without hardware info — skip gracefully
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            hardware = self._hardware_info or {}
            pledge = pledge_resources(
                node_id=node_id or "BIZRA-UNKNOWN",
                hardware_info=hardware,
            )
            step.details = pledge.to_dict()
            step.status = "success"
            logger.info(
                "✓ urp_pledge: %.1fGB RAM pledged | hash=%s",
                pledge.ram_gb,
                pledge.pledge_hash[:12],
            )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ urp_pledge: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_mobile_pair(self, config: GenesisConfig) -> GenesisStep:
        """Mobile Pairing — stub. Future: BLE/NFC device companion."""
        step = GenesisStep(name="mobile_pair")
        if not config.mobile_pair:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            pair_result = pair_mobile(config.mobile_pair)
            step.details = pair_result.to_dict()
            step.status = "success"
            logger.info(
                "✓ mobile_pair: %s (%s) paired",
                pair_result.device_name,
                pair_result.model,
            )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ mobile_pair: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_guild_join(self, config: GenesisConfig, node_id: str) -> GenesisStep:
        """Guild Join — join an impact-domain community."""
        step = GenesisStep(name="guild_join")
        if not config.guild_join:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            from core.guild.registry import GuildRegistry

            registry = GuildRegistry()
            join_result = registry.join_guild(
                guild_id=config.guild_join,
                node_id=node_id or "BIZRA-UNKNOWN",
                role="member",
            )
            if not join_result.success:
                step.status = "failed"
                step.error = join_result.message
            else:
                guild = join_result.guild
                step.details = {
                    "guild_id": config.guild_join,
                    "guild_name": guild.name if guild else "",
                    "member_count": guild.member_count if guild else 0,
                }
                step.status = "success"
                logger.info(
                    "✓ guild_join: '%s' (%d members)",
                    step.details["guild_name"],
                    step.details["member_count"],
                )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ guild_join: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_quest_accept(self, config: GenesisConfig, node_id: str) -> GenesisStep:
        """Quest Accept — accept an impact mission."""
        step = GenesisStep(name="quest_accept")
        if not config.quest_accept:
            step.status = "skipped"
            return step

        t0 = time.monotonic()
        try:
            from core.quest.engine import QuestEngine

            engine = QuestEngine()
            accept_result = engine.accept_quest(
                quest_id=config.quest_accept,
                node_id=node_id or "BIZRA-UNKNOWN",
            )
            if not accept_result.success:
                step.status = "failed"
                step.error = accept_result.message
            else:
                quest = accept_result.quest
                step.details = {
                    "quest_id": config.quest_accept,
                    "quest_title": quest.title if quest else "",
                    "reward": quest.reward.description if quest else "",
                }
                step.status = "success"
                logger.info(
                    "✓ quest_accept: '%s' | reward: %s",
                    step.details["quest_title"],
                    step.details["reward"],
                )
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            logger.warning("✗ quest_accept: %s", exc)
        finally:
            step.duration_ms = (time.monotonic() - t0) * 1000

        return step

    def _run_ihsan_target(self, config: GenesisConfig) -> GenesisStep:
        """Ihsan Target — record constitutional excellence commitment."""
        step = GenesisStep(name="ihsan_target")
        t0 = time.monotonic()

        target = config.ihsan_target
        threshold = UNIFIED_IHSAN_THRESHOLD

        step.details = {
            "target": target,
            "constitutional_threshold": threshold,
            "delta": round(target - threshold, 4),
            "commitment": "Ihsān (إحسان) — do everything as if you see God",
        }
        step.status = "success"
        step.duration_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "✓ ihsan_target: %.4f (threshold=%.2f, delta=+%.4f)",
            target,
            threshold,
            target - threshold,
        )
        return step
