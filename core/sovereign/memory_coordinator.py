"""
Memory Coordinator — Unified Auto-Save & Persistent Memory Engine
=================================================================
Coordinates all memory subsystems for Node0 auto-persistence:
- StateCheckpointer (runtime state)
- LivingMemoryCore (episodic/semantic/procedural/working/prospective)
- ProactiveSovereignEntity state (scheduler, monitor, agents, goals)
- Genesis identity binding (every save stamped with node_id)

Auto-save loop runs in the background, saving all memory subsystems
at configurable intervals. On startup, restores last known state.

Standing on Giants: Event Sourcing + Living Memory + Checkpointing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .state_checkpointer import StateCheckpointer, StorageBackend

logger = logging.getLogger(__name__)


class RestorePriority(IntEnum):
    """Priority for state restoration ordering.

    Safety-critical state restores first (lower number = higher priority).
    Cherry-picked from Proposal 3 (AMNESTIA) — priority-aware warm-start.
    """

    SAFETY = 0  # Rate limiters, constitutional filters (MUST restore first)
    CORE = 1  # Runtime metrics, component state
    QUALITY = 2  # Trend baselines, predictions, optimizations
    AUXILIARY = 3  # Preferences, cosmetic state


@dataclass
class MemoryCoordinatorConfig:
    """Configuration for the unified memory coordinator."""

    state_dir: Path = field(default_factory=lambda: Path("sovereign_state"))
    auto_save_interval: float = 120.0  # seconds between auto-saves
    checkpoint_backend: StorageBackend = StorageBackend.FILE
    max_checkpoints: int = 20
    enable_living_memory: bool = True
    enable_proactive_state: bool = True
    living_memory_path: Optional[Path] = None


class MemoryCoordinator:
    """
    Unified memory coordinator for Node0.

    Manages auto-save across all memory subsystems and provides
    a single interface for save/restore operations.

    Usage:
        coordinator = MemoryCoordinator(config)
        await coordinator.initialize(genesis=genesis_state)
        await coordinator.start_auto_save(state_providers)
    """

    def __init__(self, config: Optional[MemoryCoordinatorConfig] = None):
        self.config = config or MemoryCoordinatorConfig()

        # Core checkpointer (delegates to StateCheckpointer for versioned snapshots)
        self._checkpointer = StateCheckpointer(
            checkpoint_dir=self.config.state_dir / "checkpoints",
            max_checkpoints=self.config.max_checkpoints,
            auto_interval_seconds=self.config.auto_save_interval,
            backend=self.config.checkpoint_backend,
        )

        # Living memory (lazy — initialized when register_living_memory called)
        self._living_memory: Optional[object] = None

        # Genesis identity stamp
        self._node_id: Optional[str] = None
        self._node_name: Optional[str] = None

        # State providers (registered by runtime components)
        # Maps name -> (provider_fn, priority)
        self._state_providers: Dict[
            str, Tuple[Callable[[], Dict[str, Any]], RestorePriority]
        ] = {}

        # State
        self._running = False
        self._save_count = 0
        self._last_save: Optional[datetime] = None
        self._auto_save_task: Optional[asyncio.Task] = None

    def initialize(
        self,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> None:
        """Initialize the memory coordinator with genesis identity."""
        self._node_id = node_id
        self._node_name = node_name

        # Ensure directories exist
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MemoryCoordinator initialized for {node_id or 'ephemeral'}"
            f" (auto-save: {self.config.auto_save_interval}s,"
            f" backend: {self.config.checkpoint_backend.name})"
        )

    def register_state_provider(
        self,
        name: str,
        provider: Callable[[], Dict[str, Any]],
        priority: RestorePriority = RestorePriority.CORE,
    ) -> None:
        """Register a state provider for checkpoint collection.

        Args:
            name: Unique provider name (e.g. "runtime", "scheduler")
            provider: Callable returning state dict
            priority: Restore priority — SAFETY restores first
        """
        self._state_providers[name] = (provider, priority)
        logger.debug(f"Registered state provider: {name} (priority={priority.name})")

    def register_living_memory(self, living_memory: object) -> None:
        """Register the LivingMemoryCore instance for auto-save."""
        self._living_memory = living_memory
        logger.debug("LivingMemoryCore registered for auto-save")

    async def save_all(self, source: str = "manual") -> bool:
        """
        Save all memory subsystems.

        Collects state from all registered providers, saves living memory,
        and creates a versioned checkpoint.
        """
        try:
            # Collect state from all providers
            state: Dict[str, Any] = {
                "coordinator": {
                    "node_id": self._node_id,
                    "node_name": self._node_name,
                    "save_count": self._save_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

            for name, (provider, priority) in self._state_providers.items():
                try:
                    state[name] = provider()
                    state[name]["_restore_priority"] = priority.value
                except Exception as e:
                    logger.warning(f"State provider '{name}' failed: {e}")
                    state[name] = {"error": str(e)}

            # Save living memory (persists to its own JSONL file)
            if self._living_memory is not None:
                try:
                    save_fn = getattr(self._living_memory, "_save_memories", None)
                    if save_fn is not None:
                        await save_fn()
                    # Include stats in checkpoint
                    stats_fn = getattr(self._living_memory, "get_stats", None)
                    if stats_fn is not None:
                        stats = stats_fn()
                        state["living_memory"] = (
                            stats.to_dict() if hasattr(stats, "to_dict") else {}
                        )
                except Exception as e:
                    logger.warning(f"Living memory save failed: {e}")

            # Create versioned checkpoint
            checkpoint = await self._checkpointer.checkpoint(
                state=state,
                source=source,
                metadata={
                    "node_id": self._node_id,
                    "providers": list(self._state_providers.keys()),
                },
            )

            self._save_count += 1
            self._last_save = datetime.now(timezone.utc)

            logger.info(
                f"Memory saved: {checkpoint.id} "
                f"({len(self._state_providers)} providers, "
                f"checksum: {checkpoint.checksum})"
            )
            return True

        except Exception as e:
            logger.error(f"save_all failed: {e}")
            return False

    async def restore_latest(self) -> Optional[Dict[str, Any]]:
        """Restore the latest checkpoint state."""
        checkpoint = await self._checkpointer.restore()
        if checkpoint is None:
            logger.info("No checkpoint found to restore")
            return None

        logger.info(
            f"Restored checkpoint {checkpoint.id} "
            f"(version {checkpoint.version}, source: {checkpoint.source})"
        )
        return checkpoint.state

    async def start_auto_save(self) -> None:
        """Start the background auto-save loop."""
        if self._running:
            return

        self._running = True
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"Auto-save started (interval: {self.config.auto_save_interval}s)")

    async def _auto_save_loop(self) -> None:
        """Background loop that periodically saves all memory."""
        while self._running:
            await asyncio.sleep(self.config.auto_save_interval)
            if self._running:
                await self.save_all(source="auto")

    async def stop(self) -> None:
        """Stop auto-save and perform final save."""
        if not self._running:
            return

        self._running = False

        if self._auto_save_task is not None:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            self._auto_save_task = None

        # Final save on shutdown
        await self.save_all(source="shutdown")
        self._checkpointer.close()
        logger.info("MemoryCoordinator stopped (final save complete)")

    def restore_by_priority(
        self, state: Dict[str, Any]
    ) -> List[Tuple[RestorePriority, str, Dict[str, Any]]]:
        """Return state sections sorted by restore priority (SAFETY first).

        Args:
            state: Full state dict from restore_latest()

        Returns:
            List of (priority, name, state_dict) sorted by priority
        """
        prioritized: List[Tuple[RestorePriority, str, Dict[str, Any]]] = []
        for name, section in state.items():
            if isinstance(section, dict):
                prio_val = section.pop("_restore_priority", RestorePriority.CORE.value)
                try:
                    prio = RestorePriority(prio_val)
                except ValueError:
                    prio = RestorePriority.CORE
                prioritized.append((prio, name, section))
        prioritized.sort(key=lambda x: x[0].value)
        return prioritized

    def stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "node_id": self._node_id,
            "running": self._running,
            "save_count": self._save_count,
            "last_save": self._last_save.isoformat() if self._last_save else None,
            "auto_save_interval": self.config.auto_save_interval,
            "providers": list(self._state_providers.keys()),
            "living_memory_registered": self._living_memory is not None,
            "checkpointer": self._checkpointer.stats(),
        }


__all__ = [
    "MemoryCoordinator",
    "MemoryCoordinatorConfig",
    "RestorePriority",
]
