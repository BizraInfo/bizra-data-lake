# core/federation/sape_bridge.py - SAPE ↔ Federation Bridge
#
# Connects SAPE pattern elevation to the Pattern Federation Protocol.
# When SAPE auto-elevates a pattern (>3 repetitions), it flows through
# this bridge to the network.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.federation.protocol import PatternType, PatternEnvelope
from core.federation.federation import PatternFederation, FederationConfig


logger = logging.getLogger("federation.sape_bridge")


class SAPEFederationBridge:
    """
    Bridge between SAPE pattern elevation and Pattern Federation.
    
    This class provides the integration layer between:
    1. SAPE (core/sape.py) - local pattern detection
    2. Federation (core/federation/) - network pattern sharing
    
    Flow:
    ─────
    SAPE detects pattern → Bridge converts → Federation broadcasts
    Federation receives → Bridge converts → SAPE registers
    
    Usage:
    ──────
        from core.sape import SAPEEngine
        from core.federation.sape_bridge import SAPEFederationBridge
        
        # Create SAPE engine and federation bridge
        sape = SAPEEngine()
        bridge = SAPEFederationBridge()
        
        # Hook them together
        bridge.connect_sape(sape)
        
        # Start federation
        await bridge.start()
        
        # Now patterns flow automatically:
        # - Local elevations broadcast to network
        # - Network patterns integrate into local SAPE
    """
    
    def __init__(self, config: Optional[FederationConfig] = None):
        self.federation = PatternFederation(config)
        self._sape = None
        self._running = False
        
        # Statistics
        self.patterns_elevated = 0
        self.patterns_adopted = 0
    
    async def start(self):
        """Start the bridge and federation."""
        if self._running:
            return
        
        self._running = True
        await self.federation.start()
        
        logger.info("🔗 SAPE ↔ Federation Bridge active")
    
    async def stop(self):
        """Stop the bridge."""
        if not self._running:
            return
        
        self._running = False
        await self.federation.stop()
        
        logger.info("🔌 SAPE ↔ Federation Bridge stopped")
    
    def connect_sape(self, sape_engine):
        """
        Connect SAPE engine to federation.
        
        Args:
            sape_engine: Instance of core.sape.SAPEEngine
        """
        self._sape = sape_engine
        
        # Register callback for adopted patterns
        self.federation.on_pattern_adopted(self._on_pattern_adopted)
        
        # Monkey-patch SAPE to emit elevations
        self._hook_sape_elevation()
        
        logger.info("🔗 Connected to SAPE engine")
    
    def _hook_sape_elevation(self):
        """Hook into SAPE elevation to broadcast patterns."""
        if self._sape is None:
            return
        
        # Store original auto_elevate method
        original_auto_elevate = getattr(self._sape, 'auto_elevate', None)
        
        if original_auto_elevate is None:
            logger.warning("SAPE engine has no auto_elevate method to hook")
            return
        
        bridge = self  # Capture for closure
        
        def hooked_auto_elevate(sequence: List[str]):
            """Hooked version that broadcasts elevated patterns."""
            # Call original
            result = original_auto_elevate(sequence)
            
            # Broadcast if elevation occurred
            if result and bridge._running:
                asyncio.create_task(bridge._broadcast_elevation(sequence))
            
            return result
        
        # Replace method
        self._sape.auto_elevate = hooked_auto_elevate
        logger.debug("Hooked SAPE auto_elevate for federation")
    
    async def _broadcast_elevation(self, sequence: List[str]):
        """Broadcast an elevated pattern to the network."""
        try:
            # Get pattern info from SAPE
            pattern = self._find_sape_pattern(sequence)
            if pattern is None:
                return
            
            # Elevate to federation
            envelope = await self.federation.elevate_pattern(
                trigger_sequence=sequence,
                optimization=pattern.get("optimization", "SAPE-elevated pattern"),
                pattern_type=PatternType.SAPE_PROBE,
                repetition_count=pattern.get("activation_count", 3),
                success_rate=pattern.get("success_rate", 0.85),
                ihsan_score=pattern.get("ihsan_score", 0.92),
                latency_reduction_ms=pattern.get("latency_reduction_ms", 50),
                token_savings_percent=pattern.get("token_savings_percent", 20.0),
                snr_improvement=pattern.get("snr_improvement", 0.1),
            )
            
            self.patterns_elevated += 1
            logger.info(f"📡 Broadcast elevated pattern: {sequence[:2]}...")
            
        except Exception as e:
            logger.error(f"Failed to broadcast elevation: {e}")
    
    def _find_sape_pattern(self, sequence: List[str]) -> Optional[Dict[str, Any]]:
        """Find pattern info in SAPE engine."""
        if self._sape is None:
            return None
        
        # Look for matching pattern
        patterns = getattr(self._sape, 'patterns', {})
        
        for pattern_id, pattern in patterns.items():
            if getattr(pattern, 'trigger_sequence', []) == sequence:
                return {
                    "id": pattern_id,
                    "optimization": getattr(pattern, 'optimization', ''),
                    "activation_count": getattr(pattern, 'activation_count', 0),
                    "latency_reduction_ms": getattr(pattern, 'latency_reduction_ms', 0),
                    "token_savings_percent": getattr(pattern, 'token_savings_percent', 0),
                    "snr_improvement": getattr(pattern, 'snr_improvement', 0),
                }
        
        return None
    
    def _on_pattern_adopted(self, envelope: PatternEnvelope):
        """Handle pattern adopted from network."""
        if self._sape is None:
            return
        
        try:
            # Convert to SAPE format
            pattern = self._envelope_to_sape_pattern(envelope)
            
            # Register with SAPE
            register_fn = getattr(self._sape, 'register_pattern', None)
            if register_fn:
                register_fn(pattern)
                self.patterns_adopted += 1
                logger.info(f"📥 Adopted network pattern: {envelope.metadata.pattern_id[:16]}")
        
        except Exception as e:
            logger.error(f"Failed to adopt pattern: {e}")
    
    def _envelope_to_sape_pattern(self, envelope: PatternEnvelope) -> Dict[str, Any]:
        """Convert PatternEnvelope to SAPE ElevatedPattern format."""
        return {
            "id": envelope.metadata.pattern_id,
            "name": f"Network: {envelope.payload.trigger_sequence[0]}...",
            "trigger_sequence": envelope.payload.trigger_sequence,
            "optimization": envelope.payload.optimization,
            "snr_improvement": envelope.payload.snr_improvement,
            "latency_reduction_ms": envelope.payload.latency_reduction_ms,
            "token_savings_percent": envelope.payload.token_savings_percent,
            "activation_count": envelope.metadata.adoption_count,
            "origin": envelope.metadata.origin_node_id,
            "federated": True,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Manual Pattern Sharing
    # ─────────────────────────────────────────────────────────────────────────
    
    async def share_pattern(
        self,
        trigger_sequence: List[str],
        optimization: str,
        **kwargs,
    ) -> PatternEnvelope:
        """
        Manually share a pattern with the network.
        
        Useful for programmatic pattern sharing without SAPE detection.
        """
        return await self.federation.elevate_pattern(
            trigger_sequence=trigger_sequence,
            optimization=optimization,
            **kwargs,
        )
    
    def get_all_patterns(self) -> List[PatternEnvelope]:
        """Get all patterns (local + federated)."""
        return self.federation.get_all_patterns()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        fed_stats = self.federation.get_stats()
        return {
            "patterns_elevated": self.patterns_elevated,
            "patterns_adopted": self.patterns_adopted,
            "federation": fed_stats.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_global_bridge: Optional[SAPEFederationBridge] = None


def get_federation_bridge() -> SAPEFederationBridge:
    """Get or create global federation bridge."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = SAPEFederationBridge()
    return _global_bridge


async def start_federation(sape_engine=None) -> SAPEFederationBridge:
    """
    Start pattern federation with optional SAPE integration.
    
    This is the main entry point for enabling network effects.
    
    Example:
        from core.sape import get_sape_engine
        from core.federation.sape_bridge import start_federation
        
        sape = get_sape_engine()
        bridge = await start_federation(sape)
        
        # Now patterns flow between nodes!
    """
    bridge = get_federation_bridge()
    
    if sape_engine:
        bridge.connect_sape(sape_engine)
    
    await bridge.start()
    
    return bridge


async def share_pattern(
    trigger_sequence: List[str],
    optimization: str,
    **kwargs,
) -> PatternEnvelope:
    """
    Share a pattern with the network.
    
    Convenience function for pattern sharing.
    """
    bridge = get_federation_bridge()
    return await bridge.share_pattern(trigger_sequence, optimization, **kwargs)
