# core/federation/federation.py - Pattern Federation Coordinator
#
# Main entry point for pattern federation.
# Integrates gossip protocol, consensus, and SAPE elevation.

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.federation.protocol import (
    PatternEnvelope,
    PatternMetadata,
    PatternPayload,
    PatternType,
    generate_keypair,
    PATTERN_TTL_SECONDS,
    MIN_REPETITIONS,
    MIN_IMPACT_SCORE,
)
from core.federation.gossip import GossipProtocol, PeerInfo
from core.federation.consensus import PatternConsensus, ConsensusResult


logger = logging.getLogger("federation")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FederationConfig:
    """Configuration for pattern federation."""
    
    # Node identity
    node_id: str = field(default_factory=lambda: f"node_{secrets.token_hex(8)}")
    
    # Network
    host: str = "0.0.0.0"
    port: int = 9999
    
    # Paths
    patterns_dir: Path = field(default_factory=lambda: Path("data/federation/patterns"))
    keys_dir: Path = field(default_factory=lambda: Path("data/federation/keys"))
    
    # Behavior
    auto_propagate: bool = True  # Automatically propagate elevated patterns
    auto_adopt: bool = True  # Automatically adopt consensus-accepted patterns
    
    # Limits
    max_patterns: int = 10000
    max_peers: int = 50
    
    @classmethod
    def from_env(cls) -> "FederationConfig":
        """Load config from environment."""
        return cls(
            node_id=os.getenv("BIZRA_NODE_ID", f"node_{secrets.token_hex(8)}"),
            host=os.getenv("BIZRA_FED_HOST", "0.0.0.0"),
            port=int(os.getenv("BIZRA_FED_PORT", "9999")),
            patterns_dir=Path(os.getenv("BIZRA_FED_PATTERNS", "data/federation/patterns")),
            keys_dir=Path(os.getenv("BIZRA_FED_KEYS", "data/federation/keys")),
        )


@dataclass
class FederationStats:
    """Federation statistics."""
    
    # Patterns
    local_patterns: int = 0
    federated_patterns: int = 0
    patterns_sent: int = 0
    patterns_received: int = 0
    
    # Consensus
    consensus_proposed: int = 0
    consensus_accepted: int = 0
    consensus_rejected: int = 0
    
    # Network
    connected_peers: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    # Performance
    avg_consensus_time_ms: float = 0.0
    network_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_patterns": self.local_patterns,
            "federated_patterns": self.federated_patterns,
            "patterns_sent": self.patterns_sent,
            "patterns_received": self.patterns_received,
            "consensus_proposed": self.consensus_proposed,
            "consensus_accepted": self.consensus_accepted,
            "consensus_rejected": self.consensus_rejected,
            "connected_peers": self.connected_peers,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "avg_consensus_time_ms": self.avg_consensus_time_ms,
            "network_multiplier": self.network_multiplier,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN FEDERATION
# ═══════════════════════════════════════════════════════════════════════════════

class PatternFederation:
    """
    Pattern Federation Coordinator.
    
    Manages the lifecycle of federated patterns:
    1. Elevation: SAPE elevates local pattern
    2. Broadcast: Gossip announces pattern to network
    3. Consensus: Validators vote on pattern
    4. Adoption: Accepted patterns integrated into local SAPE
    
    Network Effect Activation:
    ─────────────────────────
    As nodes join and share patterns, the collective intelligence grows.
    Each node benefits from discoveries made anywhere in the network.
    
    Value ∝ n² (Metcalfe's Law) when patterns are shared.
    
    Usage:
        fed = PatternFederation(config)
        await fed.start()
        
        # When SAPE elevates a pattern
        await fed.elevate_pattern(pattern)
        
        # Patterns from network automatically integrate
    """
    
    def __init__(self, config: Optional[FederationConfig] = None):
        self.config = config or FederationConfig.from_env()
        
        # Ensure directories exist
        self.config.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.config.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or generate keys
        self._private_key, self._public_key = self._load_or_generate_keys()
        
        # Pattern storage
        self.local_patterns: Dict[str, PatternEnvelope] = {}
        self.federated_patterns: Dict[str, PatternEnvelope] = {}
        
        # Protocol components
        self.gossip = GossipProtocol(
            node_id=self.config.node_id,
            host=self.config.host,
            port=self.config.port,
        )
        
        self.consensus = PatternConsensus(
            node_id=self.config.node_id,
            private_key=self._private_key,
            public_key=self._public_key,
        )
        
        # Callbacks for SAPE integration
        self._on_pattern_adopted: Optional[Callable[[PatternEnvelope], None]] = None
        
        # Statistics
        self.stats = FederationStats()
        
        # Setup internal callbacks
        self._setup_callbacks()
        
        # Running state
        self._running = False
    
    def _load_or_generate_keys(self) -> tuple[bytes, bytes]:
        """Load existing keys or generate new ones."""
        private_path = self.config.keys_dir / "node.key"
        public_path = self.config.keys_dir / "node.pub"
        
        if private_path.exists() and public_path.exists():
            private_key = private_path.read_bytes()
            public_key = public_path.read_bytes()
            logger.info(f"🔑 Loaded existing keypair for {self.config.node_id}")
        else:
            private_key, public_key = generate_keypair()
            private_path.write_bytes(private_key)
            public_path.write_bytes(public_key)
            logger.info(f"🔑 Generated new keypair for {self.config.node_id}")
        
        return private_key, public_key
    
    def _setup_callbacks(self):
        """Setup internal protocol callbacks."""
        
        # When gossip receives a pattern
        def on_pattern_received(envelope: PatternEnvelope):
            asyncio.create_task(self._handle_received_pattern(envelope))
        
        self.gossip.on_pattern_received(on_pattern_received)
        
        # When consensus accepts a pattern
        def on_pattern_accepted(envelope: PatternEnvelope):
            self._adopt_pattern(envelope)
        
        self.consensus.on_pattern_accepted(on_pattern_accepted)
        
        # Pattern getter for gossip requests
        def get_pattern(pattern_id: str) -> Optional[PatternEnvelope]:
            return self.local_patterns.get(pattern_id) or self.federated_patterns.get(pattern_id)
        
        self.gossip.set_pattern_getter(get_pattern)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    
    async def start(self):
        """Start federation services."""
        if self._running:
            return
        
        self._running = True
        
        # Load persisted patterns
        self._load_patterns()
        
        # Start gossip protocol
        await self.gossip.start()
        
        logger.info(f"🌐 Pattern Federation started on port {self.config.port}")
        logger.info(f"   Node ID: {self.config.node_id}")
        logger.info(f"   Local patterns: {len(self.local_patterns)}")
        logger.info(f"   Federated patterns: {len(self.federated_patterns)}")
    
    async def stop(self):
        """Stop federation services."""
        if not self._running:
            return
        
        self._running = False
        
        # Save patterns
        self._save_patterns()
        
        # Stop gossip
        await self.gossip.stop()
        
        logger.info("🔌 Pattern Federation stopped")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pattern Elevation (SAPE Integration)
    # ─────────────────────────────────────────────────────────────────────────
    
    async def elevate_pattern(
        self,
        trigger_sequence: List[str],
        optimization: str,
        pattern_type: PatternType = PatternType.SAPE_PROBE,
        repetition_count: int = MIN_REPETITIONS,
        success_rate: float = 0.8,
        ihsan_score: float = 0.95,
        latency_reduction_ms: int = 50,
        token_savings_percent: float = 20.0,
        snr_improvement: float = 0.1,
        tags: Optional[List[str]] = None,
    ) -> PatternEnvelope:
        """
        Elevate a locally discovered pattern and broadcast to network.
        
        This is the main integration point with SAPE.
        When SAPE detects a pattern worth elevating, it calls this method.
        
        Args:
            trigger_sequence: The probe/action sequence that triggers the pattern
            optimization: Description of the optimization
            pattern_type: Type of pattern
            repetition_count: How many times pattern was observed
            success_rate: Success rate [0-1]
            ihsan_score: Ihsān compliance score
            latency_reduction_ms: Latency saved by pattern
            token_savings_percent: Token reduction percentage
            snr_improvement: SNR improvement
            tags: Searchable tags
            
        Returns:
            PatternEnvelope: The created pattern envelope
        """
        # Create pattern payload
        payload = PatternPayload(
            trigger_sequence=trigger_sequence,
            optimization=optimization,
            latency_reduction_ms=latency_reduction_ms,
            token_savings_percent=token_savings_percent,
            snr_improvement=snr_improvement,
        )
        
        # Compute impact score
        impact_score = self._compute_impact_score(
            repetition_count=repetition_count,
            success_rate=success_rate,
            ihsan_score=ihsan_score,
            latency_reduction_ms=latency_reduction_ms,
        )
        
        # Create metadata
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=PATTERN_TTL_SECONDS)
        
        # Pattern ID is hash of trigger sequence
        from core.federation.protocol import domain_separated_hash, canonical_json
        pattern_id = domain_separated_hash(canonical_json(trigger_sequence))[:32]
        
        metadata = PatternMetadata(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            version=1,
            origin_node_id=self.config.node_id,
            origin_timestamp=now.isoformat(),
            repetition_count=repetition_count,
            success_rate=success_rate,
            impact_score=impact_score,
            ihsan_score=ihsan_score,
            expires_at=expires.isoformat(),
            tags=tags or [],
        )
        
        # Create signed envelope
        envelope = PatternEnvelope.create(
            metadata=metadata,
            payload=payload,
            private_key=self._private_key,
            public_key=self._public_key,
        )
        
        # Store locally
        self.local_patterns[pattern_id] = envelope
        self.stats.local_patterns = len(self.local_patterns)
        
        logger.info(f"📤 Elevated pattern {pattern_id[:16]} (impact={impact_score:.2f})")
        
        # Broadcast to network
        if self.config.auto_propagate:
            await self.gossip.broadcast_pattern(envelope)
            self.stats.patterns_sent += 1
            
            # Propose for consensus
            await self.consensus.propose_pattern(envelope)
            self.stats.consensus_proposed += 1
        
        # Save to disk
        self._save_pattern(envelope)
        
        return envelope
    
    def _compute_impact_score(
        self,
        repetition_count: int,
        success_rate: float,
        ihsan_score: float,
        latency_reduction_ms: int,
    ) -> float:
        """Compute pattern impact score."""
        # Logarithmic scaling for repetitions (diminishing returns)
        rep_score = min(1.0, (repetition_count / 10) ** 0.5)
        
        # Latency contribution (capped at 200ms)
        latency_score = min(1.0, latency_reduction_ms / 200)
        
        # Weighted combination
        impact = (
            0.30 * rep_score +
            0.30 * success_rate +
            0.25 * ihsan_score +
            0.15 * latency_score
        )
        
        return round(impact, 3)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pattern Reception
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _handle_received_pattern(self, envelope: PatternEnvelope):
        """Handle pattern received from network."""
        pattern_id = envelope.metadata.pattern_id
        
        # Already have it?
        if pattern_id in self.local_patterns or pattern_id in self.federated_patterns:
            return
        
        self.stats.patterns_received += 1
        
        logger.info(f"📥 Received pattern {pattern_id[:16]} from {envelope.metadata.origin_node_id[:16]}")
        
        # Verify and propose for local consensus
        valid, reason = envelope.verify()
        if not valid:
            logger.warning(f"Invalid pattern {pattern_id[:16]}: {reason}")
            return
        
        # Propose for consensus
        state = await self.consensus.propose_pattern(envelope)
        
        # If consensus already complete (cached result), adopt immediately
        if state.result and state.result.accepted:
            self._adopt_pattern(envelope)
    
    def _adopt_pattern(self, envelope: PatternEnvelope):
        """Adopt a consensus-accepted pattern into local SAPE."""
        pattern_id = envelope.metadata.pattern_id
        
        if not self.config.auto_adopt:
            logger.info(f"Pattern {pattern_id[:16]} accepted but auto-adopt disabled")
            return
        
        # Store in federated patterns
        self.federated_patterns[pattern_id] = envelope
        self.stats.federated_patterns = len(self.federated_patterns)
        self.stats.consensus_accepted += 1
        
        logger.info(f"✅ Adopted federated pattern {pattern_id[:16]}")
        
        # Notify SAPE integration callback
        if self._on_pattern_adopted:
            self._on_pattern_adopted(envelope)
        
        # Persist
        self._save_pattern(envelope, federated=True)
        
        # Update network multiplier
        self._update_network_multiplier()
    
    def _update_network_multiplier(self):
        """Update network multiplier based on federation state."""
        # M = 1 + log₁₀(patterns + 1) / 10 × (peers / 10)
        patterns = len(self.local_patterns) + len(self.federated_patterns)
        peers = len([p for p in self.gossip.peers.values() if p.is_alive()])
        
        if patterns > 0 and peers > 0:
            import math
            pattern_factor = math.log10(patterns + 1) / 10
            peer_factor = min(1.0, peers / 10)
            self.stats.network_multiplier = 1.0 + pattern_factor * peer_factor
        else:
            self.stats.network_multiplier = 1.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # SAPE Integration
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_pattern_adopted(self, callback: Callable[[PatternEnvelope], None]):
        """
        Register callback for when patterns are adopted from network.
        
        SAPE should register this to integrate federated patterns.
        
        Example:
            def integrate_pattern(envelope):
                # Convert to SAPE ElevatedPattern format
                sape.register_pattern(
                    id=envelope.metadata.pattern_id,
                    trigger=envelope.payload.trigger_sequence,
                    optimization=envelope.payload.optimization,
                )
            
            federation.on_pattern_adopted(integrate_pattern)
        """
        self._on_pattern_adopted = callback
    
    def get_all_patterns(self) -> List[PatternEnvelope]:
        """Get all patterns (local + federated)."""
        return list(self.local_patterns.values()) + list(self.federated_patterns.values())
    
    def get_pattern(self, pattern_id: str) -> Optional[PatternEnvelope]:
        """Get pattern by ID."""
        return self.local_patterns.get(pattern_id) or self.federated_patterns.get(pattern_id)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_pattern(self, envelope: PatternEnvelope, federated: bool = False):
        """Save pattern to disk."""
        subdir = "federated" if federated else "local"
        pattern_dir = self.config.patterns_dir / subdir
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        path = pattern_dir / f"{envelope.metadata.pattern_id}.json"
        path.write_bytes(envelope.to_wire())
    
    def _save_patterns(self):
        """Save all patterns to disk."""
        for envelope in self.local_patterns.values():
            self._save_pattern(envelope, federated=False)
        for envelope in self.federated_patterns.values():
            self._save_pattern(envelope, federated=True)
    
    def _load_patterns(self):
        """Load patterns from disk."""
        # Load local patterns
        local_dir = self.config.patterns_dir / "local"
        if local_dir.exists():
            for path in local_dir.glob("*.json"):
                try:
                    envelope = PatternEnvelope.from_wire(path.read_bytes())
                    self.local_patterns[envelope.metadata.pattern_id] = envelope
                except Exception as e:
                    logger.warning(f"Failed to load pattern {path}: {e}")
        
        # Load federated patterns
        fed_dir = self.config.patterns_dir / "federated"
        if fed_dir.exists():
            for path in fed_dir.glob("*.json"):
                try:
                    envelope = PatternEnvelope.from_wire(path.read_bytes())
                    self.federated_patterns[envelope.metadata.pattern_id] = envelope
                except Exception as e:
                    logger.warning(f"Failed to load pattern {path}: {e}")
        
        self.stats.local_patterns = len(self.local_patterns)
        self.stats.federated_patterns = len(self.federated_patterns)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Peer Management
    # ─────────────────────────────────────────────────────────────────────────
    
    async def connect_to_peer(self, host: str, port: int) -> Optional[PeerInfo]:
        """Connect to a peer node."""
        return await self.gossip.connect_to_peer(host, port)
    
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return [p for p in self.gossip.peers.values() if p.is_alive()]
    
    def get_stats(self) -> FederationStats:
        """Get federation statistics."""
        self.stats.connected_peers = len(self.get_peers())
        self.stats.messages_sent = self.gossip.stats.messages_sent
        self.stats.messages_received = self.gossip.stats.messages_received
        return self.stats
