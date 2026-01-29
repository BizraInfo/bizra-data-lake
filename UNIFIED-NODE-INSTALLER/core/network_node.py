"""
BIZRA Network Node - Resource Sharing & Token System
=====================================================

This module handles:
1. Resource contribution (CPU, GPU, storage, bandwidth)
2. Token earning through Proof-of-Impact
3. P2P network connectivity
4. The "Free Virtual World" gateway

Core Principle: More users = Better for everyone
- Faster: Distributed compute
- Smarter: Collective learning (privacy-preserving)
- Safer: More redundancy
- Private: Anonymity in the crowd
"""

import json
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import threading


# ============================================================================
# TOKEN SYSTEM
# ============================================================================

class TokenType(Enum):
    """BIZRA token types."""
    BZT = "BZT"      # Utility token - used for services
    BZG = "BZG"      # Governance token - voting rights


@dataclass
class Wallet:
    """User's token wallet."""
    address: str
    bzt_balance: float = 0.0
    bzg_balance: float = 0.0
    pending_rewards: float = 0.0
    contribution_score: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def generate_address(cls, seed: str) -> str:
        """Generate deterministic wallet address from seed."""
        return "bzra_" + hashlib.sha256(seed.encode()).hexdigest()[:32]


@dataclass
class Transaction:
    """Token transaction record."""
    tx_id: str
    from_address: str
    to_address: str
    token_type: TokenType
    amount: float
    timestamp: float
    reason: str
    signature: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['token_type'] = self.token_type.value
        return d


class TokenLedger:
    """
    Local token ledger - tracks balances and transactions.
    Syncs with network for consensus.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.ledger_file = data_dir / "ledger.json"
        self.wallets: Dict[str, Wallet] = {}
        self.transactions: List[Transaction] = []
        self._load()

    def _load(self):
        """Load ledger from disk."""
        if self.ledger_file.exists():
            data = json.loads(self.ledger_file.read_text())
            for addr, w in data.get("wallets", {}).items():
                self.wallets[addr] = Wallet(**w)
            # Transactions loaded separately for efficiency

    def _save(self):
        """Save ledger to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "wallets": {addr: w.to_dict() for addr, w in self.wallets.items()},
            "last_updated": time.time(),
        }
        self.ledger_file.write_text(json.dumps(data, indent=2))

    def get_or_create_wallet(self, address: str) -> Wallet:
        """Get existing wallet or create new one."""
        if address not in self.wallets:
            self.wallets[address] = Wallet(address=address)
            self._save()
        return self.wallets[address]

    def add_reward(self, address: str, amount: float, reason: str):
        """Add pending reward to wallet."""
        wallet = self.get_or_create_wallet(address)
        wallet.pending_rewards += amount
        wallet.bzt_balance += amount  # Immediately credited (testnet mode)
        self._save()

        # Record transaction
        tx = Transaction(
            tx_id=hashlib.sha256(f"{address}{amount}{time.time()}".encode()).hexdigest()[:16],
            from_address="network_rewards",
            to_address=address,
            token_type=TokenType.BZT,
            amount=amount,
            timestamp=time.time(),
            reason=reason,
        )
        self.transactions.append(tx)

    def get_balance(self, address: str) -> Dict[str, float]:
        """Get wallet balances."""
        wallet = self.get_or_create_wallet(address)
        return {
            "bzt": wallet.bzt_balance,
            "bzg": wallet.bzg_balance,
            "pending": wallet.pending_rewards,
            "contribution_score": wallet.contribution_score,
        }


# ============================================================================
# RESOURCE METERING
# ============================================================================

@dataclass
class ResourceContribution:
    """Record of resource contribution."""
    node_id: str
    period_start: float
    period_end: float
    compute_units: float      # CPU-seconds contributed
    gpu_units: float          # GPU-seconds contributed
    storage_mb: float         # MB stored for network
    bandwidth_mb: float       # MB relayed
    uptime_percent: float     # % online during period
    tasks_completed: int      # Number of tasks processed
    quality_score: float      # Quality of contributions (0-1)

    @property
    def impact_score(self) -> float:
        """
        Calculate Proof-of-Impact score.
        This is NOT proof-of-work (wasting energy).
        This rewards USEFUL contributions.
        """
        # Weights for different contribution types
        compute_weight = 0.2
        gpu_weight = 0.3
        storage_weight = 0.1
        bandwidth_weight = 0.1
        uptime_weight = 0.1
        tasks_weight = 0.15
        quality_weight = 0.05

        # Normalize values (example normalization)
        normalized_compute = min(self.compute_units / 3600, 1.0)  # Max 1 hour
        normalized_gpu = min(self.gpu_units / 1800, 1.0)          # Max 30 min
        normalized_storage = min(self.storage_mb / 1000, 1.0)     # Max 1GB
        normalized_bandwidth = min(self.bandwidth_mb / 500, 1.0)  # Max 500MB
        normalized_tasks = min(self.tasks_completed / 100, 1.0)   # Max 100 tasks

        score = (
            compute_weight * normalized_compute +
            gpu_weight * normalized_gpu +
            storage_weight * normalized_storage +
            bandwidth_weight * normalized_bandwidth +
            uptime_weight * self.uptime_percent +
            tasks_weight * normalized_tasks +
            quality_weight * self.quality_score
        )

        return score


class ResourceMeter:
    """
    Meters resource contributions for fair reward distribution.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.current_period_start = time.time()
        self.compute_seconds = 0.0
        self.gpu_seconds = 0.0
        self.storage_mb = 0.0
        self.bandwidth_mb = 0.0
        self.tasks_completed = 0
        self.quality_scores: List[float] = []
        self._lock = threading.Lock()

    def record_compute(self, seconds: float):
        """Record CPU time contributed."""
        with self._lock:
            self.compute_seconds += seconds

    def record_gpu(self, seconds: float):
        """Record GPU time contributed."""
        with self._lock:
            self.gpu_seconds += seconds

    def record_storage(self, mb: float):
        """Record storage contributed."""
        with self._lock:
            self.storage_mb = mb  # Current snapshot

    def record_bandwidth(self, mb: float):
        """Record bandwidth used for network."""
        with self._lock:
            self.bandwidth_mb += mb

    def record_task_completion(self, quality: float = 1.0):
        """Record a completed task."""
        with self._lock:
            self.tasks_completed += 1
            self.quality_scores.append(quality)

    def finalize_period(self, uptime_percent: float) -> ResourceContribution:
        """Finalize the current period and return contribution record."""
        with self._lock:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 1.0

            contribution = ResourceContribution(
                node_id=self.node_id,
                period_start=self.current_period_start,
                period_end=time.time(),
                compute_units=self.compute_seconds,
                gpu_units=self.gpu_seconds,
                storage_mb=self.storage_mb,
                bandwidth_mb=self.bandwidth_mb,
                uptime_percent=uptime_percent,
                tasks_completed=self.tasks_completed,
                quality_score=avg_quality,
            )

            # Reset for next period
            self.current_period_start = time.time()
            self.compute_seconds = 0.0
            self.gpu_seconds = 0.0
            self.bandwidth_mb = 0.0
            self.tasks_completed = 0
            self.quality_scores = []

            return contribution


# ============================================================================
# REWARD CALCULATOR
# ============================================================================

class RewardCalculator:
    """
    Calculates token rewards based on Proof-of-Impact.

    Key principle: Rewards are proportional to USEFUL contribution,
    not raw resource burning. A helpful answer to another user is
    worth more than 1000 idle CPU cycles.
    """

    # Reward pool per epoch (e.g., per hour)
    EPOCH_REWARD_POOL = 1000.0  # BZT per epoch

    # Bonus multipliers
    EARLY_ADOPTER_BONUS = 1.5
    VALIDATOR_BONUS = 1.2
    QUALITY_BONUS_THRESHOLD = 0.9  # Quality > 90% gets bonus

    def calculate_reward(
        self,
        contribution: ResourceContribution,
        network_total_impact: float,
        is_early_adopter: bool = False,
        is_validator: bool = False,
    ) -> float:
        """
        Calculate token reward for a contribution.

        reward = (individual_impact / network_total_impact) * pool * multipliers
        """
        if network_total_impact == 0:
            return 0.0

        # Base share of pool
        share = contribution.impact_score / network_total_impact
        base_reward = share * self.EPOCH_REWARD_POOL

        # Apply multipliers
        multiplier = 1.0
        if is_early_adopter:
            multiplier *= self.EARLY_ADOPTER_BONUS
        if is_validator:
            multiplier *= self.VALIDATOR_BONUS
        if contribution.quality_score >= self.QUALITY_BONUS_THRESHOLD:
            multiplier *= 1.1  # 10% quality bonus

        return base_reward * multiplier


# ============================================================================
# P2P NETWORK NODE
# ============================================================================

class NodeStatus(Enum):
    """Node operational status."""
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    SYNCING = "syncing"
    VALIDATING = "validating"


@dataclass
class PeerInfo:
    """Information about a network peer."""
    node_id: str
    address: str
    port: int
    last_seen: float
    reputation: float = 1.0
    is_validator: bool = False


class NetworkNode:
    """
    BIZRA Network Node - The gateway to the decentralized world.

    This handles:
    - P2P connectivity
    - Resource sharing
    - Token rewards
    - Content distribution (the "free social space")
    """

    def __init__(
        self,
        node_id: str,
        data_dir: Path,
        resource_config: Dict[str, Any],
    ):
        self.node_id = node_id
        self.data_dir = data_dir
        self.status = NodeStatus.OFFLINE
        self.peers: Dict[str, PeerInfo] = {}

        # Token system
        self.wallet_address = Wallet.generate_address(node_id)
        self.ledger = TokenLedger(data_dir / "ledger")

        # Resource metering
        self.resource_config = resource_config
        self.meter = ResourceMeter(node_id)
        self.reward_calculator = RewardCalculator()

        # Network stats
        self.network_size = 1  # Starts with just us
        self.connected_peers = 0
        self.uptime_start = time.time()

        print(f"Network Node initialized: {node_id}")
        print(f"Wallet: {self.wallet_address}")

    async def start(self):
        """Start the network node."""
        self.status = NodeStatus.CONNECTING
        print("Starting BIZRA Network Node...")

        # Start background tasks
        asyncio.create_task(self._peer_discovery_loop())
        asyncio.create_task(self._reward_distribution_loop())
        asyncio.create_task(self._resource_sharing_loop())

        self.status = NodeStatus.ONLINE
        print(f"Node online. Status: {self.status.value}")

    async def stop(self):
        """Stop the network node gracefully."""
        self.status = NodeStatus.OFFLINE
        # Finalize any pending rewards
        await self._finalize_period()
        print("Node stopped.")

    async def _peer_discovery_loop(self):
        """Continuously discover and connect to peers."""
        while self.status != NodeStatus.OFFLINE:
            try:
                # In production, this would use libp2p/DHT
                # For now, simulate peer discovery
                await asyncio.sleep(30)  # Check every 30 seconds

                # Simulated network growth
                self.network_size = max(1, self.network_size + 1)
                self.connected_peers = min(self.network_size - 1, 10)

            except Exception as e:
                print(f"Peer discovery error: {e}")
                await asyncio.sleep(5)

    async def _reward_distribution_loop(self):
        """Distribute rewards at the end of each epoch."""
        epoch_duration = 3600  # 1 hour epochs

        while self.status != NodeStatus.OFFLINE:
            try:
                await asyncio.sleep(epoch_duration)
                await self._finalize_period()

            except Exception as e:
                print(f"Reward distribution error: {e}")

    async def _finalize_period(self):
        """Finalize contribution period and distribute rewards."""
        # Calculate uptime
        uptime_duration = time.time() - self.uptime_start
        uptime_percent = min(1.0, uptime_duration / 3600)  # Max 1 hour

        # Get contribution record
        contribution = self.meter.finalize_period(uptime_percent)

        # Calculate network total (in production, this comes from consensus)
        # For single node testing, just use our own impact
        network_total = contribution.impact_score * self.network_size

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            contribution=contribution,
            network_total_impact=network_total,
            is_early_adopter=True,  # Early adopter bonus for now
        )

        # Credit reward
        if reward > 0:
            self.ledger.add_reward(
                self.wallet_address,
                reward,
                f"Epoch contribution - Impact: {contribution.impact_score:.4f}",
            )
            print(f"Epoch complete! Earned {reward:.4f} BZT")

        # Reset uptime tracking
        self.uptime_start = time.time()

    async def _resource_sharing_loop(self):
        """Share resources with the network."""
        while self.status != NodeStatus.OFFLINE:
            try:
                # Simulate resource contribution
                # In production, this would handle actual distributed tasks

                # Record some compute contribution (simulated)
                self.meter.record_compute(1.0)  # 1 second of compute

                # Record storage (current allocated)
                storage_gb = self.resource_config.get("storage_gb", 10)
                self.meter.record_storage(storage_gb * 1024)  # Convert to MB

                await asyncio.sleep(1)

            except Exception as e:
                print(f"Resource sharing error: {e}")
                await asyncio.sleep(5)

    def get_status(self) -> Dict[str, Any]:
        """Get current node status."""
        balances = self.ledger.get_balance(self.wallet_address)

        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "wallet": self.wallet_address,
            "balances": balances,
            "network": {
                "size": self.network_size,
                "connected_peers": self.connected_peers,
            },
            "resources": {
                "configured": self.resource_config,
                "current_contribution": {
                    "compute_seconds": self.meter.compute_seconds,
                    "gpu_seconds": self.meter.gpu_seconds,
                    "storage_mb": self.meter.storage_mb,
                    "tasks_completed": self.meter.tasks_completed,
                },
            },
        }


# ============================================================================
# FREE VIRTUAL WORLD - Content Layer
# ============================================================================

@dataclass
class ContentItem:
    """A piece of content in the decentralized network."""
    content_id: str
    author_node: str
    content_type: str  # text, image, link, etc.
    data: str
    timestamp: float
    signatures: List[str] = field(default_factory=list)
    reactions: Dict[str, int] = field(default_factory=dict)


class ContentFeed:
    """
    Decentralized content feed - the "free social space".

    Key principles:
    - No algorithmic manipulation (YOU choose what to see)
    - User-owned content (your data is YOURS)
    - Censorship-resistant (distributed across nodes)
    - Privacy-preserving (encrypted by default)
    """

    def __init__(self, node: NetworkNode):
        self.node = node
        self.local_content: Dict[str, ContentItem] = {}
        self.subscriptions: List[str] = []  # Node IDs we follow

    def create_post(self, content_type: str, data: str) -> ContentItem:
        """Create a new post."""
        content_id = hashlib.sha256(
            f"{self.node.node_id}{data}{time.time()}".encode()
        ).hexdigest()[:16]

        item = ContentItem(
            content_id=content_id,
            author_node=self.node.node_id,
            content_type=content_type,
            data=data,
            timestamp=time.time(),
        )

        self.local_content[content_id] = item
        return item

    def get_feed(self, algorithm: str = "chronological") -> List[ContentItem]:
        """
        Get content feed.

        Available algorithms (USER chooses, not forced):
        - chronological: Newest first
        - followed: From subscribed nodes only
        - random: Random shuffle (discovery mode)
        """
        content = list(self.local_content.values())

        if algorithm == "chronological":
            content.sort(key=lambda x: x.timestamp, reverse=True)
        elif algorithm == "followed":
            content = [c for c in content if c.author_node in self.subscriptions]
            content.sort(key=lambda x: x.timestamp, reverse=True)
        elif algorithm == "random":
            import random
            random.shuffle(content)

        return content

    def subscribe(self, node_id: str):
        """Subscribe to another node's content."""
        if node_id not in self.subscriptions:
            self.subscriptions.append(node_id)

    def unsubscribe(self, node_id: str):
        """Unsubscribe from a node."""
        if node_id in self.subscriptions:
            self.subscriptions.remove(node_id)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_network_node(
    username: str,
    data_dir: Path,
    resource_config: Dict[str, Any],
) -> NetworkNode:
    """Create a network node for a user."""
    node_id = hashlib.sha256(username.encode()).hexdigest()[:16]
    return NetworkNode(node_id, data_dir, resource_config)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        # Create a test node
        data_dir = Path.home() / ".bizra" / "test"
        resource_config = {
            "cpu_cores": 2,
            "ram_gb": 4,
            "storage_gb": 25,
            "gpu_enabled": False,
        }

        node = create_network_node("test_user", data_dir, resource_config)
        await node.start()

        # Check status
        print("\nNode Status:")
        print(json.dumps(node.get_status(), indent=2))

        # Simulate some activity
        for i in range(5):
            node.meter.record_compute(10.0)
            node.meter.record_task_completion(quality=0.95)
            await asyncio.sleep(1)

        # Finalize and get rewards
        await node._finalize_period()

        print("\nFinal Status:")
        print(json.dumps(node.get_status(), indent=2))

        await node.stop()

    asyncio.run(test())
