"""
BIZRA ACCUMULATOR
═══════════════════════════════════════════════════════════════════════════════

The Accumulator: A Self-Sustaining Value Accrual Engine

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE BIZRA ACCUMULATOR                                │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LAYER 4: IMPACT LAYER (PoI - Proof of Impact)                      │   │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │   │
│   │  │ SEED Token     │  │ BLOOM Accrual  │  │ FRUIT Harvest  │        │   │
│   │  │ (Utility)      │  │ (Impact Score) │  │ (Distribution) │        │   │
│   │  └────────────────┘  └────────────────┘  └────────────────┘        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  LAYER 3: KNOWLEDGE LAYER       │                                   │   │
│   │  ┌────────────────┐  ┌─────────┴────────┐  ┌────────────────┐      │   │
│   │  │ Sovereign      │  │ Vector           │  │ Quran          │      │   │
│   │  │ Catalog        │  │ Embeddings       │  │ Knowledge      │      │   │
│   │  │ (709K nodes)   │  │ (Semantic)       │  │ (6,236 verses) │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  LAYER 2: AGENT LAYER           │                                   │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  PERSONAL TEAM (MAG7)                                       │   │   │
│   │  │  Planner │ Researcher │ Coder │ Evaluator │ Ethicist │ +2   │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  OPS AGENTS                                                 │   │   │
│   │  │  Self-Diag │ Auto-Debug │ Perf-Tune │ Sec-Monitor           │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  LAYER 1: INFERENCE LAYER       │                                   │   │
│   │  ┌────────────────┐  ┌─────────┴────────┐  ┌────────────────┐      │   │
│   │  │ Flywheel       │  │ LM Studio        │  │ Ollama         │      │   │
│   │  │ (Autopoietic)  │  │ (Host GPU)       │  │ (Container)    │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│   ┌─────────────────────────────────┼───────────────────────────────────┐   │
│   │  LAYER 0: FOUNDATION LAYER      │                                   │   │
│   │  ┌────────────────┐  ┌─────────┴────────┐  ┌────────────────┐      │   │
│   │  │ Fail-Closed    │  │ State            │  │ Event          │      │   │
│   │  │ Security       │  │ Persistence      │  │ Bus            │      │   │
│   │  └────────────────┘  └──────────────────┘  └────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

The Accumulator Metaphor:
━━━━━━━━━━━━━━━━━━━━━━━━━
SEED → BLOOM → FRUIT

1. SEED (Utility Token)
   - Initial stake/contribution
   - Computational resources committed
   - Knowledge deposited

2. BLOOM (Impact Accrual)  
   - Actions generate impact scores
   - Proof-of-Impact attestations
   - Compounding value over time

3. FRUIT (Distribution)
   - Harvestable impact rewards
   - Redistributable to contributors
   - Zakat-compliant overflow routing

Giants Protocol Alignment:
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Al-Khwarizmi: Algorithmic accumulation rules
- Ibn Khaldun: Civilizational wealth cycles (asabiyyah)
- Al-Ghazali: Ethical constraints on accumulation

Islamic Finance Principles:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No riba (interest) - value from real impact
- Gharar minimized - transparent attestations
- Zakat integration - automatic tithing at thresholds
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable
from enum import Enum
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ACCUMULATOR_VERSION = "1.0.0"
GENESIS_MERKLE_ROOT = "d9c9fa504add65a1be737f3fe3447bc056fd1aad2850f491184208354f41926f"

# Paths (configurable via environment)
ACCUMULATOR_STATE_PATH = Path(os.getenv("BIZRA_ACCUMULATOR_STATE", "/var/lib/bizra/accumulator_state.json"))
POI_LEDGER_PATH = Path(os.getenv("BIZRA_POI_LEDGER", "/var/lib/bizra/poi_ledger.jsonl"))
BLOOM_SNAPSHOT_PATH = Path(os.getenv("BIZRA_BLOOM_SNAPSHOT", "/var/lib/bizra/bloom_snapshot.json"))

# Accumulator Thresholds
ZAKAT_THRESHOLD = 0.025  # 2.5% - Islamic zakat rate
BLOOM_DECAY_RATE = 0.001  # Daily decay to encourage activity
FRUIT_HARVEST_MINIMUM = 100.0  # Minimum bloom to harvest

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ImpactCategory(str, Enum):
    """Categories of impact that generate bloom."""
    COMPUTATION = "computation"     # LLM inference, processing
    KNOWLEDGE = "knowledge"         # Research, learning, documentation
    CODE = "code"                   # Development, bug fixes
    ETHICS = "ethics"               # Compliance, review, validation
    ORCHESTRATION = "orchestration" # Coordination, planning
    COMMUNITY = "community"         # Helping others, sadaqah
    ZAKAT = "zakat"                 # Charitable distribution


class AccumulatorState(str, Enum):
    """Accumulator lifecycle states."""
    DORMANT = "dormant"       # Not active, no accumulation
    SEEDING = "seeding"       # Initial deposits, building base
    BLOOMING = "blooming"     # Active accumulation
    FRUITING = "fruiting"     # Ready for harvest
    HARVESTING = "harvesting" # Distribution in progress


@dataclass
class Seed:
    """
    Initial contribution to the accumulator.
    
    Seeds represent:
    - Computational resources (GPU time, inference tokens)
    - Knowledge deposits (documents, embeddings)
    - Code contributions (commits, reviews)
    """
    seed_id: str
    contributor: str
    category: ImpactCategory
    amount: float  # Base value
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    attestation_hash: str = ""
    
    def finalize(self) -> "Seed":
        """Generate cryptographic attestation."""
        payload = json.dumps({
            "id": self.seed_id,
            "contributor": self.contributor,
            "category": self.category.value,
            "amount": self.amount,
            "timestamp": self.timestamp,
        }, sort_keys=True)
        self.attestation_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self


@dataclass
class Bloom:
    """
    Accumulated impact from activities.
    
    Bloom grows through:
    - Successful task completions
    - Knowledge synthesis
    - Ethical validations
    - Community contributions
    """
    contributor: str
    total_bloom: float = 0.0
    category_bloom: Dict[str, float] = field(default_factory=dict)
    last_activity: str = ""
    streak_days: int = 0
    multiplier: float = 1.0  # Activity streak bonus
    
    def add_bloom(self, amount: float, category: ImpactCategory) -> float:
        """Add bloom with category tracking."""
        adjusted = amount * self.multiplier
        self.total_bloom += adjusted
        self.category_bloom[category.value] = self.category_bloom.get(category.value, 0) + adjusted
        self.last_activity = datetime.now(timezone.utc).isoformat()
        return adjusted
    
    def apply_decay(self, days_inactive: int) -> float:
        """Apply decay for inactivity."""
        if days_inactive > 0:
            decay = 1 - (BLOOM_DECAY_RATE * days_inactive)
            decay = max(0.5, decay)  # Cap at 50% decay
            lost = self.total_bloom * (1 - decay)
            self.total_bloom *= decay
            self.multiplier = max(1.0, self.multiplier - 0.1 * days_inactive)
            return lost
        return 0.0


@dataclass 
class Fruit:
    """
    Harvestable value from accumulated bloom.
    
    Fruit represents:
    - Claimable rewards
    - Redistributable value
    - Zakat overflow
    """
    fruit_id: str
    contributor: str
    bloom_source: float  # Bloom converted to fruit
    zakat_portion: float = 0.0  # 2.5% for charitable distribution
    net_fruit: float = 0.0  # After zakat
    harvested: bool = False
    harvest_timestamp: Optional[str] = None
    
    def calculate(self) -> "Fruit":
        """Calculate zakat and net fruit."""
        self.zakat_portion = self.bloom_source * ZAKAT_THRESHOLD
        self.net_fruit = self.bloom_source - self.zakat_portion
        return self


@dataclass
class ProofOfImpact:
    """
    Cryptographic attestation of impact.
    
    Aligned with System Contract §5.
    """
    version: str = "poi-1.0"
    chain_id: str = "bizra-main"
    genesis_merkle_root: str = GENESIS_MERKLE_ROOT
    contributor: str = ""
    action: str = ""
    category: str = ""
    impact_score: float = 0.0
    resources: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    attestation_hash: str = ""
    previous_hash: str = ""  # Chain linkage
    
    def finalize(self, previous_hash: str = "") -> "ProofOfImpact":
        """Generate cryptographic attestation with chain linkage."""
        self.previous_hash = previous_hash
        payload = json.dumps({
            "v": self.version,
            "c": self.chain_id,
            "g": self.genesis_merkle_root,
            "contributor": self.contributor,
            "action": self.action,
            "category": self.category,
            "impact": self.impact_score,
            "t": self.timestamp,
            "prev": self.previous_hash,
        }, sort_keys=True)
        self.attestation_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# IMPACT SCORERS
# ═══════════════════════════════════════════════════════════════════════════════

class ImpactScorer:
    """
    Calculate impact scores for different activity types.
    
    Scoring Philosophy (Al-Khwarizmi's Algorithmic Precision):
    - Transparent formulas
    - Reproducible results
    - Category-specific weights
    """
    
    # Base weights per category
    CATEGORY_WEIGHTS = {
        ImpactCategory.COMPUTATION: 1.0,
        ImpactCategory.KNOWLEDGE: 2.0,   # Knowledge is valued highly
        ImpactCategory.CODE: 1.5,
        ImpactCategory.ETHICS: 3.0,      # Ethical work has highest weight
        ImpactCategory.ORCHESTRATION: 1.2,
        ImpactCategory.COMMUNITY: 2.5,   # Community contribution valued
        ImpactCategory.ZAKAT: 0.0,       # Zakat is outflow, not scored
    }
    
    @classmethod
    def score_computation(cls, tokens_processed: int, latency_ms: float) -> float:
        """Score computation impact."""
        # Efficiency bonus for low latency
        efficiency = max(0.5, 1.0 - (latency_ms / 5000))
        base_score = (tokens_processed / 1000) * efficiency
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.COMPUTATION]
    
    @classmethod
    def score_knowledge(cls, documents_processed: int, synthesis_quality: float) -> float:
        """Score knowledge impact."""
        # Quality multiplier (0.0 - 1.0)
        base_score = documents_processed * synthesis_quality
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.KNOWLEDGE]
    
    @classmethod
    def score_code(cls, lines_changed: int, test_coverage: float, bugs_fixed: int) -> float:
        """Score code impact."""
        coverage_bonus = 1.0 + test_coverage  # Up to 2x for 100% coverage
        base_score = (lines_changed / 100 + bugs_fixed * 10) * coverage_bonus
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.CODE]
    
    @classmethod
    def score_ethics(cls, reviews_completed: int, violations_caught: int) -> float:
        """Score ethics/compliance impact."""
        base_score = reviews_completed * 5 + violations_caught * 20
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.ETHICS]
    
    @classmethod
    def score_orchestration(cls, tasks_coordinated: int, success_rate: float) -> float:
        """Score orchestration impact."""
        base_score = tasks_coordinated * success_rate
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.ORCHESTRATION]
    
    @classmethod
    def score_community(cls, users_helped: int, satisfaction_score: float) -> float:
        """Score community impact."""
        base_score = users_helped * satisfaction_score * 10
        return base_score * cls.CATEGORY_WEIGHTS[ImpactCategory.COMMUNITY]


# ═══════════════════════════════════════════════════════════════════════════════
# THE ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class BizraAccumulator:
    """
    The BIZRA Accumulator: Self-Sustaining Value Accrual Engine.
    
    Core Responsibilities:
    1. Accept seeds (initial contributions)
    2. Track bloom (impact accumulation)
    3. Generate fruit (harvestable value)
    4. Maintain PoI ledger (attestation chain)
    5. Enforce zakat distribution
    
    Design Principles:
    - Transparency: All calculations visible
    - Fairness: Category-weighted scoring
    - Sustainability: Decay prevents hoarding
    - Compliance: Built-in zakat
    """
    
    def __init__(self):
        self.state = AccumulatorState.DORMANT
        self.seeds: List[Seed] = []
        self.blooms: Dict[str, Bloom] = {}  # contributor -> bloom
        self.fruits: List[Fruit] = []
        self.poi_ledger: List[ProofOfImpact] = []
        self.zakat_pool: float = 0.0
        
        # Callbacks for external integration
        self._on_bloom: Optional[Callable[[str, float, ImpactCategory], Awaitable[None]]] = None
        self._on_fruit: Optional[Callable[[Fruit], Awaitable[None]]] = None
        self._on_zakat: Optional[Callable[[float], Awaitable[None]]] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load persisted state
        self._load_state()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def _load_state(self):
        """Load persisted accumulator state."""
        if ACCUMULATOR_STATE_PATH.exists():
            try:
                with open(ACCUMULATOR_STATE_PATH, 'r') as f:
                    data = json.load(f)
                    self.state = AccumulatorState(data.get("state", "dormant"))
                    self.zakat_pool = data.get("zakat_pool", 0.0)
                    
                    # Load blooms
                    for contrib, bloom_data in data.get("blooms", {}).items():
                        self.blooms[contrib] = Bloom(
                            contributor=contrib,
                            total_bloom=bloom_data.get("total_bloom", 0.0),
                            category_bloom=bloom_data.get("category_bloom", {}),
                            last_activity=bloom_data.get("last_activity", ""),
                            streak_days=bloom_data.get("streak_days", 0),
                            multiplier=bloom_data.get("multiplier", 1.0),
                        )
            except Exception as e:
                print(f"[Accumulator] Failed to load state: {e}")
        
        # Load PoI ledger
        if POI_LEDGER_PATH.exists():
            try:
                with open(POI_LEDGER_PATH, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.poi_ledger.append(json.loads(line))
            except Exception as e:
                print(f"[Accumulator] Failed to load PoI ledger: {e}")
    
    def _save_state(self):
        """Persist accumulator state."""
        ACCUMULATOR_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "state": self.state.value,
            "zakat_pool": self.zakat_pool,
            "blooms": {
                contrib: {
                    "total_bloom": bloom.total_bloom,
                    "category_bloom": bloom.category_bloom,
                    "last_activity": bloom.last_activity,
                    "streak_days": bloom.streak_days,
                    "multiplier": bloom.multiplier,
                }
                for contrib, bloom in self.blooms.items()
            },
            "last_saved": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(ACCUMULATOR_STATE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _append_poi(self, poi: ProofOfImpact):
        """Append PoI to ledger with chain linkage."""
        # Get previous hash for chain
        previous_hash = self.poi_ledger[-1].get("attestation_hash", "") if self.poi_ledger else ""
        poi.finalize(previous_hash)
        
        POI_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(POI_LEDGER_PATH, 'a') as f:
            f.write(json.dumps({
                "version": poi.version,
                "chain_id": poi.chain_id,
                "contributor": poi.contributor,
                "action": poi.action,
                "category": poi.category,
                "impact_score": poi.impact_score,
                "resources": poi.resources,
                "benchmarks": poi.benchmarks,
                "timestamp": poi.timestamp,
                "attestation_hash": poi.attestation_hash,
                "previous_hash": poi.previous_hash,
            }) + "\n")
        
        self.poi_ledger.append({
            "attestation_hash": poi.attestation_hash,
            "contributor": poi.contributor,
            "impact_score": poi.impact_score,
        })
    
    # ─────────────────────────────────────────────────────────────────────────
    # SEEDING
    # ─────────────────────────────────────────────────────────────────────────
    
    def plant_seed(
        self,
        contributor: str,
        category: ImpactCategory,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Seed:
        """
        Plant a seed - initial contribution to the accumulator.
        
        Seeds kickstart the accumulation process.
        """
        with self._lock:
            seed_id = hashlib.sha256(
                f"{contributor}:{category.value}:{amount}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            seed = Seed(
                seed_id=seed_id,
                contributor=contributor,
                category=category,
                amount=amount,
                metadata=metadata or {},
            ).finalize()
            
            self.seeds.append(seed)
            
            # Initialize bloom for contributor if not exists
            if contributor not in self.blooms:
                self.blooms[contributor] = Bloom(contributor=contributor)
            
            # Seed generates initial bloom
            initial_bloom = seed.amount * 0.1  # 10% of seed value
            self.blooms[contributor].add_bloom(initial_bloom, category)
            
            # Transition state
            if self.state == AccumulatorState.DORMANT:
                self.state = AccumulatorState.SEEDING
            
            # Record PoI
            poi = ProofOfImpact(
                contributor=contributor,
                action="seed_planted",
                category=category.value,
                impact_score=initial_bloom,
                resources={"seed_amount": amount, "seed_id": seed_id},
            )
            self._append_poi(poi)
            self._save_state()
            
            return seed
    
    # ─────────────────────────────────────────────────────────────────────────
    # BLOOMING
    # ─────────────────────────────────────────────────────────────────────────
    
    def record_impact(
        self,
        contributor: str,
        action: str,
        category: ImpactCategory,
        impact_score: float,
        resources: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Record an impact and add to bloom.
        
        This is the primary accumulation mechanism.
        """
        with self._lock:
            # Ensure contributor has bloom
            if contributor not in self.blooms:
                self.blooms[contributor] = Bloom(contributor=contributor)
            
            # Add bloom
            bloom = self.blooms[contributor]
            actual_bloom = bloom.add_bloom(impact_score, category)
            
            # Check for state transition
            if self.state == AccumulatorState.SEEDING and bloom.total_bloom >= FRUIT_HARVEST_MINIMUM:
                self.state = AccumulatorState.BLOOMING
            
            # Record PoI
            poi = ProofOfImpact(
                contributor=contributor,
                action=action,
                category=category.value,
                impact_score=actual_bloom,
                resources=resources or {},
                benchmarks=benchmarks or {},
            )
            self._append_poi(poi)
            self._save_state()
            
            # Trigger callback if registered
            if self._on_bloom:
                asyncio.create_task(self._on_bloom(contributor, actual_bloom, category))
            
            return actual_bloom
    
    def get_bloom(self, contributor: str) -> Optional[Bloom]:
        """Get bloom status for a contributor."""
        return self.blooms.get(contributor)
    
    def get_total_bloom(self) -> float:
        """Get total bloom across all contributors."""
        return sum(b.total_bloom for b in self.blooms.values())
    
    # ─────────────────────────────────────────────────────────────────────────
    # FRUITING
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_fruitability(self, contributor: str) -> bool:
        """Check if contributor has enough bloom to harvest."""
        bloom = self.blooms.get(contributor)
        return bloom is not None and bloom.total_bloom >= FRUIT_HARVEST_MINIMUM
    
    def harvest_fruit(self, contributor: str) -> Optional[Fruit]:
        """
        Convert bloom to harvestable fruit.
        
        Automatically deducts zakat (2.5%) for charitable distribution.
        """
        with self._lock:
            if not self.check_fruitability(contributor):
                return None
            
            bloom = self.blooms[contributor]
            
            # Create fruit from bloom
            fruit_id = hashlib.sha256(
                f"{contributor}:fruit:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            fruit = Fruit(
                fruit_id=fruit_id,
                contributor=contributor,
                bloom_source=bloom.total_bloom,
            ).calculate()
            
            # Add zakat to pool
            self.zakat_pool += fruit.zakat_portion
            
            # Reset bloom (harvested)
            self.blooms[contributor] = Bloom(
                contributor=contributor,
                multiplier=bloom.multiplier,  # Keep streak multiplier
                streak_days=bloom.streak_days,
            )
            
            self.fruits.append(fruit)
            
            # State transition
            self.state = AccumulatorState.FRUITING
            
            # Record PoI
            poi = ProofOfImpact(
                contributor=contributor,
                action="fruit_harvested",
                category="harvest",
                impact_score=fruit.net_fruit,
                resources={
                    "fruit_id": fruit_id,
                    "bloom_source": fruit.bloom_source,
                    "zakat_portion": fruit.zakat_portion,
                    "net_fruit": fruit.net_fruit,
                },
            )
            self._append_poi(poi)
            self._save_state()
            
            # Trigger callbacks
            if self._on_fruit:
                asyncio.create_task(self._on_fruit(fruit))
            if self._on_zakat and fruit.zakat_portion > 0:
                asyncio.create_task(self._on_zakat(fruit.zakat_portion))
            
            return fruit
    
    # ─────────────────────────────────────────────────────────────────────────
    # ZAKAT DISTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def distribute_zakat(self, recipients: List[str], amounts: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Distribute accumulated zakat to recipients.
        
        If amounts not specified, distributes equally.
        """
        with self._lock:
            if self.zakat_pool <= 0:
                return {}
            
            if amounts is None:
                # Equal distribution
                per_recipient = self.zakat_pool / len(recipients)
                amounts = [per_recipient] * len(recipients)
            
            total_requested = sum(amounts)
            if total_requested > self.zakat_pool:
                # Scale down proportionally
                scale = self.zakat_pool / total_requested
                amounts = [a * scale for a in amounts]
            
            distribution = {}
            for recipient, amount in zip(recipients, amounts):
                distribution[recipient] = amount
                
                # Record PoI for each distribution
                poi = ProofOfImpact(
                    contributor="system:zakat",
                    action="zakat_distributed",
                    category=ImpactCategory.ZAKAT.value,
                    impact_score=0.0,  # Outflow, not impact
                    resources={
                        "recipient": recipient,
                        "amount": amount,
                    },
                )
                self._append_poi(poi)
            
            # Deduct from pool
            self.zakat_pool -= sum(distribution.values())
            self._save_state()
            
            return distribution
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def record_computation(
        self,
        contributor: str,
        tokens_processed: int,
        latency_ms: float,
        model: str = "unknown",
    ) -> float:
        """Record computational impact (LLM inference, etc.)."""
        score = ImpactScorer.score_computation(tokens_processed, latency_ms)
        return self.record_impact(
            contributor=contributor,
            action="computation",
            category=ImpactCategory.COMPUTATION,
            impact_score=score,
            resources={"tokens": tokens_processed, "model": model},
            benchmarks={"latency_ms": latency_ms},
        )
    
    def record_knowledge(
        self,
        contributor: str,
        documents_processed: int,
        synthesis_quality: float,
        sources: Optional[List[str]] = None,
    ) -> float:
        """Record knowledge synthesis impact."""
        score = ImpactScorer.score_knowledge(documents_processed, synthesis_quality)
        return self.record_impact(
            contributor=contributor,
            action="knowledge_synthesis",
            category=ImpactCategory.KNOWLEDGE,
            impact_score=score,
            resources={"documents": documents_processed, "sources": sources or []},
            benchmarks={"quality": synthesis_quality},
        )
    
    def record_code(
        self,
        contributor: str,
        lines_changed: int,
        test_coverage: float,
        bugs_fixed: int = 0,
        commit_hash: Optional[str] = None,
    ) -> float:
        """Record code contribution impact."""
        score = ImpactScorer.score_code(lines_changed, test_coverage, bugs_fixed)
        return self.record_impact(
            contributor=contributor,
            action="code_contribution",
            category=ImpactCategory.CODE,
            impact_score=score,
            resources={"lines": lines_changed, "bugs_fixed": bugs_fixed, "commit": commit_hash},
            benchmarks={"test_coverage": test_coverage},
        )
    
    def record_ethics(
        self,
        contributor: str,
        reviews_completed: int,
        violations_caught: int = 0,
    ) -> float:
        """Record ethics/compliance impact."""
        score = ImpactScorer.score_ethics(reviews_completed, violations_caught)
        return self.record_impact(
            contributor=contributor,
            action="ethics_review",
            category=ImpactCategory.ETHICS,
            impact_score=score,
            resources={"reviews": reviews_completed, "violations": violations_caught},
        )
    
    def record_community(
        self,
        contributor: str,
        users_helped: int,
        satisfaction_score: float,
    ) -> float:
        """Record community contribution impact."""
        score = ImpactScorer.score_community(users_helped, satisfaction_score)
        return self.record_impact(
            contributor=contributor,
            action="community_help",
            category=ImpactCategory.COMMUNITY,
            impact_score=score,
            resources={"users_helped": users_helped},
            benchmarks={"satisfaction": satisfaction_score},
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATUS & REPORTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def status(self) -> Dict[str, Any]:
        """Get accumulator status."""
        return {
            "version": ACCUMULATOR_VERSION,
            "state": self.state.value,
            "total_seeds": len(self.seeds),
            "total_contributors": len(self.blooms),
            "total_bloom": self.get_total_bloom(),
            "total_fruits": len(self.fruits),
            "zakat_pool": self.zakat_pool,
            "poi_attestations": len(self.poi_ledger),
            "top_contributors": sorted(
                [(c, b.total_bloom) for c, b in self.blooms.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
    
    def leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get contributor leaderboard."""
        sorted_blooms = sorted(
            self.blooms.items(),
            key=lambda x: x[1].total_bloom,
            reverse=True,
        )[:limit]
        
        return [
            {
                "rank": i + 1,
                "contributor": contrib,
                "total_bloom": bloom.total_bloom,
                "categories": bloom.category_bloom,
                "streak_days": bloom.streak_days,
                "multiplier": bloom.multiplier,
                "harvestable": bloom.total_bloom >= FRUIT_HARVEST_MINIMUM,
            }
            for i, (contrib, bloom) in enumerate(sorted_blooms)
        ]
    
    # ─────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_bloom(self, callback: Callable[[str, float, ImpactCategory], Awaitable[None]]):
        """Register callback for bloom events."""
        self._on_bloom = callback
    
    def on_fruit(self, callback: Callable[[Fruit], Awaitable[None]]):
        """Register callback for fruit harvest events."""
        self._on_fruit = callback
    
    def on_zakat(self, callback: Callable[[float], Awaitable[None]]):
        """Register callback for zakat distribution events."""
        self._on_zakat = callback


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_accumulator: Optional[BizraAccumulator] = None

def get_accumulator() -> BizraAccumulator:
    """Get the singleton Accumulator instance."""
    global _accumulator
    if _accumulator is None:
        _accumulator = BizraAccumulator()
    return _accumulator


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("═" * 70)
    print("   BIZRA ACCUMULATOR — Self-Sustaining Value Accrual Engine")
    print("═" * 70)
    
    acc = get_accumulator()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "status":
            status = acc.status()
            print(f"\n📊 Status: {status['state']}")
            print(f"   Seeds: {status['total_seeds']}")
            print(f"   Contributors: {status['total_contributors']}")
            print(f"   Total Bloom: {status['total_bloom']:.2f}")
            print(f"   Zakat Pool: {status['zakat_pool']:.2f}")
            print(f"   PoI Attestations: {status['poi_attestations']}")
        
        elif cmd == "leaderboard":
            leaders = acc.leaderboard()
            print("\n🏆 Leaderboard:")
            for entry in leaders:
                status = "🍎" if entry["harvestable"] else "🌱"
                print(f"   {entry['rank']}. {entry['contributor']}: {entry['total_bloom']:.2f} {status}")
        
        elif cmd == "seed":
            # Test seed
            seed = acc.plant_seed(
                contributor="test:user",
                category=ImpactCategory.COMPUTATION,
                amount=100.0,
                metadata={"test": True},
            )
            print(f"\n🌱 Seed planted: {seed.seed_id}")
        
        elif cmd == "impact":
            # Test impact recording
            bloom = acc.record_computation(
                contributor="test:user",
                tokens_processed=1000,
                latency_ms=250.0,
                model="llama3.1:8b",
            )
            print(f"\n🌸 Impact recorded: +{bloom:.2f} bloom")
        
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python accumulator.py [status|leaderboard|seed|impact]")
    else:
        print(acc.status())
