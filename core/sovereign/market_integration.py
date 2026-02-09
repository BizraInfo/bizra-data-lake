"""
Market Integration — OpportunityEngine ↔ Muraqabah Engine
═══════════════════════════════════════════════════════════════════════════════

Integrates the Apex OpportunityEngine as a Muraqabah sensor array for
market intelligence with SNR-based filtering and autonomy level mapping.

Standing on the Shoulders of Giants:
- Shannon (1948): Information theory, SNR as quality metric
- Lo (2004): Adaptive Markets Hypothesis
- Markowitz (1952): Risk-adjusted returns
- Al-Ghazali (1058-1111): Muraqabah continuous vigilance

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                  MarketAwareMuraqabah                         │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
    │  │ Muraqabah      │  │ Opportunity    │  │ Autonomy       │  │
    │  │ Engine (base)  │  │ Engine (Apex)  │  │ Mapping        │  │
    │  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  │
    │           └───────────────────┴───────────────────┘          │
    └──────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Apex Integration v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.apex import (
    OpportunityEngine,
)
from core.sovereign.autonomy_matrix import AutonomyLevel
from core.sovereign.muraqabah_engine import (
    MonitorDomain,
    MuraqabahEngine,
    SensorReading,
)

logger = logging.getLogger(__name__)


from core.integration.constants import (
    SNR_THRESHOLD_T1_HIGH,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# Shannon-inspired SNR thresholds (from single source of truth)
SNR_FLOOR: float = UNIFIED_SNR_THRESHOLD
SNR_HIGH_CONFIDENCE: float = SNR_THRESHOLD_T1_HIGH
SNR_VERY_HIGH: float = STRICT_IHSAN_THRESHOLD
DATA_STALENESS_MINUTES: int = 5


class MarketSensorType(str, Enum):
    """Types of market sensors."""

    MARKET_ANALYSIS = "market_analysis"
    TRADING_SIGNAL = "trading_signal"
    ARBITRAGE = "arbitrage"
    PRICE_MOVEMENT = "price_movement"


@dataclass
class MarketSensorReading:
    """
    A reading from a market sensor adapted for Muraqabah.

    Extends SensorReading with market-specific fields.
    """

    sensor_id: str = ""
    sensor_type: MarketSensorType = MarketSensorType.MARKET_ANALYSIS
    domain: MonitorDomain = MonitorDomain.FINANCIAL
    symbol: str = ""
    value: Dict[str, Any] = field(default_factory=dict)
    snr_score: float = 0.5
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_stale: bool = False

    def to_sensor_reading(self) -> SensorReading:
        """Convert to base SensorReading for Muraqabah compatibility."""
        return SensorReading(
            sensor_id=self.sensor_id,
            domain=self.domain,
            metric_name=f"{self.sensor_type.value}:{self.symbol}",
            value=self.snr_score,
            unit="snr",
            confidence=self.confidence,
            timestamp=self.timestamp,
            metadata=self.value,
        )


@dataclass
class MarketGoal:
    """
    A proactive goal generated from market intelligence.

    Includes autonomy level based on SNR score.
    """

    goal_id: str = ""
    domain: str = "financial"
    description: str = ""
    urgency: float = 0.5
    estimated_value: float = 0.0
    autonomy_level: AutonomyLevel = AutonomyLevel.OBSERVER
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    source_reading: Optional[MarketSensorReading] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketSensorAdapter:
    """
    Adapts OpportunityEngine to Muraqabah sensor protocol.

    Transforms market signals into sensor readings compatible
    with the Muraqabah monitoring framework.

    Standing on Giants:
    - Shannon: SNR as information quality metric
    - Lo (2004): Adaptive markets hypothesis
    """

    def __init__(
        self,
        snr_threshold: float = SNR_FLOOR,
        staleness_minutes: int = DATA_STALENESS_MINUTES,
    ):
        """
        Initialize the market sensor adapter.

        Args:
            snr_threshold: Minimum SNR to pass filtering
            staleness_minutes: Minutes after which data is considered stale
        """
        self.snr_threshold = snr_threshold
        self.staleness_minutes = staleness_minutes

        # Initialize opportunity engine
        self.opportunity_engine = OpportunityEngine(snr_threshold=snr_threshold)

        # Tracking
        self._watched_symbols: set = {"COMPUTE/USD", "STORAGE/USD", "BANDWIDTH/USD"}
        self._last_scan: Optional[datetime] = None

        logger.info(f"MarketSensorAdapter initialized: SNR threshold={snr_threshold}")

    def add_watched_symbol(self, symbol: str) -> None:
        """Add a symbol to watch."""
        self._watched_symbols.add(symbol)

    def remove_watched_symbol(self, symbol: str) -> None:
        """Remove a symbol from watch list."""
        self._watched_symbols.discard(symbol)

    def _check_staleness(self, timestamp: datetime) -> bool:
        """Check if data is stale."""
        age = datetime.now(timezone.utc) - timestamp
        return age > timedelta(minutes=self.staleness_minutes)

    def scan(self) -> List[MarketSensorReading]:
        """
        Scan for market opportunities.

        Returns sensor readings compatible with Muraqabah.
        """
        readings: List[MarketSensorReading] = []
        self._last_scan = datetime.now(timezone.utc)

        # 1. Get market analysis readings
        for symbol in self._watched_symbols:
            analysis = self.opportunity_engine.analyzer.analyze(symbol)

            if analysis is None:
                continue

            is_stale = self._check_staleness(analysis.timestamp)
            # MarketAnalysis has no snr_score; derive from efficiency_score
            # Lower market efficiency → higher signal (Lo's AMH)
            base_snr = 1.0 - analysis.efficiency_score
            effective_snr = base_snr * (0.7 if is_stale else 1.0)

            reading = MarketSensorReading(
                sensor_id=f"analysis:{symbol}",
                sensor_type=MarketSensorType.MARKET_ANALYSIS,
                symbol=symbol,
                value={
                    "condition": (
                        analysis.condition.value
                        if hasattr(analysis, "condition")
                        else "unknown"
                    ),
                    "volatility": getattr(analysis, "volatility", 0.0),
                    "trend": getattr(analysis, "trend_direction", "neutral"),
                },
                snr_score=effective_snr,
                confidence=(
                    analysis.confidence if hasattr(analysis, "confidence") else 0.5
                ),
                timestamp=analysis.timestamp,
                is_stale=is_stale,
            )
            readings.append(reading)

        # 2. Get trading signals
        active_signals: List[Any] = []
        for symbol in self._watched_symbols:
            analysis = self.opportunity_engine.analyzer.analyze(symbol)
            history = [
                d.price
                for d in self.opportunity_engine.analyzer._price_history.get(symbol, [])
            ]
            active_signals.extend(
                self.opportunity_engine.signal_generator.generate_signals(
                    symbol, analysis, history
                )
            )

        for signal in active_signals:
            # Filter by SNR
            if signal.snr_score < self.snr_threshold:
                logger.debug(
                    f"Filtering low-SNR signal: {signal.id}, SNR={signal.snr_score:.2f}"
                )
                continue

            is_stale = self._check_staleness(signal.timestamp)
            effective_snr = signal.snr_score * (0.8 if is_stale else 1.0)

            reading = MarketSensorReading(
                sensor_id=f"signal:{signal.id}",
                sensor_type=MarketSensorType.TRADING_SIGNAL,
                symbol=signal.symbol,
                value={
                    "signal_type": signal.signal_type.value,
                    "strength": signal.strength.value,
                    "expected_return": signal.expected_return,
                    "expected_risk": getattr(signal, "expected_risk", 0.0),
                },
                snr_score=effective_snr,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                is_stale=is_stale,
            )
            readings.append(reading)

        # 3. Check for arbitrage opportunities
        arb_opportunities = self.opportunity_engine.arbitrage_detector.detect()

        for arb in arb_opportunities:
            reading = MarketSensorReading(
                sensor_id=f"arb:{arb.symbol}:{arb.market_a}-{arb.market_b}",
                sensor_type=MarketSensorType.ARBITRAGE,
                symbol=arb.symbol,
                value={
                    "buy_market": arb.market_a,
                    "sell_market": arb.market_b,
                    "profit_pct": arb.spread_pct,
                    "volume_available": getattr(arb, "volume_available", 0.0),
                },
                snr_score=1.0 - arb.execution_risk,
                confidence=1.0 - arb.execution_risk,
                timestamp=arb.timestamp,
                is_stale=False,  # Arbitrage is always time-sensitive
            )
            readings.append(reading)

        logger.debug(f"Market scan complete: {len(readings)} readings")
        return readings


class MarketAwareMuraqabah(MuraqabahEngine):
    """
    Muraqabah engine enhanced with market intelligence.

    Integrates OpportunityEngine sensors into the 24/7 monitoring loop
    with SNR-based filtering and autonomy level determination.

    Key Features:
    - Market sensor array (analysis, signals, arbitrage)
    - SNR-based filtering (Shannon)
    - Autonomy level mapping (SNR → autonomy)
    - Goal generation with Ihsan validation
    """

    def __init__(
        self,
        node_id: str,
        snr_threshold: float = SNR_FLOOR,
        ihsan_threshold: float = 0.95,
    ):
        """
        Initialize market-aware Muraqabah.

        Args:
            node_id: Unique identifier for this node
            snr_threshold: Minimum SNR for signal acceptance
            ihsan_threshold: Constitutional constraint threshold
        """
        super().__init__()

        self.node_id = node_id
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold

        # Initialize market sensor
        self.market_sensor = MarketSensorAdapter(snr_threshold=snr_threshold)

        # Register market sensor
        self._register_market_sensor()

        # Goal tracking
        self._pending_goals: Dict[str, MarketGoal] = {}

        logger.info(f"MarketAwareMuraqabah initialized: {node_id}")

    def _register_market_sensor(self) -> None:
        """Register market sensor with Muraqabah."""
        # The market sensor provides financial domain readings
        logger.debug("Registered market sensor for financial domain")

    async def scan_financial_domain(self) -> List[MarketSensorReading]:
        """Scan financial domain using market sensor."""
        return self.market_sensor.scan()

    async def scan_all_domains(self) -> Dict[str, List[SensorReading]]:
        """
        Scan all domains including market.

        Returns readings organized by domain.
        """
        readings: Dict[str, List[SensorReading]] = {
            "financial": [],
            "health": [],
            "social": [],
            "cognitive": [],
            "environmental": [],
        }

        # Get market readings
        market_readings = await self.scan_financial_domain()
        readings["financial"] = [r.to_sensor_reading() for r in market_readings]

        # Other domains would be scanned by parent class sensors
        # (placeholder for now)

        return readings

    def snr_to_autonomy(self, snr: float) -> AutonomyLevel:
        """
        Map SNR score to autonomy level.

        Higher SNR = more autonomous action allowed.

        Mapping:
        - SNR < 0.85: Observer (watch only)
        - SNR 0.85-0.90: Suggester (require approval)
        - SNR 0.90-0.95: AutoLow (small actions)
        - SNR 0.95-0.99: AutoMedium (medium actions)
        - SNR >= 0.99: AutoHigh (larger actions)
        """
        if snr >= SNR_VERY_HIGH:
            return AutonomyLevel.AUTOHIGH
        elif snr >= SNR_HIGH_CONFIDENCE:
            return AutonomyLevel.AUTOMEDIUM
        elif snr >= 0.90:
            return AutonomyLevel.AUTOLOW
        elif snr >= SNR_FLOOR:
            return AutonomyLevel.SUGGESTER
        else:
            return AutonomyLevel.OBSERVER

    def calculate_urgency(self, reading: MarketSensorReading) -> float:
        """
        Calculate urgency based on reading type and content.

        Arbitrage opportunities are time-sensitive (high urgency).
        """
        if reading.sensor_type == MarketSensorType.ARBITRAGE:
            return 0.95  # Very urgent — arbitrage windows close fast

        if reading.sensor_type == MarketSensorType.TRADING_SIGNAL:
            strength = reading.value.get("strength", "weak")
            strength_map = {
                "strong": 0.8,
                "moderate": 0.5,
                "weak": 0.2,
            }
            return strength_map.get(strength, 0.3)

        # Market analysis is less urgent
        return 0.3

    def estimate_value(self, reading: MarketSensorReading) -> float:
        """Estimate value of acting on this reading."""
        if reading.sensor_type == MarketSensorType.ARBITRAGE:
            profit_pct = reading.value.get("profit_pct", 0.0)
            return profit_pct * 100  # Scale to basis points

        if reading.sensor_type == MarketSensorType.TRADING_SIGNAL:
            return reading.value.get("expected_return", 0.0) * 100

        return 0.0

    def process_market_reading(
        self,
        reading: MarketSensorReading,
    ) -> Optional[MarketGoal]:
        """
        Process a market sensor reading into a potential goal.

        Returns goal if reading passes constitutional filters.

        Validation:
        1. SNR gate (must be above threshold)
        2. Staleness penalty (reduces effective SNR)
        3. Ihsan validation (constitutional compliance)
        """
        # SNR gate
        if reading.snr_score < SNR_FLOOR:
            logger.debug(
                f"Filtered by SNR: {reading.sensor_id}, SNR={reading.snr_score:.2f}"
            )
            return None

        # Calculate effective SNR with staleness penalty
        effective_snr = reading.snr_score
        if reading.is_stale:
            effective_snr *= 0.85
            if effective_snr < SNR_FLOOR:
                logger.debug(f"Filtered by staleness: {reading.sensor_id}")
                return None

        # Determine autonomy level
        autonomy_level = self.snr_to_autonomy(effective_snr)

        # Calculate urgency and value
        urgency = self.calculate_urgency(reading)
        estimated_value = self.estimate_value(reading)

        # Create goal
        goal = MarketGoal(
            goal_id=f"goal:{reading.sensor_id}",
            domain="financial",
            description=self._format_goal_description(reading),
            urgency=urgency,
            estimated_value=estimated_value,
            autonomy_level=autonomy_level,
            snr_score=effective_snr,
            source_reading=reading,
        )

        # Ihsan validation (simplified)
        ihsan_score = self._calculate_ihsan(goal)
        if ihsan_score < self.ihsan_threshold:
            logger.warning(
                f"Goal failed Ihsan: {goal.goal_id}, score={ihsan_score:.3f}"
            )
            return None

        goal.ihsan_score = ihsan_score

        # Track pending goal
        self._pending_goals[goal.goal_id] = goal

        logger.info(
            f"Created goal: {goal.goal_id}, autonomy={autonomy_level.name}, "
            f"urgency={urgency:.2f}, SNR={effective_snr:.2f}"
        )

        return goal

    def _format_goal_description(self, reading: MarketSensorReading) -> str:
        """Format a human-readable goal description."""
        if reading.sensor_type == MarketSensorType.ARBITRAGE:
            return (
                f"Arbitrage opportunity on {reading.symbol}: "
                f"buy from {reading.value.get('buy_market')}, "
                f"sell to {reading.value.get('sell_market')}, "
                f"profit {reading.value.get('profit_pct', 0) * 100:.1f}%"
            )

        if reading.sensor_type == MarketSensorType.TRADING_SIGNAL:
            return (
                f"{reading.value.get('signal_type', 'Signal').upper()} signal for {reading.symbol}: "
                f"strength={reading.value.get('strength')}, "
                f"expected return={reading.value.get('expected_return', 0) * 100:.1f}%"
            )

        return f"Market analysis update for {reading.symbol}"

    def _calculate_ihsan(self, goal: MarketGoal) -> float:
        """
        Calculate Ihsan score for a goal.

        Simplified validation based on:
        - SNR quality
        - Risk/reward ratio
        - Urgency appropriateness
        """
        snr_factor = goal.snr_score

        # Value/urgency balance (high urgency for low value is suspicious)
        if goal.urgency > 0.8 and goal.estimated_value < 10:
            balance_factor = 0.8
        else:
            balance_factor = 1.0

        # Autonomy appropriateness
        if goal.autonomy_level == AutonomyLevel.AUTOHIGH and goal.snr_score < 0.98:
            autonomy_factor = 0.9
        else:
            autonomy_factor = 1.0

        return snr_factor * balance_factor * autonomy_factor

    def get_pending_goals(self) -> List[MarketGoal]:
        """Get all pending market goals."""
        return list(self._pending_goals.values())

    def clear_expired_goals(self, max_age_minutes: int = 30) -> int:
        """Clear goals older than max_age_minutes."""
        now = datetime.now(timezone.utc)
        expired = [
            gid
            for gid, goal in self._pending_goals.items()
            if (now - goal.created_at) > timedelta(minutes=max_age_minutes)
        ]

        for gid in expired:
            del self._pending_goals[gid]

        if expired:
            logger.info(f"Cleared {len(expired)} expired goals")

        return len(expired)
