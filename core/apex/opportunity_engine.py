"""
BIZRA Opportunity Engine — Active Market Intelligence
═══════════════════════════════════════════════════════════════════════════════

SNR-maximizing autonomous engine for market opportunity detection,
trading signal generation, and arbitrage discovery.

Standing on the Shoulders of Giants:
- Shannon (1948): Information theory, channel capacity
- Markowitz (1952): Portfolio theory, risk-return optimization
- Black-Scholes (1973): Options pricing, Greeks
- Fama (1970): Efficient Market Hypothesis (and its limits)
- Lo (2004): Adaptive Markets Hypothesis

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    OpportunityEngine                          │
    │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
    │  │ Market     │  │ Signal     │  │ Arbitrage              │ │
    │  │ Analyzer   │  │ Generator  │  │ Detector               │ │
    │  └──────┬─────┘  └──────┬─────┘  └──────────┬─────────────┘ │
    │         └───────────────┼───────────────────┘               │
    │                         ▼                                   │
    │              ┌──────────────────────┐                       │
    │              │  Position Manager    │                       │
    │              └──────────────────────┘                       │
    └──────────────────────────────────────────────────────────────┘

Performance Targets:
- Signal latency: <100ms
- SNR threshold: ≥0.85
- Sharpe ratio target: >1.5

Created: 2026-02-04 | BIZRA Apex System v1.0
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (Standing on Giants)
# =============================================================================

# SNR threshold (Shannon)
SNR_THRESHOLD = 0.85

# Risk-free rate assumption (for Sharpe calculation)
RISK_FREE_RATE = 0.02

# Maximum position size (% of portfolio)
MAX_POSITION_SIZE = 0.10

# Minimum Sharpe ratio for trade (Markowitz)
MIN_SHARPE_RATIO = 1.0

# Arbitrage minimum profit threshold
ARBITRAGE_MIN_PROFIT = 0.001  # 0.1%

# Signal decay half-life (minutes)
SIGNAL_HALFLIFE_MINUTES = 5

# Market efficiency decay (Lo's AMH)
EFFICIENCY_DECAY_HOURS = 24


# =============================================================================
# ENUMS
# =============================================================================

class MarketCondition(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class SignalType(str, Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ARBITRAGE = "arbitrage"


class SignalStrength(str, Enum):
    """Signal confidence levels."""
    WEAK = "weak"        # SNR 0.6-0.7
    MODERATE = "moderate"  # SNR 0.7-0.85
    STRONG = "strong"    # SNR 0.85-0.95
    EXTREME = "extreme"  # SNR >0.95


class PositionStatus(str, Enum):
    """Status of trading positions."""
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketData:
    """Point-in-time market data."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bid: Optional[float] = None
    ask: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        if self.bid and self.ask:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.bid and self.ask and self.price > 0:
            return self.spread / self.price
        return 0.0


@dataclass
class MarketAnalysis:
    """Analysis of market conditions."""
    symbol: str
    condition: MarketCondition
    volatility: float  # Standard deviation of returns
    trend_strength: float  # [0,1]
    volume_profile: str  # "high", "normal", "low"
    efficiency_score: float  # Market efficiency [0,1] (Lo's AMH)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_tradeable(self) -> bool:
        """Market is tradeable if not too efficient and not too volatile."""
        return self.efficiency_score < 0.9 and self.volatility < 0.5


@dataclass
class TradingSignal:
    """
    A trading signal with SNR scoring.

    SNR Calculation (Shannon-inspired):
    - Signal = strength of directional conviction
    - Noise = uncertainty, conflicting indicators
    - SNR = Signal / (Signal + Noise)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    signal_type: SignalType = SignalType.HOLD
    strength: SignalStrength = SignalStrength.WEAK

    # Scores
    snr_score: float = 0.5  # [0,1]
    confidence: float = 0.5  # [0,1]
    expected_return: float = 0.0  # Expected return %
    expected_risk: float = 0.0  # Expected volatility

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=30))

    # Context
    reasoning: List[str] = field(default_factory=list)
    indicators_used: List[str] = field(default_factory=list)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio for this signal."""
        if self.expected_risk <= 0:
            return 0.0
        return (self.expected_return - RISK_FREE_RATE) / self.expected_risk

    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if SNR and Sharpe meet thresholds."""
        return (
            self.snr_score >= SNR_THRESHOLD and
            self.sharpe_ratio >= MIN_SHARPE_RATIO and
            datetime.now(timezone.utc) < self.expiry
        )

    @property
    def decay_factor(self) -> float:
        """Signal strength decay over time."""
        age_minutes = (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 60
        return math.pow(0.5, age_minutes / SIGNAL_HALFLIFE_MINUTES)


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    market_a: str = ""
    market_b: str = ""
    price_a: float = 0.0
    price_b: float = 0.0
    spread_pct: float = 0.0
    estimated_profit: float = 0.0
    execution_risk: float = 0.1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_profitable(self) -> bool:
        return self.spread_pct > ARBITRAGE_MIN_PROFIT


@dataclass
class Position:
    """A trading position."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    side: str = "long"  # "long" or "short"
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.PENDING

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Timestamps
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def pnl(self) -> float:
        if self.status == PositionStatus.CLOSED:
            return self.realized_pnl
        return self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def update_price(self, price: float):
        """Update current price and unrealized P&L."""
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.size


# =============================================================================
# MARKET ANALYZER
# =============================================================================

class MarketAnalyzer:
    """
    Analyzes market conditions for trading opportunities.

    Implements Adaptive Markets Hypothesis (Lo, 2004):
    - Markets cycle between efficiency states
    - Opportunities exist in inefficient regimes
    - Strategies must adapt to changing conditions
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._price_history: Dict[str, Deque[MarketData]] = {}
        self._analysis_cache: Dict[str, MarketAnalysis] = {}

    def update(self, data: MarketData):
        """Update market data."""
        if data.symbol not in self._price_history:
            self._price_history[data.symbol] = deque(maxlen=self.window_size)
        self._price_history[data.symbol].append(data)

    def analyze(self, symbol: str) -> MarketAnalysis:
        """Analyze market conditions for symbol."""
        history = self._price_history.get(symbol)

        if not history or len(history) < 10:
            return MarketAnalysis(
                symbol=symbol,
                condition=MarketCondition.UNKNOWN,
                volatility=0.0,
                trend_strength=0.0,
                volume_profile="unknown",
                efficiency_score=1.0
            )

        prices = [d.price for d in history]
        volumes = [d.volume for d in history]

        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
                   for i in range(1, len(prices))]

        # Volatility (standard deviation of returns)
        if returns:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance)
        else:
            volatility = 0.0

        # Trend strength (simple momentum)
        if len(prices) >= 20:
            recent_avg = sum(prices[-10:]) / 10
            older_avg = sum(prices[-20:-10]) / 10
            trend_strength = abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            trend_strength = 0.0

        # Market condition classification
        if trend_strength > 0.05:
            condition = MarketCondition.TRENDING_UP if prices[-1] > prices[-10] else MarketCondition.TRENDING_DOWN
        elif volatility > 0.03:
            condition = MarketCondition.VOLATILE
        else:
            condition = MarketCondition.RANGING

        # Volume profile
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        recent_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_volume

        if recent_volume > avg_volume * 1.5:
            volume_profile = "high"
        elif recent_volume < avg_volume * 0.5:
            volume_profile = "low"
        else:
            volume_profile = "normal"

        # Market efficiency (variance ratio test approximation)
        if len(returns) >= 20:
            # Compare short-term vs long-term variance
            short_var = sum(r ** 2 for r in returns[-10:]) / 10
            long_var = sum(r ** 2 for r in returns) / len(returns)
            efficiency_score = min(1.0, short_var / long_var if long_var > 0 else 1.0)
        else:
            efficiency_score = 0.8

        analysis = MarketAnalysis(
            symbol=symbol,
            condition=condition,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            efficiency_score=efficiency_score
        )

        self._analysis_cache[symbol] = analysis
        return analysis


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals with SNR optimization.

    SNR Maximization (Shannon foundation):
    1. Multiple indicator confluence (reduces noise)
    2. Regime-filtered signals (context awareness)
    3. Confidence-weighted output (uncertainty quantification)
    """

    def __init__(self, snr_threshold: float = SNR_THRESHOLD):
        self.snr_threshold = snr_threshold
        self._signals: Dict[str, List[TradingSignal]] = {}

    def generate_signals(
        self,
        symbol: str,
        analysis: MarketAnalysis,
        price_history: List[float]
    ) -> List[TradingSignal]:
        """Generate trading signals for symbol."""
        signals = []

        if len(price_history) < 20:
            return signals

        # Indicator 1: Moving average crossover
        ma_signal, ma_confidence = self._moving_average_signal(price_history)

        # Indicator 2: Momentum
        mom_signal, mom_confidence = self._momentum_signal(price_history)

        # Indicator 3: Mean reversion (for ranging markets)
        mr_signal, mr_confidence = self._mean_reversion_signal(price_history, analysis)

        # Combine signals with regime filter
        if analysis.condition in [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN]:
            # Trend-following signals
            combined_signal = (ma_signal * 0.5 + mom_signal * 0.5)
            combined_confidence = (ma_confidence + mom_confidence) / 2
        elif analysis.condition == MarketCondition.RANGING:
            # Mean reversion signals
            combined_signal = mr_signal
            combined_confidence = mr_confidence
        else:
            # Uncertain regime - reduce confidence
            combined_signal = ma_signal * 0.3 + mom_signal * 0.3 + mr_signal * 0.4
            combined_confidence = (ma_confidence + mom_confidence + mr_confidence) / 3 * 0.7

        # Calculate SNR
        signal_strength = abs(combined_signal)
        noise = 1.0 - combined_confidence
        snr = signal_strength / (signal_strength + noise) if signal_strength + noise > 0 else 0

        # Determine signal type
        if combined_signal > 0.3:
            signal_type = SignalType.BUY
        elif combined_signal < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Determine strength
        if snr >= 0.95:
            strength = SignalStrength.EXTREME
        elif snr >= 0.85:
            strength = SignalStrength.STRONG
        elif snr >= 0.7:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Expected return and risk
        expected_return = combined_signal * analysis.trend_strength * 0.1  # Conservative estimate
        expected_risk = analysis.volatility

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            snr_score=snr,
            confidence=combined_confidence,
            expected_return=expected_return,
            expected_risk=expected_risk,
            reasoning=[
                f"MA signal: {ma_signal:.3f}",
                f"Momentum: {mom_signal:.3f}",
                f"Mean reversion: {mr_signal:.3f}",
                f"Market condition: {analysis.condition.value}"
            ],
            indicators_used=["MA", "Momentum", "MeanReversion"]
        )

        signals.append(signal)

        # Cache signal
        if symbol not in self._signals:
            self._signals[symbol] = []
        self._signals[symbol].append(signal)

        # Keep only recent signals
        if len(self._signals[symbol]) > 100:
            self._signals[symbol] = self._signals[symbol][-100:]

        return signals

    def _moving_average_signal(self, prices: List[float]) -> Tuple[float, float]:
        """Moving average crossover signal."""
        if len(prices) < 20:
            return (0.0, 0.0)

        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20

        if long_ma == 0:
            return (0.0, 0.0)

        # Signal: positive = bullish, negative = bearish
        signal = (short_ma - long_ma) / long_ma

        # Confidence based on how clear the crossover is
        confidence = min(1.0, abs(signal) / 0.05)

        return (signal * 10, confidence)  # Scale signal

    def _momentum_signal(self, prices: List[float]) -> Tuple[float, float]:
        """Rate of change momentum signal."""
        if len(prices) < 10:
            return (0.0, 0.0)

        current = prices[-1]
        past = prices[-10]

        if past == 0:
            return (0.0, 0.0)

        momentum = (current - past) / past
        confidence = min(1.0, abs(momentum) / 0.1)

        return (momentum * 5, confidence)

    def _mean_reversion_signal(self, prices: List[float], analysis: MarketAnalysis) -> Tuple[float, float]:
        """Mean reversion signal for ranging markets."""
        if len(prices) < 20 or analysis.condition != MarketCondition.RANGING:
            return (0.0, 0.3)

        mean_price = sum(prices[-20:]) / 20
        current = prices[-1]

        if mean_price == 0:
            return (0.0, 0.0)

        # Z-score from mean
        std_dev = math.sqrt(sum((p - mean_price) ** 2 for p in prices[-20:]) / 20)

        if std_dev == 0:
            return (0.0, 0.0)

        z_score = (current - mean_price) / std_dev

        # Reversion signal: oversold = buy, overbought = sell
        signal = -z_score / 2  # Normalize

        confidence = min(1.0, abs(z_score) / 2)

        return (signal, confidence)


# =============================================================================
# ARBITRAGE DETECTOR
# =============================================================================

class ArbitrageDetector:
    """
    Detects arbitrage opportunities across markets.

    Types of arbitrage:
    - Spatial: Same asset, different exchanges
    - Temporal: Price lags between related assets
    - Statistical: Mean-reverting spreads
    """

    def __init__(self, min_profit: float = ARBITRAGE_MIN_PROFIT):
        self.min_profit = min_profit
        self._market_prices: Dict[str, Dict[str, float]] = {}  # symbol -> market -> price

    def update_price(self, symbol: str, market: str, price: float):
        """Update price for symbol on market."""
        if symbol not in self._market_prices:
            self._market_prices[symbol] = {}
        self._market_prices[symbol][market] = price

    def detect(self) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities."""
        opportunities = []

        for symbol, markets in self._market_prices.items():
            if len(markets) < 2:
                continue

            # Find min and max prices
            prices = list(markets.items())
            min_market, min_price = min(prices, key=lambda x: x[1])
            max_market, max_price = max(prices, key=lambda x: x[1])

            if min_price <= 0:
                continue

            spread_pct = (max_price - min_price) / min_price

            if spread_pct > self.min_profit:
                # Estimate execution risk (higher spread = higher risk)
                execution_risk = min(0.9, spread_pct * 10)

                opp = ArbitrageOpportunity(
                    symbol=symbol,
                    market_a=min_market,
                    market_b=max_market,
                    price_a=min_price,
                    price_b=max_price,
                    spread_pct=spread_pct,
                    estimated_profit=spread_pct - execution_risk * 0.1,
                    execution_risk=execution_risk
                )

                if opp.is_profitable:
                    opportunities.append(opp)

        return sorted(opportunities, key=lambda o: o.estimated_profit, reverse=True)


# =============================================================================
# OPPORTUNITY ENGINE (UNIFIED)
# =============================================================================

class OpportunityEngine:
    """
    Unified opportunity detection and trading engine.

    Integrates:
    - MarketAnalyzer for regime detection
    - SignalGenerator for trade signals
    - ArbitrageDetector for cross-market opportunities

    SNR Optimization:
    - Only surfaces opportunities above SNR threshold
    - Filters by Sharpe ratio for risk-adjusted returns
    - Adapts to market regime (Lo's AMH)
    """

    def __init__(
        self,
        snr_threshold: float = SNR_THRESHOLD,
        max_positions: int = 10
    ):
        self.snr_threshold = snr_threshold
        self.max_positions = max_positions

        self.analyzer = MarketAnalyzer()
        self.signal_generator = SignalGenerator(snr_threshold)
        self.arbitrage_detector = ArbitrageDetector()

        # Position tracking
        self._positions: Dict[str, Position] = {}

        # Performance tracking
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0

        logger.info("OpportunityEngine initialized")

    def process_market_data(self, data: MarketData) -> Dict[str, Any]:
        """Process market data and return opportunities."""
        # Update analyzers
        self.analyzer.update(data)
        self.arbitrage_detector.update_price(data.symbol, "primary", data.price)

        # Analyze market
        analysis = self.analyzer.analyze(data.symbol)

        # Generate signals
        history = [d.price for d in self.analyzer._price_history.get(data.symbol, [])]
        signals = self.signal_generator.generate_signals(data.symbol, analysis, history)

        # Detect arbitrage
        arbitrage = self.arbitrage_detector.detect()

        # Filter actionable opportunities
        actionable_signals = [s for s in signals if s.is_actionable]
        actionable_arbitrage = [a for a in arbitrage if a.is_profitable]

        return {
            "symbol": data.symbol,
            "analysis": analysis,
            "signals": actionable_signals,
            "arbitrage": actionable_arbitrage,
            "timestamp": datetime.now(timezone.utc)
        }

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Position]:
        """Open a new position."""
        if len(self._positions) >= self.max_positions:
            logger.warning("Max positions reached")
            return None

        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            current_price=price,
            status=PositionStatus.OPEN,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.now(timezone.utc)
        )

        self._positions[position.id] = position
        logger.info(f"Opened {side} position {position.id} for {symbol} @ {price}")
        return position

    def close_position(self, position_id: str, price: float) -> Optional[Position]:
        """Close a position."""
        position = self._positions.get(position_id)
        if not position:
            return None

        position.update_price(price)
        position.realized_pnl = position.unrealized_pnl
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now(timezone.utc)

        self._total_trades += 1
        if position.realized_pnl > 0:
            self._winning_trades += 1
        self._total_pnl += position.realized_pnl

        logger.info(f"Closed position {position_id} with P&L: {position.realized_pnl:.2f}")
        return position

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get trading performance statistics."""
        win_rate = self._winning_trades / self._total_trades if self._total_trades > 0 else 0

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": win_rate,
            "total_pnl": self._total_pnl,
            "open_positions": len([p for p in self._positions.values() if p.status == PositionStatus.OPEN]),
            "max_positions": self.max_positions
        }


# =============================================================================
# FACTORY
# =============================================================================

_opportunity_engine: Optional[OpportunityEngine] = None


def get_opportunity_engine() -> OpportunityEngine:
    """Get singleton opportunity engine."""
    global _opportunity_engine
    if _opportunity_engine is None:
        _opportunity_engine = OpportunityEngine()
    return _opportunity_engine
