# Market Integration Specification

## Phase 2: OpportunityEngine ↔ Muraqabah Engine

**Module**: `core/sovereign/market_integration.py`
**Dependencies**: `core/apex/opportunity_engine.py`, `core/sovereign/muraqabah_engine.py`
**Lines**: ~180

---

## Functional Requirements

### FR-1: Market Sensor Registration

Register OpportunityEngine components as Muraqabah sensors:
- MarketAnalyzer → Financial domain sensor
- SignalGenerator → Opportunity detection
- ArbitrageDetector → Cross-market opportunities

### FR-2: SNR-Based Filtering

All signals must pass SNR threshold before action:
- SNR < 0.85 → Log and discard
- SNR ≥ 0.85 → Publish to opportunity pipeline
- SNR ≥ 0.95 → High-confidence, eligible for autonomous action

### FR-3: Autonomy Level Mapping

Map signal confidence to autonomy levels:
- SNR 0.85-0.90 → Level 1 (Suggester) — Require approval
- SNR 0.90-0.95 → Level 2 (AutoLow) — Execute small actions
- SNR 0.95-0.99 → Level 3 (AutoMedium) — Execute medium actions
- SNR ≥ 0.99 → Level 4 (AutoHigh) — Execute larger actions

---

## Pseudocode

```
MODULE market_integration

IMPORT OpportunityEngine, TradingSignal, MarketData FROM core.apex
IMPORT MuraqabahEngine, SensorReading FROM core.sovereign.muraqabah_engine
IMPORT AutonomyMatrix, AutonomyLevel FROM core.sovereign.autonomy_matrix

# Constants (Shannon-inspired thresholds)
CONST SNR_FLOOR: FLOAT = 0.85
CONST SNR_HIGH_CONFIDENCE: FLOAT = 0.95
CONST SNR_VERY_HIGH: FLOAT = 0.99

CLASS MarketSensorAdapter:
    """
    Adapts OpportunityEngine to Muraqabah sensor protocol.

    Standing on Giants:
    - Shannon: SNR as information quality metric
    - Lo (2004): Adaptive markets hypothesis
    - Markowitz: Risk-adjusted returns
    """

    PROPERTY opportunity_engine: OpportunityEngine
    PROPERTY snr_threshold: FLOAT = SNR_FLOOR

    CONSTRUCTOR(self, snr_threshold: FLOAT = SNR_FLOOR):
        self.opportunity_engine = OpportunityEngine(snr_threshold=snr_threshold)
        self.snr_threshold = snr_threshold

    METHOD scan(self) -> LIST[SensorReading]:
        """
        Scan for market opportunities.

        Returns sensor readings compatible with Muraqabah.
        """
        readings = []

        # 1. Get latest market analysis
        FOR symbol IN self.opportunity_engine.watched_symbols:
            analysis = self.opportunity_engine.market_analyzer.get_analysis(symbol)

            IF analysis IS NONE:
                CONTINUE

            reading = SensorReading(
                domain="financial",
                sensor_type="market_analysis",
                value={
                    "symbol": symbol,
                    "condition": analysis.condition.value,
                    "volatility": analysis.volatility,
                    "trend": analysis.trend_direction,
                },
                snr=analysis.snr_score,
                timestamp=analysis.timestamp
            )
            readings.APPEND(reading)

        # 2. Get trading signals
        signals = self.opportunity_engine.signal_generator.get_active_signals()

        FOR signal IN signals:
            IF signal.snr_score < self.snr_threshold:
                LOG.debug(f"Filtering low-SNR signal: {signal.id}, SNR={signal.snr_score}")
                CONTINUE

            reading = SensorReading(
                domain="financial",
                sensor_type="trading_signal",
                value={
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "type": signal.signal_type.value,
                    "strength": signal.strength.value,
                    "expected_return": signal.expected_return,
                },
                snr=signal.snr_score,
                timestamp=signal.timestamp
            )
            readings.APPEND(reading)

        # 3. Check for arbitrage opportunities
        arb_opportunities = self.opportunity_engine.arbitrage_detector.detect()

        FOR arb IN arb_opportunities:
            reading = SensorReading(
                domain="financial",
                sensor_type="arbitrage",
                value={
                    "symbol": arb.symbol,
                    "buy_market": arb.buy_market,
                    "sell_market": arb.sell_market,
                    "profit_pct": arb.profit_percentage,
                },
                snr=arb.confidence,
                timestamp=arb.detected_at
            )
            readings.APPEND(reading)

        RETURN readings


CLASS MarketAwareMuraqabah EXTENDS MuraqabahEngine:
    """
    Muraqabah engine enhanced with market intelligence.

    Integrates OpportunityEngine sensors into the 24/7 monitoring loop.
    """

    PROPERTY market_sensor: MarketSensorAdapter
    PROPERTY autonomy_matrix: AutonomyMatrix

    CONSTRUCTOR(self, node_id: STRING):
        SUPER().__init__(node_id)
        self.market_sensor = MarketSensorAdapter(snr_threshold=SNR_FLOOR)
        self.autonomy_matrix = AutonomyMatrix()

        # Register market sensor
        self.register_sensor("market", self.market_sensor)

    METHOD process_market_reading(self, reading: SensorReading) -> Optional[ProactiveGoal]:
        """
        Process a market sensor reading into a potential goal.

        Returns goal if reading passes constitutional filters.
        """
        # SNR gate
        IF reading.snr < SNR_FLOOR:
            RETURN NONE

        # Determine autonomy level based on SNR
        autonomy_level = self._snr_to_autonomy(reading.snr)

        # Create goal
        goal = ProactiveGoal(
            domain="financial",
            description=self._format_goal_description(reading),
            urgency=self._calculate_urgency(reading),
            estimated_value=reading.value.get("expected_return", 0) * 100,
            autonomy_level=autonomy_level,
            source_reading=reading
        )

        # Constitutional filter
        ihsan_score = self._calculate_ihsan(goal)
        IF ihsan_score < IHSAN_THRESHOLD:
            LOG.warning(f"Goal failed Ihsan check: {ihsan_score}")
            RETURN NONE

        goal.constitutional_score = ihsan_score
        RETURN goal

    METHOD _snr_to_autonomy(self, snr: FLOAT) -> AutonomyLevel:
        """
        Map SNR score to autonomy level.

        Higher SNR = more autonomous action allowed.
        """
        IF snr >= SNR_VERY_HIGH:
            RETURN AutonomyLevel.AUTOHIGH
        ELIF snr >= SNR_HIGH_CONFIDENCE:
            RETURN AutonomyLevel.AUTOMEDIUM
        ELIF snr >= 0.90:
            RETURN AutonomyLevel.AUTOLOW
        ELSE:
            RETURN AutonomyLevel.SUGGESTER

    METHOD _calculate_urgency(self, reading: SensorReading) -> FLOAT:
        """
        Calculate urgency based on signal type and strength.

        Arbitrage opportunities are time-sensitive.
        """
        IF reading.sensor_type == "arbitrage":
            RETURN 0.95  # Very urgent — arbitrage windows close fast

        IF reading.sensor_type == "trading_signal":
            strength_map = {
                "strong": 0.8,
                "moderate": 0.5,
                "weak": 0.2
            }
            RETURN strength_map.get(reading.value.get("strength"), 0.3)

        RETURN 0.3  # Default moderate urgency

---

## TDD Anchors

### Test: SNR Filtering

```python
def test_low_snr_signals_filtered():
    """Signals below SNR threshold should not create goals."""
    muraqabah = MarketAwareMuraqabah(node_id="test")

    low_snr_reading = SensorReading(
        domain="financial",
        sensor_type="trading_signal",
        value={"symbol": "TEST/USD", "type": "buy"},
        snr=0.60,  # Below threshold
        timestamp=datetime.now()
    )

    goal = muraqabah.process_market_reading(low_snr_reading)
    assert goal is None
```

### Test: SNR to Autonomy Mapping

```python
def test_snr_maps_to_correct_autonomy():
    """SNR scores should map to correct autonomy levels."""
    muraqabah = MarketAwareMuraqabah(node_id="test")

    # Very high SNR → AutoHigh
    assert muraqabah._snr_to_autonomy(0.99) == AutonomyLevel.AUTOHIGH

    # High SNR → AutoMedium
    assert muraqabah._snr_to_autonomy(0.96) == AutonomyLevel.AUTOMEDIUM

    # Medium SNR → AutoLow
    assert muraqabah._snr_to_autonomy(0.91) == AutonomyLevel.AUTOLOW

    # Low (but passing) SNR → Suggester
    assert muraqabah._snr_to_autonomy(0.86) == AutonomyLevel.SUGGESTER
```

### Test: Arbitrage Urgency

```python
def test_arbitrage_has_high_urgency():
    """Arbitrage opportunities should have high urgency."""
    muraqabah = MarketAwareMuraqabah(node_id="test")

    arb_reading = SensorReading(
        domain="financial",
        sensor_type="arbitrage",
        value={"symbol": "COMPUTE/USD", "profit_pct": 0.02},
        snr=0.92,
        timestamp=datetime.now()
    )

    urgency = muraqabah._calculate_urgency(arb_reading)
    assert urgency >= 0.9
```

---

## Data Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Market Data     │────▶│ MarketAnalyzer   │────▶│ Analysis        │
│ (price, volume) │     │ (trend, vol)     │     │ (SNR scored)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Market Patterns │────▶│ SignalGenerator  │────▶│ Trading Signals │
│ (indicators)    │     │ (SNR filter)     │     │ (SNR ≥ 0.85)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Cross-Market    │────▶│ ArbitrageDetector│────▶│ Arbitrage Opps  │
│ Prices          │     │ (min profit)     │     │ (time-sensitive)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │ MarketSensorAdapter  │
                                               │ (Muraqabah sensor)   │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │ MarketAwareMuraqabah │
                                               │ (goal generation)    │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │ AutonomyMatrix       │
                                               │ (action decision)    │
                                               └──────────────────────┘
```

---

## Edge Cases

| Case | Handling |
|------|----------|
| No market data available | Return empty readings, log warning |
| Stale data (>5 min old) | Mark reading as stale, reduce SNR |
| Conflicting signals | Average SNR, prefer higher confidence |
| Market closed | Skip market scan, return cached analysis |

---

## File Output

**Target**: `core/sovereign/market_integration.py`
**Lines**: ~180
**Imports**:
```python
from core.apex import OpportunityEngine, TradingSignal, MarketData
from core.sovereign.muraqabah_engine import MuraqabahEngine, SensorReading
from core.sovereign.autonomy_matrix import AutonomyMatrix, AutonomyLevel
```
