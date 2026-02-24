# 04: Strict Rollback Automation

## Standing on Giants
Nygard (Release It! circuit breakers, 2007) · Fowler (canary releases, 2010) · Lamport (state machines, 1978)

## Overview

Automatic rollback when Phase 46 canaries breach SLO thresholds. Strict policy: any two consecutive breached evaluation windows triggers rollback. The rollback sequence is ordered (HMM -> GoT -> Search -> hard kill) and persists a receipt with timestamp, trigger, and last-good config.

## Rollback Policy

### Evaluation Windows

| Component | Window | Breach Threshold |
|-----------|--------|------------------|
| Search | 15m | error rate > 2% |
| GoT Bridge | 15m | fallback rate > 20% |
| HMM | 30m | confidence p50 < 0.55 |
| Resonance | 30m | combined_snr p50 drops > 15% from baseline |
| Latency | 30m | p95 delta > 30% from 1h-ago baseline |

### Strict Rule

```
IF breached_windows_count >= 2 (consecutive):
    EXECUTE rollback_sequence
```

Not cumulative — the 2 windows must be consecutive. A clean window resets the counter.

## Pseudocode

### `core/rollout/rollback.py`

```
MODULE rollback

IMPORT os, json, logging
FROM datetime IMPORT datetime, timezone
FROM pathlib IMPORT Path
FROM typing IMPORT Dict, List, Optional, Any
FROM dataclasses IMPORT dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
CLASS RollbackReceipt:
    """Immutable receipt for every rollback event."""
    timestamp: str
    trigger: str                    # "search_error_rate" | "got_fallback" | etc.
    breach_count: int               # How many consecutive breaches
    component: str                  # "search" | "got_bridge" | "hmm" | "all"
    action: str                     # "percent_zero" | "hard_kill"
    previous_config: Dict[str, str] # Snapshot of env vars before rollback
    metrics_snapshot: Dict[str, Any]


@dataclass
CLASS BreachWindow:
    """Tracks consecutive breaches for a single metric."""
    metric_name: str
    consecutive_count: int = 0
    last_evaluation: Optional[datetime] = None
    last_breached: bool = False


CLASS RollbackEngine:
    """Strict rollback automation for Phase 46 canary rollout.

    Policy:
    - 2 consecutive breached evaluation windows -> rollback
    - Rollback sequence: HMM % -> GoT % -> Search % -> hard kill
    - Every rollback emits a receipt to artifacts/
    """

    FUNCTION __init__(
        self,
        receipt_dir: str = "artifacts/rollback_receipts",
        metrics: Optional[Any] = None,  # Phase46Metrics
    ):
        self._receipt_dir = Path(receipt_dir)
        self._receipt_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = metrics
        self._breach_windows: Dict[str, BreachWindow] = {
            "search_error_rate": BreachWindow("search_error_rate"),
            "got_fallback_rate": BreachWindow("got_fallback_rate"),
            "hmm_confidence": BreachWindow("hmm_confidence"),
            "resonance_snr": BreachWindow("resonance_snr"),
            "latency_regression": BreachWindow("latency_regression"),
        }
        self._rollback_in_progress = False

    FUNCTION evaluate(self, metric_name: str, breached: bool) -> Optional[RollbackReceipt]:
        """Evaluate a metric window. Returns RollbackReceipt if rollback triggered.

        Call this at the end of each evaluation window for each metric.
        """
        window = self._breach_windows.get(metric_name)
        IF window IS None:
            logger.warning("Unknown metric for rollback: %s", metric_name)
            RETURN None

        window.last_evaluation = datetime.now(timezone.utc)

        IF breached:
            window.consecutive_count += 1
            window.last_breached = True
            logger.warning(
                "Rollback eval: %s breached (%d consecutive)",
                metric_name, window.consecutive_count
            )

            IF window.consecutive_count >= 2:
                # TRIGGER ROLLBACK
                RETURN self._execute_rollback(metric_name, window)
        ELSE:
            # Clean window — reset counter
            IF window.consecutive_count > 0:
                logger.info(
                    "Rollback eval: %s clean — resetting breach counter from %d",
                    metric_name, window.consecutive_count
                )
            window.consecutive_count = 0
            window.last_breached = False

        RETURN None

    FUNCTION _execute_rollback(self, trigger: str, window: BreachWindow) -> RollbackReceipt:
        """Execute the rollback sequence and emit receipt."""
        self._rollback_in_progress = True

        # Snapshot current config
        previous_config = self._snapshot_config()

        # Determine rollback scope based on trigger
        component, action = self._determine_rollback_scope(trigger)

        # Execute rollback
        self._apply_rollback(component, action)

        # Snapshot metrics
        metrics_snap = self._metrics.snapshot() IF self._metrics ELSE {}

        # Create receipt
        receipt = RollbackReceipt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            breach_count=window.consecutive_count,
            component=component,
            action=action,
            previous_config=previous_config,
            metrics_snapshot=metrics_snap,
        )

        # Persist receipt
        self._persist_receipt(receipt)

        # Reset breach counter
        window.consecutive_count = 0
        self._rollback_in_progress = False

        logger.critical(
            "ROLLBACK EXECUTED: trigger=%s component=%s action=%s",
            trigger, component, action
        )

        RETURN receipt

    FUNCTION _determine_rollback_scope(self, trigger: str) -> Tuple[str, str]:
        """Determine which component to roll back and how.

        Rollback order (reverse activation): HMM -> GoT -> Search -> hard kill
        """
        # Check current state to determine appropriate rollback
        hmm_pct = int(os.getenv("BIZRA_PHASE46_HMM_PERCENT", "0"))
        got_pct = int(os.getenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "0"))
        search_pct = int(os.getenv("BIZRA_PHASE46_SEARCH_PERCENT", "0"))

        # Component-specific triggers roll back that component first
        IF trigger IN ("hmm_confidence",) AND hmm_pct > 0:
            RETURN ("hmm", "percent_zero")
        IF trigger IN ("got_fallback_rate",) AND got_pct > 0:
            RETURN ("got_bridge", "percent_zero")
        IF trigger IN ("search_error_rate",) AND search_pct > 0:
            RETURN ("search", "percent_zero")

        # Cross-cutting triggers (latency, SNR) roll back in reverse order
        IF hmm_pct > 0:
            RETURN ("hmm", "percent_zero")
        IF got_pct > 0:
            RETURN ("got_bridge", "percent_zero")
        IF search_pct > 0:
            RETURN ("search", "percent_zero")

        # All percents already zero — hard kill
        RETURN ("all", "hard_kill")

    FUNCTION _apply_rollback(self, component: str, action: str):
        """Apply the rollback action to environment."""
        IF action == "percent_zero":
            env_map = {
                "search": "BIZRA_PHASE46_SEARCH_PERCENT",
                "got_bridge": "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
                "hmm": "BIZRA_PHASE46_HMM_PERCENT",
            }
            env_key = env_map.get(component)
            IF env_key:
                os.environ[env_key] = "0"
                logger.info("Rollback: set %s=0", env_key)

        ELIF action == "hard_kill":
            os.environ["BIZRA_PHASE46_SEARCH_ENABLED"] = "0"
            os.environ["BIZRA_PHASE46_GOT_BRIDGE_ENABLED"] = "0"
            os.environ["BIZRA_PHASE46_HMM_ENABLED"] = "0"
            os.environ["BIZRA_PHASE46_SEARCH_PERCENT"] = "0"
            os.environ["BIZRA_PHASE46_GOT_BRIDGE_PERCENT"] = "0"
            os.environ["BIZRA_PHASE46_HMM_PERCENT"] = "0"
            logger.critical("Rollback: HARD KILL — all Phase 46 disabled")

    FUNCTION _snapshot_config(self) -> Dict[str, str]:
        """Capture current Phase 46 environment state."""
        keys = [
            "BIZRA_PHASE46_SEARCH_ENABLED", "BIZRA_PHASE46_SEARCH_PERCENT",
            "BIZRA_PHASE46_GOT_BRIDGE_ENABLED", "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
            "BIZRA_PHASE46_HMM_ENABLED", "BIZRA_PHASE46_HMM_PERCENT",
        ]
        RETURN {k: os.getenv(k, "unset") FOR k IN keys}

    FUNCTION _persist_receipt(self, receipt: RollbackReceipt):
        """Write receipt to artifacts directory."""
        filename = f"rollback_{receipt.timestamp.replace(':', '-')}.json"
        path = self._receipt_dir / filename
        path.write_text(json.dumps(asdict(receipt), indent=2, default=str))
        logger.info("Rollback receipt persisted: %s", path)

    @property
    FUNCTION status(self) -> Dict[str, Any]:
        """Current rollback engine status."""
        RETURN {
            "rollback_in_progress": self._rollback_in_progress,
            "breach_windows": {
                name: {
                    "consecutive": w.consecutive_count,
                    "last_breached": w.last_breached,
                }
                FOR name, w IN self._breach_windows.items()
            },
            "receipts_dir": str(self._receipt_dir),
        }
```

## TDD Anchors

```python
class TestRollbackEngine:

    def test_single_breach_no_rollback(self):
        """One breach does not trigger rollback."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        result = engine.evaluate("search_error_rate", breached=True)
        assert result is None

    def test_two_consecutive_breaches_trigger_rollback(self):
        """Two consecutive breaches trigger rollback."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("search_error_rate", breached=True)
        receipt = engine.evaluate("search_error_rate", breached=True)
        assert receipt is not None
        assert receipt.trigger == "search_error_rate"
        assert receipt.breach_count == 2

    def test_clean_window_resets_counter(self):
        """A clean window between breaches resets the counter."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("search_error_rate", breached=True)  # count=1
        engine.evaluate("search_error_rate", breached=False)  # reset
        result = engine.evaluate("search_error_rate", breached=True)  # count=1 again
        assert result is None  # No rollback — only 1 consecutive

    def test_rollback_sets_percent_zero(self):
        """Rollback sets component percent to 0."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine = RollbackEngine(receipt_dir=tmp_path)
            engine.evaluate("search_error_rate", breached=True)
            engine.evaluate("search_error_rate", breached=True)
            assert os.environ["BIZRA_PHASE46_SEARCH_PERCENT"] == "0"

    def test_hard_kill_when_all_percents_zero(self):
        """Hard kill when all percents already at 0."""
        with patch.dict(os.environ, {
            "BIZRA_PHASE46_SEARCH_PERCENT": "0",
            "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "0",
            "BIZRA_PHASE46_HMM_PERCENT": "0",
        }):
            engine = RollbackEngine(receipt_dir=tmp_path)
            engine.evaluate("latency_regression", breached=True)
            receipt = engine.evaluate("latency_regression", breached=True)
            assert receipt.action == "hard_kill"
            assert os.environ["BIZRA_PHASE46_SEARCH_ENABLED"] == "0"

    def test_receipt_persisted_to_disk(self):
        """Rollback receipt is written as JSON."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("hmm_confidence", breached=True)
        engine.evaluate("hmm_confidence", breached=True)
        receipts = list(tmp_path.glob("rollback_*.json"))
        assert len(receipts) == 1
        data = json.loads(receipts[0].read_text())
        assert data["trigger"] == "hmm_confidence"

    def test_rollback_order_hmm_first(self):
        """Cross-cutting breach rolls back HMM first (reverse activation order)."""
        with patch.dict(os.environ, {
            "BIZRA_PHASE46_SEARCH_PERCENT": "50",
            "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "50",
            "BIZRA_PHASE46_HMM_PERCENT": "50",
        }):
            engine = RollbackEngine(receipt_dir=tmp_path)
            engine.evaluate("latency_regression", breached=True)
            receipt = engine.evaluate("latency_regression", breached=True)
            assert receipt.component == "hmm"
            assert os.environ["BIZRA_PHASE46_HMM_PERCENT"] == "0"
            # Search and GoT unchanged
            assert os.environ["BIZRA_PHASE46_SEARCH_PERCENT"] == "50"
```
