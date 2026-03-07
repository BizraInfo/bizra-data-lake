# Module 05 — Economic Engine

> **Domain:** 3 tokens, Proof of Impact, treasury, flywheel, Agent-as-a-Service
> **Source Specs:** Phase 42 (brand token), Phase 64 (asset registry), V3 specs
> **Key Paths:** `core/token/`, `core/treasury/`, `core/bridges/`

## 5.1 Three-Token System (SEED, BRANCH, FRUIT)

**Status:** [x] BUILT
**Path:** `core/token/`

**Token roles:**
- **SEED** — base unit, minted at node creation, 2.5% zakat deduction at mint
- **BRANCH** — earned through contribution, enables governance weight
- **FRUIT** — reward token, distributed from treasury surplus

**Key class:** `TokenMinter` — accepts `ledger=`, `db_path=`, `log_path=` for isolation
**Ledger:** `.swarm/memory.db` + `04_GOLD/token_ledger.jsonl`
**Zakat:** 2.5% applied at mint time (1.0 SEED -> 0.975 net balance)

**Tests:** `tests/core/token/` — mint, transfer, zakat deduction, Gini gate

---

## 5.2 Emission Decay

**Status:** [x] BUILT
**Path:** `core/token/emission_decay.py`

Token emission follows a decay curve to prevent inflation. New token supply
decreases over time, incentivizing early adoption while maintaining scarcity.

---

## 5.3 ADL Gini Justice Gate

**Status:** [x] BUILT
**Path:** `core/integration/constants.py` (ADL_GINI_THRESHOLD = 0.35)
**Enforcement:** Simulates post-transaction Gini coefficient. Rejects if > 0.35
AND transaction increases concentration. Genesis mint exempt.

---

## 5.4 Harberger Tax

**Status:** [x] BUILT
**Path:** `core/integration/constants.py` (ADL_HARBERGER_TAX_RATE = 0.05)

5% annual wealth tax on declared asset values. Self-assessed valuation
with forced sale mechanism (if someone offers your declared price, you must sell).

---

## 5.5 Computational Zakat

**Status:** [x] BUILT
**Path:** `core/token/` (TOKEN_ZAKAT_RATE = 0.025)

2.5% of every minted token flows to community treasury.
Constitutional invariant — cannot be bypassed or reduced.

---

## 5.6 Treasury Management

**Status:** [x] BUILT
**Path:** `core/treasury/`

Resource allocation and surplus distribution. Treasury accumulates from
zakat, Harberger tax, and service fees.

---

## 5.7 Asset Registry

**Status:** [~] PARTIAL
**Path:** `core/sovereign/asset_registry.py` (Phase 64)
**Built:** Asset registration, floor constraints, hardware body detection
**Gap:** No marketplace, no valuation oracle, no Harberger forced-sale mechanism

### TDD Anchor (marketplace)
```
def test_asset_marketplace_listing():
    registry = AssetRegistry()
    registry.register("gpu_compute", declared_value=100, owner="node_a")
    listing = registry.list_marketplace()
    assert len(listing) == 1
    assert listing[0]["price"] == 100  # Self-assessed

def test_harberger_forced_sale():
    registry = AssetRegistry()
    registry.register("gpu_compute", declared_value=100, owner="node_a")
    result = registry.offer_purchase("gpu_compute", buyer="node_b", offer=100)
    assert result.accepted  # Must accept at declared price
    assert registry.owner("gpu_compute") == "node_b"
```

---

## 5.8 Proof of Impact (PoI)

**Status:** [~] PARTIAL
**Path:** Evidence ledger tracks actions, but no dedicated PoI scoring engine
**Built:** Receipt chain proves what happened
**Gap:** No impact quantification, no impact-to-token conversion rate

### TDD Anchor
```
def test_poi_score_from_evidence():
    poi = ProofOfImpactEngine(ledger=evidence_ledger)
    score = poi.compute_impact("node_a", window_hours=24)
    assert 0.0 <= score <= 1.0
    assert score > 0  # Node performed approved actions

def test_poi_to_token_emission():
    poi = ProofOfImpactEngine(ledger=evidence_ledger)
    tokens = poi.emit_rewards(period="daily")
    assert all(t.amount > 0 for t in tokens)
    assert sum(t.amount for t in tokens) <= DAILY_EMISSION_CAP
```

---

## 5.9 Economic Flywheel

**Status:** [ ] NOT BUILT
**Spec:** Users -> Data -> AI Quality -> Token Value -> More Users
**Gap:** No flywheel orchestrator, no feedback loop implementation

### Pseudocode
```
class EconomicFlywheel:
    """Virtuous cycle: usage -> quality -> value -> adoption"""

    def compute_cycle_metrics(self, period: str) -> FlywheelMetrics:
        usage = self.measure_network_usage(period)
        quality = self.measure_ai_quality(period)  # SNR aggregate
        value = self.measure_token_value(period)
        adoption = self.measure_new_nodes(period)
        return FlywheelMetrics(
            usage=usage, quality=quality,
            value=value, adoption=adoption,
            momentum=self._compute_momentum(usage, quality, value, adoption)
        )

    def _compute_momentum(self, u, q, v, a) -> float:
        """Positive momentum = flywheel accelerating"""
        return (delta(u) + delta(q) + delta(v) + delta(a)) / 4
```

---

## 5.10 Agent-as-a-Service (AaaS)

**Status:** [ ] NOT BUILT
**Spec:** Monetize agent capabilities via metered API
**Gap:** No billing, no metering, no service catalog

### Pseudocode
```
class AgentServiceCatalog:
    """Registry of agent capabilities available for purchase"""

    def list_services(self) -> List[AgentService]:
        return self.registry.query(status="available")

    def meter_usage(self, service_id: str, caller: NodeID, units: int):
        """Track usage for billing"""
        self.usage_ledger.append(service_id, caller, units, timestamp=now())

    def settle_period(self, period: str) -> List[Settlement]:
        """Convert metered usage to token transfers"""
        usage = self.usage_ledger.aggregate(period)
        return [Settlement(
            from_node=u.caller, to_node=u.provider,
            amount=u.units * u.rate, token="BRANCH"
        ) for u in usage]
```

---

## 5.11 Brand Token Integration

**Status:** [ ] NOT BUILT
**Spec:** Phase 42 (docs/specs/) — brand-specific token injection
**Gap:** No brand token factory, no injection pipeline

### Pseudocode
```
class BrandTokenFactory:
    """Create brand-specific tokens within BIZRA ecosystem"""

    def create_brand_token(self, brand: str, config: BrandConfig) -> BrandToken:
        # Validate brand against constitutional gates
        gate_result = self.fate_gate.evaluate(brand, config)
        if not gate_result.passed:
            raise ConstitutionalViolation(gate_result.reason)
        return BrandToken(
            symbol=config.symbol,
            backed_by="SEED",  # All brand tokens backed by SEED
            exchange_rate=config.initial_rate,
            zakat_rate=TOKEN_ZAKAT_RATE  # Cannot bypass
        )
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 5.1 Three Tokens | BUILT | Full |
| 5.2 Emission Decay | BUILT | Full |
| 5.3 Gini Gate | BUILT | Full |
| 5.4 Harberger Tax | BUILT | Constant |
| 5.5 Zakat | BUILT | Full |
| 5.6 Treasury | BUILT | Full |
| 5.7 Asset Registry | PARTIAL | No marketplace |
| 5.8 Proof of Impact | PARTIAL | Evidence only |
| 5.9 Flywheel | NOT BUILT | Zero |
| 5.10 AaaS | NOT BUILT | Zero |
| 5.11 Brand Token | NOT BUILT | Zero |
| **TOTAL** | **6/11 + 2P + 3N** | **64%** |
