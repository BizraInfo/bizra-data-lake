# Phase 52.9: Agent as a Service + Interactive Agentic Ads (Phase 8+9)

> Standing on Giants: General Magic (Telescript agent marketplace, 1994) · Harberger (self-assessed tax for resource allocation, 1962) · Ostrom (commons governance, 1990) · Lamport (Byzantine fault tolerance, 1982) · Nakamoto (trustless receipts, 2008) · Shannon (CPVA as information-theoretic pricing, 1948) · Al-Ghazali (Riba-zero economic ethics, 1095)

## 1. Overview

Phases 0-7 demonstrate a single node in isolation. Phases 8+9 open the node to the
network: Ahmed can both consume and provide agent services, and encounter a
fundamentally new form of advertising that respects sovereignty.

**AASP (Agent as a Service Protocol):** Discovery, access control, sandboxed execution,
and Harberger-tax pricing. Cross-reference: [Phase 49](../phase_49_agent_as_a_service.md).

**Interactive Agentic Ads:** Ads packaged as Telescript agents with capability,
budget, and advertiser signature. The user's HDA intercepts, consults preferences,
negotiates via AASP, and executes only with explicit consent and receipt-chain audit.
Cross-reference: [Phase 50](../phase_50_telescript_mobile_agents.md).

---

## 2. Data Flow

```
  ┌──────────────────────────────────────────────┐
  │  EXPERT REGISTRY (Federated Discovery)        │
  │  Agent Cards indexed by capability + Ihsan    │
  └──────────────────┬───────────────────────────┘
                     │ discovery
  ┌──────────────────▼───────────────────────────┐
  │  AASP PROTOCOL                                │
  │  1. Discover: search by capability            │
  │  2. Negotiate: permit + SEED payment          │
  │  3. Execute: Telescript in sandbox            │
  │  4. Verify: receipt chain + Ihsan gate        │
  │  5. Price: Harberger tax on agent slots       │
  └──────────────────┬───────────────────────────┘
                     │
  ┌──────────────────▼───────────────────────────┐
  │  INTERACTIVE AGENTIC AD                       │
  │  1. Ad arrives as Telescript + budget         │
  │  2. HDA intercepts → checks user prefs       │
  │  3. AASP negotiation (agent-to-agent)        │
  │  4. Execute if approved → receipt chain       │
  │  5. CPVA billing (pay per verified action)    │
  └──────────────────────────────────────────────┘
```

---

## 3. Pseudocode

### 3.1 AASP Protocol

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from core.integration.constants import (
    ADL_HARBERGER_TAX_RATE,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
)


@dataclass
class AgentCard:
    """Identity + capability manifest for a service agent.
    Cross-reference: Phase 49, core/a2a/schema.py."""
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    ihsan_score: float = 0.0
    success_rate: float = 0.0
    tasks_completed: int = 0
    price_per_action_seed: float = 0.0
    self_assessed_value: float = 0.0  # For Harberger tax
    owner_profile_id: str = ""
    signature: str = ""  # Ed25519

    def harberger_tax_annual(self) -> float:
        """Annual Harberger tax = self_assessed_value * ADL_HARBERGER_TAX_RATE."""
        return self.self_assessed_value * ADL_HARBERGER_TAX_RATE


@dataclass
class BizraAgentPackage:
    """.bizra-agent package structure."""
    manifest: AgentCard
    telescript_bundle: list[dict]   # Telescript actions
    test_suite: list[dict]          # Self-test definitions
    readme: str = ""
    version: str = "1.0.0"
    checksum: str = ""              # BLAKE3 of entire package


class ExpertRegistry:
    """Federated registry of agent capabilities."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}

    async def register(self, card: AgentCard) -> None:
        if card.ihsan_score < IHSAN_THRESHOLD:
            raise ValueError(f"Agent Ihsan {card.ihsan_score} below threshold")
        self._agents[card.agent_id] = card

    async def discover(self, capability: str, min_ihsan: float = IHSAN_THRESHOLD,
                       max_results: int = 10) -> list[AgentCard]:
        """Search agents by capability, filtered by Ihsan threshold."""
        matches = [a for a in self._agents.values()
                   if capability in a.capabilities and a.ihsan_score >= min_ihsan]
        return sorted(matches, key=lambda a: -a.ihsan_score)[:max_results]

    async def capability_match(self, required: list[str]) -> list[AgentCard]:
        """Find agents matching ALL required capabilities."""
        return [a for a in self._agents.values()
                if all(cap in a.capabilities for cap in required)
                and a.ihsan_score >= IHSAN_THRESHOLD]
```

### 3.2 AASP Transaction

```python
@dataclass
class AASPTransaction:
    """A single agent-to-agent service transaction."""
    tx_id: str = field(default_factory=lambda: str(uuid4()))
    requester_profile: str = ""
    provider_agent_id: str = ""
    capability: str = ""
    seed_amount: float = 0.0
    status: str = "proposed"  # proposed → accepted → executing → completed → verified
    receipt_hash: str = ""
    ihsan_score: float = 0.0
    created_at: float = field(default_factory=time.time)


class AASPProtocol:
    """Agent as a Service Protocol orchestrator."""

    def __init__(self, registry: ExpertRegistry, receipt_chain: ReceiptChain):
        self.registry = registry
        self.receipt_chain = receipt_chain

    async def request_service(self, requester: str, capability: str,
                              budget_seed: float) -> Optional[AASPTransaction]:
        # Step 1: Discover capable agents
        agents = await self.registry.discover(capability)
        if not agents:
            return None

        # Step 2: Select best (highest Ihsan, within budget)
        for agent in agents:
            if agent.price_per_action_seed <= budget_seed:
                # Step 3: Create transaction
                tx = AASPTransaction(
                    requester_profile=requester,
                    provider_agent_id=agent.agent_id,
                    capability=capability,
                    seed_amount=agent.price_per_action_seed,
                    status="accepted")

                # Step 4: Execute in sandbox
                tx.status = "executing"
                result = await self._execute_sandboxed(agent, tx)
                tx.status = "completed" if result["success"] else "failed"

                # Step 5: Receipt
                receipt = await self.receipt_chain.append(
                    action_type="aasp_service",
                    description=f"Service: {capability} from {agent.name}",
                    domain="federation", cpva_usd=tx.seed_amount,
                    ihsan_score=agent.ihsan_score)
                tx.receipt_hash = receipt.receipt_hash
                tx.ihsan_score = agent.ihsan_score
                tx.status = "verified"
                return tx

        return None

    async def _execute_sandboxed(self, agent: AgentCard,
                                  tx: AASPTransaction) -> dict:
        """Execute agent service in isolated sandbox."""
        # Telescript sandbox: no filesystem access, network-restricted
        return {"success": True, "output": "service_completed"}


def compute_harberger_price(card: AgentCard, holding_days: float) -> float:
    """Harberger tax: self-assessed value * rate * holding_period.
    Standing on Giants: Harberger (1962)."""
    daily_rate = ADL_HARBERGER_TAX_RATE / 365.0
    return card.self_assessed_value * daily_rate * holding_days
```

### 3.3 Interactive Agentic Ads

```python
@dataclass
class AgenticAd:
    """An advertisement packaged as a Telescript agent.
    Pay per verified action, not impressions."""
    ad_id: str = field(default_factory=lambda: str(uuid4()))
    advertiser_id: str = ""
    advertiser_signature: str = ""     # Ed25519
    capability_offered: str = ""       # What the ad can do for the user
    budget_seed: float = 0.0           # Max spend per interaction
    telescript: list[dict] = field(default_factory=list)
    user_benefit_description: str = "" # What the user gets
    cpva_per_action: float = 0.0       # Advertiser pays per verified action
    ihsan_score: float = 0.0

    def is_valid(self) -> bool:
        return (self.ihsan_score >= IHSAN_THRESHOLD
                and self.budget_seed > 0
                and bool(self.advertiser_signature))


class AgenticAdInterceptor:
    """User-side ad processing. Respects sovereignty and preferences."""

    def __init__(self, user_prefs: dict, aasp: AASPProtocol):
        self.prefs = user_prefs
        self.aasp = aasp

    async def process_ad(self, ad: AgenticAd) -> dict:
        """Intercept ad, check preferences, negotiate, maybe execute."""
        # Step 1: Validate ad integrity
        if not ad.is_valid():
            return {"action": "rejected", "reason": "invalid_ad"}

        # Step 2: Check user preferences
        if not self._matches_preferences(ad):
            return {"action": "ignored", "reason": "not_interested"}

        # Step 3: Present to user for approval
        approval = await self._request_approval(ad)
        if not approval:
            return {"action": "declined", "reason": "user_declined"}

        # Step 4: Execute via AASP (agent-to-agent negotiation)
        tx = await self.aasp.request_service(
            requester=self.prefs.get("profile_id", ""),
            capability=ad.capability_offered,
            budget_seed=ad.budget_seed)

        if tx and tx.status == "verified":
            return {"action": "executed", "tx_id": tx.tx_id,
                    "cpva": ad.cpva_per_action, "receipt": tx.receipt_hash}
        return {"action": "failed", "reason": "execution_failed"}

    def _matches_preferences(self, ad: AgenticAd) -> bool:
        blocked = self.prefs.get("blocked_categories", [])
        return ad.capability_offered not in blocked

    async def _request_approval(self, ad: AgenticAd) -> bool:
        """Present ad to Ghost Overlay for user approval."""
        return True  # Pseudocode: actual impl shows Ghost suggestion
```

---

## 4. Market Implications

- **Zero friction:** Discovery, negotiation, execution, payment all automated
- **Privacy-preserving:** User preferences never leave the node
- **Trust via receipts:** Every transaction has a cryptographic audit trail
- **CPVA billing:** Advertisers pay per verified action, not impressions
- **Riba-zero:** No interest, no exploitation -- Harberger tax redistributes to UBC
- **Anti-centralization:** No ad network middleman; direct agent-to-agent exchange

---

## 5. TDD Anchors

```python
import pytest

class TestAgentAsServiceAds:
    """Phase 52.9: AaaS + Agentic Ads tests."""

    @pytest.mark.asyncio
    async def test_agent_discovery(self):
        registry = ExpertRegistry()
        card = AgentCard(name="PDF Organizer", capabilities=["file_organization"],
                         ihsan_score=0.97, price_per_action_seed=0.01)
        await registry.register(card)
        results = await registry.discover("file_organization")
        assert len(results) == 1 and results[0].name == "PDF Organizer"

    @pytest.mark.asyncio
    async def test_agent_discovery_ihsan_filter(self):
        registry = ExpertRegistry()
        await registry.register(AgentCard(capabilities=["test"], ihsan_score=0.80))
        results = await registry.discover("test", min_ihsan=0.95)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_capability_match(self):
        registry = ExpertRegistry()
        await registry.register(AgentCard(
            capabilities=["ocr", "file_organization"], ihsan_score=0.96))
        matches = await registry.capability_match(["ocr", "file_organization"])
        assert len(matches) == 1

    def test_harberger_pricing(self):
        card = AgentCard(self_assessed_value=100.0)
        tax = compute_harberger_price(card, holding_days=365)
        assert abs(tax - 7.0) < 0.01  # 7% annual

    @pytest.mark.asyncio
    async def test_aasp_transaction(self):
        registry = ExpertRegistry()
        await registry.register(AgentCard(
            name="Organizer", capabilities=["file_org"],
            ihsan_score=0.97, price_per_action_seed=0.01))
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        aasp = AASPProtocol(registry=registry, receipt_chain=chain)
        tx = await aasp.request_service("ahmed-001", "file_org", 0.05)
        assert tx is not None and tx.status == "verified"

    def test_ad_validation_valid(self):
        ad = AgenticAd(ihsan_score=0.96, budget_seed=0.10,
                       advertiser_signature="sig123")
        assert ad.is_valid() is True

    def test_ad_validation_low_ihsan(self):
        ad = AgenticAd(ihsan_score=0.80, budget_seed=0.10,
                       advertiser_signature="sig")
        assert ad.is_valid() is False

    @pytest.mark.asyncio
    async def test_ad_negotiation(self):
        interceptor = AgenticAdInterceptor(
            user_prefs={"profile_id": "ahmed", "blocked_categories": []},
            aasp=mock_aasp())
        ad = AgenticAd(capability_offered="file_org", ihsan_score=0.97,
                       budget_seed=0.10, advertiser_signature="sig",
                       cpva_per_action=0.02)
        result = await interceptor.process_ad(ad)
        assert result["action"] == "executed"

    @pytest.mark.asyncio
    async def test_ad_blocked_category(self):
        interceptor = AgenticAdInterceptor(
            user_prefs={"blocked_categories": ["gambling"]},
            aasp=mock_aasp())
        ad = AgenticAd(capability_offered="gambling", ihsan_score=0.97,
                       budget_seed=0.10, advertiser_signature="sig")
        result = await interceptor.process_ad(ad)
        assert result["action"] == "ignored"

    @pytest.mark.asyncio
    async def test_ad_transaction_receipt(self):
        """Ad execution generates receipt in chain."""
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        aasp = AASPProtocol(registry=registry_with_agent(), receipt_chain=chain)
        tx = await aasp.request_service("ahmed-001", "test_cap", 0.05)
        assert chain.length() > 1  # genesis + service receipt
```
