# Phase 65.7: Network Effects & Federated Commons

> Standing on Giants: Metcalfe (network value, 1980) · Diffie-Hellman (privacy-preserving exchange, 1976) · Al-Ghazali (commons ethics, 1095)

## 1. Purpose

Enable privacy-preserving sharing of compiled reflexes between BIZRA nodes. A user who
has myelinated a high-value reflex can contribute an anonymized version to the BIZRA
Commons. Other nodes can install community reflexes to bootstrap their own learning.
Contributors earn IMPT rewards proportional to adoption.

**Entry State**: `[FLOURISHING]` — Multiple high-success reflexes
**Exit State**: `[FLOURISHING_NETWORKED]` — Contributing to / benefiting from commons
**Trigger**: Reflex has 50+ uses, 95%+ success rate, high Ihsan average

---

## 2. Pseudocode

### 2.1 Reflex Anonymization

```
FUNCTION anonymize_reflex(
    reflex: CompiledReflex,
    pattern: ActionPattern
) -> AnonymizedReflex:
    """Strip personal data from reflex while preserving utility."""

    # CRITICAL: Never share user file paths, names, or directory structures

    anonymized = AnonymizedReflex(
        pattern_structure=generalize_regex(reflex.trigger_regex),
        action_type=reflex.action_type,
        method=reflex.method,
        safety_gates=[
            gate.generalize() FOR gate IN reflex.safety_gates
        ],
        # Performance stats (aggregate, not individual)
        avg_latency_ms=pattern.avg_latency_ms,
        success_rate=pattern.successes / pattern.occurrences,
        avg_reward=pattern.total_reward / pattern.successes,
        # Anonymized source info
        contributed_by="anonymous",  # NEVER include node_id
        contribution_hash=blake3_hash(reflex.reflex_id + utc_now_iso()),
    )

    RETURN anonymized


FUNCTION generalize_regex(trigger_regex: str) -> str:
    """Convert specific regex to general pattern."""
    # "organize C:\\Users\\Sarah\\.*  by topic"
    # → "organize .* by topic"
    # Strip all path-specific components
    RETURN strip_path_references(trigger_regex)
```

### 2.2 Contribution Protocol

```
CONSTANT CONTRIBUTION_MIN_USES = 50
CONSTANT CONTRIBUTION_MIN_SUCCESS_RATE = 0.95
CONSTANT CONTRIBUTION_MIN_IHSAN = 0.90

FUNCTION propose_contribution(
    reflex: CompiledReflex,
    pattern: ActionPattern,
    system_state: SystemState
) -> ContributionProposal | None:
    """Check if reflex qualifies for contribution to commons."""

    # Eligibility checks
    IF pattern.occurrences < CONTRIBUTION_MIN_USES:
        RETURN None
    IF pattern.successes / pattern.occurrences < CONTRIBUTION_MIN_SUCCESS_RATE:
        RETURN None

    # Ihsan check: reflex must have consistently high quality
    avg_ihsan = compute_avg_ihsan_for_pattern(pattern, system_state.ledger)
    IF avg_ihsan < CONTRIBUTION_MIN_IHSAN:
        RETURN None

    anonymized = anonymize_reflex(reflex, pattern)

    RETURN ContributionProposal(
        anonymized_reflex=anonymized,
        stats=ContributionStats(
            uses=pattern.occurrences,
            success_rate=pattern.successes / pattern.occurrences,
            avg_ihsan=avg_ihsan,
            avg_latency_ms=pattern.avg_latency_ms,
        ),
        zero_knowledge_proof=generate_success_proof(pattern, system_state)
    )
```

### 2.3 User Consent for Contribution

```
FUNCTION request_contribution_consent(
    proposal: ContributionProposal,
    system_state: SystemState
) -> ConsentResult:
    """Present contribution proposal and await user consent."""

    presentation = {
        "type": "CONTRIBUTE_TO_COMMONS",
        "pattern": proposal.anonymized_reflex.action_type,
        "your_uses": proposal.stats.uses,
        "your_success_rate": f"{proposal.stats.success_rate:.0%}",
        "what_shared": [
            "Pattern structure (e.g., 'organize by topic')",
            "Safety gates (TeleScript checks)",
            "Performance stats (latency, success rate)"
        ],
        "what_not_shared": [
            "Your file names or content",
            "Your directory structure",
            "Your identity or usage patterns"
        ],
        "reward": "50 IMPT if 10+ users benefit",
        "options": ["Contribute", "Decline"]
    }

    user_response = present_consent_dialog(presentation)

    IF user_response.choice == "Contribute":
        signature = ed25519_sign(
            system_state.identity.private_key,
            blake3_hash(json.dumps(presentation, sort_keys=True))
        )
        RETURN ConsentResult(approved=True, signature=signature)
    ELSE:
        RETURN ConsentResult(approved=False)
```

### 2.4 Commons Publishing

```
FUNCTION publish_to_commons(
    proposal: ContributionProposal,
    consent: ConsentResult,
    system_state: SystemState
) -> PublishResult:
    """Publish anonymized reflex to BIZRA Commons."""

    # Step 1: Generate zero-knowledge proof of success
    zkp = proposal.zero_knowledge_proof

    # Step 2: Publish to federated commons
    # Source: core/federation/ (P2P gossip protocol)
    commons_entry = CommonsEntry(
        reflex=proposal.anonymized_reflex,
        stats=proposal.stats,
        zkp=zkp,
        published_at=utc_now_iso()
    )

    federation.publish(commons_entry)

    # Step 3: Emit contribution receipt
    receipt = {
        "type": "COMMONS_CONTRIBUTION",
        "contribution_hash": proposal.anonymized_reflex.contribution_hash,
        "pattern_type": proposal.anonymized_reflex.action_type,
        "consent_signature": consent.signature,
        "reason_codes": ["NETWORK_CONTRIBUTION", "USER_CONSENTED"]
    }
    system_state.ledger.append(receipt=receipt)

    RETURN PublishResult(
        contribution_hash=proposal.anonymized_reflex.contribution_hash,
        tracking_id=commons_entry.tracking_id
    )
```

### 2.5 Adoption Tracking & Rewards

```
FUNCTION track_adoption(
    contribution_hash: str,
    system_state: SystemState
) -> AdoptionReport:
    """Track how many nodes installed and used the contribution."""

    # Source: core/federation/ (aggregated anonymous counters)
    adoption = federation.query_adoption(contribution_hash)

    RETURN AdoptionReport(
        installed_count=adoption.installed,
        successful_executions=adoption.executions,
        aggregate_ihsan=adoption.avg_ihsan,
        time_saved_hours=adoption.total_time_saved_hours
    )


FUNCTION claim_contribution_reward(
    contribution_hash: str,
    adoption: AdoptionReport,
    system_state: SystemState
) -> float:
    """Claim IMPT reward based on adoption metrics."""

    ADOPTION_THRESHOLD = 10  # Min users for base reward
    BASE_REWARD = 50         # IMPT
    HIGH_ADOPTION_BONUS = 25 # IMPT if > 30 users

    IF adoption.installed_count < ADOPTION_THRESHOLD:
        RETURN 0.0  # Not yet eligible

    reward = BASE_REWARD
    IF adoption.installed_count > 30:
        reward += HIGH_ADOPTION_BONUS

    # Mint reward
    system_state.impt_balance += reward

    # Emit reward receipt
    receipt = {
        "type": "COMMONS_REWARD",
        "contribution_hash": contribution_hash,
        "adopters": adoption.installed_count,
        "reward_impt": reward,
        "reason_codes": ["NETWORK_REWARD", "ADOPTION_MILESTONE"]
    }
    system_state.ledger.append(receipt=receipt)

    RETURN reward
```

### 2.6 Installing Community Reflexes

```
FUNCTION install_community_reflex(
    commons_entry: CommonsEntry,
    system_state: SystemState,
    reflex_registry: ReflexRegistry
) -> InstallResult:
    """Install a reflex from the commons into local registry."""

    # Step 1: Verify zero-knowledge proof
    IF NOT verify_zkp(commons_entry.zkp):
        RETURN InstallResult(success=False, reason="ZKP verification failed")

    # Step 2: FATE gate check on installation
    fate = fate_gate_verify_install(commons_entry, system_state)
    IF NOT fate.allowed:
        RETURN InstallResult(success=False, reason="FATE gate rejected install")

    # Step 3: Adapt to local context
    local_reflex = localize_reflex(commons_entry.reflex, system_state)

    # Step 4: Register (starts with low confidence, upgrades with local usage)
    local_reflex.confidence = 0.5  # Untested locally
    register_reflex(reflex_registry, local_reflex)

    RETURN InstallResult(
        success=True,
        reflex_id=local_reflex.reflex_id,
        source="commons"
    )
```

---

## 3. Data Structures

```
@dataclass
class AnonymizedReflex:
    pattern_structure: str         # Generalized regex (no personal paths)
    action_type: str               # "FILE_ORGANIZATION"
    method: str                    # "TOPIC_EXTRACTION"
    safety_gates: list[str]        # Generalized gate types
    avg_latency_ms: float
    success_rate: float
    avg_reward: float
    contributed_by: str            # Always "anonymous"
    contribution_hash: str         # Unique hash for tracking

@dataclass
class ContributionStats:
    uses: int
    success_rate: float
    avg_ihsan: float
    avg_latency_ms: float

@dataclass
class CommonsEntry:
    reflex: AnonymizedReflex
    stats: ContributionStats
    zkp: bytes                     # Zero-knowledge proof of success
    published_at: str
    tracking_id: str               # For adoption tracking

@dataclass
class AdoptionReport:
    installed_count: int
    successful_executions: int
    aggregate_ihsan: float
    total_time_saved_hours: float
```

---

## 4. Privacy Invariant

```
INVARIANT: Network sharing never leaks personal data.

What IS shared:     Pattern structure, gate types, aggregate stats
What is NOT shared: File paths, directory names, user identity, usage timing
What is PROVEN:     Success rate via zero-knowledge proof (no raw data)

The anonymization function is one-way:
  given AnonymizedReflex, it is computationally infeasible to recover:
    - The user's file system layout
    - The user's identity or node_id
    - The specific files acted upon
```

---

## 5. TDD Anchors

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_network.py

class TestReflexAnonymization:

    def test_anonymized_reflex_has_no_paths(self):
        """Personal file paths stripped from anonymized reflex."""
        anon = anonymize_reflex(reflex_with_user_paths, pattern)
        assert "Sarah" not in anon.pattern_structure
        assert "C:\\" not in anon.pattern_structure
        assert "/home/" not in anon.pattern_structure

    def test_anonymized_reflex_has_no_node_id(self):
        """Contributor identity is always 'anonymous'."""
        anon = anonymize_reflex(reflex, pattern)
        assert anon.contributed_by == "anonymous"


class TestContributionEligibility:

    def test_requires_50_plus_uses(self):
        """Pattern needs >= 50 uses to qualify."""
        pattern = make_pattern(occurrences=49)
        proposal = propose_contribution(reflex, pattern, state)
        assert proposal is None

    def test_requires_95_percent_success(self):
        """Pattern needs >= 95% success rate."""
        pattern = make_pattern(occurrences=100, successes=90)
        proposal = propose_contribution(reflex, pattern, state)
        assert proposal is None


class TestAdoptionRewards:

    def test_reward_at_10_adopters(self):
        """Base reward (50 IMPT) unlocked at 10 adopters."""
        adoption = AdoptionReport(installed_count=10, ...)
        reward = claim_contribution_reward(hash, adoption, state)
        assert reward == 50.0

    def test_bonus_at_30_adopters(self):
        """Bonus (25 IMPT) unlocked at 30+ adopters."""
        adoption = AdoptionReport(installed_count=47, ...)
        reward = claim_contribution_reward(hash, adoption, state)
        assert reward == 75.0

    def test_no_reward_below_threshold(self):
        """No reward if < 10 adopters."""
        adoption = AdoptionReport(installed_count=5, ...)
        reward = claim_contribution_reward(hash, adoption, state)
        assert reward == 0.0
```
