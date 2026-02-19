# Phase 45.4 — Proof-of-Impact Engine

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Nakamoto (proof-of-work, 2008) · Rawls (veil of ignorance, 1971) · Gini (inequality coefficient, 1912) · Harberger (1962) · Al-Ghazali (Ihsan, 1095) · Friston (Active Inference, 2006)

## 4.1 Purpose

Define the incentive engine that makes reverse scale sustainable.
Without incentives, nodes don't participate. Without quality gates,
participation degrades. Without fairness constraints, plutocracy emerges.

Proof-of-Impact (PoI) replaces Proof-of-Work: you earn by demonstrating
measurable positive impact, not by burning electricity.

## 4.2 Core Economics — The Dual Token System

```
DUAL_TOKEN_ECONOMY:
  -- Standing on Giants: BIZRA Phase 39 (already implemented in core.token.ledger)

  SEED (بذرة):
    TYPE: utility token
    PEG: 1 SEED = 1 compute hour
    MINTED_BY: verified compute contribution
    USED_FOR: purchasing compute, priority access, task rewards
    TAXED_BY: Harberger tax (7% annual) -> UBC pool

  BLOOM (إزهار):
    TYPE: impact token
    NOT_PEGGED: value emerges from verified positive impact
    MINTED_BY: Proof-of-Impact engine
    USED_FOR: governance weight, reputation boost, priority in Shura
    REDISTRIBUTION: 50% flows back to network (thermodynamic necessity)

  CONSTITUTIONAL_CONSTRAINTS:
    RIBA_ZERO: No interest, no exploitation, no rent-seeking
    ADL_GINI: Gini coefficient <= 0.35 (hard gate, from constants.py)
    HARBERGER: Self-assessed resource tax prevents hoarding
    UBC: Universal Basic Compute pool ensures minimum participation
```

## 4.3 Impact Categories — Pseudocode

```
MODULE: core.proof_of_impact.categories

ENUM ImpactCategory:
  """What kind of positive impact was produced?"""

  COMPUTE_CONTRIBUTION    = "compute"      -- shared CPU/GPU cycles
  KNOWLEDGE_CONTRIBUTION  = "knowledge"    -- added valuable knowledge to mesh
  REASONING_CONTRIBUTION  = "reasoning"    -- solved problems, generated insights
  VALIDATION_CONTRIBUTION = "validation"   -- verified others' work, found errors
  TEACHING_CONTRIBUTION   = "teaching"     -- helped other nodes learn
  GOVERNANCE_CONTRIBUTION = "governance"   -- participated in Shura decisions
  INFRASTRUCTURE          = "infrastructure" -- maintained mesh health

CLASS ImpactWeights:
  """
  How different contributions are valued.

  Standing on Giants:
    Ihsan dimension weights from constants.py
    Adjusted for distributed context.
  """
  WEIGHTS = {
    ImpactCategory.COMPUTE_CONTRIBUTION:    0.15,
    ImpactCategory.KNOWLEDGE_CONTRIBUTION:  0.20,
    ImpactCategory.REASONING_CONTRIBUTION:  0.25,
    ImpactCategory.VALIDATION_CONTRIBUTION: 0.20,
    ImpactCategory.TEACHING_CONTRIBUTION:   0.10,
    ImpactCategory.GOVERNANCE_CONTRIBUTION: 0.05,
    ImpactCategory.INFRASTRUCTURE:          0.05,
  }
  -- Weights sum to 1.0
  -- Reasoning and validation weighted highest (quality over quantity)
```

## 4.4 Proof-of-Impact Scoring — Pseudocode

```
MODULE: core.proof_of_impact.scorer

IMPORT: blake3_digest, canonical_bytes from core.proof_engine.canonical
IMPORT: UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD from core.integration.constants
IMPORT: MerkleTree from core.hashtable.merkle_tree

CLASS ProofOfImpact:
  """
  Score and verify positive impact from node contributions.

  Every impact claim must be:
  1. Backed by a PCI receipt (proof it happened)
  2. Validated by at least one other node (no self-dealing)
  3. Above Ihsan floor (quality minimum)
  4. Recorded in an immutable Merkle chain (audit trail)
  """

  FIELDS:
    impact_tree: MerkleTree         -- Merkle chain of all impact records
    pending_claims: list[ImpactClaim]
    validated_impacts: list[ValidatedImpact]

  METHOD submit_claim(claim: ImpactClaim) -> str:
    """
    Submit an impact claim for validation.
    Returns claim_id for tracking.
    """
    -- Gate 1: Receipt must be present and valid
    IF NOT claim.receipt.is_valid():
      RAISE InvalidReceiptError("ZANN: no unverified impact claims")

    -- Gate 2: SNR must meet minimum
    IF claim.snr_score < UNIFIED_SNR_THRESHOLD:
      RAISE QualityBelowFloorError(
        f"SNR {claim.snr_score} below floor {UNIFIED_SNR_THRESHOLD}"
      )

    -- Gate 3: Must not be self-validated
    IF claim.claimed_by == claim.primary_validator:
      RAISE SelfDealingError("Cannot validate your own impact")

    claim.claim_id = blake3_digest(canonical_bytes(claim.to_dict())).hex()[:16]
    self.pending_claims.append(claim)
    RETURN claim.claim_id

  METHOD validate_claim(claim_id: str, validator_node_id: str,
                         validation: ValidationResult) -> bool:
    """
    Another node validates (or rejects) an impact claim.
    Requires independent verification.
    """
    claim = find_pending(claim_id)

    IF validator_node_id == claim.claimed_by:
      RAISE SelfDealingError("Cannot validate your own claim")

    -- Record validation
    claim.validations.append(validation)

    -- Check if enough validations to finalize
    IF len(claim.validations) >= claim.required_validations:
      RETURN self._finalize_claim(claim)

    RETURN false  -- still pending

  METHOD _finalize_claim(claim: ImpactClaim) -> bool:
    """Finalize a claim that has enough validations."""

    -- Majority must approve
    approvals = [v for v in claim.validations if v.approved]
    IF len(approvals) < len(claim.validations) / 2:
      claim.status = "rejected"
      RETURN false

    -- Calculate impact score
    base_score = claim.snr_score * ImpactWeights.WEIGHTS[claim.category]

    -- Adjust by validator consensus strength
    consensus_factor = len(approvals) / len(claim.validations)
    final_score = base_score * consensus_factor

    -- Ihsan floor check
    IF final_score < UNIFIED_IHSAN_THRESHOLD * 0.5:
      claim.status = "rejected_quality"
      RETURN false

    -- Create validated impact record
    validated = ValidatedImpact(
      claim_id = claim.claim_id,
      node_id = claim.claimed_by,
      category = claim.category,
      impact_score = final_score,
      validators = [v.validator_id for v in claim.validations if v.approved],
      receipt_digest = claim.receipt.digest(),
      timestamp = utc_now(),
    )

    -- Record in Merkle chain (immutable audit trail)
    self.impact_tree.append(canonical_bytes(validated.to_dict()))
    self.validated_impacts.append(validated)

    -- Mint BLOOM tokens proportional to impact
    bloom_reward = final_score * BLOOM_MINT_RATE
    self.ledger.mint_bloom(claim.claimed_by, bloom_reward)

    -- Also reward validators (incentivize honest validation)
    validator_reward = bloom_reward * 0.10  -- 10% of minted BLOOM
    FOR v IN claim.validations:
      IF v.approved:
        self.ledger.mint_bloom(v.validator_id, validator_reward / len(approvals))

    claim.status = "validated"
    RETURN true

CONST BLOOM_MINT_RATE: float = 10.0  -- BLOOM per unit impact score
```

## 4.5 Reputation Engine — Pseudocode

```
MODULE: core.proof_of_impact.reputation

CLASS ReputationEngine:
  """
  Network-wide reputation scoring.

  Reputation = f(impact_history, consistency, validation_accuracy, uptime)

  Standing on Giants:
    EigenTrust (Kamvar et al., 2003) — distributed reputation
    PageRank (Page et al., 1998) — authority from network structure
  """

  FIELDS:
    scores: dict[str, ReputationScore]    -- node_id -> score
    impact_engine: ProofOfImpact

  METHOD update_reputation(node_id: str) -> float:
    """Recalculate reputation from impact history."""
    impacts = self.impact_engine.get_impacts_for(node_id)

    IF len(impacts) == 0:
      -- New node or inactive: apply decay only
      self.scores[node_id].apply_decay()
      RETURN self.scores[node_id].score

    -- Component 1: Impact quality (weighted average of recent impacts)
    recent = impacts[-50:]  -- last 50 impacts
    quality_score = mean([i.impact_score for i in recent])

    -- Component 2: Consistency (standard deviation — lower is better)
    consistency = 1.0 - min(1.0, stdev([i.impact_score for i in recent]))

    -- Component 3: Validation accuracy (for nodes that validate others)
    validation_accuracy = self._validation_accuracy(node_id)

    -- Component 4: Uptime / availability
    uptime = self._uptime_score(node_id)

    -- Weighted combination
    reputation = (
      quality_score * 0.40
      + consistency * 0.25
      + validation_accuracy * 0.20
      + uptime * 0.15
    )

    self.scores[node_id].score = clamp(reputation, 0.0, 1.0)
    self.scores[node_id].last_active = utc_now()

    RETURN self.scores[node_id].score

  METHOD _validation_accuracy(node_id: str) -> float:
    """
    How accurately does this node validate others' work?

    If a node approves garbage or rejects good work,
    their validation accuracy drops.
    """
    validations = get_validations_by(node_id)
    IF len(validations) < 5:
      RETURN 0.5  -- insufficient data

    -- Compare to consensus: did this node agree with the majority?
    agreements = sum(1 for v in validations if v.agreed_with_consensus)
    RETURN agreements / len(validations)

  METHOD sybil_resistance_check(node_id: str) -> bool:
    """
    Detect potential sybil nodes.

    Red flags:
    - Very new node with high activity
    - Impact only from self-dealing circles
    - Reputation spikes without diverse validators
    """
    score = self.scores.get(node_id)
    IF score IS None:
      RETURN true  -- new node, not yet suspicious

    -- Flag: new node with rapid reputation growth
    age_days = (utc_now() - score.genesis_timestamp).days
    IF age_days < 7 AND score.score > 0.5:
      RETURN false  -- suspicious

    -- Flag: all impacts validated by same small set of nodes
    validators = get_all_validators_for(node_id)
    unique_validators = set(validators)
    IF len(unique_validators) < 3 AND score.total_tasks_completed > 10:
      RETURN false  -- suspicious cluster

    RETURN true
```

## 4.6 Shura (Collective Decision-Making) — Pseudocode

```
MODULE: core.proof_of_impact.shura

CLASS ShuraCouncil:
  """
  Collective decision-making weighted by reputation.

  Standing on Giants:
    Shura (Islamic consultation) — obligatory collective counsel
    Condorcet (jury theorem, 1785) — diverse independent judges converge on truth
    BFT (Lamport et al., 1982) — agreement despite Byzantine faults

  Key principle: reputation-weighted voting with ADL Gini cap.
  No single node, no matter how reputed, can dominate decisions.
  """

  METHOD propose(proposal: Proposal) -> str:
    """Submit a governance proposal for Shura voting."""
    -- Proposer must have minimum reputation
    IF proposer_reputation < 0.30:
      RAISE InsufficientReputationError("Min 0.30 reputation to propose")

    proposal.id = generate_proposal_id()
    proposal.status = "voting"
    proposal.voting_deadline = utc_now() + timedelta(hours=72)
    broadcast_to_mesh(proposal)
    RETURN proposal.id

  METHOD vote(proposal_id: str, node_id: str, vote: "approve" | "reject",
              reason: str) -> None:
    """Cast a reputation-weighted vote."""
    reputation = self.reputation_engine.scores[node_id].score

    -- ADL Gini enforcement: cap any single vote weight
    max_weight = 1.0 / max(10, len(eligible_voters))  -- no node > 10% weight
    vote_weight = min(reputation, max_weight)

    record_vote(proposal_id, node_id, vote, vote_weight, reason)

  METHOD tally(proposal_id: str) -> ShuraResult:
    """Count votes after deadline."""
    votes = get_votes_for(proposal_id)

    approve_weight = sum(v.weight for v in votes if v.vote == "approve")
    reject_weight = sum(v.weight for v in votes if v.vote == "reject")
    total_weight = approve_weight + reject_weight

    IF total_weight == 0:
      RETURN ShuraResult(status="no_quorum")

    -- Supermajority required for governance changes: 67%
    approval_ratio = approve_weight / total_weight
    passed = approval_ratio >= 0.67

    -- Verify ADL: no single node dominated the outcome
    max_single_weight = max(v.weight for v in votes)
    gini_check = max_single_weight / total_weight < ADL_GINI_THRESHOLD

    RETURN ShuraResult(
      status = "passed" if (passed and gini_check) else "rejected",
      approval_ratio = approval_ratio,
      total_votes = len(votes),
      gini_compliant = gini_check,
    )
```

## 4.7 Anti-Exploitation Safeguards

```
SAFEGUARDS:

  riba_zero:
    -- No interest on compute loans
    -- No rent-seeking on idle resources
    -- Harberger tax ensures resources flow to active users
    ENFORCED_BY: core.token.ledger Harberger tax gate

  adl_gini:
    -- Max Gini coefficient 0.35
    -- Transactions that would push Gini above threshold are REJECTED
    -- Applied to: SEED holdings, BLOOM holdings, governance weight
    ENFORCED_BY: core.token.ledger ADL Gini gate

  zann_zero:
    -- No impact claim without PCI receipt
    -- No reputation without verified impact
    -- No self-dealing (validate your own work)
    ENFORCED_BY: ProofOfImpact.submit_claim gates

  sybil_resistance:
    -- Reputation decay prevents stockpiling
    -- Diverse validator requirement
    -- New node probation period
    -- Cluster detection
    ENFORCED_BY: ReputationEngine.sybil_resistance_check

  daughter_test:
    -- "Would I be proud if my daughter saw this interaction?"
    -- Applied to: task assignments, knowledge sharing, governance
    ENFORCED_BY: Constitutional Ihsan gate
```

## 4.8 TDD Anchors

```
TEST_SUITE: tests/core/proof_of_impact/

  test_impact_submission:
    - valid claim with receipt accepted
    - claim without receipt rejected (ZANN)
    - claim below SNR floor rejected
    - self-validated claim rejected
    - claim_id is deterministic BLAKE3 hash

  test_validation:
    - majority approval finalizes claim
    - majority rejection rejects claim
    - validators earn BLOOM reward
    - self-validation prevented
    - minimum validators required before finalization

  test_reputation:
    - new node starts at 0.10
    - consistent high-quality impacts raise score
    - inactivity decays score
    - validation accuracy affects score
    - score clamped to [0.0, 1.0]

  test_sybil_resistance:
    - rapid reputation growth flagged
    - small validator circle flagged
    - diverse, organic growth passes

  test_shura:
    - low-reputation proposer rejected
    - supermajority (67%) required to pass
    - ADL Gini cap prevents single-node dominance
    - no quorum returns no_quorum status

  test_economics:
    - SEED minted for compute (1:1 hour peg)
    - BLOOM minted for validated impact
    - Harberger tax flows to UBC pool
    - Gini gate blocks concentrating transactions
    - BLOOM redistribution at 50%
```
