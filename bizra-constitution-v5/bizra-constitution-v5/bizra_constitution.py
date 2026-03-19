"""
BIZRA Constitution Parser — Single Source of Truth
═══════════════════════════════════════════════════

Loads constitution.toml into typed dataclasses.
Every Python module imports thresholds FROM HERE.
Zero hardcoded constants anywhere else in the codebase.

Usage:
    from bizra_constitution import load_constitution
    const = load_constitution()
    threshold = const.ihsan.thresholds.gate_minimum  # 0.85
    agents = const.pat.agents                         # List[PatAgent]
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # pip install tomli for Python <3.11


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Typed constitutional structure
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Meta:
    version: str
    created: str
    author: str
    hash_algorithm: str
    signature_scheme: str
    sacred_documents: list[str]
    constitutional_hash: str


@dataclass(frozen=True)
class IdentityRights:
    rights: list[str]
    minimum_rights_count: int

    def __post_init__(self):
        if len(self.rights) < self.minimum_rights_count:
            raise ConstitutionalViolation(
                f"Identity requires ≥{self.minimum_rights_count} rights, "
                f"got {len(self.rights)}"
            )


@dataclass(frozen=True)
class Identity:
    genesis_domain: str
    key_algorithm: str
    id_derivation: str
    sovereignty_classes: list[str]
    agent_derivation: str
    agents_per_node: int
    rights: IdentityRights


@dataclass(frozen=True)
class InteractionLaws:
    law_1: str
    law_2: str
    law_3: str
    boundary_count: int
    connection_types: int
    eliminated_attacks: list[str]
    remaining_attacks: list[str]
    sybil_mitigation: str


@dataclass(frozen=True)
class IhsanWeights:
    moral_clarity: float
    epistemic_humility: float
    structural_integrity: float
    verifiability: float
    contextual_relevance: float
    intent_alignment: float
    resilience: float
    efficiency: float

    def sum(self) -> float:
        return (
            self.moral_clarity
            + self.epistemic_humility
            + self.structural_integrity
            + self.verifiability
            + self.contextual_relevance
            + self.intent_alignment
            + self.resilience
            + self.efficiency
        )

    def validate(self):
        s = self.sum()
        if abs(s - 1.0) > 0.001:
            raise ConstitutionalViolation(f"Ihsan weights must sum to 1.0, got {s:.4f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "moral_clarity": self.moral_clarity,
            "epistemic_humility": self.epistemic_humility,
            "structural_integrity": self.structural_integrity,
            "verifiability": self.verifiability,
            "contextual_relevance": self.contextual_relevance,
            "intent_alignment": self.intent_alignment,
            "resilience": self.resilience,
            "efficiency": self.efficiency,
        }

    def project(self, dimensions: list[str]) -> dict[str, float]:
        """Project to a subset of dimensions and renormalize."""
        full = self.as_dict()
        subset = {d: full[d] for d in dimensions if d in full}
        total = sum(subset.values())
        if total == 0:
            raise ConstitutionalViolation("Projection produced zero-weight tensor")
        return {k: v / total for k, v in subset.items()}


@dataclass(frozen=True)
class IhsanThresholds:
    gate_minimum: float
    poi_consensus: float
    bloom_eligibility: float
    ihsan_excellence: float
    conformance_join: float


@dataclass(frozen=True)
class IhsanTensor:
    dimensions: int
    fail_mode: str
    canonical_weights: IhsanWeights
    operational_dimensions: list[str]
    thresholds: IhsanThresholds

    def operational_weights(self) -> dict[str, float]:
        """Get the 6-dim operational projection, renormalized."""
        return self.canonical_weights.project(self.operational_dimensions)


@dataclass(frozen=True)
class PatAgent:
    name: str
    trust_stage: str
    role: str
    key_derivation_index: int


@dataclass(frozen=True)
class Pat:
    agent_count: int
    trust_monotonicity: bool
    agents: list[PatAgent]

    def __post_init__(self):
        if len(self.agents) != self.agent_count:
            raise ConstitutionalViolation(
                f"PAT requires {self.agent_count} agents, got {len(self.agents)}"
            )


@dataclass(frozen=True)
class SatServiceTypes:
    types: list[str]


@dataclass(frozen=True)
class Sat:
    agents_per_node: int
    total_formula: str
    bootstrap_roles: list[str]
    dynamic_roles_enabled: bool
    minimum_infrastructure_pct: int
    rebalance_interval_s: int
    service_types: list[str]


@dataclass(frozen=True)
class Gate:
    name: str
    weight: float
    description: str
    pass_criterion: str


@dataclass(frozen=True)
class Gates:
    count: int
    composition: str
    fail_mode: str
    total_overhead_budget_ms: int
    alpha_4: Gate
    alpha_7: Gate
    alpha_8: Gate
    alpha_9: Gate
    alpha_10: Gate

    def all_gates(self) -> list[Gate]:
        return [self.alpha_4, self.alpha_7, self.alpha_8, self.alpha_9, self.alpha_10]

    def total_weight(self) -> float:
        return sum(g.weight for g in self.all_gates())

    def validate(self):
        w = self.total_weight()
        if abs(w - 1.0) > 0.001:
            raise ConstitutionalViolation(f"Gate weights must sum to 1.0, got {w:.3f}")


@dataclass(frozen=True)
class ComplexityTier:
    range_low: float
    range_high: float
    handler: str
    latency_budget_ms: int


@dataclass(frozen=True)
class Hhmm:
    hidden_states: int
    observation_window: int
    max_em_iterations: int
    initial_live_states: int
    expansion_trigger: int
    tiers: dict[str, ComplexityTier]
    gcd_tick_ms: int
    max_concurrent_missions: int
    max_missions_per_hour: int
    priority_formula: str


@dataclass(frozen=True)
class Economics:
    seed_yearly_cap: int
    bloom_ihsan_threshold: float
    zakat_rate: float
    zakat_constitutional: bool
    gini_threshold: float
    gini_measurement_interval_s: int
    no_riba: bool
    no_gharar: bool
    local_cost_per_mission: float
    cloud_cost_per_mission: float


@dataclass(frozen=True)
class ReflexConfig:
    store_type: str
    max_entries: int
    persistence: str
    consecutive_hits: int
    ihsan_minimum: float
    template_similarity: float
    invalidation_interval: int
    invalidation_delta: float
    staleness_max_days: int
    publish_ihsan_minimum: float


@dataclass(frozen=True)
class Conformance:
    hhmm_state_mapping_accuracy: float
    poi_calculation_variance: float
    crown_entropy_accuracy: float
    reflex_abstraction_semantic: float
    pool_latency_ms: int
    cross_language_tolerance: float


@dataclass(frozen=True)
class DomainSeparation:
    evidence_receipt: str
    urp_lease: str
    poi_attestation: str
    identity_genesis: str
    telescript_publish: str
    bloom_mint: str


@dataclass(frozen=True)
class Security:
    signature_scheme: str
    hash_algorithm: str
    domain_separation: DomainSeparation
    byzantine_tolerance_formula: str
    equivocation_possible: bool
    privacy_classes: list[str]
    default_privacy: str


@dataclass(frozen=True)
class DaughterTest:
    description: str
    type: str
    enforcement: str
    test_safe_rejection: str
    test_safe_acceptance: str


@dataclass(frozen=True)
class PsiTargets:
    targets: dict[str, float]


@dataclass(frozen=True)
class Constitution:
    """The complete BIZRA Constitution as a typed Python object."""

    meta: Meta
    identity: Identity
    interaction_laws: InteractionLaws
    ihsan: IhsanTensor
    pat: Pat
    sat: Sat
    gates: Gates
    hhmm: Hhmm
    economics: Economics
    reflex: ReflexConfig
    conformance: Conformance
    security: Security
    daughter_test: DaughterTest
    psi: PsiTargets
    raw_hash: str  # SHA-256 of the TOML file itself

    def validate(self) -> list[str]:
        """Run all constitutional invariant checks. Returns list of violations."""
        violations = []

        # Ihsan weights sum to 1.0
        try:
            self.ihsan.canonical_weights.validate()
        except ConstitutionalViolation as e:
            violations.append(str(e))

        # Gate weights sum to 1.0
        try:
            self.gates.validate()
        except ConstitutionalViolation as e:
            violations.append(str(e))

        # PAT agent count matches
        if len(self.pat.agents) != self.pat.agent_count:
            violations.append(
                f"PAT agent count mismatch: {len(self.pat.agents)} != {self.pat.agent_count}"
            )

        # Trust monotonicity: agents must define sequential stages
        if self.pat.trust_monotonicity:
            stages = [a.trust_stage for a in self.pat.agents]
            if len(set(stages)) != len(stages):
                violations.append("PAT trust stages must be unique for monotonicity")

        # Identity rights minimum
        if len(self.identity.rights.rights) < self.identity.rights.minimum_rights_count:
            violations.append("Insufficient identity rights")

        # Zakat is constitutional (cannot be zero)
        if self.economics.zakat_rate <= 0:
            violations.append("Zakat rate must be > 0 (constitutional requirement)")

        # Fail modes must be closed
        if self.ihsan.fail_mode != "closed":
            violations.append("Ihsan fail_mode must be 'closed'")
        if self.gates.fail_mode != "closed":
            violations.append("Gates fail_mode must be 'closed'")

        # Gate count matches
        if len(self.gates.all_gates()) != self.gates.count:
            violations.append("Gate count mismatch")

        # Bloom threshold ≥ gate threshold
        if self.economics.bloom_ihsan_threshold < self.ihsan.thresholds.gate_minimum:
            violations.append("BLOOM threshold must be ≥ gate minimum")

        return violations


class ConstitutionalViolation(Exception):
    """Raised when the constitution's internal invariants are violated."""

    pass


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER — Load TOML → Constitution dataclass
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_file_hash(path: Path) -> str:
    """SHA-256 of the raw TOML file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_gate(data: dict[str, Any]) -> Gate:
    return Gate(
        name=data["name"],
        weight=data["weight"],
        description=data["description"],
        pass_criterion=data["pass_criterion"],
    )


def _parse_tier(data: dict[str, Any]) -> ComplexityTier:
    r = data["range"]
    return ComplexityTier(
        range_low=r[0],
        range_high=r[1],
        handler=data["handler"],
        latency_budget_ms=data["latency_budget_ms"],
    )


def load_constitution(path: str | Path | None = None) -> Constitution:
    """
    Load the BIZRA constitution from a TOML file.

    Args:
        path: Path to constitution.toml. If None, looks for it in:
              1. ./constitution.toml
              2. ../constitution.toml
              3. $BIZRA_CONSTITUTION_PATH env var

    Returns:
        Constitution dataclass with all sections parsed and validated.

    Raises:
        ConstitutionalViolation: If any invariant check fails.
        FileNotFoundError: If constitution.toml not found.
    """
    import os

    if path is None:
        candidates = [
            Path("constitution.toml"),
            Path("../constitution.toml"),
            Path(os.environ.get("BIZRA_CONSTITUTION_PATH", "")),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                "constitution.toml not found. Set BIZRA_CONSTITUTION_PATH or "
                "place it in the project root."
            )

    path = Path(path)
    file_hash = _compute_file_hash(path)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Parse each section
    meta_data = raw["meta"]
    meta = Meta(
        version=meta_data["version"],
        created=meta_data["created"],
        author=meta_data["author"],
        hash_algorithm=meta_data["hash_algorithm"],
        signature_scheme=meta_data["signature_scheme"],
        sacred_documents=meta_data["sacred_documents"],
        constitutional_hash=file_hash,
    )

    id_data = raw["identity"]
    identity = Identity(
        genesis_domain=id_data["genesis_domain"],
        key_algorithm=id_data["key_algorithm"],
        id_derivation=id_data["id_derivation"],
        sovereignty_classes=id_data["sovereignty_classes"],
        agent_derivation=id_data["agent_derivation"],
        agents_per_node=id_data["agents_per_node"],
        rights=IdentityRights(
            rights=id_data["rights"]["rights"],
            minimum_rights_count=id_data["rights"]["minimum_rights_count"],
        ),
    )

    il_data = raw["interaction_laws"]
    interaction_laws = InteractionLaws(
        law_1=il_data["law_1"],
        law_2=il_data["law_2"],
        law_3=il_data["law_3"],
        boundary_count=il_data["boundary_count"],
        connection_types=il_data["connection_types"],
        eliminated_attacks=il_data["eliminated_attacks"],
        remaining_attacks=il_data["remaining_attacks"],
        sybil_mitigation=il_data["sybil_mitigation"],
    )

    ihsan_data = raw["ihsan_tensor"]
    ihsan = IhsanTensor(
        dimensions=ihsan_data["dimensions"],
        fail_mode=ihsan_data["fail_mode"],
        canonical_weights=IhsanWeights(**ihsan_data["canonical_weights"]),
        operational_dimensions=ihsan_data["operational_dimensions"]["dimensions"],
        thresholds=IhsanThresholds(**ihsan_data["thresholds"]),
    )

    pat_data = raw["pat"]
    pat = Pat(
        agent_count=pat_data["agent_count"],
        trust_monotonicity=pat_data["trust_monotonicity"],
        agents=[
            PatAgent(
                name=a["name"],
                trust_stage=a["trust_stage"],
                role=a["role"],
                key_derivation_index=a["key_derivation_index"],
            )
            for a in pat_data["agents"]
        ],
    )

    sat_data = raw["sat"]
    sat = Sat(
        agents_per_node=sat_data["agents_per_node"],
        total_formula=sat_data["total_formula"],
        bootstrap_roles=sat_data["bootstrap_roles"]["roles"],
        dynamic_roles_enabled=sat_data["dynamic_roles"]["enabled"],
        minimum_infrastructure_pct=sat_data["dynamic_roles"][
            "minimum_infrastructure_pct"
        ],
        rebalance_interval_s=sat_data["dynamic_roles"]["rebalance_interval_s"],
        service_types=sat_data["service_types"]["types"],
    )

    gates_data = raw["gates"]
    gates = Gates(
        count=gates_data["count"],
        composition=gates_data["composition"],
        fail_mode=gates_data["fail_mode"],
        total_overhead_budget_ms=gates_data["total_overhead_budget_ms"],
        alpha_4=_parse_gate(gates_data["alpha_4"]),
        alpha_7=_parse_gate(gates_data["alpha_7"]),
        alpha_8=_parse_gate(gates_data["alpha_8"]),
        alpha_9=_parse_gate(gates_data["alpha_9"]),
        alpha_10=_parse_gate(gates_data["alpha_10"]),
    )

    hhmm_data = raw["hhmm"]
    ab = hhmm_data["action_bus"]
    tiers_data = hhmm_data["complexity_tiers"]
    hhmm = Hhmm(
        hidden_states=hhmm_data["hidden_states"],
        observation_window=hhmm_data["observation_window"],
        max_em_iterations=hhmm_data["max_em_iterations"],
        initial_live_states=hhmm_data["initial_live_states"],
        expansion_trigger=hhmm_data["expansion_trigger"],
        tiers={k: _parse_tier(v) for k, v in tiers_data.items()},
        gcd_tick_ms=ab["gcd_tick_ms"],
        max_concurrent_missions=ab["max_concurrent_missions"],
        max_missions_per_hour=ab["max_missions_per_hour"],
        priority_formula=ab["priority_formula"],
    )

    econ = raw["economics"]
    economics = Economics(
        seed_yearly_cap=econ["seed"]["yearly_cap"],
        bloom_ihsan_threshold=econ["bloom"]["ihsan_threshold"],
        zakat_rate=econ["zakat"]["rate"],
        zakat_constitutional=econ["zakat"]["constitutional"],
        gini_threshold=econ["gini"]["threshold"],
        gini_measurement_interval_s=econ["gini"]["measurement_interval_s"],
        no_riba=econ["seed"]["no_riba"],
        no_gharar=econ["seed"]["no_gharar"],
        local_cost_per_mission=econ["local_model_advantage"]["local_cost_per_mission"],
        cloud_cost_per_mission=econ["local_model_advantage"]["cloud_cost_per_mission"],
    )

    rx = raw["reflex"]
    reflex = ReflexConfig(
        store_type=rx["store_type"],
        max_entries=rx["max_entries"],
        persistence=rx["persistence"],
        consecutive_hits=rx["precipitation"]["consecutive_hits"],
        ihsan_minimum=rx["precipitation"]["ihsan_minimum"],
        template_similarity=rx["precipitation"]["template_similarity"],
        invalidation_interval=rx["invalidation"]["check_interval"],
        invalidation_delta=rx["invalidation"]["ihsan_delta_threshold"],
        staleness_max_days=rx["invalidation"]["staleness_max_days"],
        publish_ihsan_minimum=rx["telescript_bridge"]["publish_ihsan_minimum"],
    )

    conf_data = raw["conformance"]
    conformance = Conformance(
        hhmm_state_mapping_accuracy=conf_data["hhmm_state_mapping_accuracy"],
        poi_calculation_variance=conf_data["poi_calculation_variance"],
        crown_entropy_accuracy=conf_data["crown_entropy_accuracy"],
        reflex_abstraction_semantic=conf_data["reflex_abstraction_semantic"],
        pool_latency_ms=conf_data["pool_latency_ms"],
        cross_language_tolerance=conf_data["cross_language"]["tolerance"],
    )

    sec = raw["security"]
    security = Security(
        signature_scheme=sec["signature_scheme"],
        hash_algorithm=sec["hash_algorithm"],
        domain_separation=DomainSeparation(**sec["domain_separation"]),
        byzantine_tolerance_formula=sec["byzantine"]["tolerance_formula"],
        equivocation_possible=sec["byzantine"]["equivocation_possible"],
        privacy_classes=sec["privacy_classes"]["classes"],
        default_privacy=sec["privacy_classes"]["default"],
    )

    dt = raw["daughter_test"]
    daughter_test = DaughterTest(
        description=dt["description"],
        type=dt["type"],
        enforcement=dt["enforcement"],
        test_safe_rejection=dt["test_safe_rejection"],
        test_safe_acceptance=dt["test_safe_acceptance"],
    )

    psi = PsiTargets(targets=raw["psi_targets"])

    constitution = Constitution(
        meta=meta,
        identity=identity,
        interaction_laws=interaction_laws,
        ihsan=ihsan,
        pat=pat,
        sat=sat,
        gates=gates,
        hhmm=hhmm,
        economics=economics,
        reflex=reflex,
        conformance=conformance,
        security=security,
        daughter_test=daughter_test,
        psi=psi,
        raw_hash=file_hash,
    )

    # Validate all invariants
    violations = constitution.validate()
    if violations:
        raise ConstitutionalViolation(
            f"Constitution has {len(violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    return constitution


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — Quick verification
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        c = load_constitution(path)
        print(f"✅ BIZRA Constitution v{c.meta.version} loaded successfully")
        print(f"   SHA-256: {c.raw_hash[:16]}...")
        print(f"   Ihsan weights sum: {c.ihsan.canonical_weights.sum():.4f}")
        print(f"   Gate weights sum:  {c.gates.total_weight():.3f}")
        print(f"   PAT agents:        {c.pat.agent_count}")
        print(f"   SAT per node:      {c.sat.agents_per_node}")
        print(f"   Identity rights:   {len(c.identity.rights.rights)}")
        print(f"   Eliminated attacks: {len(c.interaction_laws.eliminated_attacks)}")
        print(f"   Gate minimum:      {c.ihsan.thresholds.gate_minimum}")
        print(f"   Fail mode:         {c.gates.fail_mode}")
        print(f"   Zakat rate:        {c.economics.zakat_rate}")
        print(f"   Gini threshold:    {c.economics.gini_threshold}")
        print()
        print("   Operational Ihsan (6-dim projection):")
        for dim, weight in c.ihsan.operational_weights().items():
            print(f"     {dim:25s} {weight:.4f}")
        print()
        print(f"   0 violations. Constitution is valid.")
    except ConstitutionalViolation as e:
        print(f"❌ CONSTITUTIONAL VIOLATION: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
