"""
BIZRA Generated Constants — DO NOT EDIT MANUALLY
Generated from constitution.toml v5.0.0-GENESIS
SHA-256: 8b020123e2b04e8ade720eb285339ee3967a321f142a7a9d253d7d47cd562422
Generated: 2026-03-03T19:57:52.695668+00:00

To modify: edit constitution.toml, then re-run generate_from_constitution.py
"""

# Constitution Reference
CONSTITUTION_VERSION = "5.0.0-GENESIS"
CONSTITUTION_HASH = "8b020123e2b04e8ade720eb285339ee3967a321f142a7a9d253d7d47cd562422"

# Ihsan Tensor: 8-dim Canonical Weights
IHSAN_CANONICAL_WEIGHTS = {
    "moral_clarity":        0.1200,
    "epistemic_humility":   0.1400,
    "structural_integrity": 0.1300,
    "verifiability":        0.1300,
    "contextual_relevance": 0.1100,
    "intent_alignment":     0.1400,
    "resilience":           0.1100,
    "efficiency":           0.1200,
}

# Ihsan Tensor: 6-dim Operational Projection (renormalized)
IHSAN_OPERATIONAL_WEIGHTS = {
    "moral_clarity":        0.1558,
    "epistemic_humility":   0.1818,
    "structural_integrity": 0.1688,
    "verifiability":        0.1688,
    "intent_alignment":     0.1818,
    "resilience":           0.1429,
}

IHSAN_DIMENSIONS_CANONICAL = 8
IHSAN_DIMENSIONS_OPERATIONAL = 6
IHSAN_OPERATIONAL_NAMES = ['moral_clarity', 'epistemic_humility', 'structural_integrity', 'verifiability', 'intent_alignment', 'resilience']

# Ihsan Thresholds
IHSAN_GATE_MINIMUM = 0.85
IHSAN_POI_CONSENSUS = 0.85
IHSAN_BLOOM_ELIGIBILITY = 0.9
IHSAN_EXCELLENCE = 0.95
IHSAN_CONFORMANCE_JOIN = 0.95

# Gate Configuration
GATE_FAIL_MODE = "closed"
GATE_OVERHEAD_BUDGET_MS = 50
GATE_WEIGHTS = {
    "alpha_4": 0.15,
    "alpha_7": 0.25,
    "alpha_8": 0.2,
    "alpha_9": 0.25,
    "alpha_10": 0.15,
}

# HHMM
HMM_NUM_HIDDEN_STATES = 47
HMM_OBSERVATION_WINDOW = 50
HMM_MAX_EM_ITERATIONS = 100
HMM_INITIAL_LIVE_STATES = 5
HMM_EXPANSION_TRIGGER = 1000

# Complexity Tier Budgets (ms)
TIER_TRIVIAL_BUDGET_MS = 100
TIER_SIMPLE_BUDGET_MS = 3000
TIER_COMPLEX_BUDGET_MS = 15000
TIER_SOVEREIGN_BUDGET_MS = 60000

# Action Bus
ACTION_BUS_GCD_TICK_MS = 100
ACTION_BUS_MAX_CONCURRENT = 10
ACTION_BUS_MAX_PER_HOUR = 100

# Economics
SEED_YEARLY_CAP = 1000000
BLOOM_IHSAN_THRESHOLD = 0.9
ZAKAT_RATE = 0.025
GINI_THRESHOLD = 0.45
GINI_MEASUREMENT_INTERVAL_S = 3600
NO_RIBA = True
NO_GHARAR = True

# Reflex Cache
REFLEX_STORE_TYPE = "HashMap"
REFLEX_MAX_ENTRIES = 500
REFLEX_PRECIPITATION_HITS = 3
REFLEX_PRECIPITATION_IHSAN = 0.9
REFLEX_SIMILARITY_THRESHOLD = 0.95
REFLEX_INVALIDATION_INTERVAL = 100
REFLEX_INVALIDATION_DELTA = 0.05
REFLEX_STALENESS_DAYS = 30

# Security: Domain Separation
DOMAIN_EVIDENCE_RECEIPT = "bizra-evidence-v1"
DOMAIN_URP_LEASE = "bizra-urp-lease-v1"
DOMAIN_POI_ATTESTATION = "bizra-poi-v1"
DOMAIN_IDENTITY_GENESIS = "bizra-identity-genesis-v1"
DOMAIN_TELESCRIPT_PUBLISH = "bizra-telescript-v1"
DOMAIN_BLOOM_MINT = "bizra-bloom-mint-v1"

# Identity
IDENTITY_KEY_ALGORITHM = "Ed25519"
IDENTITY_AGENTS_PER_NODE = 12
IDENTITY_GENESIS_DOMAIN = "bizra-identity-genesis-v1"
IDENTITY_RIGHTS = ['Exist', 'Privacy', 'Earn', 'Grow', 'Leave', 'Migrate', 'FairTreatment']

# PAT
PAT_AGENT_COUNT = 7
PAT_AGENT_NAMES = ['Planner', 'Researcher', 'Coder', 'Evaluator', 'Ethicist', 'Publisher', 'Integrator']
PAT_TRUST_STAGES = ['abstracting', 'gathering', 'executing', 'attesting', 'certifying', 'publishing', 'chaining']

# SAT
SAT_AGENTS_PER_NODE = 5
SAT_BOOTSTRAP_ROLES = ['ComputeScheduler', 'SecurityMonitor', 'PerformanceAnalyzer', 'ConsensusValidator', 'NetworkOrchestrator']
SAT_INFRASTRUCTURE_FLOOR_PCT = 20
SAT_REBALANCE_INTERVAL_S = 300
SAT_SERVICE_TYPES = ['ComputeAllocation', 'NetworkRoute', 'ConsensusVerification', 'SecurityCheck', 'TemplatePublish', 'EconomicSettlement']

# Conformance
CONFORMANCE_HHMM_ACCURACY = 1.0
CONFORMANCE_POI_VARIANCE = 0.01
CONFORMANCE_CROWN_ENTROPY = 0.95
CONFORMANCE_REFLEX_SEMANTIC = 0.9
CONFORMANCE_POOL_LATENCY_MS = 200

# Privacy
PRIVACY_CLASSES = ['LOCAL_ONLY', 'ABSTRACT_OK', 'SHAREABLE']
PRIVACY_DEFAULT = "LOCAL_ONLY"
