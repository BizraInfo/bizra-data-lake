//! BIZRA Eval Engine v1 — Genesis Valuation via Proof-of-Impact
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/eval/v1.rs
//! Domain tag: bizra-eval-v1
//!
//! Integration points:
//!   - receipts.rs: Blake3Hash, ReceiptPayload, ReceiptPayloadDecode, ByteReader, ReceiptKind
//!   - configure_cognition.rs: GraphNodeFactory, ConfigureError
//!   - thought_graph.rs: GraphNode, AgentCtx, Thought
//!   - canonical_hasher.rs: blake3_domain
//!
//! Constitutional authority:
//!   البذرة §"كل أرباح المشروع من جميع الخدمات والأدوات ستحول نصف الأرباح إلى الحوض"
//!
//! Design invariant:
//!   E1: evaluate(config, evidence, comparables) is a PURE FUNCTION.
//!       Same inputs → same output. No RNG. No clock. No external state.
//!       Any node can re-run and verify.

use std::collections::BTreeMap;

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash,
    ReceiptPayload, ReceiptPayloadDecode, ReceiptKind,
    ByteReader, DecodeError,
};
use crate::configure::cognition::{GraphNodeFactory, ConfigureError};

// ════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum EvalError {
    EmptyEvidenceChain,
    ComparableSetEmpty,
    IhsanBelowFloor { score: f64, floor: f64 },
    ReproducibilityMismatch { expected: Blake3Hash, got: Blake3Hash },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEvidenceChain => write!(f, "Evidence chain is empty"),
            Self::ComparableSetEmpty => write!(f, "Comparable set is empty"),
            Self::IhsanBelowFloor { score, floor } =>
                write!(f, "Ihsan score {:.4} below floor {:.4}", score, floor),
            Self::ReproducibilityMismatch { expected, got } =>
                write!(f, "Reproducibility hash mismatch"),
        }
    }
}

impl std::error::Error for EvalError {}

// ════════════════════════════════════════════════════════════
// Evidence Chain — the user's indexed work
// ════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ActionCategory {
    RustCode            = 0,
    PythonCode          = 1,
    TypeScriptCode      = 2,
    Test                = 3,
    Documentation       = 4,
    Architecture        = 5,
    ConstitutionalDesign = 6,
    SpiritualFoundation = 7,
    InfrastructureOps   = 8,
    Research            = 9,
}

impl ActionCategory {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::RustCode),
            1 => Some(Self::PythonCode),
            2 => Some(Self::TypeScriptCode),
            3 => Some(Self::Test),
            4 => Some(Self::Documentation),
            5 => Some(Self::Architecture),
            6 => Some(Self::ConstitutionalDesign),
            7 => Some(Self::SpiritualFoundation),
            8 => Some(Self::InfrastructureOps),
            9 => Some(Self::Research),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HashedAction {
    pub hash: Blake3Hash,
    pub timestamp_unix: u64,
    pub category: ActionCategory,
    pub size_metric: u64,
}

/// The indexed evidence chain for a POI claim.
///
/// `root_hash` = blake3_domain("bizra-eval-v1-evidence", canonical_bytes(actions))
#[derive(Clone, Debug)]
pub struct EvidenceChain {
    pub actions: Vec<HashedAction>,
    pub root_hash: Blake3Hash,
    pub dedup_removed: u64,
    pub raw_action_count: u64,
}

impl EvidenceChain {
    /// Build from raw actions: sort, dedup, compute root.
    pub fn from_actions(mut actions: Vec<HashedAction>) -> Self {
        let raw_count = actions.len() as u64;

        // Canonical sort: timestamp, then hash bytes
        actions.sort_by(|a, b| {
            a.timestamp_unix.cmp(&b.timestamp_unix)
                .then_with(|| a.hash.cmp(&b.hash))
        });

        let before = actions.len();
        actions.dedup_by(|a, b| a.hash == b.hash);
        let dedup_removed = (before - actions.len()) as u64;

        let canonical = Self::to_canonical_bytes(&actions);
        let root_hash = blake3_domain("bizra-eval-v1-evidence", &canonical);

        EvidenceChain { actions, root_hash, dedup_removed, raw_action_count: raw_count }
    }

    fn to_canonical_bytes(actions: &[HashedAction]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(actions.len() * 49); // 32+8+1+8
        for a in actions {
            buf.extend_from_slice(&a.hash);
            buf.extend_from_slice(&a.timestamp_unix.to_le_bytes());
            buf.push(a.category as u8);
            buf.extend_from_slice(&a.size_metric.to_le_bytes());
        }
        buf
    }

    pub fn action_count(&self) -> u64 { self.actions.len() as u64 }

    pub fn span_months(&self) -> f64 {
        if self.actions.len() < 2 { return 0.0; }
        let first = self.actions.first().unwrap().timestamp_unix;
        let last = self.actions.last().unwrap().timestamp_unix;
        (last - first) as f64 / (30.44 * 24.0 * 3600.0)
    }

    pub fn category_diversity(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for a in &self.actions { seen.insert(a.category); }
        seen.len()
    }

    pub fn total_size(&self) -> u64 {
        self.actions.iter().map(|a| a.size_metric).sum()
    }

    pub fn count_by_category(&self, cat: ActionCategory) -> u64 {
        self.actions.iter().filter(|a| a.category == cat).map(|a| a.size_metric).sum()
    }
}

// ════════════════════════════════════════════════════════════
// Comparable Set — algorithmically derived
// ════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct ComparableEntry {
    pub project_id: String,
    pub loc: u64,
    pub test_density: f64,    // tests per 1K LOC
    pub commit_cadence: f64,  // commits per month
    pub age_months: u32,
    pub contributor_count: u32,
}

#[derive(Clone, Debug)]
pub struct ComparableFilter {
    pub min_loc: u64,
    pub min_age_months: u32,
    pub license_families: Vec<String>,
    pub exclude_self: bool,
}

#[derive(Clone, Debug)]
pub struct ComparableSet {
    pub entries: Vec<ComparableEntry>,
    pub set_hash: Blake3Hash,
    pub filter: ComparableFilter,
}

impl ComparableSet {
    pub fn from_entries(mut entries: Vec<ComparableEntry>, filter: ComparableFilter) -> Self {
        entries.sort_by(|a, b| a.project_id.cmp(&b.project_id));
        let canonical = Self::to_canonical_bytes(&entries, &filter);
        let set_hash = blake3_domain("bizra-eval-v1-comparables", &canonical);
        ComparableSet { entries, set_hash, filter }
    }

    fn to_canonical_bytes(entries: &[ComparableEntry], filter: &ComparableFilter) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&filter.min_loc.to_le_bytes());
        buf.extend_from_slice(&filter.min_age_months.to_le_bytes());
        for lic in &filter.license_families {
            buf.extend_from_slice(lic.as_bytes());
            buf.push(0x00);
        }
        buf.push(filter.exclude_self as u8);
        for e in entries {
            buf.extend_from_slice(e.project_id.as_bytes());
            buf.push(0x00);
            buf.extend_from_slice(&e.loc.to_le_bytes());
            buf.extend_from_slice(&e.test_density.to_le_bytes());
            buf.extend_from_slice(&e.commit_cadence.to_le_bytes());
            buf.extend_from_slice(&e.age_months.to_le_bytes());
            buf.extend_from_slice(&e.contributor_count.to_le_bytes());
        }
        buf
    }

    fn sorted_field<F: Fn(&ComparableEntry) -> f64>(&self, f: F) -> Vec<f64> {
        let mut vals: Vec<f64> = self.entries.iter().map(f).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals
    }

    pub fn median_loc(&self) -> u64 {
        if self.entries.is_empty() { return 1; }
        let mut locs: Vec<u64> = self.entries.iter().map(|e| e.loc).collect();
        locs.sort();
        locs[locs.len() / 2].max(1)
    }

    pub fn median_test_density(&self) -> f64 {
        let vals = self.sorted_field(|e| e.test_density);
        if vals.is_empty() { return 1.0; }
        vals[vals.len() / 2].max(0.01)
    }

    pub fn median_commit_cadence(&self) -> f64 {
        let vals = self.sorted_field(|e| e.commit_cadence);
        if vals.is_empty() { return 1.0; }
        vals[vals.len() / 2].max(0.01)
    }

    pub fn median_contributors(&self) -> f64 {
        if self.entries.is_empty() { return 1.0; }
        let mut counts: Vec<u32> = self.entries.iter().map(|e| e.contributor_count).collect();
        counts.sort();
        (counts[counts.len() / 2] as f64).max(1.0)
    }
}

// ════════════════════════════════════════════════════════════
// Valuation Config — the function definition
// ════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct ValuationConfig {
    pub ihsan_floor: f64,      // 0.95 per commit 0115016b

    pub weight_loc: f64,
    pub weight_test_density: f64,
    pub weight_commit_cadence: f64,
    pub weight_diversity: f64,
    pub weight_duration: f64,
    pub weight_solo: f64,

    pub seed_per_impact_unit: u64,
}

/// Distribution ratio is NOT a field. It is a constant.
/// 50/50 per البذرة. Hardcoded. Not negotiable.
const DISTRIBUTION_URP_RATIO: f64 = 0.50;

impl ValuationConfig {
    /// The canonical v1 config. Any change produces a different function_hash.
    pub fn canonical_v1() -> Self {
        ValuationConfig {
            ihsan_floor: 0.95,
            weight_loc: 0.20,
            weight_test_density: 0.20,
            weight_commit_cadence: 0.15,
            weight_diversity: 0.15,
            weight_duration: 0.15,
            weight_solo: 0.15,
            seed_per_impact_unit: 1000,
        }
    }

    pub fn function_hash(&self) -> Blake3Hash {
        blake3_domain("bizra-eval-v1-function", &self.canonical_bytes())
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.ihsan_floor.to_le_bytes());
        buf.extend_from_slice(&self.weight_loc.to_le_bytes());
        buf.extend_from_slice(&self.weight_test_density.to_le_bytes());
        buf.extend_from_slice(&self.weight_commit_cadence.to_le_bytes());
        buf.extend_from_slice(&self.weight_diversity.to_le_bytes());
        buf.extend_from_slice(&self.weight_duration.to_le_bytes());
        buf.extend_from_slice(&self.weight_solo.to_le_bytes());
        buf.extend_from_slice(&self.seed_per_impact_unit.to_le_bytes());
        // Distribution ratio is constant, but included in hash for completeness
        buf.extend_from_slice(&DISTRIBUTION_URP_RATIO.to_le_bytes());
        buf
    }
}

// ════════════════════════════════════════════════════════════
// The Evaluation — pure function, deterministic (E1)
// ════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct ValuationResult {
    pub valuation_seed: u64,
    pub ihsan_score: f64,
    pub factors: BTreeMap<String, f64>,
    pub raw_impact: f64,
    pub distribution_urp: u64,
    pub distribution_user: u64,
    pub reproducibility_hash: Blake3Hash,
}

/// Pure function. Same inputs → same output. No RNG, no clock, no external state.
/// Applies to every user equally. Founder is user-zero, not special case.
pub fn evaluate(
    config: &ValuationConfig,
    evidence: &EvidenceChain,
    comparables: &ComparableSet,
) -> Result<ValuationResult, EvalError> {
    if evidence.actions.is_empty() { return Err(EvalError::EmptyEvidenceChain); }
    if comparables.entries.is_empty() { return Err(EvalError::ComparableSetEmpty); }

    let mut factors = BTreeMap::new();

    // Factor 1: LOC normalized against comparable median (capped 3x)
    let f_loc = (evidence.total_size() as f64 / comparables.median_loc() as f64).min(3.0);
    factors.insert("loc_normalized".into(), f_loc);

    // Factor 2: Test density normalized
    let test_size = evidence.count_by_category(ActionCategory::Test);
    let total_size = evidence.total_size().max(1);
    let test_density = (test_size as f64 / total_size as f64) * 1000.0;
    let f_test = (test_density / comparables.median_test_density()).min(3.0);
    factors.insert("test_density".into(), f_test);

    // Factor 3: Commit cadence normalized
    let span = evidence.span_months().max(1.0);
    let cadence = evidence.action_count() as f64 / span;
    let f_cadence = (cadence / comparables.median_commit_cadence()).min(3.0);
    factors.insert("commit_cadence".into(), f_cadence);

    // Factor 4: Category diversity (out of 10)
    let f_diversity = evidence.category_diversity() as f64 / 10.0;
    factors.insert("category_diversity".into(), f_diversity);

    // Factor 5: Sustained duration (normalized to 36 months, capped 1.5)
    let f_duration = (span / 36.0).min(1.5);
    factors.insert("sustained_duration".into(), f_duration);

    // Factor 6: Solo-builder multiplier = log2(median_team_size), capped 4.0
    // Universal rule: any solo user gets this against team-built comparables
    let f_solo = comparables.median_contributors().log2().min(4.0);
    factors.insert("solo_builder_multiplier".into(), f_solo);

    // Raw impact (weighted sum)
    let raw_impact = f_loc * config.weight_loc
        + f_test * config.weight_test_density
        + f_cadence * config.weight_commit_cadence
        + f_diversity * config.weight_diversity
        + f_duration * config.weight_duration
        + f_solo * config.weight_solo;

    // Ihsan score — computed from evidence quality, not vibes
    let code_count = [ActionCategory::RustCode, ActionCategory::PythonCode,
                      ActionCategory::TypeScriptCode]
        .iter()
        .map(|c| evidence.actions.iter().filter(|a| a.category == *c).count() as f64)
        .sum::<f64>()
        .max(1.0);

    let doc_count = [ActionCategory::Documentation, ActionCategory::Architecture,
                     ActionCategory::ConstitutionalDesign, ActionCategory::SpiritualFoundation]
        .iter()
        .map(|c| evidence.actions.iter().filter(|a| a.category == *c).count() as f64)
        .sum::<f64>();

    let test_count = evidence.actions.iter()
        .filter(|a| a.category == ActionCategory::Test).count() as f64;

    let test_ratio = (test_count / code_count).min(1.0);
    let doc_ratio = (doc_count / code_count).min(1.0);
    let diversity_norm = evidence.category_diversity() as f64 / 10.0;

    let ihsan_score = 0.40 * test_ratio + 0.30 * doc_ratio + 0.30 * diversity_norm;

    // Ihsan gate — frozen anchor
    if ihsan_score < config.ihsan_floor {
        return Err(EvalError::IhsanBelowFloor {
            score: ihsan_score,
            floor: config.ihsan_floor,
        });
    }

    // Convert to SEED
    let valuation_seed = (raw_impact * config.seed_per_impact_unit as f64) as u64;

    // Distribution — 50/50 per البذرة, hardcoded constant
    let distribution_urp = (valuation_seed as f64 * DISTRIBUTION_URP_RATIO) as u64;
    let distribution_user = valuation_seed - distribution_urp;

    // Reproducibility hash
    let mut repro_buf = Vec::with_capacity(80);
    repro_buf.extend_from_slice(&config.function_hash());
    repro_buf.extend_from_slice(&evidence.root_hash);
    repro_buf.extend_from_slice(&comparables.set_hash);
    repro_buf.extend_from_slice(&valuation_seed.to_le_bytes());
    let reproducibility_hash = blake3_domain("bizra-eval-v1-repro", &repro_buf);

    Ok(ValuationResult {
        valuation_seed, ihsan_score, factors, raw_impact,
        distribution_urp, distribution_user, reproducibility_hash,
    })
}

/// Independent verification: re-run evaluate, check hash match.
pub fn verify_valuation(
    config: &ValuationConfig,
    evidence: &EvidenceChain,
    comparables: &ComparableSet,
    claimed: &GenesisValuationReceipt,
) -> Result<bool, EvalError> {
    let result = evaluate(config, evidence, comparables)?;
    Ok(result.reproducibility_hash == claimed.reproducibility_hash
        && result.valuation_seed == claimed.valuation_seed
        && result.distribution_urp == claimed.distribution_urp
        && result.distribution_user == claimed.distribution_user)
}

// ════════════════════════════════════════════════════════════
// Genesis Valuation Receipt — two-layer model integration
// ════════════════════════════════════════════════════════════

/// NOTE: Add to receipts.rs ReceiptKind enum:
///   GenesisValuation = 0x60,
/// And in from_byte():
///   0x60 => Some(Self::GenesisValuation),

#[derive(Debug, Clone)]
pub struct GenesisValuationReceipt {
    pub user_id: String,
    pub evidence_chain_hash: Blake3Hash,
    pub comparable_set_hash: Blake3Hash,
    pub function_hash: Blake3Hash,
    pub valuation_seed: u64,
    pub ihsan_score: f64,
    pub distribution_urp: u64,
    pub distribution_user: u64,
    pub reproducibility_hash: Blake3Hash,
    pub factor_count: u32,
    pub factor_keys: Vec<String>,
    pub factor_values: Vec<f64>,
}

impl GenesisValuationReceipt {
    pub fn from_result(
        user_id: String,
        evidence: &EvidenceChain,
        comparables: &ComparableSet,
        config: &ValuationConfig,
        result: &ValuationResult,
    ) -> Self {
        let factor_keys: Vec<String> = result.factors.keys().cloned().collect();
        let factor_values: Vec<f64> = result.factors.values().copied().collect();
        GenesisValuationReceipt {
            user_id,
            evidence_chain_hash: evidence.root_hash,
            comparable_set_hash: comparables.set_hash,
            function_hash: config.function_hash(),
            valuation_seed: result.valuation_seed,
            ihsan_score: result.ihsan_score,
            distribution_urp: result.distribution_urp,
            distribution_user: result.distribution_user,
            reproducibility_hash: result.reproducibility_hash,
            factor_count: factor_keys.len() as u32,
            factor_keys,
            factor_values,
        }
    }
}

impl ReceiptPayload for GenesisValuationReceipt {
    fn kind(&self) -> ReceiptKind {
        // Uses the new variant added to ReceiptKind
        // ReceiptKind::GenesisValuation  (0x60)
        // For now, using GovernanceDecision as placeholder until
        // the enum variant is added to receipts.rs
        ReceiptKind::GovernanceDecision // TODO: change to GenesisValuation
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // user_id: length-prefixed
        buf.extend_from_slice(&(self.user_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.user_id.as_bytes());
        // fixed hashes (32 bytes each)
        buf.extend_from_slice(&self.evidence_chain_hash);
        buf.extend_from_slice(&self.comparable_set_hash);
        buf.extend_from_slice(&self.function_hash);
        // numerics
        buf.extend_from_slice(&self.valuation_seed.to_le_bytes());
        buf.extend_from_slice(&self.ihsan_score.to_le_bytes());
        buf.extend_from_slice(&self.distribution_urp.to_le_bytes());
        buf.extend_from_slice(&self.distribution_user.to_le_bytes());
        buf.extend_from_slice(&self.reproducibility_hash);
        // factors (already sorted via BTreeMap origin)
        buf.extend_from_slice(&self.factor_count.to_le_bytes());
        for (k, v) in self.factor_keys.iter().zip(self.factor_values.iter()) {
            buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-eval-v1-receipt", &self.canonical_bytes())
    }
}

impl ReceiptPayloadDecode for GenesisValuationReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);

        // user_id
        let user_id_bytes = r.read_length_prefixed()?;
        let user_id = std::str::from_utf8(user_id_bytes)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();

        // hashes
        let evidence_chain_hash = r.read_hash()?;
        let comparable_set_hash = r.read_hash()?;
        let function_hash = r.read_hash()?;

        // numerics
        let valuation_seed = r.read_u64()?;
        let ihsan_score = r.read_f64()?;
        let distribution_urp = r.read_u64()?;
        let distribution_user = r.read_u64()?;
        let reproducibility_hash = r.read_hash()?;

        // factors
        let factor_count = r.read_u32()?;
        let mut factor_keys = Vec::with_capacity(factor_count as usize);
        let mut factor_values = Vec::with_capacity(factor_count as usize);
        for _ in 0..factor_count {
            let key_bytes = r.read_length_prefixed()?;
            let key = std::str::from_utf8(key_bytes)
                .map_err(|e| DecodeError::Utf8(e.to_string()))?
                .to_string();
            let val = r.read_f64()?;
            factor_keys.push(key);
            factor_values.push(val);
        }

        Ok(GenesisValuationReceipt {
            user_id, evidence_chain_hash, comparable_set_hash,
            function_hash, valuation_seed, ihsan_score,
            distribution_urp, distribution_user, reproducibility_hash,
            factor_count, factor_keys, factor_values,
        })
    }
}

// ════════════════════════════════════════════════════════════
// ValuationFactory — plugs into configure_cognition.rs
// ════════════════════════════════════════════════════════════

use crate::cognition::thought_graph::{GraphNode, AgentCtx, Thought};

/// A GraphNode that runs the valuation when traversed.
/// Used by the configure layer to wire eval_v1 into the standard
/// boot → build → reason → receipt pipeline.
pub struct ValuationNode {
    config: ValuationConfig,
    evidence: EvidenceChain,
    comparables: ComparableSet,
}

impl GraphNode for ValuationNode {
    fn traverse(&self, ctx: &mut AgentCtx) -> Vec<Thought> {
        // The valuation node doesn't produce Thoughts in the normal sense.
        // It runs evaluate() and emits a receipt via the runtime loop.
        // This traverse() is a passthrough that signals "valuation available."
        //
        // The actual mint event is handled by the runtime's event loop
        // via CognitionEvent::ValuationRequest, not via graph traversal.
        // This node exists so the valuation engine can be declared as an
        // EdgeDeclaration in the configure layer and participate in
        // the boot digest / canonical sort / provenance chain.
        vec![]
    }
}

/// Factory for producing ValuationNode instances.
/// Carries the config; evidence and comparables are provided at runtime.
pub struct ValuationFactory {
    pub config: ValuationConfig,
}

impl GraphNodeFactory for ValuationFactory {
    fn build(&self) -> Result<Box<dyn GraphNode>, ConfigureError> {
        // At boot, the factory produces a node with the config embedded.
        // Evidence and comparables are bound later when the actual
        // evaluation is triggered via the runtime event loop.
        //
        // This matches the factory pattern: declaration (config) is
        // separate from instantiation (evidence + comparables arrive
        // at runtime, not at boot).
        Ok(Box::new(ValuationNodeStub {
            config_hash: self.config.function_hash(),
        }))
    }

    fn factory_kind(&self) -> &'static str {
        "bizra-eval-v1"
    }
}

/// Stub node produced at boot. Real evaluation happens via runtime event.
struct ValuationNodeStub {
    config_hash: Blake3Hash,
}

impl GraphNode for ValuationNodeStub {
    fn traverse(&self, _ctx: &mut AgentCtx) -> Vec<Thought> {
        // Signals that eval_v1 is available in this node's boot config.
        // Actual computation deferred to runtime event.
        vec![]
    }
}

// ════════════════════════════════════════════════════════════
// Integration: ReceiptKind patch (apply to receipts.rs)
// ════════════════════════════════════════════════════════════

/// To integrate, add to receipts.rs ReceiptKind:
///
/// ```rust
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// #[repr(u8)]
/// pub enum ReceiptKind {
///     Genesis            = 0x00,
///     CognitionBoot      = 0x10,
///     Myelination        = 0x20,
///     Demyelination      = 0x21,
///     ReasoningSession   = 0x30,
///     GovernanceDecision = 0x40,
///     NodeLifecycle      = 0x50,
///     GenesisValuation   = 0x60,  // ← NEW
///     DegradedPath       = 0xF0,
/// }
/// ```
///
/// And add `0x60 => Some(Self::GenesisValuation)` to from_byte().

// ════════════════════════════════════════════════════════════
// Integration: CognitionEvent patch (apply to runtime.rs)
// ════════════════════════════════════════════════════════════

/// To integrate, add to runtime.rs CognitionEvent:
///
/// ```rust
/// pub enum CognitionEvent {
///     ReasoningRequest { request_id: Blake3Hash },
///     ConsolidationTick,
///     GovernanceDemyelination { edge: Blake3Hash, decision: Blake3Hash },
///     ValuationRequest {          // ← NEW
///         user_id: String,
///         evidence: EvidenceChain,
///         comparables: ComparableSet,
///     },
///     Shutdown,
/// }
/// ```
