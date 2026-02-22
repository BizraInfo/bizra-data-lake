// bizra-agent/src/reflex_cache.rs
// ============================================================
// GENESIS Reflex Cache — System-1 compiled rule store
// ============================================================

use std::collections::{HashMap, VecDeque};

use blake3::Hasher;

use crate::hash_namespace::TriggerHash;

/// All-zeros policy hash identifying bootstrap (pre-compiled) rules.
/// Bootstrap rules are loaded on cold start and replaced once the
/// reflex compiler produces higher-SNR compiled rules.
pub const BOOTSTRAP_POLICY_HASH: [u8; 32] = [0u8; 32];

/// Domain prefix used when hashing bootstrap trigger pattern names.
const BOOTSTRAP_TRIGGER_DOMAIN: &str = "genesis/bootstrap/v1";

/// Compute a deterministic `TriggerHash` for a bootstrap pattern name
/// (e.g. `"bootstrap:greeting"`).  Uses BLAKE3 with domain separation
/// identical in structure to `hash_namespace::domain_hash`.
fn bootstrap_trigger_hash(pattern_name: &str) -> TriggerHash {
    let mut hasher = Hasher::new();
    hasher.update(BOOTSTRAP_TRIGGER_DOMAIN.as_bytes());
    hasher.update(b":");
    hasher.update(pattern_name.as_bytes());
    TriggerHash(*hasher.finalize().as_bytes())
}

/// Returns `true` when `rule` was loaded as a bootstrap reflex
/// (policy_hash == all-zeros).
pub fn is_bootstrap_rule(rule: &ReflexRule) -> bool {
    rule.policy_hash == BOOTSTRAP_POLICY_HASH
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflexMode {
    Disabled,
    Shadow,
    Active,
}

impl ReflexMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ReflexMode::Disabled => "disabled",
            ReflexMode::Shadow => "shadow",
            ReflexMode::Active => "active",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "disabled" => Some(Self::Disabled),
            "shadow" => Some(Self::Shadow),
            "active" => Some(Self::Active),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineReason {
    GuardianVeto,
    RevalidationFailed,
    PolicyHashMismatch,
    ManualInvalidation,
    MissingPolicyHash,
}

impl QuarantineReason {
    pub fn as_str(self) -> &'static str {
        match self {
            QuarantineReason::GuardianVeto => "guardian_veto",
            QuarantineReason::RevalidationFailed => "revalidation_failed",
            QuarantineReason::PolicyHashMismatch => "policy_hash_mismatch",
            QuarantineReason::ManualInvalidation => "manual_invalidation",
            QuarantineReason::MissingPolicyHash => "missing_policy_hash",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "guardian_veto" => Some(Self::GuardianVeto),
            "revalidation_failed" => Some(Self::RevalidationFailed),
            "policy_hash_mismatch" => Some(Self::PolicyHashMismatch),
            "manual_invalidation" => Some(Self::ManualInvalidation),
            "missing_policy_hash" => Some(Self::MissingPolicyHash),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionTemplate {
    pub route_signature: String,
    pub primary_agent: String,
}

#[derive(Debug, Clone)]
pub struct ReflexRule {
    pub trigger_hash: TriggerHash,
    pub action_template: ActionTemplate,
    pub compile_ihsan: f32,
    pub compile_snr: f32,
    pub compiled_at: u64,
    pub use_count: u64,
    pub last_used_at: u64,
    pub last_validated_at: u64,
    pub quarantined: bool,
    pub quarantine_reason: Option<QuarantineReason>,
    pub policy_hash: [u8; 32],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReflexStats {
    pub hits: u64,
    pub misses: u64,
    pub compiled: u64,
    pub quarantined: u64,
    pub invalidated: u64,
    pub revalidations: u64,
    pub revalidation_failures: u64,
    pub size: usize,
}

pub struct ReflexCache {
    by_trigger: HashMap<TriggerHash, ReflexRule>,
    lru: VecDeque<TriggerHash>,
    max_entries: usize,
    stats: ReflexStats,
}

impl ReflexCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            by_trigger: HashMap::new(),
            lru: VecDeque::new(),
            max_entries: max_entries.max(1),
            stats: ReflexStats::default(),
        }
    }

    /// Seed the cache with universal bootstrap reflexes that cover the
    /// most common cold-start message patterns (~40% of traffic).
    ///
    /// Bootstrap rules are only loaded when the cache is completely empty
    /// -- they never overwrite compiled rules.  Each rule carries:
    ///   - `policy_hash = BOOTSTRAP_POLICY_HASH` (all zeros)
    ///   - `compiled_at = 0`, `last_validated_at = 0`
    ///   - `compile_ihsan = 0.95`, `compile_snr = 0.90`
    ///
    /// Returns the number of bootstrap rules inserted (0 when the cache
    /// already contains rules).
    pub fn load_bootstrap_rules(&mut self) -> usize {
        if !self.by_trigger.is_empty() {
            return 0;
        }

        let bootstrap_defs: &[(&str, &str, &str, f32)] = &[
            // (pattern_name, primary_agent, route_signature, confidence)
            (
                "bootstrap:greeting",
                "Diplomat",
                "GreetUser>GenerateResponse",
                0.95,
            ),
            (
                "bootstrap:help",
                "Scholar",
                "RetrieveContext>GenerateResponse",
                0.90,
            ),
            (
                "bootstrap:remember",
                "Oracle",
                "RecallMemory>GenerateResponse",
                0.92,
            ),
            (
                "bootstrap:profile_recall",
                "Oracle",
                "ProfileRecall>GenerateResponse",
                0.95,
            ),
        ];

        let mut loaded: usize = 0;

        for &(pattern_name, agent, route, confidence) in bootstrap_defs {
            let trigger = bootstrap_trigger_hash(pattern_name);
            let rule = ReflexRule {
                trigger_hash: trigger,
                action_template: ActionTemplate {
                    route_signature: route.to_string(),
                    primary_agent: agent.to_string(),
                },
                compile_ihsan: confidence,
                compile_snr: 0.90,
                compiled_at: 0,
                use_count: 0,
                last_used_at: 0,
                last_validated_at: 0,
                quarantined: false,
                quarantine_reason: None,
                policy_hash: BOOTSTRAP_POLICY_HASH,
            };

            self.by_trigger.insert(trigger, rule);
            self.lru.push_back(trigger);
            loaded += 1;
        }

        self.stats.size = self.by_trigger.len();
        // Track bootstrap rules in `compiled` so stats reflect cache
        // population, but the caller can distinguish them via
        // `is_bootstrap_rule()`.
        self.stats.compiled += loaded as u64;
        loaded
    }

    pub fn get_active(
        &mut self,
        mode: ReflexMode,
        trigger: &TriggerHash,
        current_policy_hash: Option<[u8; 32]>,
        now: u64,
    ) -> Option<ReflexRule> {
        if mode != ReflexMode::Active {
            self.stats.misses += 1;
            return None;
        }

        let Some(policy_hash) = current_policy_hash else {
            self.stats.misses += 1;
            if self.by_trigger.contains_key(trigger) {
                let _ = self.quarantine(*trigger, QuarantineReason::MissingPolicyHash);
            }
            return None;
        };

        let trigger_val = *trigger;
        let result = {
            let Some(rule) = self.by_trigger.get_mut(&trigger_val) else {
                self.stats.misses += 1;
                return None;
            };

            if rule.quarantined {
                self.stats.misses += 1;
                return None;
            }

            if rule.policy_hash != policy_hash {
                rule.quarantined = true;
                rule.quarantine_reason = Some(QuarantineReason::PolicyHashMismatch);
                self.stats.quarantined += 1;
                self.stats.misses += 1;
                return None;
            }

            rule.use_count += 1;
            rule.last_used_at = now;
            rule.clone()
        };
        self.touch(trigger_val);
        self.stats.hits += 1;
        Some(result)
    }

    pub fn insert_compiled(&mut self, mode: ReflexMode, rule: ReflexRule) {
        if mode == ReflexMode::Disabled {
            return;
        }

        let key = rule.trigger_hash;
        self.by_trigger.insert(key, rule);
        self.touch(key);
        self.stats.compiled += 1;
        self.evict_if_needed();
        self.stats.size = self.by_trigger.len();
    }

    pub fn needs_revalidation(
        &self,
        trigger: &TriggerHash,
        now: u64,
        revalidate_after_seconds: u64,
        revalidate_after_uses: u64,
    ) -> bool {
        let Some(rule) = self.by_trigger.get(trigger) else {
            return false;
        };
        if rule.quarantined {
            return false;
        }
        let by_time = now.saturating_sub(rule.last_validated_at) >= revalidate_after_seconds;
        let by_use = rule.use_count >= revalidate_after_uses;
        by_time || by_use
    }

    pub fn mark_revalidated(&mut self, trigger: &TriggerHash, now: u64, passed: bool) {
        self.stats.revalidations += 1;
        if let Some(rule) = self.by_trigger.get_mut(trigger) {
            if passed {
                rule.last_validated_at = now;
                rule.use_count = 0;
                return;
            }
            self.stats.revalidation_failures += 1;
            rule.quarantined = true;
            rule.quarantine_reason = Some(QuarantineReason::RevalidationFailed);
            self.stats.quarantined += 1;
        }
    }

    pub fn quarantine(&mut self, trigger: TriggerHash, reason: QuarantineReason) -> bool {
        if let Some(rule) = self.by_trigger.get_mut(&trigger) {
            if !rule.quarantined {
                self.stats.quarantined += 1;
            }
            rule.quarantined = true;
            rule.quarantine_reason = Some(reason);
            return true;
        }
        false
    }

    pub fn invalidate(&mut self, trigger: TriggerHash) -> bool {
        let existed = self.by_trigger.remove(&trigger).is_some();
        if existed {
            self.stats.invalidated += 1;
            self.lru.retain(|k| *k != trigger);
            self.stats.size = self.by_trigger.len();
        }
        existed
    }

    pub fn stats(&self) -> ReflexStats {
        let mut out = self.stats;
        out.size = self.by_trigger.len();
        out
    }

    pub fn all_rules(&self) -> Vec<ReflexRule> {
        self.by_trigger.values().cloned().collect()
    }

    pub fn replace_rules(&mut self, rules: Vec<ReflexRule>) {
        self.by_trigger.clear();
        self.lru.clear();
        for rule in rules {
            let key = rule.trigger_hash;
            self.by_trigger.insert(key, rule);
            self.lru.push_back(key);
        }
        self.evict_if_needed();
        self.stats.size = self.by_trigger.len();
    }

    fn touch(&mut self, key: TriggerHash) {
        self.lru.retain(|k| *k != key);
        self.lru.push_back(key);
    }

    fn evict_if_needed(&mut self) {
        while self.by_trigger.len() > self.max_entries {
            if let Some(oldest) = self.lru.pop_front() {
                self.by_trigger.remove(&oldest);
                self.stats.invalidated += 1;
            } else {
                break;
            }
        }
    }
}

impl Default for ReflexCache {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(trigger: TriggerHash, policy_hash: [u8; 32]) -> ReflexRule {
        ReflexRule {
            trigger_hash: trigger,
            action_template: ActionTemplate {
                route_signature: "RetrieveContext>GenerateResponse".to_string(),
                primary_agent: "Scholar".to_string(),
            },
            compile_ihsan: 0.96,
            compile_snr: 0.93,
            compiled_at: 100,
            use_count: 0,
            last_used_at: 0,
            last_validated_at: 100,
            quarantined: false,
            quarantine_reason: None,
            policy_hash,
        }
    }

    #[test]
    fn mode_disabled_never_routes() {
        let mut cache = ReflexCache::new(16);
        let t = TriggerHash([1u8; 32]);
        cache.insert_compiled(ReflexMode::Shadow, rule(t, [7u8; 32]));
        assert!(cache
            .get_active(ReflexMode::Disabled, &t, Some([7u8; 32]), 200)
            .is_none());
    }

    #[test]
    fn policy_mismatch_quarantines() {
        let mut cache = ReflexCache::new(16);
        let t = TriggerHash([1u8; 32]);
        cache.insert_compiled(ReflexMode::Active, rule(t, [7u8; 32]));
        assert!(cache
            .get_active(ReflexMode::Active, &t, Some([8u8; 32]), 200)
            .is_none());
        let rules = cache.all_rules();
        assert!(rules[0].quarantined);
    }

    #[test]
    fn revalidation_dual_trigger() {
        let mut cache = ReflexCache::new(16);
        let t = TriggerHash([1u8; 32]);
        let mut r = rule(t, [7u8; 32]);
        r.use_count = 210;
        r.last_validated_at = 10;
        cache.insert_compiled(ReflexMode::Active, r);
        assert!(cache.needs_revalidation(&t, 20, 604800, 200));
    }

    // ── Bootstrap reflex tests ──────────────────────────────────

    #[test]
    fn bootstrap_rules_load_when_empty() {
        let mut cache = ReflexCache::new(64);
        assert!(cache.by_trigger.is_empty());

        let loaded = cache.load_bootstrap_rules();
        assert_eq!(loaded, 4, "should load exactly 4 bootstrap reflexes");
        assert_eq!(cache.by_trigger.len(), 4);
        assert_eq!(cache.lru.len(), 4);
        assert_eq!(cache.stats().compiled, 4);
        assert_eq!(cache.stats().size, 4);

        // Verify each rule has bootstrap markers
        for rule in cache.all_rules() {
            assert_eq!(rule.policy_hash, BOOTSTRAP_POLICY_HASH);
            assert_eq!(rule.compiled_at, 0);
            assert_eq!(rule.last_validated_at, 0);
            assert!(rule.compile_ihsan >= 0.90, "bootstrap ihsan must be >= 0.90");
            assert!((rule.compile_snr - 0.90).abs() < f32::EPSILON);
            assert!(!rule.quarantined);
            assert!(is_bootstrap_rule(&rule));
        }

        // Verify expected agents are present
        let mut agents: Vec<String> = cache
            .all_rules()
            .iter()
            .map(|r| r.action_template.primary_agent.clone())
            .collect();
        agents.sort();
        assert_eq!(agents, vec!["Diplomat", "Oracle", "Oracle", "Scholar"]);

        // Verify expected route signatures are present
        let mut routes: Vec<String> = cache
            .all_rules()
            .iter()
            .map(|r| r.action_template.route_signature.clone())
            .collect();
        routes.sort();
        assert_eq!(
            routes,
            vec![
                "GreetUser>GenerateResponse",
                "ProfileRecall>GenerateResponse",
                "RecallMemory>GenerateResponse",
                "RetrieveContext>GenerateResponse",
            ]
        );
    }

    #[test]
    fn bootstrap_rules_skip_when_populated() {
        let mut cache = ReflexCache::new(64);

        // Insert a compiled rule first
        let t = TriggerHash([42u8; 32]);
        cache.insert_compiled(ReflexMode::Active, rule(t, [7u8; 32]));
        assert_eq!(cache.by_trigger.len(), 1);
        let compiled_before = cache.stats().compiled;

        // Attempt to load bootstrap rules -- should be a no-op
        let loaded = cache.load_bootstrap_rules();
        assert_eq!(loaded, 0, "bootstrap must not overwrite existing rules");
        assert_eq!(cache.by_trigger.len(), 1);
        assert_eq!(cache.stats().compiled, compiled_before);

        // The existing rule should be untouched
        let existing = cache.by_trigger.get(&t).unwrap();
        assert_eq!(existing.policy_hash, [7u8; 32]);
        assert!(!is_bootstrap_rule(existing));
    }

    #[test]
    fn bootstrap_rule_identification() {
        // A rule with BOOTSTRAP_POLICY_HASH is a bootstrap rule
        let bootstrap = ReflexRule {
            trigger_hash: TriggerHash([1u8; 32]),
            action_template: ActionTemplate {
                route_signature: "GreetUser>GenerateResponse".to_string(),
                primary_agent: "Diplomat".to_string(),
            },
            compile_ihsan: 0.95,
            compile_snr: 0.90,
            compiled_at: 0,
            use_count: 0,
            last_used_at: 0,
            last_validated_at: 0,
            quarantined: false,
            quarantine_reason: None,
            policy_hash: BOOTSTRAP_POLICY_HASH,
        };
        assert!(is_bootstrap_rule(&bootstrap));

        // A rule with a non-zero policy hash is NOT a bootstrap rule
        let compiled = rule(TriggerHash([2u8; 32]), [7u8; 32]);
        assert!(!is_bootstrap_rule(&compiled));

        // Edge case: a rule with almost-zero hash (only last byte differs)
        let mut nearly_zero = BOOTSTRAP_POLICY_HASH;
        nearly_zero[31] = 1;
        let edge = rule(TriggerHash([3u8; 32]), nearly_zero);
        assert!(!is_bootstrap_rule(&edge));
    }

    #[test]
    fn bootstrap_trigger_hashes_are_deterministic() {
        let h1 = bootstrap_trigger_hash("bootstrap:greeting");
        let h2 = bootstrap_trigger_hash("bootstrap:greeting");
        assert_eq!(h1, h2, "same pattern name must produce identical hash");

        let h3 = bootstrap_trigger_hash("bootstrap:help");
        assert_ne!(h1, h3, "different patterns must produce different hashes");
    }

    #[test]
    fn bootstrap_rules_are_retrievable_with_bootstrap_policy() {
        let mut cache = ReflexCache::new(64);
        cache.load_bootstrap_rules();

        // Bootstrap rules should be retrievable when the caller
        // presents BOOTSTRAP_POLICY_HASH as the current policy.
        let greeting_trigger = bootstrap_trigger_hash("bootstrap:greeting");
        let result = cache.get_active(
            ReflexMode::Active,
            &greeting_trigger,
            Some(BOOTSTRAP_POLICY_HASH),
            100,
        );
        assert!(
            result.is_some(),
            "bootstrap rule should match on bootstrap policy"
        );
        let matched = result.unwrap();
        assert_eq!(matched.action_template.primary_agent, "Diplomat");
        assert_eq!(
            matched.action_template.route_signature,
            "GreetUser>GenerateResponse"
        );
    }
}
