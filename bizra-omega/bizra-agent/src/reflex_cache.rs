// bizra-agent/src/reflex_cache.rs
// ============================================================
// GENESIS Reflex Cache — System-1 compiled rule store
// ============================================================

use std::collections::{HashMap, VecDeque};

use crate::hash_namespace::TriggerHash;

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
}
