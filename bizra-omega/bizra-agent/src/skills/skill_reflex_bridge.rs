// bizra-agent/src/skills/skill_reflex_bridge.rs
// ============================================================
// Neural-Symbolic Bridge: SkillTree ↔ OmniKernel ReflexCache
// ============================================================
//
// The OmniKernel routes through CyclePaths:
//   ReflexHit (S1, 50ms) → EngramHit → FullInference (S2, 1800ms)
//
// The SkillTree has mastery levels:
//   Expert (reflexive, S1) → Competent → Novice (full S2)
//
// These are the SAME routing decision. This bridge connects them:
//   - When a skill reaches Expert → register as ReflexRule
//   - When OmniKernel hits a reflex → verify SkillTree mastery
//   - When a skill is quarantined → quarantine the reflex rule
//
// Standing on Giants:
//   Kahneman (2011): System 1/2 → fast/slow thinking
//   Hebb (1949): neurons that fire together wire together
//   BIZRA: skills that succeed together compile together
// ============================================================

use blake3::Hasher;

use crate::hash_namespace::TriggerHash;
use crate::reflex_cache::{ActionTemplate, ReflexCache, ReflexMode, ReflexRule};
use super::skill_tree::{Mastery, SkillId, SkillTree};

/// Domain prefix for skill-derived trigger hashes.
const SKILL_TRIGGER_DOMAIN: &str = "bizra/skill/v1";

/// Policy hash identifying skill-tree-derived reflexes.
/// Distinct from BOOTSTRAP_POLICY_HASH (all-zeros) so that
/// skill reflexes can be revalidated independently.
pub const SKILL_POLICY_HASH: [u8; 32] = {
    let mut h = [0u8; 32];
    // "skill" in ASCII as first 5 bytes — deterministic, non-zero
    h[0] = b's'; h[1] = b'k'; h[2] = b'i'; h[3] = b'l'; h[4] = b'l';
    h
};

/// Compute a deterministic TriggerHash for a skill ID.
/// Uses BLAKE3 with domain separation so skill triggers never
/// collide with bootstrap or user-compiled triggers.
pub fn skill_trigger_hash(skill_id: SkillId) -> TriggerHash {
    let mut hasher = Hasher::new();
    hasher.update(SKILL_TRIGGER_DOMAIN.as_bytes());
    hasher.update(b":");
    hasher.update(skill_id.as_bytes());
    TriggerHash(*hasher.finalize().as_bytes())
}

/// Create a ReflexRule from a mastered skill node.
/// Called when SkillTree promotes a node to Expert (10+ successes).
fn skill_to_reflex_rule(skill_id: SkillId, tree: &SkillTree, now: u64) -> Option<ReflexRule> {
    let node = tree.nodes.get(skill_id)?;

    // Only Expert+ skills become reflexes
    if !node.mastery.is_reflexive() {
        return None;
    }

    let trigger = skill_trigger_hash(skill_id);
    let primary_agent = node.affinity_roles.first().copied().unwrap_or("Navigator");

    Some(ReflexRule {
        trigger_hash: trigger,
        action_template: ActionTemplate {
            route_signature: format!("skill:{}", skill_id),
            primary_agent: primary_agent.to_string(),
        },
        compile_ihsan: node.success_rate(),
        compile_snr: node.success_rate(), // SNR mirrors success rate for skills
        compiled_at: now,
        use_count: 0,
        last_used_at: now,
        last_validated_at: now,
        quarantined: false,
        quarantine_reason: None,
        policy_hash: SKILL_POLICY_HASH,
    })
}

/// Promote a newly-Expert skill into the OmniKernel's ReflexCache.
/// Returns true if a reflex rule was inserted.
///
/// Call this after SkillTree::record_success() returns Some(Mastery::Expert).
/// The skill's execution pattern becomes a System-1 fast path in the OmniKernel.
pub fn promote_skill_to_reflex(
    skill_id: SkillId,
    tree: &SkillTree,
    cache: &mut ReflexCache,
    now: u64,
) -> bool {
    if let Some(rule) = skill_to_reflex_rule(skill_id, tree, now) {
        cache.insert_compiled(ReflexMode::Active, rule);
        true
    } else {
        false
    }
}

/// Verify that a skill-based reflex hit is still backed by SkillTree mastery.
/// Returns true if the skill exists AND is at Expert+ level.
///
/// Call this when OmniKernel matches a reflex with route_signature "skill:*".
/// If false, the reflex should be quarantined (skill may have been degraded).
pub fn verify_skill_mastery(skill_id: SkillId, tree: &SkillTree) -> bool {
    tree.nodes.get(skill_id)
        .map(|n| n.mastery.is_reflexive())
        .unwrap_or(false)
}

/// Bulk-sync all Expert+ skills from the SkillTree into the ReflexCache.
/// Returns the count of reflexes registered.
///
/// Call this on node boot to populate the cache from persisted skill state.
pub fn sync_skill_tree_to_cache(
    tree: &SkillTree,
    cache: &mut ReflexCache,
    now: u64,
) -> usize {
    let mut count = 0;
    for skill_id in tree.reflexive_skills() {
        if promote_skill_to_reflex(skill_id, tree, cache, now) {
            count += 1;
        }
    }
    count
}

/// Quarantine a skill-derived reflex when the skill fails or is degraded.
/// Returns true if a reflex was found and quarantined.
pub fn quarantine_skill_reflex(
    skill_id: SkillId,
    cache: &mut ReflexCache,
) -> bool {
    let trigger = skill_trigger_hash(skill_id);
    cache.quarantine(trigger, crate::reflex_cache::QuarantineReason::OwnerDrift)
}

/// Check if a route_signature corresponds to a skill-derived reflex.
/// Route signatures from skills have the format "skill:{skill_id}".
pub fn is_skill_route(route_signature: &str) -> Option<&str> {
    route_signature.strip_prefix("skill:")
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::skill_tree::filesystem_skill_tree;

    #[test]
    fn skill_trigger_hash_deterministic() {
        let h1 = skill_trigger_hash("fs_classify");
        let h2 = skill_trigger_hash("fs_classify");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_skills_different_hashes() {
        let h1 = skill_trigger_hash("fs_classify");
        let h2 = skill_trigger_hash("fs_organize");
        assert_ne!(h1, h2);
    }

    #[test]
    fn skill_policy_hash_not_bootstrap() {
        assert_ne!(SKILL_POLICY_HASH, crate::reflex_cache::BOOTSTRAP_POLICY_HASH);
    }

    #[test]
    fn novice_skill_does_not_become_reflex() {
        let tree = filesystem_skill_tree();
        // fs_classify starts at Novice
        assert_eq!(tree.nodes["fs_classify"].mastery, Mastery::Novice);
        let rule = skill_to_reflex_rule("fs_classify", &tree, 1000);
        assert!(rule.is_none());
    }

    #[test]
    fn expert_skill_becomes_reflex() {
        let mut tree = filesystem_skill_tree();
        // Promote fs_classify: 3→Competent, 10→Expert
        for _ in 0..10 { tree.record_success("fs_classify"); }
        assert_eq!(tree.nodes["fs_classify"].mastery, Mastery::Expert);

        let rule = skill_to_reflex_rule("fs_classify", &tree, 1000).unwrap();
        assert_eq!(rule.action_template.route_signature, "skill:fs_classify");
        assert_eq!(rule.action_template.primary_agent, "Navigator");
        assert_eq!(rule.policy_hash, SKILL_POLICY_HASH);
        assert!(rule.compile_ihsan > 0.0);
    }

    #[test]
    fn promote_registers_in_cache() {
        let mut tree = filesystem_skill_tree();
        let mut cache = ReflexCache::new(100);

        // Before: no skill reflex
        let trigger = skill_trigger_hash("fs_classify");
        assert!(cache.lookup_readonly(ReflexMode::Active, &trigger, None).is_none());

        // Promote to Expert
        for _ in 0..10 { tree.record_success("fs_classify"); }
        let inserted = promote_skill_to_reflex("fs_classify", &tree, &mut cache, 1000);
        assert!(inserted);

        // After: reflex exists in cache
        let rule = cache.lookup_readonly(ReflexMode::Active, &trigger, Some(SKILL_POLICY_HASH)).unwrap();
        assert_eq!(rule.action_template.route_signature, "skill:fs_classify");
    }

    #[test]
    fn verify_mastery_checks_tree() {
        let mut tree = filesystem_skill_tree();
        // Novice — not reflexive
        assert!(!verify_skill_mastery("fs_classify", &tree));

        // Promote to Expert
        for _ in 0..10 { tree.record_success("fs_classify"); }
        assert!(verify_skill_mastery("fs_classify", &tree));

        // Nonexistent skill — false
        assert!(!verify_skill_mastery("nonexistent", &tree));
    }

    #[test]
    fn sync_registers_all_expert_skills() {
        let mut tree = filesystem_skill_tree();
        let mut cache = ReflexCache::new(100);

        // Promote classify and rename to Expert
        for _ in 0..10 { tree.record_success("fs_classify"); }
        for _ in 0..10 { tree.record_success("fs_rename"); }

        let count = sync_skill_tree_to_cache(&tree, &mut cache, 1000);
        assert_eq!(count, 2);

        // Both exist in cache
        assert!(cache.lookup_readonly(ReflexMode::Active, &skill_trigger_hash("fs_classify"), Some(SKILL_POLICY_HASH)).is_some());
        assert!(cache.lookup_readonly(ReflexMode::Active, &skill_trigger_hash("fs_rename"), Some(SKILL_POLICY_HASH)).is_some());
    }

    #[test]
    fn quarantine_removes_skill_reflex() {
        let mut tree = filesystem_skill_tree();
        let mut cache = ReflexCache::new(100);

        for _ in 0..10 { tree.record_success("fs_classify"); }
        promote_skill_to_reflex("fs_classify", &tree, &mut cache, 1000);

        // Quarantine it
        let q = quarantine_skill_reflex("fs_classify", &mut cache);
        assert!(q);

        // Rule is quarantined — lookup_readonly(Active) should NOT find it
        let trigger = skill_trigger_hash("fs_classify");
        assert!(cache.lookup_readonly(ReflexMode::Active, &trigger, Some(SKILL_POLICY_HASH)).is_none());
    }

    #[test]
    fn is_skill_route_parses_correctly() {
        assert_eq!(is_skill_route("skill:fs_classify"), Some("fs_classify"));
        assert_eq!(is_skill_route("skill:br_read"), Some("br_read"));
        assert_eq!(is_skill_route("bootstrap:greeting"), None);
        assert_eq!(is_skill_route("other"), None);
    }

    #[test]
    fn full_lifecycle_skill_to_reflex_to_quarantine() {
        let mut tree = filesystem_skill_tree();
        let mut cache = ReflexCache::new(100);

        // Step 1: Novice — no reflex
        assert!(!promote_skill_to_reflex("fs_classify", &tree, &mut cache, 100));

        // Step 2: Competent (3 successes) — still no reflex
        for _ in 0..3 { tree.record_success("fs_classify"); }
        assert!(!promote_skill_to_reflex("fs_classify", &tree, &mut cache, 200));

        // Step 3: Expert (10 successes) — reflex registered!
        for _ in 0..7 { tree.record_success("fs_classify"); }
        assert!(promote_skill_to_reflex("fs_classify", &tree, &mut cache, 300));

        // Step 4: Verify mastery
        assert!(verify_skill_mastery("fs_classify", &tree));

        // Step 5: Quarantine
        assert!(quarantine_skill_reflex("fs_classify", &mut cache));

        // Step 6: Verify quarantine — lookup_readonly(Active) should NOT find quarantined rules
        let trigger = skill_trigger_hash("fs_classify");
        assert!(cache.lookup_readonly(ReflexMode::Active, &trigger, Some(SKILL_POLICY_HASH)).is_none());
    }
}
