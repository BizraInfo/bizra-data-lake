// bizra-agent/src/skills/skill_tree.rs
// ============================================================
// Skill Tree — Self-configuring capability hierarchy
// ============================================================
//
// Every agent has a skill tree. Skills branch, unlock, and
// self-configure based on:
//   1. Agent role (Navigator → routing skills, Scholar → search)
//   2. Node resources (GPU → inference skills unlock)
//   3. Prerequisites (classify before organize, read before fill)
//   4. Mastery via reflex compiler (repeated success → promotion)
//
// The two default root trees every node ships with:
//   Root: FileSystem → Classify → Organize → Rename → Merge → Dedup
//   Root: Browser    → Navigate → Read → Fill → Click → Execute
//
// Constitutional invariant: no skill activates without its
// prerequisite chain being satisfied. No destructive skill
// activates without SAT approval in the tree config.
//
// This is the organism's learning architecture — the same
// pattern that makes S2→S1 reflex compilation work, but
// applied to capability acquisition.
// ============================================================

use std::collections::HashMap;

/// Unique identifier for a skill in the tree.
pub type SkillId = &'static str;

/// Mastery level — earned through repeated successful execution.
/// Maps to reflex compiler: 3+ successes at current level → promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mastery {
    /// Skill exists in tree but not yet attempted.
    Locked,
    /// Prerequisites met, skill available but unproven.
    Novice,
    /// 3+ successful executions. Can execute without GoT.
    Competent,
    /// 10+ successes, compiled into reflex. System-1 fast path.
    Expert,
    /// 50+ successes, can teach sub-agents. Eligible for delegation.
    Master,
}

impl Mastery {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Locked => "locked",
            Self::Novice => "novice",
            Self::Competent => "competent",
            Self::Expert => "expert",
            Self::Master => "master",
        }
    }

    /// Threshold for promotion to next level.
    pub fn promotion_threshold(&self) -> u32 {
        match self {
            Self::Locked => 0,        // unlock via prerequisites
            Self::Novice => 3,        // 3 successes → Competent
            Self::Competent => 10,    // 10 successes → Expert (reflex compiled)
            Self::Expert => 50,       // 50 successes → Master (can delegate)
            Self::Master => u32::MAX, // terminal
        }
    }

    /// Can this mastery level execute the skill?
    pub fn can_execute(&self) -> bool {
        matches!(
            self,
            Self::Novice | Self::Competent | Self::Expert | Self::Master
        )
    }

    /// Can this mastery level delegate to sub-agents?
    pub fn can_delegate(&self) -> bool {
        matches!(self, Self::Master)
    }

    /// Does this level use System-1 fast path (reflex compiled)?
    pub fn is_reflexive(&self) -> bool {
        matches!(self, Self::Expert | Self::Master)
    }

    /// Try to promote based on success count.
    pub fn try_promote(&self, successes: u32) -> Option<Mastery> {
        if successes >= self.promotion_threshold() {
            match self {
                Self::Locked => Some(Self::Novice),
                Self::Novice => Some(Self::Competent),
                Self::Competent => Some(Self::Expert),
                Self::Expert => Some(Self::Master),
                Self::Master => None,
            }
        } else {
            None
        }
    }
}

/// A single node in the skill tree.
#[derive(Debug, Clone)]
pub struct SkillNode {
    pub id: SkillId,
    pub name: &'static str,
    /// Parent skill — None for root skills.
    pub parent: Option<SkillId>,
    /// Skills that must be Competent+ before this unlocks.
    pub prerequisites: Vec<SkillId>,
    /// Current mastery level.
    pub mastery: Mastery,
    /// Successful execution count (drives promotion).
    pub successes: u32,
    /// Failed execution count.
    pub failures: u32,
    /// Does this skill require SAT approval before execution?
    pub sat_required: bool,
    /// Does this skill require HITL (human-in-the-loop)?
    pub hitl_required: bool,
    /// Which agent roles have natural affinity for this skill?
    pub affinity_roles: Vec<&'static str>,
    /// Children skills that branch from this one.
    pub children: Vec<SkillId>,
}

impl SkillNode {
    /// Record a successful execution. Returns new mastery if promoted.
    pub fn record_success(&mut self) -> Option<Mastery> {
        self.successes += 1;
        if let Some(new_level) = self.mastery.try_promote(self.successes) {
            self.mastery = new_level;
            Some(new_level)
        } else {
            None
        }
    }

    /// Record a failed execution.
    pub fn record_failure(&mut self) {
        self.failures += 1;
    }

    /// Success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f32 {
        let total = self.successes + self.failures;
        if total == 0 {
            0.0
        } else {
            self.successes as f32 / total as f32
        }
    }
}

/// The complete skill tree for an agent or node.
/// Self-configuring: checks prerequisites, promotes on success,
/// unlocks branches as mastery grows.
#[derive(Debug, Clone, Default)]
pub struct SkillTree {
    pub nodes: HashMap<SkillId, SkillNode>,
    pub roots: Vec<SkillId>,
}

impl SkillTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a skill node into the tree.
    fn insert(&mut self, node: SkillNode) {
        let id = node.id;
        if node.parent.is_none() {
            self.roots.push(id);
        }
        self.nodes.insert(id, node);
    }

    /// Check if a skill's prerequisites are satisfied (all at Competent+).
    pub fn prerequisites_met(&self, skill_id: SkillId) -> bool {
        let node = match self.nodes.get(skill_id) {
            Some(n) => n,
            None => return false,
        };
        node.prerequisites.iter().all(|prereq| {
            self.nodes
                .get(prereq)
                .map(|n| n.mastery >= Mastery::Competent)
                .unwrap_or(false)
        })
    }

    /// Auto-unlock skills whose prerequisites are now met.
    /// Called after every successful execution to cascade unlocks.
    pub fn cascade_unlocks(&mut self) {
        let ids: Vec<SkillId> = self.nodes.keys().copied().collect();
        for id in ids {
            let should_unlock = {
                let node = &self.nodes[id];
                node.mastery == Mastery::Locked && self.prerequisites_met_internal(node)
            };
            if should_unlock {
                if let Some(node) = self.nodes.get_mut(id) {
                    node.mastery = Mastery::Novice;
                }
            }
        }
    }

    fn prerequisites_met_internal(&self, node: &SkillNode) -> bool {
        node.prerequisites.iter().all(|prereq| {
            self.nodes
                .get(prereq)
                .map(|n| n.mastery >= Mastery::Competent)
                .unwrap_or(false)
        })
    }

    /// Record a successful execution and cascade unlocks.
    pub fn record_success(&mut self, skill_id: SkillId) -> Option<Mastery> {
        let promotion = self.nodes.get_mut(skill_id)?.record_success();
        self.cascade_unlocks();
        promotion
    }

    /// Record a failure.
    pub fn record_failure(&mut self, skill_id: SkillId) {
        if let Some(node) = self.nodes.get_mut(skill_id) {
            node.record_failure();
        }
    }

    /// Get all skills at or above a mastery level.
    pub fn skills_at_level(&self, min_level: Mastery) -> Vec<SkillId> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.mastery >= min_level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all executable skills (Novice+).
    pub fn executable_skills(&self) -> Vec<SkillId> {
        self.skills_at_level(Mastery::Novice)
    }

    /// Get all reflex-compiled skills (Expert+).
    pub fn reflexive_skills(&self) -> Vec<SkillId> {
        self.skills_at_level(Mastery::Expert)
    }

    /// Summary for node status display.
    pub fn summary(&self) -> String {
        let total = self.nodes.len();
        let unlocked = self
            .nodes
            .values()
            .filter(|n| n.mastery.can_execute())
            .count();
        let reflexive = self
            .nodes
            .values()
            .filter(|n| n.mastery.is_reflexive())
            .count();
        let mastered = self
            .nodes
            .values()
            .filter(|n| n.mastery == Mastery::Master)
            .count();
        format!(
            "Skills: {}/{} unlocked, {} reflexive, {} mastered",
            unlocked, total, reflexive, mastered
        )
    }
}

// ── Default skill trees (constitutional defaults) ─────────

/// Helper to create a skill node concisely.
#[allow(clippy::too_many_arguments)]
fn skill(
    id: SkillId,
    name: &'static str,
    parent: Option<SkillId>,
    prereqs: Vec<SkillId>,
    sat: bool,
    hitl: bool,
    roles: Vec<&'static str>,
    children: Vec<SkillId>,
    initial_mastery: Mastery,
) -> SkillNode {
    SkillNode {
        id,
        name,
        parent,
        prerequisites: prereqs,
        mastery: initial_mastery,
        successes: 0,
        failures: 0,
        sat_required: sat,
        hitl_required: hitl,
        affinity_roles: roles,
        children,
    }
}

/// Build the filesystem skill tree (default capability #1).
///
/// Tree structure:
/// ```text
/// fs_root (Filesystem)
///  ├── fs_classify (Smart Classification)    → Novice from boot
///  │    ├── fs_organize (Auto Organization)  → unlocks at Competent
///  │    ├── fs_dedup (Duplicate Detection)   → unlocks at Competent
///  │    └── fs_snr_score (SNR Scoring)       → unlocks at Competent
///  ├── fs_rename (Batch Renaming)            → Novice from boot
///  │    └── fs_sanitize (Name Sanitization)  → unlocks at Competent
///  ├── fs_merge (File Merging)               → requires classify+organize
///  ├── fs_delete (Safe Deletion)             → requires classify, SAT+HITL
///  └── fs_archive (Archive Management)       → requires classify
/// ```
pub fn filesystem_skill_tree() -> SkillTree {
    let mut tree = SkillTree::new();

    // Root
    tree.insert(skill(
        "fs_root",
        "Filesystem management",
        None,
        vec![],
        false,
        false,
        vec!["Navigator", "Artisan", "Scholar", "Sentinel"],
        vec![
            "fs_classify",
            "fs_rename",
            "fs_merge",
            "fs_delete",
            "fs_archive",
        ],
        Mastery::Novice,
    ));

    // Branch 1: Classification → Organization → Dedup → SNR
    tree.insert(skill(
        "fs_classify",
        "Smart classification",
        Some("fs_root"),
        vec![],
        false,
        false,
        vec!["Navigator", "Scholar"],
        vec!["fs_organize", "fs_dedup", "fs_snr_score"],
        Mastery::Novice, // available from boot
    ));
    tree.insert(skill(
        "fs_organize",
        "Auto organization",
        Some("fs_classify"),
        vec!["fs_classify"],
        true,
        false, // SAT validates manifest
        vec!["Artisan", "Navigator"],
        vec![],
        Mastery::Locked, // unlocks when classify reaches Competent
    ));
    tree.insert(skill(
        "fs_dedup",
        "Duplicate detection",
        Some("fs_classify"),
        vec!["fs_classify"],
        false,
        false,
        vec!["Scholar", "Sentinel"],
        vec![],
        Mastery::Locked,
    ));
    tree.insert(skill(
        "fs_snr_score",
        "SNR scoring",
        Some("fs_classify"),
        vec!["fs_classify"],
        false,
        false,
        vec!["Scholar"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 2: Rename → Sanitize
    tree.insert(skill(
        "fs_rename",
        "Batch renaming",
        Some("fs_root"),
        vec![],
        false,
        false,
        vec!["Artisan"],
        vec!["fs_sanitize"],
        Mastery::Novice,
    ));
    tree.insert(skill(
        "fs_sanitize",
        "Name sanitization",
        Some("fs_rename"),
        vec!["fs_rename"],
        false,
        false,
        vec!["Artisan"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 3: Merge (requires classify + organize both Competent)
    tree.insert(skill(
        "fs_merge",
        "File merging",
        Some("fs_root"),
        vec!["fs_classify", "fs_organize"],
        false,
        false,
        vec!["Artisan", "Scholar"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 4: Delete (constitutional gate — SAT + HITL required)
    tree.insert(skill(
        "fs_delete",
        "Safe deletion",
        Some("fs_root"),
        vec!["fs_classify"],
        true,
        true, // SAT validates + HITL confirms
        vec!["Sentinel"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 5: Archive management
    tree.insert(skill(
        "fs_archive",
        "Archive management",
        Some("fs_root"),
        vec!["fs_classify"],
        false,
        false,
        vec!["Scholar", "Artisan"],
        vec![],
        Mastery::Locked,
    ));

    tree
}

/// Build the browser MCP skill tree (default capability #2).
///
/// Tree structure:
/// ```text
/// br_root (Browser management)
///  ├── br_navigate (URL Navigation)           → Novice from boot
///  │    ├── br_history (History traversal)     → unlocks at Competent
///  │    └── br_tabs (Tab management)           → unlocks at Competent
///  ├── br_read (Page reading)                  → Novice from boot
///  │    ├── br_extract (Text extraction)       → unlocks at Competent
///  │    ├── br_find (Element finding)          → unlocks at Competent
///  │    └── br_screenshot (Visual capture)     → unlocks at Competent
///  ├── br_fill (Form interaction)              → requires read, SAT
///  │    └── br_click (Element clicking)        → requires fill, SAT
///  └── br_execute_js (JavaScript execution)    → requires all, SAT+HITL
/// ```
pub fn browser_skill_tree() -> SkillTree {
    let mut tree = SkillTree::new();

    // Root
    tree.insert(skill(
        "br_root",
        "Browser management",
        None,
        vec![],
        false,
        false,
        vec!["Navigator", "Scholar", "Artisan", "Sentinel"],
        vec!["br_navigate", "br_read", "br_fill", "br_execute_js"],
        Mastery::Novice,
    ));

    // Branch 1: Navigate → History → Tabs
    tree.insert(skill(
        "br_navigate",
        "URL navigation",
        Some("br_root"),
        vec![],
        true,
        false, // SAT validates URLs
        vec!["Navigator"],
        vec!["br_history", "br_tabs"],
        Mastery::Novice,
    ));

    tree.insert(skill(
        "br_history",
        "History traversal",
        Some("br_navigate"),
        vec!["br_navigate"],
        false,
        false,
        vec!["Navigator"],
        vec![],
        Mastery::Locked,
    ));
    tree.insert(skill(
        "br_tabs",
        "Tab management",
        Some("br_navigate"),
        vec!["br_navigate"],
        false,
        false,
        vec!["Navigator"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 2: Read → Extract → Find → Screenshot
    tree.insert(skill(
        "br_read",
        "Page reading",
        Some("br_root"),
        vec![],
        false,
        false,
        vec!["Scholar"],
        vec!["br_extract", "br_find", "br_screenshot"],
        Mastery::Novice,
    ));
    tree.insert(skill(
        "br_extract",
        "Text extraction",
        Some("br_read"),
        vec!["br_read"],
        false,
        false,
        vec!["Scholar"],
        vec![],
        Mastery::Locked,
    ));
    tree.insert(skill(
        "br_find",
        "Element finding",
        Some("br_read"),
        vec!["br_read"],
        false,
        false,
        vec!["Scholar", "Artisan"],
        vec![],
        Mastery::Locked,
    ));
    tree.insert(skill(
        "br_screenshot",
        "Visual capture",
        Some("br_read"),
        vec!["br_read"],
        false,
        false,
        vec!["Sentinel"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 3: Fill → Click (SAT required — interacts with external sites)
    tree.insert(skill(
        "br_fill",
        "Form interaction",
        Some("br_root"),
        vec!["br_read"],
        true,
        false, // SAT validates field targets
        vec!["Artisan"],
        vec!["br_click"],
        Mastery::Locked,
    ));
    tree.insert(skill(
        "br_click",
        "Element clicking",
        Some("br_fill"),
        vec!["br_fill"],
        true,
        false, // SAT validates click targets
        vec!["Artisan"],
        vec![],
        Mastery::Locked,
    ));

    // Branch 4: Execute JS (highest gate — SAT + HITL)
    tree.insert(skill(
        "br_execute_js",
        "JavaScript execution",
        Some("br_root"),
        vec!["br_read", "br_fill", "br_click"],
        true,
        true, // full gate
        vec!["Scholar", "Sentinel"],
        vec![],
        Mastery::Locked,
    ));

    tree
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mastery_promotion_thresholds() {
        assert_eq!(Mastery::Novice.promotion_threshold(), 3);
        assert_eq!(Mastery::Competent.promotion_threshold(), 10);
        assert_eq!(Mastery::Expert.promotion_threshold(), 50);
    }

    #[test]
    fn mastery_promotion_chain() {
        let mut level = Mastery::Novice;
        // 3 successes → Competent
        assert_eq!(level.try_promote(3), Some(Mastery::Competent));
        level = Mastery::Competent;
        // 10 successes → Expert (reflex)
        assert_eq!(level.try_promote(10), Some(Mastery::Expert));
        level = Mastery::Expert;
        // 50 successes → Master (delegate)
        assert_eq!(level.try_promote(50), Some(Mastery::Master));
        level = Mastery::Master;
        // Master is terminal
        assert_eq!(level.try_promote(1000), None);
    }

    #[test]
    fn locked_cannot_execute() {
        assert!(!Mastery::Locked.can_execute());
        assert!(Mastery::Novice.can_execute());
        assert!(Mastery::Expert.can_execute());
    }

    #[test]
    fn only_master_can_delegate() {
        assert!(!Mastery::Expert.can_delegate());
        assert!(Mastery::Master.can_delegate());
    }

    #[test]
    fn expert_and_master_are_reflexive() {
        assert!(!Mastery::Competent.is_reflexive());
        assert!(Mastery::Expert.is_reflexive());
        assert!(Mastery::Master.is_reflexive());
    }

    #[test]
    fn skill_node_records_success_and_promotes() {
        let mut node = skill(
            "test",
            "Test skill",
            None,
            vec![],
            false,
            false,
            vec![],
            vec![],
            Mastery::Novice,
        );
        assert_eq!(node.mastery, Mastery::Novice);

        // 2 successes — not enough
        node.record_success();
        node.record_success();
        assert_eq!(node.mastery, Mastery::Novice);

        // 3rd success → promoted to Competent
        let promotion = node.record_success();
        assert_eq!(promotion, Some(Mastery::Competent));
        assert_eq!(node.mastery, Mastery::Competent);
        assert_eq!(node.successes, 3);
    }

    #[test]
    fn skill_node_success_rate() {
        let mut node = skill(
            "test",
            "Test",
            None,
            vec![],
            false,
            false,
            vec![],
            vec![],
            Mastery::Novice,
        );
        node.successes = 8;
        node.failures = 2;
        assert!((node.success_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn filesystem_tree_has_correct_structure() {
        let tree = filesystem_skill_tree();
        assert_eq!(tree.roots.len(), 1);
        assert_eq!(tree.roots[0], "fs_root");
        assert_eq!(tree.nodes.len(), 10); // root + 9 skills

        // Root and base skills start unlocked (Novice)
        assert_eq!(tree.nodes["fs_root"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_classify"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_rename"].mastery, Mastery::Novice);

        // Dependent skills start Locked
        assert_eq!(tree.nodes["fs_organize"].mastery, Mastery::Locked);
        assert_eq!(tree.nodes["fs_dedup"].mastery, Mastery::Locked);
        assert_eq!(tree.nodes["fs_merge"].mastery, Mastery::Locked);
        assert_eq!(tree.nodes["fs_delete"].mastery, Mastery::Locked);
    }

    #[test]
    fn filesystem_delete_requires_sat_and_hitl() {
        let tree = filesystem_skill_tree();
        let delete = &tree.nodes["fs_delete"];
        assert!(delete.sat_required);
        assert!(delete.hitl_required);
    }

    #[test]
    fn browser_tree_has_correct_structure() {
        let tree = browser_skill_tree();
        assert_eq!(tree.roots.len(), 1);
        assert_eq!(tree.roots[0], "br_root");
        assert_eq!(tree.nodes.len(), 11); // root + 10 skills

        // Root and base skills start unlocked
        assert_eq!(tree.nodes["br_root"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["br_navigate"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["br_read"].mastery, Mastery::Novice);

        // Dependent skills start Locked
        assert_eq!(tree.nodes["br_fill"].mastery, Mastery::Locked);
        assert_eq!(tree.nodes["br_click"].mastery, Mastery::Locked);
        assert_eq!(tree.nodes["br_execute_js"].mastery, Mastery::Locked);
    }

    #[test]
    fn browser_js_execution_requires_sat_and_hitl() {
        let tree = browser_skill_tree();
        let js = &tree.nodes["br_execute_js"];
        assert!(js.sat_required);
        assert!(js.hitl_required);
        // Requires 3 prerequisites
        assert_eq!(js.prerequisites.len(), 3);
    }

    #[test]
    fn cascade_unlock_after_classify_mastery() {
        let mut tree = filesystem_skill_tree();

        // fs_organize is locked — needs fs_classify at Competent
        assert_eq!(tree.nodes["fs_organize"].mastery, Mastery::Locked);
        assert!(!tree.prerequisites_met("fs_organize"));

        // Simulate 3 successes on fs_classify → promotes to Competent
        tree.record_success("fs_classify");
        tree.record_success("fs_classify");
        let promotion = tree.record_success("fs_classify");
        assert_eq!(promotion, Some(Mastery::Competent));

        // Now fs_organize should have cascaded to Novice
        assert_eq!(tree.nodes["fs_organize"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_dedup"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_snr_score"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_delete"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["fs_archive"].mastery, Mastery::Novice);
    }

    #[test]
    fn merge_requires_both_classify_and_organize() {
        let mut tree = filesystem_skill_tree();

        // Promote classify to Competent
        for _ in 0..3 {
            tree.record_success("fs_classify");
        }
        // fs_organize is now Novice, but merge needs organize at Competent too
        assert_eq!(tree.nodes["fs_merge"].mastery, Mastery::Locked);

        // Promote organize to Competent
        for _ in 0..3 {
            tree.record_success("fs_organize");
        }
        // NOW merge should unlock
        assert_eq!(tree.nodes["fs_merge"].mastery, Mastery::Novice);
    }

    #[test]
    fn reflexive_skills_empty_at_boot() {
        let tree = filesystem_skill_tree();
        assert!(tree.reflexive_skills().is_empty());
    }

    #[test]
    fn skill_becomes_reflexive_at_expert() {
        let mut tree = filesystem_skill_tree();
        // Promote classify: 3→Competent, then 10→Expert
        for _ in 0..10 {
            tree.record_success("fs_classify");
        }
        assert_eq!(tree.nodes["fs_classify"].mastery, Mastery::Expert);
        assert!(tree.reflexive_skills().contains(&"fs_classify"));
    }

    #[test]
    fn summary_reflects_state() {
        let tree = filesystem_skill_tree();
        let s = tree.summary();
        // 3 skills start Novice (fs_root, fs_classify, fs_rename)
        assert!(s.contains("3/10 unlocked"));
        assert!(s.contains("0 reflexive"));
        assert!(s.contains("0 mastered"));
    }

    #[test]
    fn browser_cascade_read_to_extract() {
        let mut tree = browser_skill_tree();
        assert_eq!(tree.nodes["br_extract"].mastery, Mastery::Locked);

        // Promote br_read to Competent
        for _ in 0..3 {
            tree.record_success("br_read");
        }

        // br_extract, br_find, br_screenshot should all unlock
        assert_eq!(tree.nodes["br_extract"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["br_find"].mastery, Mastery::Novice);
        assert_eq!(tree.nodes["br_screenshot"].mastery, Mastery::Novice);
    }

    #[test]
    fn browser_fill_unlocks_after_read_competent() {
        let mut tree = browser_skill_tree();
        // br_fill needs br_read at Competent
        for _ in 0..3 {
            tree.record_success("br_read");
        }
        assert_eq!(tree.nodes["br_fill"].mastery, Mastery::Novice);
    }

    #[test]
    fn browser_js_stays_locked_until_all_prereqs() {
        let mut tree = browser_skill_tree();
        // Promote read → Competent (unlocks fill)
        for _ in 0..3 {
            tree.record_success("br_read");
        }
        // Promote fill → Competent (unlocks click)
        for _ in 0..3 {
            tree.record_success("br_fill");
        }
        // JS still locked — needs click at Competent too
        assert_eq!(tree.nodes["br_execute_js"].mastery, Mastery::Locked);

        // Promote click → Competent
        for _ in 0..3 {
            tree.record_success("br_click");
        }
        // NOW JS unlocks
        assert_eq!(tree.nodes["br_execute_js"].mastery, Mastery::Novice);
    }
}
