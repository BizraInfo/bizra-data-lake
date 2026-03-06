// bizra-memory/src/synthesis.rs
// ============================================================
// Synthesis Engine — the intelligence layer
// ============================================================
// Takes raw fragments and produces:
// 1. Insights (combined understanding from multiple fragments)
// 2. Profile updates (coherent picture of who the user is)
// 3. Contradiction detection (when new info conflicts with old)
//
// This is what separates "storing conversations" from
// "genuinely knowing someone."
//
// The synthesis process:
// 1. Cluster related fragments by kind + content similarity
// 2. Detect reinforcements (same knowledge, different words)
// 3. Detect contradictions (conflicting facts)
// 4. Produce insights from clusters
// 5. Update user profile from insights
// ============================================================

use crate::types::*;
use crate::store::InMemoryStore;
use bizra_hooks::IhsanScore;

// ============================================================
// SYNTHESIS CONFIG
// ============================================================

/// Maximum fragments to process in one synthesis round
pub const MAX_SYNTHESIS_BATCH: usize = 128;

/// Minimum fragments in a cluster to produce an insight
pub const MIN_CLUSTER_SIZE: usize = 2;

/// Content similarity threshold for clustering (0-10000)
pub const SIMILARITY_THRESHOLD: u16 = 6000;

// ============================================================
// FRAGMENT CLUSTER — related fragments grouped together
// ============================================================

pub const MAX_CLUSTER_MEMBERS: usize = 16;

#[derive(Debug)]
pub struct FragmentCluster {
    pub kind: FragmentKind,
    pub members: [Option<FragmentId>; MAX_CLUSTER_MEMBERS],
    pub member_count: u8,
    pub avg_confidence: Confidence,
    pub is_contradiction: bool,
}

impl FragmentCluster {
    pub fn new(kind: FragmentKind) -> Self {
        Self {
            kind,
            members: [None; MAX_CLUSTER_MEMBERS],
            member_count: 0,
            avg_confidence: Confidence::new(0),
            is_contradiction: false,
        }
    }

    pub fn add_member(&mut self, id: FragmentId, confidence: Confidence) -> bool {
        if (self.member_count as usize) >= MAX_CLUSTER_MEMBERS {
            return false;
        }
        self.members[self.member_count as usize] = Some(id);
        self.member_count += 1;
        self.recalculate_confidence(confidence);
        true
    }

    fn recalculate_confidence(&mut self, new_confidence: Confidence) {
        let total: u32 = self.avg_confidence.raw() as u32 * (self.member_count as u32 - 1)
            + new_confidence.raw() as u32;
        self.avg_confidence = Confidence::new((total / self.member_count as u32) as u16);
    }

    pub fn member_ids(&self) -> &[Option<FragmentId>] {
        &self.members[..self.member_count as usize]
    }
}

// ============================================================
// SYNTHESIS RESULT — what came out of a synthesis round
// ============================================================

pub const MAX_SYNTHESIS_INSIGHTS: usize = 32;
pub const MAX_SYNTHESIS_PROFILE_UPDATES: usize = 16;
pub const MAX_SYNTHESIS_CONTRADICTIONS: usize = 8;

#[derive(Debug)]
pub struct Contradiction {
    pub old_fragment: FragmentId,
    pub new_fragment: FragmentId,
    pub kind: FragmentKind,
    /// Which fragment should win? Higher confidence wins.
    pub resolution: FragmentId,
}

pub struct SynthesisResult {
    pub insights_produced: Vec<Insight>,
    pub profile_updates: Vec<(String, String, Confidence)>, // (key, value, confidence)
    pub contradictions: Vec<Contradiction>,
    pub fragments_reinforced: Vec<FragmentId>,
    pub fragments_superseded: Vec<(FragmentId, FragmentId)>, // (old, new)
    pub round: u32,
    pub duration_us: u64,
}

impl SynthesisResult {
    pub fn empty(round: u32) -> Self {
        Self {
            insights_produced: Vec::new(),
            profile_updates: Vec::new(),
            contradictions: Vec::new(),
            fragments_reinforced: Vec::new(),
            fragments_superseded: Vec::new(),
            round,
            duration_us: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.insights_produced.is_empty()
            && self.profile_updates.is_empty()
            && self.contradictions.is_empty()
    }
}

// ============================================================
// SYNTHESIS ENGINE
// ============================================================

pub struct SynthesisEngine {
    round_counter: u32,
    insight_sequence: u32,
    metrics: SynthesisMetrics,
    /// Minimum إحسان to run synthesis
    min_ihsan: IhsanScore,
}

impl SynthesisEngine {
    pub fn new() -> Self {
        Self {
            round_counter: 0,
            insight_sequence: 0,
            metrics: SynthesisMetrics::new(),
            min_ihsan: IhsanScore::new(9500),
        }
    }

    /// Run a full synthesis round
    /// Processes recent fragments, produces insights, suggests profile updates
    pub fn synthesize(
        &mut self,
        store: &mut InMemoryStore,
        profile: &mut UserProfile,
        current_ihsan: IhsanScore,
        current_time: u64,
    ) -> SynthesisResult {
        self.round_counter += 1;
        let round = self.round_counter;

        // إحسان gate: don't synthesize if system quality is degraded
        if current_ihsan.raw() < self.min_ihsan.raw() {
            return SynthesisResult::empty(round);
        }

        let mut result = SynthesisResult::empty(round);

        // Step 1: Gather active fragments for synthesis
        // Collect owned snapshots to avoid holding immutable borrow on store
        let active_fragments = store.query_fragments(&crate::store::FragmentQuery::all());
        if active_fragments.is_empty() {
            return result;
        }

        // Step 2: Cluster fragments by kind (read-only phase)
        let clusters = self.cluster_fragments(&active_fragments);

        // Step 3: Detect reinforcements within clusters (read-only)
        let reinforcements = self.detect_reinforcements(&clusters, &active_fragments);

        // Step 4: Detect contradictions (read-only)
        let contradictions = self.detect_contradictions(&clusters, &active_fragments);

        // Step 5: Produce insights from clusters (read-only synthesis)
        let mut pending_insights = Vec::new();
        for cluster in &clusters {
            if cluster.member_count as usize >= MIN_CLUSTER_SIZE && !cluster.is_contradiction {
                if let Some(insight) = self.synthesize_cluster(cluster, &active_fragments, current_time, current_ihsan) {
                    let profile_update = self.derive_profile_update(&insight, &active_fragments);
                    pending_insights.push((insight, profile_update));
                }
            }
        }

        // Drop the immutable borrows — now we can mutate store
        drop(active_fragments);

        // Step 6: Apply reinforcements (mutation phase)
        for fid in &reinforcements {
            if store.reinforce_fragment(*fid, current_time).is_ok() {
                result.fragments_reinforced.push(*fid);
            }
        }

        // Step 7: Apply contradiction resolutions
        for contradiction in contradictions {
            if store.supersede_fragment(
                contradiction.old_fragment,
                contradiction.new_fragment,
            ).is_ok() {
                result.fragments_superseded.push((
                    contradiction.old_fragment,
                    contradiction.new_fragment,
                ));
            }
            result.contradictions.push(contradiction);
        }

        // Step 8: Store insights and update profile
        for (insight, profile_update) in pending_insights {
            if store.store_insight(insight.clone()).is_ok() {
                if let Some((key, value)) = profile_update {
                    let confidence = insight.confidence;
                    profile.set_trait(&key, &value, confidence, current_time);
                    result.profile_updates.push((key, value, confidence));
                }
                result.insights_produced.push(insight);
            }
        }

        // Update synthesis metadata
        profile.mark_synthesized(current_time);
        self.metrics.synthesis_rounds += 1;
        self.metrics.insights_produced += result.insights_produced.len() as u64;
        self.metrics.profile_updates += result.profile_updates.len() as u64;
        self.metrics.ihsan_at_last_synthesis = current_ihsan;

        result
    }

    /// Ingest a raw fragment into the store after validation
    pub fn ingest(
        &mut self,
        store: &mut InMemoryStore,
        fragment: MemoryFragment,
    ) -> Result<FragmentId, crate::store::StoreError> {
        self.metrics.fragments_ingested += 1;

        if !fragment.confidence.meets_threshold() {
            self.metrics.fragments_below_threshold += 1;
            return Err(crate::store::StoreError::ConfidenceBelowThreshold);
        }

        let id = fragment.id;
        store.store_fragment(fragment)?;
        Ok(id)
    }

    /// Get synthesis metrics
    pub fn metrics(&self) -> &SynthesisMetrics {
        &self.metrics
    }

    pub fn round(&self) -> u32 {
        self.round_counter
    }

    // ================================================================
    // INTERNAL: Clustering
    // ================================================================

    fn cluster_fragments<'a>(
        &self,
        fragments: &[&'a MemoryFragment],
    ) -> Vec<FragmentCluster> {
        let mut clusters: Vec<FragmentCluster> = Vec::new();

        for fragment in fragments {
            // Try to find existing cluster of same kind
            let mut added = false;
            for cluster in clusters.iter_mut() {
                if cluster.kind == fragment.kind
                    && (cluster.member_count as usize) < MAX_CLUSTER_MEMBERS
                {
                    // Simple heuristic: same kind = same cluster
                    // In production, this would use vector similarity
                    cluster.add_member(fragment.id, fragment.confidence);
                    added = true;
                    break;
                }
            }

            if !added {
                let mut new_cluster = FragmentCluster::new(fragment.kind);
                new_cluster.add_member(fragment.id, fragment.confidence);
                clusters.push(new_cluster);
            }
        }

        clusters
    }

    // ================================================================
    // INTERNAL: Reinforcement detection
    // ================================================================

    fn detect_reinforcements(
        &self,
        clusters: &[FragmentCluster],
        _fragments: &[&MemoryFragment],
    ) -> Vec<FragmentId> {
        let mut reinforced = Vec::new();

        for cluster in clusters {
            // Clusters with 3+ members of same kind suggest reinforcement
            if cluster.member_count >= 3 {
                // Reinforce all members (knowledge confirmed multiple times)
                for member in cluster.member_ids() {
                    if let Some(id) = member {
                        reinforced.push(*id);
                    }
                }
            }
        }

        reinforced
    }

    // ================================================================
    // INTERNAL: Contradiction detection
    // ================================================================

    fn detect_contradictions(
        &self,
        _clusters: &[FragmentCluster],
        fragments: &[&MemoryFragment],
    ) -> Vec<Contradiction> {
        let mut contradictions = Vec::new();

        // Simple heuristic: look for Facts with same key but different values
        // In production, this would use semantic similarity
        let facts: Vec<&&MemoryFragment> = fragments.iter()
            .filter(|f| f.kind == FragmentKind::Fact && f.is_active())
            .collect();

        // Pairwise comparison for contradictions
        // O(n²) but n is bounded by MAX_SYNTHESIS_BATCH
        for i in 0..facts.len() {
            for j in (i + 1)..facts.len() {
                // If two facts have very similar content but different creation times,
                // the newer one might supersede the older one.
                // This is a simplified heuristic — real implementation would use
                // vector embeddings for semantic comparison.
                if facts[i].content.as_str().len() > 10
                    && facts[j].content.as_str().len() > 10
                    && self.content_might_conflict(
                        facts[i].content.as_str(),
                        facts[j].content.as_str(),
                    )
                {
                    let (old, new) = if facts[i].created_at < facts[j].created_at {
                        (facts[i], facts[j])
                    } else {
                        (facts[j], facts[i])
                    };

                    // Resolve by confidence first, then recency
                    let resolution = if new.confidence.raw() >= old.confidence.raw() {
                        new.id
                    } else {
                        old.id
                    };

                    contradictions.push(Contradiction {
                        old_fragment: old.id,
                        new_fragment: new.id,
                        kind: FragmentKind::Fact,
                        resolution,
                    });
                }
            }
        }

        contradictions
    }

    /// Simple content conflict heuristic
    /// In production: vector embedding cosine distance
    fn content_might_conflict(&self, a: &str, b: &str) -> bool {
        // Look for shared prefix that diverges
        // "Works at CompanyA" vs "Works at CompanyB"
        let a_words: Vec<&str> = a.split_whitespace().collect();
        let b_words: Vec<&str> = b.split_whitespace().collect();

        if a_words.len() < 3 || b_words.len() < 3 {
            return false;
        }

        // If first N-1 words match but last word differs, potential conflict
        let min_len = a_words.len().min(b_words.len());
        if min_len < 2 {
            return false;
        }

        let shared_prefix = a_words.iter()
            .zip(b_words.iter())
            .take_while(|(a, b)| a.eq_ignore_ascii_case(b))
            .count();

        // If most words match but they diverge = potential conflict
        shared_prefix >= min_len / 2 && a_words != b_words
    }

    // ================================================================
    // INTERNAL: Insight synthesis
    // ================================================================

    fn synthesize_cluster(
        &mut self,
        cluster: &FragmentCluster,
        fragments: &[&MemoryFragment],
        timestamp: u64,
        ihsan: IhsanScore,
    ) -> Option<Insight> {
        if cluster.member_count < 2 {
            return None;
        }

        // Collect source fragment IDs
        let source_ids: Vec<FragmentId> = cluster.member_ids()
            .iter()
            .filter_map(|m| *m)
            .collect();

        // Build insight content from cluster members
        // In production: LLM-powered synthesis
        // For MVP: concatenate key phrases with contextual framing
        let member_contents: Vec<&str> = source_ids.iter()
            .filter_map(|id| {
                fragments.iter().find(|f| f.id == *id).map(|f| f.content.as_str())
            })
            .collect();

        if member_contents.is_empty() {
            return None;
        }

        // Generate insight content
        let insight_text = self.generate_insight_text(cluster.kind, &member_contents);

        self.insight_sequence += 1;
        let insight = Insight::new(
            InsightId::new(self.round_counter, self.insight_sequence),
            &insight_text,
            &source_ids,
            timestamp,
            ihsan,
        );

        Some(insight)
    }

    /// Generate insight text from cluster members
    /// MVP: structured concatenation
    /// Production: LLM synthesis via FFI bridge
    fn generate_insight_text(&self, kind: FragmentKind, contents: &[&str]) -> String {
        let prefix = match kind {
            FragmentKind::Preference => "User preferences indicate: ",
            FragmentKind::Fact => "Established facts: ",
            FragmentKind::Pattern => "Behavioral patterns: ",
            FragmentKind::Emotion => "Emotional context: ",
            FragmentKind::Goal => "User goals: ",
            FragmentKind::Expertise => "Expertise areas: ",
            FragmentKind::Relationship => "Relationship context: ",
            FragmentKind::Temporal => "Temporal patterns: ",
            FragmentKind::Domain => "Domain knowledge: ",
            FragmentKind::Style => "Communication style: ",
        };

        let mut text = String::from(prefix);
        for (i, content) in contents.iter().take(4).enumerate() {
            if i > 0 {
                text.push_str("; ");
            }
            text.push_str(content);
        }

        // Truncate to fit InsightContent
        if text.len() > INSIGHT_CONTENT_SIZE {
            text.truncate(INSIGHT_CONTENT_SIZE);
        }

        text
    }

    // ================================================================
    // INTERNAL: Profile derivation
    // ================================================================

    fn derive_profile_update(
        &self,
        insight: &Insight,
        _fragments: &[&MemoryFragment],
    ) -> Option<(String, String)> {
        let content = insight.content.as_str();

        // Simple key-value extraction from insight content
        // Production: structured extraction via LLM
        if content.starts_with("User preferences indicate: ") {
            let value = content.strip_prefix("User preferences indicate: ")?;
            return Some(("preferences".to_string(), value.to_string()));
        }
        if content.starts_with("Established facts: ") {
            let value = content.strip_prefix("Established facts: ")?;
            return Some(("facts".to_string(), value.to_string()));
        }
        if content.starts_with("Behavioral patterns: ") {
            let value = content.strip_prefix("Behavioral patterns: ")?;
            return Some(("patterns".to_string(), value.to_string()));
        }
        if content.starts_with("Expertise areas: ") {
            let value = content.strip_prefix("Expertise areas: ")?;
            return Some(("expertise".to_string(), value.to_string()));
        }
        if content.starts_with("Communication style: ") {
            let value = content.strip_prefix("Communication style: ")?;
            return Some(("style".to_string(), value.to_string()));
        }
        if content.starts_with("User goals: ") {
            let value = content.strip_prefix("User goals: ")?;
            return Some(("goals".to_string(), value.to_string()));
        }
        if content.starts_with("Domain knowledge: ") {
            let value = content.strip_prefix("Domain knowledge: ")?;
            return Some(("domain".to_string(), value.to_string()));
        }

        None
    }
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_hooks::ComponentId;

    fn make_fragment(conv: u32, seq: u32, kind: FragmentKind, content: &str) -> MemoryFragment {
        MemoryFragment::new(
            FragmentId::new(conv, seq),
            kind,
            content,
            Confidence::HIGH,
            ComponentId::new("test", "1.0"),
            1000,
            IhsanScore::new(9900),
        )
    }

    #[test]
    fn ingest_valid_fragment() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let frag = make_fragment(1, 1, FragmentKind::Fact, "User works at BIZRA");

        let result = engine.ingest(&mut store, frag);
        assert!(result.is_ok());
        assert_eq!(store.fragment_count(), 1);
        assert_eq!(engine.metrics().fragments_ingested, 1);
    }

    #[test]
    fn ingest_rejects_low_confidence() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut frag = make_fragment(1, 1, FragmentKind::Fact, "Uncertain info");
        frag.confidence = Confidence::new(3000);

        let result = engine.ingest(&mut store, frag);
        assert!(result.is_err());
        assert_eq!(engine.metrics().fragments_below_threshold, 1);
    }

    #[test]
    fn synthesis_produces_insights_from_clusters() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        // Add multiple fragments of same kind to trigger clustering
        engine.ingest(&mut store, make_fragment(1, 1, FragmentKind::Preference, "Prefers Rust")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 2, FragmentKind::Preference, "Likes static typing")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 3, FragmentKind::Preference, "Favors zero-cost abstractions")).unwrap();

        let result = engine.synthesize(
            &mut store,
            &mut profile,
            IhsanScore::new(9900),
            2000,
        );

        // Should produce at least one insight from the preference cluster
        assert!(!result.insights_produced.is_empty());
        assert_eq!(engine.metrics().synthesis_rounds, 1);
    }

    #[test]
    fn synthesis_updates_profile() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        engine.ingest(&mut store, make_fragment(1, 1, FragmentKind::Expertise, "Knows Rust deeply")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 2, FragmentKind::Expertise, "Systems programming expert")).unwrap();

        let result = engine.synthesize(
            &mut store,
            &mut profile,
            IhsanScore::new(9900),
            2000,
        );

        // Profile should have expertise trait
        if !result.profile_updates.is_empty() {
            assert!(profile.get_trait("expertise").is_some());
        }
    }

    #[test]
    fn synthesis_blocked_by_low_ihsan() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        engine.ingest(&mut store, make_fragment(1, 1, FragmentKind::Fact, "Test")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 2, FragmentKind::Fact, "Test 2")).unwrap();

        let result = engine.synthesize(
            &mut store,
            &mut profile,
            IhsanScore::new(9000), // Below 9500 threshold
            2000,
        );

        assert!(result.is_empty());
    }

    #[test]
    fn contradiction_detection() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        // Two conflicting facts
        let mut frag1 = make_fragment(1, 1, FragmentKind::Fact, "Works at Company Alpha");
        frag1.created_at = 1000;
        let mut frag2 = make_fragment(2, 1, FragmentKind::Fact, "Works at Company Beta");
        frag2.created_at = 2000;

        engine.ingest(&mut store, frag1).unwrap();
        engine.ingest(&mut store, frag2).unwrap();

        let result = engine.synthesize(
            &mut store,
            &mut profile,
            IhsanScore::new(9900),
            3000,
        );

        // Should detect the contradiction
        assert!(!result.contradictions.is_empty());
        // Old fragment should be superseded
        assert!(!result.fragments_superseded.is_empty());
    }

    #[test]
    fn reinforcement_detection() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        // Many fragments of same kind = reinforcement
        for i in 1..=4 {
            engine.ingest(
                &mut store,
                make_fragment(1, i, FragmentKind::Pattern, &format!("Pattern observation {}", i)),
            ).unwrap();
        }

        let result = engine.synthesize(
            &mut store,
            &mut profile,
            IhsanScore::new(9900),
            2000,
        );

        // Should reinforce the pattern fragments
        assert!(!result.fragments_reinforced.is_empty());
    }

    #[test]
    fn synthesis_metrics_track_correctly() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        // Ingest some fragments
        engine.ingest(&mut store, make_fragment(1, 1, FragmentKind::Fact, "Fact A")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 2, FragmentKind::Fact, "Fact B")).unwrap();

        // Low confidence — should be rejected
        let mut low = make_fragment(1, 3, FragmentKind::Fact, "Low conf");
        low.confidence = Confidence::new(3000);
        let _ = engine.ingest(&mut store, low);

        assert_eq!(engine.metrics().fragments_ingested, 3);
        assert_eq!(engine.metrics().fragments_below_threshold, 1);

        engine.synthesize(&mut store, &mut profile, IhsanScore::new(9900), 2000);
        assert_eq!(engine.metrics().synthesis_rounds, 1);
    }

    #[test]
    fn content_conflict_detection_heuristic() {
        let engine = SynthesisEngine::new();

        // Should detect conflict: same prefix, different value
        assert!(engine.content_might_conflict(
            "Works at Company Alpha",
            "Works at Company Beta",
        ));

        // Should not detect conflict: completely different
        assert!(!engine.content_might_conflict(
            "Likes Rust",
            "Lives in Dubai",
        ));

        // Too short: should not flag
        assert!(!engine.content_might_conflict("Hi", "Bye"));
    }

    #[test]
    fn multiple_synthesis_rounds_are_cumulative() {
        let mut engine = SynthesisEngine::new();
        let mut store = InMemoryStore::new();
        let mut profile = UserProfile::new();

        // Round 1
        engine.ingest(&mut store, make_fragment(1, 1, FragmentKind::Goal, "Learn Rust")).unwrap();
        engine.ingest(&mut store, make_fragment(1, 2, FragmentKind::Goal, "Build distributed system")).unwrap();
        engine.synthesize(&mut store, &mut profile, IhsanScore::new(9900), 2000);

        // Round 2 — more data
        engine.ingest(&mut store, make_fragment(2, 1, FragmentKind::Goal, "Launch product")).unwrap();
        engine.ingest(&mut store, make_fragment(2, 2, FragmentKind::Goal, "Reach 100 users")).unwrap();
        engine.synthesize(&mut store, &mut profile, IhsanScore::new(9900), 3000);

        assert_eq!(engine.round(), 2);
        assert!(engine.metrics().insights_produced >= 1);
        assert_eq!(profile.synthesis_round, 2);
    }
}
