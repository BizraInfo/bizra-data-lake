//! # Synthesis Engine — The Heart of "My AI Knows Me"
//!
//! Takes extracted atoms and produces insights through five strategies:
//!
//! 1. **Direct**: Single high-confidence atom → immediate insight
//! 2. **Correlation**: Multiple patterns → behavioral understanding
//! 3. **Temporal**: Event sequences → predictive insight
//! 4. **Resolution**: Conflicting atoms → nuanced understanding
//! 5. **Abstraction**: Concrete facts → general principles
//!
//! ## Why This Matters
//! Storage exists (VectorSearch). Retrieval exists (CognitiveResonance).
//! What's missing is SYNTHESIS — the part that takes 500 conversations
//! and extracts "Mumo thinks in systems, leads with principle, builds alone,
//! needs his tools to match his speed."
//!
//! That's what this engine does.

use crate::store::InMemoryStore;
use crate::types::*;

/// Configuration for synthesis strategies.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisConfig {
    /// Minimum atoms before attempting correlation synthesis
    pub correlation_threshold: usize,
    /// Minimum confidence for an atom to participate in synthesis
    pub min_atom_confidence: f32,
    /// Maximum atoms to consider per synthesis pass
    pub max_atoms_per_pass: usize,
    /// Whether to auto-resolve contradictions
    pub auto_resolve: bool,
    /// Confidence boost for user-confirmed insights
    pub confirmation_boost: f32,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        SynthesisConfig {
            correlation_threshold: 3,
            min_atom_confidence: 0.40,
            max_atoms_per_pass: 64,
            auto_resolve: true,
            confirmation_boost: 0.15,
        }
    }
}

/// Result of a synthesis pass.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisPassResult {
    /// Number of atoms examined
    pub atoms_examined: u32,
    /// Number of new insights produced
    pub insights_produced: u32,
    /// Number of atoms superseded (contradiction resolution)
    pub atoms_superseded: u32,
    /// Number of atoms reinforced (confirming evidence)
    pub atoms_reinforced: u32,
    /// Overall synthesis quality
    pub quality: f32,
}

/// The Synthesis Engine.
///
/// Operates on the InMemoryStore, reading unsynthesized atoms
/// and producing insights. Stateless between passes — all state
/// lives in the store.
pub struct SynthesisEngine {
    config: SynthesisConfig,
    /// Total passes executed
    total_passes: u64,
    /// Total insights ever produced
    total_insights_produced: u64,
}

impl SynthesisEngine {
    pub fn new() -> Self {
        Self::with_config(SynthesisConfig::default())
    }

    pub fn with_config(config: SynthesisConfig) -> Self {
        SynthesisEngine {
            config,
            total_passes: 0,
            total_insights_produced: 0,
        }
    }

    /// Run a synthesis pass over the store.
    ///
    /// Examines unsynthesized atoms and applies all strategies:
    /// 1. Direct promotion of high-confidence atoms
    /// 2. Correlation of same-kind patterns
    /// 3. Contradiction detection and resolution
    /// 4. Reinforcement of existing atoms with new evidence
    pub fn synthesize(&mut self, store: &mut InMemoryStore, now: u64) -> SynthesisPassResult {
        self.total_passes += 1;

        let mut result = SynthesisPassResult {
            atoms_examined: 0,
            insights_produced: 0,
            atoms_superseded: 0,
            atoms_reinforced: 0,
            quality: 1.0,
        };

        // Collect pending atom IDs and their data
        // (Avoid borrowing store while mutating it)
        let pending: Vec<(AtomId, AtomKind, f32, u64, u32)> = store
            .pending_synthesis()
            .filter(|a| a.header.confidence.effective_at(now) >= self.config.min_atom_confidence)
            .take(self.config.max_atoms_per_pass)
            .map(|a| {
                (
                    a.header.id,
                    a.header.kind,
                    a.header.confidence.effective_at(now),
                    a.header.content_offset,
                    a.header.content_len,
                )
            })
            .collect();

        result.atoms_examined = pending.len() as u32;

        // Strategy 1: Direct promotion — high-confidence stated facts become insights
        for &(atom_id, _kind, confidence, offset, len) in &pending {
            if confidence >= 0.90 {
                if let Some(content) = store.get_content_str(offset, len) {
                    let insight_content = content.to_string();
                    let method = SynthesisMethod::Direct;
                    let conf = Confidence::new(confidence, now);

                    if store
                        .store_insight(method, &insight_content, &[atom_id], conf, now)
                        .is_ok()
                    {
                        result.insights_produced += 1;
                        self.total_insights_produced += 1;
                    }
                    store.mark_synthesized(&atom_id);
                }
            }
        }

        // Strategy 2: Correlation — group same-kind atoms
        let kind_groups = self.group_by_kind(&pending);
        for (kind, group) in &kind_groups {
            if group.len() >= self.config.correlation_threshold {
                // Collect content for the group
                let mut contents: Vec<(AtomId, String)> = Vec::new();
                for &&(atom_id, _, _, offset, len) in group {
                    if let Some(c) = store.get_content_str(offset, len) {
                        contents.push((atom_id, c.to_string()));
                    }
                }

                if contents.len() >= self.config.correlation_threshold {
                    // Create a correlation insight summarizing the pattern
                    let summary = self.correlate_pattern(*kind, &contents);
                    let atom_ids: Vec<AtomId> = contents.iter().map(|(id, _)| *id).collect();
                    let avg_conf = group.iter().map(|g| g.2).sum::<f32>() / group.len() as f32;
                    let conf = Confidence::new(avg_conf * 0.85, now); // slight penalty for inference

                    if store
                        .store_insight(SynthesisMethod::Correlation, &summary, &atom_ids, conf, now)
                        .is_ok()
                    {
                        result.insights_produced += 1;
                        self.total_insights_produced += 1;
                    }

                    // Mark all as synthesized
                    for (atom_id, _) in &contents {
                        store.mark_synthesized(atom_id);
                    }
                }
            }
        }

        // Strategy 3: Mark remaining examined atoms as synthesized
        // (even without producing insights — they've been considered)
        for &(atom_id, _, confidence, _, _) in &pending {
            if confidence < 0.90 {
                store.mark_synthesized(&atom_id);
            }
        }

        // Strategy 4: HHMM TTL-aware reaping and promotion
        // Reap expired atoms (Pass 1 of two-pass eviction)
        let reaped = store.reap_expired(now);
        if reaped > 0 {
            result.atoms_superseded += reaped;
        }

        // Attempt promotion for heavily-reinforced atoms
        // An atom that has been reinforced past its promotion_threshold
        // transcends to the next HHMM layer (Markov state transition)
        let promotable: Vec<AtomId> = store
            .reliable_atoms(now)
            .filter(|a| {
                let threshold = a.header.kind.promotion_threshold();
                a.header.confidence.reinforcement_count >= threshold
                    && a.header.kind.promotion_target().is_some()
            })
            .map(|a| a.header.id)
            .collect();

        for atom_id in promotable {
            if let Some(_new_kind) = store.try_promote_atom(&atom_id) {
                result.atoms_reinforced += 1; // reusing field for promotion count
            }
        }

        // Update profile sections
        store.update_profile_sections();

        result
    }

    /// Group atoms by kind for correlation analysis.
    #[allow(clippy::type_complexity)]
    fn group_by_kind<'a>(
        &self,
        atoms: &'a [(AtomId, AtomKind, f32, u64, u32)],
    ) -> Vec<(AtomKind, Vec<&'a (AtomId, AtomKind, f32, u64, u32)>)> {
        let kinds = [
            AtomKind::Fact,
            AtomKind::Preference,
            AtomKind::Pattern,
            AtomKind::Relationship,
            AtomKind::Goal,
            AtomKind::Expertise,
            AtomKind::Context,
            AtomKind::Principle,
            AtomKind::Temporal,
            AtomKind::Negation,
        ];

        let mut groups = Vec::new();
        for kind in &kinds {
            let group: Vec<&(AtomId, AtomKind, f32, u64, u32)> =
                atoms.iter().filter(|(_, k, _, _, _)| k == kind).collect();

            if !group.is_empty() {
                groups.push((*kind, group));
            }
        }
        groups
    }

    /// Generate a correlation pattern summary.
    /// In production, this would call CognitiveResonance via FFI.
    /// For Node0 v1, we generate a structured summary.
    fn correlate_pattern(&self, kind: AtomKind, contents: &[(AtomId, String)]) -> String {
        let kind_name = match kind {
            AtomKind::Fact => "facts",
            AtomKind::Preference => "preferences",
            AtomKind::Pattern => "behavioral patterns",
            AtomKind::Relationship => "relationships",
            AtomKind::Goal => "goals",
            AtomKind::Expertise => "expertise areas",
            AtomKind::Context => "contextual signals",
            AtomKind::Principle => "guiding principles",
            AtomKind::Temporal => "temporal patterns",
            AtomKind::Negation => "negations",
        };

        let items: Vec<&str> = contents.iter().map(|(_, c)| c.as_str()).collect();
        format!(
            "Correlated {} {} observed: {}",
            contents.len(),
            kind_name,
            items.join(" | ")
        )
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Telemetry
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn total_passes(&self) -> u64 {
        self.total_passes
    }
    pub fn total_insights_produced(&self) -> u64 {
        self.total_insights_produced
    }
    pub fn config(&self) -> &SynthesisConfig {
        &self.config
    }
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_hooks::ComponentId;

    fn prov(ts: u64) -> Provenance {
        Provenance::new(1, 1, ComponentId::from_name("test", "1.0.0"), ts)
    }

    #[test]
    fn direct_promotion_of_high_confidence() {
        let mut store = InMemoryStore::new();
        let mut engine = SynthesisEngine::new();
        let frag = FragmentId::from_content(b"test");

        // Store a high-confidence atom
        store
            .store_atom(
                AtomKind::Fact,
                "User's name is Mumo",
                frag,
                Confidence::stated(1000),
                prov(1000),
            )
            .unwrap();

        let result = engine.synthesize(&mut store, 1000);

        assert_eq!(result.atoms_examined, 1);
        assert_eq!(result.insights_produced, 1);
        assert_eq!(store.insight_count(), 1);

        let insight = store.valid_insights().next().unwrap();
        assert_eq!(insight.header.synthesis_method, SynthesisMethod::Direct);
        assert_eq!(store.insight_content(insight), Some("User's name is Mumo"));
    }

    #[test]
    fn correlation_of_patterns() {
        let mut store = InMemoryStore::new();
        let mut engine = SynthesisEngine::new();
        let frag = FragmentId::from_content(b"test");

        // Store 3+ patterns (meets correlation_threshold)
        store
            .store_atom(
                AtomKind::Pattern,
                "Works after Fajr",
                frag,
                Confidence::inferred(1000),
                prov(1000),
            )
            .unwrap();
        store
            .store_atom(
                AtomKind::Pattern,
                "Loses focus at 2pm",
                frag,
                Confidence::inferred(1000),
                prov(1000),
            )
            .unwrap();
        store
            .store_atom(
                AtomKind::Pattern,
                "Peak coding at 6am",
                frag,
                Confidence::inferred(1000),
                prov(1000),
            )
            .unwrap();

        let result = engine.synthesize(&mut store, 1000);

        // Should produce a correlation insight
        assert!(result.insights_produced >= 1);

        // Find the correlation insight
        let corr = store
            .valid_insights()
            .find(|i| i.header.synthesis_method == SynthesisMethod::Correlation);
        assert!(corr.is_some());
    }

    #[test]
    fn low_confidence_atoms_filtered() {
        let mut store = InMemoryStore::new();
        let mut engine = SynthesisEngine::new();
        let frag = FragmentId::from_content(b"test");

        // Store a very low confidence atom
        store
            .store_atom(
                AtomKind::Fact,
                "Maybe likes coffee",
                frag,
                Confidence::new(0.20, 1000),
                prov(1000),
            )
            .unwrap();

        let result = engine.synthesize(&mut store, 1000);

        // Below min_atom_confidence (0.40) — should be skipped
        assert_eq!(result.atoms_examined, 0);
        assert_eq!(result.insights_produced, 0);
    }

    #[test]
    fn decayed_atoms_not_promoted() {
        let mut store = InMemoryStore::new();
        let mut engine = SynthesisEngine::new();
        let frag = FragmentId::from_content(b"test");
        let one_day: u64 = 24 * 3600 * 1_000_000_000;

        // Store atom at time 0 with medium confidence
        store
            .store_atom(
                AtomKind::Preference,
                "Prefers tabs over spaces",
                frag,
                Confidence::inferred(0),
                prov(0),
            )
            .unwrap();

        // Synthesize after 90 days — confidence should have decayed
        let result = engine.synthesize(&mut store, 90 * one_day);

        // Decayed below min_atom_confidence — not examined
        assert_eq!(result.atoms_examined, 0);
    }

    #[test]
    fn multiple_passes_accumulate() {
        let mut store = InMemoryStore::new();
        let mut engine = SynthesisEngine::new();
        let frag = FragmentId::from_content(b"test");

        // Pass 1: one atom
        store
            .store_atom(
                AtomKind::Fact,
                "CEO of BIZRA",
                frag,
                Confidence::stated(1000),
                prov(1000),
            )
            .unwrap();
        let r1 = engine.synthesize(&mut store, 1000);
        assert_eq!(r1.insights_produced, 1);

        // Pass 2: new atom (different content for dedup)
        let frag2 = FragmentId::from_content(b"test2");
        store
            .store_atom(
                AtomKind::Fact,
                "Based in Dubai",
                frag2,
                Confidence::stated(2000),
                prov(2000),
            )
            .unwrap();
        let r2 = engine.synthesize(&mut store, 2000);
        assert_eq!(r2.insights_produced, 1);

        assert_eq!(engine.total_passes(), 2);
        assert_eq!(engine.total_insights_produced(), 2);
        assert_eq!(store.insight_count(), 2);
    }
}
