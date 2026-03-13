// bizra-memory/src/store.rs
// ============================================================
// Memory Store — where fragments and insights live
// ============================================================
// Trait-based design: InMemoryStore for Node0 MVP,
// pluggable backends (SQLite, VectorDB) via trait implementation.
//
// Design: Fixed-capacity arrays, hash-addressed slots,
// no heap allocation in core operations.
// ============================================================

use crate::types::*;
use bizra_hooks::IhsanScore;

// ============================================================
// STORE CONFIGURATION
// ============================================================

pub const MAX_FRAGMENTS: usize = 4096;
pub const MAX_INSIGHTS: usize = 1024;

// ============================================================
// STORE RESULT TYPE
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// Store is at capacity
    Full,
    /// Fragment/Insight not found
    NotFound,
    /// Duplicate ID
    Duplicate,
    /// إحسان score too low for storage
    IhsanBelowThreshold,
    /// Fragment confidence below threshold
    ConfidenceBelowThreshold,
}

pub type StoreResult<T> = Result<T, StoreError>;

// ============================================================
// STORE QUERY — how to find what you're looking for
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct FragmentQuery {
    pub kind: Option<FragmentKind>,
    pub min_confidence: Option<Confidence>,
    pub active_only: bool,
    pub max_results: usize,
}

impl FragmentQuery {
    pub fn all() -> Self {
        Self {
            kind: None,
            min_confidence: None,
            active_only: true,
            max_results: 100,
        }
    }

    pub fn by_kind(kind: FragmentKind) -> Self {
        Self {
            kind: Some(kind),
            min_confidence: None,
            active_only: true,
            max_results: 100,
        }
    }

    pub fn high_confidence() -> Self {
        Self {
            kind: None,
            min_confidence: Some(Confidence::HIGH),
            active_only: true,
            max_results: 100,
        }
    }
}

// ============================================================
// IN-MEMORY STORE — Node0 MVP implementation
// ============================================================

pub struct InMemoryStore {
    fragments: Vec<Option<MemoryFragment>>,
    insights: Vec<Option<Insight>>,
    fragment_count: usize,
    insight_count: usize,
    min_ihsan_for_storage: IhsanScore,
    total_stored: u64,
    total_evicted: u64,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            fragments: (0..MAX_FRAGMENTS).map(|_| None).collect(),
            insights: (0..MAX_INSIGHTS).map(|_| None).collect(),
            fragment_count: 0,
            insight_count: 0,
            min_ihsan_for_storage: IhsanScore::new(9500), // 0.950 floor
            total_stored: 0,
            total_evicted: 0,
        }
    }

    /// Set minimum إحسان score for accepting new fragments
    pub fn set_ihsan_floor(&mut self, score: IhsanScore) {
        self.min_ihsan_for_storage = score;
    }

    // --- Fragment Operations ---

    /// Store a new fragment
    pub fn store_fragment(&mut self, fragment: MemoryFragment) -> StoreResult<()> {
        // إحسان gate
        if fragment.ihsan_at_creation.raw() < self.min_ihsan_for_storage.raw() {
            return Err(StoreError::IhsanBelowThreshold);
        }

        // Confidence gate
        if !fragment.confidence.meets_threshold() {
            return Err(StoreError::ConfidenceBelowThreshold);
        }

        // Check for duplicate
        let slot_idx = self.fragment_slot(fragment.id);
        if let Some(existing) = &self.fragments[slot_idx] {
            if existing.id == fragment.id {
                return Err(StoreError::Duplicate);
            }
        }

        // Find slot: try hash slot first, then linear probe
        let idx = self.find_fragment_slot(fragment.id)?;
        self.fragments[idx] = Some(fragment);
        self.fragment_count += 1;
        self.total_stored += 1;
        Ok(())
    }

    /// Retrieve a fragment by ID
    pub fn get_fragment(&self, id: FragmentId) -> Option<&MemoryFragment> {
        let start = self.fragment_slot(id);
        for offset in 0..MAX_FRAGMENTS {
            let idx = (start + offset) % MAX_FRAGMENTS;
            match &self.fragments[idx] {
                Some(f) if f.id == id => return Some(f),
                None => return None, // Empty slot means ID doesn't exist
                _ => continue,       // Collision, keep probing
            }
        }
        None
    }

    /// Retrieve a mutable fragment by ID
    pub fn get_fragment_mut(&mut self, id: FragmentId) -> Option<&mut MemoryFragment> {
        let start = self.fragment_slot(id);
        for offset in 0..MAX_FRAGMENTS {
            let idx = (start + offset) % MAX_FRAGMENTS;
            match &self.fragments[idx] {
                Some(f) if f.id == id => return self.fragments[idx].as_mut(),
                None => return None,
                _ => continue,
            }
        }
        None
    }

    /// Query fragments matching criteria
    pub fn query_fragments(&self, query: &FragmentQuery) -> Vec<&MemoryFragment> {
        let mut results = Vec::new();

        for slot in self.fragments.iter() {
            if results.len() >= query.max_results {
                break;
            }
            if let Some(f) = slot {
                // Active filter
                if query.active_only && !f.is_active() {
                    continue;
                }
                // Kind filter
                if let Some(kind) = query.kind {
                    if f.kind != kind {
                        continue;
                    }
                }
                // Confidence filter
                if let Some(min_conf) = query.min_confidence {
                    if f.confidence.raw() < min_conf.raw() {
                        continue;
                    }
                }
                results.push(f);
            }
        }

        results
    }

    /// Count active fragments
    pub fn fragment_count(&self) -> usize {
        self.fragment_count
    }

    /// Reinforce an existing fragment
    pub fn reinforce_fragment(&mut self, id: FragmentId, timestamp: u64) -> StoreResult<()> {
        match self.get_fragment_mut(id) {
            Some(f) => {
                f.reinforce(timestamp);
                Ok(())
            }
            None => Err(StoreError::NotFound),
        }
    }

    /// Supersede a fragment with a new one
    pub fn supersede_fragment(&mut self, old_id: FragmentId, new_id: FragmentId) -> StoreResult<()> {
        match self.get_fragment_mut(old_id) {
            Some(f) => {
                f.supersede(new_id);
                Ok(())
            }
            None => Err(StoreError::NotFound),
        }
    }

    // --- Insight Operations ---

    /// Store a new insight
    pub fn store_insight(&mut self, insight: Insight) -> StoreResult<()> {
        if self.insight_count >= MAX_INSIGHTS {
            return Err(StoreError::Full);
        }

        let idx = self.find_insight_slot(insight.id)?;
        self.insights[idx] = Some(insight);
        self.insight_count += 1;
        Ok(())
    }

    /// Retrieve an insight by ID
    pub fn get_insight(&self, id: InsightId) -> Option<&Insight> {
        let start = self.insight_slot(id);
        for offset in 0..MAX_INSIGHTS {
            let idx = (start + offset) % MAX_INSIGHTS;
            match &self.insights[idx] {
                Some(i) if i.id == id => return Some(i),
                None => return None,
                _ => continue,
            }
        }
        None
    }

    /// Retrieve a mutable insight by ID
    pub fn get_insight_mut(&mut self, id: InsightId) -> Option<&mut Insight> {
        let start = self.insight_slot(id);
        for offset in 0..MAX_INSIGHTS {
            let idx = (start + offset) % MAX_INSIGHTS;
            match &self.insights[idx] {
                Some(i) if i.id == id => return self.insights[idx].as_mut(),
                None => return None,
                _ => continue,
            }
        }
        None
    }

    /// Validate an existing insight
    pub fn validate_insight(&mut self, id: InsightId, timestamp: u64) -> StoreResult<()> {
        match self.get_insight_mut(id) {
            Some(i) => {
                i.validate(timestamp);
                Ok(())
            }
            None => Err(StoreError::NotFound),
        }
    }

    pub fn insight_count(&self) -> usize {
        self.insight_count
    }

    /// All active insights
    pub fn all_insights(&self) -> impl Iterator<Item = &Insight> {
        self.insights.iter().filter_map(|i| i.as_ref())
    }

    // --- Eviction & Maintenance ---

    /// Evict lowest-weight fragments to make room
    /// Returns number of fragments evicted
    pub fn evict_lowest(&mut self, count: usize, current_time: u64) -> usize {
        // Collect (index, weight) for active fragments
        let mut indexed: Vec<(usize, f32)> = self.fragments.iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref().map(|f| (i, f.synthesis_weight(current_time)))
            })
            .collect();

        // Sort by weight ascending (lowest first)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let to_evict = count.min(indexed.len());
        for &(idx, _) in indexed.iter().take(to_evict) {
            self.fragments[idx] = None;
            self.fragment_count = self.fragment_count.saturating_sub(1);
            self.total_evicted += 1;
        }

        to_evict
    }

    // --- Statistics ---

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            fragment_count: self.fragment_count,
            fragment_capacity: MAX_FRAGMENTS,
            insight_count: self.insight_count,
            insight_capacity: MAX_INSIGHTS,
            total_stored: self.total_stored,
            total_evicted: self.total_evicted,
            fragment_utilization: self.fragment_count as f32 / MAX_FRAGMENTS as f32,
            insight_utilization: self.insight_count as f32 / MAX_INSIGHTS as f32,
        }
    }

    // --- Internal Helpers ---

    fn fragment_slot(&self, id: FragmentId) -> usize {
        (id.0 as usize) % MAX_FRAGMENTS
    }

    fn find_fragment_slot(&self, id: FragmentId) -> StoreResult<usize> {
        let start = self.fragment_slot(id);
        for offset in 0..MAX_FRAGMENTS {
            let idx = (start + offset) % MAX_FRAGMENTS;
            match &self.fragments[idx] {
                None => return Ok(idx),
                Some(f) if f.id == id => return Err(StoreError::Duplicate),
                _ => continue,
            }
        }
        Err(StoreError::Full)
    }

    fn insight_slot(&self, id: InsightId) -> usize {
        (id.0 as usize) % MAX_INSIGHTS
    }

    fn find_insight_slot(&self, id: InsightId) -> StoreResult<usize> {
        let start = self.insight_slot(id);
        for offset in 0..MAX_INSIGHTS {
            let idx = (start + offset) % MAX_INSIGHTS;
            match &self.insights[idx] {
                None => return Ok(idx),
                Some(i) if i.id == id => return Err(StoreError::Duplicate),
                _ => continue,
            }
        }
        Err(StoreError::Full)
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// STORE STATS
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct StoreStats {
    pub fragment_count: usize,
    pub fragment_capacity: usize,
    pub insight_count: usize,
    pub insight_capacity: usize,
    pub total_stored: u64,
    pub total_evicted: u64,
    pub fragment_utilization: f32,
    pub insight_utilization: f32,
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
    fn store_and_retrieve_fragment() {
        let mut store = InMemoryStore::new();
        let frag = make_fragment(1, 1, FragmentKind::Fact, "User works at BIZRA");

        assert!(store.store_fragment(frag).is_ok());
        assert_eq!(store.fragment_count(), 1);

        let retrieved = store.get_fragment(FragmentId::new(1, 1)).unwrap();
        assert_eq!(retrieved.content.as_str(), "User works at BIZRA");
    }

    #[test]
    fn reject_duplicate_fragment() {
        let mut store = InMemoryStore::new();
        let frag1 = make_fragment(1, 1, FragmentKind::Fact, "First");
        let frag2 = make_fragment(1, 1, FragmentKind::Fact, "Duplicate");

        assert!(store.store_fragment(frag1).is_ok());
        assert_eq!(store.store_fragment(frag2), Err(StoreError::Duplicate));
    }

    #[test]
    fn reject_low_ihsan_fragment() {
        let mut store = InMemoryStore::new();
        let mut frag = make_fragment(1, 1, FragmentKind::Fact, "Low quality");
        frag.ihsan_at_creation = IhsanScore::new(9000); // Below 9500 floor

        assert_eq!(store.store_fragment(frag), Err(StoreError::IhsanBelowThreshold));
    }

    #[test]
    fn reject_low_confidence_fragment() {
        let mut store = InMemoryStore::new();
        let mut frag = make_fragment(1, 1, FragmentKind::Fact, "Uncertain");
        frag.confidence = Confidence::new(3000); // Below 6000 threshold

        assert_eq!(store.store_fragment(frag), Err(StoreError::ConfidenceBelowThreshold));
    }

    #[test]
    fn query_fragments_by_kind() {
        let mut store = InMemoryStore::new();
        store.store_fragment(make_fragment(1, 1, FragmentKind::Fact, "Fact 1")).unwrap();
        store.store_fragment(make_fragment(1, 2, FragmentKind::Preference, "Pref 1")).unwrap();
        store.store_fragment(make_fragment(1, 3, FragmentKind::Fact, "Fact 2")).unwrap();

        let facts = store.query_fragments(&FragmentQuery::by_kind(FragmentKind::Fact));
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn query_high_confidence_only() {
        let mut store = InMemoryStore::new();
        store.store_fragment(make_fragment(1, 1, FragmentKind::Fact, "High")).unwrap();

        let mut low_frag = make_fragment(1, 2, FragmentKind::Fact, "Medium");
        low_frag.confidence = Confidence::MEDIUM;
        store.store_fragment(low_frag).unwrap();

        let high = store.query_fragments(&FragmentQuery::high_confidence());
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn reinforce_fragment() {
        let mut store = InMemoryStore::new();
        store.store_fragment(make_fragment(1, 1, FragmentKind::Pattern, "Asks for examples")).unwrap();

        store.reinforce_fragment(FragmentId::new(1, 1), 2000).unwrap();

        let frag = store.get_fragment(FragmentId::new(1, 1)).unwrap();
        assert_eq!(frag.reinforcement_count, 2);
        assert_eq!(frag.last_reinforced, 2000);
    }

    #[test]
    fn supersede_fragment() {
        let mut store = InMemoryStore::new();
        store.store_fragment(make_fragment(1, 1, FragmentKind::Fact, "Old company")).unwrap();
        store.store_fragment(make_fragment(2, 1, FragmentKind::Fact, "New company")).unwrap();

        store.supersede_fragment(FragmentId::new(1, 1), FragmentId::new(2, 1)).unwrap();

        // Query active only — should not include superseded
        let active = store.query_fragments(&FragmentQuery::all());
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content.as_str(), "New company");
    }

    #[test]
    fn store_and_retrieve_insight() {
        let mut store = InMemoryStore::new();
        let sources = [FragmentId::new(1, 1), FragmentId::new(1, 2)];
        let insight = Insight::new(
            InsightId::new(1, 1),
            "Rust developer building distributed systems",
            &sources,
            1000,
            IhsanScore::new(9900),
        );

        assert!(store.store_insight(insight).is_ok());
        assert_eq!(store.insight_count(), 1);

        let retrieved = store.get_insight(InsightId::new(1, 1)).unwrap();
        assert_eq!(retrieved.source_count, 2);
    }

    #[test]
    fn validate_insight() {
        let mut store = InMemoryStore::new();
        let insight = Insight::new(
            InsightId::new(1, 1),
            "Prefers practical examples",
            &[FragmentId::new(1, 1)],
            1000,
            IhsanScore::new(9900),
        );
        store.store_insight(insight).unwrap();

        let initial_conf = store.get_insight(InsightId::new(1, 1)).unwrap().confidence.raw();
        store.validate_insight(InsightId::new(1, 1), 2000).unwrap();
        let new_conf = store.get_insight(InsightId::new(1, 1)).unwrap().confidence.raw();

        assert!(new_conf > initial_conf);
    }

    #[test]
    fn evict_lowest_weight_fragments() {
        let mut store = InMemoryStore::new();

        // Store fragments with very different weights
        let mut old = make_fragment(1, 1, FragmentKind::Temporal, "Low weight temporal");
        old.created_at = 100; // Very old
        old.last_reinforced = 100;
        old.decay_rate = 5000; // Fast decay
        store.store_fragment(old).unwrap();

        let fresh = make_fragment(2, 1, FragmentKind::Pattern, "High weight pattern");
        store.store_fragment(fresh).unwrap();

        assert_eq!(store.fragment_count(), 2);

        // Evict 1 — should remove the lowest weight fragment
        let evicted = store.evict_lowest(1, 100000);
        assert_eq!(evicted, 1);
        assert_eq!(store.fragment_count(), 1);
    }

    #[test]
    fn store_stats() {
        let mut store = InMemoryStore::new();
        store.store_fragment(make_fragment(1, 1, FragmentKind::Fact, "F1")).unwrap();
        store.store_fragment(make_fragment(1, 2, FragmentKind::Fact, "F2")).unwrap();

        let insight = Insight::new(
            InsightId::new(1, 1),
            "Combined insight",
            &[FragmentId::new(1, 1)],
            1000,
            IhsanScore::new(9900),
        );
        store.store_insight(insight).unwrap();

        let stats = store.stats();
        assert_eq!(stats.fragment_count, 2);
        assert_eq!(stats.insight_count, 1);
        assert_eq!(stats.total_stored, 2);
        assert!(stats.fragment_utilization > 0.0);
    }
}
