//! # Memory Store — Pluggable Knowledge Persistence
//!
//! Trait-based storage abstraction. The pipeline writes to stores,
//! queries read from stores. Backends are swappable:
//!
//! - `InMemoryStore`: Default for Node0. Fast, bounded, ephemeral.
//! - Future: SQLite, RocksDB, encrypted sovereign storage.
//!
//! ## Design
//! - Trait defines the contract
//! - InMemoryStore uses fixed arrays (no heap in core paths)
//! - Content stored separately from headers (header in events, content in store)
//! - Deduplication via content-addressed FragmentId

use crate::types::*;

/// Maximum fragments in the in-memory store.
const MAX_FRAGMENTS: usize = 4096;
/// Maximum atoms in the in-memory store.
const MAX_ATOMS: usize = 8192;
/// Maximum insights in the in-memory store.
const MAX_INSIGHTS: usize = 2048;
/// Content buffer size (1 MB).
const CONTENT_BUFFER_SIZE: usize = 1024 * 1024;

/// Errors from store operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// Store capacity exceeded for this item type
    StoreFull,
    /// Content buffer exhausted
    ContentBufferFull,
    /// Item not found
    NotFound,
    /// Duplicate item (content-addressed dedup)
    Duplicate,
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::StoreFull => write!(f, "store capacity exceeded"),
            Self::ContentBufferFull => write!(f, "content buffer exhausted"),
            Self::NotFound => write!(f, "item not found"),
            Self::Duplicate => write!(f, "duplicate item"),
        }
    }
}

/// A stored fragment: header + content slice reference.
#[derive(Debug, Clone, Copy)]
pub struct StoredFragment {
    pub header: FragmentHeader,
    /// Whether this fragment has been processed by the Extract stage
    pub extracted: bool,
}

/// A stored atom: header + content slice reference.
#[derive(Debug, Clone, Copy)]
pub struct StoredAtom {
    pub header: AtomHeader,
    /// Whether this atom has been processed by the Synthesize stage
    pub synthesized: bool,
    /// Whether this atom is superseded by a newer version
    pub superseded: bool,
}

/// A stored insight: header + content slice reference.
#[derive(Debug, Clone, Copy)]
pub struct StoredInsight {
    pub header: InsightHeader,
    /// Atom IDs that contribute to this insight (up to 16)
    pub contributing_atoms: [AtomId; 16],
    pub contributing_count: u8,
    /// Whether this insight is still valid
    pub valid: bool,
}

/// The in-memory store for Node0.
///
/// Fixed-capacity arrays. No heap allocation in hot paths.
/// Content is stored in a single contiguous buffer with offset tracking.
pub struct InMemoryStore {
    // Fragment storage
    fragments: Vec<StoredFragment>,
    // Atom storage
    atoms: Vec<StoredAtom>,
    // Insight storage
    insights: Vec<StoredInsight>,

    // Content buffer (shared across all types)
    content: Vec<u8>,
    content_cursor: usize,

    // Counters
    next_atom_id: u64,
    next_insight_id: u64,

    // Profile tracking
    profile: ProfileSnapshot,
}

impl InMemoryStore {
    pub fn new() -> Self {
        InMemoryStore {
            fragments: Vec::with_capacity(MAX_FRAGMENTS),
            atoms: Vec::with_capacity(MAX_ATOMS),
            insights: Vec::with_capacity(MAX_INSIGHTS),
            content: vec![0u8; CONTENT_BUFFER_SIZE],
            content_cursor: 0,
            next_atom_id: 1,
            next_insight_id: 1,
            profile: ProfileSnapshot {
                total_atoms: 0,
                total_insights: 0,
                active_atoms: 0,
                confidence: 0.0,
                ihsan: bizra_hooks::IhsanScore::MAX,
                last_updated: 0,
                populated_sections: 0,
            },
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Content Buffer Management
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Store content and return (offset, len).
    fn store_content(&mut self, content: &[u8]) -> Result<(u64, u32), StoreError> {
        let len = content.len();
        if self.content_cursor + len > self.content.len() {
            return Err(StoreError::ContentBufferFull);
        }
        let offset = self.content_cursor as u64;
        self.content[self.content_cursor..self.content_cursor + len].copy_from_slice(content);
        self.content_cursor += len;
        Ok((offset, len as u32))
    }

    /// Retrieve content by offset and length.
    pub fn get_content(&self, offset: u64, len: u32) -> Option<&[u8]> {
        let start = offset as usize;
        let end = start + len as usize;
        if end <= self.content.len() {
            Some(&self.content[start..end])
        } else {
            None
        }
    }

    /// Retrieve content as string.
    pub fn get_content_str(&self, offset: u64, len: u32) -> Option<&str> {
        self.get_content(offset, len)
            .and_then(|bytes| core::str::from_utf8(bytes).ok())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Fragment Operations
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Ingest a raw fragment with its content.
    pub fn ingest_fragment(
        &mut self,
        kind: FragmentKind,
        content: &[u8],
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> Result<FragmentId, StoreError> {
        if self.fragments.len() >= MAX_FRAGMENTS {
            return Err(StoreError::StoreFull);
        }

        let id = FragmentId::from_content(content);

        // Dedup check
        if self.fragments.iter().any(|f| f.header.id == id) {
            return Err(StoreError::Duplicate);
        }

        let (offset, len) = self.store_content(content)?;

        let header = FragmentHeader {
            id,
            kind,
            session_id,
            turn,
            timestamp,
            content_offset: offset,
            content_len: len,
            ihsan: bizra_hooks::IhsanScore::MAX,
        };

        self.fragments.push(StoredFragment {
            header,
            extracted: false,
        });

        Ok(id)
    }

    /// Get all unextracted fragments (ready for Extract stage).
    pub fn pending_extraction(&self) -> impl Iterator<Item = &StoredFragment> {
        self.fragments.iter().filter(|f| !f.extracted)
    }

    /// Mark a fragment as extracted.
    pub fn mark_extracted(&mut self, id: &FragmentId) -> bool {
        for frag in &mut self.fragments {
            if frag.header.id == *id {
                frag.extracted = true;
                return true;
            }
        }
        false
    }

    /// Get fragment by ID.
    pub fn get_fragment(&self, id: &FragmentId) -> Option<&StoredFragment> {
        self.fragments.iter().find(|f| f.header.id == *id)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Atom Operations
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Store an extracted atom with its content.
    pub fn store_atom(
        &mut self,
        kind: AtomKind,
        content: &str,
        source_fragment: FragmentId,
        confidence: Confidence,
        provenance: Provenance,
    ) -> Result<AtomId, StoreError> {
        if self.atoms.len() >= MAX_ATOMS {
            return Err(StoreError::StoreFull);
        }

        let id = AtomId::new(self.next_atom_id);
        self.next_atom_id += 1;

        let (offset, len) = self.store_content(content.as_bytes())?;

        let header = AtomHeader {
            id,
            kind,
            source_fragment,
            confidence,
            provenance,
            content_offset: offset,
            content_len: len,
        };

        self.atoms.push(StoredAtom {
            header,
            synthesized: false,
            superseded: false,
        });

        self.profile.total_atoms += 1;
        self.profile.active_atoms += 1;
        self.profile.last_updated = provenance.extracted_at;

        Ok(id)
    }

    /// Get all unsynthesized atoms.
    pub fn pending_synthesis(&self) -> impl Iterator<Item = &StoredAtom> {
        self.atoms
            .iter()
            .filter(|a| !a.synthesized && !a.superseded)
    }

    /// Mark an atom as synthesized.
    pub fn mark_synthesized(&mut self, id: &AtomId) -> bool {
        for atom in &mut self.atoms {
            if atom.header.id == *id {
                atom.synthesized = true;
                return true;
            }
        }
        false
    }

    /// Get atom by ID.
    pub fn get_atom(&self, id: &AtomId) -> Option<&StoredAtom> {
        self.atoms.iter().find(|a| a.header.id == *id)
    }

    /// Get atom content as string.
    pub fn atom_content(&self, atom: &StoredAtom) -> Option<&str> {
        self.get_content_str(atom.header.content_offset, atom.header.content_len)
    }

    /// Find atoms by kind.
    pub fn atoms_by_kind(&self, kind: AtomKind) -> impl Iterator<Item = &StoredAtom> {
        self.atoms
            .iter()
            .filter(move |a| a.header.kind == kind && !a.superseded)
    }

    /// Find atoms with confidence above threshold at given time.
    pub fn reliable_atoms(&self, now: u64) -> impl Iterator<Item = &StoredAtom> {
        self.atoms
            .iter()
            .filter(move |a| !a.superseded && a.header.confidence.is_reliable(now))
    }

    /// Supersede an atom (mark as replaced by newer knowledge).
    pub fn supersede_atom(&mut self, id: &AtomId) -> bool {
        for atom in &mut self.atoms {
            if atom.header.id == *id {
                atom.superseded = true;
                self.profile.active_atoms = self.profile.active_atoms.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Reinforce an atom's confidence (saw confirming evidence).
    pub fn reinforce_atom(&mut self, id: &AtomId, timestamp: u64) -> bool {
        for atom in &mut self.atoms {
            if atom.header.id == *id {
                atom.header.confidence.reinforce(timestamp);
                return true;
            }
        }
        false
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // HHMM Temporal Granularity — TTL-aware eviction & promotion
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Reap expired atoms based on HHMM-aware TTL.
    ///
    /// Pass 1 of the two-pass eviction strategy:
    /// 1. Reap atoms that have exceeded their kind-specific TTL
    /// 2. (Existing) Capacity-based eviction if still over limit
    ///
    /// An atom expires when: now - last_reinforced > ttl_days * 86400 * 1e9
    /// Reinforcement count extends effective TTL via sqrt factor (Ebbinghaus).
    ///
    /// Returns the number of atoms reaped.
    pub fn reap_expired(&mut self, now: u64) -> u32 {
        let mut reaped = 0u32;
        for atom in &mut self.atoms {
            if atom.superseded {
                continue;
            }
            let kind = atom.header.kind;
            let ttl_nanos = kind.ttl_days() as u64 * 24 * 3600 * 1_000_000_000;
            // Reinforcement extends TTL: effective_ttl = ttl * sqrt(reinforcement_count)
            let count_factor = (atom.header.confidence.reinforcement_count as f64).sqrt();
            let effective_ttl = (ttl_nanos as f64 * count_factor) as u64;

            let elapsed = now.saturating_sub(atom.header.confidence.last_reinforced);
            if elapsed > effective_ttl {
                atom.superseded = true;
                self.profile.active_atoms = self.profile.active_atoms.saturating_sub(1);
                reaped += 1;
            }
        }
        reaped
    }

    /// Promote an atom to a higher HHMM layer.
    ///
    /// When an atom has been reinforced enough times (exceeds promotion_threshold),
    /// it transcends its current layer. A Goal mentioned 5+ times becomes a Pattern.
    /// A Pattern confirmed 10+ times becomes a Fact.
    ///
    /// The promotion:
    /// 1. Changes the atom's kind to the promotion target
    /// 2. Updates the half_life_nanos to match the new kind's TTL
    /// 3. Preserves all other metadata (content, provenance, confidence)
    ///
    /// Returns Some(new_kind) if promoted, None if not eligible.
    pub fn try_promote_atom(&mut self, id: &AtomId) -> Option<AtomKind> {
        // Find the atom and check promotion eligibility
        let (eligible, new_kind) = {
            let atom = self
                .atoms
                .iter()
                .find(|a| a.header.id == *id && !a.superseded)?;
            let kind = atom.header.kind;
            let count = atom.header.confidence.reinforcement_count;
            let threshold = kind.promotion_threshold();

            if count >= threshold {
                kind.promotion_target().map(|target| (true, target))
            } else {
                None
            }
            .unwrap_or((false, kind))
        };

        if !eligible {
            return None;
        }

        // Apply the promotion
        for atom in &mut self.atoms {
            if atom.header.id == *id && !atom.superseded {
                atom.header.kind = new_kind;
                atom.header.confidence.half_life_nanos = new_kind.half_life_nanos();
                return Some(new_kind);
            }
        }
        None
    }

    /// Get atoms grouped by HHMM layer for diagnostics.
    pub fn atoms_by_layer(&self) -> (u32, u32, u32) {
        let mut fast = 0u32;
        let mut slow = 0u32;
        let mut glacial = 0u32;
        for atom in &self.atoms {
            if atom.superseded {
                continue;
            }
            match atom.header.kind.hhmm_layer() {
                HhmmLayer::Fast => fast += 1,
                HhmmLayer::Slow => slow += 1,
                HhmmLayer::Glacial => glacial += 1,
            }
        }
        (fast, slow, glacial)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Insight Operations
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Store a synthesized insight.
    pub fn store_insight(
        &mut self,
        method: SynthesisMethod,
        content: &str,
        contributing_atoms: &[AtomId],
        confidence: Confidence,
        timestamp: u64,
    ) -> Result<InsightId, StoreError> {
        if self.insights.len() >= MAX_INSIGHTS {
            return Err(StoreError::StoreFull);
        }

        let id = InsightId::new(self.next_insight_id);
        self.next_insight_id += 1;

        let (offset, len) = self.store_content(content.as_bytes())?;

        let mut atoms = [AtomId::null(); 16];
        let count = contributing_atoms.len().min(16);
        atoms[..count].copy_from_slice(&contributing_atoms[..count]);

        let header = InsightHeader {
            id,
            synthesis_method: method,
            atom_count: count as u16,
            confidence,
            created_at: timestamp,
            last_validated: timestamp,
            content_offset: offset,
            content_len: len,
        };

        self.insights.push(StoredInsight {
            header,
            contributing_atoms: atoms,
            contributing_count: count as u8,
            valid: true,
        });

        self.profile.total_insights += 1;
        self.profile.last_updated = timestamp;

        Ok(id)
    }

    /// Get insight by ID.
    pub fn get_insight(&self, id: &InsightId) -> Option<&StoredInsight> {
        self.insights.iter().find(|i| i.header.id == *id)
    }

    /// Get insight content as string.
    pub fn insight_content(&self, insight: &StoredInsight) -> Option<&str> {
        self.get_content_str(insight.header.content_offset, insight.header.content_len)
    }

    /// All valid insights.
    pub fn valid_insights(&self) -> impl Iterator<Item = &StoredInsight> {
        self.insights.iter().filter(|i| i.valid)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Profile & Statistics
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get current profile snapshot.
    pub fn profile(&self) -> &ProfileSnapshot {
        &self.profile
    }

    /// Update profile section bitmask when atoms of relevant kinds arrive.
    pub fn update_profile_sections(&mut self) {
        let mut sections: u8 = 0;
        for atom in &self.atoms {
            if atom.superseded {
                continue;
            }
            let section = match atom.header.kind {
                AtomKind::Fact | AtomKind::Relationship => ProfileSection::Identity,
                AtomKind::Goal => ProfileSection::Goals,
                AtomKind::Preference => ProfileSection::Communication,
                AtomKind::Expertise => ProfileSection::Technical,
                AtomKind::Pattern | AtomKind::Temporal => ProfileSection::Temporal,
                AtomKind::Principle | AtomKind::Context => ProfileSection::Values,
                AtomKind::Negation => continue,
            };
            sections |= 1 << section as u8;
        }
        self.profile.populated_sections = sections;
    }

    /// Total items in store (all types).
    pub fn total_items(&self) -> usize {
        self.fragments.len() + self.atoms.len() + self.insights.len()
    }

    /// Content buffer usage.
    pub fn content_usage(&self) -> (usize, usize) {
        (self.content_cursor, CONTENT_BUFFER_SIZE)
    }

    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }
    pub fn insight_count(&self) -> usize {
        self.insights.len()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_hooks::ComponentId;

    fn test_provenance() -> Provenance {
        Provenance::new(1, 1, ComponentId::from_name("test", "1.0.0"), 1000)
    }

    #[test]
    fn ingest_and_retrieve_fragment() {
        let mut store = InMemoryStore::new();
        let content = b"User said: I am working on BIZRA";

        let id = store
            .ingest_fragment(FragmentKind::UserMessage, content, 1, 1, 1000)
            .unwrap();

        let frag = store.get_fragment(&id).unwrap();
        assert_eq!(frag.header.kind, FragmentKind::UserMessage);
        assert!(!frag.extracted);

        let retrieved = store
            .get_content(frag.header.content_offset, frag.header.content_len)
            .unwrap();
        assert_eq!(retrieved, content);
    }

    #[test]
    fn dedup_prevents_duplicate_fragments() {
        let mut store = InMemoryStore::new();
        let content = b"same content twice";

        store
            .ingest_fragment(FragmentKind::UserMessage, content, 1, 1, 1000)
            .unwrap();
        let result = store.ingest_fragment(FragmentKind::UserMessage, content, 1, 2, 2000);
        assert_eq!(result, Err(StoreError::Duplicate));
    }

    #[test]
    fn store_and_query_atoms() {
        let mut store = InMemoryStore::new();
        let frag_id = FragmentId::from_content(b"test");

        let atom_id = store
            .store_atom(
                AtomKind::Fact,
                "User's name is Mumo",
                frag_id,
                Confidence::stated(1000),
                test_provenance(),
            )
            .unwrap();

        let atom = store.get_atom(&atom_id).unwrap();
        assert_eq!(atom.header.kind, AtomKind::Fact);
        assert_eq!(store.atom_content(atom), Some("User's name is Mumo"));
    }

    #[test]
    fn atom_lifecycle() {
        let mut store = InMemoryStore::new();
        let frag_id = FragmentId::from_content(b"test");

        let id = store
            .store_atom(
                AtomKind::Preference,
                "Prefers Rust",
                frag_id,
                Confidence::inferred(1000),
                test_provenance(),
            )
            .unwrap();

        assert_eq!(store.profile().active_atoms, 1);

        // Reinforce
        store.reinforce_atom(&id, 2000);
        let atom = store.get_atom(&id).unwrap();
        assert_eq!(atom.header.confidence.reinforcement_count, 2);

        // Supersede
        store.supersede_atom(&id);
        assert_eq!(store.profile().active_atoms, 0);
    }

    #[test]
    fn store_insight_from_atoms() {
        let mut store = InMemoryStore::new();
        let frag_id = FragmentId::from_content(b"test");

        let a1 = store
            .store_atom(
                AtomKind::Pattern,
                "Works best after Fajr",
                frag_id,
                Confidence::inferred(1000),
                test_provenance(),
            )
            .unwrap();

        let a2 = store
            .store_atom(
                AtomKind::Pattern,
                "Loses focus at 2pm",
                frag_id,
                Confidence::inferred(1000),
                test_provenance(),
            )
            .unwrap();

        let ins_id = store
            .store_insight(
                SynthesisMethod::Correlation,
                "Schedule deep work morning, admin afternoon",
                &[a1, a2],
                Confidence::inferred(1000),
                2000,
            )
            .unwrap();

        let insight = store.get_insight(&ins_id).unwrap();
        assert_eq!(insight.contributing_count, 2);
        assert_eq!(
            insight.header.synthesis_method,
            SynthesisMethod::Correlation
        );
        assert_eq!(
            store.insight_content(insight),
            Some("Schedule deep work morning, admin afternoon")
        );
    }

    #[test]
    fn profile_section_tracking() {
        let mut store = InMemoryStore::new();
        let frag_id = FragmentId::from_content(b"test");

        store
            .store_atom(
                AtomKind::Fact,
                "name is Mumo",
                frag_id,
                Confidence::stated(1000),
                test_provenance(),
            )
            .unwrap();
        store
            .store_atom(
                AtomKind::Expertise,
                "knows distributed systems",
                frag_id,
                Confidence::inferred(1000),
                test_provenance(),
            )
            .unwrap();
        store
            .store_atom(
                AtomKind::Goal,
                "preparing investor pitch",
                frag_id,
                Confidence::stated(1000),
                test_provenance(),
            )
            .unwrap();

        store.update_profile_sections();

        let profile = store.profile();
        assert!(profile.has_section(ProfileSection::Identity));
        assert!(profile.has_section(ProfileSection::Technical));
        assert!(profile.has_section(ProfileSection::Goals));
        assert!(!profile.has_section(ProfileSection::Communication));
        assert_eq!(profile.section_count(), 3);
    }
}
