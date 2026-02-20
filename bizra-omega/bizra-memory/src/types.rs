//! # Memory Types — What BIZRA Knows About You
//!
//! Every piece of knowledge flows through these types. From raw conversation
//! fragments to synthesized insights to the living user profile.
//!
//! ## Memory Hierarchy
//! ```text
//! MemoryFragment (raw input: conversation turn, file, observation)
//!     ↓ Extract
//! MemoryAtom (single fact, preference, pattern, or context)
//!     ↓ Synthesize
//! Insight (connected understanding: "user thinks in systems")
//!     ↓ Integrate
//! UserProfile (living model: who they are, how they work, what they need)
//! ```
//!
//! ## Design
//! - Fixed-size core types for hot paths (no heap in event loop)
//! - Heap-backed types for rich content (synthesis, profile)
//! - Every type carries temporal metadata (when learned, confidence decay)
//! - Every type carries provenance (which conversation, which turn)

use bizra_hooks::types::{ComponentId, IhsanScore};
use core::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Identity Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 128-bit memory fragment identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentId(pub [u8; 16]);

/// 64-bit atom identifier (extracted fact/preference/pattern).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomId(pub u64);

/// 64-bit insight identifier (synthesized understanding).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InsightId(pub u64);

impl fmt::Debug for FragmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frag:")?;
        for b in &self.0[..4] {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl fmt::Debug for AtomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atom:{:012x}", self.0)
    }
}

impl fmt::Debug for InsightId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ins:{:012x}", self.0)
    }
}

impl FragmentId {
    /// Generate from content hash (FNV-1a 128-bit, same as ComponentId).
    pub fn from_content(content: &[u8]) -> Self {
        let mut hash: u128 = 0x6c62272e07bb0142_62b821756295c58d;
        let prime: u128 = 0x0000000001000000_000000000000013b;
        for &byte in content {
            hash ^= byte as u128;
            hash = hash.wrapping_mul(prime);
        }
        FragmentId(hash.to_le_bytes())
    }

    pub const fn null() -> Self {
        FragmentId([0u8; 16])
    }
    pub fn is_null(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl AtomId {
    pub const fn new(id: u64) -> Self {
        AtomId(id)
    }
    pub const fn null() -> Self {
        AtomId(0)
    }
}

impl InsightId {
    pub const fn new(id: u64) -> Self {
        InsightId(id)
    }
    pub const fn null() -> Self {
        InsightId(0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Temporal Metadata — When and how confident
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Confidence score (0.0 - 1.0) with temporal decay.
/// Knowledge fades unless reinforced.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence {
    /// Base confidence when first learned (0.0 - 1.0)
    pub base: f32,
    /// Timestamp when last reinforced (nanos since epoch)
    pub last_reinforced: u64,
    /// Number of times this knowledge was reinforced
    pub reinforcement_count: u32,
    /// Decay half-life in nanoseconds (default: 30 days)
    pub half_life_nanos: u64,
}

impl Confidence {
    /// 30 days in nanoseconds
    const DEFAULT_HALF_LIFE: u64 = 30 * 24 * 3600 * 1_000_000_000;

    pub fn new(base: f32, timestamp: u64) -> Self {
        Confidence {
            base: base.clamp(0.0, 1.0),
            last_reinforced: timestamp,
            reinforcement_count: 1,
            half_life_nanos: Self::DEFAULT_HALF_LIFE,
        }
    }

    /// High confidence (0.95) — directly stated by user.
    pub fn stated(timestamp: u64) -> Self {
        Self::new(0.95, timestamp)
    }

    /// Medium confidence (0.70) — inferred from behavior.
    pub fn inferred(timestamp: u64) -> Self {
        Self::new(0.70, timestamp)
    }

    /// Low confidence (0.40) — speculative pattern.
    pub fn speculative(timestamp: u64) -> Self {
        Self::new(0.40, timestamp)
    }

    /// Calculate current effective confidence with temporal decay.
    pub fn effective_at(&self, now: u64) -> f32 {
        if now <= self.last_reinforced {
            return self.base;
        }

        let elapsed = now - self.last_reinforced;
        // Exponential decay: confidence * 2^(-elapsed/half_life)
        // Reinforcement count slows decay: effective_half_life = half_life * sqrt(count)
        let count_factor = (self.reinforcement_count as f64).sqrt();
        let effective_half_life = self.half_life_nanos as f64 * count_factor;

        if effective_half_life <= 0.0 {
            return 0.0;
        }

        let decay = (-0.693147 * elapsed as f64 / effective_half_life).exp();
        (self.base as f64 * decay) as f32
    }

    /// Reinforce this knowledge (saw it again).
    pub fn reinforce(&mut self, timestamp: u64) {
        self.last_reinforced = timestamp;
        self.reinforcement_count += 1;
        // Slight boost for repeated confirmation, capped at 0.99
        self.base = (self.base + 0.02).min(0.99);
    }

    /// Is this knowledge still reliable? (effective > 0.30)
    pub fn is_reliable(&self, now: u64) -> bool {
        self.effective_at(now) > 0.30
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::new(0.50, 0)
    }
}

impl fmt::Debug for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conf({:.2}, ×{})", self.base, self.reinforcement_count)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Provenance — Where knowledge came from
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Source tracking: which conversation, which turn, which engine extracted it.
#[derive(Debug, Clone, Copy)]
pub struct Provenance {
    /// Conversation/session ID
    pub session_id: u64,
    /// Turn number within session
    pub turn: u32,
    /// Which engine produced this (ComponentId of extractor)
    pub extractor: ComponentId,
    /// Timestamp of extraction
    pub extracted_at: u64,
}

impl Provenance {
    pub fn new(session_id: u64, turn: u32, extractor: ComponentId, timestamp: u64) -> Self {
        Provenance {
            session_id,
            turn,
            extractor,
            extracted_at: timestamp,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Memory Fragment — Raw input into the pipeline
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The kind of raw input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FragmentKind {
    /// User message in a conversation
    UserMessage = 0,
    /// Assistant response
    AssistantMessage = 1,
    /// File content the user shared
    FileContent = 2,
    /// Observed user behavior (action pattern)
    Observation = 3,
    /// System event (component status change, error, etc.)
    SystemEvent = 4,
    /// External data source (calendar, email, etc.)
    ExternalData = 5,
}

/// Fixed-size fragment header (for event bus transport).
/// The actual content is referenced by content_ptr/content_len
/// and lives in the memory store, not in the event.
#[derive(Debug, Clone, Copy)]
pub struct FragmentHeader {
    pub id: FragmentId,
    pub kind: FragmentKind,
    pub session_id: u64,
    pub turn: u32,
    pub timestamp: u64,
    /// Byte offset into content store
    pub content_offset: u64,
    /// Content length in bytes
    pub content_len: u32,
    /// Initial إحسان score of the content
    pub ihsan: IhsanScore,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Memory Atom — Extracted knowledge unit
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// What kind of knowledge this atom represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AtomKind {
    /// Concrete fact: "user's name is Mumo"
    Fact = 0,
    /// Preference: "user prefers Rust over Python for core systems"
    Preference = 1,
    /// Behavioral pattern: "user works best after Fajr"
    Pattern = 2,
    /// Relationship: "user is CEO of BIZRA"
    Relationship = 3,
    /// Goal/intention: "user is preparing investor pitch"
    Goal = 4,
    /// Skill/expertise: "user understands distributed systems"
    Expertise = 5,
    /// Emotional context: "user is passionate about sovereignty"
    Context = 6,
    /// Principle/value: "إحسان is the quality standard"
    Principle = 7,
    /// Temporal: "investor pitch due next week"
    Temporal = 8,
    /// Negation: "user does NOT want centralized dependencies"
    Negation = 9,
}

/// Fixed-size atom header for event transport.
#[derive(Debug, Clone, Copy)]
pub struct AtomHeader {
    pub id: AtomId,
    pub kind: AtomKind,
    /// Which fragment this was extracted from
    pub source_fragment: FragmentId,
    pub confidence: Confidence,
    pub provenance: Provenance,
    /// Content offset into atom store
    pub content_offset: u64,
    pub content_len: u32,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Insight — Synthesized understanding
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// An insight connects multiple atoms into understanding.
/// "User works best after Fajr" (Pattern) + "User loses focus at 2pm" (Pattern)
/// → Insight: "Schedule deep work in morning, admin in afternoon"
#[derive(Debug, Clone, Copy)]
pub struct InsightHeader {
    pub id: InsightId,
    /// Synthesis method that produced this
    pub synthesis_method: SynthesisMethod,
    /// Number of atoms that contributed
    pub atom_count: u16,
    /// Aggregate confidence (min of contributing atoms, adjusted)
    pub confidence: Confidence,
    /// When this insight was first synthesized
    pub created_at: u64,
    /// When last validated against new data
    pub last_validated: u64,
    /// Content offset into insight store
    pub content_offset: u64,
    pub content_len: u32,
}

/// How was this insight produced?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SynthesisMethod {
    /// Direct extraction (single atom → insight, high confidence)
    Direct = 0,
    /// Pattern correlation (multiple patterns → behavioral insight)
    Correlation = 1,
    /// Temporal inference (sequence of events → prediction)
    Temporal = 2,
    /// Contradiction resolution (conflicting atoms → nuanced understanding)
    Resolution = 3,
    /// Abstraction (concrete facts → general principle)
    Abstraction = 4,
    /// User-confirmed (insight shown to user and validated)
    UserConfirmed = 5,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// User Profile — The Living Model
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Profile section categories — what the system knows about the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ProfileSection {
    /// Identity: name, role, organization
    Identity = 0,
    /// Work: projects, responsibilities, deadlines
    Work = 1,
    /// Communication: style preferences, tone, language
    Communication = 2,
    /// Technical: skills, tools, preferences
    Technical = 3,
    /// Temporal: schedule patterns, rhythms, deadlines
    Temporal = 4,
    /// Values: principles, priorities, what matters
    Values = 5,
    /// Goals: current objectives, aspirations
    Goals = 6,
    /// Social: relationships, collaborators, team
    Social = 7,
}

/// Profile snapshot header — lightweight summary for event transport.
#[derive(Debug, Clone, Copy)]
pub struct ProfileSnapshot {
    /// Total atoms contributing to this profile
    pub total_atoms: u32,
    /// Total insights synthesized
    pub total_insights: u32,
    /// Number of active (non-decayed) atoms
    pub active_atoms: u32,
    /// Aggregate profile confidence
    pub confidence: f32,
    /// إحسان quality score of the profile
    pub ihsan: IhsanScore,
    /// Last update timestamp
    pub last_updated: u64,
    /// Sections that have content
    pub populated_sections: u8, // bitmask of ProfileSection
}

impl ProfileSnapshot {
    pub fn has_section(&self, section: ProfileSection) -> bool {
        self.populated_sections & (1 << section as u8) != 0
    }

    pub fn section_count(&self) -> u32 {
        self.populated_sections.count_ones()
    }

    /// Profile completeness (0.0 - 1.0): fraction of sections populated.
    pub fn completeness(&self) -> f32 {
        self.section_count() as f32 / 8.0
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pipeline Stage Markers — What stage is processing at?
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The five stages of the memory synthesis pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PipelineStage {
    /// Raw content enters the system
    Ingest = 0,
    /// Facts, preferences, patterns extracted from content
    Extract = 1,
    /// Atoms connected into insights
    Synthesize = 2,
    /// Insights indexed for retrieval
    Index = 3,
    /// Knowledge served to queries
    Query = 4,
}

/// Pipeline processing result.
#[derive(Debug, Clone, Copy)]
pub struct StageResult {
    pub stage: PipelineStage,
    pub success: bool,
    pub items_in: u32,
    pub items_out: u32,
    pub duration_nanos: u64,
    pub ihsan: IhsanScore,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Query Types — How the system retrieves knowledge
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// What kind of memory retrieval is requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QueryKind {
    /// Exact fact lookup: "what is the user's name?"
    FactLookup = 0,
    /// Semantic search: "what does the user care about?"
    SemanticSearch = 1,
    /// Temporal query: "what was the user working on last week?"
    TemporalRange = 2,
    /// Profile section: "user's communication preferences"
    ProfileSection = 3,
    /// Contextual: "given current task, what's relevant?"
    Contextual = 4,
    /// Predictive: "what will the user likely need next?"
    Predictive = 5,
}

/// Fixed-size query header for event transport.
#[derive(Debug, Clone, Copy)]
pub struct QueryHeader {
    pub kind: QueryKind,
    /// Maximum results to return
    pub max_results: u16,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Time range start (0 = no filter)
    pub time_from: u64,
    /// Time range end (0 = no filter)
    pub time_to: u64,
    /// Section filter (0 = all sections)
    pub section_filter: u8,
    /// Query content offset
    pub content_offset: u64,
    pub content_len: u32,
}

/// A single query result.
#[derive(Debug, Clone, Copy)]
pub struct QueryResult {
    /// What kind of result (atom or insight)
    pub result_kind: QueryResultKind,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f32,
    /// Confidence of the knowledge
    pub confidence: f32,
    /// Content offset for the result text
    pub content_offset: u64,
    pub content_len: u32,
    /// Source provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QueryResultKind {
    Atom = 0,
    Insight = 1,
    ProfileEntry = 2,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_id_deterministic() {
        let id1 = FragmentId::from_content(b"hello world");
        let id2 = FragmentId::from_content(b"hello world");
        let id3 = FragmentId::from_content(b"hello worlD");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn confidence_decay() {
        let one_day: u64 = 24 * 3600 * 1_000_000_000;
        let c = Confidence::new(0.90, 0);

        // At creation: full confidence
        assert!((c.effective_at(0) - 0.90).abs() < 0.01);

        // After 30 days (one half-life): ~45%
        let after_30d = c.effective_at(30 * one_day);
        assert!(after_30d < 0.50);
        assert!(after_30d > 0.40);
    }

    #[test]
    fn confidence_reinforcement_slows_decay() {
        let one_day: u64 = 24 * 3600 * 1_000_000_000;
        let mut c = Confidence::new(0.90, 0);

        // Reinforce 10 times
        for i in 1..=10 {
            c.reinforce(i * one_day);
        }

        // After 30 days from last reinforcement, should retain more
        // because reinforcement_count=11 → sqrt(11)≈3.3x half-life
        let after_30d = c.effective_at(10 * one_day + 30 * one_day);
        assert!(after_30d > 0.60); // Much higher than unreinforced
    }

    #[test]
    fn confidence_reliability_threshold() {
        let c = Confidence::new(0.90, 0);
        let one_day: u64 = 24 * 3600 * 1_000_000_000;

        assert!(c.is_reliable(0));
        assert!(c.is_reliable(20 * one_day));
        // After ~50 days, should drop below 0.30
        assert!(!c.is_reliable(60 * one_day));
    }

    #[test]
    fn profile_snapshot_sections() {
        let snap = ProfileSnapshot {
            total_atoms: 100,
            total_insights: 25,
            active_atoms: 80,
            confidence: 0.85,
            ihsan: IhsanScore::from_f64(0.99),
            last_updated: 1000,
            populated_sections: 0b00111001, // Identity, Technical, Temporal, Values
        };

        assert!(snap.has_section(ProfileSection::Identity));
        assert!(!snap.has_section(ProfileSection::Work));
        assert!(snap.has_section(ProfileSection::Technical));
        assert_eq!(snap.section_count(), 4);
        assert!((snap.completeness() - 0.5).abs() < 0.01);
    }
}
