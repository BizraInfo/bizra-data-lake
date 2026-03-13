// bizra-memory/src/types.rs
// ============================================================
// Memory Synthesis Type System
// ============================================================
// The atomic building blocks of memory. Every piece of knowledge
// BIZRA holds about a user flows through these types.
//
// Design principles:
// - Fixed-size where possible (no_std compatible core)
// - Deterministic ordering via timestamps
// - إحسان score propagation from hooks layer
// - Semantic density: each type carries maximum meaning
// ============================================================

use bizra_hooks::{ComponentId, IhsanScore};

// ============================================================
// MEMORY FRAGMENT — the atom of memory
// ============================================================
// A single extractable piece of knowledge from a conversation.
// Not the raw message — the distilled meaning.
// "User prefers Rust over Python" is a fragment.
// "User said 'I like Rust better'" is raw — not stored here.

/// Unique identifier for a memory fragment
/// Combines source conversation hash + extraction sequence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentId(pub u64);

impl FragmentId {
    pub fn new(conversation_hash: u32, sequence: u32) -> Self {
        Self(((conversation_hash as u64) << 32) | sequence as u64)
    }

    pub fn conversation_hash(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn sequence(&self) -> u32 {
        self.0 as u32
    }
}

/// What kind of knowledge does this fragment represent?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FragmentKind {
    /// Stated preference: "I prefer X over Y"
    Preference = 0,
    /// Factual about user: "I work at company X"
    Fact = 1,
    /// Behavioral pattern: user always asks for examples
    Pattern = 2,
    /// Emotional signal: frustration, excitement, curiosity
    Emotion = 3,
    /// Goal or aspiration: "I want to learn Rust"
    Goal = 4,
    /// Expertise indicator: user knows advanced TypeScript
    Expertise = 5,
    /// Relationship context: mentions team members, family
    Relationship = 6,
    /// Temporal context: deadlines, schedules, time preferences
    Temporal = 7,
    /// Domain knowledge: industry-specific understanding
    Domain = 8,
    /// Communication style: prefers bullet points, formal tone
    Style = 9,
}

impl FragmentKind {
    /// Weight for synthesis priority (higher = more valuable for personalization)
    pub fn synthesis_weight(&self) -> f32 {
        match self {
            Self::Preference => 0.9,
            Self::Fact => 0.7,
            Self::Pattern => 0.95,   // Patterns are gold — implicit knowledge
            Self::Emotion => 0.6,
            Self::Goal => 0.85,
            Self::Expertise => 0.8,
            Self::Relationship => 0.5,
            Self::Temporal => 0.4,
            Self::Domain => 0.75,
            Self::Style => 0.88,     // Style matters hugely for "knows me"
        }
    }
}

/// Confidence in fragment extraction accuracy
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Confidence(u16); // 0-10000 representing 0.0000-1.0000

impl Confidence {
    pub const MAX: Self = Self(10000);
    pub const HIGH: Self = Self(9000);
    pub const MEDIUM: Self = Self(7000);
    pub const LOW: Self = Self(5000);
    pub const THRESHOLD: Self = Self(6000); // Below this, don't synthesize

    pub fn new(value: u16) -> Self {
        Self(if value > 10000 { 10000 } else { value })
    }

    pub fn as_f32(&self) -> f32 {
        self.0 as f32 / 10000.0
    }

    pub fn raw(&self) -> u16 {
        self.0
    }

    pub fn meets_threshold(&self) -> bool {
        self.0 >= Self::THRESHOLD.0
    }
}

/// Fixed-size content buffer for fragment text
/// 512 bytes — enough for a distilled insight, not raw conversation
pub const FRAGMENT_CONTENT_SIZE: usize = 512;

#[derive(Clone)]
pub struct FragmentContent {
    data: [u8; FRAGMENT_CONTENT_SIZE],
    len: u16,
}

impl FragmentContent {
    pub fn new(text: &str) -> Self {
        let mut data = [0u8; FRAGMENT_CONTENT_SIZE];
        let bytes = text.as_bytes();
        let len = bytes.len().min(FRAGMENT_CONTENT_SIZE);
        data[..len].copy_from_slice(&bytes[..len]);
        Self { data, len: len as u16 }
    }

    pub fn as_str(&self) -> &str {
        // Safe: we only store valid UTF-8 from new()
        // But handle gracefully if somehow corrupted
        core::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("[corrupted]")
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for FragmentContent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FragmentContent({:?})", self.as_str())
    }
}

/// A single memory fragment — one extracted piece of knowledge
#[derive(Clone, Debug)]
pub struct MemoryFragment {
    pub id: FragmentId,
    pub kind: FragmentKind,
    pub content: FragmentContent,
    pub confidence: Confidence,
    pub source_component: ComponentId,
    pub created_at: u64,       // Unix timestamp
    pub last_reinforced: u64,  // Updated when same knowledge is seen again
    pub reinforcement_count: u16,
    pub ihsan_at_creation: IhsanScore,
    pub decay_rate: u16,       // 0-10000: how fast this fragment loses relevance
    pub superseded_by: Option<FragmentId>, // If newer fragment replaces this
}

impl MemoryFragment {
    pub fn new(
        id: FragmentId,
        kind: FragmentKind,
        content: &str,
        confidence: Confidence,
        source: ComponentId,
        timestamp: u64,
        ihsan: IhsanScore,
    ) -> Self {
        Self {
            id,
            kind,
            content: FragmentContent::new(content),
            confidence,
            source_component: source,
            created_at: timestamp,
            last_reinforced: timestamp,
            reinforcement_count: 1,
            ihsan_at_creation: ihsan,
            decay_rate: 500, // Default: moderate decay
            superseded_by: None,
        }
    }

    /// Effective weight for synthesis: combines kind weight, confidence, reinforcement, and decay
    pub fn synthesis_weight(&self, current_time: u64) -> f32 {
        let kind_w = self.kind.synthesis_weight();
        let conf_w = self.confidence.as_f32();
        let reinforce_w = (self.reinforcement_count as f32).min(10.0) / 10.0;

        // Time decay: exponential decay based on time since last reinforcement
        let age_hours = (current_time.saturating_sub(self.last_reinforced)) / 3600;
        let decay_factor = self.decay_rate as f32 / 10000.0;
        let time_w = (-decay_factor * age_hours as f32 / 720.0).exp(); // 30-day half-life at default

        kind_w * conf_w * (0.5 + 0.5 * reinforce_w) * time_w
    }

    /// Reinforce this fragment (same knowledge observed again)
    pub fn reinforce(&mut self, timestamp: u64) {
        self.last_reinforced = timestamp;
        self.reinforcement_count = self.reinforcement_count.saturating_add(1);
        // Each reinforcement slows decay (knowledge becomes more stable)
        self.decay_rate = self.decay_rate.saturating_sub(50);
    }

    /// Check if this fragment is still active (not superseded)
    pub fn is_active(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Supersede this fragment with a newer one
    pub fn supersede(&mut self, replacement: FragmentId) {
        self.superseded_by = Some(replacement);
    }
}

// ============================================================
// INSIGHT — synthesized knowledge from multiple fragments
// ============================================================
// An insight is what emerges when multiple fragments are combined.
// "User is a Rust developer building distributed systems who
//  prefers practical examples over theory" is an insight
//  synthesized from Fact + Expertise + Pattern + Style fragments.

/// Unique identifier for a synthesized insight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InsightId(pub u64);

impl InsightId {
    pub fn new(synthesis_round: u32, sequence: u32) -> Self {
        Self(((synthesis_round as u64) << 32) | sequence as u64)
    }
}

/// How many fragments contributed to this insight?
/// More sources = higher confidence in the synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthesisDepth(pub u8);

impl SynthesisDepth {
    pub fn confidence_multiplier(&self) -> f32 {
        match self.0 {
            0..=1 => 0.5,   // Single source: low confidence
            2..=3 => 0.75,  // Corroborated: moderate
            4..=7 => 0.9,   // Well-supported: high
            _ => 1.0,       // Deeply corroborated: maximum
        }
    }
}

pub const INSIGHT_CONTENT_SIZE: usize = 1024;

#[derive(Clone)]
pub struct InsightContent {
    data: [u8; INSIGHT_CONTENT_SIZE],
    len: u16,
}

impl InsightContent {
    pub fn new(text: &str) -> Self {
        let mut data = [0u8; INSIGHT_CONTENT_SIZE];
        let bytes = text.as_bytes();
        let len = bytes.len().min(INSIGHT_CONTENT_SIZE);
        data[..len].copy_from_slice(&bytes[..len]);
        Self { data, len: len as u16 }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("[corrupted]")
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for InsightContent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "InsightContent({:?})", self.as_str())
    }
}

/// Maximum number of source fragments an insight can reference
pub const MAX_INSIGHT_SOURCES: usize = 16;

/// A synthesized insight — emergent knowledge from combined fragments
#[derive(Clone, Debug)]
pub struct Insight {
    pub id: InsightId,
    pub content: InsightContent,
    pub source_fragments: [Option<FragmentId>; MAX_INSIGHT_SOURCES],
    pub source_count: u8,
    pub depth: SynthesisDepth,
    pub confidence: Confidence,
    pub created_at: u64,
    pub last_validated: u64,
    pub validation_count: u16,
    pub ihsan_at_synthesis: IhsanScore,
}

impl Insight {
    pub fn new(
        id: InsightId,
        content: &str,
        sources: &[FragmentId],
        timestamp: u64,
        ihsan: IhsanScore,
    ) -> Self {
        let mut source_fragments = [None; MAX_INSIGHT_SOURCES];
        let count = sources.len().min(MAX_INSIGHT_SOURCES);
        for (i, &fid) in sources[..count].iter().enumerate() {
            source_fragments[i] = Some(fid);
        }

        Self {
            id,
            content: InsightContent::new(content),
            source_fragments,
            source_count: count as u8,
            depth: SynthesisDepth(count as u8),
            confidence: Confidence::new(
                (Confidence::MEDIUM.raw() as f32
                    * SynthesisDepth(count as u8).confidence_multiplier()) as u16,
            ),
            created_at: timestamp,
            last_validated: timestamp,
            validation_count: 1,
            ihsan_at_synthesis: ihsan,
        }
    }

    /// Validate this insight (re-confirmed by new evidence)
    pub fn validate(&mut self, timestamp: u64) {
        self.last_validated = timestamp;
        self.validation_count = self.validation_count.saturating_add(1);
        // Increase confidence with each validation, capped at MAX
        let boost = 200u16.saturating_sub(self.validation_count * 10);
        self.confidence = Confidence::new(self.confidence.raw().saturating_add(boost));
    }

    pub fn source_fragments(&self) -> &[Option<FragmentId>] {
        &self.source_fragments[..self.source_count as usize]
    }
}

// ============================================================
// USER PROFILE — the synthesized understanding of who they are
// ============================================================
// This is what makes conversation 501 feel like talking to an
// old colleague. Not a list of facts — a coherent understanding.

pub const MAX_PROFILE_TRAITS: usize = 64;
pub const TRAIT_KEY_SIZE: usize = 64;
pub const TRAIT_VALUE_SIZE: usize = 256;

/// A single trait in the user profile
#[derive(Clone)]
pub struct ProfileTrait {
    key: [u8; TRAIT_KEY_SIZE],
    key_len: u8,
    value: [u8; TRAIT_VALUE_SIZE],
    value_len: u16,
    pub confidence: Confidence,
    pub source_insight: Option<InsightId>,
    pub last_updated: u64,
}

impl ProfileTrait {
    pub fn new(key: &str, value: &str, confidence: Confidence, timestamp: u64) -> Self {
        let mut k = [0u8; TRAIT_KEY_SIZE];
        let klen = key.as_bytes().len().min(TRAIT_KEY_SIZE);
        k[..klen].copy_from_slice(&key.as_bytes()[..klen]);

        let mut v = [0u8; TRAIT_VALUE_SIZE];
        let vlen = value.as_bytes().len().min(TRAIT_VALUE_SIZE);
        v[..vlen].copy_from_slice(&value.as_bytes()[..vlen]);

        Self {
            key: k,
            key_len: klen as u8,
            value: v,
            value_len: vlen as u16,
            confidence,
            source_insight: None,
            last_updated: timestamp,
        }
    }

    pub fn key(&self) -> &str {
        core::str::from_utf8(&self.key[..self.key_len as usize]).unwrap_or("")
    }

    pub fn value(&self) -> &str {
        core::str::from_utf8(&self.value[..self.value_len as usize]).unwrap_or("")
    }

    pub fn update_value(&mut self, value: &str, confidence: Confidence, timestamp: u64) {
        let vlen = value.as_bytes().len().min(TRAIT_VALUE_SIZE);
        self.value = [0u8; TRAIT_VALUE_SIZE];
        self.value[..vlen].copy_from_slice(&value.as_bytes()[..vlen]);
        self.value_len = vlen as u16;
        self.confidence = confidence;
        self.last_updated = timestamp;
    }
}

impl core::fmt::Debug for ProfileTrait {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ProfileTrait({}: {:?})", self.key(), self.value())
    }
}

/// User profile — the coherent picture of who this person is
pub struct UserProfile {
    traits: [Option<ProfileTrait>; MAX_PROFILE_TRAITS],
    trait_count: u8,
    pub last_synthesis: u64,
    pub synthesis_round: u32,
    pub overall_confidence: Confidence,
}

impl UserProfile {
    pub fn new() -> Self {
        Self {
            traits: core::array::from_fn(|_| None),
            trait_count: 0,
            last_synthesis: 0,
            synthesis_round: 0,
            overall_confidence: Confidence::new(0),
        }
    }

    /// Set or update a trait
    pub fn set_trait(&mut self, key: &str, value: &str, confidence: Confidence, timestamp: u64) -> bool {
        // Check if trait already exists (update it)
        for slot in self.traits.iter_mut() {
            if let Some(t) = slot {
                if t.key() == key {
                    t.update_value(value, confidence, timestamp);
                    return true;
                }
            }
        }
        // Add new trait
        if (self.trait_count as usize) < MAX_PROFILE_TRAITS {
            for slot in self.traits.iter_mut() {
                if slot.is_none() {
                    *slot = Some(ProfileTrait::new(key, value, confidence, timestamp));
                    self.trait_count += 1;
                    return true;
                }
            }
        }
        false // No space
    }

    /// Get a trait value by key
    pub fn get_trait(&self, key: &str) -> Option<&ProfileTrait> {
        self.traits.iter().filter_map(|t| t.as_ref()).find(|t| t.key() == key)
    }

    /// Iterate all active traits
    pub fn traits(&self) -> impl Iterator<Item = &ProfileTrait> {
        self.traits.iter().filter_map(|t| t.as_ref())
    }

    pub fn trait_count(&self) -> usize {
        self.trait_count as usize
    }

    /// Remove a trait by key
    pub fn remove_trait(&mut self, key: &str) -> bool {
        for slot in self.traits.iter_mut() {
            if let Some(t) = slot {
                if t.key() == key {
                    *slot = None;
                    self.trait_count = self.trait_count.saturating_sub(1);
                    return true;
                }
            }
        }
        false
    }

    /// Update synthesis metadata
    pub fn mark_synthesized(&mut self, timestamp: u64) {
        self.synthesis_round += 1;
        self.last_synthesis = timestamp;
        self.recalculate_confidence();
    }

    fn recalculate_confidence(&mut self) {
        if self.trait_count == 0 {
            self.overall_confidence = Confidence::new(0);
            return;
        }
        let total: u32 = self.traits.iter()
            .filter_map(|t| t.as_ref())
            .map(|t| t.confidence.raw() as u32)
            .sum();
        self.overall_confidence = Confidence::new(
            (total / self.trait_count as u32) as u16
        );
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TEMPORAL CONTEXT — when things happened and how they relate
// ============================================================

/// Session marker — represents one conversation session
#[derive(Debug, Clone, Copy)]
pub struct SessionMarker {
    pub session_id: u64,
    pub started_at: u64,
    pub ended_at: u64,
    pub fragment_count: u16,
    pub insight_count: u16,
    pub avg_ihsan: IhsanScore,
}

/// Maximum sessions tracked for temporal analysis
pub const MAX_SESSIONS: usize = 256;

/// Temporal context — tracks the time dimension of memory
pub struct TemporalContext {
    sessions: [Option<SessionMarker>; MAX_SESSIONS],
    session_count: u16,
    pub current_session: Option<u64>, // Active session ID
}

impl TemporalContext {
    pub fn new() -> Self {
        Self {
            sessions: [None; MAX_SESSIONS],
            session_count: 0,
            current_session: None,
        }
    }

    pub fn start_session(&mut self, session_id: u64, timestamp: u64) -> bool {
        self.current_session = Some(session_id);
        if (self.session_count as usize) < MAX_SESSIONS {
            let marker = SessionMarker {
                session_id,
                started_at: timestamp,
                ended_at: 0,
                fragment_count: 0,
                insight_count: 0,
                avg_ihsan: IhsanScore::new(9900), // Assume excellence until proven otherwise
            };
            // Find slot (use modular index for ring-buffer behavior)
            let idx = self.session_count as usize % MAX_SESSIONS;
            self.sessions[idx] = Some(marker);
            self.session_count += 1;
            true
        } else {
            // Overwrite oldest
            let idx = (self.session_count as usize) % MAX_SESSIONS;
            let marker = SessionMarker {
                session_id,
                started_at: timestamp,
                ended_at: 0,
                fragment_count: 0,
                insight_count: 0,
                avg_ihsan: IhsanScore::new(9900),
            };
            self.sessions[idx] = Some(marker);
            self.session_count += 1;
            true
        }
    }

    pub fn end_session(&mut self, session_id: u64, timestamp: u64) -> bool {
        for slot in self.sessions.iter_mut().rev() {
            if let Some(s) = slot {
                if s.session_id == session_id {
                    s.ended_at = timestamp;
                    self.current_session = None;
                    return true;
                }
            }
        }
        false
    }

    pub fn get_session(&self, session_id: u64) -> Option<&SessionMarker> {
        self.sessions.iter()
            .filter_map(|s| s.as_ref())
            .find(|s| s.session_id == session_id)
    }

    pub fn recent_sessions(&self, count: usize) -> impl Iterator<Item = &SessionMarker> {
        self.sessions.iter()
            .filter_map(|s| s.as_ref())
            .rev()
            .take(count)
    }

    pub fn total_sessions(&self) -> u16 {
        self.session_count
    }
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// SYNTHESIS METRICS — how well is the pipeline performing?
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct SynthesisMetrics {
    pub fragments_ingested: u64,
    pub fragments_below_threshold: u64,
    pub insights_produced: u64,
    pub profile_updates: u64,
    pub synthesis_rounds: u32,
    pub avg_confidence: Confidence,
    pub last_synthesis_duration_us: u64,
    pub ihsan_at_last_synthesis: IhsanScore,
}

impl SynthesisMetrics {
    pub fn new() -> Self {
        Self {
            fragments_ingested: 0,
            fragments_below_threshold: 0,
            insights_produced: 0,
            profile_updates: 0,
            synthesis_rounds: 0,
            avg_confidence: Confidence::new(0),
            last_synthesis_duration_us: 0,
            ihsan_at_last_synthesis: IhsanScore::new(9900),
        }
    }

    /// What percentage of fragments pass the confidence threshold?
    pub fn quality_ratio(&self) -> f32 {
        if self.fragments_ingested == 0 {
            return 1.0;
        }
        1.0 - (self.fragments_below_threshold as f32 / self.fragments_ingested as f32)
    }

    /// Insights per fragment — higher means better synthesis
    pub fn synthesis_efficiency(&self) -> f32 {
        if self.fragments_ingested == 0 {
            return 0.0;
        }
        self.insights_produced as f32 / self.fragments_ingested as f32
    }
}

impl Default for SynthesisMetrics {
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

    #[test]
    fn fragment_id_roundtrip() {
        let id = FragmentId::new(0xDEAD, 42);
        assert_eq!(id.conversation_hash(), 0xDEAD);
        assert_eq!(id.sequence(), 42);
    }

    #[test]
    fn fragment_content_stores_and_retrieves() {
        let content = FragmentContent::new("User prefers Rust over Python");
        assert_eq!(content.as_str(), "User prefers Rust over Python");
        assert_eq!(content.len(), 29);
        assert!(!content.is_empty());
    }

    #[test]
    fn fragment_content_truncates_long_text() {
        let long_text = "x".repeat(1000);
        let content = FragmentContent::new(&long_text);
        assert_eq!(content.len(), FRAGMENT_CONTENT_SIZE);
    }

    #[test]
    fn confidence_threshold() {
        assert!(Confidence::HIGH.meets_threshold());
        assert!(Confidence::MEDIUM.meets_threshold());
        assert!(!Confidence::new(5000).meets_threshold());
        assert!(!Confidence::new(0).meets_threshold());
    }

    #[test]
    fn fragment_synthesis_weight_decays_over_time() {
        let source = ComponentId::new("test", "1.0");
        let fragment = MemoryFragment::new(
            FragmentId::new(1, 1),
            FragmentKind::Preference,
            "Likes Rust",
            Confidence::HIGH,
            source,
            1000,
            IhsanScore::new(9900),
        );

        let weight_fresh = fragment.synthesis_weight(1000);
        let weight_1day = fragment.synthesis_weight(1000 + 86400);
        let weight_30day = fragment.synthesis_weight(1000 + 86400 * 30);

        assert!(weight_fresh > weight_1day);
        assert!(weight_1day > weight_30day);
    }

    #[test]
    fn fragment_reinforcement_slows_decay() {
        let source = ComponentId::new("test", "1.0");
        let mut fragment = MemoryFragment::new(
            FragmentId::new(1, 1),
            FragmentKind::Pattern,
            "Always asks for examples",
            Confidence::HIGH,
            source,
            1000,
            IhsanScore::new(9900),
        );

        let initial_decay = fragment.decay_rate;
        fragment.reinforce(2000);
        assert!(fragment.decay_rate < initial_decay);
        assert_eq!(fragment.reinforcement_count, 2);
        assert_eq!(fragment.last_reinforced, 2000);
    }

    #[test]
    fn fragment_supersession() {
        let source = ComponentId::new("test", "1.0");
        let mut fragment = MemoryFragment::new(
            FragmentId::new(1, 1),
            FragmentKind::Fact,
            "Works at Company A",
            Confidence::HIGH,
            source,
            1000,
            IhsanScore::new(9900),
        );

        assert!(fragment.is_active());
        let new_id = FragmentId::new(2, 1);
        fragment.supersede(new_id);
        assert!(!fragment.is_active());
        assert_eq!(fragment.superseded_by, Some(new_id));
    }

    #[test]
    fn insight_creation_from_fragments() {
        let sources = [
            FragmentId::new(1, 1),
            FragmentId::new(1, 2),
            FragmentId::new(2, 1),
        ];
        let insight = Insight::new(
            InsightId::new(1, 1),
            "User is a Rust developer building distributed systems",
            &sources,
            1000,
            IhsanScore::new(9900),
        );

        assert_eq!(insight.source_count, 3);
        assert_eq!(insight.depth.0, 3);
        assert!(insight.confidence.raw() > Confidence::LOW.raw());
    }

    #[test]
    fn insight_validation_increases_confidence() {
        let insight_sources = [FragmentId::new(1, 1)];
        let mut insight = Insight::new(
            InsightId::new(1, 1),
            "Prefers practical examples",
            &insight_sources,
            1000,
            IhsanScore::new(9900),
        );

        let initial_confidence = insight.confidence.raw();
        insight.validate(2000);
        assert!(insight.confidence.raw() > initial_confidence);
        assert_eq!(insight.validation_count, 2);
    }

    #[test]
    fn user_profile_set_and_get_traits() {
        let mut profile = UserProfile::new();
        let ts = 1000u64;

        assert!(profile.set_trait("language", "Rust", Confidence::HIGH, ts));
        assert!(profile.set_trait("role", "architect", Confidence::MEDIUM, ts));

        let lang = profile.get_trait("language").unwrap();
        assert_eq!(lang.value(), "Rust");
        assert_eq!(profile.trait_count(), 2);
    }

    #[test]
    fn user_profile_updates_existing_trait() {
        let mut profile = UserProfile::new();

        profile.set_trait("company", "OldCo", Confidence::MEDIUM, 1000);
        profile.set_trait("company", "NewCo", Confidence::HIGH, 2000);

        assert_eq!(profile.trait_count(), 1); // Same key = update, not add
        assert_eq!(profile.get_trait("company").unwrap().value(), "NewCo");
    }

    #[test]
    fn user_profile_removes_trait() {
        let mut profile = UserProfile::new();
        profile.set_trait("temp", "value", Confidence::LOW, 1000);
        assert_eq!(profile.trait_count(), 1);

        assert!(profile.remove_trait("temp"));
        assert_eq!(profile.trait_count(), 0);
        assert!(profile.get_trait("temp").is_none());
    }

    #[test]
    fn temporal_context_session_lifecycle() {
        let mut ctx = TemporalContext::new();

        assert!(ctx.start_session(100, 1000));
        assert_eq!(ctx.current_session, Some(100));

        assert!(ctx.end_session(100, 2000));
        assert_eq!(ctx.current_session, None);

        let session = ctx.get_session(100).unwrap();
        assert_eq!(session.started_at, 1000);
        assert_eq!(session.ended_at, 2000);
    }

    #[test]
    fn synthesis_metrics_quality_ratio() {
        let mut metrics = SynthesisMetrics::new();
        metrics.fragments_ingested = 100;
        metrics.fragments_below_threshold = 15;

        assert!((metrics.quality_ratio() - 0.85).abs() < 0.001);
    }

    #[test]
    fn synthesis_metrics_efficiency() {
        let mut metrics = SynthesisMetrics::new();
        metrics.fragments_ingested = 50;
        metrics.insights_produced = 10;

        assert!((metrics.synthesis_efficiency() - 0.2).abs() < 0.001);
    }

    #[test]
    fn fragment_kind_weights_are_ordered() {
        // Patterns should be highest weight (implicit knowledge is gold)
        assert!(FragmentKind::Pattern.synthesis_weight() > FragmentKind::Fact.synthesis_weight());
        // Style matters more than temporal
        assert!(FragmentKind::Style.synthesis_weight() > FragmentKind::Temporal.synthesis_weight());
    }
}
