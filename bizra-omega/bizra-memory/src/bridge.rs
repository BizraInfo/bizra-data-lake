//! # Bridge — FFI Contract for Python Engine Integration
//!
//! BIZRA's existing AI engines (CognitiveResonance, VectorSearch, HMM)
//! are Python-based. This bridge defines the FFI contract so:
//!
//! 1. Rust pipeline calls Python for LLM-grade extraction
//! 2. Rust pipeline calls Python for semantic search/indexing
//! 3. Python engines call Rust for high-speed event routing
//!
//! ## Architecture
//! ```text
//! ┌──────────────────┐     FFI      ┌──────────────────────┐
//! │  Rust Pipeline   │ ←──────────→ │  Python Engines      │
//! │  (Orchestrator)  │  C-ABI calls │  CognitiveResonance  │
//! │  types, store,   │              │  VectorSearch         │
//! │  synthesis,      │              │  HMM Caller           │
//! │  routing         │              │  LLM Extraction       │
//! └──────────────────┘              └──────────────────────┘
//! ```
//!
//! ## Design
//! - Trait-based: `ExternalExtractor` and `ExternalSearcher` traits
//! - Default implementations are rule-based (Node0 bootstrap)
//! - FFI implementations wrap Python calls (production)
//! - All FFI uses fixed-size buffers (no heap across boundary)

use crate::types::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Extraction Trait — How atoms are pulled from fragments
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Maximum atoms a single extraction call can produce.
pub const MAX_EXTRACTION_RESULTS: usize = 32;

/// A single extraction result from an external engine.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub kind: AtomKind,
    pub content: ExtractionContent,
    pub confidence: f32,
}

/// Fixed-capacity content buffer for extraction results.
/// Avoids heap allocation across FFI boundary.
#[derive(Clone)]
pub struct ExtractionContent {
    buf: [u8; 512],
    len: usize,
}

impl ExtractionContent {
    pub fn new(s: &str) -> Self {
        let mut buf = [0u8; 512];
        let len = s.len().min(512);
        buf[..len].copy_from_slice(&s.as_bytes()[..len]);
        ExtractionContent { buf, len }
    }

    pub fn as_str(&self) -> &str {
        let slice = &self.buf[..self.len];
        debug_assert!(
            core::str::from_utf8(slice).is_ok(),
            "UTF-8 invariant violated in ExtractionContent"
        );
        // Safety: constructors validate UTF-8; debug_assert guards against regression
        unsafe { core::str::from_utf8_unchecked(slice) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for ExtractionContent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "\"{}\"", self.as_str())
    }
}

/// Batch extraction results.
pub struct ExtractionBatch {
    results: [Option<ExtractionResult>; MAX_EXTRACTION_RESULTS],
    count: usize,
}

impl ExtractionBatch {
    pub fn new() -> Self {
        ExtractionBatch {
            results: Default::default(),
            count: 0,
        }
    }

    pub fn push(&mut self, result: ExtractionResult) -> bool {
        if self.count < MAX_EXTRACTION_RESULTS {
            self.results[self.count] = Some(result);
            self.count += 1;
            true
        } else {
            false
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &ExtractionResult> {
        self.results[..self.count].iter().filter_map(|r| r.as_ref())
    }

    pub fn count(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for ExtractionBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for external extraction engines.
///
/// Default impl: rule-based (what pipeline.rs uses today).
/// Production impl: calls CognitiveResonance via FFI for LLM extraction.
pub trait Extractor {
    /// Extract atoms from a text fragment.
    fn extract(&self, content: &str, kind: FragmentKind) -> ExtractionBatch;

    /// Engine name for provenance tracking.
    fn engine_name(&self) -> &str;
}

/// Rule-based extractor — the bootstrap implementation.
/// Same logic as pipeline.rs rule_extract but exposed as trait.
pub struct RuleExtractor;

impl Extractor for RuleExtractor {
    fn extract(&self, content: &str, _kind: FragmentKind) -> ExtractionBatch {
        let mut batch = ExtractionBatch::new();
        let lower = content.to_lowercase();

        // Identity patterns
        if lower.contains("i am ") || lower.contains("i'm ") {
            batch.push(ExtractionResult {
                kind: AtomKind::Fact,
                content: ExtractionContent::new(content),
                confidence: 0.95,
            });
        }

        // Preference patterns
        if lower.contains("i prefer")
            || lower.contains("i like")
            || lower.contains("i want")
            || lower.contains("i need")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Preference,
                content: ExtractionContent::new(content),
                confidence: 0.90,
            });
        }

        // Goal patterns
        if lower.contains("working on")
            || lower.contains("building")
            || lower.contains("preparing")
            || lower.contains("planning")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Goal,
                content: ExtractionContent::new(content),
                confidence: 0.70,
            });
        }

        // Negation patterns
        if lower.contains("i don't")
            || lower.contains("i never")
            || lower.contains("i do not")
            || lower.contains("not interested")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Negation,
                content: ExtractionContent::new(content),
                confidence: 0.90,
            });
        }

        // Behavioral patterns
        if lower.contains("every day")
            || lower.contains("usually")
            || lower.contains("always")
            || lower.contains("every morning")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Pattern,
                content: ExtractionContent::new(content),
                confidence: 0.70,
            });
        }

        // Expertise patterns
        if lower.contains("i know")
            || lower.contains("i understand")
            || lower.contains("expert")
            || lower.contains("experienced")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Expertise,
                content: ExtractionContent::new(content),
                confidence: 0.80,
            });
        }

        // Principle patterns
        if lower.contains("principle")
            || lower.contains("believe")
            || lower.contains("value")
            || lower.contains("إحسان")
            || lower.contains("ihsan")
        {
            batch.push(ExtractionResult {
                kind: AtomKind::Principle,
                content: ExtractionContent::new(content),
                confidence: 0.85,
            });
        }

        // Fallback: context
        if batch.is_empty() && content.len() > 20 {
            batch.push(ExtractionResult {
                kind: AtomKind::Context,
                content: ExtractionContent::new(content),
                confidence: 0.40,
            });
        }

        batch
    }

    fn engine_name(&self) -> &str {
        "rule-extractor-v1"
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Search Trait — How knowledge is retrieved
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Maximum results from a search query.
pub const MAX_SEARCH_RESULTS: usize = 16;

/// A search result with relevance scoring.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub content: ExtractionContent,
    pub relevance: f32,
    pub confidence: f32,
    pub kind: QueryResultKind,
}

/// Batch search results.
pub struct SearchBatch {
    results: Vec<SearchResult>,
}

impl SearchBatch {
    pub fn new() -> Self {
        SearchBatch {
            results: Vec::with_capacity(MAX_SEARCH_RESULTS),
        }
    }

    pub fn push(&mut self, result: SearchResult) {
        if self.results.len() < MAX_SEARCH_RESULTS {
            self.results.push(result);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SearchResult> {
        self.results.iter()
    }

    pub fn count(&self) -> usize {
        self.results.len()
    }
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Sort by relevance descending.
    pub fn sort_by_relevance(&mut self) {
        self.results.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }
}

impl Default for SearchBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for external search/retrieval engines.
///
/// Default impl: linear scan (Node0 bootstrap).
/// Production impl: calls VectorSearch via FFI for semantic retrieval.
pub trait Searcher {
    /// Search knowledge base for relevant content.
    fn search(&self, query: &str, max_results: usize) -> SearchBatch;

    /// Engine name for telemetry.
    fn engine_name(&self) -> &str;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FFI Buffer Types — C-ABI compatible for Python bridge
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// C-ABI compatible extraction result for FFI.
#[repr(C)]
pub struct FfiExtractionResult {
    pub kind: u8, // AtomKind as u8
    pub confidence: f32,
    pub content_ptr: *const u8,
    pub content_len: u32,
}

/// C-ABI compatible search result for FFI.
#[repr(C)]
pub struct FfiSearchResult {
    pub relevance: f32,
    pub confidence: f32,
    pub kind: u8, // QueryResultKind as u8
    pub content_ptr: *const u8,
    pub content_len: u32,
}

/// C-ABI compatible batch header for FFI.
#[repr(C)]
pub struct FfiBatchHeader {
    pub count: u32,
    pub success: u8,     // 0 = error, 1 = ok
    pub error_code: u32, // 0 = none
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bridge Status — Health of the Python connection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Status of the Python engine bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeStatus {
    /// No Python engine connected — using rule-based fallback
    Disconnected,
    /// Python engine connected and healthy
    Connected,
    /// Python engine connected but degraded (high latency or errors)
    Degraded,
    /// Python engine connection failed
    Failed,
}

/// Bridge health snapshot.
#[derive(Debug, Clone, Copy)]
pub struct BridgeHealth {
    pub status: BridgeStatus,
    pub extractor_engine: BridgeStatus,
    pub searcher_engine: BridgeStatus,
    pub calls_made: u64,
    pub calls_failed: u64,
    pub avg_latency_us: u64,
}

impl BridgeHealth {
    pub fn disconnected() -> Self {
        BridgeHealth {
            status: BridgeStatus::Disconnected,
            extractor_engine: BridgeStatus::Disconnected,
            searcher_engine: BridgeStatus::Disconnected,
            calls_made: 0,
            calls_failed: 0,
            avg_latency_us: 0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.calls_made == 0 {
            return 1.0;
        }
        1.0 - (self.calls_failed as f64 / self.calls_made as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_extractor_identity() {
        let extractor = RuleExtractor;
        let batch = extractor.extract("I am the founder of BIZRA", FragmentKind::UserMessage);

        assert!(batch.count() >= 1);
        let result = batch.iter().next().unwrap();
        assert_eq!(result.kind, AtomKind::Fact);
        assert!(result.confidence >= 0.90);
    }

    #[test]
    fn rule_extractor_preference() {
        let extractor = RuleExtractor;
        let batch = extractor.extract(
            "I prefer Rust over Python for core",
            FragmentKind::UserMessage,
        );

        let pref = batch.iter().find(|r| r.kind == AtomKind::Preference);
        assert!(pref.is_some());
    }

    #[test]
    fn rule_extractor_multi_pattern() {
        let extractor = RuleExtractor;
        let batch = extractor.extract(
            "I am Mumo and I prefer sovereign architecture and I'm building BIZRA every day",
            FragmentKind::UserMessage,
        );

        // Should detect: Fact + Preference + Goal + Pattern
        assert!(batch.count() >= 3);
    }

    #[test]
    fn rule_extractor_negation() {
        let extractor = RuleExtractor;
        let batch = extractor.extract(
            "I don't want centralized dependencies",
            FragmentKind::UserMessage,
        );

        let neg = batch.iter().find(|r| r.kind == AtomKind::Negation);
        assert!(neg.is_some());
    }

    #[test]
    fn rule_extractor_principle() {
        let extractor = RuleExtractor;
        let batch = extractor.extract(
            "My core principle is إحسان excellence in everything",
            FragmentKind::UserMessage,
        );

        let prin = batch.iter().find(|r| r.kind == AtomKind::Principle);
        assert!(prin.is_some());
    }

    #[test]
    fn rule_extractor_fallback_context() {
        let extractor = RuleExtractor;
        let batch = extractor.extract(
            "The architecture uses a HyperBlockTree with dual consensus",
            FragmentKind::UserMessage,
        );

        // No specific patterns → falls back to Context
        assert_eq!(batch.count(), 1);
        let ctx = batch.iter().next().unwrap();
        assert_eq!(ctx.kind, AtomKind::Context);
    }

    #[test]
    fn extraction_content_fixed_buffer() {
        let short = ExtractionContent::new("hello");
        assert_eq!(short.as_str(), "hello");
        assert_eq!(short.len(), 5);

        // Long content gets truncated at 512
        let long_str = "x".repeat(600);
        let long = ExtractionContent::new(&long_str);
        assert_eq!(long.len(), 512);
    }

    #[test]
    fn extraction_batch_capacity() {
        let mut batch = ExtractionBatch::new();
        for i in 0..MAX_EXTRACTION_RESULTS {
            assert!(batch.push(ExtractionResult {
                kind: AtomKind::Fact,
                content: ExtractionContent::new(&format!("fact {}", i)),
                confidence: 0.90,
            }));
        }

        // 33rd should fail
        assert!(!batch.push(ExtractionResult {
            kind: AtomKind::Fact,
            content: ExtractionContent::new("overflow"),
            confidence: 0.90,
        }));
        assert_eq!(batch.count(), MAX_EXTRACTION_RESULTS);
    }

    #[test]
    fn bridge_health_tracking() {
        let health = BridgeHealth::disconnected();
        assert_eq!(health.status, BridgeStatus::Disconnected);
        assert_eq!(health.success_rate(), 1.0); // no calls = 100%

        let healthy = BridgeHealth {
            status: BridgeStatus::Connected,
            extractor_engine: BridgeStatus::Connected,
            searcher_engine: BridgeStatus::Connected,
            calls_made: 100,
            calls_failed: 3,
            avg_latency_us: 1500,
        };
        assert!(healthy.success_rate() > 0.96);
    }

    #[test]
    fn search_batch_sort() {
        let mut batch = SearchBatch::new();
        batch.push(SearchResult {
            content: ExtractionContent::new("low relevance"),
            relevance: 0.3,
            confidence: 0.9,
            kind: QueryResultKind::Atom,
        });
        batch.push(SearchResult {
            content: ExtractionContent::new("high relevance"),
            relevance: 0.95,
            confidence: 0.8,
            kind: QueryResultKind::Insight,
        });
        batch.push(SearchResult {
            content: ExtractionContent::new("mid relevance"),
            relevance: 0.6,
            confidence: 0.85,
            kind: QueryResultKind::Atom,
        });

        batch.sort_by_relevance();
        let results: Vec<f32> = batch.iter().map(|r| r.relevance).collect();
        assert!(results[0] > results[1] && results[1] > results[2]);
    }
}
