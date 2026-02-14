//! Comprehensive tests for pattern_memory — Pattern, MemoryPatternStore, PatternMemory
//!
//! Phase 13: Test Sprint — Closing coverage gap on autopoiesis

use bizra_autopoiesis::pattern_memory::*;
use bizra_autopoiesis::{ELEVATION_THRESHOLD, EMBEDDING_DIM};
use bizra_core::NodeId;

// ---------------------------------------------------------------------------
// Pattern
// ---------------------------------------------------------------------------

#[test]
fn pattern_new_generates_unique_id() {
    let node = NodeId("test_node".into());
    let p1 = Pattern::new("hello".into(), vec![0.1; 384], node.clone());
    let p2 = Pattern::new("world".into(), vec![0.2; 384], node);
    assert!(p1.id.starts_with("pat_"));
    assert_ne!(p1.id, p2.id);
}

#[test]
fn pattern_new_defaults() {
    let node = NodeId("n".into());
    let p = Pattern::new("content".into(), vec![1.0; 10], node.clone());
    assert_eq!(p.content, "content");
    assert_eq!(p.source_node, node);
    assert_eq!(p.access_count, 1);
    assert!((p.confidence - 0.5).abs() < f64::EPSILON);
    assert!(p.tags.is_empty());
    assert!(!p.is_elevated);
}

#[test]
fn pattern_update_confidence_positive() {
    let node = NodeId("n".into());
    let mut p = Pattern::new("x".into(), vec![], node);
    assert!((p.confidence - 0.5).abs() < f64::EPSILON);
    p.update_confidence(true);
    assert!((p.confidence - 0.55).abs() < f64::EPSILON);
}

#[test]
fn pattern_update_confidence_negative() {
    let node = NodeId("n".into());
    let mut p = Pattern::new("x".into(), vec![], node);
    p.update_confidence(false);
    assert!((p.confidence - 0.47).abs() < f64::EPSILON);
}

#[test]
fn pattern_confidence_clamps_at_1() {
    let node = NodeId("n".into());
    let mut p = Pattern::new("x".into(), vec![], node);
    p.confidence = 0.98;
    p.update_confidence(true); // 0.98 + 0.05 = 1.03 → clamped to 1.0
    assert!((p.confidence - 1.0).abs() < f64::EPSILON);
}

#[test]
fn pattern_confidence_clamps_at_0() {
    let node = NodeId("n".into());
    let mut p = Pattern::new("x".into(), vec![], node);
    p.confidence = 0.01;
    p.update_confidence(false); // 0.01 - 0.03 = -0.02 → clamped to 0.0
    assert!(p.confidence >= 0.0);
}

// ---------------------------------------------------------------------------
// MemoryPatternStore
// ---------------------------------------------------------------------------

#[test]
fn memory_store_new_empty() {
    let store = MemoryPatternStore::new();
    assert_eq!(store.count(), 0);
}

#[test]
fn memory_store_default_same_as_new() {
    let store = MemoryPatternStore::default();
    assert_eq!(store.count(), 0);
}

#[test]
fn memory_store_store_and_get() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p = Pattern::new("test_content".into(), vec![0.5; 10], node);
    let id = p.id.clone();
    store.store(p).unwrap();
    assert_eq!(store.count(), 1);
    let retrieved = store.get(&id).unwrap();
    assert_eq!(retrieved.content, "test_content");
}

#[test]
fn memory_store_get_missing_returns_none() {
    let store = MemoryPatternStore::new();
    assert!(store.get("nonexistent").is_none());
}

#[test]
fn memory_store_search_similar_identical_vectors() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let embedding = vec![1.0, 0.0, 0.0];
    let p = Pattern::new("exact_match".into(), embedding.clone(), node);
    store.store(p).unwrap();

    let results = store.search_similar(&embedding, 10, 0.9);
    assert_eq!(results.len(), 1);
    assert!((results[0].1 - 1.0).abs() < 0.001); // cosine similarity ≈ 1.0
}

#[test]
fn memory_store_search_similar_orthogonal_vectors() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p = Pattern::new("orthogonal".into(), vec![1.0, 0.0, 0.0], node);
    store.store(p).unwrap();

    let query = vec![0.0, 1.0, 0.0]; // orthogonal
    let results = store.search_similar(&query, 10, 0.1);
    assert!(results.is_empty()); // cosine similarity = 0.0, below threshold
}

#[test]
fn memory_store_search_similar_respects_threshold() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    // Two patterns: one similar, one less similar
    let p1 = Pattern::new("similar".into(), vec![1.0, 0.1, 0.0], node.clone());
    let p2 = Pattern::new("different".into(), vec![0.1, 1.0, 0.0], node);
    store.store(p1).unwrap();
    store.store(p2).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search_similar(&query, 10, 0.9);
    // Only the very similar one should pass 0.9 threshold
    assert!(results.len() <= 1);
}

#[test]
fn memory_store_search_similar_respects_limit() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    for i in 0..10 {
        let emb = vec![1.0, 0.1 * i as f32, 0.0];
        let p = Pattern::new(format!("p{}", i), emb, node.clone());
        store.store(p).unwrap();
    }

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search_similar(&query, 3, 0.0);
    assert!(results.len() <= 3);
}

#[test]
fn memory_store_search_similar_sorted_descending() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p1 = Pattern::new("close".into(), vec![1.0, 0.1, 0.0], node.clone());
    let p2 = Pattern::new("closer".into(), vec![1.0, 0.01, 0.0], node.clone());
    let p3 = Pattern::new("far".into(), vec![0.3, 0.9, 0.1], node);
    store.store(p1).unwrap();
    store.store(p2).unwrap();
    store.store(p3).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search_similar(&query, 10, 0.0);
    // Verify descending order
    for w in results.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

// ---------------------------------------------------------------------------
// PatternMemory
// ---------------------------------------------------------------------------

#[test]
fn pattern_memory_in_memory_starts_empty() {
    let mem = PatternMemory::in_memory(NodeId("node".into()));
    assert_eq!(mem.count(), 0);
}

#[test]
fn pattern_memory_learn_increases_count() {
    let mut mem = PatternMemory::in_memory(NodeId("learner".into()));
    let id = mem
        .learn(
            "test data".into(),
            vec![0.1; EMBEDDING_DIM],
            vec!["tag1".into()],
        )
        .unwrap();
    assert!(id.starts_with("pat_"));
    assert_eq!(mem.count(), 1);
}

#[test]
fn pattern_memory_learn_multiple() {
    let mut mem = PatternMemory::in_memory(NodeId("multi".into()));
    for i in 0..5 {
        mem.learn(format!("item_{}", i), vec![0.1 * i as f32; 10], vec![])
            .unwrap();
    }
    assert_eq!(mem.count(), 5);
}

#[test]
fn pattern_memory_recall_returns_similar() {
    let mut mem = PatternMemory::in_memory(NodeId("recaller".into()));
    let emb = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    mem.learn("target".into(), emb.clone(), vec!["test".into()])
        .unwrap();
    mem.learn("noise".into(), vec![0.0, 0.0, 0.0, 0.0, 1.0], vec![])
        .unwrap();

    let results = mem.recall(&emb, 10);
    // "target" should be in results (cosine sim = 1.0 > 0.5 threshold)
    assert!(!results.is_empty());
    assert!(results.iter().any(|p| p.content == "target"));
}

#[test]
fn pattern_memory_recall_respects_limit() {
    let mut mem = PatternMemory::in_memory(NodeId("limiter".into()));
    // Store many similar patterns
    for i in 0..20 {
        let emb = vec![1.0, 0.01 * i as f32, 0.0];
        mem.learn(format!("item_{}", i), emb, vec![]).unwrap();
    }
    let results = mem.recall(&[1.0, 0.0, 0.0], 5);
    assert!(results.len() <= 5);
}

#[test]
fn pattern_memory_recall_empty_store_returns_empty() {
    let mem = PatternMemory::in_memory(NodeId("empty".into()));
    let results = mem.recall(&[1.0, 0.0, 0.0], 10);
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// Cosine similarity edge cases (tested via search_similar)
// ---------------------------------------------------------------------------

#[test]
fn search_handles_zero_vectors() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p = Pattern::new("zero".into(), vec![0.0, 0.0, 0.0], node);
    store.store(p).unwrap();
    let results = store.search_similar(&[1.0, 0.0, 0.0], 10, 0.0);
    // Zero vector has 0 norm → cosine sim = 0 → won't meet positive threshold usually
    // but with threshold 0.0 it would pass if sim >= 0.0 (which 0.0 is)
    // Actually the function returns 0.0 for zero vectors
    assert!(results.is_empty() || results[0].1.abs() < 0.001);
}

#[test]
fn search_handles_different_length_vectors() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p = Pattern::new("short".into(), vec![1.0, 0.0], node);
    store.store(p).unwrap();
    // Query with different length → cosine_similarity returns 0.0
    let results = store.search_similar(&[1.0, 0.0, 0.0], 10, 0.0);
    assert!(results.is_empty() || results[0].1.abs() < 0.001);
}

#[test]
fn search_handles_empty_vectors() {
    let mut store = MemoryPatternStore::new();
    let node = NodeId("n".into());
    let p = Pattern::new("empty".into(), vec![], node);
    store.store(p).unwrap();
    let results = store.search_similar(&[], 10, 0.0);
    assert!(results.is_empty() || results[0].1.abs() < 0.001);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn constants_valid() {
    assert_eq!(EMBEDDING_DIM, 384);
    assert!((ELEVATION_THRESHOLD - 0.95).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// PatternError display
// ---------------------------------------------------------------------------

#[test]
fn pattern_error_display() {
    let e1 = PatternError::NotFound("x".into());
    assert!(e1.to_string().contains("x"));
    let e2 = PatternError::Storage("disk".into());
    assert!(e2.to_string().contains("disk"));
}
