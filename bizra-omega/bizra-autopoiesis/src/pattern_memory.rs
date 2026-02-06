//! Pattern Memory — Persistent storage with vector similarity

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bizra_core::NodeId;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub source_node: NodeId,
    pub created_at: DateTime<Utc>,
    pub access_count: u64,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub is_elevated: bool,
}

impl Pattern {
    pub fn new(content: String, embedding: Vec<f32>, source: NodeId) -> Self {
        Self {
            id: format!("pat_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..16].to_string()),
            content, embedding, source_node: source,
            created_at: Utc::now(), access_count: 1, confidence: 0.5,
            tags: Vec::new(), is_elevated: false,
        }
    }

    pub fn update_confidence(&mut self, positive: bool) {
        let delta = if positive { 0.05 } else { -0.03 };
        self.confidence = (self.confidence + delta).clamp(0.0, 1.0);
    }
}

pub trait PatternStore: Send + Sync {
    fn store(&mut self, pattern: Pattern) -> Result<(), PatternError>;
    fn get(&self, id: &str) -> Option<Pattern>;
    fn search_similar(&self, embedding: &[f32], limit: usize, threshold: f32) -> Vec<(Pattern, f32)>;
    fn count(&self) -> usize;
}

pub struct MemoryPatternStore {
    patterns: HashMap<String, Pattern>,
}

impl MemoryPatternStore {
    pub fn new() -> Self { Self { patterns: HashMap::new() } }
}

impl Default for MemoryPatternStore {
    fn default() -> Self { Self::new() }
}

impl PatternStore for MemoryPatternStore {
    fn store(&mut self, pattern: Pattern) -> Result<(), PatternError> {
        self.patterns.insert(pattern.id.clone(), pattern);
        Ok(())
    }

    fn get(&self, id: &str) -> Option<Pattern> {
        self.patterns.get(id).cloned()
    }

    fn search_similar(&self, embedding: &[f32], limit: usize, threshold: f32) -> Vec<(Pattern, f32)> {
        let mut results: Vec<_> = self.patterns.values()
            .map(|p| (p.clone(), cosine_similarity(embedding, &p.embedding)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        results
    }

    fn count(&self) -> usize { self.patterns.len() }
}

pub struct PatternMemory {
    store: Box<dyn PatternStore>,
    node_id: NodeId,
}

impl PatternMemory {
    pub fn new(store: Box<dyn PatternStore>, node_id: NodeId) -> Self {
        Self { store, node_id }
    }

    pub fn in_memory(node_id: NodeId) -> Self {
        Self::new(Box::new(MemoryPatternStore::new()), node_id)
    }

    pub fn learn(&mut self, content: String, embedding: Vec<f32>, tags: Vec<String>) -> Result<String, PatternError> {
        let mut pattern = Pattern::new(content, embedding, self.node_id.clone());
        pattern.tags = tags;
        let id = pattern.id.clone();
        self.store.store(pattern)?;
        Ok(id)
    }

    pub fn recall(&self, embedding: &[f32], limit: usize) -> Vec<Pattern> {
        self.store.search_similar(embedding, limit, 0.5).into_iter().map(|(p, _)| p).collect()
    }

    pub fn count(&self) -> usize { self.store.count() }
}

#[derive(Debug, thiserror::Error)]
pub enum PatternError {
    #[error("Pattern not found: {0}")]
    NotFound(String),
    #[error("Storage error: {0}")]
    Storage(String),
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_memory() {
        let node_id = NodeId("test123456789abc".into());
        let mut memory = PatternMemory::in_memory(node_id);
        let id = memory.learn("test".into(), vec![0.1; 384], vec![]).unwrap();
        assert!(id.starts_with("pat_"));
        assert_eq!(memory.count(), 1);
    }
}
