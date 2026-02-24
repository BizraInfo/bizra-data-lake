//! # Reflex Ledger — The Symbolic-Neural Bridge
//!
//! When the node encounters a task:
//!   1. Hash the task signature
//!   2. Check reflex ledger for cached plan
//!   3. IF cached AND fresh → symbolic execution (System 1, zero LLM cost)
//!   4. ELSE → neural reasoning (System 2, LLM inference)
//!   5. IF neural result achieves Iḥsān ≥ 0.95 → compile as new reflex
//!
//! Over time: cost DECREASES, speed INCREASES. Reverse scaling per node.
//!
//! ## Standing on Giants
//! - **Kahneman (2011)**: System 1 (fast) vs System 2 (slow, deliberate)
//! - **Anderson (1982)**: ACT-R — skill compilation, declarative → procedural

use crate::receipt::content_hash;
use crate::types::*;

/// A compiled reflex — cached plan for a known task pattern.
#[derive(Debug, Clone)]
pub struct Reflex {
    pub task_hash: [u8; 32],
    pub description: String,
    pub actions: Vec<BizraAction>,
    pub compiled_ihsan: IhsanScore,
    pub execution_count: u64,
    pub last_used: ActionTimestamp,
    pub compiled_at: ActionTimestamp,
    pub avg_duration_ns: u64,
    pub marketplace_eligible: bool,
}

impl Reflex {
    /// Is this reflex stale?
    pub fn is_stale(&self, now: ActionTimestamp, staleness_ns: u64) -> bool {
        now.0.saturating_sub(self.last_used.0) > staleness_ns
    }

    /// Update stats after successful execution.
    pub fn record_execution(&mut self, timestamp: ActionTimestamp, duration_ns: u64) {
        self.execution_count += 1;
        self.last_used = timestamp;
        if self.execution_count == 1 {
            self.avg_duration_ns = duration_ns;
        } else {
            // Exponential moving average (7/8 old + 1/8 new)
            self.avg_duration_ns = (self.avg_duration_ns * 7 + duration_ns) / 8;
        }
    }

    /// Compute saved vs LLM inference (~2s per call).
    pub fn compute_saved_ns(&self) -> u64 {
        let llm_cost = 2_000_000_000u64;
        self.execution_count * llm_cost.saturating_sub(self.avg_duration_ns)
    }
}

/// The Reflex Ledger — hash-indexed compiled skill store.
pub struct ReflexLedger {
    entries: Vec<Reflex>,
    capacity: usize,
    staleness_threshold_ns: u64,
    total_lookups: u64,
    hits: u64,
}

impl ReflexLedger {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            staleness_threshold_ns: 7 * 24 * 3600 * 1_000_000_000, // 7 days
            total_lookups: 0,
            hits: 0,
        }
    }

    /// Hash a task description into a lookup key.
    pub fn task_signature(description: &str) -> [u8; 32] {
        content_hash(description.as_bytes())
    }

    /// Look up a reflex by task hash. Returns None if not found or stale.
    pub fn lookup(&mut self, task_hash: &[u8; 32], now: ActionTimestamp) -> Option<&Reflex> {
        self.total_lookups += 1;
        let staleness = self.staleness_threshold_ns;
        let found = self
            .entries
            .iter()
            .find(|r| r.task_hash == *task_hash && !r.is_stale(now, staleness));
        if found.is_some() {
            self.hits += 1;
        }
        found
    }

    /// Look up mutably (for recording execution).
    pub fn lookup_mut(
        &mut self,
        task_hash: &[u8; 32],
        now: ActionTimestamp,
    ) -> Option<&mut Reflex> {
        let staleness = self.staleness_threshold_ns;
        self.entries
            .iter_mut()
            .find(|r| r.task_hash == *task_hash && !r.is_stale(now, staleness))
    }

    /// Compile a new reflex from successful neural execution.
    /// Only compiles if Iḥsān meets constitutional threshold.
    pub fn compile(
        &mut self,
        description: &str,
        actions: Vec<BizraAction>,
        ihsan: IhsanScore,
        timestamp: ActionTimestamp,
    ) -> Result<usize, ReflexError> {
        if !ihsan.meets_constitutional() {
            return Err(ReflexError::IhsanBelowThreshold {
                score: ihsan.value(),
                required: IhsanScore::PRODUCTION_FLOOR,
            });
        }

        let task_hash = Self::task_signature(description);

        // Update existing if new Iḥsān is higher
        if let Some(idx) = self.entries.iter().position(|r| r.task_hash == task_hash) {
            if ihsan.value() > self.entries[idx].compiled_ihsan.value() {
                self.entries[idx].actions = actions;
                self.entries[idx].compiled_ihsan = ihsan;
                self.entries[idx].compiled_at = timestamp;
                return Ok(idx);
            }
            return Err(ReflexError::ExistingReflexBetter);
        }

        // Evict LRU if at capacity
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        self.entries.push(Reflex {
            task_hash,
            description: description.to_string(),
            actions,
            compiled_ihsan: ihsan,
            execution_count: 0,
            last_used: timestamp,
            compiled_at: timestamp,
            avg_duration_ns: 0,
            marketplace_eligible: false,
        });

        Ok(self.entries.len() - 1)
    }

    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let lru = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| r.last_used.0)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.entries.swap_remove(lru);
    }

    /// Promote to marketplace after sufficient executions.
    pub fn promote_to_marketplace(
        &mut self,
        task_hash: &[u8; 32],
        min_executions: u64,
    ) -> Result<(), ReflexError> {
        let reflex = self
            .entries
            .iter_mut()
            .find(|r| r.task_hash == *task_hash)
            .ok_or(ReflexError::NotFound)?;

        if reflex.execution_count < min_executions {
            return Err(ReflexError::InsufficientExecutions {
                current: reflex.execution_count,
                required: min_executions,
            });
        }
        reflex.marketplace_eligible = true;
        Ok(())
    }

    // ── Stats ──
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
    pub fn total_compute_saved_ns(&self) -> u64 {
        self.entries.iter().map(|r| r.compute_saved_ns()).sum()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn marketplace_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|r| r.marketplace_eligible)
            .count()
    }

    pub fn health(&self) -> ReflexHealth {
        ReflexHealth {
            total_reflexes: self.entries.len() as u64,
            capacity: self.capacity as u64,
            hit_rate: self.hit_rate(),
            total_lookups: self.total_lookups,
            total_hits: self.hits,
            compute_saved_ns: self.total_compute_saved_ns(),
            marketplace_eligible: self.marketplace_count() as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReflexError {
    IhsanBelowThreshold { score: f64, required: f64 },
    ExistingReflexBetter,
    NotFound,
    InsufficientExecutions { current: u64, required: u64 },
}

#[derive(Debug, Clone)]
pub struct ReflexHealth {
    pub total_reflexes: u64,
    pub capacity: u64,
    pub hit_rate: f64,
    pub total_lookups: u64,
    pub total_hits: u64,
    pub compute_saved_ns: u64,
    pub marketplace_eligible: u64,
}
