// src/unified/wisdom.rs - WisdomAtom & Symbolic Harness
//
// SAPE v1.∞: Bridge Neural ↔️ Symbolic
// =====================================
// Prevents "Hallucination Amplification" by grounding thoughts
// in verifiable logical propositions.
//
// References:
// - Shannon (1948): Information Entropy for breakthrough potential
// - NIST SP 800-207: Zero Trust attestation model

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Symbol - Atomic logical unit (Horn clause component)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Symbol {
    /// Symbol name (predicate)
    pub name: String,
    /// Arguments to the predicate
    pub arguments: Vec<String>,
    /// Whether this symbol is negated
    pub negated: bool,
}

impl Symbol {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            arguments: Vec::new(),
            negated: false,
        }
    }

    pub fn with_args(name: &str, args: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            arguments: args.into_iter().map(String::from).collect(),
            negated: false,
        }
    }

    pub fn negated(mut self) -> Self {
        self.negated = true;
        self
    }

    /// Convert to Horn clause string representation
    pub fn to_horn_clause(&self) -> String {
        let prefix = if self.negated { "NOT " } else { "" };
        if self.arguments.is_empty() {
            format!("{}{}", prefix, self.name)
        } else {
            format!("{}{}({})", prefix, self.name, self.arguments.join(", "))
        }
    }
}

/// ActionPrimitive - Executable action with resource cost
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPrimitive {
    /// Action type: execute, delegate, emit, store
    pub action_type: ActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Estimated resource cost (0.0-1.0)
    pub cost: f64,
    /// Whether action is reversible
    pub reversible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    Execute,   // Run code/command
    Delegate,  // Pass to another agent
    Emit,      // Produce output
    Store,     // Persist data
    Query,     // Retrieve information
    Transform, // Modify data
}

impl ActionPrimitive {
    pub fn execute(params: HashMap<String, String>, cost: f64) -> Self {
        Self {
            action_type: ActionType::Execute,
            parameters: params,
            cost,
            reversible: false,
        }
    }

    pub fn emit(content: &str) -> Self {
        let mut params = HashMap::new();
        params.insert("content".to_string(), content.to_string());
        Self {
            action_type: ActionType::Emit,
            parameters: params,
            cost: 0.01,
            reversible: false,
        }
    }
}

/// WisdomAtom - The fundamental unit of verified knowledge
///
/// Implements the Symbolic Harness from SAPE analysis:
/// - Neural component (embedding)
/// - Symbolic component (Horn clause: preconditions → action → postconditions)
/// - Evidence chain (Isnad: cryptographic provenance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomAtom {
    /// Unique identifier
    pub id: String,

    // === Neural Component ===
    /// Semantic embedding vector
    pub embedding: Vec<f32>,

    // === Symbolic Component (Horn Clause) ===
    /// Preconditions that must be satisfied
    /// e.g., [HighLoad, LowMemory] → "IF HighLoad AND LowMemory"
    pub preconditions: Vec<Symbol>,
    /// The action to take when preconditions are met
    pub action: ActionPrimitive,
    /// Expected postconditions after action
    pub postconditions: Vec<Symbol>,

    // === Evidence Chain (Isnad) ===
    /// Cryptographic signatures from provenance chain
    pub provenance_chain: Vec<Vec<u8>>,
    /// Historical success rate (0.0-1.0)
    pub success_rate: f64,
    /// Context hash for relevance matching
    pub context_hash: [u8; 32],

    // === Metadata ===
    /// Creation timestamp (Unix epoch ms)
    pub created_at: u64,
    /// Source agent that created this wisdom
    pub source_agent: String,
    /// Generation number (evolutionary lineage)
    pub generation: u32,
    /// Number of times this wisdom was applied
    pub application_count: u64,
    /// Ihsān score of the wisdom
    pub ihsan_score: f64,
}

impl WisdomAtom {
    /// Create a new WisdomAtom
    pub fn new(
        preconditions: Vec<Symbol>,
        action: ActionPrimitive,
        postconditions: Vec<Symbol>,
        source_agent: &str,
    ) -> Self {
        let id = Self::generate_id(&preconditions, &action);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            id,
            embedding: Vec::new(),
            preconditions,
            action,
            postconditions,
            provenance_chain: Vec::new(),
            success_rate: 0.5, // Initial neutral success rate
            context_hash: [0u8; 32],
            created_at: now,
            source_agent: source_agent.to_string(),
            generation: 1,
            application_count: 0,
            ihsan_score: 0.9,
        }
    }

    /// Generate unique ID from content
    fn generate_id(preconditions: &[Symbol], action: &ActionPrimitive) -> String {
        let mut hasher = Sha256::new();
        for pre in preconditions {
            hasher.update(pre.to_horn_clause().as_bytes());
        }
        hasher.update(format!("{:?}", action.action_type).as_bytes());
        let hash = hasher.finalize();
        let hex_str: String = hash[..8].iter().map(|b| format!("{:02x}", b)).collect();
        format!("WA-{}", hex_str)
    }

    /// Convert to Horn clause representation
    pub fn to_horn_clause(&self) -> String {
        let precond_str: Vec<String> = self
            .preconditions
            .iter()
            .map(|s| s.to_horn_clause())
            .collect();
        let postcond_str: Vec<String> = self
            .postconditions
            .iter()
            .map(|s| s.to_horn_clause())
            .collect();

        format!(
            "IF ({}) THEN {:?} RESULTING IN ({})",
            precond_str.join(" AND "),
            self.action.action_type,
            postcond_str.join(" AND ")
        )
    }

    /// Check if preconditions are satisfied by current state
    pub fn preconditions_satisfied(&self, state: &HashMap<String, bool>) -> bool {
        self.preconditions.iter().all(|symbol| {
            let present = state.get(&symbol.name).copied().unwrap_or(false);
            if symbol.negated {
                !present
            } else {
                present
            }
        })
    }

    /// Update success rate after application
    pub fn record_outcome(&mut self, success: bool) {
        self.application_count += 1;
        // Exponential moving average
        let alpha = 0.1;
        let outcome = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * outcome + (1.0 - alpha) * self.success_rate;
    }

    /// Add a provenance signature
    pub fn add_provenance(&mut self, signature: Vec<u8>) {
        self.provenance_chain.push(signature);
    }

    /// Set embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        // Compute context hash from embedding
        let mut hasher = Sha256::new();
        for val in &embedding {
            hasher.update(val.to_le_bytes());
        }
        self.context_hash = hasher.finalize().into();
        self.embedding = embedding;
        self
    }

    /// Fitness score for evolutionary selection
    /// Combines success rate, Ihsān score, and efficiency
    pub fn fitness_score(&self) -> f64 {
        let efficiency = 1.0 - self.action.cost;
        let experience = (self.application_count as f64).ln().max(1.0) / 10.0;

        // Weighted combination
        0.4 * self.success_rate
            + 0.3 * self.ihsan_score
            + 0.2 * efficiency
            + 0.1 * experience.min(1.0)
    }
}

/// WisdomStore - Persistent storage for WisdomAtoms
///
/// Implements:
/// - T-Cell Memory: Store signatures of failure for early pattern recognition
/// - Genealogy tracking: Wisdom lineage for evolutionary analysis
pub struct WisdomStore {
    /// Active wisdom atoms indexed by ID
    atoms: Arc<RwLock<HashMap<String, WisdomAtom>>>,
    /// Failure signatures (T-Cell Memory)
    failure_signatures: Arc<RwLock<Vec<FailureSignature>>>,
    /// Wisdom by context hash for fast lookup
    context_index: Arc<RwLock<HashMap<[u8; 32], Vec<String>>>>,
    /// Maximum atoms to store
    max_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignature {
    /// Pattern that led to failure
    pub pattern_hash: [u8; 32],
    /// Description of the failure
    pub description: String,
    /// Severity (0.0-1.0)
    pub severity: f64,
    /// When this failure was recorded
    pub recorded_at: u64,
    /// How many times this pattern has failed
    pub occurrence_count: u32,
}

impl WisdomStore {
    pub fn new(max_capacity: usize) -> Self {
        info!(max_capacity = max_capacity, "📚 WisdomStore initialized");
        Self {
            atoms: Arc::new(RwLock::new(HashMap::new())),
            failure_signatures: Arc::new(RwLock::new(Vec::new())),
            context_index: Arc::new(RwLock::new(HashMap::new())),
            max_capacity,
        }
    }

    /// Store a new wisdom atom
    pub async fn store(&self, atom: WisdomAtom) -> Result<(), String> {
        let mut atoms = self.atoms.write().await;

        // Check capacity
        if atoms.len() >= self.max_capacity {
            // Evict lowest fitness atom
            if let Some((lowest_id, _)) = atoms.iter().min_by(|a, b| {
                a.1.fitness_score()
                    .partial_cmp(&b.1.fitness_score())
                    .unwrap()
            }) {
                let id = lowest_id.clone();
                atoms.remove(&id);
                debug!(evicted_id = %id, "Evicted low-fitness wisdom atom");
            }
        }

        // Update context index
        {
            let mut index = self.context_index.write().await;
            index
                .entry(atom.context_hash)
                .or_insert_with(Vec::new)
                .push(atom.id.clone());
        }

        atoms.insert(atom.id.clone(), atom);
        Ok(())
    }

    /// Retrieve wisdom by ID
    pub async fn get(&self, id: &str) -> Option<WisdomAtom> {
        self.atoms.read().await.get(id).cloned()
    }

    /// Find relevant wisdom for a context
    pub async fn find_relevant(&self, context_hash: &[u8; 32], limit: usize) -> Vec<WisdomAtom> {
        let index = self.context_index.read().await;
        let atoms = self.atoms.read().await;

        if let Some(ids) = index.get(context_hash) {
            ids.iter()
                .filter_map(|id| atoms.get(id).cloned())
                .take(limit)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find wisdom applicable to current state
    pub async fn find_applicable(&self, state: &HashMap<String, bool>) -> Vec<WisdomAtom> {
        self.atoms
            .read()
            .await
            .values()
            .filter(|atom| atom.preconditions_satisfied(state))
            .cloned()
            .collect()
    }

    /// Record a failure pattern (T-Cell Memory)
    pub async fn record_failure(&self, pattern_hash: [u8; 32], description: &str, severity: f64) {
        let mut failures = self.failure_signatures.write().await;
        let pattern_hex: String = pattern_hash[..8]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();

        // Check if pattern already exists
        if let Some(existing) = failures.iter_mut().find(|f| f.pattern_hash == pattern_hash) {
            existing.occurrence_count += 1;
            existing.severity = existing.severity.max(severity);
            warn!(
                pattern = %pattern_hex,
                occurrences = existing.occurrence_count,
                "Recurring failure pattern detected"
            );
        } else {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            failures.push(FailureSignature {
                pattern_hash,
                description: description.to_string(),
                severity,
                recorded_at: now,
                occurrence_count: 1,
            });
            info!(
                pattern = %pattern_hex,
                description = description,
                "New failure signature recorded (T-Cell Memory)"
            );
        }
    }

    /// Check if a pattern matches known failures
    pub async fn matches_failure_pattern(
        &self,
        pattern_hash: &[u8; 32],
    ) -> Option<FailureSignature> {
        self.failure_signatures
            .read()
            .await
            .iter()
            .find(|f| &f.pattern_hash == pattern_hash)
            .cloned()
    }

    /// Get top wisdom by fitness
    pub async fn top_wisdom(&self, limit: usize) -> Vec<WisdomAtom> {
        let atoms = self.atoms.read().await;
        let mut sorted: Vec<_> = atoms.values().cloned().collect();
        sorted.sort_by(|a, b| b.fitness_score().partial_cmp(&a.fitness_score()).unwrap());
        sorted.into_iter().take(limit).collect()
    }

    /// Get statistics
    pub async fn stats(&self) -> WisdomStats {
        let atoms = self.atoms.read().await;
        let failures = self.failure_signatures.read().await;

        let total = atoms.len();
        let avg_fitness = if total > 0 {
            atoms.values().map(|a| a.fitness_score()).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let avg_success = if total > 0 {
            atoms.values().map(|a| a.success_rate).sum::<f64>() / total as f64
        } else {
            0.0
        };

        WisdomStats {
            total_atoms: total,
            average_fitness: avg_fitness,
            average_success_rate: avg_success,
            failure_patterns: failures.len(),
            capacity_used: total as f64 / self.max_capacity as f64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomStats {
    pub total_atoms: usize,
    pub average_fitness: f64,
    pub average_success_rate: f64,
    pub failure_patterns: usize,
    pub capacity_used: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_to_horn_clause() {
        let s1 = Symbol::new("HighLoad");
        assert_eq!(s1.to_horn_clause(), "HighLoad");

        let s2 = Symbol::with_args("Memory", vec!["low", "critical"]);
        assert_eq!(s2.to_horn_clause(), "Memory(low, critical)");

        let s3 = Symbol::new("Available").negated();
        assert_eq!(s3.to_horn_clause(), "NOT Available");
    }

    #[test]
    fn test_wisdom_atom_creation() {
        let preconditions = vec![Symbol::new("HighLoad"), Symbol::new("LowMemory")];
        let action = ActionPrimitive::emit("Throttle resources");
        let postconditions = vec![Symbol::new("NormalLoad")];

        let atom = WisdomAtom::new(preconditions, action, postconditions, "TestAgent");

        assert!(atom.id.starts_with("WA-"));
        assert_eq!(atom.source_agent, "TestAgent");
        assert_eq!(atom.generation, 1);
    }

    #[test]
    fn test_precondition_satisfaction() {
        let preconditions = vec![Symbol::new("HighLoad"), Symbol::new("Available").negated()];
        let action = ActionPrimitive::emit("Scale up");
        let postconditions = vec![];

        let atom = WisdomAtom::new(preconditions, action, postconditions, "Test");

        let mut state = HashMap::new();
        state.insert("HighLoad".to_string(), true);
        state.insert("Available".to_string(), false);

        assert!(atom.preconditions_satisfied(&state));

        state.insert("Available".to_string(), true);
        assert!(!atom.preconditions_satisfied(&state));
    }

    #[tokio::test]
    async fn test_wisdom_store() {
        let store = WisdomStore::new(100);

        let atom = WisdomAtom::new(
            vec![Symbol::new("Test")],
            ActionPrimitive::emit("Output"),
            vec![],
            "TestAgent",
        );

        let id = atom.id.clone();
        store.store(atom).await.unwrap();

        let retrieved = store.get(&id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().source_agent, "TestAgent");
    }
}
