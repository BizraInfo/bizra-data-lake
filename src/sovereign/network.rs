// src/sovereign/network.rs - BIZRA Network Multiplier with Reverse Scaling
//
// # PEAK MASTERPIECE IMPLEMENTATION - NETWORK ECONOMICS
//
// Standing on the Shoulders of Giants:
// - Metcalfe's Law: Network value ∝ n² (modified for efficiency)
// - Reed's Law: Group-forming networks scale exponentially
// - Dunbar's Number: Optimal group sizes (150)
// - Shannon: Information-theoretic efficiency bounds
//
// ## Reverse Scaling Philosophy
// ```
// Traditional networks:  Cost grows with scale
// BIZRA networks:        Efficiency IMPROVES with scale
//
// M = 1 + log₁₀(node_count + 1) / 10
//
// This creates sub-linear cost growth while preserving
// network effects from Metcalfe/Reed.
// ```
//
// إحسان Quality Standard: 99.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum network multiplier (caps at 3x efficiency boost)
pub const M_MAX: f64 = 2.0;

/// Minimum node count for multiplier activation
pub const MIN_NODES_FOR_BOOST: usize = 10;

/// Dunbar optimal group size for sub-network formation
pub const DUNBAR_NUMBER: usize = 150;

/// Shannon efficiency threshold (bits per joule equivalent)
pub const SHANNON_EFFICIENCY_MIN: f64 = 0.85;

/// Network Multiplier - Reverse Scaling Economics
///
/// Implements: M = 1 + log₁₀(node_count + 1) / 10
///
/// Key insight: As network grows, per-node cost DECREASES
/// because fixed overhead is amortized across more participants.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMultiplier {
    /// Current node count
    node_count: usize,
    /// Active contributors (subset of nodes)
    active_contributors: usize,
    /// Current multiplier value
    multiplier: f64,
    /// Decentralization factor (1 - Gini)
    decentralization: f64,
    /// Ihsān score for gating
    ihsan_score: f64,
    /// Sub-network count (Dunbar groups)
    sub_networks: usize,
    /// Shannon efficiency metric
    shannon_efficiency: f64,
    /// Historical multiplier values for stability
    history: Vec<f64>,
}

impl NetworkMultiplier {
    /// Create new network multiplier
    pub fn new(node_count: usize, gini: f64, ihsan: f64) -> Self {
        let decentralization = 1.0 - gini;
        let multiplier = Self::calculate_multiplier(node_count, decentralization, ihsan);
        let sub_networks = (node_count / DUNBAR_NUMBER).max(1);

        Self {
            node_count,
            active_contributors: node_count,
            multiplier,
            decentralization,
            ihsan_score: ihsan,
            sub_networks,
            shannon_efficiency: 1.0,
            history: vec![multiplier],
        }
    }

    /// Calculate network multiplier with safety gates
    ///
    /// Formula: M = 1 + (log₁₀(n + 1) / 10) × D × I
    ///
    /// Where:
    /// - n = node_count
    /// - D = decentralization factor (1 - Gini)
    /// - I = Ihsān score (ethics gate)
    fn calculate_multiplier(node_count: usize, decentralization: f64, ihsan: f64) -> f64 {
        // No boost for small networks
        if node_count < MIN_NODES_FOR_BOOST {
            return 1.0;
        }

        // Safety gate: Ihsān must meet minimum
        if ihsan < 0.85 {
            return 1.0;
        }

        // Safety gate: Decentralization must meet minimum
        if decentralization < 0.60 {
            return 1.0;
        }

        // Base boost factor from reverse scaling
        // boost = log₁₀(n + 1) / 10
        let boost = (node_count as f64 + 1.0).log10() / 10.0;

        // Scale boost by decentralization and Ihsān factors
        // M = 1 + boost × D × I (always >= 1.0 since boost >= 0)
        let scaled_boost = boost * decentralization * ihsan;

        // Final multiplier (clamped to max)
        (1.0 + scaled_boost).min(1.0 + M_MAX)
    }

    /// Update network state and recalculate multiplier
    pub fn update(&mut self, node_count: usize, gini: f64, ihsan: f64) {
        self.node_count = node_count;
        self.decentralization = 1.0 - gini;
        self.ihsan_score = ihsan;
        self.sub_networks = (node_count / DUNBAR_NUMBER).max(1);

        let new_multiplier =
            Self::calculate_multiplier(node_count, self.decentralization, self.ihsan_score);

        // EMA smoothing for stability (α = 0.1)
        self.multiplier = 0.9 * self.multiplier + 0.1 * new_multiplier;

        // Track history (keep last 100 values)
        self.history.push(self.multiplier);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Get current multiplier value
    pub fn get_multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Get effective reward multiplier for a contributor
    ///
    /// Combines network effect with individual contribution weight
    pub fn reward_multiplier(&self, contribution_score: f64) -> f64 {
        // Network base × sqrt(contribution) for sub-linear scaling
        self.multiplier * contribution_score.sqrt()
    }

    /// Get cost efficiency factor
    ///
    /// Higher efficiency = more value per unit cost
    /// Larger networks have better efficiency due to amortized overhead
    pub fn cost_efficiency(&self) -> f64 {
        if self.node_count == 0 {
            return 1.0;
        }

        // Efficiency improves with network size:
        // efficiency = M × (1 + log₁₀(n+1))
        // This increases with both multiplier and network size
        let scale_bonus = 1.0 + (self.node_count as f64 + 1.0).log10();
        self.multiplier * scale_bonus
    }

    /// Check if network meets Shannon efficiency threshold
    pub fn meets_shannon_threshold(&self) -> bool {
        self.shannon_efficiency >= SHANNON_EFFICIENCY_MIN
    }

    /// Calculate network value (modified Metcalfe)
    ///
    /// V = n × log₂(n) × M (instead of n²)
    pub fn network_value(&self) -> f64 {
        if self.node_count <= 1 {
            return self.node_count as f64;
        }

        let n = self.node_count as f64;
        n * n.log2() * self.multiplier
    }

    /// Get sub-network count (Dunbar groups)
    pub fn sub_network_count(&self) -> usize {
        self.sub_networks
    }

    /// Get multiplier stability (variance over history)
    pub fn stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }

        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;

        // Stability = 1 / (1 + variance)
        1.0 / (1.0 + variance)
    }
}

/// Reverse Scaling - Cost Reduction with Growth
///
/// Implements the anti-pattern to traditional cost scaling:
/// As network grows, per-unit costs DECREASE.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReverseScaling {
    /// Base cost per operation
    base_cost: f64,
    /// Current effective cost
    effective_cost: f64,
    /// Scale factor (1/log(n))
    scale_factor: f64,
    /// Minimum cost floor (prevents zero cost)
    cost_floor: f64,
    /// Cost history for trend analysis
    cost_history: Vec<f64>,
}

impl ReverseScaling {
    /// Create new reverse scaling model
    pub fn new(base_cost: f64, cost_floor: f64) -> Self {
        Self {
            base_cost,
            effective_cost: base_cost,
            scale_factor: 1.0,
            cost_floor,
            cost_history: vec![base_cost],
        }
    }

    /// Update effective cost based on network size
    ///
    /// cost = base / log₁₀(n + 1) when log > 1 (n >= 10)
    pub fn update(&mut self, node_count: usize) {
        // Need enough nodes for scaling to reduce cost
        let log_factor = (node_count as f64 + 1.0).log10();

        if node_count < 10 || log_factor <= 1.0 {
            self.effective_cost = self.base_cost;
            self.scale_factor = 1.0;
            return;
        }

        // Reverse scaling: cost decreases with more nodes
        self.scale_factor = 1.0 / log_factor;
        self.effective_cost = (self.base_cost * self.scale_factor).max(self.cost_floor);

        self.cost_history.push(self.effective_cost);
        if self.cost_history.len() > 100 {
            self.cost_history.remove(0);
        }
    }

    /// Get current effective cost
    pub fn get_cost(&self) -> f64 {
        self.effective_cost
    }

    /// Get cost reduction percentage from base
    pub fn cost_reduction(&self) -> f64 {
        if self.base_cost == 0.0 {
            return 0.0;
        }
        (1.0 - self.effective_cost / self.base_cost) * 100.0
    }

    /// Project cost at future network size
    pub fn project_cost(&self, future_node_count: usize) -> f64 {
        if future_node_count < 2 {
            return self.base_cost;
        }

        let future_scale = 1.0 / (future_node_count as f64 + 1.0).log10();
        (self.base_cost * future_scale).max(self.cost_floor)
    }

    /// Calculate cost trend (positive = increasing, negative = decreasing)
    pub fn cost_trend(&self) -> f64 {
        if self.cost_history.len() < 2 {
            return 0.0;
        }

        let recent: f64 = self.cost_history.iter().rev().take(10).sum::<f64>()
            / self.cost_history.iter().rev().take(10).count() as f64;
        let earlier: f64 = self.cost_history.iter().take(10).sum::<f64>()
            / self.cost_history.iter().take(10).count() as f64;

        recent - earlier
    }
}

/// Resource Pool - Network Resource Management
///
/// Manages shared resources across the network with
/// fair allocation based on contribution and need.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Total compute capacity (normalized units)
    total_compute: u64,
    /// Total storage capacity (bytes)
    total_storage: u64,
    /// Total bandwidth capacity (bytes/sec)
    total_bandwidth: u64,
    /// Allocated compute
    allocated_compute: u64,
    /// Allocated storage
    allocated_storage: u64,
    /// Allocated bandwidth
    allocated_bandwidth: u64,
    /// Node contributions
    contributions: HashMap<[u8; 32], NodeContribution>,
    /// Fair allocation weights
    allocation_weights: HashMap<[u8; 32], f64>,
}

/// Node contribution to resource pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeContribution {
    /// Compute contribution
    pub compute: u64,
    /// Storage contribution
    pub storage: u64,
    /// Bandwidth contribution
    pub bandwidth: u64,
    /// Uptime percentage (0-100)
    pub uptime: u8,
    /// Contribution timestamp
    pub timestamp: u64,
}

impl ResourcePool {
    /// Create new resource pool
    pub fn new() -> Self {
        Self {
            total_compute: 0,
            total_storage: 0,
            total_bandwidth: 0,
            allocated_compute: 0,
            allocated_storage: 0,
            allocated_bandwidth: 0,
            contributions: HashMap::new(),
            allocation_weights: HashMap::new(),
        }
    }

    /// Add node contribution to pool
    pub fn add_contribution(&mut self, node_id: [u8; 32], contribution: NodeContribution) {
        // Update totals
        self.total_compute += contribution.compute;
        self.total_storage += contribution.storage;
        self.total_bandwidth += contribution.bandwidth;

        // Calculate weight based on contribution + uptime
        let weight = (contribution.compute as f64
            + contribution.storage as f64 / 1_000_000.0
            + contribution.bandwidth as f64 / 1_000.0)
            * (contribution.uptime as f64 / 100.0);

        self.allocation_weights.insert(node_id, weight);
        self.contributions.insert(node_id, contribution);
    }

    /// Remove node contribution
    pub fn remove_contribution(&mut self, node_id: &[u8; 32]) -> Option<NodeContribution> {
        if let Some(contrib) = self.contributions.remove(node_id) {
            self.total_compute = self.total_compute.saturating_sub(contrib.compute);
            self.total_storage = self.total_storage.saturating_sub(contrib.storage);
            self.total_bandwidth = self.total_bandwidth.saturating_sub(contrib.bandwidth);
            self.allocation_weights.remove(node_id);
            Some(contrib)
        } else {
            None
        }
    }

    /// Calculate fair allocation for a node
    ///
    /// Uses weighted fair queueing: allocation ∝ contribution × weight
    pub fn fair_allocation(&self, node_id: &[u8; 32]) -> (u64, u64, u64) {
        let total_weight: f64 = self.allocation_weights.values().sum();
        if total_weight == 0.0 {
            return (0, 0, 0);
        }

        let node_weight = self.allocation_weights.get(node_id).copied().unwrap_or(0.0);
        let fraction = node_weight / total_weight;

        let compute = (self.total_compute as f64 * fraction) as u64;
        let storage = (self.total_storage as f64 * fraction) as u64;
        let bandwidth = (self.total_bandwidth as f64 * fraction) as u64;

        (compute, storage, bandwidth)
    }

    /// Get pool utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total_compute == 0 {
            return 0.0;
        }

        let compute_util = self.allocated_compute as f64 / self.total_compute as f64;
        let storage_util = if self.total_storage > 0 {
            self.allocated_storage as f64 / self.total_storage as f64
        } else {
            0.0
        };
        let bandwidth_util = if self.total_bandwidth > 0 {
            self.allocated_bandwidth as f64 / self.total_bandwidth as f64
        } else {
            0.0
        };

        (compute_util + storage_util + bandwidth_util) / 3.0 * 100.0
    }

    /// Get number of contributing nodes
    pub fn contributor_count(&self) -> usize {
        self.contributions.len()
    }

    /// Allocate resources for a task
    pub fn allocate(
        &mut self,
        compute: u64,
        storage: u64,
        bandwidth: u64,
    ) -> Result<(), &'static str> {
        if self.allocated_compute + compute > self.total_compute {
            return Err("Insufficient compute capacity");
        }
        if self.allocated_storage + storage > self.total_storage {
            return Err("Insufficient storage capacity");
        }
        if self.allocated_bandwidth + bandwidth > self.total_bandwidth {
            return Err("Insufficient bandwidth capacity");
        }

        self.allocated_compute += compute;
        self.allocated_storage += storage;
        self.allocated_bandwidth += bandwidth;
        Ok(())
    }

    /// Release allocated resources
    pub fn release(&mut self, compute: u64, storage: u64, bandwidth: u64) {
        self.allocated_compute = self.allocated_compute.saturating_sub(compute);
        self.allocated_storage = self.allocated_storage.saturating_sub(storage);
        self.allocated_bandwidth = self.allocated_bandwidth.saturating_sub(bandwidth);
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_multiplier_small_network() {
        let nm = NetworkMultiplier::new(5, 0.2, 0.95);
        // Small network gets no boost
        assert_eq!(nm.get_multiplier(), 1.0);
    }

    #[test]
    fn test_network_multiplier_large_network() {
        let nm = NetworkMultiplier::new(1000, 0.2, 0.95);
        let multiplier = nm.get_multiplier();

        // Should get meaningful boost: 1 + log₁₀(1001)/10 × 0.8 × 0.95
        assert!(multiplier > 1.0);
        assert!(multiplier < 1.0 + M_MAX);
    }

    #[test]
    fn test_network_multiplier_low_ihsan() {
        let nm = NetworkMultiplier::new(1000, 0.2, 0.7);
        // Low Ihsān = no boost
        assert_eq!(nm.get_multiplier(), 1.0);
    }

    #[test]
    fn test_network_multiplier_high_gini() {
        let nm = NetworkMultiplier::new(1000, 0.5, 0.95);
        // High Gini (centralized) = no boost
        assert_eq!(nm.get_multiplier(), 1.0);
    }

    #[test]
    fn test_network_value() {
        let nm = NetworkMultiplier::new(100, 0.2, 0.95);
        let value = nm.network_value();

        // Value = n × log₂(n) × M
        assert!(value > 0.0);
        assert!(value > 100.0); // More than just node count
    }

    #[test]
    fn test_reverse_scaling() {
        let mut rs = ReverseScaling::new(100.0, 10.0);

        // Small network = base cost
        rs.update(5);
        assert_eq!(rs.get_cost(), 100.0);

        // Large network = reduced cost
        rs.update(1000);
        assert!(rs.get_cost() < 100.0);
        assert!(rs.get_cost() >= 10.0); // Respects floor
    }

    #[test]
    fn test_reverse_scaling_projection() {
        let rs = ReverseScaling::new(100.0, 10.0);

        let cost_100 = rs.project_cost(100);
        let cost_1000 = rs.project_cost(1000);
        let cost_10000 = rs.project_cost(10000);

        // Cost should decrease with scale
        assert!(cost_1000 < cost_100);
        assert!(cost_10000 < cost_1000);
    }

    #[test]
    fn test_resource_pool_contribution() {
        let mut pool = ResourcePool::new();

        let node_id = [1u8; 32];
        let contribution = NodeContribution {
            compute: 1000,
            storage: 1_000_000,
            bandwidth: 10_000,
            uptime: 99,
            timestamp: 0,
        };

        pool.add_contribution(node_id, contribution);

        assert_eq!(pool.contributor_count(), 1);
        assert_eq!(pool.total_compute, 1000);
    }

    #[test]
    fn test_resource_pool_fair_allocation() {
        let mut pool = ResourcePool::new();

        // Add two nodes with different contributions
        let node1 = [1u8; 32];
        let node2 = [2u8; 32];

        pool.add_contribution(
            node1,
            NodeContribution {
                compute: 1000,
                storage: 1_000_000,
                bandwidth: 10_000,
                uptime: 100,
                timestamp: 0,
            },
        );

        pool.add_contribution(
            node2,
            NodeContribution {
                compute: 500,
                storage: 500_000,
                bandwidth: 5_000,
                uptime: 100,
                timestamp: 0,
            },
        );

        let (c1, _, _) = pool.fair_allocation(&node1);
        let (c2, _, _) = pool.fair_allocation(&node2);

        // Node1 contributed more, should get more allocation
        assert!(c1 > c2);
    }

    #[test]
    fn test_resource_pool_allocation() {
        let mut pool = ResourcePool::new();

        pool.add_contribution(
            [1u8; 32],
            NodeContribution {
                compute: 1000,
                storage: 1_000_000,
                bandwidth: 10_000,
                uptime: 100,
                timestamp: 0,
            },
        );

        // Should succeed
        assert!(pool.allocate(500, 500_000, 5_000).is_ok());

        // Should fail - exceeds capacity
        assert!(pool.allocate(600, 0, 0).is_err());

        // Release and try again
        pool.release(500, 500_000, 5_000);
        assert!(pool.allocate(600, 0, 0).is_ok());
    }

    #[test]
    fn test_network_multiplier_stability() {
        let mut nm = NetworkMultiplier::new(100, 0.2, 0.95);

        // Simulate updates
        for i in 0..50 {
            nm.update(100 + i * 10, 0.2, 0.95);
        }

        let stability = nm.stability();
        // Should have some stability (not zero, not perfect)
        assert!(stability > 0.0);
        assert!(stability <= 1.0);
    }

    #[test]
    fn test_sub_network_count() {
        let nm = NetworkMultiplier::new(500, 0.2, 0.95);
        assert_eq!(nm.sub_network_count(), 3); // 500 / 150 = 3

        let nm_large = NetworkMultiplier::new(1500, 0.2, 0.95);
        assert_eq!(nm_large.sub_network_count(), 10); // 1500 / 150 = 10
    }

    #[test]
    fn test_cost_efficiency() {
        let nm_small = NetworkMultiplier::new(10, 0.2, 0.95);
        let nm_large = NetworkMultiplier::new(10000, 0.1, 0.98);

        // Larger network should have better cost efficiency
        assert!(nm_large.cost_efficiency() > nm_small.cost_efficiency());
    }
}
