//! # bizra-hypergraph
//!
//! N-ary knowledge graph with incidence-list storage and SIMD-friendly traversal.
//!
//! A hyperedge connects **N >= 2** nodes simultaneously, generalising the
//! standard binary edge model used by `core/graph/semantic_layer.py`.
//!
//! ## Standing on Giants
//! - Berge (1973) — Hypergraph theory
//! - Vaswani (2017) — Attention as soft hyperedge
//! - Shannon (1948) — Information content of hyperedge membership
//! - Besta (2024) — Graph-of-Thoughts as directed hypergraph

pub mod hyperedge;
pub mod store;
pub mod traversal;

pub use hyperedge::{HyperEdge, HyperEdgeType};
pub use store::HyperGraphStore;
pub use traversal::bfs_reachable;
