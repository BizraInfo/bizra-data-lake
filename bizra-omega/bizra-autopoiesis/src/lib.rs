//! BIZRA Autopoiesis — Self-Creating Systems

pub mod pattern_memory;
pub mod preference_tracker;

pub use pattern_memory::{Pattern, PatternMemory, PatternStore};
pub use preference_tracker::{Preference, PreferenceTracker, PreferenceType};

pub const EMBEDDING_DIM: usize = 384;
pub const ELEVATION_THRESHOLD: f64 = 0.95;
