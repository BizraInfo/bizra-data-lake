//! BIZRA Autopoiesis — Self-Creating Systems

pub mod pattern_memory;
pub mod preference_tracker;

pub use pattern_memory::{PatternMemory, Pattern, PatternStore};
pub use preference_tracker::{PreferenceTracker, Preference, PreferenceType};

pub const EMBEDDING_DIM: usize = 384;
pub const ELEVATION_THRESHOLD: f64 = 0.95;
