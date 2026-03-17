// bizra-agent/src/skills/mod.rs
// ============================================================
// BIZRA Skills — Sovereign capability modules
// ============================================================
// Each skill defines both PAT (execution) and SAT (validation)
// behaviors. PAT serves the user. SAT validates independently.
//
// Constitutional defaults (every node ships with these):
//   1. file_management    — classify, organize, rename, merge, dedup
//   2. browser_management — navigate, read, fill, click, screenshot
//
// Default config (no user setup required):
//   NodeCapabilityConfig::genesis() → ready from minute one
// ============================================================

pub mod browser_management;
pub mod default_capabilities;
pub mod file_management;
pub mod skill_tree;
