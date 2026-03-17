// bizra-agent/src/persistence.rs
// ============================================================
// GENESIS Reflex Persistence — Sovereign file-based store
// ============================================================
//
// Standing on Giants:
// - Lampson (1974): Naming = addressing. TriggerHash IS the filename.
// - Al-Ghazali (1111): Persistence of learned virtue (compiled reflexes).
// - Shannon (1948): Content-addressed storage = maximum information density.
//
// Architecture:
//   Each compiled ReflexRule is serialized to JSON and stored as a file
//   named by its TriggerHash hex (64 chars + .json). On cold start,
//   the store restores all rules, giving the agent instant System-1
//   capability without replaying System-2 traces.
//
//   File layout:
//     {store_root}/
//       {trigger_hash_hex}.json   — one file per compiled reflex
//       _manifest.json            — integrity manifest (BLAKE3 of all rules)
//
// Zero external dependencies. Pure std::fs + serde_json.
// ============================================================

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::hash_namespace::TriggerHash;
use crate::reflex_cache::ReflexRule;

/// Persistence error types.
#[derive(Debug)]
pub enum PersistError {
    /// IO error from filesystem operations.
    Io(io::Error),
    /// JSON serialization/deserialization error.
    Json(serde_json::Error),
    /// Store directory does not exist and could not be created.
    StoreNotCreated(PathBuf),
    /// Integrity check failed: stored hash does not match recomputed hash.
    IntegrityViolation {
        trigger_hex: String,
        reason: &'static str,
    },
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "persistence IO error: {e}"),
            Self::Json(e) => write!(f, "persistence JSON error: {e}"),
            Self::StoreNotCreated(p) => write!(f, "cannot create store at {}", p.display()),
            Self::IntegrityViolation { trigger_hex, reason } => {
                write!(f, "integrity violation for {trigger_hex}: {reason}")
            }
        }
    }
}

impl From<io::Error> for PersistError {
    fn from(e: io::Error) -> Self { Self::Io(e) }
}
impl From<serde_json::Error> for PersistError {
    fn from(e: serde_json::Error) -> Self { Self::Json(e) }
}

/// Integrity manifest stored alongside rules.
#[derive(Debug, Serialize, Deserialize)]
struct StoreManifest {
    /// Number of rules in the store.
    rule_count: usize,
    /// BLAKE3 hash of all trigger hashes concatenated (sorted order).
    content_hash: String,
    /// Timestamp of last write (Unix seconds).
    last_written_at: u64,
    /// Store format version (for forward compatibility).
    version: u32,
}

const MANIFEST_FILENAME: &str = "_manifest.json";
const STORE_VERSION: u32 = 1;

/// File-based reflex persistence store.
///
/// Each `ReflexRule` is stored as `{trigger_hash_hex}.json` in the
/// store directory. The manifest tracks integrity.
pub struct ReflexStore {
    root: PathBuf,
}

impl ReflexStore {
    /// Open or create a store at the given directory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, PersistError> {
        let root = root.into();
        if !root.exists() {
            std::fs::create_dir_all(&root).map_err(|_| {
                PersistError::StoreNotCreated(root.clone())
            })?;
            info!(path = %root.display(), "created reflex store directory");
        }
        Ok(Self { root })
    }

    /// Persist a single compiled rule to disk.
    ///
    /// Overwrites any existing rule with the same trigger hash.
    /// This is the hot path — called after every successful compilation.
    pub fn save_rule(&self, rule: &ReflexRule) -> Result<(), PersistError> {
        let hex = rule.trigger_hash.to_hex();
        let path = self.rule_path(&hex);
        let json = serde_json::to_string_pretty(rule)?;
        std::fs::write(&path, json.as_bytes())?;
        debug!(trigger = %hex, "persisted reflex rule");
        Ok(())
    }

    /// Remove a rule from disk (called on invalidation).
    pub fn remove_rule(&self, trigger: &TriggerHash) -> Result<bool, PersistError> {
        let hex = trigger.to_hex();
        let path = self.rule_path(&hex);
        if path.exists() {
            std::fs::remove_file(&path)?;
            debug!(trigger = %hex, "removed persisted reflex rule");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Restore all persisted rules from disk.
    ///
    /// Skips files that fail to deserialize (logs warning, does not
    /// propagate error — partial restoration is better than none).
    pub fn restore_all(&self) -> Result<Vec<ReflexRule>, PersistError> {
        let mut rules = Vec::new();
        let mut errors = 0usize;

        let entries = std::fs::read_dir(&self.root)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            // Skip manifest and non-JSON files
            let Some(ext) = path.extension() else { continue };
            if ext != "json" { continue }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            if stem.starts_with('_') { continue }

            // Must be a 64-char hex filename (32-byte BLAKE3 hash)
            if stem.len() != 64 { continue }

            match std::fs::read_to_string(&path) {
                Ok(json) => match serde_json::from_str::<ReflexRule>(&json) {
                    Ok(rule) => {
                        // Integrity check: filename must match trigger hash
                        if rule.trigger_hash.to_hex() != stem {
                            warn!(
                                file = %stem,
                                actual = %rule.trigger_hash.to_hex(),
                                "integrity violation: filename != trigger hash, skipping"
                            );
                            errors += 1;
                            continue;
                        }
                        rules.push(rule);
                    }
                    Err(e) => {
                        warn!(file = %stem, err = %e, "failed to deserialize reflex rule");
                        errors += 1;
                    }
                },
                Err(e) => {
                    warn!(file = %stem, err = %e, "failed to read reflex rule file");
                    errors += 1;
                }
            }
        }

        info!(
            restored = rules.len(),
            errors = errors,
            path = %self.root.display(),
            "reflex store restoration complete"
        );
        Ok(rules)
    }

    /// Write the integrity manifest after a bulk operation.
    pub fn write_manifest(&self, rules: &[ReflexRule]) -> Result<(), PersistError> {
        let content_hash = self.compute_content_hash(rules);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let manifest = StoreManifest {
            rule_count: rules.len(),
            content_hash,
            last_written_at: now,
            version: STORE_VERSION,
        };

        let json = serde_json::to_string_pretty(&manifest)?;
        let path = self.root.join(MANIFEST_FILENAME);
        std::fs::write(&path, json.as_bytes())?;
        debug!(rules = rules.len(), "wrote reflex store manifest");
        Ok(())
    }

    /// Persist ALL rules at once (snapshot). Used for graceful shutdown.
    pub fn snapshot(&self, rules: &[ReflexRule]) -> Result<usize, PersistError> {
        let mut saved = 0usize;
        for rule in rules {
            if !rule.quarantined {
                self.save_rule(rule)?;
                saved += 1;
            }
        }
        self.write_manifest(rules)?;
        info!(saved = saved, total = rules.len(), "reflex store snapshot complete");
        Ok(saved)
    }

    /// Compute BLAKE3 content hash of all trigger hashes (sorted).
    fn compute_content_hash(&self, rules: &[ReflexRule]) -> String {
        let mut hashes: Vec<[u8; 32]> = rules.iter().map(|r| r.trigger_hash.0).collect();
        hashes.sort();
        let mut hasher = blake3::Hasher::new();
        for h in &hashes {
            hasher.update(h);
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Path to a rule file given its hex trigger hash.
    fn rule_path(&self, hex: &str) -> PathBuf {
        self.root.join(format!("{hex}.json"))
    }

    /// Path to the store root (for diagnostics).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Count of .json rule files on disk (excludes manifest).
    pub fn file_count(&self) -> usize {
        std::fs::read_dir(&self.root)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        let p = e.path();
                        p.extension().is_some_and(|ext| ext == "json")
                            && p.file_stem()
                                .and_then(|s| s.to_str())
                                .is_some_and(|s| !s.starts_with('_') && s.len() == 64)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflex_cache::{ActionTemplate, QuarantineReason, BOOTSTRAP_POLICY_HASH};
    use tempfile::TempDir;

    fn test_rule(trigger_bytes: u8, policy_bytes: u8) -> ReflexRule {
        ReflexRule {
            trigger_hash: TriggerHash([trigger_bytes; 32]),
            action_template: ActionTemplate {
                route_signature: "Retrieve>Generate|roles=Scholar>Artisan".to_string(),
                primary_agent: "Scholar".to_string(),
            },
            compile_ihsan: 0.97,
            compile_snr: 0.93,
            compiled_at: 1700000000,
            use_count: 42,
            last_used_at: 1700001000,
            last_validated_at: 1700000500,
            quarantined: false,
            quarantine_reason: None,
            policy_hash: [policy_bytes; 32],
        }
    }

    #[test]
    fn save_and_restore_single_rule() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let rule = test_rule(0xAA, 0x07);
        store.save_rule(&rule).unwrap();

        // Verify file exists on disk
        assert_eq!(store.file_count(), 1);

        // Restore
        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 1);

        let r = &restored[0];
        assert_eq!(r.trigger_hash, rule.trigger_hash);
        assert_eq!(r.action_template.primary_agent, "Scholar");
        assert_eq!(r.action_template.route_signature, "Retrieve>Generate|roles=Scholar>Artisan");
        assert!((r.compile_ihsan - 0.97).abs() < f32::EPSILON);
        assert!((r.compile_snr - 0.93).abs() < f32::EPSILON);
        assert_eq!(r.compiled_at, 1700000000);
        assert_eq!(r.use_count, 42);
        assert_eq!(r.policy_hash, [0x07; 32]);
        assert!(!r.quarantined);
    }

    #[test]
    fn restore_multiple_rules() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let rules: Vec<ReflexRule> = (1u8..=5)
            .map(|i| test_rule(i, 0x07))
            .collect();

        for rule in &rules {
            store.save_rule(rule).unwrap();
        }
        assert_eq!(store.file_count(), 5);

        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 5);
    }

    #[test]
    fn overwrite_existing_rule() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let mut rule = test_rule(0xBB, 0x07);
        store.save_rule(&rule).unwrap();

        // Overwrite with updated use_count
        rule.use_count = 999;
        rule.compile_ihsan = 0.99;
        store.save_rule(&rule).unwrap();

        assert_eq!(store.file_count(), 1); // Still 1 file

        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].use_count, 999);
        assert!((restored[0].compile_ihsan - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn remove_rule_from_disk() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let rule = test_rule(0xCC, 0x07);
        store.save_rule(&rule).unwrap();
        assert_eq!(store.file_count(), 1);

        let removed = store.remove_rule(&rule.trigger_hash).unwrap();
        assert!(removed);
        assert_eq!(store.file_count(), 0);

        // Removing again returns false
        let removed_again = store.remove_rule(&rule.trigger_hash).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn integrity_check_rejects_tampered_filename() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let rule = test_rule(0xDD, 0x07);
        store.save_rule(&rule).unwrap();

        // Tamper: rename file to wrong hash
        let correct_hex = rule.trigger_hash.to_hex();
        let correct_path = dir.path().join(format!("{correct_hex}.json"));
        let wrong_hex = TriggerHash([0xEE; 32]).to_hex();
        let wrong_path = dir.path().join(format!("{wrong_hex}.json"));
        std::fs::rename(&correct_path, &wrong_path).unwrap();

        // Restore should skip the tampered file
        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 0, "tampered file must be rejected");
    }

    #[test]
    fn skips_malformed_json() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        // Write valid rule
        let rule = test_rule(0x11, 0x07);
        store.save_rule(&rule).unwrap();

        // Write malformed JSON with valid-looking filename
        let fake_hex = TriggerHash([0x22; 32]).to_hex();
        let fake_path = dir.path().join(format!("{fake_hex}.json"));
        std::fs::write(&fake_path, b"{ this is not valid json }").unwrap();

        // Restore should get only the valid rule
        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].trigger_hash, TriggerHash([0x11; 32]));
    }

    #[test]
    fn snapshot_skips_quarantined() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let mut quarantined = test_rule(0xF1, 0x07);
        quarantined.quarantined = true;
        quarantined.quarantine_reason = Some(QuarantineReason::GuardianVeto);

        let healthy = test_rule(0xF2, 0x07);

        let saved = store.snapshot(&[quarantined, healthy]).unwrap();
        assert_eq!(saved, 1, "only non-quarantined rules should be persisted");
        assert_eq!(store.file_count(), 1);
    }

    #[test]
    fn manifest_written_and_readable() {
        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        let rules: Vec<ReflexRule> = (1u8..=3).map(|i| test_rule(i, 0x07)).collect();
        store.snapshot(&rules).unwrap();

        // Manifest should exist
        let manifest_path = dir.path().join(MANIFEST_FILENAME);
        assert!(manifest_path.exists());

        let json = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: StoreManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.rule_count, 3);
        assert_eq!(manifest.version, STORE_VERSION);
        assert!(!manifest.content_hash.is_empty());
        assert!(manifest.last_written_at > 0);
    }

    #[test]
    fn open_creates_directory() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("reflexes").join("v1");
        assert!(!nested.exists());

        let store = ReflexStore::open(&nested).unwrap();
        assert!(nested.exists());
        assert_eq!(store.file_count(), 0);
    }

    #[test]
    fn bootstrap_rules_roundtrip() {
        use crate::reflex_cache::{ReflexCache, is_bootstrap_rule};

        let dir = TempDir::new().unwrap();
        let store = ReflexStore::open(dir.path()).unwrap();

        // Load bootstrap rules into cache
        let mut cache = ReflexCache::new(64);
        cache.load_bootstrap_rules();
        let rules = cache.all_rules();
        assert_eq!(rules.len(), 4);

        // Snapshot to disk
        let saved = store.snapshot(&rules).unwrap();
        assert_eq!(saved, 4);

        // Restore from disk
        let restored = store.restore_all().unwrap();
        assert_eq!(restored.len(), 4);

        // All restored rules should still be identified as bootstrap
        for rule in &restored {
            assert!(is_bootstrap_rule(rule));
            assert_eq!(rule.policy_hash, BOOTSTRAP_POLICY_HASH);
        }

        // Load into fresh cache
        let mut cache2 = ReflexCache::new(64);
        cache2.replace_rules(restored);
        assert_eq!(cache2.stats().size, 4);
    }
}
