//! BIZRA Resource Registry Cache — §Cycle-7 G3 Commit-5 seed
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/resource_registry_cache.rs
//! Authority: cycle-7/niyyah.md §G3 "Persistent Local Memory" +
//!            §G4 "Local resource registry + URP view"
//! Cycle position: 7, Phase 3 (seed) — Phase 4 (fill)
//!
//! Sixth and final dema_cache surface. Seeded empty in G3 so G4 can
//! assume the file exists and has a schema version already locked.
//!
//! G3 scope (this commit):
//!   - Module, schema-versioned JSON, atomic write/read, tests.
//!   - `seed_empty_if_missing()` creates an empty registry on boot.
//!   - No runtime API for mutating resources yet — G4 owns that.
//!
//! G4 scope (next phase):
//!   - `dema register-resource` CLI + gateway endpoint
//!   - Allowlist enforcement on `dema organize <path>`
//!   - URP (Universal Resource Pattern) projection from the registry

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

pub const CACHE_SCHEMA_VERSION: &str = "v1";
pub const DEMA_CACHE_DIRNAME: &str = "dema_cache";
pub const RESOURCE_REGISTRY_FILENAME: &str = "resource_registry.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRegistrySnapshot {
    pub resources: Vec<ResourceEntry>,
}

/// Minimal resource descriptor stub. G4 will extend this enum with
/// typed variants (FilesystemPath, NetworkEndpoint, ProcessHandle, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEntry {
    pub id: String,
    pub kind: String,
    pub summary: String,
    pub allowlisted: bool,
}

impl Default for ResourceRegistrySnapshot {
    fn default() -> Self {
        ResourceRegistrySnapshot {
            resources: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum ResourceRegistryCacheError {
    DirCreate {
        path: PathBuf,
        msg: String,
    },
    TempWrite {
        path: PathBuf,
        msg: String,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
        msg: String,
    },
    ReadFailed {
        path: PathBuf,
        msg: String,
    },
    ParseFailed {
        path: PathBuf,
        msg: String,
    },
    Malformed {
        path: PathBuf,
        reason: &'static str,
    },
    SchemaMismatch {
        path: PathBuf,
        got: String,
        want: String,
    },
    Serialize(String),
}

impl std::fmt::Display for ResourceRegistryCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirCreate { path, msg } => {
                write!(f, "create dema_cache dir {}: {}", path.display(), msg)
            }
            Self::TempWrite { path, msg } => {
                write!(f, "write cache temp {}: {}", path.display(), msg)
            }
            Self::Rename { from, to, msg } => {
                write!(f, "rename {} -> {}: {}", from.display(), to.display(), msg)
            }
            Self::ReadFailed { path, msg } => {
                write!(f, "read cache {}: {}", path.display(), msg)
            }
            Self::ParseFailed { path, msg } => {
                write!(f, "parse cache {}: {}", path.display(), msg)
            }
            Self::Malformed { path, reason } => {
                write!(f, "cache {} malformed: {}", path.display(), reason)
            }
            Self::SchemaMismatch { path, got, want } => {
                write!(
                    f,
                    "cache {} schema {}, expected {}",
                    path.display(),
                    got,
                    want
                )
            }
            Self::Serialize(s) => write!(f, "serialize resource registry: {}", s),
        }
    }
}

impl std::error::Error for ResourceRegistryCacheError {}

#[derive(Debug, Clone)]
pub struct ResourceRegistryCache {
    cache_dir: PathBuf,
}

impl ResourceRegistryCache {
    pub fn at_sovereign_root(sovereign_root: &Path) -> Self {
        ResourceRegistryCache {
            cache_dir: sovereign_root.join(DEMA_CACHE_DIRNAME),
        }
    }

    pub fn at_cache_dir(cache_dir: &Path) -> Self {
        ResourceRegistryCache {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn registry_path(&self) -> PathBuf {
        self.cache_dir.join(RESOURCE_REGISTRY_FILENAME)
    }

    pub fn write(
        &self,
        snapshot: &ResourceRegistrySnapshot,
    ) -> Result<(), ResourceRegistryCacheError> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| ResourceRegistryCacheError::DirCreate {
            path: self.cache_dir.clone(),
            msg: e.to_string(),
        })?;

        let mut resources_json = Vec::with_capacity(snapshot.resources.len());
        for r in &snapshot.resources {
            resources_json.push(json!({
                "id": r.id,
                "kind": r.kind,
                "summary": r.summary,
                "allowlisted": r.allowlisted,
            }));
        }

        let payload = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "resources": resources_json,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| ResourceRegistryCacheError::Serialize(e.to_string()))?;

        let final_path = self.registry_path();
        let tmp_path = self.cache_dir.join(format!(
            "{}.tmp.{}",
            RESOURCE_REGISTRY_FILENAME,
            std::process::id()
        ));

        {
            let mut f =
                fs::File::create(&tmp_path).map_err(|e| ResourceRegistryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.write_all(&bytes)
                .map_err(|e| ResourceRegistryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
            f.sync_all()
                .map_err(|e| ResourceRegistryCacheError::TempWrite {
                    path: tmp_path.clone(),
                    msg: e.to_string(),
                })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| ResourceRegistryCacheError::Rename {
            from: tmp_path,
            to: final_path,
            msg: e.to_string(),
        })?;

        Ok(())
    }

    pub fn read(&self) -> Result<Option<ResourceRegistrySnapshot>, ResourceRegistryCacheError> {
        let path = self.registry_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| ResourceRegistryCacheError::ReadFailed {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            ResourceRegistryCacheError::ParseFailed {
                path: path.clone(),
                msg: e.to_string(),
            }
        })?;
        let obj = v.as_object().ok_or(ResourceRegistryCacheError::Malformed {
            path: path.clone(),
            reason: "root is not an object",
        })?;
        let schema = obj.get("schema_version").and_then(|x| x.as_str()).ok_or(
            ResourceRegistryCacheError::Malformed {
                path: path.clone(),
                reason: "missing schema_version",
            },
        )?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(ResourceRegistryCacheError::SchemaMismatch {
                path,
                got: schema.into(),
                want: CACHE_SCHEMA_VERSION.into(),
            });
        }
        let arr = obj.get("resources").and_then(|x| x.as_array()).ok_or(
            ResourceRegistryCacheError::Malformed {
                path: path.clone(),
                reason: "missing resources array",
            },
        )?;
        let mut resources = Vec::with_capacity(arr.len());
        for r in arr {
            let robj = r.as_object().ok_or(ResourceRegistryCacheError::Malformed {
                path: path.clone(),
                reason: "resource is not an object",
            })?;
            let id = robj
                .get("id")
                .and_then(|x| x.as_str())
                .ok_or(ResourceRegistryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing id",
                })?
                .to_string();
            let kind = robj
                .get("kind")
                .and_then(|x| x.as_str())
                .ok_or(ResourceRegistryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing kind",
                })?
                .to_string();
            let summary = robj
                .get("summary")
                .and_then(|x| x.as_str())
                .ok_or(ResourceRegistryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing summary",
                })?
                .to_string();
            let allowlisted = robj.get("allowlisted").and_then(|x| x.as_bool()).ok_or(
                ResourceRegistryCacheError::Malformed {
                    path: path.clone(),
                    reason: "missing allowlisted",
                },
            )?;
            resources.push(ResourceEntry {
                id,
                kind,
                summary,
                allowlisted,
            });
        }
        Ok(Some(ResourceRegistrySnapshot { resources }))
    }

    /// Write an empty registry if the file does not yet exist. Idempotent.
    /// Called from gateway bootstrap so G4 can assume the file is present
    /// with a valid schema version.
    pub fn seed_empty_if_missing(&self) -> Result<bool, ResourceRegistryCacheError> {
        if self.registry_path().exists() {
            return Ok(false);
        }
        self.write(&ResourceRegistrySnapshot::default())?;
        Ok(true)
    }

    pub fn delete(&self) -> Result<(), ResourceRegistryCacheError> {
        let path = self.registry_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|e| ResourceRegistryCacheError::ReadFailed {
                path,
                msg: e.to_string(),
            })?;
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_resource(seed: u8) -> ResourceEntry {
        ResourceEntry {
            id: format!("resource-{:02x}", seed),
            kind: "filesystem".into(),
            summary: format!("sample resource #{}", seed),
            allowlisted: seed % 2 == 0,
        }
    }

    fn sample_snapshot() -> ResourceRegistrySnapshot {
        ResourceRegistrySnapshot {
            resources: vec![sample_resource(1), sample_resource(2), sample_resource(3)],
        }
    }

    #[test]
    fn default_is_empty() {
        let d = ResourceRegistrySnapshot::default();
        assert!(d.resources.is_empty());
    }

    #[test]
    fn round_trip_preserves_resources() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        let snap = sample_snapshot();
        cache.write(&snap).unwrap();
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_absent_returns_none() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        assert!(cache.read().unwrap().is_none());
    }

    #[test]
    fn seed_empty_if_missing_creates_default_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        assert!(!cache.registry_path().exists());
        let created = cache.seed_empty_if_missing().unwrap();
        assert!(created);
        assert!(cache.registry_path().exists());
        let loaded = cache.read().unwrap().unwrap();
        assert!(loaded.resources.is_empty());
    }

    #[test]
    fn seed_is_idempotent_and_preserves_existing_content() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        // Pre-populate.
        cache.write(&sample_snapshot()).unwrap();
        // Seed should not overwrite.
        let created = cache.seed_empty_if_missing().unwrap();
        assert!(!created);
        let loaded = cache.read().unwrap().unwrap();
        assert_eq!(loaded, sample_snapshot());
    }

    #[test]
    fn empty_snapshot_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        let empty = ResourceRegistrySnapshot::default();
        cache.write(&empty).unwrap();
        assert_eq!(cache.read().unwrap().unwrap(), empty);
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        for entry in fs::read_dir(cache.cache_dir()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            assert!(!name.contains(".tmp."), "leftover temp file: {}", name);
        }
    }

    #[test]
    fn read_rejects_wrong_schema() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(
            cache.registry_path(),
            br#"{"schema_version":"v999","resources":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            ResourceRegistryCacheError::SchemaMismatch { .. }
        ));
    }

    #[test]
    fn read_rejects_malformed_json() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        fs::create_dir_all(cache.cache_dir()).unwrap();
        fs::write(cache.registry_path(), b"not json").unwrap();
        assert!(matches!(
            cache.read().unwrap_err(),
            ResourceRegistryCacheError::ParseFailed { .. }
        ));
    }

    #[test]
    fn delete_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let cache = ResourceRegistryCache::at_sovereign_root(td.path());
        cache.write(&sample_snapshot()).unwrap();
        cache.delete().unwrap();
        cache.delete().unwrap();
    }

    #[test]
    fn restart_survival() {
        let td = tempfile::TempDir::new().unwrap();
        let snap = sample_snapshot();
        {
            let cache = ResourceRegistryCache::at_sovereign_root(td.path());
            cache.write(&snap).unwrap();
        }
        let cache2 = ResourceRegistryCache::at_sovereign_root(td.path());
        assert_eq!(cache2.read().unwrap().unwrap(), snap);
    }
}
